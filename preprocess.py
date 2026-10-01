"""영수증 이미지 전처리.

OCR(PP-OCR)과 VLM 에 들어갈 이미지를 따로 만든다. 좌표 복원(layout.py)은 OCR 좌표만
쓰므로 두 입력이 달라도 안전하다.

단계는 요청마다 골라 켠다. 어느 단계가 실제로 인식률을 올리는지 실측 영수증으로
하나씩 비교하기 위해서다. 아무 단계도 켜지 않으면 예전과 똑같이 동작한다.

- crop : 사진에서 영수증 영역만 잘라낸다. 배경(책상 등)에 쓰이던 픽셀을 영수증에
         몰아줘, 같은 해상도 상한 안에서 글자가 커진다. OCR·VLM 모두에 적용한다.
- hires: OCR 입력만 긴 변 상한을 높인다. 긴 영수증을 1536px 로 줄이면 글자 높이가
         15~20px 까지 떨어져 금액이 `,00`·`.00` 으로 깨진다. VLM 은 토큰이 늘어 느려지므로
         건드리지 않는다.
- clahe: OCR 입력만 국소 대비를 올린다. 바랜 감열지 글씨용이다. 흑백 이진화는 옅은
         글씨를 지워버려서 쓰지 않는다. VLM 은 컬러 사진에 익숙하므로 건드리지 않는다.

어떤 단계든 실패하면 그 단계만 건너뛰고 이유를 info 에 남긴다. 전처리 때문에 분석
자체가 실패하는 일은 없어야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

#: 지금까지 쓰던 긴 변 상한. VLM 입력과, hires 를 끈 OCR 입력에 쓴다.
BASE_MAX_SIDE = 1536
#: hires 를 켰을 때 OCR 입력의 긴 변 상한.
HIRES_MAX_SIDE = 2560

STEPS = ("crop", "hires", "clahe")

# crop 판정 기준. 후보가 화면의 이보다 작으면 영수증이 아니라 작은 물체를 잡았다고 본다.
# 낮게 둔다: 멀리서 찍어 영수증이 작게 나온 사진이 잘라내기가 가장 필요한 경우다.
# 잘라내기는 해상도를 떨어뜨리지 않으므로(확대 없이 줄이기만 한다) 위험은 엉뚱한 물체를
# 잡는 것뿐이다. 이보다 크면 이미 꽉 찬 사진이라 잘라도 이득이 없다.
_CROP_MIN_AREA = 0.05
_CROP_MAX_AREA = 0.90
# 가장 큰 종이 덩어리에 함께 묶을 조각의 최소 크기(가장 큰 덩어리 넓이 대비)와,
# 가로 범위가 얼마나 겹쳐야 같은 영수증으로 보는지(좁은 쪽 폭 대비).
# 손이나 폰 그림자가 영수증을 가로지르면 종이가 위아래 두 덩어리로 갈라진다. 큰 덩어리
# 하나만 남기면 합계·결제 부분이 잘려 나간다.
_CROP_JOIN_MIN_AREA = 0.10
_CROP_JOIN_MIN_OVERLAP = 0.5
# 잘라낸 영역 둘레에 남길 여백(긴 변 대비). 가장자리 글자가 잘리지 않게 한다.
_CROP_MARGIN = 0.02
# 종이는 채도로 가른다. 실측: 영수증 종이 채도 15~50, 나무 책상 117~137. 밝기로는
# 갈리지 않는다. 밝은 책상이 종이보다 밝은 사진도 있었다(책상 201, 종이 197).
# 채도 기준은 사진마다 Otsu 로 정하되 이 값보다 낮추지 않는다. 배경까지 무채색이면 사진
# 전체가 채도 0 근처라 Otsu 가 기준을 0 가까이 잡고, 그러면 종이(채도 수~수십)까지 빠진다.
_CROP_MIN_SATURATION_THRESHOLD = 70
# 이보다 어두운 무채색(검은·짙은 회색 바닥 등)은 종이가 아니다.
_CROP_MIN_VALUE = 90


class PreprocessError(ValueError):
    """알 수 없는 전처리 단계 이름."""


def parse_steps(spec: str | None) -> tuple[str, ...]:
    """`"crop,hires"` 를 정해진 순서의 단계 튜플로. 빈 값이면 ()."""
    if spec is None:
        return ()
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise PreprocessError(f"알 수 없는 전처리 단계: {', '.join(unknown)} (가능: {', '.join(STEPS)})")
    return tuple(s for s in STEPS if s in names)


@dataclass
class Prepared:
    vlm_image: Image.Image
    ocr_image: Image.Image
    info: dict = field(default_factory=dict)


def _limit(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image.copy()
    out = image.copy()
    out.thumbnail((max_side, max_side))
    return out


def find_receipt_box(image: Image.Image) -> tuple[tuple[int, int, int, int] | None, str]:
    """영수증(무채색 종이) 영역의 외접 사각형을 원본 좌표로. 못 찾으면 (None, 이유).

    밝기가 아니라 채도로 종이를 가른다. 감열지는 하얗고 채도가 거의 없지만 나무 책상은
    밝기가 비슷해도 채도가 높다. 흰 책상처럼 배경도 밝은 무채색이면 가를 수 없으므로
    건너뛴다(후보가 화면을 채운다).
    """
    import cv2

    width, height = image.size
    scale = 1000 / max(width, height)
    small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    hsv = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2HSV)
    saturation = cv2.GaussianBlur(hsv[..., 1], (7, 7), 0)
    value = hsv[..., 2]
    otsu, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(otsu, _CROP_MIN_SATURATION_THRESHOLD)
    mask = ((saturation <= threshold) & (value >= _CROP_MIN_VALUE)).astype(np.uint8) * 255
    # 글자 구멍을 메워 종이를 한 덩어리로 만든다.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "무채색 영역을 찾지 못함"

    x, y, w, h = _join_receipt_pieces(contours, cv2)
    area = (w * h) / (mask.shape[0] * mask.shape[1])
    if area < _CROP_MIN_AREA:
        return None, f"후보가 너무 작음({area:.2f})"
    if area > _CROP_MAX_AREA:
        return None, f"이미 영수증이 화면을 채움({area:.2f})"

    # 배경이 무채색이고 밝으면(흰 책상) 배경까지 후보가 되어 위의 "화면을 채움"에서 걸러진다.
    # 배경이 어두운 무채색이면 밝기 조건에서 빠지므로 영수증만 남는다. 둘 다 따로 막을 필요가
    # 없다. 채도 차로 한 번 더 거르면 검은 책상 위 영수증처럼 쉬운 경우를 오히려 버린다.

    margin = round(max(mask.shape) * _CROP_MARGIN)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(mask.shape[1], x + w + margin), min(mask.shape[0], y + h + margin)
    box = (round(x0 / scale), round(y0 / scale), min(width, round(x1 / scale)), min(height, round(y1 / scale)))
    return box, f"영역 {area:.2f}"


def _join_receipt_pieces(contours, cv2) -> tuple[int, int, int, int]:
    """가장 큰 종이 덩어리와, 그 위아래로 이어지는 큰 조각들을 합친 외접 사각형."""
    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    x0, y0, w, h = cv2.boundingRect(largest)
    x1, y1 = x0 + w, y0 + h
    for contour in contours:
        if contour is largest or cv2.contourArea(contour) < largest_area * _CROP_JOIN_MIN_AREA:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        overlap = min(x1, cx + cw) - max(x0, cx)
        if overlap < _CROP_JOIN_MIN_OVERLAP * min(w, cw):
            continue  # 옆에 놓인 다른 물체
        x0, y0 = min(x0, cx), min(y0, cy)
        x1, y1 = max(x1, cx + cw), max(y1, cy + ch)
    return x0, y0, x1 - x0, y1 - y0


def apply_clahe(image: Image.Image) -> Image.Image:
    import cv2

    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    rgb = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def prepare(image: Image.Image, steps: tuple[str, ...]) -> Prepared:
    """EXIF 회전을 마친 RGB 원본을 받아 VLM·OCR 입력을 만든다."""
    info: dict = {"steps": list(steps)}
    source = image

    if "crop" in steps:
        try:
            box, reason = find_receipt_box(image)
        except Exception as exc:  # 전처리 실패가 분석 실패가 되면 안 된다
            box, reason = None, f"오류: {exc}"
        info["crop"] = {"box": list(box) if box else None, "reason": reason}
        if box:
            source = image.crop(box)

    vlm_image = _limit(source, BASE_MAX_SIDE)
    # hires 가 꺼져 있으면 OCR 입력은 VLM 입력과 같은 이미지다. 같은 객체를 돌려줘 호출부가
    # 한 번만 저장하게 한다.
    ocr_image = _limit(source, HIRES_MAX_SIDE) if "hires" in steps else vlm_image

    if "clahe" in steps:
        try:
            ocr_image = apply_clahe(ocr_image)
        except Exception as exc:
            info["clahe"] = f"오류: {exc}"

    info["vlm_size"] = list(vlm_image.size)
    info["ocr_size"] = list(ocr_image.size)
    return Prepared(vlm_image=vlm_image, ocr_image=ocr_image, info=info)
