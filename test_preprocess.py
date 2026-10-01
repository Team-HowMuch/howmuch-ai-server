"""preprocess.py 회귀 테스트.

실행: .venv/bin/python -m pytest test_preprocess.py -v
"""
import numpy as np
from PIL import Image, ImageDraw

import preprocess as pre

_WOOD = (196, 140, 72)  # 실측 책상: 채도 높음(약 120~140)
_PAPER = (236, 234, 230)  # 감열지: 채도 낮음


def _photo(size=(3000, 4000), receipt=(900, 300, 2100, 3700), background=_WOOD):
    """배경 위에 영수증(밝은 무채색 사각형)과 글자 줄을 그린 합성 사진."""
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    draw.rectangle(receipt, fill=_PAPER)
    x0, y0, x1, y1 = receipt
    for y in range(y0 + 80, y1 - 80, 90):
        draw.rectangle((x0 + 60, y, x1 - 200, y + 30), fill=(30, 30, 30))
    return image


def test_parse_steps_orders_and_validates():
    assert pre.parse_steps(None) == ()
    assert pre.parse_steps("") == ()
    assert pre.parse_steps(" HIRES , crop ") == ("crop", "hires")  # 정해진 순서로
    assert pre.parse_steps("clahe,clahe") == ("clahe",)
    try:
        pre.parse_steps("crop,sharpen")
    except pre.PreprocessError as exc:
        assert "sharpen" in str(exc)
    else:
        raise AssertionError("알 수 없는 단계를 받아들였다")


def test_no_steps_keeps_previous_behaviour():
    """아무 단계도 없으면 예전처럼 긴 변 1536 으로 줄인 같은 이미지가 두 입력이 된다."""
    prepared = pre.prepare(_photo(), ())
    assert prepared.vlm_image.size == prepared.ocr_image.size == (1152, 1536)
    assert np.array_equal(np.asarray(prepared.vlm_image), np.asarray(prepared.ocr_image))
    assert prepared.info["steps"] == []
    assert "crop" not in prepared.info


def test_hires_enlarges_only_ocr_input():
    prepared = pre.prepare(_photo(), ("hires",))
    assert max(prepared.vlm_image.size) == pre.BASE_MAX_SIDE
    assert max(prepared.ocr_image.size) == pre.HIRES_MAX_SIDE


def test_hires_never_upscales_small_photo():
    prepared = pre.prepare(_photo(size=(900, 1200), receipt=(300, 100, 600, 1100)), ("hires",))
    assert prepared.ocr_image.size == (900, 1200)


def test_clahe_changes_only_ocr_input():
    base = pre.prepare(_photo(), ())
    boosted = pre.prepare(_photo(), ("clahe",))
    assert np.array_equal(np.asarray(base.vlm_image), np.asarray(boosted.vlm_image))
    assert not np.array_equal(np.asarray(base.ocr_image), np.asarray(boosted.ocr_image))


def test_crop_finds_receipt_on_saturated_background():
    receipt = (900, 300, 2100, 3700)
    box, reason = pre.find_receipt_box(_photo(receipt=receipt))
    assert box is not None, reason
    x0, y0, x1, y1 = box
    # 영수증을 다 덮고, 여백은 긴 변의 몇 % 이내
    assert x0 <= receipt[0] and y0 <= receipt[1] and x1 >= receipt[2] and y1 >= receipt[3]
    slack = 0.05 * 4000
    assert receipt[0] - x0 < slack and x1 - receipt[2] < slack
    assert receipt[1] - y0 < slack and y1 - receipt[3] < slack


def test_crop_ignores_brightness_and_uses_saturation():
    """실측: 밝은 책상이 종이보다 밝은 사진이 있었다. 밝기로 가르면 책상까지 잡는다."""
    bright_wood = (250, 205, 120)  # 종이보다 밝지만 채도가 높다
    box, reason = pre.find_receipt_box(_photo(background=bright_wood))
    assert box is not None, reason
    assert box[2] - box[0] < 1500  # 책상 폭(3000)까지 번지지 않는다


def test_crop_skips_when_background_is_also_bright_and_colourless():
    """흰 책상처럼 배경도 밝은 무채색이면 가를 수 없으니 자르지 않는다."""
    box, reason = pre.find_receipt_box(_photo(background=(200, 200, 200)))
    assert box is None
    assert reason


def test_crop_finds_receipt_on_dark_colourless_background():
    """검은·짙은 회색 책상은 채도가 0 이어도 어두워서 종이와 갈린다. 쉬운 경우를 버리면 안 된다."""
    receipt = (900, 300, 2100, 3700)
    box, reason = pre.find_receipt_box(_photo(receipt=receipt, background=(45, 45, 45)))
    assert box is not None, reason
    assert box[2] - box[0] < 1500


def test_crop_keeps_small_receipt_and_rejects_speck():
    """멀리서 찍어 작게 나온 영수증은 자른다. 화면의 몇 % 도 안 되는 조각은 영수증으로 보지 않는다."""
    box, _ = pre.find_receipt_box(_photo(receipt=(1100, 1000, 1900, 3000)))  # 약 13%
    assert box is not None
    box, reason = pre.find_receipt_box(_photo(receipt=(1300, 1700, 1600, 2300)))  # 약 1.5%
    assert box is None
    assert "작음" in reason


def test_crop_skips_when_receipt_fills_frame():
    box, reason = pre.find_receipt_box(_photo(receipt=(20, 20, 2980, 3980)))
    assert box is None
    assert "채움" in reason


def test_prepare_crop_applies_to_both_inputs_and_records_box():
    prepared = pre.prepare(_photo(), ("crop",))
    info = prepared.info["crop"]
    assert info["box"] is not None
    # 영수증 비율(1200x3400)로 잘렸으니 두 입력 모두 세로로 길다
    for image in (prepared.vlm_image, prepared.ocr_image):
        width, height = image.size
        assert height / width > 2.5


def test_crop_failure_falls_back_to_full_image():
    original = pre.find_receipt_box

    def boom(image):
        raise RuntimeError("cv2 폭발")

    pre.find_receipt_box = boom
    try:
        prepared = pre.prepare(_photo(), ("crop",))
    finally:
        pre.find_receipt_box = original
    assert prepared.info["crop"]["box"] is None
    assert "cv2 폭발" in prepared.info["crop"]["reason"]
    assert prepared.vlm_image.size == (1152, 1536)


def test_crop_keeps_receipt_split_by_shadow():
    """리뷰 P2: 손·폰 그림자가 영수증을 가로지르면 종이가 위아래 두 덩어리로 갈라진다.

    큰 덩어리 하나만 남기면 합계·결제 부분이 잘려 나간다.
    """
    receipt = (900, 300, 2100, 3700)
    image = _photo(receipt=receipt)
    ImageDraw.Draw(image).rectangle((0, 2300, 3000, 2550), fill=(40, 40, 40))  # 가로 그림자 띠
    box, reason = pre.find_receipt_box(image)
    assert box is not None, reason
    assert box[1] <= receipt[1] and box[3] >= receipt[3], box


def test_crop_does_not_join_object_beside_receipt():
    """영수증 옆에 놓인 다른 흰 물체(가로 범위가 겹치지 않음)는 묶지 않는다."""
    receipt = (900, 300, 2100, 3700)
    image = _photo(receipt=receipt)
    ImageDraw.Draw(image).rectangle((2400, 500, 2900, 1500), fill=_PAPER)  # 옆의 냅킨
    box, _ = pre.find_receipt_box(image)
    assert box is not None
    assert box[2] < 2400, box


def test_default_path_uses_one_image_for_both_inputs():
    """리뷰 P3: 같은 이미지를 두 번 만들어 저장하지 않는다(예전처럼 파일 하나)."""
    prepared = pre.prepare(_photo(), ())
    assert prepared.ocr_image is prepared.vlm_image
    cropped = pre.prepare(_photo(), ("crop",))
    assert cropped.ocr_image is cropped.vlm_image
    hires = pre.prepare(_photo(), ("hires",))
    assert hires.ocr_image is not hires.vlm_image
