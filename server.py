#!/usr/bin/env python3
"""폰 카메라로 영수증을 찍어 OCR 정확도를 실측하는 로컬 웹 서버.

파이프라인: VLM(Qwen2.5-VL) + OCR(PP-OCRv5 korean) 병렬 실행
 -> 품목명은 두 결과를 비교해 실존 단어 쪽을 채택, 오독은 혼동자모 보정
 -> 합계 금액은 OCR 숫자와 교차 검증

실행: .venv/bin/python server.py
접속: 같은 와이파이에서 http://<맥북IP>:8600
"""
import json
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from corrector import KoreanCorrector, _jamo_distance, _jamo_seq
from layout import analyze_receipt

MAX_SIDE = 1536

PROMPT = """이 한국 영수증 이미지를 읽고 아래 JSON 형식으로만 답해줘. 다른 설명은 쓰지 마.
{
  "store_name": "상호명",
  "purchased_at": "YYYY-MM-DD HH:MM",
  "items": [{"name": "품목명", "quantity": 1, "price": 0, "sub_items": [{"name": "옵션명", "price": 0}]}],
  "total_amount": 0,
  "payment_method": "카드 또는 현금"
}
규칙:
- items에는 실제 구매한 상품만 넣어. 소계/부가세/과세물품가액/면세물품가액/합계/할인 같은 요약 줄은 품목이 아니야.
- 품목 아래 들여쓰기나 -, * 기호로 붙는 옵션/추가선택(예: 샷추가, 사이즈업)은 별도 품목이 아니라 그 품목의 sub_items에 넣어. 없으면 빈 배열로 해.
- 한국 영수증 날짜는 보통 년/월/일 순서야. 25/09/21은 2025-09-21이야.
- 금액은 숫자만(콤마 없이) 적어.
- 읽을 수 없는 값은 null로 해."""

app = FastAPI(
    title="HowMuch 영수증 OCR API",
    description=(
        "영수증 이미지를 분석해 구조화된 정산 데이터(상호/일시/품목/합계)를 반환합니다.\n\n"
        "- 비전 모델(Qwen2.5-VL)과 텍스트 OCR(PP-OCRv5)을 병렬 실행 후 병합\n"
        "- 한글 오독(예: 잠치김밥 → 참치김밥)은 사전 기반으로 자동 보정\n"
        "- 합계 금액은 OCR 숫자와 교차 검증 (`total_verified`)\n\n"
        "처리 시간은 이미지당 약 3초입니다."
    ),
    version="0.1.0",
)


class SubReceiptItem(BaseModel):
    name: str | None = Field(None, description="하위 옵션명", examples=["샷추가"])
    price: int | None = Field(None, description="옵션 금액(원), 없으면 null", examples=[500])


class ReceiptItem(BaseModel):
    name: str | None = Field(None, description="품목명 (보정 적용 후)", examples=["참치김밥"])
    quantity: int | None = Field(None, description="수량", examples=[1])
    price: int | None = Field(None, description="금액(원)", examples=[3500])
    sub_items: list[SubReceiptItem] = Field(
        default_factory=list, description="품목에 붙는 하위 옵션/추가선택 (좌표 기반 복원 포함)"
    )


class ReceiptResult(BaseModel):
    store_name: str | None = Field(None, description="상호명", examples=["Starfield"])
    purchased_at: str | None = Field(
        None, description="구매 일시 (YYYY-MM-DD HH:MM, 시각이 없으면 날짜만)", examples=["2025-10-03 16:47"]
    )
    items: list[ReceiptItem] = Field(default_factory=list, description="구매 품목 목록")
    total_amount: int | None = Field(None, description="합계 금액(원)", examples=[60000])
    payment_method: str | None = Field(None, description="결제 수단", examples=["카드"])
    total_verified: bool = Field(
        False, description="합계 금액이 OCR 인식 숫자와 일치하는지 (true면 신뢰도 높음)"
    )


class CorrectionEntry(BaseModel):
    field: str = Field(description="보정된 필드", examples=["item.name"])
    before: str = Field(description="보정 전 (모델이 읽은 값)", examples=["잠치김밥"])
    after: str = Field(description="보정 후", examples=["참치김밥"])
    reason: str = Field(
        description=(
            "보정 경로: ocr(OCR 교차검증) | confusion_swap(혼동 자모 치환) | "
            "ocr_recovered(VLM 누락 품목을 좌표 기반으로 복구) | ocr_layout(좌표 기반 합계 보완) | "
            "barcode_price_paired(품목명 줄과 바코드·금액 줄 병합)"
        ),
        examples=["confusion_swap"],
    )


class OcrLine(BaseModel):
    text: str = Field(description="OCR이 인식한 텍스트 라인")
    confidence: float = Field(description="인식 신뢰도 (0~1)")
    box: list[int] | None = Field(
        None, description="텍스트 위치 [x1, y1, x2, y2] (리사이즈된 이미지 픽셀 기준)"
    )


class OcrReceiptResponse(BaseModel):
    ok: bool = Field(description="분석 성공 여부")
    elapsed_sec: float = Field(description="처리 시간(초)")
    result: ReceiptResult | None = Field(None, description="구조화된 영수증 데이터 (실패 시 null)")
    corrections: list[CorrectionEntry] = Field(default_factory=list, description="품목명 보정 내역")
    ocr_lines: list[OcrLine] = Field(default_factory=list, description="OCR 원본 인식 결과 (디버깅용)")
    raw: str = Field(description="비전 모델 원본 응답 (디버깅용)")


class OcrErrorResponse(BaseModel):
    ok: bool = Field(False)
    error: str = Field(description="오류 메시지")
_ocr_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)
_vlm = None
_ocr_engine = None
_corrector = None


def _load_models():
    global _vlm, _ocr_engine, _corrector
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    from vlm import create_vlm

    _vlm = create_vlm()

    print("OCR 로드 중: PP-OCRv5 korean mobile")
    _ocr_engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.KOREAN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        }
    )

    _corrector = KoreanCorrector()
    print("모델 로드 완료")


def _run_vlm(image_path: str) -> str:
    return _vlm.generate(image_path, PROMPT)


def _run_text_ocr(image_path: str) -> list[dict]:
    with _ocr_lock:
        result = _ocr_engine(image_path)
    if result.txts is None:
        return []
    lines = []
    for i, (txt, score) in enumerate(zip(result.txts, result.scores)):
        entry = {"text": txt, "confidence": round(float(score), 3)}
        if result.boxes is not None:
            quad = result.boxes[i]  # 4점 사각형 -> 외접 박스
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            entry["box"] = [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]
        lines.append(entry)
    return lines


def _parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _ocr_numbers(ocr_lines: list[dict]) -> set[int]:
    numbers = set()
    for line in ocr_lines:
        for m in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", line["text"]):
            numbers.add(int(m.replace(",", "")))
    return numbers


# VLM이 품목으로 착각하는 영수증 요약 줄 (프롬프트로도 완전히 안 걸러짐)
_SUMMARY_LINE = re.compile(
    r"^(면세|과세)\s*물품\s*가액$|^부\s*가\s*세$|^소\s*계$|^합\s*계$|^할인|^봉사료$|^공급가액$"
)


def _names_match(a: str | None, b: str | None) -> bool:
    """한글 기준으로 같은 품목명인지 판단 (포함 관계 또는 자모 편집거리)."""
    na = re.sub(r"[^가-힣]", "", a or "")
    nb = re.sub(r"[^가-힣]", "", b or "")
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return _jamo_distance(na, nb) <= max(2, len(_jamo_seq(na)) // 3)


def _correct_name(item: dict, key: str, field_name: str, ocr_texts: list[str], corrections: list):
    name = item.get(key)
    if not name:
        return
    res = _corrector.correct(name, ocr_texts)
    if res.changed:
        item[key] = res.corrected
        corrections.append(
            {"field": field_name, "before": res.original, "after": res.corrected, "reason": res.reason}
        )


_BARCODE_NAME = re.compile(r"^[\d\s\-*]{8,}$")


def _pair_orphan_prices(kept: list[dict], corrections: list):
    """VLM이 '품목명 줄'과 '바코드·금액 줄'을 별도 품목으로 쪼갠 경우 하나로 합친다."""
    i = 1
    while i < len(kept):
        item = kept[i]
        name = (item.get("name") or "").strip()
        prev = kept[i - 1]
        if (
            _BARCODE_NAME.match(name)
            and (item.get("price") or 0) > 0
            and prev.get("price") in (None, 0)
            and re.search(r"[가-힣]", prev.get("name") or "")
        ):
            prev["price"] = item["price"]
            if not prev.get("quantity") and item.get("quantity"):
                prev["quantity"] = item["quantity"]
            corrections.append(
                {
                    "field": "items",
                    "before": name,
                    "after": prev["name"],
                    "reason": "barcode_price_paired",
                }
            )
            kept.pop(i)
            continue
        i += 1


def _apply_layout(kept: list[dict], layout, ocr_texts: list[str], corrections: list):
    """좌표 기반 구조(layout)를 VLM 결과와 교차: 하위 옵션 부착 + 누락 품목 복구."""
    for li in layout.items:
        if li.name and _SUMMARY_LINE.match(li.name.strip()):
            continue

        matched = next(
            (
                it
                for it in kept
                if _names_match(it.get("name"), li.name)
                or (li.price is not None and it.get("price") == li.price)
            ),
            None,
        )

        subs = [{"name": s.name, "price": s.price} for s in li.sub_items]
        if matched is not None:
            if subs and not matched.get("sub_items"):
                matched["sub_items"] = subs
            continue

        # VLM이 놓친 품목 복구: 좌표상 품목 영역에서 이름+금액이 확실한 줄만
        if (
            li.name is None
            or li.price is None
            or li.name_confidence < 0.8
            or len(re.sub(r"[^가-힣]", "", li.name)) < 2
        ):
            continue
        res = _corrector.correct(li.name, ocr_texts)
        name = res.corrected if res.changed else li.name
        kept.append({"name": name, "quantity": li.quantity, "price": li.price, "sub_items": subs})
        corrections.append(
            {"field": "items", "before": "(VLM 누락)", "after": name, "reason": "ocr_recovered"}
        )


def _merge(parsed: dict, ocr_lines: list[dict]) -> tuple[dict, list[dict]]:
    """VLM 결과를 OCR 텍스트·좌표와 대조해 품목명 보정, 구조 복원, 합계 검증을 수행한다."""
    ocr_texts = [line["text"] for line in ocr_lines if line["confidence"] >= 0.8]
    corrections = []
    layout = analyze_receipt(ocr_lines)

    items = parsed.get("items") or []
    kept = [it for it in items if not _SUMMARY_LINE.match((it.get("name") or "").strip())]
    parsed["items"] = kept
    _pair_orphan_prices(kept, corrections)

    for item in kept:
        item["sub_items"] = [s for s in (item.get("sub_items") or []) if isinstance(s, dict)]
        _correct_name(item, "name", "item.name", ocr_texts, corrections)
        for sub in item["sub_items"]:
            _correct_name(sub, "name", "item.sub_items.name", ocr_texts, corrections)

    _apply_layout(kept, layout, ocr_texts, corrections)

    total = parsed.get("total_amount")
    if isinstance(total, (int, float)):
        if layout.total_amount is not None:
            # "합계" 라벨 옆 숫자와 정확히 일치해야 검증 통과 (임의 숫자 일치 오검증 방지)
            parsed["total_verified"] = int(total) == layout.total_amount
        else:
            parsed["total_verified"] = int(total) in _ocr_numbers(ocr_lines)
    elif layout.total_amount is not None:
        parsed["total_amount"] = layout.total_amount
        parsed["total_verified"] = True
        corrections.append(
            {
                "field": "total_amount",
                "before": "null",
                "after": str(layout.total_amount),
                "reason": "ocr_layout",
            }
        )
    else:
        parsed["total_verified"] = False
    return parsed, corrections


@app.on_event("startup")
def startup():
    _load_models()


@app.post(
    "/ocr/receipt",
    response_model=OcrReceiptResponse,
    responses={500: {"model": OcrErrorResponse, "description": "분석 중 서버 오류"}},
    summary="영수증 이미지 분석",
    description=(
        "영수증 사진을 업로드하면 상호/일시/품목/합계를 구조화해 반환합니다.\n\n"
        "- `file`: 영수증 이미지 (JPEG/PNG, 폰 카메라 원본 그대로 가능 — 서버에서 리사이즈)\n"
        "- 프론트는 `result`만 사용하면 되고, `corrections`/`ocr_lines`/`raw`는 디버깅용입니다.\n"
        "- `result.total_verified`가 false면 합계를 사용자에게 확인받는 UX를 권장합니다."
    ),
)
async def ocr_receipt(file: UploadFile = File(..., description="영수증 이미지 파일")):
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        image = Image.open(BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE))
        image.save(tmp_path, "JPEG", quality=92)

        start = time.perf_counter()
        vlm_future = _executor.submit(_run_vlm, tmp_path)
        ocr_future = _executor.submit(_run_text_ocr, tmp_path)
        raw_text = vlm_future.result()
        ocr_lines = ocr_future.result()
        elapsed = round(time.perf_counter() - start, 1)

        parsed = _parse_json(raw_text)
        corrections = []
        if parsed is not None:
            parsed, corrections = _merge(parsed, ocr_lines)

        return JSONResponse(
            {
                "ok": parsed is not None,
                "elapsed_sec": elapsed,
                "result": parsed,
                "corrections": corrections,
                "ocr_lines": ocr_lines,
                "raw": raw_text,
            }
        )
    except Exception as e:  # 실측용 서버라 원인 그대로 노출
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8600")))
