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
from layout import _money_value, _sum_fees, analyze_receipt, delivery_fee_label, _discount_kind

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
    price: int | None = Field(
        None, description="옵션 추가금(원). 품목 price 에 더한다. 없으면 null", examples=[500]
    )


class ReceiptItem(BaseModel):
    name: str | None = Field(None, description="품목명 (보정 적용 후)", examples=["참치김밥"])
    quantity: int | None = Field(None, description="수량", examples=[1])
    price: int | None = Field(
        None,
        description=(
            "줄 금액(원). 옵션 제외, 할인 **전**, 수량 반영. "
            "줄 최종 금액은 price + Σsub_items.price − discount 다."
        ),
        examples=[3500],
    )
    discount: int = Field(
        0,
        description=(
            "이 품목에 귀속되는 할인액(원). 항상 0 이상이고 없으면 0. "
            "실제로 낸 금액은 price - discount 다. 영수증 품목 줄 바로 아래 붙는 "
            "행사할인·특매할인·에누리 줄이 여기로 들어온다."
        ),
        examples=[500],
    )
    sub_items: list[SubReceiptItem] = Field(
        default_factory=list, description="품목에 붙는 하위 옵션/추가선택 (좌표 기반 복원 포함)"
    )


class ReceiptResult(BaseModel):
    store_name: str | None = Field(None, description="상호명", examples=["Starfield"])
    purchased_at: str | None = Field(
        None, description="구매 일시 (YYYY-MM-DD HH:MM, 시각이 없으면 날짜만)", examples=["2025-10-03 16:47"]
    )
    items: list[ReceiptItem] = Field(default_factory=list, description="구매 품목 목록")
    discount: int = Field(
        0,
        description=(
            "특정 품목에 귀속되지 않는 영수증 전체 단위 할인액(원). 항상 0 이상이고 없으면 0. "
            "품목별 할인은 items[].discount 에 따로 담기므로 여기에 중복해서 넣지 않는다."
        ),
        examples=[1000],
    )
    delivery_fee: int = Field(
        0,
        description=(
            "배달비(원). 배달팁 할인·무료배달을 상계한 순액이라 항상 0 이상이고, 없으면 0. "
            "배달비는 items 에 넣지 않는다. 합계는 "
            "Σ(price + Σsub_items.price − discount) − discount + delivery_fee 다."
        ),
        examples=[3000],
    )
    total_amount: int | None = Field(None, description="할인이 모두 반영된 최종 결제 금액(원)", examples=[60000])
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
            "barcode_price_paired(품목명 줄과 바코드·금액 줄 병합) | "
            "sub_promoted(잘못 하위로 묶인 품목을 최상위로 승격) | "
            "sub_reassigned(하위 옵션을 좌표상 실제 부모 품목으로 이동) | "
            "sub_demoted(옵션 마커가 남은 품목을 직전 품목의 하위로 강등) | "
            "discount_from_sub(하위 옵션으로 잘못 올라온 할인 줄을 discount 로 이동) | "
            "delivery_fee_from_item(품목·옵션으로 올라온 배달비를 delivery_fee 로 이동)"
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
    """좌표 기반 구조(layout)를 VLM 결과와 교차.

    layout이 계층(최상위/하위)의 기준이다. VLM은 옵션 기호(>)에 끌려서
    최상위 품목까지 한 품목의 하위로 묶는 오류를 낸다.
    1. 승격: VLM이 하위로 넣었지만 layout상 최상위 품목이면 최상위로 꺼낸다.
    2. 재배치: layout상 다른 품목 소속의 하위 항목은 그 품목으로 옮긴다.
    3. 부착/복구: 하위 옵션 부착, VLM 누락 품목 복구.
    """
    layout_items = [
        li for li in layout.items if not (li.name and _SUMMARY_LINE.match(li.name.strip()))
    ]

    def _layout_subs(li) -> list[dict]:
        # 금액을 못 읽은 layout 하위 항목은 오독 잔재일 가능성이 높아 복사하지 않는다
        return [{"name": s.name, "price": s.price} for s in li.sub_items if s.price is not None]

    # 1) 승격: VLM 이름(오독이 적음)을 유지한 채 최상위 품목으로 꺼낸다
    for v in list(kept):
        remaining = []
        for s in v.get("sub_items", []):
            li = next(
                (
                    li
                    for li in layout_items
                    if li.price is not None
                    and s.get("price") == li.price
                    and _names_match(s.get("name"), li.name)
                ),
                None,
            )
            if li is None:
                remaining.append(s)
                continue
            already = next(
                (k for k in kept if k is not v and _names_match(k.get("name"), li.name)), None
            )
            if already is None:
                kept.append(
                    {
                        "name": s.get("name"),
                        "quantity": li.quantity,
                        "price": s.get("price"),
                        "sub_items": _layout_subs(li),
                    }
                )
            corrections.append(
                {
                    "field": "items",
                    "before": f"{v.get('name')} > {s.get('name')}",
                    "after": s.get("name") or "",
                    "reason": "sub_promoted",
                }
            )
        v["sub_items"] = remaining

    # 2) 재배치: layout이 다른 부모 소속이라고 판정한 하위 항목 이동
    for v in kept:
        lv = next((li for li in layout_items if _names_match(v.get("name"), li.name)), None)
        if lv is None:
            continue
        remaining = []
        for s in v["sub_items"]:
            parent = next(
                (
                    li
                    for li in layout_items
                    if any(
                        _names_match(s.get("name"), ls.name) and s.get("price") == ls.price
                        for ls in li.sub_items
                    )
                ),
                None,
            )
            if parent is not None and parent is not lv:
                owner = next(
                    (k for k in kept if _names_match(k.get("name"), parent.name)), None
                )
                if owner is not None:
                    existing = next(
                        (x for x in owner["sub_items"] if _names_match(x.get("name"), s.get("name"))),
                        None,
                    )
                    if existing is not None:
                        existing["name"] = s.get("name") or existing["name"]  # VLM 이름 우선
                    else:
                        owner["sub_items"].append(s)
                    corrections.append(
                        {
                            "field": "items",
                            "before": f"{v.get('name')} > {s.get('name')}",
                            "after": f"{owner.get('name')} > {s.get('name')}",
                            "reason": "sub_reassigned",
                        }
                    )
                    continue
            remaining.append(s)
        v["sub_items"] = remaining

    # 3) 부착 + 복구
    for li in layout_items:
        matched = next(
            (
                it
                for it in kept
                if _names_match(it.get("name"), li.name)
                or (li.price is not None and it.get("price") == li.price)
            ),
            None,
        )

        subs = _layout_subs(li)
        if matched is not None:
            if subs and not matched.get("sub_items"):
                matched["sub_items"] = subs
            # 할인은 좌표 기반 layout 만 안다. VLM 응답에는 할인 필드가 없다.
            if li.discount:
                matched["discount"] = li.discount
                matched["_discount_amounts"] = list(li.discount_amounts)
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
        kept.append(
            {
                "name": name,
                "quantity": li.quantity,
                "price": li.price,
                "sub_items": subs,
                "discount": li.discount,
                "_discount_amounts": list(li.discount_amounts),
            }
        )
        corrections.append(
            {"field": "items", "before": "(VLM 누락)", "after": name, "reason": "ocr_recovered"}
        )


# VLM이 값을 못 찾을 때 프롬프트 예시를 그대로 돌려주는 경우
_PLACEHOLDERS = {"store_name": "상호명", "payment_method": "카드 또는 현금", "purchased_at": "YYYY-MM-DD HH:MM"}


def _scrub_placeholders(parsed: dict):
    for key, placeholder in _PLACEHOLDERS.items():
        if parsed.get(key) == placeholder:
            parsed[key] = None


def _merge_wrapped_names(kept: list[dict], corrections: list):
    """줄바꿈으로 잘린 메뉴명 병합.

    긴 메뉴명(예: '갓 튀긴 옛날통닭 + 콜라 + 치킨 무 [한그릇 세트]')이 두 줄로
    인쇄되면 VLM이 두 품목으로 쪼갠다. 금액이 0원인데 하위 옵션을 거느린 품목은
    실제 상품이 아니라 직전 품목명의 이어짐이므로 합친다.
    """
    i = 1
    while i < len(kept):
        item = kept[i]
        prev = kept[i - 1]
        if (
            item.get("price") in (None, 0)
            and item.get("sub_items")
            and item.get("name")
            and prev.get("price")
        ):
            joiner = " + " if "+" in (prev.get("name") or "") else " "
            merged_name = f"{prev.get('name')}{joiner}{item['name']}"
            corrections.append(
                {
                    "field": "item.name",
                    "before": f"{prev.get('name')} / {item['name']}",
                    "after": merged_name,
                    "reason": "name_wrapped",
                }
            )
            prev["name"] = merged_name
            for sub in item["sub_items"]:
                if not any(
                    _names_match(x.get("name"), sub.get("name"))
                    and x.get("price") == sub.get("price")
                    for x in prev["sub_items"]
                ):
                    prev["sub_items"].append(sub)
            kept.pop(i)
            continue
        i += 1


_ITEM_MARKER = re.compile(r"^[▶►▸>›»└+*~]\s*|^-\s+")


def _demote_marked_items(kept: list[dict], corrections: list):
    """이름에 옵션 마커(▶, >, + 등)가 남은 최상위 품목을 직전 품목의 하위로 내린다.

    OCR이 마커 글리프를 놓치면 layout은 최상위로 판정하지만, 영수증에 인쇄된
    마커는 명시적 계층 표기다(하삼동커피 실측: ▶개인텀블러지참). VLM이 이름에
    남긴 마커를 계층 증거로 쓴다. 모든 품목에 마커가 있으면 불릿 장식으로 보고
    건너뛴다. layout에서 승격된 품목은 이름 정규화로 마커가 이미 벗겨져 있어
    다시 강등되지 않는다.
    """
    marked = [it for it in kept if _ITEM_MARKER.match((it.get("name") or "").strip())]
    if not marked or len(marked) == len(kept):
        return
    parent = None
    for it in list(kept):
        name = (it.get("name") or "").strip()
        if not _ITEM_MARKER.match(name):
            parent = it
            continue
        if parent is None:
            continue  # 첫 품목이 마커면 붙일 부모가 없다
        stripped = _ITEM_MARKER.sub("", name)
        parent["sub_items"].append({"name": stripped, "price": it.get("price")})
        parent["sub_items"].extend(it.get("sub_items") or [])
        kept.remove(it)
        corrections.append(
            {
                "field": "items",
                "before": name,
                "after": f"{parent.get('name')} > {stripped}",
                "reason": "sub_demoted",
            }
        )


def _fold_discount_subitems(kept: list[dict], corrections: list) -> int:
    """VLM 이 sub_items 로 올려보낸 할인 줄을 item.discount 로 접는다.

    VLM 프롬프트에는 할인 필드가 없어서, 모델은 `$특매할인 -400` 같은 줄을 들여쓰기된
    옵션으로 보고 sub_items 에 음수 금액으로 넣는다. 신세계처럼 이름까지 잃고
    `옵션 -27,800` 으로 오는 경우도 있다. 그대로 두면 같은 할인이 sub_items 와
    discount 양쪽에 남아 백엔드·프론트가 두 번 빼게 된다.

    금액 단위로 대조해 중복을 지운다. layout 이 이미 센 금액과 같은 값이면 한 번만
    센다. 이름에 할인 어휘가 없어도 금액이 음수인 하위 항목은 할인으로 본다.
    실측 63장에서 음수 하위 항목은 전부 할인이었다.

    이름이 합계부 요약 라벨(`합인금액:`, `총할인액` 등)인 하위 항목은 품목 할인이
    아니라 **영수증 전체 요약**이 잘못 붙은 것이다. 품목에 더하면 같은 돈이 품목
    할인과 요약 양쪽에 남는다. 그런 값은 품목이 아니라 반환값으로 돌려보내
    호출부가 품목 할인 합과 대조하게 한다.

    반환값: 하위 항목에서 건진 요약 할인액의 합.
    """
    summary_from_subs = 0
    for item in kept:
        # layout 이 이 품목에 귀속시킨 개별 할인 금액들. item["discount"] 는 이미 이 합이다.
        counted = [int(a) for a in item.pop("_discount_amounts", [])]
        folded: list[tuple[str | None, int, bool]] = []
        remaining = []
        for sub in item.get("sub_items", []):
            price = sub.get("price")
            if not isinstance(price, (int, float)) or not price:
                remaining.append(sub)
                continue
            price = int(price)
            # 음수 금액이거나 이름이 할인 어휘면 할인으로 본다.
            if price >= 0 and _discount_kind(sub.get("name"), price) is None:
                remaining.append(sub)
                continue
            amount = abs(price)
            if _discount_kind(sub.get("name"), price) == "total":
                # 합계부 요약줄이 하위로 잘못 붙은 것 — 품목이 아니라 영수증 단위다
                summary_from_subs += amount
                corrections.append(
                    {
                        "field": "item.sub_items",
                        "before": f"{item.get('name')} / {sub.get('name')} -{amount:,}",
                        "after": "영수증 단위 할인 요약으로 분리",
                        "reason": "discount_from_sub",
                    }
                )
                continue
            duplicate = amount in counted
            if duplicate:
                counted.remove(amount)  # layout 이 이미 센 같은 할인 — 더하지 않는다
            folded.append((sub.get("name"), amount, duplicate))
        item["sub_items"] = remaining
        if not folded:
            continue
        extra = sum(amount for _, amount, duplicate in folded if not duplicate)
        item["discount"] = int(item.get("discount") or 0) + extra
        for name, amount, duplicate in folded:
            corrections.append(
                {
                    "field": "item.discount",
                    "before": f"sub_items: {name} -{amount:,}",
                    "after": ("discount 에 이미 반영됨 (중복 제거)" if duplicate
                              else f"discount: {amount:,}"),
                    "reason": "discount_from_sub",
                }
            )
    return summary_from_subs


def _as_int(value) -> int:
    """VLM 금액을 정수로. 숫자가 아니면 `3,000` 같은 문자열까지 읽고, 못 읽으면 0."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return _money_value(value.strip()) or 0
    return 0


def _is_nameless_negative(entry: dict) -> bool:
    """무료배달 상계 줄: 이름이 없고(한글 없음) 금액이 음수."""
    name = entry.get("name")
    return (not isinstance(name, str) or not re.search(r"[가-힣]", name)) and _as_int(entry.get("price")) < 0


def _take_fees(entries: list[dict]) -> tuple[list[dict], list[tuple[str, int, str]]]:
    """배달비 항목과 그 바로 뒤의 무료배달 상계 항목을 떼어낸다.

    반환: (남은 항목, [(이름, 금액, 종류)]). 종류는 parent·child·off·waiver.
    """
    remaining: list[dict] = []
    taken: list[tuple[str, int, str]] = []
    after_fee = False
    for entry in entries:
        name = entry.get("name")
        kind = delivery_fee_label(name)
        if kind is not None:
            taken.append((name, _as_int(entry.get("price")), kind))
            after_fee = True
            continue
        if after_fee and _is_nameless_negative(entry):
            taken.append((name or "(이름 없음)", _as_int(entry.get("price")), "waiver"))
            continue
        after_fee = False
        remaining.append(entry)
    return remaining, taken


def _extract_delivery_fee(kept: list[dict], layout, corrections: list) -> int:
    """배달비를 품목·하위 옵션에서 떼어내 delivery_fee 로 돌려준다.

    VLM 프롬프트에는 배달비 필드가 없어서, 모델은 배달비를 품목(`배달비 2,000`)이나
    직전 품목의 옵션으로 올려보내거나 아예 빠뜨린다. 품목에 남겨 두면 정산에서 메뉴처럼
    나뉘고, delivery_fee 에도 넣으면 합계에서 두 번 더해진다.

    **다른 병합 단계보다 먼저** 돌아야 한다. 늦게 떼면 그 사이에
    - 무료배달 상계 `-4,100` 이 할인 접기(_fold_discount_subitems)에서 품목 할인이 되어 두 번 빠지고,
    - 좌표 복구(_apply_layout)가 VLM 이 놓친 `군만두 3,000` 을 금액이 같은 `배달팁 3,000` 과
      짝지어 되살리지 않고,
    - 이름 병합(_merge_wrapped_names)이 뒤 품목을 배달비에 붙였다가 같이 버린다.

    VLM 값도 좌표와 같은 규칙(상위·내역·감액, _sum_fees)으로 합친다. VLM 이 `배달팁 3,000` 을
    품목과 옵션 양쪽에 올리거나 `ㄴ기본배달팁` 내역까지 올리면 그냥 더해서는 부풀어 오른다.
    상계 줄이 배달비와 다른 층(배달비는 옵션, `-4,100` 은 품목)에 오면 같은 금액끼리 짝짓는다.

    값은 좌표(layout)를 우선한다. layout 은 `기본배달팁 4,100` 바로 아래의 `-4,100`
    (무료배달) 까지 보고 순액을 내지만, VLM 은 4,100 만 옮겨 적는 경우가 있다. layout 이
    금액을 읽은 배달비 줄이 하나도 없을 때만 VLM 이 올린 값을 쓴다.
    """
    moved: list[tuple[str, int, str]] = []
    remaining, taken = _take_fees(kept)
    moved.extend(taken)
    for item in remaining:
        subs, taken = _take_fees(item.get("sub_items", []))
        item["sub_items"] = subs
        moved.extend(taken)

    # 다른 층의 상계: 떼어낸 배달비 금액과 같은 크기의 이름 없는 음수 항목
    unmatched = [price for _, price, kind in moved if kind in ("parent", "child") and price > 0]
    for price in [price for _, price, kind in moved if kind == "waiver"]:
        if -price in unmatched:
            unmatched.remove(-price)

    def strip_cross_level(entries: list[dict]) -> list[dict]:
        out = []
        for entry in entries:
            amount = -_as_int(entry.get("price"))
            if _is_nameless_negative(entry) and amount in unmatched:
                unmatched.remove(amount)
                moved.append((entry.get("name") or "(이름 없음)", -amount, "waiver"))
                continue
            out.append(entry)
        return out

    if unmatched:
        remaining = strip_cross_level(remaining)
        for item in remaining:
            item["sub_items"] = strip_cross_level(item.get("sub_items", []))
    kept[:] = remaining  # parsed["items"] 와 같은 리스트라 제자리에서 바꾼다

    parents = [price for _, price, kind in moved if kind == "parent"]
    children = [price for _, price, kind in moved if kind == "child"]
    offs = [-abs(price) for _, price, kind in moved if kind == "off"]
    waivers = [price for _, price, kind in moved if kind == "waiver"]
    from_vlm = _sum_fees(parents, children, offs + waivers) if moved else 0

    fee = layout.delivery_fee if layout.delivery_fee is not None else from_vlm
    for name, price, _ in moved:
        corrections.append(
            {
                "field": "delivery_fee",
                "before": f"items: {name} {price:,}",
                "after": f"delivery_fee: {fee:,}",
                "reason": "delivery_fee_from_item",
            }
        )
    return fee


def _items_sum(items: list[dict]) -> int:
    """품목 합계. price 는 할인 전 금액이므로 귀속 할인을 여기서 뺀다.

    예전에는 할인이 sub_items 안에 음수로 들어와 우연히 상계됐는데, 합계부 요약줄까지
    하위로 붙는 바람에 같은 할인이 두 번 빠졌다. 이제 할인은 discount 로만 센다.
    """
    total = 0
    for it in items:
        if isinstance(it.get("price"), (int, float)):
            total += int(it["price"])
        for s in it.get("sub_items", []):
            if isinstance(s.get("price"), (int, float)):
                total += int(s["price"])
        total -= int(it.get("discount") or 0)
    return total


def _merge(parsed: dict, ocr_lines: list[dict]) -> tuple[dict, list[dict]]:
    """VLM 결과를 OCR 텍스트·좌표와 대조해 품목명 보정, 구조 복원, 합계 검증을 수행한다."""
    ocr_texts = [line["text"] for line in ocr_lines if line["confidence"] >= 0.8]
    corrections = []
    layout = analyze_receipt(ocr_lines)

    _scrub_placeholders(parsed)
    items = parsed.get("items") or []
    kept = [it for it in items if not _SUMMARY_LINE.match((it.get("name") or "").strip())]
    parsed["items"] = kept
    _pair_orphan_prices(kept, corrections)
    for item in kept:
        item["sub_items"] = [s for s in (item.get("sub_items") or []) if isinstance(s, dict)]
    # 배달비는 다른 병합 단계보다 먼저 뗀다. 이유는 _extract_delivery_fee 참고.
    parsed["delivery_fee"] = _extract_delivery_fee(kept, layout, corrections)
    _merge_wrapped_names(kept, corrections)

    for item in kept:
        _correct_name(item, "name", "item.name", ocr_texts, corrections)
        for sub in item["sub_items"]:
            if isinstance(sub.get("name"), str):
                sub["name"] = re.sub(r"^[-*+└>›»▶►▸~\s]+", "", sub["name"])  # 옵션 기호 제거
            _correct_name(sub, "name", "item.sub_items.name", ocr_texts, corrections)

    _apply_layout(kept, layout, ocr_texts, corrections)
    _demote_marked_items(kept, corrections)
    summary_from_subs = _fold_discount_subitems(kept, corrections)

    # 할인은 좌표 기반 layout 과, VLM 이 하위 옵션으로 잘못 올린 줄에서 온다.
    # VLM 프롬프트에는 할인 필드가 없어서 모델이 할인 줄을 옵션으로 보거나 빠뜨린다.
    for item in kept:
        value = item.get("discount")
        item["discount"] = int(value) if isinstance(value, (int, float)) and value > 0 else 0

    # 영수증 단위 할인 = 요약줄이 말하는 총 할인 - 이미 품목에 귀속시킨 할인.
    # 요약줄은 같은 할인을 다시 적은 것이므로 그대로 더하면 이중계상이다. 좌표에서
    # 본 요약값과 하위 옵션에서 건진 요약값 중 큰 쪽을 총 할인으로 본다(같은 줄을
    # 양쪽에서 봤을 뿐이지 서로 더할 값이 아니다).
    item_discount_sum = sum(item["discount"] for item in kept)
    summary_total = max(layout.summary_discount, summary_from_subs)
    parsed["discount"] = max(0, summary_total - item_discount_sum)

    total = parsed.get("total_amount")
    layout_total = layout.total_amount
    if isinstance(total, (int, float)):
        t = int(total)
        if layout_total is None:
            parsed["total_verified"] = t in _ocr_numbers(ocr_lines)
        elif t == layout_total:
            # "합계" 라벨 옆 숫자와 정확히 일치 (임의 숫자 일치 오검증 방지)
            parsed["total_verified"] = True
        elif layout_total == _items_sum(kept) - parsed["discount"] + parsed["delivery_fee"]:
            # VLM 합계가 라벨·품목합 모두와 어긋남 (과세물품가액을 합계로 착각하는 유형)
            # -> 라벨 값과 (품목합 - 할인 + 배달비)가 서로 일치하면 그 값을 채택
            parsed["total_amount"] = layout_total
            parsed["total_verified"] = True
            corrections.append(
                {
                    "field": "total_amount",
                    "before": str(t),
                    "after": str(layout_total),
                    "reason": "ocr_layout",
                }
            )
        else:
            parsed["total_verified"] = False
    elif layout_total is not None:
        parsed["total_amount"] = layout_total
        parsed["total_verified"] = True
        corrections.append(
            {
                "field": "total_amount",
                "before": "null",
                "after": str(layout_total),
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
        error = None
        if parsed is not None:
            parsed, corrections = _merge(parsed, ocr_lines)
        else:
            # ok=false 만 내려보내면 원인을 알 수 없다. 실측에서 토큰 한도로 JSON 이
            # 잘린 경우가 있었는데 응답만 보고는 구분이 안 됐다.
            error = (
                "VLM 응답을 JSON 으로 읽지 못했습니다 "
                f"(길이 {len(raw_text or '')}자, 끝: {(raw_text or '')[-40:]!r})"
            )

        return JSONResponse(
            {
                "ok": parsed is not None,
                "error": error,
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
