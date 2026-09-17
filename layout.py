"""OCR box 좌표 기반 영수증 구조 복원.

PP-OCR이 주는 box 좌표로 다음을 복원한다.

1. 줄 그룹핑: y중심 거리가 박스 높이 중앙값의 0.6배 이내면 같은 줄
2. 품목 영역 판별: 헤더 줄(상품명/수량/금액) 다음부터 요약 줄(소계/합계/가액...) 전까지
3. 줄 역할 분류: 완결 품목 / 가격 줄 / 품목명 줄 / 하위 옵션 / 무시
4. 품목-수량-금액 연결: 한국 영수증은 "품목명 줄 -> 바코드·수량·금액 줄" 인접 패턴이
   흔해서, 이름 없는 가격 줄은 직전 품목명 줄에 붙인다
5. 합계 추출: "합계/총액" 라벨이 있는 줄의 최우측 금액

들여쓰기는 한국 영수증에서 드물어 보조 신호로만 쓴다(문자 폭 단위 정규화).
"""
import re
from dataclasses import dataclass, field
from statistics import median

_BARCODE = re.compile(r"^\d{8,}$")
_MONEY = re.compile(r"^-?\d{1,3}(?:,\d{3})+$|^-?\d{3,7}$")
_PLAIN_INT = re.compile(r"^\d{1,3}$")
_HANGUL = re.compile(r"[가-힣]")

# 품목 표 헤더: "상품명  수량  금액" 형태
_HEADER_NAME = re.compile(r"상\s*품\s*명|품\s*명|메\s*뉴")
_HEADER_COLS = re.compile(r"수\s*량|단\s*가|금\s*액")

# 품목 영역이 끝나는 요약 줄 (공백 제거 후 대조).
# "할인"은 품목 아래 행사할인 줄, "과세/면세"는 품목명 접미(예: 유니클로(과세))로도
# 나오므로 단독으로 쓰지 않고 "과세물품/면세물품" 형태만 매칭한다.
# 주문금액/배달비/총결제는 배달앱(쿠팡이츠 등) 영수증의 요약 시작 줄이다.
_REGION_END = re.compile(
    r"소계|합계|과세물품|면세물품|부가세|가액|공급가|봉사료|총구매|총액"
    r"|주문금액|배달비|총결제|결제금액"
)

# 합계로 인정하는 라벨 (공백 제거 후 대조). 선결제는 배민 주문서의 결제액 줄.
_TOTAL_LABEL = re.compile(r"합계|총액|총구매|총결제|결제금액|받을금액|선결제")

# 강한 라벨이 없을 때만 쓰는 보조 합계 라벨 (배달 영수증의 주문금액 = 품목 합)
_TOTAL_LABEL_WEAK = re.compile(r"주문금액")

# 품목이 될 수 없는 금액 라벨 줄 (카드 전표의 판매금액/부가세 등, 공백 제거 후 대조)
_LABEL_NAME = re.compile(
    r"판매금액|받은금액|받을금액|거스름|매출표|승인번호|부가세|봉사료|공급가"
    r"|합계|소계|총액|총구매|과세물품|면세물품|가액"
)

# 하위 옵션 접두 기호
_SUB_PREFIX = re.compile(r"^[-*+└ㄴ>›»▶►▸~]|^\(추가|^옵션")


@dataclass
class Token:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class Row:
    tokens: list[Token]

    @property
    def x1(self) -> float:
        return self.tokens[0].x1

    @property
    def y_center(self) -> float:
        return sum(t.y_center for t in self.tokens) / len(self.tokens)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def compact(self) -> str:
        return re.sub(r"\s+", "", self.text)


@dataclass
class LayoutSubItem:
    name: str
    price: int | None


@dataclass
class LayoutItem:
    name: str | None
    quantity: int | None
    price: int | None
    sub_items: list[LayoutSubItem] = field(default_factory=list)
    name_confidence: float = 1.0


@dataclass
class Layout:
    rows: list[Row]
    items: list[LayoutItem]
    total_amount: int | None


def _to_tokens(ocr_lines: list[dict]) -> list[Token]:
    tokens = []
    for line in ocr_lines:
        box = line.get("box")
        if not box:
            continue
        tokens.append(
            Token(
                text=line["text"],
                confidence=float(line.get("confidence", 1.0)),
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )
        )
    return tokens


def group_rows(tokens: list[Token]) -> list[Row]:
    """y중심 클러스터링으로 토큰을 줄 단위로 묶는다.

    임계값은 전체가 아니라 지금 줄의 로컬 박스 높이 기준이다. 배달앱 영수증처럼
    옵션 줄 폰트가 작고 줄 간격이 좁으면(간격 ~17px < 전체 중앙값 0.6배)
    전역 기준으로는 서로 다른 줄이 병합된다.
    """
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: t.y_center)
    rows: list[list[Token]] = [[ordered[0]]]
    for token in ordered[1:]:
        current = rows[-1]
        center = sum(t.y_center for t in current) / len(current)
        row_h = median(t.height for t in current)
        if abs(token.y_center - center) < 0.6 * min(row_h, token.height):
            current.append(token)
        else:
            rows.append([token])
    return [Row(tokens=sorted(row, key=lambda t: t.x1)) for row in rows]


def _money_value(text: str) -> int | None:
    if _BARCODE.match(text):
        return None
    if _MONEY.match(text):
        return int(text.replace(",", ""))
    return None


def _parse_row(row: Row) -> dict:
    """줄에서 금액(최우측)·수량·이름 토큰을 분리한다."""
    price = None
    price_token = None
    for token in reversed(row.tokens):
        value = _money_value(token.text)
        if value is not None:
            price, price_token = value, token
            break

    quantity = None
    qty_token = None
    for token in row.tokens:
        if token is price_token:
            continue
        if _PLAIN_INT.match(token.text) and 1 <= int(token.text) <= 999:
            quantity, qty_token = int(token.text), token

    name_tokens = [
        t
        for t in row.tokens
        if t not in (price_token, qty_token) and not _BARCODE.match(t.text) and _HANGUL.search(t.text)
    ]
    name = " ".join(t.text for t in name_tokens) if name_tokens else None
    name_conf = min((t.confidence for t in name_tokens), default=1.0)
    return {"price": price, "quantity": quantity, "name": name, "name_conf": name_conf}


def _find_item_region(rows: list[Row]) -> tuple[int, int]:
    """헤더 줄 다음부터 요약 줄 전까지를 품목 영역으로 본다."""
    start = 0
    for i, row in enumerate(rows):
        compact = row.compact
        if _HEADER_NAME.search(row.text) and _HEADER_COLS.search(row.text):
            start = i + 1
            break
    end = len(rows)
    for i in range(start, len(rows)):
        if _REGION_END.search(rows[i].compact):
            end = i
            break
    return start, end


def _char_width(tokens: list[Token]) -> float:
    widths = [
        (t.x2 - t.x1) / len(t.text)
        for t in tokens
        if t.text and _HANGUL.search(t.text)
    ]
    return median(widths) if widths else 1.0


def analyze_receipt(ocr_lines: list[dict]) -> Layout:
    tokens = _to_tokens(ocr_lines)
    rows = group_rows(tokens)
    if not rows:
        return Layout(rows=[], items=[], total_amount=None)

    start, end = _find_item_region(rows)
    region = rows[start:end]

    char_w = _char_width(tokens)
    name_rows_x1 = [r.x1 for r in region if _parse_row(r)["name"]]
    base_x = min(name_rows_x1) if name_rows_x1 else 0.0

    items: list[LayoutItem] = []
    pending: dict | None = None  # 가격 없는 품목명 줄 (다음 가격 줄을 기다림)

    for row in region:
        parsed = _parse_row(row)
        name, price, quantity = parsed["name"], parsed["price"], parsed["quantity"]
        compact_name = re.sub(r"\s+", "", name) if name else ""
        if compact_name and (_LABEL_NAME.search(compact_name) or _TOTAL_LABEL.search(compact_name)):
            continue  # 요약/전표/합계 라벨 줄은 품목도 pending도 아니다
        indent = (row.x1 - base_x) / char_w if char_w else 0.0
        is_sub = name is not None and (
            bool(_SUB_PREFIX.match(name))
            or indent >= 1.5
            or (price is not None and price < 0)  # 행사할인 등 음수 금액 줄
        )

        if is_sub:
            if items:
                items[-1].sub_items.append(
                    LayoutSubItem(name=re.sub(r"^[-*+└>›»▶►▸~\s]+", "", name), price=price)
                )
            continue
        if name and price is not None:
            items.append(
                LayoutItem(
                    name=name,
                    quantity=quantity,
                    price=price,
                    name_confidence=parsed["name_conf"],
                )
            )
            pending = None
        elif name:
            pending = parsed
        elif price is not None:
            if pending is not None:
                items.append(
                    LayoutItem(
                        name=pending["name"],
                        quantity=quantity or pending["quantity"],
                        price=price,
                        name_confidence=pending["name_conf"],
                    )
                )
                pending = None
            # 직전 품목명 줄이 없으면 이름 없는 가격 줄이므로 버린다
        # 이름도 가격도 없는 줄(바코드 조각 등)은 무시

    total = None
    weak_total = None
    for row in rows:
        compact = row.compact
        if _TOTAL_LABEL.search(compact):
            value = _parse_row(row)["price"]
            # "받을금액: 0"(미수금) 같은 0원 라벨이 실제 합계를 덮어쓰지 않게 한다
            if value is not None and (value > 0 or total is None):
                total = value
        elif _TOTAL_LABEL_WEAK.search(compact):
            value = _parse_row(row)["price"]
            if value is not None and (value > 0 or weak_total is None):
                weak_total = value
    if total is None:
        total = weak_total
    return Layout(rows=rows, items=items, total_amount=total)
