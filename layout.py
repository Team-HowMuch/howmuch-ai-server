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
# 실측 표기: -4,195 / -₩4,195(쿠팡이츠) / 10,500원(EGG DROP) / ₩4,900(StoryWay)
_MONEY = re.compile(r"^-?₩?\d{1,3}(?:,\d{3})+원?$|^-?₩?\d{3,7}원?$")
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

# 할인 어휘. 63장 실측에서 나온 것만 담았다.
# `에누리`가 `할인`만큼 자주 나오고, 홈푸드마트는 `$특매할인`, 신세계는 `직원 에누리20%`처럼 쓴다.
# `말인`·`합인`은 OCR 오독이다. 실측: `다품말인 20%`(신세계), `쿠폰말인`(하나로마트),
# `합인금액:`(홈푸드마트). `에누ㄹ`도 `에누리`가 자모로 깨진 것이다(이마트).
# 셋 다 한국어에 없는 표기라 오탐 위험이 거의 없다.
_DISCOUNT_WORD = re.compile(r"할인|말인|합인|에누리|에누ㄹ|쿠폰|특매|D/?C")

# 합계부의 할인 "요약" 라벨. 품목별 할인 줄들의 합계를 다시 적은 것이라 또 빼면 이중차감이다.
# 실측: `할인금액 : -1,510`(홈푸드마트), `총할인액 -12,000`·`쿠폰할인 -12,000`(하나로마트),
# `할인(에누리) -23,980`(홈플러스), `에누리계 -22,240`(신세계)
_DISCOUNT_TOTAL_LABEL = re.compile(r"할인금액|합인금액|총할인|할인계|할인액|에누리계|할인\(에누리\)|쿠폰할인|쿠폰말인")

# 할인처럼 보이지만 할인이 아닌 줄.
# `할인: 0, 현재잔액: 5,000` 은 농·축산물 할인지원금 한도 안내이고,
# `배달팁 할인` 은 배달팁 순액 안에서 이미 상계된 값이라 총 할인에 더하면 이중계상이다.
_DISCOUNT_DECOY = re.compile(r"현재잔액|잔액|배달팁할인|배달팁|바코드|이용권")

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
    #: 이 품목 줄에 귀속되는 할인액. 항상 0 이상이고, price 는 할인 **전** 금액이다.
    discount: int = 0
    #: 위 discount 를 이룬 개별 할인 금액들. VLM 이 같은 할인 줄을 sub_items 로도
    #: 올려보내기 때문에, 금액 단위로 대조해야 같은 돈을 두 번 빼지 않는다.
    discount_amounts: list[int] = field(default_factory=list)


@dataclass
class Layout:
    rows: list[Row]
    items: list[LayoutItem]
    total_amount: int | None
    #: 특정 품목에 귀속되지 않는 영수증 전체 단위 할인액. 항상 0 이상.
    discount: int = 0
    #: 합계부 할인 "요약" 줄들의 합(상계 전 원값). 요약줄은 품목별 할인의 재기재라
    #: 품목 합과 대조해야 하는데, VLM 이 같은 줄을 하위 옵션으로도 올려보내기 때문에
    #: 병합 단계에서 한 번 더 대조하려면 상계 전 값이 필요하다.
    summary_discount: int = 0


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
        return int(text.replace(",", "").replace("₩", "").replace("원", ""))
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


def _discount_kind(name: str | None, price: int | None) -> str | None:
    """할인 줄이면 종류를, 아니면 None을 돌려준다.

    반환값은 "item"(품목에 귀속) 또는 "total"(영수증 전체 요약)이다.

    금액이 0이면 할인이 아니다. `할인: 0`, `할  인  0`, `배달팁 할인 0` 처럼 값이 0인
    자리표시자가 실측 63장 중 여러 장에 있었다. 부호는 보지 않는다. 배달앱과 백화점은
    할인을 부호 없는 양수로 찍는다(`할인금액 3,000`, `할인금액 27,800`).
    """
    if not name or not price:
        return None
    compact = re.sub(r"\s+", "", name)
    if not _DISCOUNT_WORD.search(compact) or _DISCOUNT_DECOY.search(compact):
        return None
    return "total" if _DISCOUNT_TOTAL_LABEL.search(compact) else "item"


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
    summary_discount = 0  # 합계부 할인 요약줄의 값 (품목별 합과 대조해 이중계상을 막는다)

    for row in region:
        parsed = _parse_row(row)
        name, price, quantity = parsed["name"], parsed["price"], parsed["quantity"]
        compact_name = re.sub(r"\s+", "", name) if name else ""
        if compact_name and (_LABEL_NAME.search(compact_name) or _TOTAL_LABEL.search(compact_name)):
            continue  # 요약/전표/합계 라벨 줄은 품목도 pending도 아니다

        # 할인 줄은 하위 옵션이 아니라 discount 로 뺀다. 예전에는 음수 금액이라는 이유로
        # sub_items 에 들어갔는데, 그러면 합계부 요약줄(`할인금액 -1,510`)까지 직전 품목의
        # 옵션이 되어 품목 합에서 할인이 두 번 빠졌다.
        kind = _discount_kind(name, price)
        if kind == "total":
            summary_discount += abs(price)
            continue
        if kind == "item":
            if items:
                items[-1].discount += abs(price)
                items[-1].discount_amounts.append(abs(price))
            else:
                summary_discount += abs(price)  # 귀속할 품목이 없으면 전체 할인으로 본다
            continue

        indent = (row.x1 - base_x) / char_w if char_w else 0.0
        is_sub = name is not None and (
            bool(_SUB_PREFIX.match(name))
            or indent >= 1.5
            or (price is not None and price < 0)  # 할인 어휘가 없는 음수 줄
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

    # 품목 영역 밖의 할인 요약줄도 줍는다. 배달앱은 할인을 품목 영역이 끝난 뒤
    # (`소계금액` 다음) 부호 없는 양수로 찍어서 영역 안에서는 보이지 않는다.
    for row in rows[end:]:
        parsed = _parse_row(row)
        if _discount_kind(parsed["name"], parsed["price"]) is not None:
            summary_discount += abs(parsed["price"])

    # 요약값이 품목별 할인 합과 같으면 같은 돈을 두 번 적은 것이므로 버린다.
    # 다르면 요약이 더 큰 만큼만 전체 단위 할인으로 본다. 신세계처럼 요약줄이 품목별
    # 할인 중 일부만 집계하는 영수증이 있어, 요약값을 그대로 믿으면 오히려 줄어든다.
    item_discount_sum = sum(i.discount for i in items)
    receipt_discount = max(0, summary_discount - item_discount_sum)

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
    return Layout(
        rows=rows,
        items=items,
        total_amount=total,
        discount=receipt_discount,
        summary_discount=summary_discount,
    )
