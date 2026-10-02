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
import math
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

# 다른 결제 수단 줄. 합계 라벨(`결제금액`)이 붙어 있어도 분할 결제의 일부라 합계로 쓰지 않는다.
_TENDER_ROW = re.compile(r"포인트|상품권|캐시|기프트|마일리지|예치금")

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
# 배달비 줄의 할인(`배달팁 할인`, `배달비 할인`)은 delivery_fee 에서 상계하므로 여기서 또 세지 않는다.
_DISCOUNT_DECOY = re.compile(r"현재잔액|잔액|배달팁|배달비|배달료|배달요금|배송비|무료배달|바코드|이용권")

# 배달비 라벨 판정. 두 가지 모양을 인정한다(공백 제거 후).
#
# 1) 이름이 배달비 어휘로 **끝난다**: `03.배달비`, `ㄴ기본배달팁`, `맨드본 배달팁`(실측),
#    `배달팁 할인쿠폰`, `와우 무료배달`, `배달팁(기본)`, `배달비:`.
# 2) 이름이 배달비 어휘로 **시작**하고, 뒤에 붙은 한글이 수식어뿐이다: VLM 이 붙여 보내는
#    `배달팁 3,000원`, `배달팁(거리할증 1,000원 포함)`, `배달팁 기본`, `배달비 합계`.
#
# 포함만 되면 잡는 식은 안 된다. 온라인 쇼핑 영수증의 `[배송비무료] 제주감귤`,
# `사과 1박스(배송비포함)` 은 상품이고, 안내문 `배달 비대면 요청` 은 공백을 지우면
# `배달비대면요청` 이 된다 — 셋 다 배달비 어휘 뒤에 수식어가 아닌 한글이 남는다.
# OCR 이 두 줄을 한 행으로 합친 `03.배달비 --케이준양념감자(중)` 도 같은 이유로 걸러진다.
_FEE_WORD = r"(배달(비|팁|료|요금)|배송비|무료배달)"
_DELIVERY_FEE_END = re.compile(_FEE_WORD + r"(할인|무료|쿠폰)*(\([^)]{0,12}\))?[:：]?$")
_DELIVERY_FEE_START = re.compile("^" + _FEE_WORD)
_FEE_QUALIFIER = re.compile(r"기본|합계|할증|추가|거리|할인|무료|쿠폰|포함|금액")

# 내역 줄 표시. `기본배달팁`·`배달팁 기본`·`추가배달비` 는 상위 `배달팁` 의 내역이다.
# `ㄴ` 기호는 기준으로 못 쓴다. OCR 이 `ㄴ` 을 `L`·`A` 로 읽거나 떨어뜨린다(실측 `A 기본배달팁`).
_FEE_BREAKDOWN = re.compile(r"기본|할증|추가|거리")

# 배달비를 깎는 줄. `배달팁 할인 1,000`·`무료배달 -3,000` 은 부호와 상관없이 뺀다.
_DELIVERY_FEE_OFF = re.compile(r"할인|쿠폰|무료")


def delivery_fee_label(name) -> str | None:
    """배달비 줄이면 "parent"(상위)·"child"(내역)·"off"(감액), 아니면 None."""
    if not isinstance(name, str):
        return None
    compact = re.sub(r"\s+", "", name)
    core = re.sub(r"^[^가-힣]+", "", compact)
    start = _DELIVERY_FEE_START.match(core)
    if _DELIVERY_FEE_END.search(compact):
        pass
    elif start:
        rest = re.sub(r"\([^)]*\)", "", core[start.end():])
        rest = re.sub(r"\d[\d,.]*원?", "", rest)
        if re.search(r"[가-힣]", _FEE_QUALIFIER.sub("", rest)):
            return None
    else:
        return None
    if _DELIVERY_FEE_OFF.search(compact):
        return "off"
    if start and not _FEE_BREAKDOWN.search(core[start.end():]):
        return "parent"
    return "child"


# 좌표 복구에서 품목으로 되살리지 않을 줄(넓게 잡는다). 합쳐진 행이 품목으로 복구되면
# 배달비가 품목과 delivery_fee 양쪽에서 더해진다. 넓게 잡아 생기는 손해는 VLM 이 놓친
# 배송비 들어간 상품을 좌표로 못 되살리는 것뿐이다.
_DELIVERY_WORD = re.compile(r"배달(비(?!대면)|팁|료|요금)|배송비|무료배달")

# 하위 옵션 접두 기호
# `(선택)곱빼기`·`(더하기 선택(최대 5개))야채` 도 옵션 표기다.
_SUB_PREFIX = re.compile(r"^[-*+└ㄴ>›»▶►▸~]|^\((추가|선택|변경|더하기)|^옵션")
# 기호만으로 옵션이라고 믿을 만한 것. `-`·`*`·`~` 나 `(선택)` 은 안내문·장식에도 쓰여 약하다.
_STRONG_SUB_MARK = re.compile(r"^[+└ㄴ>›»▶►▸]")


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
    marked: bool = False  # 이름 앞에 옵션 기호(`ㄴ`·`+`·`-`…)가 찍혀 있었다(들여쓰기만으로 옵션이라 본 줄이 아니다)
    strong: bool = False  # 그 기호가 `ㄴ`·`+`·`>` 처럼 옵션에만 쓰이는 것이다


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
    #: 배달비 순액(배달팁 할인·무료배달 상계 후). 0 이상.
    #: None 이면 영수증에서 배달비 줄을 아예 보지 못했다는 뜻이다. 0(무료배달)과 구분해야
    #: 병합 단계에서 VLM 이 품목으로 올린 배달비를 믿을지 정할 수 있다.
    delivery_fee: int | None = None
    #: 품목 영역이 끝나는 줄 번호(rows 기준). 그 뒤는 합계·세금·결제 같은 요약부다.
    summary_start: int = 0


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


# 기울기 보정: 같은 줄에서 옆에 붙은 토큰 쌍의 기울기 중 가장 촘촘히 몰린 값을 쓴다.
_SKEW_MIN_DX = 80.0  # 이보다 가까운 두 토큰은 기울기를 재기엔 너무 짧다(px)
_SKEW_BANDWIDTH = 0.008  # 같은 기울기로 볼 허용 오차(tan, 약 0.46도)
_SKEW_MIN_SUPPORT = 5  # 이 기울기를 말하는 쌍이 이만큼은 있어야 믿는다
_SKEW_MIN_SHARE = 0.25  # 그리고 전체 쌍의 이 비율 이상이어야 한다
_SKEW_MIN_DEG = 0.8  # 이보다 덜 기울면 손대지 않는다
_SKEW_MAX_DEG = 12.0  # 이보다 많이 기울었다면 줄 단위 구조를 믿을 수 없다


def estimate_skew(tokens: list[Token]) -> float:
    """사진이 기운 정도를 tan 값으로 돌려준다(오른쪽이 아래로 내려가면 양수). 못 믿으면 0.

    영수증 한 줄은 OCR 이 바코드·단가·금액처럼 여러 토큰으로 쪼개는데, 기운 사진에서는
    그 토큰들의 y 중심이 오른쪽으로 갈수록 한쪽으로 밀린다. 그래서 줄 묶기가 오른쪽 금액을
    아래 줄 이름에 붙인다. 줄 간격이 20px 안팎일 때 기울기 1.6도만 돼도 금액 칸이 8px 밀려,
    "이름 줄은 위, 금액 줄은 아래" 패턴이 한 줄씩 어긋난다.
    """
    if len(tokens) < 8:
        return 0.0
    height = median(t.height for t in tokens)
    ordered = sorted(tokens, key=lambda t: (t.x1 + t.x2) / 2)
    slopes: list[float] = []
    for i, a in enumerate(ordered):
        ax = (a.x1 + a.x2) / 2
        best: tuple[float, float] | None = None
        for b in ordered[i + 1:]:
            bx = (b.x1 + b.x2) / 2
            if b.x1 < a.x2 - 0.2 * height or b.x1 - a.x2 > 4.0 * height or bx - ax < _SKEW_MIN_DX:
                continue
            dy = abs(b.y_center - a.y_center)
            if dy > 0.5 * min(a.height, b.height) + 0.1 * (bx - ax):
                continue
            if best is None or dy < best[0]:
                best = (dy, (b.y_center - a.y_center) / (bx - ax))
        if best is not None:
            slopes.append(best[1])
    if len(slopes) < _SKEW_MIN_SUPPORT:
        return 0.0
    support, centre = 0, 0.0
    for candidate in slopes:
        members = [s for s in slopes if abs(s - candidate) <= _SKEW_BANDWIDTH]
        if len(members) > support:
            support, centre = len(members), median(members)
    if support < _SKEW_MIN_SUPPORT or support < _SKEW_MIN_SHARE * len(slopes):
        return 0.0
    if not math.radians(_SKEW_MIN_DEG) <= abs(math.atan(centre)) <= math.radians(_SKEW_MAX_DEG):
        return 0.0
    return centre


def deskew(tokens: list[Token]) -> list[Token]:
    """품목 구간의 기울기를 펴서 같은 인쇄 줄의 토큰이 같은 y 에 오게 한다.

    합계부는 건드리지 않는다. 합계부는 `합계 ····· 5,300` 처럼 라벨과 금액이 멀리 떨어진 줄이
    많고 줄 간격이 넓어서, 기울기를 펴면 오히려 금액이 아래 줄 라벨에 붙는다(홈플러스 영수증 실측).
    품목 구간은 `이름 → 바코드·수량·금액` 이 20px 간격으로 촘촘히 이어져 기울기에 가장 약하다.
    안 기울었거나 품목 구간을 못 찾으면 그대로 돌려준다.
    """
    rows = group_rows(tokens)
    start, end = _find_item_region(rows)
    # 표 머리글(`상품명 단가 수량 금액`)을 찾았을 때만 한다. 못 찾으면 구간이 어디서 끝나는지도
    # 믿을 수 없어서(`합 계` 라벨이 깨지면 합계 줄까지 품목 구간이 된다) 기울기를 펴지 않는다.
    if start == 0 or end - start < 3:
        return tokens
    inside = {id(t) for row in rows[start:end] for t in row.tokens}
    region = [t for t in tokens if id(t) in inside]
    slope = estimate_skew(region)
    if not slope:
        return tokens
    pivot = median((t.x1 + t.x2) / 2 for t in region)
    shifted = []
    for t in tokens:
        if id(t) not in inside:
            shifted.append(t)
            continue
        dy = ((t.x1 + t.x2) / 2 - pivot) * slope
        shifted.append(Token(t.text, t.confidence, t.x1, t.y1 - dy, t.x2, t.y2 - dy))
    return shifted


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


def _money_values(row: Row) -> set[int]:
    return {v for v in (_money_value(tok.text) for tok in row.tokens) if v is not None}


def _has_value_digits(row: Row) -> bool:
    """이름 토큰이 아닌 곳에 숫자가 있는가. 금액을 찍었지만 OCR 이 깨뜨린 줄(`3.000`)을 가린다."""
    return any(re.search(r"\d", tok.text) and not _HANGUL.search(tok.text) for tok in row.tokens)


def _sum_fees(parents: list[int], children: list[int], offs: list[int]) -> int:
    """상위 줄이 있으면 상위만, 없으면 내역을 더하고 감액을 뺀다. 0 미만은 0.

    같은 금액의 상위 줄은 한 번만 센다. `배달팁 3,000` 이 두 군데 찍히거나 VLM 이 품목과
    옵션으로 두 번 올린 경우다. 서로 다른 배달비가 우연히 같은 금액일 가능성보다
    같은 줄을 두 번 읽었을 가능성이 훨씬 크다.
    """
    base = sorted(set(parents)) if parents else children
    return max(0, sum(base) + sum(offs))


def _delivery_fee(rows: list[Row]) -> int | None:
    """영수증 전체에서 배달비 순액을 구한다.

    금액을 읽은 배달비 줄이 없으면 순액은 None 이다. 0 으로 치면 VLM 이 제대로 읽은
    배달비까지 버리게 되므로, None 을 돌려 병합 단계가 VLM 값을 쓰게 한다.

    배민 주문전표는 배달비를 이렇게 찍는다.

        배달팁                 0
        ㄴ기본배달팁       4,100
                          -4,100     <- 무료배달이면 바로 아래 음수 줄로 상계

    - 배달비 줄 금액에 **바로 뒤따르는 이름 없는 음수 줄**을 더한다. 양수는 더하지 않는다.
      이름 없는 양수 줄은 대개 다음 요약값(합계 등)이 줄바꿈된 것이다.
    - 라벨 줄에 숫자가 아예 없으면 바로 **아래** 이름 없는 양수 줄을 값으로 빌린다. 라벨과
      값이 y 로 갈라져 다른 행이 되는 경우다. 단,
        * 라벨 줄에 숫자가 있는데 못 읽었으면(`3.000`) 빌리지 않는다. 그 아래 값은 남의 것이다.
        * 그 다음 줄이 값 없는 라벨(`총결제금액`)이면 빌리지 않는다. 그 라벨의 값이다.
        * 위 줄은 보지 않는다. 위의 이름 없는 값은 대개 앞 요약줄(주문금액)의 값이다.
    - 한 행에 서로 다른 금액이 둘 이상이면 두 줄이 합쳐진 것이라 버린다.
    - 상위·내역·감액 구분은 delivery_fee_label, 합산은 _sum_fees.
    """
    parents: list[int] = []
    children: list[int] = []
    offs: list[int] = []
    for i, row in enumerate(rows):
        parsed = _parse_row(row)
        kind = delivery_fee_label(parsed["name"])
        if kind is None:
            continue
        if len(_money_values(row)) > 1:
            continue  # 두 줄이 한 행으로 합쳐졌다

        amount = parsed["price"]
        tail = i + 1
        if amount is None and not _has_value_digits(row) and i + 1 < len(rows):
            value = _parse_row(rows[i + 1])
            after = _parse_row(rows[i + 2]) if i + 2 < len(rows) else None
            # 다음 줄이 숫자가 아예 없는 라벨이면 이 값은 그 라벨 몫이다. 라벨에 깨진 숫자라도
            # 있으면(`배달팁 할인 1.000`) 제 값을 들고 있는 것이라 남의 값을 빌리지 않는다.
            owned_by_next_label = (
                after is not None
                and after["name"] is not None
                and after["price"] is None
                and not _has_value_digits(rows[i + 2])
            )
            if value["name"] is None and value["price"] and value["price"] > 0 and not owned_by_next_label:
                amount = value["price"]
                tail = i + 2
        if amount is None:
            continue  # 금액을 못 읽었다(`4.100`, `1;000` 등)

        for j in range(tail, min(tail + 2, len(rows))):
            follow = _parse_row(rows[j])
            if follow["name"] is not None or follow["price"] is None or follow["price"] >= 0:
                break
            amount += follow["price"]

        if kind == "off":
            offs.append(-abs(amount))
        elif kind == "parent":
            parents.append(amount)
        else:
            children.append(amount)

    if not (parents or children or offs):
        return None
    return _sum_fees(parents, children, offs)


def analyze_receipt(ocr_lines: list[dict]) -> Layout:
    tokens = deskew(_to_tokens(ocr_lines))
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
    delivery_fee = _delivery_fee(rows)

    after_fee = False  # 바로 앞이 배달비 줄이면 그 아래 음수는 무료배달 상계다
    negatives: list[int] = []  # 품목 영역에서 지금까지 본 음수 금액들
    prev_was_sub = False  # 바로 앞 줄이 하위 옵션 줄이었나
    prev_sub_x = 0.0  # 바로 앞 옵션 줄이 시작한 x
    prev_orphan_sub = False  # 바로 앞 줄이 붙일 품목이 없는 옵션 줄이었나
    for ri, row in enumerate(region):
        was_sub, was_orphan = prev_was_sub, prev_orphan_sub
        prev_was_sub = prev_orphan_sub = False
        nxt = _parse_row(region[ri + 1]) if ri + 1 < len(region) else None
        if _DELIVERY_WORD.search(row.compact):
            # 배달비는 품목이 아니라 delivery_fee 로 따로 센다. pending 도 비운다. 안 비우면
            # 앞의 가격 없는 품목명 줄(`단무지 많이`)이 배달비 아래 값 행(`3,000`)이나
            # 상계 행(`-3,000`)을 제 금액으로 가져가 품목이 된다. 비우고 나면 그 행들은
            # 주인 없는 가격 줄이라 아래에서 버려진다.
            pending = None
            after_fee = True
            continue
        parsed = _parse_row(row)
        name, price, quantity = parsed["name"], parsed["price"], parsed["quantity"]
        if name is not None:
            after_fee = False

        # 이름 없는 음수 줄은 할인이다. 홈푸드마트는 `$특매할인 -400` 의 라벨을 OCR 이 자주
        # 떨어뜨려 `1  -400  -400` 만 남는다. 지금까지 품목에 붙인 할인의 합과 같으면
        # 품목별 할인을 다시 적은 요약줄(`할인금액 -1,910` 의 라벨이 떨어진 것)이다.
        if price is not None and price < 0:
            restated = _is_restated_discount(price, negatives, named=name is not None)
            negatives.append(abs(price))
            if name is None or restated:
                if after_fee:
                    continue  # 무료배달 상계 — 배달비 순액에서 이미 셌다
                if restated or not items:
                    summary_discount += abs(price)
                else:
                    items[-1].discount += abs(price)
                    items[-1].discount_amounts.append(abs(price))
                pending = None
                continue
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

        # 기호가 떨어진 옵션 줄: 앞 옵션 줄보다 기호 폭만큼(한 글자쯤) 안쪽에서 시작한다. 옵션이 있는
        # 품목이 연달아 오면 품목 줄도 옵션 줄 사이에 끼므로(`김밥 / +치즈 / 라면 / +계란`), 끼어 있다는 것만으로는
        # 옵션이라 하지 않고 안쪽에서 시작할 때만 그렇게 본다.
        if (not is_sub and name is not None and price is not None and price > 0 and was_sub and char_w):
            dx = row.x1 - prev_sub_x
            sandwiched = nxt is not None and nxt["name"] is not None and bool(_SUB_PREFIX.match(nxt["name"]))
            if dx >= 0.6 * char_w or (sandwiched and dx >= 0.3 * char_w):
                is_sub = True

        if is_sub:
            if items:
                items[-1].sub_items.append(
                    LayoutSubItem(
                        name=re.sub(r"^[-*+└ㄴ>›»▶►▸~\s]+", "", name),
                        price=price,
                        marked=bool(_SUB_PREFIX.match(name)),
                        strong=bool(_STRONG_SUB_MARK.match(name)),
                    )
                )
                prev_was_sub = True  # 붙일 품목이 없던 줄은 옵션 블록이 아니다
                prev_sub_x = row.x1
            else:
                prev_orphan_sub = True
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
            # 부모를 못 찾은 옵션 줄 바로 뒤의 이름 조각(`추가`, `도 추가`)은 그 옵션 이름이 줄바꿈된 것이다
            pending = None if was_orphan else parsed
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
            elif price > 0:
                # 이름을 못 읽은 가격 줄도 자리는 남긴다. 이름 없이 금액만 있는 품목이라 복구는
                # 안 하지만, 바로 아래 할인 줄이 엉뚱한 위 품목에 붙지 않고, 병합 단계가 금액으로
                # VLM 품목과 짝지어 할인을 제 품목에 붙일 수 있다.
                items.append(LayoutItem(name=None, quantity=quantity, price=price))
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
        if _TENDER_ROW.search(compact):
            continue  # `포인트결제금액 3,000`·`상품권 5,000` 은 분할 결제의 일부지 합계가 아니다
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
        delivery_fee=delivery_fee,
        summary_start=end,
    )


def _is_restated_discount(price: int, negatives: list[int], named: bool) -> bool:
    """이 음수가 위에서 본 할인들의 합을 다시 적은 요약값인가.

    할인이 하나뿐이면 같은 금액이 품목마다 우연히 반복될 수 있다(`-400`, `-400`). 그래서
    하나짜리는 라벨이 붙은 줄(`과 -1,510` = 깨진 `할인금액`)일 때만 요약으로 본다.
    """
    if not negatives or abs(price) != sum(negatives):
        return False
    return len(negatives) >= 2 or named
