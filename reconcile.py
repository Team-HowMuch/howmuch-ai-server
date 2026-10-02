"""합계를 검산 기준으로 품목·할인·배달비를 맞춘다.

VLM 과 좌표 복원은 줄 하나하나를 읽다 보니 두 종류의 실수를 낸다.

- 품목이 아닌 줄이 품목·옵션으로 들어온다. 라벨이 깨진 합계·세금·결제 줄(`합할 계 2,400`,
  `과세물풀: 29,219`, `거자이체 4,150`)은 이름 규칙을 빠져나간다.
- 있어야 할 돈을 빠뜨린다. 라벨이 떨어진 할인 줄, 번진 배달팁, 배민 `결제 금액 상세` 의
  채널할인처럼 품목 영역 밖에 찍힌 할인.

영수증에는 이 둘을 가려낼 검산식이 이미 찍혀 있다. 손님이 실제로 낸 돈(결제금액)은
Σ(품목 + 옵션 − 품목할인) − 영수증할인 + 배달비 와 같아야 한다. 그래서 마지막에

1. 이름만 봐도 확실히 품목이 아닌 줄(라벨)을 지우고,
2. 실제 결제액 후보를 OCR 에서 찾은 뒤,
3. 의심 가는 줄을 빼거나 OCR 에 찍혀 있는 할인·배달비를 더하는 **가장 작은 수정**으로
   검산식이 맞는지 찾는다.

원칙
- 수정은 OCR 에 실제로 찍힌 금액에서만 고른다(지어내지 않는다).
- 같은 비용으로 서로 다른 결제액이 맞으면 모호하므로 아무것도 바꾸지 않는다.
- 결제액이 OCR 에 없고 VLM 만 말한 값이면 검산이 맞아도 "검증됨" 으로 표시하지 않는다.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 품목이 아닌 이름

# 상품명에 나올 일이 없는 요약·결제 라벨(한글만 남긴 뒤 대조). 들어 있으면 바로 지운다.
_LABEL_DEFINITE = re.compile(
    r"부가세|부가가치세|과세물품|면세물품|과세물|면세물|공급가액|총상품금액|상품금액|주문금액|주무금액"
    r"|결제금액|카드결제|간편결제|카드지불|신용카드|계좌이체|현재잔액|거스름|승인번호|승인금액|일시불"
    r"|주문번호|청구금액|받을금액|받은금액|내실금액|쿠팡캐시|소계금액|합계금액|적립포인트|받은포인트|할인합계"
)
# 이름 전체가 이것뿐이면 라벨이다. `합할 계`·`부 세` 처럼 깨진 모양도 여기서 잡는다.
_LABEL_WHOLE = re.compile(
    r"(받은|적립|사용|보유|잔여|가용)?포인트|과세|면세|부세|가세|부가|합.{0,2}계.?|총액|소계|합계|공급가"
)
# 라벨일 가능성이 있는 어휘. 상품명에도 나올 수 있어(`종합계란`, `포인트 스티커`) 바로 지우지 않고,
# 지워야 검산이 맞을 때만 지운다.
_LABEL_PROBABLE = re.compile(r"합계|결제|지불|이체|잔액|할부|포인트|카드|금액$|캐시|현금")
# `부가세(VAT):`, `키 류: 삼성카드` 처럼 콜론 앞이나 뒤에 라벨 어휘가 붙으면 라벨이다.
# 콜론만으로는 판단하지 않는다(`샷추가: 1샷`, `[행사]콜라:500ml`).
_LABEL_COLON = re.compile(
    r"(부가|과세|면세|공급|합.?계|결제|지불|이체|잔액|포인트|승인|일시불|개월|번호|청구|받을|받은|내실|거스름|VAT)"
    r"[^:：]{0,6}[:：]|[:：].{0,8}(카드|은행|페이|일시불)"
)
_VAT_WORD = re.compile(r"(?<![A-Za-z])VAT(?![A-Za-z])")


def label_level(name) -> int:
    """2: 확실한 라벨, 1: 라벨일 수 있음(검산으로 판단), 0: 상품명."""
    if not isinstance(name, str) or not name.strip():
        return 0
    raw = name.strip()
    hangul = _hangul(raw)
    if _LABEL_COLON.search(raw) or _VAT_WORD.search(raw):
        return 2
    if hangul and (_LABEL_DEFINITE.search(hangul) or _LABEL_WHOLE.fullmatch(hangul)):
        return 2
    if hangul and _LABEL_PROBABLE.search(hangul):
        return 1
    return 0


def is_label_name(name) -> bool:
    """확실히 영수증 요약·결제 줄의 라벨이면 True."""
    return label_level(name) == 2


def drop_label_rows(items: list[dict], corrections: list) -> None:
    """품목·옵션 중 확실한 라벨 이름인 것을 지운다(제자리)."""
    kept = []
    for item in items:
        if is_label_name(item.get("name")):
            corrections.append(_corr("items", f"{item.get('name')} {item.get('price')}", "(삭제)", "label_dropped"))
            continue
        subs = []
        for sub in item.get("sub_items") or []:
            if is_label_name(sub.get("name")):
                corrections.append(
                    _corr("item.sub_items", f"{item.get('name')} > {sub.get('name')} {sub.get('price')}",
                          "(삭제)", "label_dropped")
                )
                continue
            subs.append(sub)
        item["sub_items"] = subs
        kept.append(item)
    items[:] = kept


# ---------------------------------------------------------------- OCR 에서 결제액·할인 후보

# 손님이 실제로 낸 돈을 말하는 라벨(공백 제거 후 대조).
_PAID_LABEL = re.compile(
    r"결제금액(?!상세)|총결제|결제액|받을금액|카드결제|카드지불|신용카드|승인금액|내실금액|선결제|바로결제|간편결제"
)
_PAID_SECTION = re.compile(r"결제금액상세|결제내역")
_CARD_SLIP = re.compile(r"일시불\s*/\s*(\d{1,3}(?:,\d{3})+)")
_DISCOUNT_WORD = re.compile(r"할인|말인|합인|에누리|쿠폰|특매")
# 결제 수단(돈을 깎는 게 아니라 다른 방법으로 낸 것). 할인 후보에서 빼고, 분할 결제 판단에 쓴다.
_TENDER_WORD = re.compile(r"포인트|상품권|캐시|기프트|마일리지|예치금|선불|현금|머니")
# 결제에 쓰지 않은 포인트(적립·잔여 안내). 결제 수단 합계에도, 할인에도 넣지 않는다.
_NOT_SPENT = re.compile(r"적립|잔여|가용|보유|예정|발생|누적|잔액")
_DELIVERY_RECEIPT = re.compile(r"배달|배민|쿠팡이츠|요기요|땡겨요")
_FEE_WORD = re.compile(r"배달팁|배달비|배달료|배달요금|배송비")
_FEE_NOT_FEE = re.compile(r"요청|주소|메모|라이더|문앞|도착|시간|전화|기사|주문서|전표|고객센터|완료|의민족|예정|안내")
# 금액 자리가 비거나 깨진 모양(`--`, `;;`, `3,0O0`). 전화번호·시각·날짜는 아니다.
_PLACEHOLDER = re.compile(r"[-–—~=_.,;:'`]{1,6}")
_BROKEN_AMOUNT = re.compile(r"(?=.*\d)(?=.*[A-Za-z])[\dA-Za-z,.;']{2,8}")

# 깨진 금액 표기. 후보로만 쓴다(검산이 맞아야 채택된다).
# `1;000`·`1.000`·`W4.195` 는 구분자가 깨진 것, `-1,00`·`2,00` 은 끝자리 0 이 떨어진 것이다.
_FUZZY_FULL = re.compile(r"(?<![\d])(-?)[W₩]?(\d{1,3}(?:[,.;:'`]\d{3})+)(?![\d])")
_FUZZY_SHORT = re.compile(r"(?<![\d,.])(-?)[W₩]?(\d{1,3})[,.;](\d{2})(?![\d,.])")


def fuzzy_amounts(text: str) -> list[int]:
    """깨진 표기까지 포함해 금액으로 읽을 수 있는 값들(부호 포함). 구분자 없는 숫자는 뺀다."""
    out = []
    for sign, digits in _FUZZY_FULL.findall(text):
        value = int(re.sub(r"\D", "", digits))
        out.append(-value if sign else value)
    for sign, head, tail in _FUZZY_SHORT.findall(text):
        value = int(head + tail) * 10
        out.append(-value if sign else value)
    return [v for v in out if v]


def _row_money(row) -> list[int]:
    """행의 금액들(토큰 단위로 정확히 금액 모양인 것만)."""
    from layout import _money_value

    return [v for v in (_money_value(t.text) for t in row.tokens) if v is not None]


def _formatted_money(row) -> list[int]:
    """쉼표·₩·원이 붙은 금액만. 카드번호 끝자리(`**** 1000`)·전화번호 조각을 결제액으로 보지 않는다."""
    from layout import _money_value

    out = []
    for tok in row.tokens:
        if re.search(r"[,₩원]", tok.text):
            value = _money_value(tok.text)
            if value is not None:
                out.append(value)
    return out


@dataclass
class Evidence:
    paid: list[int] = field(default_factory=list)  # 실제 결제액 후보(결제 라벨 근거)
    tenders: list[int] = field(default_factory=list)  # 포인트·상품권 등 다른 결제 수단 금액
    discounts: Counter = field(default_factory=Counter)  # OCR 에 찍힌 할인 금액 후보
    summary: set[int] = field(default_factory=set)  # 요약부에 찍힌 금액들
    amounts: set[int] = field(default_factory=set)  # 영수증 전체에 찍힌 금액들(절댓값)
    amount_counts: Counter = field(default_factory=Counter)  # 금액별로 몇 줄에 찍혔나
    delivery: bool = False
    #: 요약부에 배달비 줄이 있는데 금액을 못 읽었다(번짐). 차액을 배달비로 채워도 되는 유일한 경우.
    fee_unread: bool = False
    #: 요약부에 두 번 이상 찍힌 금액. 합계는 보통 `합계`·`카드결제`·승인 금액으로 여러 번 찍힌다.
    repeated: set[int] = field(default_factory=set)
    #: 배민 `결제 금액 상세` 에서 결제액 아래 찍힌 할인들의 합(채널할인·배달앱할인).
    section_discount: int = 0
    #: 라벨 없는 양수 금액(라벨이 깨진 할인 줄 후보). 근거가 약해 비용을 높게 둔다.
    unlabeled_positive: set[int] = field(default_factory=set)
    #: 요약부 행들의 한글(라벨일 수 있는 이름이 정말 요약부에 찍혔는지 볼 때 쓴다)
    summary_text: list[str] = field(default_factory=list)


def _is_unread_fee_row(row) -> bool:
    """배달비 줄인데 금액 자리가 비었거나 깨졌다(`알종배달 --`, `배달팁 ;;`).

    라벨은 배달비 어휘(`배달팁`·`배달비`…)거나, 짧게 깨진 `…배달` 이어야 한다. 안내문·고객센터·
    시각 줄(`배달의민족 고객센터 1600-0987`, `배달완료 19:42`)은 아니다.
    """
    hangul = _hangul(row.text)
    if _FEE_NOT_FEE.search(hangul):
        return False
    if not (_FEE_WORD.search(hangul) or ("배달" in hangul and len(hangul) <= 5)):
        return False
    if fuzzy_amounts(row.text):
        return False
    return any(
        _PLACEHOLDER.fullmatch(t.text) or _BROKEN_AMOUNT.fullmatch(t.text)
        for t in row.tokens
        if not _hangul(t.text)
    )


def collect_evidence(layout, ocr_lines: list[dict]) -> Evidence:
    ev = Evidence()
    rows = layout.rows
    ev.delivery = any(_DELIVERY_RECEIPT.search(line.get("text") or "") for line in ocr_lines)
    if layout.delivery_fee is None:
        ev.fee_unread = any(_is_unread_fee_row(row) for row in rows[layout.summary_start:])

    in_section = False  # 배민 `결제 금액 상세` 아래
    section_paid_seen = False
    summary_counts: Counter = Counter()
    for i, row in enumerate(rows):
        compact = row.compact
        hangul = _hangul(row.text)
        money = _row_money(row)
        fuzzy = fuzzy_amounts(row.text)
        ev.amounts.update(abs(v) for v in money)
        ev.amount_counts.update(abs(v) for v in set(money))
        if i >= layout.summary_start:
            ev.summary.update(abs(v) for v in money)
            summary_counts.update(abs(v) for v in set(money))
            ev.summary_text.append(hangul)
        labeled = bool(label_level(row.text) or _PAID_LABEL.search(compact)
                       or (i >= layout.summary_start and _TENDER_WORD.search(hangul)))
        if not labeled:
            ev.unlabeled_positive.update(v for v in money if v > 0)

        if _PAID_SECTION.search(compact):
            in_section, section_paid_seen = True, False
            continue
        if in_section and money:
            if not section_paid_seen:
                # 상세의 첫 금액이 실제 결제액이고, 그 아래는 결제액을 줄인 할인들이다.
                ev.paid.append(abs(money[-1]))
                section_paid_seen = True
            elif _TENDER_WORD.search(hangul) and not _DISCOUNT_WORD.search(hangul):
                ev.tenders.append(abs(money[-1]))
            else:
                ev.discounts[abs(money[-1])] += 1
                ev.section_discount += abs(money[-1])
            continue

        # 결제 수단 줄은 요약부에만 있다. 품목 영역의 `캐시미어 머플러`·`포인트 니트` 는 상품이다.
        if (i >= layout.summary_start and _TENDER_WORD.search(hangul)
                and not _DISCOUNT_WORD.search(hangul)):  # `포인트할인` 은 할인이다
            if not _NOT_SPENT.search(hangul):
                ev.tenders.extend(abs(v) for v in money)
            continue  # 포인트·상품권은 할인이 아니다
        slip = _CARD_SLIP.search(row.text)
        if slip:
            ev.paid.append(int(slip.group(1).replace(",", "")))
        if _PAID_LABEL.search(compact):
            positive = [v for v in (_formatted_money(row) or [v for v in fuzzy if v > 0]) if v > 0]
            if positive:
                ev.paid.append(positive[-1])

        fee_row = bool(_FEE_WORD.search(hangul) or "배달" in hangul)
        if fee_row:
            continue
        for v in set(money) | set(fuzzy):
            if v < 0:
                ev.discounts[-v] += 1
            elif _DISCOUNT_WORD.search(hangul):
                ev.discounts[v] += 1
    ev.repeated = {v for v, n in summary_counts.items() if n >= 2 and v >= 1000}
    if ev.section_discount and ev.discounts[ev.section_discount] == 0:
        # 상세 할인이 여러 줄(`채널할인 1,000` + `배달앱할인 1,000`)이면 합계도 한 번에 더할 수 있게 한다
        ev.discounts[ev.section_discount] += 1
    return ev


# ---------------------------------------------------------------- 검산

def line_total(item: dict) -> int:
    return _int(item.get("price")) + sum(_int(s.get("price")) for s in item.get("sub_items") or []) - _int(
        item.get("discount")
    )


def computed_total(items: list[dict], discount: int, fee: int) -> int:
    return sum(line_total(i) for i in items) - discount + fee


# 품목 하나를 통째로 바꾸는 수정. 한 품목에 둘 이상 겹치면 안 된다.
_ITEM_LEVEL = ("drop_item", "opts_included", "price_is_net", "drop_item_discount", "unit_price", "layout_price")
# 강한 할인 근거(라벨·음수·결제상세)로 더하는 할인의 비용 상한
_STRONG_DISCOUNT_COST = 1.1


@dataclass(frozen=True)
class Edit:
    kind: str
    cost: float
    item: int = -1
    sub: int = -1
    amount: int = 0
    ref: int = -1  # add_item 이 되살릴 후보(parsed["_restorable"])의 번호

    def delta(self, items: list[dict]) -> int:
        """적용하면 계산 합계가 얼마나 바뀌는가."""
        if self.kind == "drop_item":
            return -line_total(items[self.item])
        if self.kind == "drop_sub":
            return -_int(items[self.item]["sub_items"][self.sub].get("price"))
        if self.kind == "opts_included":
            return -sum(_int(s.get("price")) for s in items[self.item].get("sub_items") or [])
        if self.kind == "price_is_net":
            return -_int(items[self.item].get("price"))
        if self.kind == "add_discount":
            return -self.amount
        if self.kind == "drop_item_discount":
            return _int(items[self.item].get("discount"))
        if self.kind in ("unit_price", "layout_price", "add_item"):
            return self.amount
        raise ValueError(self.kind)


def _tax_split(total: int) -> set[int]:
    """합계에서 나오는 부가세·공급가액(반올림 방식이 가게마다 달라 셋 다)."""
    out = set()
    for vat in (total // 11, -(-total // 11), round(total / 11)):
        out.update((vat, total - vat))
    return out


# 이름이 흐려 그 줄만으로는 안 되살린 품목을 되살리는 비용. 합계가 정확히 맞아야만 쓰이도록 할인
# 추가(1.0)·금액 교정(1.0~1.2)보다 비싸게, 품목 삭제(1.5)보다는 싸게 둔다.
_ADD_ITEM_COST = 1.3
# 품목이 아닌 줄의 어휘: 표 머리글, 배달비, 결제·할인 줄
_NON_ITEM_WORD = re.compile(r"상품명|품명|메뉴|수량|단가|금액|배달|배송|결제|할인|쿠폰|적립|영수|합계|소계|총액|거래|매출|사업자|전화")


def _looks_like_non_item(name) -> bool:
    if label_level(name):
        return True
    return bool(_NON_ITEM_WORD.search(_hangul(name)))


def _candidate_edits(items: list[dict], ev: Evidence, totals: set[int], used_discounts: Counter,
                     current_discount: int = 0, restorable: list[dict] | None = None) -> list[Edit]:
    edits: list[Edit] = []
    total_like = set(totals)
    for total in totals:
        total_like |= _tax_split(total)
    not_an_item = (ev.summary | total_like) - {0}
    for ref, cand in enumerate(restorable or []):
        price, disc = _int(cand.get("price")), _int(cand.get("discount"))
        # 되살리면 합계는 (금액 − 품목 할인) 만큼 늘지만, 그 할인이 이미 영수증 단위 할인으로 센 것이면
        # 전체 할인이 그만큼 줄어 도로 늘어난다. 둘이 상쇄되는 경우만 후보로 둔다.
        if price <= 0 or disc > current_discount:
            continue
        # 합계·결제·헤더·배달비 줄이나 요약부 금액(합계·부가세)을 품목으로 되살리면, 합계가 맞아 보여도 틀린 결과다.
        if price in not_an_item or _looks_like_non_item(cand.get("name")):
            continue
        edits.append(Edit("add_item", _ADD_ITEM_COST, amount=price, ref=ref))
    suspicious_amounts = (ev.summary | total_like) - {0}
    for i, item in enumerate(items):
        price = _int(item.get("price"))
        subs = item.get("sub_items") or []
        sub_sum = sum(_int(s.get("price")) for s in subs)
        level = label_level(item.get("name"))
        if level == 1:
            # 라벨일 수 있는 이름(`거자이체`). 상품명(`카드케이스`)일 수도 있어, 그 이름이 요약부에 찍혔거나
            # 금액이 합계·세금 금액일 때만 싸게 지운다. 아니면 찍힌 할인보다 비싸게 둔다.
            key = _hangul(item.get("name"))
            echoed = price in total_like or any(key and key in t for t in ev.summary_text)
            edits.append(Edit("drop_item", 0.3 if echoed else 1.25, item=i))
        elif item.get("_src") == "ocr":
            edits.append(Edit("drop_item", 1.0, item=i))
        elif len(items) > 1 and (price in suspicious_amounts or 0 < price < 100):
            edits.append(Edit("drop_item", 1.5, item=i))
        other_prices = {_int(o.get("price")) for k, o in enumerate(items) if k != i}
        for j, sub in enumerate(subs):
            sp = _int(sub.get("price"))
            if sp <= 0:
                continue
            if label_level(sub.get("name")) == 1:
                edits.append(Edit("drop_sub", 0.3 if sp in total_like else 1.25, item=i, sub=j))
            elif sp in suspicious_amounts:
                edits.append(Edit("drop_sub", 1.0, item=i, sub=j))
            elif sp in other_prices:
                # 다른 품목의 금액을 옵션으로 또 적은 것(VLM 이 옆 줄 금액을 끌어옴)
                edits.append(Edit("drop_sub", 1.3, item=i, sub=j))
        if sub_sum > 0 and price > sub_sum:
            edits.append(Edit("opts_included", 1.7, item=i))
        layout_price = _int(item.get("_layout_price"))
        if layout_price > 0 and layout_price != price:
            # 좌표가 같은 품목 줄에서 다른 금액을 읽었다(VLM 이 단가나 옆 줄 금액을 적음)
            edits.append(Edit("layout_price", 1.0, item=i, amount=layout_price - price))
        quantity = _int(item.get("quantity"))
        if 1 < quantity <= 50 and price > 0 and price * quantity in ev.amounts:
            # 단가를 금액으로 적은 경우(`1,000 × 4 = 4,000` 에서 1,000). 곱한 금액이 영수증에 찍혀 있어야 한다.
            edits.append(Edit("unit_price", 1.2, item=i, amount=price * (quantity - 1)))
        if sub_sum > price > 0:
            edits.append(Edit("price_is_net", 1.5, item=i))
        if _int(item.get("discount")) > 0:
            # 요약줄(`할인금액 -400`)의 라벨이 깨져 품목 할인으로 한 번 더 붙은 경우.
            # 같은 할인이 두 번이면 뒤에 붙은 쪽이 요약줄이라 뒤쪽을 조금 싸게 둔다.
            edits.append(Edit("drop_item_discount", 1.2 - 0.001 * i, item=i))

    # 할인 후보는 금액마다 하나(가장 싼 근거)만 둔다. 같은 할인을 두 번 더하지 않는다.
    discount_cost: dict[int, float] = {}

    def offer(amount: int, cost: float) -> None:
        if amount > 0 and cost < discount_cost.get(amount, 99):
            discount_cost[amount] = cost

    remaining = ev.discounts - used_discounts
    for amount in remaining:
        offer(amount, 1.0)
    # 할인 요약줄(`할인 -23,980`)이 총 할인액이면, 이미 센 품목 할인과의 차액만 영수증 할인으로 더한다.
    counted = sum(_int(it.get("discount")) for it in items) + current_discount
    if counted > 0:
        for amount in set(ev.discounts):
            offer(amount - counted, 1.1)
    # 라벨을 못 읽은 할인 줄: 라벨 없는 양수 금액 중 품목·옵션 금액도, 합계·세금도 아닌 것.
    line_amounts = {_int(it.get("price")) for it in items} | {
        _int(s.get("price")) for it in items for s in it.get("sub_items") or []
    }
    for amount in ev.unlabeled_positive - total_like - line_amounts:
        offer(amount, 1.6)
    edits.extend(Edit("add_discount", cost, amount=a) for a, cost in sorted(discount_cost.items()))
    return edits


def _compatible(combo: tuple[Edit, ...]) -> bool:
    touched_items: set[int] = set()
    sub_edits: dict[int, set[int]] = {}
    for e in combo:
        if e.kind in ("add_discount", "add_item"):
            continue
        if e.kind in _ITEM_LEVEL:
            if e.item in touched_items or e.item in sub_edits:
                return False
            touched_items.add(e.item)
        else:  # drop_sub
            if e.item in touched_items:
                return False
            subs = sub_edits.setdefault(e.item, set())
            if e.sub in subs:
                return False
            subs.add(e.sub)
    return True


@dataclass
class Solution:
    paid: int
    edits: tuple[Edit, ...]
    fee_added: int
    cost: float
    ocr_backed: bool  # 결제액이 OCR 에 찍힌 값인가(아니면 VLM 만 말한 값)


@dataclass(frozen=True)
class Candidate:
    value: int
    tier: int  # 0: 결제 라벨, 1: 합계 라벨·반복 금액·합계−할인, 2: VLM 만 말한 값


_MAX_EDITS = 2
_MAX_CANDIDATES = 120
_MAX_TOTALS = 8
_PREFERENCE = ["add_discount", "drop_sub", "drop_item", "layout_price", "unit_price",
               "drop_item_discount", "opts_included", "price_is_net", "add_item"]
_FEE_GAP_COST = 1.5


def solve(items: list[dict], discount: int, fee: int, candidates: list[Candidate], ev: Evidence,
          used_discounts: Counter, restorable: list[dict] | None = None) -> Solution | None:
    """믿을 만한 후보 등급부터, 그 등급의 결제액을 맞추는 최소 비용 수정을 찾는다.

    - VLM 만 말한 결제액(등급 2)은 OCR 근거가 하나도 없을 때만 본다.
    - 합계 라벨(등급 1)이 수정 없이 맞는데 결제 라벨(등급 0)이 다르면, 결제 라벨은 강한 할인 근거만으로
      설명될 때만 채택한다. 분할 결제(카드 10,000 + 상품권 5,000)에서 품목을 지워 카드 금액에 맞추면 안 된다.
    """
    base = computed_total(items, discount, fee)
    by_tier: dict[int, set[int]] = {}
    for c in candidates:
        by_tier.setdefault(c.tier, set()).add(c.value)
    # VLM 합계는 결제액 후보에서 빠져도 "합계처럼 생긴 금액" 으로는 남긴다(그 금액의 줄은 의심한다)
    totals = {c.value for c in candidates}
    if 0 in by_tier or 1 in by_tier:
        by_tier.pop(2, None)

    # 품목이 아주 많은 영수증에서도 조합 수가 터지지 않게 싼(근거가 강한) 수정만 남긴다.
    edits = sorted(_candidate_edits(items, ev, totals, used_discounts, discount, restorable),
                   key=lambda e: e.cost)[:_MAX_CANDIDATES]
    deltas = [e.delta(items) for e in edits]
    combos: list[tuple[int, float, tuple[Edit, ...]]] = [(base, 0.0, ())]
    for size in range(1, _MAX_EDITS + 1):
        for idx in itertools.combinations(range(len(edits)), size):
            combo = tuple(edits[k] for k in idx)
            if _compatible(combo):
                combos.append((base + sum(deltas[k] for k in idx), sum(e.cost for e in combo), combo))

    label_balances = bool(by_tier.get(1)) and base in by_tier[1]

    for tier in sorted(by_tier):
        best: list[Solution] = []
        for paid in sorted(by_tier[tier])[:_MAX_TOTALS]:
            for value, cost, combo in combos:
                if tier == 0 and label_balances and paid not in by_tier[1]:
                    if not combo or any(e.kind != "add_discount" or e.cost > _STRONG_DISCOUNT_COST for e in combo):
                        continue
                if tier == 2 and combo:
                    continue  # VLM 만 말한 결제액에 맞추려고 품목을 고치지 않는다
                gap = paid - value
                fee_added = 0
                if gap != 0:
                    if tier == 2:
                        continue
                    # 마지막 수단: 배달 영수증에서 번져 못 읽은 배달팁을 차액으로 채운다.
                    if tier == 0 and label_balances:
                        continue
                    if not (ev.fee_unread and fee == 0 and 100 <= gap <= 10000 and gap % 100 == 0):
                        continue
                    fee_added, cost = gap, cost + _FEE_GAP_COST
                if any(e.kind == "add_item" for e in combo) and not _restore_consistent(
                    items, discount, fee, combo, restorable or [], paid - fee_added
                ):
                    continue  # 되살린 품목의 할인이 전체 할인과 겹쳐 합계가 어긋난다
                best.append(Solution(paid, combo, fee_added, cost, ocr_backed=tier < 2))
        if not best:
            continue
        low = min(s.cost for s in best)
        top = [s for s in best if abs(s.cost - low) < 1e-9]
        if len({s.paid for s in top}) > 1:
            return None  # 같은 비용으로 서로 다른 결제액이 맞는다 — 모호하면 손대지 않는다
        # 서로 다른 줄을 되살려도 같은 비용으로 맞는다면 어느 쪽이 맞는지 모른다 — 손대지 않는다.
        restored_sets = {frozenset(e.ref for e in s.edits if e.kind == "add_item") for s in top}
        if len(restored_sets - {frozenset()}) > 1:
            return None
        # 결제액은 같고 고치는 방법만 다르면, 근거가 강한 수정부터 고른다.
        return min(top, key=lambda s: sorted(_PREFERENCE.index(e.kind) for e in s.edits))
    return None


def _restored_item(cand: dict) -> dict:
    return {
        "name": cand.get("name"),
        "quantity": cand.get("quantity"),
        "price": _int(cand.get("price")),
        "sub_items": [],
        "discount": _int(cand.get("discount")),
        "_src": "ocr",
        "_pos": cand.get("_pos"),
    }


def _restore_consistent(items: list[dict], discount: int, fee: int, combo: tuple[Edit, ...],
                        restorable: list[dict], paid: int) -> bool:
    """되살린 품목을 실제로 넣고 다른 수정도 적용했을 때 합계가 정말 결제액이 되는가."""
    trial = [dict(i, sub_items=[dict(s) for s in i.get("sub_items") or []]) for i in items]
    top = discount
    for e in combo:
        if e.kind == "add_item":
            cand = restorable[e.ref]
            trial.append(_restored_item(cand))
            top = max(0, top - _int(cand.get("discount")))
    # add_item 외의 수정은 합계를 더할 뿐이라(할인 추가·금액 교정…) 여기서는 그 변화량만 더한다
    others = sum(e.delta(items) for e in combo if e.kind != "add_item")
    return computed_total(trial, top, fee) + others == paid


def apply(items: list[dict], parsed: dict, sol: Solution, corrections: list) -> None:
    drops_items = {e.item for e in sol.edits if e.kind == "drop_item"}
    drops_subs = {(e.item, e.sub) for e in sol.edits if e.kind == "drop_sub"}
    restored: list[dict] = []
    for e in sol.edits:
        item = items[e.item] if e.item >= 0 else None
        if e.kind == "drop_item":
            corrections.append(_corr("items", f"{item.get('name')} {item.get('price')}", "(삭제: 합계 검산)", "reconciled"))
        elif e.kind == "drop_sub":
            sub = item["sub_items"][e.sub]
            corrections.append(_corr("item.sub_items", f"{item.get('name')} > {sub.get('name')} {sub.get('price')}",
                                     "(삭제: 합계 검산)", "reconciled"))
        elif e.kind == "opts_included":
            sub_sum = sum(_int(s.get("price")) for s in item.get("sub_items") or [])
            before = _int(item.get("price"))
            item["price"] = before - sub_sum
            corrections.append(_corr("item.price", str(before), f"{item['price']} (옵션 포함 금액이었음)", "reconciled"))
        elif e.kind == "price_is_net":
            before = _int(item.get("price"))
            item["price"] = 0
            corrections.append(_corr("item.price", str(before), "0 (옵션이 금액을 담고 있음)", "reconciled"))
        elif e.kind in ("unit_price", "layout_price"):
            before = _int(item.get("price"))
            item["price"] = before + e.amount
            why = "단가 × 수량" if e.kind == "unit_price" else "좌표가 읽은 금액"
            corrections.append(_corr("item.price", str(before), f"{item['price']} ({why})", "reconciled"))
        elif e.kind == "drop_item_discount":
            before = _int(item.get("discount"))
            item["discount"] = 0
            corrections.append(_corr("item.discount", str(before), "0 (요약줄 중복)", "reconciled"))
        elif e.kind == "add_discount":
            parsed["discount"] = _int(parsed.get("discount")) + e.amount
            corrections.append(_corr("discount", "", f"+{e.amount:,} (OCR 할인 줄)", "reconciled"))
        elif e.kind == "add_item":
            restored.append(parsed["_restorable"][e.ref])
    for i, item in enumerate(items):
        item["sub_items"] = [s for j, s in enumerate(item.get("sub_items") or []) if (i, j) not in drops_subs]
    items[:] = [it for i, it in enumerate(items) if i not in drops_items]
    # 되살린 품목은 영수증에 찍힌 순서대로 끼워 넣는다(순서를 모르면 맨 뒤)
    for cand in sorted(restored, key=lambda c: c.get("_pos") if c.get("_pos") is not None else 10**6):
        new_item = _restored_item(cand)
        pos = cand.get("_pos")
        at = len(items)
        if pos is not None:
            at = 0
            for k, it in enumerate(items):
                if it.get("_pos") is not None and it["_pos"] < pos:
                    at = k + 1
        items.insert(at, new_item)
        parsed["discount"] = max(0, _int(parsed.get("discount")) - new_item["discount"])
        corrections.append(_corr("items", "(VLM 누락, 합계 검산)", str(new_item.get("name")), "ocr_recovered"))
    if sol.fee_added:
        parsed["delivery_fee"] = _int(parsed.get("delivery_fee")) + sol.fee_added
        corrections.append(_corr("delivery_fee", "0", f"{sol.fee_added:,} (결제액 − 품목 차액)", "reconciled"))


# ---------------------------------------------------------------- 구조 정리

def fold_negative_items(parsed: dict, items: list[dict], corrections: list) -> None:
    """금액이 음수인 최상위 품목(`(카드쿠폰)청정원 올리유 -12,000`)을 할인으로 바꾼다.

    이미 같은 금액을 할인으로 셌으면(합계부 요약줄 등) 품목만 지운다.
    포인트·상품권 사용은 할인이 아니라 결제 수단이라 할인으로 만들지 않고 지우기만 한다.
    """
    counted = Counter(_int(i.get("discount")) for i in items if _int(i.get("discount")))
    if _int(parsed.get("discount")):
        counted[_int(parsed["discount"])] += 1
    kept = []
    for item in items:
        price = _int(item.get("price"))
        if price < 0 and not item.get("sub_items"):
            amount = -price
            if _TENDER_WORD.search(_hangul(item.get("name"))):
                after = "(삭제: 결제 수단)"
            elif counted[amount]:
                counted[amount] -= 1
                after = "(삭제: 이미 할인으로 셈)"
            else:
                parsed["discount"] = _int(parsed.get("discount")) + amount
                after = f"discount +{amount:,}"
            corrections.append(_corr("items", f"{item.get('name')} {price}", after, "negative_item"))
            continue
        kept.append(item)
    items[:] = kept


# OCR 이 읽은 옵션 줄 표시. `ㄴ` 은 `L` 로 자주 읽힌다(`L타피오카필 1개추가`).
# `▶` 는 `※` 로 읽히기도 한다(`※우라이드` = `▶후라이드`).
_OCR_OPTION_ROW = re.compile(r"^\s*(?:[ㄴ└#+>›»▶►▸※]|L(?=[가-힣(\[]))\s*(.+)$")
_NAME_MARKER = re.compile(r"^[#ㄴ└+>›»▶►▸※\s]+")


def _hangul(text) -> str:
    return re.sub(r"[^가-힣]", "", text or "")


def _similar(a: str, b: str) -> bool:
    """한글 키 두 개가 같은 줄을 읽은 것인가(포함 관계 또는 자모 몇 개 차이)."""
    from corrector import _jamo_distance, _jamo_seq

    if not a or not b:
        return False
    if a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a)):
        return True
    return min(len(a), len(b)) >= 4 and _jamo_distance(a, b) <= max(1, len(_jamo_seq(a)) // 8)


def demote_options(items: list[dict], layout, ev: Evidence, corrections: list) -> None:
    """옵션인데 품목으로 올라온 줄을 바로 위 품목의 옵션으로 내린다. 금액 합은 그대로다.

    1. 영수증에 옵션 표시(`ㄴ`, `#` 등)와 함께 찍힌 줄: OCR 행 이름이 VLM 품목 이름과 같고,
       그 OCR 행 바로 위 품목 줄이 VLM 의 직전 품목과도 같을 때만 내린다. 따로 산 `콜라` 가
       세트의 `ㄴ콜라` 옵션에 끌려 내려가지 않게 한다.
    2. 표시된 옵션 바로 다음의 0원 줄(`▶후라이드 0` 다음 `간장 0`).
    3. 배달 영수증에서 금액 0 인 줄이 금액 있는 품목 바로 아래 오면 옵션이다(`보통맛 0`).
       편의점 증정품(`스낵면대컵 0`)처럼 POS 영수증의 0원 품목은 품목이라 배달 영수증만 본다.
    """
    marked: list[tuple[str, str]] = []  # (옵션 키, 그 위 품목 키)
    plain_priced: list[str] = []  # 표시 없이 금액과 함께 찍힌 줄(따로 산 품목)
    last_parent = ""
    for row in layout.rows:
        m = _OCR_OPTION_ROW.match(row.text)
        if m and len(_hangul(m.group(1))) >= 2:
            marked.append((_hangul(m.group(1)), last_parent))
        elif len(_hangul(row.text)) >= 2:
            last_parent = _hangul(row.text)
            # 금액을 못 읽었어도(`2,0O0`) 숫자가 찍혀 있으면 따로 산 품목 줄로 본다
            if any(re.search(r"\d", t.text) and not _hangul(t.text) for t in row.tokens):
                plain_priced.append(last_parent)

    def is_marked_under(name: str, parent_name: str) -> bool:
        key = _hangul(_NAME_MARKER.sub("", name or ""))
        if len(key) < 2:
            return False
        if any(key == k or _similar(key, k) for k in plain_priced):
            return False  # 같은 이름이 표시 없이 금액과 함께 따로 찍혔다 — 따로 산 품목이다
        parent_key = _hangul(parent_name)
        for option_key, option_parent in marked:
            if not _similar(key, option_key):
                continue
            # 표시된 옵션이 같은 메뉴 아래였거나, 같은 이름 옵션이 다른 메뉴 아래 다시 찍혀
            # OCR 이 못 읽은 경우(`타피오카펄 1개추가` 가 두 메뉴 아래 각각)
            if not option_parent or _similar(parent_key, option_parent) or _similar_menu(parent_key, option_parent):
                return True
        return False

    out: list[dict] = []
    previous_demoted = False
    for item in items:
        name = item.get("name") or ""
        price = _int(item.get("price"))
        parent = out[-1] if out else None
        reason = None
        if parent is not None and not item.get("sub_items") and not _int(item.get("discount")):
            if name.lstrip().startswith("#") or is_marked_under(name, parent.get("name") or ""):
                reason = "OCR 옵션 표시"
            elif price == 0 and previous_demoted:
                reason = "옵션 줄 다음의 0원 줄"  # `▶후라이드 0` 다음 `간장 0`
            elif ev.delivery and price == 0 and line_total(parent) + _int(parent.get("discount")) > 0:
                reason = "배달 영수증의 0원 줄"
        previous_demoted = reason is not None
        if reason:
            clean = _NAME_MARKER.sub("", name)
            parent.setdefault("sub_items", []).append({"name": clean, "price": price})
            corrections.append(_corr("items", name, f"{parent.get('name')} > {clean} ({reason})", "option_demoted"))
            continue
        out.append(item)
    items[:] = out


def _similar_menu(parent_key: str, option_parent: str) -> bool:
    """같은 메뉴가 두 번 주문된 경우(`달고나카페라떼` 1잔, 2잔) 메뉴 이름 일부만 겹쳐도 같다고 본다."""
    if len(parent_key) < 3 or len(option_parent) < 3:
        return False
    return parent_key[:3] in option_parent or option_parent[:3] in parent_key or parent_key[-3:] in option_parent


def drop_total_echoes(items: list[dict], totals: set[int], corrections: list,
                      confirmed: set[int] | None = None) -> None:
    """좌표로 되살린 품목 중 금액이 합계와 똑같은 것은 라벨이 깨진 합계 줄이다.

    `즈ㅁ그애 15,200`(주문금액), `수군금액 31,900` 처럼 이름이 깨져 라벨 규칙을 빠져나간다.
    다른 품목이 있는데 한 품목이 합계와 같을 수는 없으므로(다른 품목 금액만큼 넘친다) 지운다.
    부가세·공급가액과 같은 금액은 진짜 품목일 수 있어(`마들렌 1,000`) 검산에 맡긴다.
    VLM 이 직접 적은 품목은 건드리지 않는다.
    """
    if len(items) < 2 or not totals:
        return
    # "합계보다 큰 품목" 은 OCR 이 확인한 합계로만 판단한다. VLM 합계가 품목을 놓쳐 작게 나온 경우
    # 되살린 진짜 품목(`탕수육 15,000`)이 VLM 합계보다 클 수 있다.
    largest = max(confirmed) if confirmed else None
    kept = []
    for item in items:
        others = sum(line_total(o) for o in items if o is not item)
        price = _int(item.get("price"))
        if (
            item.get("_src") == "ocr"
            and (price in totals or (largest is not None and price > largest))  # 합계와 같거나, 혼자서 합계를 넘는 금액
            and not item.get("sub_items")
            and others > 0
        ):
            corrections.append(_corr("items", f"{item.get('name')} {item.get('price')}", "(삭제: 합계 줄)", "label_dropped"))
            continue
        kept.append(item)
    items[:] = kept


# ---------------------------------------------------------------- 진입점

def _candidates(parsed: dict, layout, ev: Evidence) -> list[Candidate]:
    out: list[Candidate] = []
    label_total = layout.total_amount if layout.total_amount and layout.total_amount > 0 else None

    paid = [p for p in ev.paid if p > 0]
    # 분할 결제: 서로 다른 결제 금액(카드 두 장, 카드 + 상품권·포인트)의 합이 합계면 부분 금액은 결제액이
    # 아니다. 합계는 합계 라벨·VLM 합계·반복 금액 중 하나와 같으면 된다(라벨이 깨진 영수증이 있다).
    targets = {t for t in {label_total or 0, _int(parsed.get("total_amount"))} | ev.repeated if t > 0}
    if paid and targets:
        parts = sorted(set(paid) | set(t for t in ev.tenders if t > 0))
        for target in sorted(targets, reverse=True):
            small = [p for p in parts if p < target]
            if any(
                sum(c) == target
                for size in range(2, min(4, len(small)) + 1)
                for c in itertools.combinations(small, size)
            ):
                paid = [p for p in paid if p >= target]
                out.append(Candidate(target, 1))
                break
    out.extend(Candidate(p, 0) for p in paid)
    if label_total:
        out.append(Candidate(label_total, 1))
    out.extend(Candidate(v, 1) for v in ev.repeated)
    # 요약부에 찍힌 금액 중 `합계 − 할인` 과 같은 값은 할인 뒤 결제액이다(홈플러스: 100,640 − 23,980 = 76,660).
    for base_total in {label_total or 0, _int(parsed.get("total_amount"))} - {0}:
        for d in ev.discounts:
            value = base_total - d
            # 할인이 합계의 절반을 넘거나, 그 값이 부가세·공급가액이면 우연히 맞은 숫자다
            if value in ev.summary and value * 2 >= base_total and value not in _tax_split(base_total):
                out.append(Candidate(value, 1))
    vlm_total = _int(parsed.get("total_amount"))
    if vlm_total > 0:
        out.append(Candidate(vlm_total, 2))
    return out


def reconcile(parsed: dict, layout, ocr_lines: list[dict], corrections: list) -> bool:
    """검산으로 맞출 수 있으면 고치고 True. 결제액은 parsed["total_amount"] 에 쓴다.

    반환값이 True 라도 결제액이 OCR 근거 없이 VLM 만 말한 값이면 parsed["_paid_from_ocr"] 가 False 다.
    """
    items = parsed.get("items")
    if not isinstance(items, list):
        return False
    vlm_total = parsed.get("total_amount")
    if isinstance(vlm_total, (int, float)) and not isinstance(vlm_total, bool) and vlm_total < 0:
        return False  # 환불 영수증은 검산 대상이 아니다
    drop_label_rows(items, corrections)
    fold_negative_items(parsed, items, corrections)
    ev = collect_evidence(layout, ocr_lines)
    demote_options(items, layout, ev, corrections)

    candidates = _candidates(parsed, layout, ev)
    if not candidates or not items:
        return False

    used = Counter()
    for item in items:
        if _int(item.get("discount")):
            used[_int(item["discount"])] += 1
    if _int(parsed.get("discount")):
        used[_int(parsed["discount"])] += 1

    drop_total_echoes(items, {c.value for c in candidates}, corrections,
                      confirmed={c.value for c in candidates if c.tier < 2})
    sol = solve(items, _int(parsed.get("discount")), _int(parsed.get("delivery_fee")), candidates, ev, used,
                parsed.get("_restorable"))
    if sol is None:
        return False
    apply(items, parsed, sol, corrections)
    before = parsed.get("total_amount")
    parsed["total_amount"] = sol.paid
    # 합계 라벨을 못 읽었어도 VLM 합계가 영수증에 금액으로 따로 찍혀 있으면 확인된 것으로 본다.
    # 품목 줄 금액은 세지 않는다. 품목 금액으로 찍힌 횟수보다 더 많이 찍혀야 합계 줄이 따로 있는 것이다
    # (품목 하나를 놓쳐 VLM 합계 = 남은 품목 금액이 된 경우를 거른다).
    same_priced = sum(1 for it in items if _int(it.get("price")) == sol.paid)
    parsed["_paid_from_ocr"] = sol.ocr_backed or ev.amount_counts[sol.paid] > same_priced
    if before != sol.paid:
        # 합계 라벨 값으로 바꾼 경우는 예전과 같은 이유 코드를 쓴다(응답 소비자 호환)
        reason = "ocr_layout" if sol.paid == layout.total_amount else "reconciled"
        corrections.append(_corr("total_amount", str(before), str(sol.paid), reason))
    return True


def _int(value) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == value and abs(value) != float("inf"):
        return int(value)
    return 0


def _corr(field_name: str, before: str, after: str, reason: str) -> dict:
    return {"field": field_name, "before": before, "after": after, "reason": reason}
