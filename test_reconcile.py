"""reconcile.py(합계 검산) 회귀 테스트.

각 테스트의 OCR 줄은 실측 63장에서 실제로 나온 모양을 줄여 옮겼다.
실행: .venv/bin/python -m pytest test_reconcile.py -v
"""
import reconcile as rec
from test_layout import _HEADER, _line, _merge_with


def _items(merged):
    return [(i["name"], i["price"], i["discount"], [s["price"] for s in i["sub_items"]]) for i in merged["items"]]


def _reasons(corrections):
    return [c["reason"] for c in corrections]


# ---------------------------------------------------------------- 라벨 이름

def test_label_names_are_caught_even_when_garbled():
    for name in ["합할 계", "합\"계:", "가세(VAT):", "트가세(VAT):", "부 세", "과세", "과세물풀:",
                 "총상품금액", "주무금액", "카드/간편결제", "키 류: 삼성카드", "적립포인트",
                 "받은포인트:", "신카드지불:", "인 0,현재잔액:", "부개월:일시불", "주문번호:466670217"]:
        assert rec.is_label_name(name), name
    # 라벨일 수 있지만 상품명에도 나오는 어휘는 검산으로만 지운다
    for name in ["거자이체", "종합계란 15구", "포인트 스티커", "카드지갑"]:
        assert rec.label_level(name) == 1, name


def test_product_names_are_not_labels():
    """리뷰 지적: `종합계란`(합계), `PRIVATE 생수`(VAT), 콜론 붙은 옵션·상품명을 지웠다."""
    for name in ["종합계란 15구", "PRIVATE 생수 2L", "샷추가: 1샷", "[행사]콜라:500ml", "포인트 스티커",
                 "카드지갑", "시계", "합성세제", "미니꿀약과", "농심)매운새우깡 90g", "(BEST)메가치킨마요"]:
        assert not rec.is_label_name(name), name


# ---------------------------------------------------------------- 새는 줄 제거

def test_vat_and_points_rows_dropped_and_items_verified():
    """홈푸드마트 실측: `가세(VAT): 363`, `신카드지불: 4,000`, `적립포인트 26,374` 가 품목으로 들어왔다."""
    lines = _HEADER + [
        _line("스크류바", 100, 150), _line("600", 600, 150),
        _line("파이리 핫소스팡", 100, 200), _line("3,400", 600, 200),
        _line("부가세(VAT)", 100, 300), _line("363", 600, 300),
        _line("합계", 100, 350), _line("4,000", 600, 350),
        _line("신용카드지불", 100, 400), _line("4,000", 600, 400),
    ]
    parsed = {
        "items": [
            {"name": "스크류바", "quantity": 1, "price": 600},
            {"name": "파이리 핫소스팡", "quantity": 1, "price": 3400},
            {"name": "트가세(VAT):", "price": 363},
            {"name": "신카드지불:", "price": 4000},
            {"name": "적립포인트", "price": 26374},
        ],
        "total_amount": 4000,
    }
    merged, corrections = _merge_with(parsed, lines)
    assert [i["name"] for i in merged["items"]] == ["스크류바", "파이리 핫소스팡"]
    assert merged["total_amount"] == 4000
    assert merged["items_verified"] is True
    assert _reasons(corrections).count("label_dropped") == 3


def test_garbled_total_line_recovered_as_item_is_dropped_by_checksum():
    """하이마트 실측(hires): `합할 계 2,400` 이 좌표 복구로 품목이 되어 합계가 두 배가 됐다."""
    lines = _HEADER + [
        _line("몬스터 파이프라인펀치", 100, 150), _line("1,900", 600, 150),
        _line("테니스볼껌", 100, 200), _line("500", 600, 200),
        _line("합할 계인", 100, 260), _line("2,400", 600, 260),  # 라벨이 깨져 품목 영역이 안 끝남
        _line("합계", 100, 320), _line("2,400", 600, 320),
    ]
    parsed = {
        "items": [
            {"name": "몬스터 파이프라인펀치", "quantity": 1, "price": 1900},
            {"name": "테니스볼껌", "quantity": 1, "price": 500},
        ],
        "total_amount": 2400,
    }
    merged, _ = _merge_with(parsed, lines)
    assert [i["name"] for i in merged["items"]] == ["몬스터 파이프라인펀치", "테니스볼껌"]
    assert merged["items_verified"] is True


def test_supply_value_sub_item_dropped():
    """마트 실측(crop): VLM 이 `세 10,810`(공급가액)을 옵션으로 붙였다. 11,890 의 공급가액이다."""
    lines = _HEADER + [
        _line("샴푸", 100, 150), _line("9,900", 600, 150),
        _line("쫀디기", 100, 200), _line("1,000", 600, 200),
        _line("합계", 100, 300), _line("10,900", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "샴푸", "quantity": 1, "price": 9900},
            {"name": "쫀디기", "quantity": 1, "price": 1000, "sub_items": [{"name": "세", "price": 9909}]},
        ],
        "total_amount": 10900,
    }
    merged, corrections = _merge_with(parsed, lines)
    assert _items(merged) == [("샴푸", 9900, 0, []), ("쫀디기", 1000, 0, [])]
    assert "reconciled" in _reasons(corrections)


# ---------------------------------------------------------------- 실제 결제액

def test_baemin_payment_detail_discounts_make_total_the_paid_amount():
    """배민 실측: 합계금액 23,300, `결제 금액 상세` 21,300 아래 채널할인 1,000·배달앱할인 1,000.

    정산할 돈은 손님이 낸 21,300 이다. 상세 할인 둘은 합쳐서 한 번에 뺀다.
    """
    lines = _HEADER + [
        _line("불닭마요연어덮밥", 100, 150), _line("10,400", 600, 150),
        _line("+ 연어 추가", 100, 190), _line("2,000", 600, 190),
        _line("들기름육회막국수", 100, 230), _line("10,900", 600, 230),
        _line("합계금액", 100, 300), _line("23,300", 600, 300),
        _line("-결제 금액 상세-", 200, 350),
        _line("21,300", 600, 400),
        _line("1,000", 600, 450),
        _line("1,000", 600, 500),
    ]
    parsed = {
        "items": [
            {"name": "불닭마요연어덮밥", "quantity": 1, "price": 10400,
             "sub_items": [{"name": "연어 추가", "price": 2000}]},
            {"name": "들기름육회막국수", "quantity": 1, "price": 10900},
        ],
        "total_amount": 23300,
    }
    merged, _ = _merge_with(parsed, lines)
    assert merged["total_amount"] == 21300
    assert merged["discount"] == 2000
    assert _items(merged)[0] == ("불닭마요연어덮밥", 10400, 0, [2000])  # 옵션을 잘못 지우지 않는다
    assert merged["items_verified"] is True


def test_coupon_total_discount_makes_paid_from_subtotal_minus_discount():
    """홈플러스 실측: 100,640 − 23,980(쿠폰 합) = 76,660 결제. VLM 은 쿠폰 하나를 놓쳤다."""
    lines = _HEADER + [
        _line("오징어양파", 100, 150), _line("19,980", 600, 150),
        _line("-9,990", 600, 180),
        _line("오징어120G", 100, 220), _line("19,980", 600, 220),
        _line("에누리(2604260014838)", 100, 250), _line("-9,990", 600, 250),
        _line("한우 앞다리", 100, 290), _line("60,680", 600, 290),
        _line("과세물품가액", 100, 340), _line("44,545", 600, 340),
        _line("100,640", 600, 380),
        _line("(이", 100, 420), _line("-23,980", 600, 420),
        _line("76,660", 600, 460),
    ]
    parsed = {
        "items": [
            {"name": "오징어양파", "quantity": 1, "price": 19980},
            {"name": "오징어120G", "quantity": 1, "price": 19980},
            {"name": "한우 앞다리", "quantity": 1, "price": 60680, "sub_items": [{"name": "과세물풀", "price": 44545}]},
        ],
        "total_amount": 100640,
    }
    merged, _ = _merge_with(parsed, lines)
    assert merged["total_amount"] == 76660
    assert sum(i["discount"] for i in merged["items"]) + merged["discount"] == 23980
    assert merged["items_verified"] is True


# ---------------------------------------------------------------- 배달비

def test_smudged_delivery_tip_filled_from_paid_minus_order_amount():
    """배민 실측: 배달팁 칸이 번져 `알종배달 --` 만 남았다. 31,300 − 30,800 = 500."""
    lines = _HEADER + [
        _line("딸기버블티", 100, 150), _line("5,000", 600, 150),
        _line("아메리카노", 100, 200), _line("25,800", 600, 200),
        _line("주문금액", 100, 260), _line("30,800", 600, 260),
        _line("알종배달", 100, 310), _line("--", 600, 310),
        _line("총 결제금액", 100, 360), _line("31,300", 600, 360),
    ]
    parsed = {
        "items": [
            {"name": "딸기버블티", "quantity": 1, "price": 5000},
            {"name": "아메리카노", "quantity": 1, "price": 25800},
        ],
        "total_amount": 31300,
    }
    merged, _ = _merge_with(parsed, lines)
    assert merged["delivery_fee"] == 500
    assert merged["total_amount"] == 31300
    assert merged["items_verified"] is True


def test_gap_is_not_filled_as_delivery_fee_when_fee_row_was_read():
    """한집배달 실측: 배달팁 1,000·-1,000 을 읽었는데 VLM 이 품목 하나를 놓쳤다.

    차액(4,200)은 배달비가 아니다. 모르는 돈을 배달비로 만들지 않고 검증 실패로 둔다.
    """
    lines = _HEADER + [
        _line("아이스티", 100, 150), _line("3,000", 600, 150),
        _line("상하이 버터떡", 100, 200), _line("8,000", 600, 200),
        _line("주문금액", 100, 260), _line("15,200", 600, 260),
        _line("맨드본 배달팁", 100, 300), _line("1,000", 600, 300),
        _line("-1,000", 600, 330),
        _line("총결제금액", 100, 380), _line("15,200", 600, 380),
    ]
    parsed = {
        "items": [
            {"name": "아이스티", "quantity": 1, "price": 3000},
            {"name": "상하이 버터떡", "quantity": 1, "price": 8000},
        ],
        "total_amount": 15200,
    }
    merged, _ = _merge_with(parsed, lines)
    assert merged["delivery_fee"] == 0
    assert merged["items_verified"] is False


# ---------------------------------------------------------------- 금액 해석

def test_options_already_included_in_line_amount():
    """카페 실측: 금액칸 11,800 이 옵션까지 담은 줄 금액이다(옵션 줄은 참고 표기)."""
    lines = _HEADER + [
        _line("폭탄 딸기라떼", 100, 150), _line("2", 400, 150), _line("11,800", 600, 150),
        _line("합계 금액", 100, 300), _line("11,800", 600, 300),
    ]
    parsed = {
        "items": [{"name": "폭탄 딸기라떼", "quantity": 2, "price": 11800,
                   "sub_items": [{"name": "아이스크림 추가", "price": 2400}, {"name": "펄 추가", "price": 2000}]}],
        "total_amount": 11800,
    }
    merged, _ = _merge_with(parsed, lines)
    item = merged["items"][0]
    assert item["price"] + sum(s["price"] for s in item["sub_items"]) == 11800
    assert merged["items_verified"] is True


def test_unit_price_written_as_line_amount():
    """마트 실측: `1,000 × 4 = 4,000` 에서 VLM 이 단가 1,000 을 적었다."""
    lines = _HEADER + [
        _line("고래밥", 100, 150), _line("1,980", 600, 150),
        _line("얼라이브 오렌지", 100, 200), _line("4", 400, 200), _line("4,000", 600, 200),
        _line("합계", 100, 300), _line("5,980", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "고래밥", "quantity": 1, "price": 1980},
            {"name": "얼라이브 오렌지", "quantity": 4, "price": 1000},
        ],
        "total_amount": 5980,
    }
    merged, _ = _merge_with(parsed, lines)
    assert [i["price"] for i in merged["items"]] == [1980, 4000]


def test_negative_item_becomes_discount_once():
    """하나로마트 실측: `(카드쿠폰)올리유 -12,000` 이 품목으로, 같은 쿠폰이 요약 할인으로도 왔다."""
    lines = _HEADER + [
        _line("올리유", 100, 150), _line("98,700", 600, 150),
        _line("쿠폰할인", 100, 250), _line("-12,000", 600, 250),
        _line("합계", 100, 300), _line("86,700", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "올리유", "quantity": 3, "price": 98700},
            {"name": "(카드쿠폰)올리유", "quantity": 3, "price": -12000},
        ],
        "total_amount": 86700,
    }
    merged, corrections = _merge_with(parsed, lines)
    assert [i["name"] for i in merged["items"]] == ["올리유"]
    assert sum(i["discount"] for i in merged["items"]) + merged["discount"] == 12000
    assert "negative_item" in _reasons(corrections)


def test_unlabelled_negative_row_under_item_is_its_discount():
    """홈푸드마트 실측: `$특매할인` 라벨을 OCR 이 떨어뜨려 `-1,510` 만 남았다. 요약줄은 다시 세지 않는다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("샴푸", 100, 150), _line("9,900", 600, 150),
        _line("미니꿀약과", 100, 200), _line("2,500", 600, 200),
        _line("-1,510", 600, 230),
        _line("쫀디기", 100, 270), _line("1,000", 600, 270),
        _line("과", 100, 320), _line("-1,510", 600, 320),  # `할인금액` 라벨이 깨진 요약줄
        _line("합계", 100, 380), _line("11,890", 600, 380),
    ]
    layout = analyze_receipt(lines)
    assert [(i.name, i.discount) for i in layout.items] == [("샴푸", 0), ("미니꿀약과", 1510), ("쫀디기", 0)]
    assert layout.summary_discount == 1510


def test_two_equal_item_discounts_are_not_mistaken_for_a_summary():
    """`-400` 이 두 품목에 연달아 붙으면 둘 다 품목 할인이다(두 번째가 요약줄이 아니다)."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("몬스터 시트라", 100, 150), _line("2,000", 600, 150),
        _line("1", 300, 180), _line("-400", 600, 180),
        _line("몬스터 레모네이드", 100, 220), _line("2,000", 600, 220),
        _line("1", 300, 250), _line("-400", 600, 250),
        _line("-800", 600, 300),
        _line("합계", 100, 350), _line("3,200", 600, 350),
    ]
    layout = analyze_receipt(lines)
    assert [i.discount for i in layout.items] == [400, 400]
    assert layout.summary_discount == 800


def test_free_delivery_waiver_row_is_not_an_item_discount():
    """배달팁 바로 아래 `-1,000` 은 무료배달 상계다. 앞 품목의 할인으로 잡으면 두 번 빠진다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("배달팁", 100, 200), _line("1,000", 600, 200),
        _line("-1,000", 600, 230),
        _line("합계", 100, 300), _line("9,000", 600, 300),
    ]
    layout = analyze_receipt(lines)
    assert [i.discount for i in layout.items if i.name] == [0]
    assert layout.delivery_fee == 0


# ---------------------------------------------------------------- 옵션 구조

def test_marked_option_rows_are_demoted_even_when_ocr_misreads_marker():
    """배민 실측: `ㄴ 타피오카펄 1개추가` 를 OCR 은 `L타피오카필 1개추가`, VLM 은 독립 품목으로 읽었다."""
    lines = _HEADER + [
        _line("달고나카페라떼", 100, 150), _line("4,500", 600, 150),
        _line("L타피오카필 1개추가", 120, 190), _line("1,000", 600, 190),
        _line("합계", 100, 300), _line("5,500", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "달고나카페라떼", "quantity": 1, "price": 4500},
            {"name": "타피오카펄 1개추가", "quantity": 1, "price": 1000},
        ],
        "total_amount": 5500,
    }
    merged, _ = _merge_with(parsed, lines)
    assert _items(merged) == [("달고나카페라떼", 4500, 0, [1000])]


def test_zero_price_lines_under_menu_on_delivery_receipt_are_options():
    lines = [_line("[배달]주문서", 100, 50)] + _HEADER + [
        _line("순살해장국", 100, 150), _line("12,500", 600, 150),
        _line("합계금액", 100, 300), _line("12,500", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "순살해장국", "quantity": 1, "price": 12500},
            {"name": "조리 (끓여서)", "quantity": 1, "price": 0},
            {"name": "보통맛", "quantity": 1, "price": 0},
        ],
        "total_amount": 12500,
    }
    merged, _ = _merge_with(parsed, lines)
    assert _items(merged) == [("순살해장국", 12500, 0, [0, 0])]


def test_zero_price_gift_on_pos_receipt_stays_an_item():
    """편의점 증정품(`스낵면대컵 0`)은 품목이다. 배달 영수증이 아니면 0원 줄을 옵션으로 내리지 않는다."""
    lines = _HEADER + [
        _line("신라면큰사발", 100, 150), _line("1,600", 600, 150),
        _line("스낵면대컵", 100, 200), _line("0", 600, 200),
        _line("합계", 100, 300), _line("1,600", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "신라면큰사발", "quantity": 1, "price": 1600},
            {"name": "스낵면대컵", "quantity": 1, "price": 0},
        ],
        "total_amount": 1600,
    }
    merged, _ = _merge_with(parsed, lines)
    assert [i["name"] for i in merged["items"]] == ["신라면큰사발", "스낵면대컵"]


# ---------------------------------------------------------------- 안전장치

def test_ambiguous_paid_amounts_change_nothing():
    """같은 비용으로 서로 다른 결제액이 맞으면 손대지 않는다."""
    ev = rec.Evidence()
    ev.discounts.update({1000: 1, 2000: 1})
    items = [{"name": "A", "price": 10000, "sub_items": []}]
    # 9,000(할인 1,000) 과 8,000(할인 2,000) 이 같은 순위·같은 비용
    candidates = [rec.Candidate(9000, 0), rec.Candidate(8000, 0)]
    assert rec.solve(items, 0, 0, candidates, ev, rec.Counter()) is None


def test_no_candidates_means_no_change():
    parsed = {"items": [{"name": "아메리카노", "quantity": 1, "price": 4500}], "total_amount": None}
    merged, _ = _merge_with(parsed, [])
    assert merged["items"][0]["price"] == 4500
    assert merged["total_verified"] is False


def test_fuzzy_amounts():
    assert rec.fuzzy_amounts("-1,00") == [-1000]
    assert rec.fuzzy_amounts("ㄴ기본배달팁 1;000") == [1000]
    assert rec.fuzzy_amounts("결제금이 W11.305") == [11305]
    assert rec.fuzzy_amounts("2026-09-03") == []
    assert rec.fuzzy_amounts("010-4007-0371") == []


def test_private_keys_never_leak_into_response():
    lines = _HEADER + [
        _line("꼬마김밥", 100, 150), _line("3,500", 600, 150),
        _line("참치김밥", 100, 200), _line("4,000", 600, 200),
        _line("합계", 100, 300), _line("7,500", 600, 300),
    ]
    parsed = {"items": [{"name": "꼬마김밥", "quantity": 1, "price": 3000}], "total_amount": 7500}
    merged, _ = _merge_with(parsed, lines)
    for item in merged["items"]:
        assert not [k for k in item if k.startswith("_")], item


# ---------------------------------------------------------------- 변이 테스트에서 살아남은 경우

def test_unit_price_fixed_from_quantity_when_ocr_lost_the_name():
    """좌표가 이름을 못 읽어(`00 O6MO( 너T`) 금액으로 짝지을 수 없어도 수량으로 고친다."""
    lines = _HEADER + [
        _line("고래밥", 100, 150), _line("1,980", 600, 150),
        _line("8801155735135", 100, 200), _line("1,000", 400, 200), _line("4", 500, 200), _line("4,000", 600, 200),
        _line("합계", 100, 300), _line("5,980", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "고래밥", "quantity": 1, "price": 1980},
            {"name": "얼라이브 오렌지", "quantity": 4, "price": 1000},
        ],
        "total_amount": 5980,
    }
    merged, _ = _merge_with(parsed, lines)
    assert [i["price"] for i in merged["items"]] == [1980, 4000]


def test_subtotal_amount_is_never_a_discount_candidate():
    """배민 실측(crop): `소계금액 16,000` 을 할인으로 더해 맞춘 일이 있었다. 라벨 붙은 금액은 할인 후보가 아니다.

    VLM 이 옵션을 품목으로 올려 품목 합이 29,000 이 됐다. 29,000 − 16,000 = 13,000 이 우연히
    결제액과 맞지만, 16,000 은 소계라 할인일 수 없다. 못 맞추면 검증 실패로 남겨야 한다.
    """
    lines = [_line("[배달]주문서", 100, 50)] + _HEADER + [
        _line("한그릇 정식", 100, 150), _line("13,000", 600, 150),
        _line("소계금액", 100, 300), _line("16,000", 600, 300),
        _line("합계금액", 100, 380), _line("13,000", 600, 380),
    ]
    parsed = {
        "items": [
            {"name": "한그릇 정식", "quantity": 1, "price": 13000},
            {"name": "순살", "quantity": 1, "price": 15000},
            {"name": "담은 무", "quantity": 1, "price": 1000},
        ],
        "total_amount": 13000,
    }
    merged, _ = _merge_with(parsed, lines)
    assert merged["discount"] != 16000
    assert merged["items_verified"] is False


def test_zero_price_line_after_marked_option_is_also_an_option():
    """통닭집 실측: `▶후라이드 0`(OCR `※우라이드`) 다음 `간장 0` 도 같은 메뉴의 옵션이다."""
    lines = _HEADER + [
        _line("순살 (반반)", 100, 150), _line("13,000", 600, 150),
        _line("※우라이드", 120, 190), _line("0", 600, 190),
        _line("간장", 120, 230), _line("0", 600, 230),
        _line("합계", 100, 300), _line("13,000", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "순살 (반반)", "quantity": 1, "price": 13000},
            {"name": "후라이드", "quantity": 1, "price": 0},
            {"name": "간장", "quantity": 1, "price": 0},
        ],
        "total_amount": 13000,
    }
    merged, _ = _merge_with(parsed, lines)
    assert _items(merged) == [("순살 (반반)", 13000, 0, [0, 0])]


# ---------------------------------------------------------------- 독립 리뷰가 만든 반례
# 리뷰어가 실측에 없는 모양으로 만든 영수증들이다. 예전 코드는 여기서 품목을 지우거나 틀린 결제액을
# "검증됨" 으로 내보냈다.

def _row(name, price, y, qty=None):
    out = [_line(name, 100, y)]
    if qty is not None:
        out.append(_line(str(qty), 400, y))
    if price is not None:
        out.append(_line(price, 600, y))
    return out


def _it(name, price, qty=1, subs=None):
    return {"name": name, "quantity": qty, "price": price, "sub_items": subs or []}


def _run(parsed, lines):
    merged, corrections = _merge_with(parsed, lines)
    return merged, corrections


def test_split_payment_voucher_plus_card_keeps_all_items():
    lines = _HEADER + _row("돈까스정식", "10,000", 150) + _row("치즈스틱", "5,000", 200) \
        + _row("합계", "15,000", 300) + _row("상품권", "5,000", 350) + _row("카드결제", "10,000", 400)
    merged, _ = _run({"items": [_it("돈까스정식", 10000), _it("치즈스틱", 5000)], "total_amount": 15000}, lines)
    assert merged["total_amount"] == 15000
    assert len(merged["items"]) == 2


def test_split_payment_card_plus_points_keeps_all_items():
    lines = _HEADER + _row("아메리카노", "4,500", 150) + _row("케이크", "6,500", 200) + _row("샌드위치", "3,000", 250) \
        + _row("합계", "14,000", 300) + _row("카드결제금액", "11,000", 350) + _row("포인트결제금액", "3,000", 400)
    parsed = {"items": [_it("아메리카노", 4500), _it("케이크", 6500), _it("샌드위치", 3000)], "total_amount": 14000}
    merged, _ = _run(parsed, lines)
    assert merged["total_amount"] == 14000
    assert len(merged["items"]) == 3


def test_split_payment_two_cards_keeps_all_items():
    lines = _HEADER + _row("삼겹살", "30,000", 150) + _row("된장찌개", "8,000", 200) + _row("소주", "5,000", 250) \
        + _row("합계", "43,000", 300) + _row("신용카드", "35,000", 350) + _row("신용카드", "8,000", 400)
    parsed = {"items": [_it("삼겹살", 30000), _it("된장찌개", 8000), _it("소주", 5000)], "total_amount": 43000}
    merged, _ = _run(parsed, lines)
    assert merged["total_amount"] == 43000
    assert len(merged["items"]) == 3


def test_vlm_total_contradicting_ocr_is_not_verified():
    """OCR 은 15,000 을 두 번 찍었는데 VLM 은 품목 하나를 잘못 읽고 합계도 14,000 으로 맞췄다."""
    lines = _HEADER + _row("김치찌개", "9,000", 150) + [_line("제육볶음", 100, 200), _line("6.0O0", 600, 200)] \
        + _row("합계", "15,000", 300) + _row("카드결제", "15,000", 350)
    merged, _ = _run({"items": [_it("김치찌개", 9000), _it("제육볶음", 5000)], "total_amount": 14000}, lines)
    assert merged["total_verified"] is False
    assert merged["items_verified"] is False


def test_vlm_only_total_is_never_verified():
    lines = _HEADER + _row("김치찌개", "9,000", 150) + [_line("제육볶음", 100, 200)]
    merged, _ = _run({"items": [_it("김치찌개", 9000), _it("제육볶음", 5000)], "total_amount": 14000}, lines)
    assert merged["total_verified"] is False
    assert merged["items_verified"] is False


def test_card_number_digits_are_not_a_paid_amount():
    lines = _HEADER + _row("커피", "10,000", 150) + _row("쿠키", "1,000", 200) + _row("합계", "11,000", 300) \
        + [_line("신용카드", 100, 350), _line("****", 300, 350), _line("1000", 600, 350)]
    merged, _ = _run({"items": [_it("커피", 10000), _it("쿠키", 1000)], "total_amount": 11000}, lines)
    assert merged["total_amount"] == 11000
    assert len(merged["items"]) == 2


def test_point_use_is_payment_not_discount():
    """포인트는 결제 수단이다. 정산할 돈은 20,000 이고 할인이 아니다."""
    lines = _HEADER + _row("소고기", "12,000", 150) + _row("양파", "8,000", 200) \
        + _row("합계", "20,000", 300) + _row("포인트사용", "-5,000", 350) + _row("결제금액", "15,000", 400)
    merged, _ = _run({"items": [_it("소고기", 12000), _it("양파", 8000)], "total_amount": 20000}, lines)
    assert merged["discount"] == 0
    assert merged["total_amount"] == 20000


def test_restored_item_priced_like_vat_is_kept():
    lines = _HEADER + _row("아메리카노", "4,500", 150) + _row("카페라떼", "5,500", 200) + _row("마들렌", "1,000", 250) \
        + _row("합계", "11,000", 350) + _row("부가세", "1,000", 400) + _row("카드결제", "11,000", 450)
    merged, _ = _run({"items": [_it("아메리카노", 4500), _it("카페라떼", 5500)], "total_amount": 11000}, lines)
    assert [i["name"] for i in merged["items"]] == ["아메리카노", "카페라떼", "마들렌"]
    assert merged["items_verified"] is True


def test_unit_price_needs_the_multiplied_amount_on_the_receipt():
    """콜라 ×2 3,000 은 맞게 읽었고 사이다 3,000 을 놓쳤다. 콜라를 6,000 으로 부풀려 맞추면 안 된다."""
    lines = _HEADER + _row("콜라", "3,000", 150, qty=2) + _row("김밥", "3,500", 200) + [_line("사이다", 100, 250)] \
        + _row("합계", "9,500", 300) + _row("카드결제", "9,500", 350)
    merged, _ = _run({"items": [_it("콜라", 3000, qty=2), _it("김밥", 3500)], "total_amount": 9500}, lines)
    assert merged["items"][0]["price"] == 3000
    assert merged["items_verified"] is False


def test_delivery_request_row_does_not_unlock_fee_gap():
    lines = _HEADER + _row("후라이드치킨", "18,000", 150) + [_line("콜라1.25L", 100, 200)] \
        + _row("주문금액", "20,000", 300) + [_line("배달 요청사항: 문 앞에 놔주세요", 100, 350)] \
        + _row("결제금액", "20,000", 400) + [_line("배달의민족", 100, 450)]
    merged, _ = _run({"items": [_it("후라이드치킨", 18000)], "total_amount": 20000}, lines)
    assert merged["delivery_fee"] == 0
    assert merged["items_verified"] is False


def test_colon_in_option_or_product_name_is_kept():
    lines = _HEADER + _row("아메리카노", "4,000", 150) + [_line("- 샷추가: 1샷", 120, 200), _line("500", 600, 200)] \
        + _row("합계", "4,500", 300) + _row("카드결제", "4,500", 350)
    parsed = {"items": [_it("아메리카노", 4000, subs=[{"name": "샷추가: 1샷", "price": 500}])], "total_amount": 4500}
    merged, _ = _run(parsed, lines)
    assert merged["items"][0]["sub_items"] == [{"name": "샷추가: 1샷", "price": 500}]

    lines = _HEADER + _row("[행사]콜라:500ml", "2,000", 150) + _row("새우깡", "1,500", 200) \
        + _row("합계", "3,500", 300) + _row("카드결제", "3,500", 350)
    merged, _ = _run({"items": [_it("[행사]콜라:500ml", 2000), _it("새우깡", 1500)], "total_amount": 3500}, lines)
    assert [i["name"] for i in merged["items"]] == ["[행사]콜라:500ml", "새우깡"]


def test_product_names_with_label_words_are_kept_when_sum_balances():
    lines = _HEADER + _row("종합계란 15구", "6,900", 150) + _row("PRIVATE 생수", "900", 200) + _row("카드", "3,000", 250) \
        + _row("합계", "10,800", 300) + _row("카드결제", "10,800", 350)
    parsed = {"items": [_it("종합계란 15구", 6900), _it("PRIVATE 생수", 900), _it("카드", 3000)], "total_amount": 10800}
    merged, _ = _run(parsed, lines)
    assert [i["name"] for i in merged["items"]] == ["종합계란 15구", "PRIVATE 생수", "카드"]
    assert merged["items_verified"] is True


def test_separately_bought_item_is_not_demoted_into_set_option():
    lines = _HEADER + _row("불고기버거세트", "8,900", 150) + [_line("ㄴ콜라", 120, 200)] + [_line("ㄴ감자튀김", 120, 230)] \
        + _row("콜라", "2,000", 280) + _row("합계", "10,900", 340) + _row("카드결제", "10,900", 380)
    parsed = {"items": [_it("불고기버거세트", 8900, subs=[{"name": "콜라", "price": 0}, {"name": "감자튀김", "price": 0}]),
                        _it("콜라", 2000)], "total_amount": 10900}
    merged, _ = _run(parsed, lines)
    assert [i["name"] for i in merged["items"]] == ["불고기버거세트", "콜라"]


def test_nameless_price_row_does_not_steal_a_named_items_discount():
    lines = _HEADER + _row("서울우유1L", "3,000", 150) + _row("행사할인", "-500", 200) \
        + [_line("3,000", 600, 250)] + [_line("-300", 600, 300)] \
        + _row("합계", "5,200", 400) + _row("카드결제", "5,200", 450)
    merged, _ = _run({"items": [_it("서울우유1L", 3000), _it("풀무원두부", 3000)], "total_amount": 5200}, lines)
    assert [i["discount"] for i in merged["items"]] == [500, 300]


def test_float_and_string_prices_are_returned_as_int():
    lines = _HEADER + _row("라면", "3,000", 150) + _row("김밥", "2,500", 200) + _row("합계", "5,500", 300)
    parsed = {"items": [_it("라면", 3000.0), _it("김밥", "2,500", subs=[{"name": "치즈", "price": "500"}])],
              "total_amount": 5500.0}
    merged, _ = _run(parsed, lines)
    prices = [merged["items"][0]["price"], merged["items"][1]["price"], merged["items"][1]["sub_items"][0]["price"]]
    assert all(type(p) is int for p in prices), prices


def test_refund_receipt_is_left_alone():
    lines = _HEADER + _row("반품 셔츠", "-29,000", 150) + _row("합계", "-29,000", 300)
    merged, corrections = _run({"items": [_it("반품 셔츠", -29000)], "total_amount": -29000}, lines)
    assert "reconciled" not in _reasons(corrections)


# ---------------------------------------------------------------- 변이 테스트로 보강한 경우

def test_contested_vlm_total_does_not_justify_dropping_items():
    """OCR 합계(16,000)와 VLM 합계(10,000)가 다르면 VLM 합계에 맞추려고 품목을 지우지 않는다."""
    lines = _HEADER + _row("돈까스", "10,000", 150) + _row("우동", "5,000", 200) + _row("합계", "16,000", 300)
    merged, _ = _run({"items": [_it("돈까스", 10000)], "total_amount": 10000}, lines)
    assert [i["name"] for i in merged["items"]] == ["돈까스", "우동"]
    assert merged["total_verified"] is False


def test_card_digits_without_total_label_do_not_become_paid_amount():
    lines = _HEADER + _row("커피", "10,000", 150) + _row("쿠키", "1,000", 200) \
        + [_line("신용카드", 100, 350), _line("****", 300, 350), _line("1000", 600, 350)]
    merged, _ = _run({"items": [_it("커피", 10000), _it("쿠키", 1000)], "total_amount": 11000}, lines)
    assert len(merged["items"]) == 2
    assert merged["total_amount"] == 11000


def test_paid_label_differing_from_balanced_total_needs_strong_discount():
    """합계 15,000 이 품목과 맞는데 카드 10,000 만 읽혔다(나머지 현금 줄은 깨짐). 약한 근거로 맞추지 않는다."""
    lines = _HEADER + _row("돈까스정식", "10,000", 150) + _row("치즈스틱", "5,000", 200) \
        + _row("합계", "15,000", 300) + _row("헌큼", "5,000", 350) + _row("카드결제", "10,000", 400)
    merged, _ = _run({"items": [_it("돈까스정식", 10000), _it("치즈스틱", 5000)], "total_amount": 15000}, lines)
    assert merged["total_amount"] == 15000
    assert merged["discount"] == 0


def test_probable_label_item_dropped_only_when_it_balances():
    """VLM 이 결제 수단 줄 `현금 3,000` 을 품목으로 올렸다. 지워야 합계가 맞으므로 지운다."""
    lines = _HEADER + _row("김밥", "4,000", 150) + _row("합계", "4,000", 300) + _row("현금", "3,0O0", 350)
    merged, _ = _run({"items": [_it("김밥", 4000), _it("현금", 3000)], "total_amount": 4000}, lines)
    assert [i["name"] for i in merged["items"]] == ["김밥"]
    assert merged["items_verified"] is True


def test_point_and_voucher_rows_are_tenders_not_discounts():
    from layout import analyze_receipt

    lines = _HEADER + _row("소고기", "12,000", 150) + _row("합계", "12,000", 300) \
        + _row("포인트사용", "-5,000", 350) + _row("상품권", "2,000", 400)
    layout = analyze_receipt(lines)
    ev = rec.collect_evidence(layout, lines)
    assert not ev.discounts
    assert sorted(ev.tenders) == [2000, 5000]

    # 적립·잔여 포인트는 결제에 쓴 돈이 아니라 분할 결제 계산에 넣지 않는다
    lines = _HEADER + _row("운동화", "50,000", 150) + _row("합계", "50,000", 300) \
        + _row("카드결제", "45,000", 360) + _row("적립포인트", "5,000", 400) + _row("잔여포인트", "1,200", 430)
    ev = rec.collect_evidence(analyze_receipt(lines), lines)
    assert ev.tenders == []


def test_delivery_notice_rows_are_not_unread_fee_rows():
    from layout import _to_tokens, group_rows

    def row_of(*texts):
        return group_rows(_to_tokens([_line(t, 100 + 300 * k, 100) for k, t in enumerate(texts)]))[0]

    assert rec._is_unread_fee_row(row_of("알종배달", "--"))  # 번진 배달팁
    assert rec._is_unread_fee_row(row_of("배달팁", "3,0O0"))  # 깨진 금액
    assert not rec._is_unread_fee_row(row_of("배달 요청사항", "--"))
    assert not rec._is_unread_fee_row(row_of("배달팁", "3,000"))  # 금액을 읽었다
    assert not rec._is_unread_fee_row(row_of("배달", "1600-0987"))  # 전화번호는 금액 자리가 아니다
    assert not rec._is_unread_fee_row(row_of("배달", "19:42"))
    assert not rec._is_unread_fee_row(row_of("배달 정성 가득", "--"))  # 긴 문구의 `배달` 은 라벨이 아니다


def test_reconcile_skips_refunds():
    from layout import analyze_receipt

    lines = _HEADER + _row("반품 셔츠", "-29,000", 150) + _row("합계", "29,000", 300)
    parsed = {"items": [_it("반품 셔츠", 29000)], "total_amount": -29000}
    assert rec.reconcile(parsed, analyze_receipt(lines), lines, []) is False


# ---------------------------------------------------------------- 독립 리뷰 2차 반례

def test_footer_phone_and_time_rows_do_not_unlock_fee_gap():
    for tail in ([_line("배달의민족 고객센터", 100, 450), _line("1600-0987", 500, 450)],
                 [_line("배달완료", 100, 450), _line("19:42", 500, 450)]):
        lines = _HEADER + _row("후라이드치킨", "18,000", 150) + [_line("콜라1.25L", 100, 200)] \
            + _row("합계", "20,000", 300) + _row("결제금액", "20,000", 350) + tail
        merged, _ = _run({"items": [_it("후라이드치킨", 18000)], "total_amount": 20000}, lines)
        assert merged["delivery_fee"] == 0
        assert merged["items_verified"] is False


def test_vlm_total_equal_to_an_item_price_is_not_verified():
    """합계 줄을 못 읽었고 VLM 합계(7,000)는 남은 품목 금액일 뿐이다. 영수증에 따로 찍힌 합계가 아니다."""
    lines = _HEADER + _row("짜장면", "7,000", 150) + [_line("짬뽕", 100, 200, conf=0.5), _line("7,0O0", 600, 200)] \
        + [_line("합 계", 100, 300, conf=0.4)]
    merged, _ = _run({"items": [_it("짜장면", 7000)], "total_amount": 7000}, lines)
    assert merged["items_verified"] is False


def test_restored_item_is_not_dropped_to_fit_a_vlm_only_total():
    lines = _HEADER + _row("짜장면", "7,000", 150) + _row("탕수육", "15,000", 200) + [_line("합 계", 100, 300, conf=0.4)]
    merged, _ = _run({"items": [_it("짜장면", 7000)], "total_amount": 7000}, lines)
    assert [i["name"] for i in merged["items"]] == ["짜장면", "탕수육"]
    assert merged["items_verified"] is False


def test_printed_discount_beats_dropping_a_product_with_label_words():
    """`카드케이스 2,000` 과 할인 `-2,000` 이 같은 금액이면 상품을 지우지 말고 할인을 더한다."""
    lines = _HEADER + _row("카드케이스", "2,000", 150) + _row("보조배터리", "20,000", 200) \
        + _row("합계", "22,000", 300) + _row("제휴사 청구", "-2,000", 330) + _row("결제금액", "20,000", 360)
    merged, _ = _run({"items": [_it("카드케이스", 2000), _it("보조배터리", 20000)], "total_amount": 20000}, lines)
    assert [i["name"] for i in merged["items"]] == ["카드케이스", "보조배터리"]
    assert merged["discount"] == 2000


def test_products_with_tender_words_are_not_payments():
    """`캐시미어 머플러` 는 상품이다(캐시 = 결제 수단 어휘). 분할 결제로 오인하면 안 된다."""
    lines = _HEADER + _row("캐시미어 머플러", "39,000", 150) + _row("양말", "5,000", 200) \
        + _row("합계", "44,000", 300) + _row("할인", "-39,000", 330) + _row("카드결제", "5,000", 400)
    merged, _ = _run({"items": [_it("캐시미어 머플러", 39000), _it("양말", 5000)], "total_amount": 5000}, lines)
    assert merged["total_amount"] == 5000
    assert merged["discount"] == 39000
    assert merged["items_verified"] is True


def test_point_discount_is_a_discount_and_earned_points_are_ignored():
    lines = _HEADER + _row("크림빵", "3,000", 150) + _row("우유", "2,000", 200) \
        + _row("합계", "5,000", 300) + _row("T멤버십 포인트할인", "-500", 350) + _row("카드결제", "4,500", 400) \
        + _row("적립포인트", "4,000", 450)
    merged, _ = _run({"items": [_it("크림빵", 3000), _it("우유", 2000)], "total_amount": 4500}, lines)
    assert merged["total_amount"] == 4500
    assert merged["discount"] == 500
    assert merged["items_verified"] is True


def test_item_with_unreadable_price_printed_separately_is_not_demoted():
    lines = _HEADER + _row("불고기버거세트", "8,900", 150) + [_line("ㄴ콜라", 120, 200)] \
        + [_line("콜라", 100, 260), _line("2,0O0", 600, 260)] + _row("합계", "10,900", 340) + _row("카드결제", "10,900", 380)
    merged, _ = _run({"items": [_it("불고기버거세트", 8900), _it("콜라", 2000)], "total_amount": 10900}, lines)
    assert [i["name"] for i in merged["items"]] == ["불고기버거세트", "콜라"]


# ---------------------------------------------------------------- 빠진 품목 되살리기 (add_item)

def _blurry_receipt(printed_total=12000, blurry_conf=0.7, blurry_name="흐린이름"):
    """VLM 이 두 번째 품목을 빠뜨렸고, OCR 은 그 줄을 흐리게(신뢰도 낮게) 읽은 영수증."""
    lines = _HEADER + [
        _line("떡볶이", 100, 150), _line("5,000", 600, 150),
        _line(blurry_name, 100, 200, conf=blurry_conf), _line("3,000", 600, 200),
        _line("순대", 100, 250), _line("4,000", 600, 250),
        _line("합계", 100, 300), _line(f"{printed_total:,}", 600, 300),
        _line("카드결제", 100, 350), _line(f"{printed_total:,}", 600, 350),
    ]
    parsed = {
        "items": [
            {"name": "떡볶이", "quantity": 1, "price": 5000, "sub_items": []},
            {"name": "순대", "quantity": 1, "price": 4000, "sub_items": []},
        ],
        "total_amount": printed_total,
        "payment_method": "카드",
    }
    return parsed, lines


def test_blurry_line_is_restored_when_the_printed_total_needs_it():
    parsed, lines = _blurry_receipt()
    merged, corrections = _merge_with(parsed, lines)
    assert [(i["price"], i["discount"]) for i in merged["items"]] == [(5000, 0), (3000, 0), (4000, 0)]
    assert merged["items"][1]["name"] == "흐린이름"  # 영수증 순서대로 끼워 넣는다
    assert merged["total_verified"] is True and merged["items_verified"] is True
    assert [c["reason"] for c in corrections if c["field"] == "items"] == ["ocr_recovered"]


def test_blurry_line_is_not_restored_when_the_total_already_balances():
    """합계가 이미 맞으면 흐린 줄은 중복 읽기일 수 있다. 되살리지 않는다."""
    parsed, lines = _blurry_receipt(printed_total=9000)
    merged, corrections = _merge_with(parsed, lines)
    assert [i["price"] for i in merged["items"]] == [5000, 4000]
    assert "ocr_recovered" not in _reasons(corrections)


def test_blurry_line_is_not_restored_when_it_still_would_not_balance():
    """되살려도 합계가 안 맞으면(다른 품목도 빠졌다) 추측으로 채우지 않는다."""
    parsed, lines = _blurry_receipt(printed_total=14000)
    merged, corrections = _merge_with(parsed, lines)
    assert [i["price"] for i in merged["items"]] == [5000, 4000]
    assert merged["items_verified"] is False
    assert "ocr_recovered" not in _reasons(corrections)


def test_a_name_with_fewer_than_two_hangul_syllables_is_never_restored():
    """한글이 한 글자뿐인 흐린 이름(`가`, `a가1`)은 이름이라 하기 어렵다. 되살리지 않는다."""
    for name in ("가", "a가1"):
        parsed, lines = _blurry_receipt(blurry_name=name)
        merged, _ = _merge_with(parsed, lines)
        assert [i["price"] for i in merged["items"]] == [5000, 4000], name


def test_blurry_line_with_a_discount_is_restored_with_its_discount():
    """되살린 품목의 할인은 영수증 단위 할인으로 이미 센 것이라, 품목으로 옮기면 전체 할인이 그만큼 준다."""
    lines = _HEADER + [
        _line("떡볶이", 100, 150), _line("5,000", 600, 150),
        _line("흐린이름", 100, 200, conf=0.7), _line("3,000", 600, 200),
        _line("-500", 600, 240),
        _line("순대", 100, 280), _line("4,000", 600, 280),
        _line("할인금액", 100, 330), _line("-500", 600, 330),
        _line("합계", 100, 380), _line("11,500", 600, 380),
        _line("카드결제", 100, 430), _line("11,500", 600, 430),
    ]
    parsed = {
        "items": [
            {"name": "떡볶이", "quantity": 1, "price": 5000, "sub_items": []},
            {"name": "순대", "quantity": 1, "price": 4000, "sub_items": []},
        ],
        "total_amount": 11500, "payment_method": "카드",
    }
    merged, _ = _merge_with(parsed, lines)
    assert [(i["price"], i["discount"]) for i in merged["items"]] == [(5000, 0), (3000, 500), (4000, 0)]
    assert merged["discount"] == 0
    assert rec.computed_total(merged["items"], merged["discount"], merged["delivery_fee"]) == 11500
    assert merged["items_verified"] is True


def test_restore_never_uses_a_total_only_the_vlm_claims():
    """결제액이 영수증에 찍혀 있지 않으면 품목을 지어내서 VLM 합계에 맞추지 않는다."""
    parsed, lines = _blurry_receipt()
    lines = [l for l in lines if l["text"] not in ("합계", "12,000", "카드결제")]
    merged, corrections = _merge_with(parsed, lines)
    assert [i["price"] for i in merged["items"]] == [5000, 4000]
    assert "ocr_recovered" not in _reasons(corrections)


def test_two_blurry_lines_can_be_restored_together_but_not_three():
    lines = _HEADER + [
        _line("김치볶음밥", 100, 150, conf=0.7), _line("1,000", 600, 150),
        _line("된장찌개정식", 100, 200, conf=0.7), _line("2,000", 600, 200),
        _line("불고기덮밥", 100, 250, conf=0.7), _line("4,000", 600, 250),
        _line("치즈돈까스", 100, 300), _line("8,000", 600, 300),
        _line("합계", 100, 350), _line("15,000", 600, 350),
        _line("카드결제", 100, 400), _line("15,000", 600, 400),
    ]

    def run(vlm_items):
        parsed = {"items": [dict(name=n, quantity=1, price=p, sub_items=[]) for n, p in vlm_items],
                  "total_amount": 15000, "payment_method": "카드"}
        return _merge_with(parsed, [dict(l) for l in lines])[0]

    two_missing = run([("불고기덮밥", 4000), ("치즈돈까스", 8000)])
    assert [i["price"] for i in two_missing["items"]] == [1000, 2000, 4000, 8000]  # 영수증 순서
    assert two_missing["items_verified"] is True
    three_missing = run([("치즈돈까스", 8000)])
    assert [i["price"] for i in three_missing["items"]] == [8000]  # 셋을 한꺼번에 지어내진 않는다


def test_restore_edit_cost_is_between_discounts_and_deletions():
    assert rec._ADD_ITEM_COST > 1.0
    assert rec._ADD_ITEM_COST < 1.5


def test_restore_consistency_checks_the_real_total_after_restoring():
    """합산 추정(가 5,000 + 나 3,000 = 8,000)과 실제(나는 500 할인돼 2,500)가 다르면 해를 버린다."""
    items = [{"name": "가", "price": 5000, "discount": 0, "sub_items": []}]
    restorable = [{"name": "나", "price": 3000, "discount": 500, "_pos": 1}]
    combo = (rec.Edit("add_item", 1.3, amount=3000, ref=0),)
    # 전체 할인이 없을 때: 품목 할인 500 이 그대로 합계를 깎아 실제 합계는 7,500
    assert not rec._restore_consistent(items, 0, 0, combo, restorable, 8000)
    assert rec._restore_consistent(items, 0, 0, combo, restorable, 7500)
    # 전체 할인 500 이 이미 있으면 그 500 이 품목 할인으로 옮겨 갈 뿐이라 합계는 여전히 7,500
    assert rec._restore_consistent(items, 500, 0, combo, restorable, 7500)
    assert not rec._restore_consistent(items, 500, 0, combo, restorable, 8000)


def test_restore_is_offered_only_if_the_receipt_wide_discount_can_absorb_its_discount():
    from collections import Counter

    items = [{"name": "가", "price": 5000, "discount": 0, "sub_items": []}]
    restorable = [{"name": "나", "price": 3000, "discount": 500, "_pos": 1}]
    kinds = lambda discount: [e.kind for e in rec._candidate_edits(items, rec.Evidence(), set(), Counter(),
                                                                    discount, restorable)]
    assert "add_item" not in kinds(0)
    assert "add_item" in kinds(500)


def test_no_private_keys_leak_after_restoring():
    parsed, lines = _blurry_receipt()
    merged, _ = _merge_with(parsed, lines)
    assert "_restorable" not in merged
    assert all(not k.startswith("_") for i in merged["items"] for k in i)


def test_restoring_two_items_is_rejected_when_their_discounts_exceed_the_receipt_wide_discount():
    """각각은 전체 할인(500)에 흡수되지만(300 ≤ 500) 둘을 함께 되살리면(600) 넘친다. 합산 추정이 틀린다."""
    from collections import Counter

    items = [{"name": "가", "price": 5000, "discount": 0, "sub_items": []}]
    restorable = [
        {"name": "나", "price": 2000, "discount": 300, "_pos": 1},
        {"name": "다", "price": 3000, "discount": 300, "_pos": 2},
    ]
    ev = rec.Evidence()
    # 합산 추정: 5000 - 500 + 2000 + 3000 = 9500. 실제로는 5000 + 1700 + 2700 = 9400
    assert rec.solve(items, 500, 0, [rec.Candidate(9500, 0)], ev, Counter(), restorable) is None
    solution = rec.solve(items, 500, 0, [rec.Candidate(9400, 0)], ev, Counter(), restorable)
    assert solution is None  # 합산 추정(9500)과 맞지 않는 값은 후보 수정 조합으로 만들 수 없다


# ---------------------------------------------------------------- add_item 후보 거르기

def _restore_case(restorable, items=None):
    from reconcile import Evidence, solve, Candidate
    from collections import Counter

    items = items if items is not None else [{"name": "과자", "quantity": 1, "price": 1000, "sub_items": [], "discount": 0}]
    ev = Evidence()
    return solve(items, 0, 0, [Candidate(3000, 0)], ev, Counter(), restorable)


def _cand(name, price):
    return {"name": name, "quantity": 1, "price": price, "discount": 0, "_pos": 5}


def test_restore_offers_a_plain_product_line():
    sol = _restore_case([_cand("새우깡", 2000)])
    assert sol is not None and [e.kind for e in sol.edits] == ["add_item"]


def test_restore_never_offers_header_delivery_or_payment_lines():
    for name in ("상품명 수량 금액", "배달팁", "카드결제", "할인쿠폰", "합계금액"):
        assert _restore_case([_cand(name, 2000)]) is None, name


def test_restore_never_offers_a_summary_amount():
    from reconcile import Evidence, solve, Candidate
    from collections import Counter

    items = [{"name": "과자", "quantity": 1, "price": 1000, "sub_items": [], "discount": 0}]
    ev = Evidence()
    ev.summary = {2000}
    assert solve(items, 0, 0, [Candidate(3000, 0)], ev, Counter(), [_cand("새우깡", 2000)]) is None


def test_restore_two_different_lines_that_both_fit_is_ambiguous():
    assert _restore_case([_cand("새우깡", 2000), _cand("감자깡", 2000)]) is None


def test_restore_never_offers_a_label_like_name_outside_the_word_list():
    # `거자이체`(`이체` = 계좌이체 라벨) 는 헤더·배달 어휘는 아니지만 품목이 아니다
    assert _restore_case([_cand("거자이체", 2000)]) is None


def test_restore_item_discount_larger_than_the_receipt_discount_is_not_offered():
    cand = dict(_cand("새우깡", 2500), discount=500)
    # 2500 - 500 = 2000 이 더해져야 3000 이 되지만, 할인 500 이 영수증 할인 0 에 이미 들어 있지 않다
    assert _restore_case([cand]) is None
