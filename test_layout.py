"""layout.py 구조 복원 및 server._merge 좌표 병합 회귀 테스트.

실행: .venv/bin/python -m pytest test_layout.py -v
"""
from layout import _to_tokens, analyze_receipt, group_rows


def _line(text: str, x1: int, y: int, conf: float = 0.95, char_w: int = 20) -> dict:
    """글자 폭 20px 가정의 합성 OCR 라인."""
    w = max(char_w, char_w * len(text))
    return {"text": text, "confidence": conf, "box": [x1, y, x1 + w, y + 30]}


_HEADER = [_line("상품명", 100, 100), _line("수량", 400, 102), _line("금액", 600, 98)]


def test_group_rows_y_clustering():
    rows = group_rows(_to_tokens(_HEADER + [_line("참치김밥", 100, 150)]))
    assert len(rows) == 2
    assert rows[0].text == "상품명 수량 금액"
    assert rows[1].text == "참치김밥"


def test_single_line_item():
    """유니클로형: 품목명·수량·금액이 한 줄."""
    lines = _HEADER + [
        _line("유니클로(과세)", 100, 150),
        _line("1", 400, 150),
        _line("60,000", 600, 150),
        _line("소계", 100, 200),
        _line("60,000", 600, 200),
    ]
    layout = analyze_receipt(lines)
    assert len(layout.items) == 1
    item = layout.items[0]
    assert (item.name, item.quantity, item.price) == ("유니클로(과세)", 1, 60000)


def test_name_line_then_price_line():
    """백화점형: 품목명 줄 다음 바코드·수량·금액 줄."""
    lines = _HEADER + [
        _line("꼬마김밥A", 100, 150),
        _line("2810000316074", 100, 200),
        _line("1", 400, 200),
        _line("3,500", 600, 200),
        _line("과세물품", 100, 250),
        _line("가액", 250, 250),
        _line("3,182", 600, 250),
    ]
    layout = analyze_receipt(lines)
    assert len(layout.items) == 1
    item = layout.items[0]
    assert (item.name, item.quantity, item.price) == ("꼬마김밥A", 1, 3500)


def test_sub_item_by_prefix():
    lines = [
        _line("아메리카노", 100, 100),
        _line("1", 400, 100),
        _line("4,500", 600, 100),
        _line("- 샷추가", 100, 150),
        _line("500", 600, 150),
    ]
    layout = analyze_receipt(lines)
    assert len(layout.items) == 1
    assert [(s.name, s.price) for s in layout.items[0].sub_items] == [("샷추가", 500)]


def test_sub_item_by_indent():
    lines = [
        _line("아메리카노", 100, 100),
        _line("1", 400, 100),
        _line("4,500", 600, 100),
        _line("사이즈업", 160, 150),  # 3글자 폭 들여쓰기
        _line("500", 600, 150),
    ]
    layout = analyze_receipt(lines)
    assert len(layout.items) == 1
    assert [(s.name, s.price) for s in layout.items[0].sub_items] == [("사이즈업", 500)]


def test_discount_line_becomes_item_discount_not_sub():
    """할인 줄은 하위 옵션이 아니라 그 품목의 discount 다.

    sub_items 에 음수로 넣으면 품목 합에 섞여, 할인을 따로 셀 수 없고 앱도
    옵션과 구분하지 못한다. price 는 할인 전 금액으로 남는다.
    """
    lines = [
        _line("케이크", 100, 100),
        _line("1", 400, 100),
        _line("15,000", 600, 100),
        _line("행사할인", 100, 150),
        _line("-1,000", 600, 150),
    ]
    layout = analyze_receipt(lines)
    assert len(layout.items) == 1
    assert layout.items[0].price == 15000
    assert layout.items[0].discount == 1000
    assert layout.items[0].sub_items == []
    assert layout.discount == 0


def test_summary_discount_row_is_not_double_counted():
    """합계부 `할인금액` 요약줄은 품목별 할인의 재기재이므로 또 빼지 않는다.

    예전에는 이 줄이 직전 품목의 하위 옵션(-1,510)으로 붙어 품목 합에서 할인이
    두 번 빠졌다. 실측 홈푸드마트 영수증이 `할인금액 : -1,510` 다음에 `과세물품`을
    찍기 때문에 품목 영역 안에서 만난다.
    """
    lines = _HEADER + [
        _line("미니꿀약과", 100, 150),
        _line("2,500", 600, 150),
        _line("$특매할인", 130, 200),
        _line("-1,510", 600, 200),
        _line("콘칩", 100, 250),
        _line("1,000", 600, 250),
        _line("할인금액", 100, 300),
        _line("-1,510", 600, 300),
        _line("합계", 100, 350),
        _line("1,990", 600, 350),
    ]
    layout = analyze_receipt(lines)
    assert [(i.name, i.price, i.discount) for i in layout.items] == [
        ("미니꿀약과", 2500, 1510),
        ("콘칩", 1000, 0),
    ]
    assert layout.discount == 0  # 요약줄은 품목별 합과 같으므로 버린다
    assert layout.total_amount == 1990


def test_receipt_level_discount_after_item_region():
    """배달앱은 할인을 품목 영역이 끝난 뒤 부호 없는 양수로 찍는다."""
    lines = _HEADER + [
        _line("파스타", 100, 150),
        _line("12,900", 600, 150),
        _line("소계금액", 100, 200),
        _line("16,000", 600, 200),
        _line("할인금액", 100, 250),
        _line("3,000", 600, 250),
        _line("합계금액", 100, 300),
        _line("13,000", 600, 300),
    ]
    layout = analyze_receipt(lines)
    assert layout.discount == 3000
    assert all(i.discount == 0 for i in layout.items)
    assert layout.total_amount == 13000


def test_zero_and_decoy_discount_rows_are_ignored():
    """금액 0인 자리표시자와 할인이 아닌 `할인` 줄은 세지 않는다.

    실측: `할인: 0, 현재잔액: 5,000`(농·축산물 지원금 한도), `할  인  0`(무인 POS),
    `배달팁 할인 -1,900`(배달팁 순액 안에서 이미 상계됨).
    """
    lines = _HEADER + [
        _line("아이스크림", 100, 150),
        _line("5,300", 600, 150),
        _line("할인", 100, 200),
        _line("0", 600, 200),
        _line("현재잔액", 100, 250),
        _line("5,000", 600, 250),
        _line("배달팁 할인", 100, 300),
        _line("-1,900", 600, 300),
        _line("합계", 100, 350),
        _line("5,300", 600, 350),
    ]
    layout = analyze_receipt(lines)
    assert layout.discount == 0
    assert all(i.discount == 0 for i in layout.items)


def test_money_accepts_won_sign_and_suffix():
    """실측 금액 표기: -₩4,195(쿠팡이츠), 10,500원(EGG DROP), ₩4,900(StoryWay)."""
    from layout import _money_value

    assert _money_value("-₩4,195") == -4195
    assert _money_value("10,500원") == 10500
    assert _money_value("₩4,900") == 4900
    assert _money_value("-520") == -520
    assert _money_value("1,200)") is None  # 옵션 금액의 닫는 괄호 — 음수 아님
    assert _money_value("△1,000") is None  # 63장에서 0건, 도입하지 않는다


def test_total_from_label_row():
    lines = _HEADER + [
        _line("유니클로(과세)", 100, 150),
        _line("60,000", 600, 150),
        _line("소", 100, 200),
        _line("계", 200, 200),
        _line("60,000", 600, 200),
        _line("합", 100, 250),
        _line("계", 200, 250),
        _line("60,000", 600, 250),
    ]
    layout = analyze_receipt(lines)
    assert layout.total_amount == 60000


def test_barcode_and_noise_rows_ignored():
    lines = _HEADER + [
        _line("꼬마김밥A", 100, 150),
        _line("3,500", 600, 200),
        _line("00", 900, 250),  # 잘린 인식 조각
        _line("9,500", 600, 300),  # 품목명 줄 없는 가격 줄은 이름 없는 자리로만 남는다
    ]
    layout = analyze_receipt(lines)
    assert [(i.name, i.price) for i in layout.items if i.name] == [("꼬마김밥A", 3500)]
    # 자리만 남길 뿐 품목으로 되살리지는 않는다(이름 없는 금액은 병합에서 복구 대상이 아니다)
    assert [(i.name, i.price) for i in layout.items if not i.name] == [(None, 9500)]


def test_delivery_receipt_options_and_tight_line_spacing():
    """쿠팡이츠형: + 접두 옵션, 좁은 줄 간격(로컬 높이 기준 그룹핑), 주문금액 요약."""

    def tight(text, x1, y, h=23):
        w = max(20, 20 * len(text))
        return {"text": text, "confidence": 0.95, "box": [x1, y, x1 + w, y + h]}

    lines = [
        _line("메뉴", 100, 100),
        _line("수량", 400, 102),
        _line("금액", 600, 98),
        _line("(BEST)메가치킨마요", 100, 150),
        _line("7,800", 600, 150),
        # 옵션 줄: 큰 폰트 중앙값 기준(0.6*30=18)이면 병합되는 17px 간격
        tight("+ 스팸 1조각", 100, 200),
        tight("1,700", 600, 201),
        tight("+ 치킨1조각", 100, 217),
        tight("1,600", 600, 218),
        _line("해시 포테이토 스틱", 100, 260),
        _line("2,600", 600, 260),
        _line("주문금액", 100, 310),
        _line("20,500", 600, 310),
        _line("총결제금액", 100, 360),
        _line("20,500", 600, 360),
    ]
    layout = analyze_receipt(lines)
    assert [(i.name, i.price) for i in layout.items] == [
        ("(BEST)메가치킨마요", 7800),
        ("해시 포테이토 스틱", 2600),
    ]
    assert [(s.name, s.price) for s in layout.items[0].sub_items] == [
        ("스팸 1조각", 1700),
        ("치킨1조각", 1600),
    ]
    assert layout.total_amount == 20500


def test_card_slip_amount_labels_are_not_items():
    """카드 매출전표: 품목 없이 판매금액/부가세 줄만 있는 경우."""
    lines = [
        _line("현대백화점카드", 100, 100),
        _line("매출표", 400, 100),
        _line("판매금액", 100, 150),
        _line("143,273", 500, 150),
        _line("원", 700, 150),
        _line("부가세", 100, 200),
        _line("14,327", 500, 200),
    ]
    layout = analyze_receipt(lines)
    assert layout.items == []


def test_merge_recovers_missing_item_and_verifies_total():
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    ocr_lines = _HEADER + [
        _line("꼬마김밥A", 100, 150),
        _line("2810000316074", 100, 200),
        _line("1", 400, 200),
        _line("3,500", 600, 200),
        _line("고추장떡볶이", 100, 250),
        _line("1", 400, 250),
        _line("9,500", 600, 250),
        _line("합계", 100, 300),
        _line("13,000", 600, 300),
    ]
    parsed = {
        "items": [{"name": "고추장떡볶이", "quantity": 1, "price": 9500}],
        "total_amount": 13000,
    }
    merged, corrections = server._merge(parsed, ocr_lines)

    names = [it["name"] for it in merged["items"]]
    assert "꼬마김밥A" in names  # VLM 누락 품목이 좌표로 복구됨
    assert merged["total_verified"] is True  # 합계 라벨 옆 숫자와 일치
    assert any(c["reason"] == "ocr_recovered" for c in corrections)


def test_merge_pairs_orphan_barcode_price_with_previous_item():
    """VLM이 품목명 줄과 바코드·금액 줄을 별도 품목으로 쪼갠 경우 병합."""
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    parsed = {
        "items": [
            {"name": "고추장 떡볶이", "quantity": 1, "price": 0},
            {"name": "2810000269400", "quantity": 1, "price": 9500},
        ],
        "total_amount": 9500,
    }
    ocr_lines = [
        _line("고추장떡볶이", 100, 100),
        _line("2810000269400", 100, 150),
        _line("1", 400, 150),
        _line("9,500", 600, 150),
    ]
    merged, corrections = server._merge(parsed, ocr_lines)
    assert [(it["name"], it["price"]) for it in merged["items"]] == [("고추장 떡볶이", 9500)]
    assert any(c["reason"] == "barcode_price_paired" for c in corrections)


def test_merge_baemin_receipt_structure_reconciliation():
    """배민형: VLM이 최상위 품목까지 한 세트의 하위로 묶고 과세가액을 합계로 착각한 경우.

    layout 계층 기준으로 승격(sub_promoted)·재배치(sub_reassigned)하고,
    합계는 결제금액 라벨·품목 합이 일치하는 값으로 바로잡는다(ocr_layout).
    """
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    ocr_lines = [
        _line("제품명", 100, 100),
        _line("수량", 400, 102),
        _line("단가", 600, 98),
        _line("치즈스틱", 100, 150),
        _line("3,800", 600, 150),
        _line("데리세트", 100, 200),
        _line("7,700", 600, 200),
        _line(">선택양념치즈", 130, 250),
        _line("600", 600, 250),
        _line("토네이도쿠키", 100, 300),
        _line("4,200", 600, 300),
        _line("통다리매운셋", 100, 350),
        _line("10,300", 600, 350),
        _line(">L포테이토", 130, 400),
        _line("500", 600, 400),
        _line("배달팁", 100, 450),
        _line("2,700", 600, 450),
        _line("부가세 과세 물품가액", 100, 500),
        _line("27,100", 600, 500),
        _line("결제금액:", 100, 550),
        _line("29,800", 600, 550),
        _line("받을금액:", 100, 600),
        _line("0", 600, 600),
    ]
    # VLM: 최상위 품목 2개를 데리세트 하위로 잘못 묶고, L포테이토도 데리세트에 붙임
    parsed = {
        "items": [
            {"name": "치즈스틱", "quantity": 1, "price": 3800, "sub_items": []},
            {
                "name": "데리세트",
                "quantity": 1,
                "price": 7700,
                "sub_items": [
                    {"name": ">선택양념치즈", "price": 600},
                    {"name": ">토네이도쿠키", "price": 4200},
                    {"name": ">통다리매운셋", "price": 10300},
                    {"name": ">L포테이토", "price": 500},
                ],
            },
            {"name": "배달팁", "quantity": 1, "price": 2700, "sub_items": []},
        ],
        "total_amount": 27100,  # 과세물품가액을 합계로 착각
    }
    merged, corrections = server._merge(parsed, ocr_lines)

    by_name = {it["name"]: it for it in merged["items"]}
    # 배달팁은 품목이 아니라 delivery_fee 로 간다
    assert set(by_name) == {"치즈스틱", "데리세트", "토네이도쿠키", "통다리매운셋"}
    assert merged["delivery_fee"] == 2700
    assert [s["name"] for s in by_name["데리세트"]["sub_items"]] == ["선택양념치즈"]
    assert [(s["name"], s["price"]) for s in by_name["통다리매운셋"]["sub_items"]] == [
        ("L포테이토", 500)
    ]
    assert by_name["토네이도쿠키"]["sub_items"] == []
    # 품목 3800+7700+600+4200+10300+500 = 27,100, 여기에 배달비 2,700 을 더해
    # 29,800 == 결제금액 라벨 (받을금액 0은 무시). 배달비를 검증식에 넣지 않으면
    # 27,100 이 되어 과세물품가액과 헷갈린다.
    assert merged["total_amount"] == 29800
    assert merged["total_verified"] is True
    reasons = {c["reason"] for c in corrections}
    assert {"sub_promoted", "sub_reassigned", "ocr_layout"} <= reasons


def test_merge_demotes_marker_named_items():
    """카페형: OCR이 ▶ 마커를 놓쳐 layout은 최상위로 보지만, VLM 이름의 마커로 강등."""
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    ocr_lines = [
        _line("수박주스(Only Ice)", 100, 100),
        _line("4,300", 600, 150),
        _line("백도스무디", 100, 200),
        _line("4,200", 600, 250),
        _line("펄추가(Only Ice)", 102, 300),  # OCR이 ▶ 글리프를 놓침 (들여쓰기도 없음)
        _line("1,000", 600, 350),
        _line("합계 금액", 100, 400),
        _line("9,500", 600, 400),
    ]
    parsed = {
        "items": [
            {"name": "수박주스(Only Ice)", "quantity": 1, "price": 4300, "sub_items": []},
            {"name": "백도스무디", "quantity": 1, "price": 4200, "sub_items": []},
            {"name": "▶ 개인텀블러지참", "quantity": 1, "price": 0, "sub_items": []},
            {"name": "▶ 펄추가(Only Ice)", "quantity": 1, "price": 1000, "sub_items": []},
        ],
        "total_amount": 9500,
    }
    merged, corrections = server._merge(parsed, ocr_lines)
    names = [it["name"] for it in merged["items"]]
    assert names == ["수박주스(Only Ice)", "백도스무디"]
    assert [(s["name"], s["price"]) for s in merged["items"][1]["sub_items"]] == [
        ("개인텀블러지참", 0),
        ("펄추가(Only Ice)", 1000),
    ]
    assert merged["total_verified"] is True
    assert sum(1 for c in corrections if c["reason"] == "sub_demoted") == 2


def test_demote_skipped_when_all_items_have_markers():
    """모든 품목에 마커가 있으면 불릿 장식으로 보고 강등하지 않는다."""
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    parsed = {
        "items": [
            {"name": "▶아메리카노", "price": 4500, "sub_items": []},
            {"name": "▶카페라떼", "price": 5000, "sub_items": []},
        ],
        "total_amount": 9500,
    }
    merged, _ = server._merge(parsed, [])
    assert [it["name"] for it in merged["items"]] == ["▶아메리카노", "▶카페라떼"]


def test_merge_wrapped_menu_name_and_placeholder_scrub():
    """줄바꿈으로 잘린 세트 메뉴명 병합 + VLM 플레이스홀더 제거 (치킨집 주문서 실측).

    금액 0원인데 하위 옵션을 거느린 품목은 직전 품목명의 이어짐이다.
    """
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    parsed = {
        "store_name": "상호명",  # VLM이 못 찾으면 프롬프트 예시를 그대로 돌려줌
        "payment_method": "카드 또는 현금",
        "items": [
            {"name": "갓 튀긴 옛날통닭 + 콜라", "quantity": 1, "price": 12000, "sub_items": []},
            {
                "name": "치킨 무 [한그릇 세트]",
                "quantity": 1,
                "price": 0,
                "sub_items": [
                    {"name": "치킨 무 + 콜라 500ml", "price": 0},
                    {"name": "양념소스 추가", "price": 500},
                ],
            },
            {"name": "배달비", "quantity": 1, "price": 0, "sub_items": []},
        ],
        "total_amount": 12500,
    }
    ocr_lines = [
        _line("배민배달선결제", 100, 100),
        _line("12,500", 600, 100),
    ]
    merged, corrections = server._merge(parsed, ocr_lines)

    names = [it["name"] for it in merged["items"]]
    assert names == ["갓 튀긴 옛날통닭 + 콜라 + 치킨 무 [한그릇 세트]"]  # 배달비는 품목이 아니다
    assert merged["delivery_fee"] == 0
    assert [(s["name"], s["price"]) for s in merged["items"][0]["sub_items"]] == [
        ("치킨 무 + 콜라 500ml", 0),
        ("양념소스 추가", 500),
    ]
    assert merged["store_name"] is None
    assert merged["payment_method"] is None
    assert merged["total_verified"] is True  # 선결제 라벨과 일치
    assert any(c["reason"] == "name_wrapped" for c in corrections)


def test_merge_total_mismatch_fails_verification():
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    ocr_lines = [
        _line("아메리카노", 100, 100),
        _line("4,200", 600, 100),
        _line("합계", 100, 150),
        _line("4,500", 600, 150),
    ]
    # VLM 합계·라벨 합계·품목 합이 전부 어긋나면 검증 실패 (사용자 확인 유도)
    parsed = {"items": [{"name": "아메리카노", "price": 4200}], "total_amount": 4000}
    merged, _ = server._merge(parsed, ocr_lines)
    assert merged["total_amount"] == 4000  # 임의 값으로 바꾸지 않는다
    assert merged["total_verified"] is False


def test_weak_total_label_used_when_strong_label_missing():
    """결제금액/합계 인식 실패 시 주문금액을 보조 합계 라벨로 사용."""
    lines = [
        _line("치즈스틱", 100, 100),
        _line("3,800", 600, 100),
        _line("주문금액:", 100, 150),
        _line("3,800", 600, 150),
    ]
    assert analyze_receipt(lines).total_amount == 3800


def test_merge_adopts_label_total_when_it_matches_items_sum():
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()

    ocr_lines = [
        _line("아메리카노", 100, 100),
        _line("4,500", 600, 100),
        _line("합계", 100, 150),
        _line("4,500", 600, 150),
    ]
    # VLM만 틀리고 라벨 합계와 품목 합이 일치하면 그 값을 채택
    parsed = {"items": [{"name": "아메리카노", "price": 4500}], "total_amount": 4000}
    merged, corrections = server._merge(parsed, ocr_lines)
    assert merged["total_amount"] == 4500
    assert merged["total_verified"] is True
    assert any(c["reason"] == "ocr_layout" for c in corrections)


def test_merge_item_discount_and_summary_row_not_double_counted():
    """마트형: 품목별 할인은 item.discount 로, 합계부 요약줄은 버린다."""
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()
    ocr_lines = _HEADER + [
        _line("미니꿀약과", 100, 150),
        _line("2,500", 600, 150),
        _line("$특매할인", 130, 200),
        _line("-1,510", 600, 200),
        _line("콘칩", 100, 250),
        _line("1,000", 600, 250),
        _line("할인금액", 100, 300),
        _line("-1,510", 600, 300),
        _line("합계", 100, 350),
        _line("1,990", 600, 350),
    ]
    parsed = {
        "items": [
            {"name": "미니꿀약과", "quantity": 1, "price": 2500, "sub_items": []},
            {"name": "콘칩", "quantity": 1, "price": 1000, "sub_items": []},
        ],
        "total_amount": 1990,
    }
    merged, _ = server._merge(parsed, ocr_lines)
    assert [(i["name"], i["price"], i["discount"]) for i in merged["items"]] == [
        ("미니꿀약과", 2500, 1510),
        ("콘칩", 1000, 0),
    ]
    assert merged["discount"] == 0
    assert merged["total_verified"] is True


def test_merge_receipt_discount_fixes_total_from_subtotal():
    """배달앱형: VLM 이 소계를 합계로 착각해도 전체 할인을 알면 라벨 합계를 채택한다."""
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()
    ocr_lines = _HEADER + [
        _line("파스타", 100, 150),
        _line("12,900", 600, 150),
        _line("소계금액", 100, 200),
        _line("16,000", 600, 200),
        _line("할인금액", 100, 250),
        _line("3,000", 600, 250),
        _line("합계금액", 100, 300),
        _line("9,900", 600, 300),
    ]
    parsed = {
        "items": [{"name": "파스타", "quantity": 1, "price": 12900, "sub_items": []}],
        "total_amount": 16000,
    }
    merged, corrections = server._merge(parsed, ocr_lines)
    assert merged["discount"] == 3000
    assert merged["items"][0]["discount"] == 0
    assert merged["total_amount"] == 9900  # 12,900 - 3,000
    assert merged["total_verified"] is True
    assert "ocr_layout" in [c["reason"] for c in corrections]


def test_merge_without_discount_keeps_previous_behaviour():
    """할인이 없는 영수증은 discount 가 0이고 기존 검증이 그대로 동작한다."""
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()
    ocr_lines = _HEADER + [
        _line("신라면큰사발", 100, 150),
        _line("1,600", 600, 150),
        _line("포장봉투", 100, 200),
        _line("100", 600, 200),
        _line("합계", 100, 250),
        _line("1,700", 600, 250),
    ]
    parsed = {
        "items": [
            {"name": "신라면큰사발", "quantity": 1, "price": 1600, "sub_items": []},
            {"name": "포장봉투", "quantity": 1, "price": 100, "sub_items": []},
        ],
        "total_amount": 1700,
    }
    merged, _ = server._merge(parsed, ocr_lines)
    assert merged["discount"] == 0
    assert all(i["discount"] == 0 for i in merged["items"])
    assert merged["total_verified"] is True


def test_discount_word_tolerates_ocr_misreads():
    """실측 오독: `다품말인 20%`(신세계), `쿠폰말인`(하나로마트), `합인금액:`(홈푸드마트)."""
    from layout import _discount_kind

    assert _discount_kind("다품말인 20%", -27800) == "item"
    assert _discount_kind("쿠폰말인", -12000) == "total"
    assert _discount_kind("합인금액:", -520) == "total"
    assert _discount_kind("에누ㄹ1()2604260014760", -4000) == "item"
    # 오독 어휘가 실재 단어를 잡아먹지 않는지
    assert _discount_kind("말인당 1인분", 9000) == "item"  # `말인`이 들어가면 할인으로 본다
    assert _discount_kind("합계", 9000) is None


def test_merge_folds_unnamed_negative_subitem_into_item_discount():
    """신세계형: VLM 이 할인 줄의 이름을 잃고 `옵션 -27,800` 으로 올려보낸다.

    이름에 할인 어휘가 없어도 음수 하위 항목은 할인이다. 실측 63장에서 음수
    하위 항목은 전부 할인이었다.
    """
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()
    ocr_lines = _HEADER + [
        _line("나이키트레이닝", 100, 150),
        _line("139,000", 600, 150),
        _line("직원 에누리20%", 130, 200),
        _line("-22,240", 600, 200),
        _line("계", 100, 300),
        _line("88,960", 600, 300),
    ]
    parsed = {
        "items": [
            {
                "name": "나이키 트레이닝",
                "quantity": 1,
                "price": 139000,
                "sub_items": [{"name": "옵션", "price": -27800}, {"name": "옵션", "price": -22240}],
            }
        ],
        "total_amount": 88960,
    }
    merged, corrections = server._merge(parsed, ocr_lines)
    item = merged["items"][0]
    assert item["price"] == 139000
    assert item["discount"] == 50040
    assert item["sub_items"] == []  # 할인은 옵션이 아니다
    assert merged["discount"] == 0  # 품목에 귀속됐으므로 영수증 단위 할인은 없다
    assert any(c["reason"] == "discount_from_sub" for c in corrections)


def test_merge_summary_label_subitem_is_not_charged_to_the_item():
    """홈푸드마트형: 합계부 요약줄 `합인금액: -520` 이 엉뚱한 품목의 하위로 붙는다.

    같은 520 이 품목 할인($특매할인)·요약줄·영수증 단위에 세 번 세지면 안 된다.
    """
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()
    ocr_lines = _HEADER + [
        _line("CJ)맛밤 36g", 100, 150),
        _line("2,500", 600, 150),
        _line("$특매할인", 130, 200),
        _line("-520", 600, 200),
        _line("정성)쫑디기 95g", 100, 250),
        _line("1,000", 600, 250),
        _line("합계", 100, 300),
        _line("2,980", 600, 300),
        _line("합인금액:", 100, 350),
        _line("-520", 600, 350),
    ]
    parsed = {
        "items": [
            {"name": "CJ)맛밤 36g", "quantity": 1, "price": 2500,
             "sub_items": [{"name": "$특매할인", "price": -520}]},
            {"name": "정성)쫑디기 95g", "quantity": 1, "price": 1000,
             "sub_items": [{"name": "합인금액:", "price": -520}]},
        ],
        "total_amount": 2980,
    }
    merged, _ = server._merge(parsed, ocr_lines)
    assert [(i["name"], i["discount"]) for i in merged["items"]] == [
        ("CJ)맛밤 36g", 520),
        ("정성)쫑디기 95g", 0),  # 요약줄은 이 품목의 할인이 아니다
    ]
    assert merged["discount"] == 0  # 요약줄이 말하는 520 은 이미 품목에 있다
    assert all(i["sub_items"] == [] for i in merged["items"])


# ---------------------------------------------------------------- 배달비

def _merge_with(parsed, ocr_lines):
    import server
    from corrector import KoreanCorrector

    if server._corrector is None:
        server._corrector = KoreanCorrector()
    return server._merge(parsed, ocr_lines)


def test_delivery_fee_waived_nets_to_zero_and_beats_vlm():
    """배민 한집배달 실측: `맨드본 배달팁 1,000` 바로 아래 `-1,000`(무료배달).

    VLM 은 1,000 만 옮겨 적어 품목으로 올린다. 좌표가 본 순액 0 이 이겨야 한다.
    """
    ocr_lines = _HEADER + [
        _line("아이스티", 100, 150), _line("3,000", 600, 150),
        _line("주문금액", 100, 200), _line("3,000", 600, 200),
        _line("맨드본 배달팁", 100, 250), _line("1,000", 600, 250),
        _line("-1,000", 600, 290),
        _line("총결제금액", 100, 340), _line("3,000", 600, 340),
    ]
    parsed = {
        "items": [
            {"name": "아이스티", "quantity": 1, "price": 3000, "sub_items": []},
            {"name": "배달팁", "quantity": 1, "price": 1000, "sub_items": []},
        ],
        "total_amount": 3000,
    }
    merged, corrections = _merge_with(parsed, ocr_lines)
    assert [i["name"] for i in merged["items"]] == ["아이스티"]
    assert merged["delivery_fee"] == 0
    assert merged["discount"] == 0  # 무료배달 상계는 영수증 할인이 아니다
    assert merged["total_verified"] is True
    assert any(c["reason"] == "delivery_fee_from_item" for c in corrections)


def test_delivery_fee_parent_and_breakdown_rows_count_once():
    """`배달팁 3,000` 아래 `ㄴ기본배달팁 3,000` 내역 — 같은 돈을 두 번 적은 것이다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250), _line("3,000", 600, 250),
        _line("ㄴ기본배달팁", 100, 290), _line("3,000", 600, 290),
        _line("총결제금액", 100, 340), _line("12,000", 600, 340),
    ]
    assert analyze_receipt(lines).delivery_fee == 3000


def test_delivery_fee_unreadable_parent_falls_back_to_breakdown():
    """상위 줄 금액이 `3.000` 으로 깨져 못 읽으면 내역 줄을 쓴다. 깨진 줄 때문에 죽지도 않는다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250), _line("3.000", 600, 250),
        _line("ㄴ기본배달팁", 100, 290), _line("3,000", 600, 290),
    ]
    assert analyze_receipt(lines).delivery_fee == 3000


def test_delivery_fee_with_garbled_waiver_does_not_go_negative():
    """실측(1790229097208): `기본배달팁 4.100` / `-4.100` 둘 다 구분자가 깨졌다.

    배달비 줄을 못 읽었으면 "봤다"고 치지 않는다(None). 음수 상계만 남아 음수 배달비가
    되면 안 되고, 0 으로 단정해서도 안 된다 — 0 이면 VLM 이 제대로 읽은 배달비를 버린다.
    이 영수증은 VLM 도 배달비를 올리지 않아 최종값은 0 이다.
    """
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("순살해장국", 100, 150), _line("12,500", 600, 150),
        _line("주문금액", 100, 200), _line("12,500", 600, 200),
        _line("기본배달팁", 100, 250), _line("4.100", 600, 250),
        _line("-4,100", 600, 290),
    ]
    assert analyze_receipt(lines).delivery_fee is None
    parsed = {"items": [{"name": "순살해장국", "quantity": 1, "price": 12500, "sub_items": []}],
              "total_amount": 12500}
    merged, _ = _merge_with(parsed, lines)
    assert merged["delivery_fee"] == 0


def test_delivery_fee_discount_row_is_netted_not_counted_as_discount():
    """`배달팁 할인 1,000` 이 부호 없이 찍혀도 배달비에서 빼고, 영수증 할인에는 넣지 않는다."""
    ocr_lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250), _line("3,000", 600, 250),
        _line("배달팁 할인", 100, 300), _line("1,000", 600, 300),
        _line("총결제금액", 100, 350), _line("11,000", 600, 350),
    ]
    parsed = {
        "items": [{"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []}],
        "total_amount": 11000,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert merged["delivery_fee"] == 2000
    assert merged["discount"] == 0
    assert merged["total_verified"] is True


def test_delivery_fee_from_vlm_sub_item_when_layout_saw_none():
    """좌표가 배달비 줄을 못 봤으면(오독 등) VLM 이 옵션으로 붙인 배달비를 쓴다."""
    ocr_lines = [_line("배민배달선결제", 100, 100), _line("14,000", 600, 100)]
    parsed = {
        "items": [
            {
                "name": "치킨 반마리",
                "quantity": 1,
                "price": 12000,
                "sub_items": [{"name": "배달비", "price": 2000}],
            }
        ],
        "total_amount": 14000,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert merged["items"][0]["sub_items"] == []
    assert merged["delivery_fee"] == 2000
    assert merged["total_verified"] is True


def test_delivery_words_that_are_not_fees():
    """`배달주소`·`배달메모`·`한집배달`·`배민배달선결제 32,400` 은 배달비가 아니다."""
    from layout import analyze_receipt

    lines = [
        _line("한집배달 주문전표", 100, 50),
        _line("배달주소: 대구 달성군", 100, 80),
        _line("배달메모: 문 앞에 두세요", 100, 110),
    ] + _HEADER[:0] + [
        _line("상품명", 100, 140), _line("수량", 400, 142), _line("금액", 600, 138),
        _line("짬뽕", 100, 190), _line("9,000", 600, 190),
        _line("배민배달선결제", 100, 240), _line("32,400", 600, 240),
    ]
    layout = analyze_receipt(lines)
    assert layout.delivery_fee is None
    assert [i.name for i in layout.items] == ["짬뽕"]


def test_delivery_fee_row_inside_item_region_is_not_recovered_as_item():
    """품목 영역 안의 `배달팁 3,000` 을 좌표 복구가 품목으로 되살리면 합계에서 두 번 더해진다."""
    ocr_lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("배달팁", 100, 200), _line("3,000", 600, 200),
        _line("합계", 100, 250), _line("12,000", 600, 250),
    ]
    parsed = {
        "items": [{"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []}],
        "total_amount": 12000,
    }
    merged, corrections = _merge_with(parsed, ocr_lines)
    assert [i["name"] for i in merged["items"]] == ["짬뽕"]
    assert not any(c["reason"] == "ocr_recovered" for c in corrections)
    assert merged["delivery_fee"] == 3000
    assert merged["total_verified"] is True


def test_delivery_fee_ignores_following_nameless_positive_row():
    """배달비 줄 다음에 이름 없는 양수 줄(줄바꿈된 합계 값)이 오면 더하지 않는다.

    실측 전표에서 `총결제금액` 라벨과 값이 다른 줄로 갈라지는 일이 흔하다
    (20260924_144101 의 `15,200` 단독 줄). 더하면 배달비가 합계만큼 부풀어 오른다.
    """
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250), _line("3,000", 600, 250),
        _line("12,000", 600, 290),
        _line("총결제금액", 100, 330),
    ]
    assert analyze_receipt(lines).delivery_fee == 3000


# ---- 독립 리뷰에서 재현된 결함들 (각 테스트는 수정 전 코드에서 실패했다)

def test_delivery_waiver_sent_as_vlm_subs_is_not_charged_as_item_discount():
    """리뷰 #1a: VLM 이 `ㄴ기본배달팁 4,100` 과 이름 없는 `-4,100` 을 마지막 메뉴 옵션으로 붙인다.

    무료배달 상계가 할인 접기에서 품목 할인이 되면 4,100 이 한 번 더 빠진다.
    VLM 합계를 일부러 틀리게 줘서, 검증이 라벨 일치가 아니라 항등식 경로를 타게 한다.
    """
    ocr_lines = _HEADER + [
        _line("순살해장국", 100, 150), _line("12,500", 600, 150),
        _line("주문금액", 100, 200), _line("12,500", 600, 200),
        _line("배달팁", 100, 240), _line("4,100", 600, 240),
        _line("-4,100", 600, 280),
        _line("총결제금액", 100, 330), _line("12,500", 600, 330),
    ]
    parsed = {
        "items": [{
            "name": "순살해장국", "quantity": 1, "price": 12500,
            "sub_items": [{"name": "ㄴ기본배달팁", "price": 4100}, {"name": None, "price": -4100}],
        }],
        "total_amount": 16600,  # VLM 이 배달팁을 더해 틀림
    }
    merged, corrections = _merge_with(parsed, ocr_lines)
    item = merged["items"][0]
    assert (item["discount"], item["sub_items"]) == (0, [])
    assert merged["delivery_fee"] == 0
    assert merged["total_amount"] == 12500
    assert merged["total_verified"] is True
    assert any(c["reason"] == "ocr_layout" for c in corrections)  # 항등식 경로로 검증됐다


def test_delivery_discount_sent_as_vlm_sub_goes_to_fee_not_item():
    """리뷰 #1b: `배달팁 할인 -1,000` 이 옵션으로 오면 배달비에서 빼고, 품목 할인이 아니다."""
    ocr_lines = [_line("배민배달선결제", 100, 100), _line("14,500", 600, 100)]
    parsed = {
        "items": [{
            "name": "순살해장국", "quantity": 1, "price": 12500,
            "sub_items": [{"name": "배달팁", "price": 3000}, {"name": "배달팁 할인", "price": -1000}],
        }],
        "total_amount": 14500,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert merged["items"][0]["discount"] == 0
    assert merged["delivery_fee"] == 2000
    assert server_identity(merged) == 14500


def server_identity(merged):
    import server
    return server._items_sum(merged["items"]) - merged["discount"] + merged["delivery_fee"]


def test_delivery_label_and_value_on_split_rows():
    """리뷰 #2a: 라벨과 값이 y 로 갈라져 다른 행이 된다. 바로 아래 이름 없는 값을 쓴다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250),
        _line("3,000", 600, 285),
        _line("총결제금액", 100, 330), _line("12,000", 600, 330),
    ]
    assert analyze_receipt(lines).delivery_fee == 3000


def test_unreadable_fee_row_falls_back_to_vlm_value():
    """리뷰 #2b: `배달팁 3.000` 을 못 읽었다고 0 으로 단정하면 VLM 의 3,000 을 버리게 된다."""
    ocr_lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250), _line("3.000", 600, 250),
        _line("총결제금액", 100, 330), _line("12,000", 600, 330),
    ]
    parsed = {
        "items": [
            {"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []},
            {"name": "배달팁", "quantity": 1, "price": 3000, "sub_items": []},
        ],
        "total_amount": 12000,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert merged["delivery_fee"] == 3000
    assert server_identity(merged) == 12000


def test_delivery_notice_is_not_a_fee_row():
    """리뷰 #2c: `배달 비대면` 은 공백을 지우면 `배달비대면` — 배달비가 아니다.

    VLM 이 요청사항을 금액 0 옵션으로 올려도 옵션으로 남아야 하고, 좌표에서 금액이 붙은
    안내 줄(`배달 비대면 할인 1,000원` 같은 이벤트 문구)도 배달비로 읽으면 안 된다.
    (2차 리뷰: 금액 없는 줄만 넣으면 이름 판정 없이도 통과해 아무것도 검증하지 않았다.)
    """
    from layout import analyze_receipt, delivery_fee_label

    assert delivery_fee_label("배달 비대면 요청") is None
    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("배달 비대면 요청 시", 100, 200), _line("1,000", 600, 200),
    ]
    assert analyze_receipt(lines).delivery_fee is None
    parsed = {
        "items": [{"name": "짬뽕", "quantity": 1, "price": 9000,
                   "sub_items": [{"name": "배달 비대면 요청", "price": 0}]}],
        "total_amount": 9000,
    }
    merged, _ = _merge_with(parsed, [])
    assert [s["name"] for s in merged["items"][0]["sub_items"]] == ["배달 비대면 요청"]


def test_breakdown_row_without_leading_marker_is_not_a_parent():
    """리뷰 #3: OCR 이 `ㄴ` 을 따로 떼거나 `L`·`A` 로 읽거나 떨어뜨려도 내역 줄은 내역 줄이다."""
    from layout import analyze_receipt

    for child in (["ㄴ", "기본배달팁"], ["L기본배달팁"], ["A", "기본배달팁"], ["기본배달팁"]):
        lines = _HEADER + [
            _line("짬뽕", 100, 150), _line("9,000", 600, 150),
            _line("주문금액", 100, 200), _line("9,000", 600, 200),
            _line("배달팁", 100, 250), _line("3,000", 600, 250),
        ]
        x = 100
        for text in child:
            lines.append(_line(text, x, 290))
            x += 20 * len(text) + 10
        lines.append(_line("3,000", 600, 290))
        assert analyze_receipt(lines).delivery_fee == 3000, child


def test_shipping_words_inside_product_names_are_not_fees():
    """리뷰 #4: 온라인 쇼핑 영수증의 상품명 안 `배송비` 는 배달비가 아니다."""
    ocr_lines = [_line("결제금액", 100, 100), _line("44,900", 600, 100)]
    parsed = {
        "items": [
            {"name": "[배송비무료] 제주감귤 5kg", "quantity": 1, "price": 19900, "sub_items": []},
            {"name": "사과 1박스(배송비포함)", "quantity": 1, "price": 25000, "sub_items": []},
        ],
        "total_amount": 44900,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert [i["name"] for i in merged["items"]] == ["[배송비무료] 제주감귤 5kg", "사과 1박스(배송비포함)"]
    assert merged["delivery_fee"] == 0
    assert server_identity(merged) == 44900


def test_missed_item_with_same_price_as_fee_is_still_recovered():
    """리뷰 #5: VLM 이 `군만두 3,000` 을 놓치고 `배달팁 3,000` 을 품목으로 올렸다.

    배달비를 늦게 떼면 좌표 복구가 군만두를 금액이 같은 배달팁 품목과 짝지어 되살리지 않는다.
    """
    ocr_lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("군만두", 100, 200), _line("3,000", 600, 200),
        _line("주문금액", 100, 250), _line("12,000", 600, 250),
        _line("배달팁", 100, 300), _line("3,000", 600, 300),
        _line("총결제금액", 100, 350), _line("15,000", 600, 350),
    ]
    parsed = {
        "items": [
            {"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []},
            {"name": "배달팁", "quantity": 1, "price": 3000, "sub_items": []},
        ],
        "total_amount": 15000,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert [i["name"] for i in merged["items"]] == ["짬뽕", "군만두"]
    assert merged["delivery_fee"] == 3000
    assert server_identity(merged) == 15000


def test_string_price_on_fee_item_does_not_crash():
    """리뷰 #6: VLM 이 금액을 `"3,000"` 문자열로 주면 500 으로 죽었다."""
    parsed = {
        "items": [
            {"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []},
            {"name": "배달비", "quantity": 1, "price": "3,000", "sub_items": []},
        ],
        "total_amount": 12000,
    }
    merged, _ = _merge_with(parsed, [])
    assert merged["delivery_fee"] == 3000


def test_free_delivery_rows_without_fee_word_are_netted():
    """리뷰 #7: `무료배달 -3,000`, `와우 무료배달 -3,000`, `무료배달 할인 -3,000` 은 배달비 상계다.

    영수증 할인으로 세지도 않는다(순액 배달비라는 계약).
    """
    for label in ("무료배달", "와우 무료배달", "무료배달 할인"):
        ocr_lines = _HEADER + [
            _line("짬뽕", 100, 150), _line("9,000", 600, 150),
            _line("주문금액", 100, 200), _line("9,000", 600, 200),
            _line("배달비", 100, 250), _line("3,000", 600, 250),
            _line(label, 100, 290), _line("-3,000", 600, 290),
            _line("총결제금액", 100, 340), _line("9,000", 600, 340),
        ]
        parsed = {"items": [{"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []}],
                  "total_amount": 9000}
        merged, _ = _merge_with(parsed, ocr_lines)
        assert (merged["delivery_fee"], merged["discount"]) == (0, 0), label
        assert server_identity(merged) == 9000, label


def test_merged_fee_and_option_row_is_not_trusted():
    """리뷰 #8 (실측 20260924_144401 모양): `03.배달비 --케이준양념감자(중) 3,000 2,000`.

    두 줄이 한 행으로 합쳐지면 최우측 금액이 배달비라는 보장이 없다. 좌표는 포기하고 VLM 값을 쓴다.
    """
    ocr_lines = _HEADER + [
        _line("후라이드", 100, 150), _line("15,000", 600, 150),
        _line("03.배달비", 100, 250), _line("3,000", 600, 250),
        _line("--케이준양념감자(중)", 300, 262), _line("2,000", 650, 262),
        _line("총결제금액", 100, 340), _line("20,000", 600, 340),
    ]
    parsed = {
        "items": [
            {"name": "후라이드", "quantity": 1, "price": 15000,
             "sub_items": [{"name": "케이준양념감자(중)", "price": 2000}]},
            {"name": "배달비", "quantity": 1, "price": 3000, "sub_items": []},
        ],
        "total_amount": 20000,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert merged["delivery_fee"] == 3000
    assert server_identity(merged) == 20000


def test_wrapped_name_merge_does_not_swallow_item_into_fee():
    """리뷰 #9: 금액 0 + 옵션 품목이 이름 병합으로 배달팁에 붙었다가 배달팁과 같이 버려졌다."""
    parsed = {
        "items": [
            {"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []},
            {"name": "배달팁", "quantity": 1, "price": 3000, "sub_items": []},
            {"name": "음료 서비스", "quantity": 1, "price": 0,
             "sub_items": [{"name": "콜라 1.25L", "price": 2000}]},
        ],
        "total_amount": 14000,
    }
    merged, _ = _merge_with(parsed, [])
    assert merged["delivery_fee"] == 3000
    assert server_identity(merged) == 14000


def test_merged_row_ending_in_fee_label_with_two_amounts_is_not_trusted():
    """합쳐진 행의 반대 순서: 옵션이 왼쪽, 배달비가 오른쪽이면 이름이 `...배달비` 로 끝난다.

    이름 끝 판정은 통과하지만 금액이 둘(2,000·3,000)이라 어느 쪽이 배달비인지 모른다.
    """
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("후라이드", 100, 150), _line("15,000", 600, 150),
        _line("--케이준양념감자(중)", 100, 250), _line("2,000", 450, 250),
        _line("03.배달비", 550, 258), _line("3,000", 750, 258),
    ]
    assert analyze_receipt(lines).delivery_fee is None


# ---- 2차 독립 리뷰에서 재현된 결함들

def test_garbled_fee_value_does_not_borrow_total_below():
    """2차 #1A: `배달팁 3.000`(깨짐) 아래의 `12,000` 은 다음 라벨 `총결제금액` 의 값이다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250), _line("3.000", 600, 250),
        _line("12,000", 600, 290),
        _line("총결제금액", 100, 330),
    ]
    assert analyze_receipt(lines).delivery_fee is None  # VLM 값으로 넘긴다


def test_fee_does_not_borrow_subtotal_above():
    """2차 #1B: 위 줄의 이름 없는 `9,000` 은 앞 요약줄(주문금액)의 값이다. 위는 보지 않는다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200),
        _line("9,000", 600, 240),
        _line("배달팁", 100, 290),
        _line("총결제금액", 100, 340), _line("12,000", 600, 340),
    ]
    assert analyze_receipt(lines).delivery_fee is None


def test_split_value_owned_by_next_label_without_digits_is_not_borrowed():
    """2차 #1: 값 행 다음이 숫자 없는 라벨이면 그 값은 다음 라벨 몫이다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("배달팁", 100, 250),
        _line("12,000", 600, 285),
        _line("총결제금액", 100, 320),
    ]
    assert analyze_receipt(lines).delivery_fee is None


def test_split_value_is_borrowed_once_even_if_next_label_is_garbled():
    """2차 #3(C): `배달팁` / `3,000` / `배달팁 할인 1.000`(깨짐). 3,000 은 배달팁 값이고,
    깨진 할인 줄이 그 값을 다시 가져가 0 이 되면 안 된다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("주문금액", 100, 200), _line("9,000", 600, 200),
        _line("배달팁", 100, 250),
        _line("3,000", 600, 285),
        _line("배달팁 할인", 100, 330), _line("1.000", 600, 330),
        _line("총결제금액", 100, 380), _line("11,000", 600, 380),
    ]
    assert analyze_receipt(lines).delivery_fee == 3000


def test_pending_item_name_does_not_swallow_fee_value_or_waiver():
    """2차 #2(D·D2, 1차 수정이 만든 회귀): 가격 없는 품목명 줄(`단무지 많이`) 뒤에
    배달팁 값 행이나 무료배달 상계 행이 오면, 그 금액이 품목으로 복구되면 안 된다."""
    for rows, fee in (
        ([_line("배달팁", 100, 250), _line("3,000", 600, 285)], 3000),
        ([_line("배달팁", 100, 250), _line("3,000", 600, 250), _line("-3,000", 600, 285)], 0),
    ):
        ocr_lines = _HEADER + [
            _line("짬뽕", 100, 150), _line("9,000", 600, 150),
            _line("단무지 많이", 100, 200),
        ] + rows + [_line("합계", 100, 330), _line(f"{9000 + fee:,}", 600, 330)]
        parsed = {"items": [{"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []}],
                  "total_amount": 9000 + fee}
        merged, corrections = _merge_with(parsed, ocr_lines)
        assert [i["name"] for i in merged["items"]] == ["짬뽕"], fee
        assert merged["delivery_fee"] == fee
        assert server_identity(merged) == 9000 + fee


def test_offset_labels_ending_in_coupon_are_netted():
    """2차 #4: `배달팁 무료쿠폰`·`배달비 쿠폰`·`배달팁 할인쿠폰`·`배송비할인쿠폰` 도 상계 줄이다."""
    # `와우 배달팁 쿠폰` 은 배달비 어휘로 시작하지 않아 끝 판정(쿠폰 접미)만으로 잡혀야 한다.
    for label in ("배달팁 무료쿠폰", "배달비 쿠폰", "배달팁 할인쿠폰", "배송비할인쿠폰", "와우 배달팁 쿠폰"):
        ocr_lines = _HEADER + [
            _line("짬뽕", 100, 150), _line("9,000", 600, 150),
            _line("주문금액", 100, 200), _line("9,000", 600, 200),
            _line("배달팁", 100, 250), _line("3,000", 600, 250),
            _line(label, 100, 290), _line("-3,000", 600, 290),
            _line("총결제금액", 100, 340), _line("9,000", 600, 340),
        ]
        parsed = {"items": [{"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []}],
                  "total_amount": 9000}
        merged, _ = _merge_with(parsed, ocr_lines)
        assert (merged["delivery_fee"], merged["discount"]) == (0, 0), label
        assert server_identity(merged) == 9000, label


def test_vlm_fee_names_with_extra_text_are_recognised():
    """2차 #5: VLM 은 `배달팁 3,000원`, `배달팁(거리할증 1,000원 포함)`, `배달팁 기본` 처럼 보낸다."""
    from layout import delivery_fee_label

    for name in ("배달팁 3,000원", "배달팁(거리할증 1,000원 포함)", "배달팁 기본", "배달팁 - 기본",
                 "배달비 합계", "배달요금", "배달비:", "배달 팁", "배달팁(기본)", "배달팁 할인"):
        assert delivery_fee_label(name) is not None, name
    for name in ("[배송비무료] 제주감귤 5kg", "사과 1박스(배송비포함)", "배달 비대면 요청",
                 "배달의민족 쿠폰", "한집배달 주문전표", "배민배달선결제", "배달주소: 대구"):
        assert delivery_fee_label(name) is None, name
    # 상위·내역·감액
    assert delivery_fee_label("배달팁") == "parent"
    assert delivery_fee_label("03.배달비") == "parent"
    assert delivery_fee_label("배달팁 기본") == "child"
    assert delivery_fee_label("ㄴ기본배달팁") == "child"
    assert delivery_fee_label("와우 무료배달") == "off"

    ocr_lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("배달팁", 100, 250), _line("3,000", 600, 250),
        _line("합계", 100, 300), _line("12,000", 600, 300),
    ]
    parsed = {
        "items": [
            {"name": "짬뽕", "quantity": 1, "price": 9000, "sub_items": []},
            {"name": "배달팁 3,000원", "quantity": 1, "price": 3000, "sub_items": []},
        ],
        "total_amount": 12000,
    }
    merged, _ = _merge_with(parsed, ocr_lines)
    assert [i["name"] for i in merged["items"]] == ["짬뽕"]
    assert server_identity(merged) == 12000


def test_waiver_at_other_nesting_level_is_matched_by_amount():
    """2차 #6(H): 배달비는 옵션, 상계 `-4,100` 은 이름 없는 최상위 품목으로 왔다."""
    parsed = {
        "items": [
            {"name": "짬뽕", "quantity": 1, "price": 9000,
             "sub_items": [{"name": "기본배달팁", "price": 4100}]},
            {"name": None, "quantity": 1, "price": -4100, "sub_items": []},
        ],
        "total_amount": 9000,
    }
    merged, _ = _merge_with(parsed, [])
    assert [i["name"] for i in merged["items"]] == ["짬뽕"]
    assert merged["delivery_fee"] == 0
    assert server_identity(merged) == 9000


def test_vlm_fee_reported_twice_counts_once():
    """2차 #7(I): VLM 이 `배달팁 3,000` 을 품목과 옵션 양쪽에 올렸다(좌표는 못 읽음)."""
    parsed = {
        "items": [
            {"name": "짬뽕", "quantity": 1, "price": 9000,
             "sub_items": [{"name": "배달팁", "price": 3000}]},
            {"name": "배달팁", "quantity": 1, "price": 3000, "sub_items": []},
        ],
        "total_amount": 12000,
    }
    merged, _ = _merge_with(parsed, [])
    assert merged["delivery_fee"] == 3000
    assert server_identity(merged) == 12000


def test_same_parent_fee_printed_twice_counts_once():
    """2차 #8(L): 같은 `배달팁 3,000` 이 두 군데 찍혔다."""
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("배달팁", 100, 200), _line("3,000", 600, 200),
        _line("주문금액", 100, 250), _line("9,000", 600, 250),
        _line("배달팁", 100, 300), _line("3,000", 600, 300),
    ]
    assert analyze_receipt(lines).delivery_fee == 3000


def test_garbled_fee_value_does_not_borrow_nameless_row_below():
    """2차 #1: 라벨 줄에 숫자가 있는데 깨졌으면(`3.000`) 아래 값은 남의 것이라 빌리지 않는다.

    다음 줄이 라벨이 아니어도(영수증 끝) 마찬가지다.
    """
    from layout import analyze_receipt

    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("9,000", 600, 150),
        _line("배달팁", 100, 250), _line("3.000", 600, 250),
        _line("12,000", 600, 290),
    ]
    assert analyze_receipt(lines).delivery_fee is None


def test_partial_waiver_in_same_list_is_netted_not_item_discount():
    """VLM 옵션 [배달팁 3,000, (이름 없음) -1,000]: 부분 상계라 금액으로는 짝이 안 맞는다.

    바로 뒤따르는 이름 없는 음수라서 배달비 상계로 떼야 한다. 안 떼면 할인 접기가
    품목 할인으로 만들어 -1,000 이 품목에서 빠진다.
    """
    parsed = {
        "items": [{"name": "짬뽕", "quantity": 1, "price": 9000,
                   "sub_items": [{"name": "배달팁", "price": 3000}, {"name": None, "price": -1000}]}],
        "total_amount": 11000,
    }
    merged, _ = _merge_with(parsed, [])
    assert merged["items"][0]["discount"] == 0
    assert merged["delivery_fee"] == 2000
    assert server_identity(merged) == 11000
