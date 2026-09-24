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
        _line("9,500", 600, 300),  # 품목명 줄 없는 가격 줄은 버린다
    ]
    layout = analyze_receipt(lines)
    assert [(i.name, i.price) for i in layout.items] == [("꼬마김밥A", 3500)]


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
    assert set(by_name) == {"치즈스틱", "데리세트", "토네이도쿠키", "통다리매운셋", "배달팁"}
    assert [s["name"] for s in by_name["데리세트"]["sub_items"]] == ["선택양념치즈"]
    assert [(s["name"], s["price"]) for s in by_name["통다리매운셋"]["sub_items"]] == [
        ("L포테이토", 500)
    ]
    assert by_name["토네이도쿠키"]["sub_items"] == []
    # 3800+7700+600+4200+10300+500+2700 = 29,800 == 결제금액 라벨 (받을금액 0은 무시)
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
    assert names == ["갓 튀긴 옛날통닭 + 콜라 + 치킨 무 [한그릇 세트]", "배달비"]
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
