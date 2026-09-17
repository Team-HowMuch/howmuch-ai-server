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


def test_discount_line_attaches_as_sub():
    lines = [
        _line("케이크", 100, 100),
        _line("1", 400, 100),
        _line("15,000", 600, 100),
        _line("행사할인", 100, 150),
        _line("-1,000", 600, 150),
    ]
    layout = analyze_receipt(lines)
    assert len(layout.items) == 1
    assert [(s.name, s.price) for s in layout.items[0].sub_items] == [("행사할인", -1000)]


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
