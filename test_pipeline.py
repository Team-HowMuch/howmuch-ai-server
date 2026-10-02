"""프롬프트 버전·전처리 선택과, 프롬프트 v2 가 추가한 필드의 병합 회귀 테스트.

실행: .venv/bin/python -m pytest test_pipeline.py -v
"""
from test_layout import _HEADER, _line, _merge_with, server_identity


# ---------------------------------------------------------------- 요청 쿼리 해석

def test_pipeline_defaults_keep_previous_behaviour():
    import server

    assert server._resolve_pipeline(None, None) == ("v1", ())


def test_pipeline_query_overrides_and_normalises():
    import server

    assert server._resolve_pipeline("V2 ", "hires, crop") == ("v2", ("crop", "hires"))


def test_pipeline_empty_preprocess_turns_off_server_default():
    """서버 기본값으로 전처리를 켜 둬도, 요청이 빈 문자열을 주면 끈 결과와 비교할 수 있어야 한다."""
    import server

    saved = server.DEFAULT_PREPROCESS
    server.DEFAULT_PREPROCESS = "crop"
    try:
        assert server._resolve_pipeline(None, None)[1] == ("crop",)
        assert server._resolve_pipeline(None, "")[1] == ()
    finally:
        server.DEFAULT_PREPROCESS = saved


def test_pipeline_rejects_unknown_values():
    import server

    for prompt, preprocess in (("v9", None), (None, "sharpen")):
        try:
            server._resolve_pipeline(prompt, preprocess)
        except ValueError:
            continue
        raise AssertionError(f"받아들이면 안 된다: {prompt!r}, {preprocess!r}")


def test_both_prompts_ask_for_the_same_keys_we_merge():
    """v2 만 delivery_fee 를 묻는다. 형식 예시의 플레이스홀더는 스크럽 대상과 같아야 한다."""
    import server

    assert '"delivery_fee"' not in server.PROMPT_V1
    assert '"delivery_fee"' in server.PROMPT_V2
    for prompt in server.PROMPTS.values():
        for placeholders in server._PLACEHOLDERS.values():
            assert placeholders[0] in prompt
    # v2 가 새로 보여준 날짜만 형식도 지운다
    assert "YYYY-MM-DD" in server._PLACEHOLDERS["purchased_at"]


# ---------------------------------------------------------------- v2 배달비 필드

def _baemin_lines(fee_rows=()):
    """배민 주문전표 모양. fee_rows 를 비우면 배달팁 칸이 번져 OCR 이 못 읽은 경우다."""
    return _HEADER + [
        _line("딸기버블티", 100, 150), _line("5,000", 600, 150),
        _line("주문금액", 100, 200), _line("5,000", 600, 200),
        *fee_rows,
        _line("총결제금액", 100, 400), _line("5,500", 600, 400),
    ]


def test_vlm_fee_field_used_when_ocr_could_not_read_fee_row():
    """실측(1790229082109): 배달팁 칸이 번져 OCR 은 못 읽었지만 v2 의 VLM 은 500 을 적는다."""
    parsed = {
        "items": [{"name": "딸기버블티", "quantity": 1, "price": 5000, "sub_items": []}],
        "delivery_fee": 500,
        "total_amount": 5500,
    }
    merged, _ = _merge_with(parsed, _baemin_lines())
    assert merged["delivery_fee"] == 500
    assert server_identity(merged) == 5500


def test_ocr_fee_row_beats_vlm_fee_field():
    """좌표가 배달비 줄을 읽었으면 그 값이 이긴다(무료배달 상계까지 본다)."""
    parsed = {
        "items": [{"name": "딸기버블티", "quantity": 1, "price": 5000, "sub_items": []}],
        "delivery_fee": 3000,  # VLM 이 상계 전 금액을 적음
        "total_amount": 5000,
    }
    lines = _baemin_lines([
        _line("배달팁", 100, 250), _line("3,000", 600, 250),
        _line("-3,000", 600, 290),
    ])
    lines[-1] = _line("5,000", 600, 400)
    merged, _ = _merge_with(parsed, lines)
    assert merged["delivery_fee"] == 0


def test_vlm_fee_item_used_when_field_is_zero():
    """v2 라도 모델이 배달비를 품목으로 올리고 필드는 0 으로 둘 수 있다."""
    parsed = {
        "items": [
            {"name": "딸기버블티", "quantity": 1, "price": 5000, "sub_items": []},
            {"name": "배달팁", "quantity": 1, "price": 500, "sub_items": []},
        ],
        "delivery_fee": 0,
        "total_amount": 5500,
    }
    merged, _ = _merge_with(parsed, _baemin_lines())
    assert [i["name"] for i in merged["items"]] == ["딸기버블티"]
    assert merged["delivery_fee"] == 500


def test_vlm_fee_field_as_string_and_garbage():
    for raw, expected in (("500", 500), ("500원", 500), ("배달팁", 0), (None, 0), (-300, 0)):
        parsed = {
            "items": [{"name": "딸기버블티", "quantity": 1, "price": 5000, "sub_items": []}],
            "delivery_fee": raw,
            "total_amount": 5500,
        }
        merged, _ = _merge_with(parsed, _baemin_lines())
        assert merged["delivery_fee"] == expected, raw
        assert isinstance(merged["delivery_fee"], int)


# ---------------------------------------------------------------- 지어낸 00:00

def test_invented_midnight_is_dropped():
    parsed = {"purchased_at": "2026-09-03 00:00", "items": [], "total_amount": None}
    merged, corrections = _merge_with(parsed, [_line("2026-09-03", 100, 100)])
    assert merged["purchased_at"] == "2026-09-03"
    assert any(c["reason"] == "invented_time" for c in corrections)


def test_real_midnight_on_receipt_is_kept():
    parsed = {"purchased_at": "2026-09-03 00:00", "items": [], "total_amount": None}
    merged, corrections = _merge_with(parsed, [_line("2026-09-03 00:00:12", 100, 100)])
    assert merged["purchased_at"] == "2026-09-03 00:00"
    assert not any(c["reason"] == "invented_time" for c in corrections)


def test_other_times_untouched():
    for value in ("2026-09-03 21:05", "2026-09-03", None):
        parsed = {"purchased_at": value, "items": [], "total_amount": None}
        merged, _ = _merge_with(parsed, [])
        assert merged["purchased_at"] == value


# ---------------------------------------------------------------- 독립 리뷰 반영

def test_vlm_fee_field_does_not_override_waived_fee_items():
    """리뷰 P2: VLM 이 `배달팁 3,000`·`-3,000`(무료배달)을 옵션으로 올리고 칸에는 3,000 을 적었다.

    품목에서 뗀 배달비가 상계까지 담고 있으므로 칸이 덮어쓰면 안 된다.
    """
    lines = _HEADER + [
        _line("짬뽕", 100, 150), _line("5,000", 600, 150),
        _line("주문금액", 100, 200), _line("5,000", 600, 200),
        _line("총결제금액", 100, 400), _line("5,000", 600, 400),
    ]
    parsed = {
        "items": [{"name": "짬뽕", "quantity": 1, "price": 5000,
                   "sub_items": [{"name": "배달팁", "price": 3000}, {"name": None, "price": -3000}]}],
        "delivery_fee": 3000,
        "total_amount": 5000,
    }
    merged, _ = _merge_with(parsed, lines)
    assert merged["delivery_fee"] == 0
    assert server_identity(merged) == 5000


def test_vlm_fee_field_non_finite_does_not_crash():
    """리뷰 P3: json.loads 는 NaN·Infinity·1e400 을 float 로 준다. 500 이 되면 안 된다."""
    for raw in (float("nan"), float("inf"), float("-inf"), 1e400):
        parsed = {
            "items": [{"name": "딸기버블티", "quantity": 1, "price": 5000, "sub_items": []}],
            "delivery_fee": raw,
            "total_amount": 5500,
        }
        merged, _ = _merge_with(parsed, _baemin_lines())
        assert merged["delivery_fee"] == 0, raw


def test_midnight_detection_is_not_a_substring_match():
    """리뷰 P3: `10:00:00` 안의 `00:00` 은 자정이 아니고, `오전 12:00`·`00 : 00` 은 자정이다."""
    cases = [
        ("주문시각 10:00:00", "2026-09-03"),        # 다른 시각의 일부 → 지어낸 00:00 은 지운다
        ("2026-09-03 00:00:12", "2026-09-03 00:00"),  # 진짜 자정
        ("오전 12:00", "2026-09-03 00:00"),
        ("AM 12:00", "2026-09-03 00:00"),
        ("결제 00 : 00", "2026-09-03 00:00"),
        ("100:00", "2026-09-03"),
    ]
    for text, expected in cases:
        parsed = {"purchased_at": "2026-09-03 00:00", "items": [], "total_amount": None}
        merged, _ = _merge_with(parsed, [_line(text, 100, 100)])
        assert merged["purchased_at"] == expected, text


def test_invented_midnight_variants_from_vlm_are_dropped():
    for value in ("2026-09-03T00:00", "2026-09-03 00:00:00", "2026-09-03 0:00"):
        parsed = {"purchased_at": value, "items": [], "total_amount": None}
        merged, _ = _merge_with(parsed, [])
        assert merged["purchased_at"] == "2026-09-03", value


def test_date_only_placeholder_from_prompt_v2_is_scrubbed():
    parsed = {"purchased_at": "YYYY-MM-DD", "items": [], "total_amount": None}
    merged, _ = _merge_with(parsed, [])
    assert merged["purchased_at"] is None


# ---------------------------------------------------------------- 영수증에 찍힌 날짜로 VLM 날짜 바로잡기

_TODAY = __import__("datetime").date(2026, 10, 1)  # 시계에 기대지 않게 고정한다


def _fix_date(vlm_value, *texts, conf=0.94):
    import server

    parsed = {"purchased_at": vlm_value}
    corrections = []
    lines = [{"text": t, "confidence": conf, "box": [0, i * 40, 100, i * 40 + 30]} for i, t in enumerate(texts)]
    server._fix_date_from_ocr(parsed, lines, corrections, today=_TODAY)
    return parsed["purchased_at"], [c["reason"] for c in corrections]


def test_date_the_vlm_made_up_is_replaced_by_the_printed_one():
    """홈푸드마트 실측: `26-07-01` 을 VLM 이 `2023-07-26` 으로 읽었다. 수요일이 맞는 날짜는 2026-07-01 이다."""
    assert _fix_date("2023-07-26 18:47", "판매입:26-07-0118:47,수요일 계산대:002") == (
        "2026-07-01 18:47", ["date_from_ocr"])


def test_date_the_vlm_read_correctly_is_left_alone():
    assert _fix_date("2026-07-01 18:47", "판매일:26-07-01 18:47,수요일") == ("2026-07-01 18:47", [])


def test_date_ocr_misreading_is_rejected_by_the_printed_weekday():
    """실측: 인쇄는 `26-08-19 수요일` 인데 OCR 이 `23-08-19` 로 읽었다. 2023-08-19 는 토요일이다."""
    assert _fix_date("2026-08-19 19:07", "판매일:23-08-19 19:07,수요일 계산대:004") == ("2026-08-19 19:07", [])


def test_date_with_four_digit_year_and_bracketed_weekday():
    """홈플러스: `2026/05/3115:36:00[일]`. 2026-05-31 은 일요일이다."""
    assert _fix_date("2023-05-31 15:36", "2026/05/3115:36:00[일]") == ("2026-05-31 15:36", ["date_from_ocr"])
    # 요일이 안 맞으면 그 줄은 믿지 않는다
    assert _fix_date("2023-05-31 15:36", "2026/05/3115:36:00[월]") == ("2023-05-31 15:36", [])


def test_date_only_the_year_differs_keeps_the_later_year():
    """OCR 이 6 을 3 으로 읽었을 때(요일이 안 찍힌 영수증)에도 더 최근 연도를 지킨다."""
    assert _fix_date("2026-08-19 19:07", "판매일:23-08-19 19:07") == ("2026-08-19 19:07", [])
    assert _fix_date("2023-08-19 19:07", "판매일:26-08-19 19:07") == ("2026-08-19 19:07", ["date_from_ocr"])


def test_date_two_different_printed_dates_are_ambiguous_and_left_alone():
    assert _fix_date("2023-01-01 10:00", "승인일시: 2026-07-01 18:47", "판매일: 2026-06-28") == (
        "2023-01-01 10:00", [])


def test_date_labeled_line_beats_unlabeled_numbers():
    value, reasons = _fix_date("2023-03-15", "주문번호 2026-07-02", "주문일시: 2026-07-01 18:47")
    assert (value, reasons) == ("2026-07-01 18:47", ["date_from_ocr"])


def test_date_phone_numbers_and_business_numbers_are_not_dates():
    texts = ["248-88-01934 박혜선 053-286-4777", "711-22-01853 010-4667-1565", "TEL 02-123-45"]
    assert _fix_date("2026-07-01 18:47", *texts) == ("2026-07-01 18:47", [])


def test_date_far_future_or_past_is_not_a_date():
    assert _fix_date("2026-07-01", "99-12-31", "1999/01/01") == ("2026-07-01", [])


def test_date_low_confidence_line_is_ignored():
    assert _fix_date("2023-07-26", "판매일:26-07-01", conf=0.5) == ("2023-07-26", [])


def test_date_vlm_time_is_kept_only_if_printed():
    value, _ = _fix_date("2023-07-26 18:47", "판매일:26-07-01", "계산 18:47:12")
    assert value == "2026-07-01 18:47"
    value, _ = _fix_date("2023-07-26 18:47", "판매일:26-07-01")
    assert value == "2026-07-01"


def test_date_missing_or_malformed_vlm_value_is_left_alone():
    for vlm in (None, "", "어제", "26-07-01"):
        assert _fix_date(vlm, "판매일:26-07-01 18:47")[0] == vlm


def test_date_window_rejects_a_single_far_future_or_far_past_date():
    assert _fix_date("2026-07-01 10:00", "판매일:99-12-31") == ("2026-07-01 10:00", [])
    assert _fix_date("2026-07-01 10:00", "판매일:1999/01/01") == ("2026-07-01 10:00", [])
    assert _fix_date("2026-07-01 10:00", "판매일:2030-01-01") == ("2026-07-01 10:00", [])


def test_line_numbers_are_recognised_only_when_they_run_in_order():
    from types import SimpleNamespace as N

    import server

    def has(*names):
        return server._has_line_numbers([N(name=n) for n in names])

    assert has("001 몬스터", "002 데날개", "003데)스나개")
    assert has("01 국산쇠고기", "02*안심한우", "04후라이드", "05 자유방목")
    assert not has("7UP 500ml", "1+1 라면", "2% 부족할때")  # 상품명이 숫자로 시작
    assert not has("001 가나다", "003 마바사", "002 아자차")  # 순서가 아니다
    assert not has("001 가나다", "001 마바사", "002 아자차")  # 번호가 겹친다
    assert not has("001 가나다", "002 마바사")  # 둘로는 모른다
    assert not has("001 가나다", "002 마바사", "003 아자차", "카타파", "하하하", "거거거", "너너너")  # 일부만


def test_line_number_is_stripped_from_a_name():
    import server

    assert server._strip_line_number("002 데날개608") == "데날개608"
    assert server._strip_line_number("003데)스나개559") == "데)스나개559"
    assert server._strip_line_number("05*자유방목동물복지란") == "자유방목동물복지란"
    assert server._strip_line_number("005 )ABC쿠키1528") == "ABC쿠키1528"
    # 이 함수는 무조건 벗긴다. 상품명이 숫자로 시작하는 영수증은 _has_line_numbers 가 걸러서 호출되지 않는다.
    assert server._strip_line_number("7UP") == "UP"
    assert server._strip_line_number("001") == "001"  # 이름이 통째로 번호면 그대로


def test_date_day_followed_by_more_digits_is_not_a_date():
    """`25-12-2025`(일-월-년)·`19/01/26` 을 연-월-일로 억지로 읽지 않는다."""
    import server

    assert server._printed_dates([{"text": "25-12-2025", "confidence": 0.95}], _TODAY) == []
    # 날짜와 시각이 붙어 읽힌 것(`26-07-0118:47`)은 그대로 읽는다
    assert server._printed_dates([{"text": "26-07-0118:47", "confidence": 0.95}], _TODAY)[0][:2] == (
        "2026-07-01", "18:47")


def test_date_vlm_date_printed_in_other_formats_is_left_alone():
    for text in ("2026년 7월 1일", "2026,07,01", "26.7.1", "20260701", "일시 2026-07-01~2026-07-02"):
        assert _fix_date("2026-07-01 10:00", text, "승인일: 2026-06-20") == ("2026-07-01 10:00", []), text


def test_date_one_digit_apart_without_weekday_is_not_overwritten():
    # OCR 이 1 을 7 로 읽었을 수도 있다 — 요일이 확인해 주지 않으면 건드리지 않는다
    assert _fix_date("2026-07-01 18:47", "판매일:26-07-07 18:47") == ("2026-07-01 18:47", [])
    # 2026-07-07 은 화요일 — 요일이 맞게 찍혀 있으면 OCR 을 믿는다
    assert _fix_date("2026-07-01 18:47", "판매일:26-07-07 화요일 18:47") == ("2026-07-07 18:47", ["date_from_ocr"])


def test_date_several_candidates_pick_the_one_with_the_vlm_month_and_day():
    assert _fix_date("2023-07-01", "판매일: 2026-07-01", "승인일: 2026-07-09") == ("2026-07-01", ["date_from_ocr"])
    assert _fix_date("2023-07-01", "판매일: 2026-07-05", "승인일: 2026-07-09") == ("2023-07-01", [])


def test_date_label_ilsibul_is_not_a_date_label():
    # `일시불` 은 날짜 라벨이 아니라 카드 할부 표기다. 라벨 줄로 치면 다른 날짜를 이긴다.
    assert _fix_date("2023-01-01", "일시불 2026-06-28", "2026-07-01 18:47") == ("2023-01-01", [])


def test_date_12_hour_clock_keeps_the_vlm_time():
    value, _ = _fix_date("2023-07-26 18:49", "판매일:26-07-01", "오후6:49")
    assert value == "2026-07-01 18:49"


def test_date_future_vlm_year_is_not_protected_by_the_later_year_rule():
    # VLM 이 오늘보다 뒤인 연도를 말하면 지어낸 값이다
    assert _fix_date("2027-08-19 19:07", "판매일:26-08-19 19:07") == ("2026-08-19 19:07", ["date_from_ocr"])


def test_date_vlm_date_glued_to_the_time_counts_as_printed():
    assert _fix_date("2026-08-08 18:32", "2026.08.0818:32:43", "2026.08.0818:33:44") == ("2026-08-08 18:32", [])
