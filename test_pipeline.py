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
