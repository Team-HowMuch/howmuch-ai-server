"""기울어 찍힌 영수증(홈푸드마트)의 회귀 테스트.

2026-10-01 앱에서 올린 실제 사진의 OCR 줄 66개와 VLM 원문을 그대로 옮겼다. 이 영수증은 약 1.6도 기울어
있어서 이름 줄과 금액 줄이 한 줄씩 어긋나게 묶였고, 그 결과 품목 2개가 빠지고 할인이 엉뚱한 곳에 붙었고
날짜가 2023-07-26 으로 나왔다. 정답은 영수증을 눈으로 읽어 적은 값이다.

실행: .venv/bin/python -m pytest test_tilted_receipt.py -v
"""
import copy

from layout import _to_tokens, analyze_receipt, deskew, estimate_skew, group_rows
from test_layout import _merge_with

# (텍스트, 신뢰도, [x1, y1, x2, y2])
_OCR = [
    ('홍풍등마트(선육통)', 0.809, [407, 142, 628, 227]),
    ('L1/-982-6908610-8882', 0.746, [413, 193, 743, 249]),
    ('판매입:26-07-0118:47,수요일계산대:002', 0.942, [411, 249, 812, 307]),
    ('NO.상품명', 0.904, [410, 289, 537, 335]),
    ('단가수량', 0.995, [614, 301, 735, 339]),
    ('001 몬스터 시트라 355m!', 0.832, [411, 332, 649, 371]),
    ('002 데날개608', 0.793, [409, 389, 674, 432]),
    ('4897036682073', 0.965, [452, 356, 593, 388]),
    ('2,000', 0.994, [611, 363, 671, 388]),
    ('-400', 0.984, [622, 382, 672, 407]),
    ('2,000', 0.999, [760, 365, 820, 393]),
    ('007-', 0.689, [770, 385, 821, 414]),
    ('8801123205851', 0.973, [450, 414, 593, 446]),
    ('003데)스나개559', 0.789, [407, 428, 653, 471]),
    ('2', 0.996, [710, 425, 732, 449]),
    ('2,200', 0.943, [759, 424, 820, 452]),
    ('68011232039181,000', 0.901, [449, 452, 709, 490]),
    ('004 스터오지레모네이드 355m|', 0.899, [407, 469, 719, 508]),
    ('3', 1, [709, 464, 730, 486]),
    ('3,000', 0.997, [759, 462, 822, 490]),
    ('7', 0.946, [249, 503, 269, 526]),
    ('005 )ABC쿠키1528', 0.789, [404, 527, 673, 566]),
    ('4897036694817-2,000', 0.953, [448, 492, 673, 527]),
    ('-400', 0.968, [616, 513, 672, 543]),
    ('2,000', 0.998, [761, 500, 822, 525]),
    ('-400', 0.889, [771, 519, 823, 545]),
    ('88010628935393,700', 0.983, [448, 553, 671, 580]),
    ('3,700', 1, [763, 555, 824, 583]),
    ('006바리스타모카프레소250m|', 0.954, [405, 569, 699, 601]),
    ('88011211079422200', 0.923, [446, 591, 673, 621]),
    ('3', 0.999, [714, 596, 731, 618]),
    ('6,600', 0.958, [763, 594, 824, 623]),
    ('007 매맛 ii0g', 0.701, [404, 608, 667, 641]),
    ('88864671000552,900', 0.98, [445, 629, 672, 659]),
    ('2,900', 0.998, [762, 633, 824, 661]),
    ('-700', 0.991, [617, 652, 671, 677]),
    ('-700', 0.931, [769, 654, 824, 683]),
    ('008 해태)금세 44', 0.719, [403, 668, 632, 698]),
    ('8801019309946', 0.968, [445, 689, 589, 717]),
    (',000', 0.827, [604, 691, 672, 719]),
    ('2,000', 1, [762, 692, 825, 724]),
    ('009 )매새우09', 0.674, [400, 705, 652, 738]),
    ('8801043036078', 0.962, [444, 730, 589, 757]),
    ('1,400', 0.993, [611, 729, 672, 757]),
    ('1,400', 0.997, [764, 733, 824, 761]),
    ('-410', 0.995, [620, 748, 673, 779]),
    ('-410', 0.996, [772, 754, 825, 784]),
    ('-1,910', 0.938, [753, 794, 825, 822]),
    ('21,719', 0.974, [752, 814, 825, 842]),
    ('부가세(VAT)', 0.911, [400, 828, 524, 861]),
    ('2,171', 0.955, [754, 833, 825, 863]),
    ('신용카드지골', 0.851, [398, 867, 533, 907]),
    ('23,890', 0.987, [693, 854, 824, 884]),
    ('23,890', 0.998, [755, 877, 825, 905]),
    ('객:', 0.95, [485, 917, 519, 940]),
    ('92', 0.981, [796, 941, 824, 966]),
    ('27,768', 0.918, [754, 959, 826, 991]),
    ('-카드승인(IC)', 0.971, [545, 981, 680, 1012]),
    ('카카오뱅크', 0.996, [392, 1003, 513, 1033]),
    ('있시볼', 0.695, [666, 1003, 736, 1035]),
    ('/23,890', 0.997, [732, 1002, 827, 1033]),
    ('****-****-**E0-!676', 0.877, [392, 1028, 619, 1055]),
    ('(29174213)', 0.997, [610, 1021, 725, 1057]),
    ('박흙신 영수 직찰해주세요', 0.648, [387, 1064, 687, 1109]),
    ('거래ND:0701105756 계산원 최타숙(005)', 0.918, [387, 1086, 785, 1127]),
    ('250707105759', 0.834, [495, 1158, 748, 1196]),
]

_VLM_RAW = """```json
{
  "store_name": "홈푸드마트(선우유통)",
  "purchased_at": "2023-07-26 18:47",
  "items": [
    {"name": "몬스터 시트러 355ml", "quantity": 1, "price": 2000, "sub_items": []},
    {"name": "몬스터오지레모네이드 355ml", "quantity": 1, "price": 2000, "sub_items": []},
    {"name": "ABC초코쿠키", "quantity": 1, "price": 3700, "sub_items": []},
    {"name": "바리스타모카프레소", "quantity": 3, "price": 6600, "sub_items": []},
    {"name": "프링글스 매운맛", "quantity": 1, "price": 2900, "sub_items": []},
    {"name": "해태)포키극세", "quantity": 1, "price": 2000, "sub_items": []},
    {"name": "농심)매운새우깡", "quantity": 1, "price": 1400, "sub_items": []}
  ],
  "total_amount": 23890,
  "payment_method": "카드"
}
```"""


def _ocr_lines():
    return [{"text": t, "confidence": c, "box": list(b)} for t, c, b in _OCR]


def _parsed():
    import server

    return server._parse_json(_VLM_RAW)


def _items(merged):
    return [(i["name"], i["quantity"], i["price"], i["discount"]) for i in merged["items"]]


# ---------------------------------------------------------------- 줄 묶기

def test_without_deskew_the_amounts_land_on_the_next_items_name_row():
    """문제의 재현: 기울기를 펴지 않으면 002 의 금액(2,200)이 003 의 이름 줄에 붙는다."""
    rows = group_rows(_to_tokens(_ocr_lines()))
    row_of_003 = next(r for r in rows if "003" in r.text)
    assert "2,200" in [t.text for t in row_of_003.tokens]


def test_deskew_puts_each_amount_on_its_own_items_row():
    layout = analyze_receipt(_ocr_lines())
    got = [(li.price, li.quantity, li.discount) for li in layout.items]
    assert got == [
        (2000, None, 400), (2200, 2, 0), (3000, 3, 0), (2000, None, 400), (3700, None, 0),
        (6600, 3, 0), (2900, None, 700), (2000, None, 0), (1400, None, 410),
    ]
    # 합계부 `-1,910` 이 요약 할인이다. 품목 줄에 섞인 -400 을 또 더해 2,310 이 되면 안 된다.
    assert layout.summary_discount == 1910


def test_tilt_estimate_matches_the_photo():
    """손으로 잰 기울기는 약 1.7도(tan 0.03). 합계부를 뺀 품목 구간에서 잰다."""
    tokens = _to_tokens(_ocr_lines())
    rows = group_rows(tokens)
    from layout import _find_item_region

    start, end = _find_item_region(rows)
    region = [t for r in rows[start:end] for t in r.tokens]
    assert 0.02 < estimate_skew(region) < 0.04


def test_deskew_leaves_the_summary_section_alone():
    tokens = _to_tokens(_ocr_lines())
    fixed = deskew(tokens)
    rows = group_rows(tokens)
    from layout import _find_item_region

    _, end = _find_item_region(rows)
    summary = {id(t) for r in rows[end:] for t in r.tokens}
    assert summary
    for before, after in zip(tokens, fixed):
        if id(before) in summary:
            assert after is before


# ---------------------------------------------------------------- 전체 병합

def test_every_item_discount_date_and_total_come_out_right():
    merged, corrections = _merge_with(_parsed(), _ocr_lines())
    assert _items(merged) == [
        ("몬스터 시트러 355ml", 1, 2000, 400),
        (merged["items"][1]["name"], 2, 2200, 0),
        (merged["items"][2]["name"], 3, 3000, 0),
        ("몬스터오지레모네이드 355ml", 1, 2000, 400),
        ("ABC초코쿠키", 1, 3700, 0),
        ("바리스타모카프레소", 3, 6600, 0),
        ("프링글스 매운맛", 1, 2900, 700),
        ("해태)포키극세", 1, 2000, 0),
        ("농심)매운새우깡", 1, 1400, 410),
    ]
    assert merged["discount"] == 0 and merged["delivery_fee"] == 0
    assert merged["total_amount"] == 23890
    assert merged["total_verified"] is True and merged["items_verified"] is True
    assert merged["purchased_at"] == "2026-07-01 18:47"
    reasons = [c["reason"] for c in corrections]
    assert reasons.count("ocr_recovered") == 2 and "date_from_ocr" in reasons
    assert not any(k.startswith("_") for k in merged) and not any(k.startswith("_") for i in merged["items"] for k in i)


def test_restored_items_sit_where_they_are_printed():
    merged, _ = _merge_with(_parsed(), _ocr_lines())
    assert [i["price"] for i in merged["items"]][:4] == [2000, 2200, 3000, 2000]


def test_blurry_lines_are_not_restored_unless_the_total_demands_them():
    """이름이 흐린 줄(신뢰도 0.79)은 그 줄만으로는 되살리지 않는다. 합계가 안 맞으면 7개 그대로."""
    lines = _ocr_lines()
    for line in lines:
        line["text"] = line["text"].replace("23,890", "24,890")
    parsed = _parsed()
    parsed["total_amount"] = 24890
    merged, corrections = _merge_with(parsed, lines)
    assert len(merged["items"]) == 7
    assert merged["items_verified"] is False
    assert "ocr_recovered" not in [c["reason"] for c in corrections]


def test_nothing_is_restored_when_the_total_is_only_the_vlms_claim():
    """영수증에 결제액이 찍혀 있지 않으면(VLM 만 23,890 이라고 한다) 검산이 품목을 만들지 않는다."""
    lines = [l for l in _ocr_lines() if "23,890" not in l["text"]]
    merged, corrections = _merge_with(_parsed(), lines)
    assert len(merged["items"]) == 7
    assert merged["items_verified"] is False
    assert "ocr_recovered" not in [c["reason"] for c in corrections]


def test_input_is_not_mutated_between_runs():
    """같은 입력을 두 번 돌려도 같은 결과(복구 후보 같은 내부 상태가 새지 않는다)."""
    first, _ = _merge_with(_parsed(), copy.deepcopy(_ocr_lines()))
    second, _ = _merge_with(_parsed(), copy.deepcopy(_ocr_lines()))
    assert first == second
