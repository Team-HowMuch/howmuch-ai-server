#!/usr/bin/env python3
"""품목명 후보정이 실존 오독만 고치고 상품명을 훼손하지 않는지 검증한다."""
from corrector import KoreanCorrector


def _assert_kept(corrector: KoreanCorrector, word: str, ocr_texts=None):
    result = corrector.correct(word, ocr_texts)
    assert result.corrected == word, f"{word!r} -> {result.corrected!r} ({result.reason})"
    assert not result.changed


def _assert_fixed(corrector: KoreanCorrector, word: str, expected: str, ocr_texts=None):
    result = corrector.correct(word, ocr_texts)
    assert result.corrected == expected, f"{word!r} -> {result.corrected!r}, expected {expected!r} ({result.reason})"
    assert result.changed


def main():
    corrector = KoreanCorrector()

    _assert_fixed(corrector, "잠치김밥", "참치김밥")
    _assert_fixed(corrector, "잠치김밥", "참치김밥", ["참치김밥 3,500"])
    _assert_kept(corrector, "참치김밥")
    _assert_fixed(corrector, "떡복이", "떡볶이")
    _assert_kept(corrector, "고추정")
    _assert_kept(corrector, "꼬마김밥A", ["꼬마김밥A"])

    _assert_kept(corrector, "구운란")
    _assert_kept(corrector, "구운란", ["구운난"])
    _assert_kept(corrector, "신라면소컵")
    _assert_kept(corrector, "신라면 소컵")
    _assert_kept(corrector, "소컵")
    _assert_kept(corrector, "신라면소컵", ["신라면수컵", "수컵", "1,200"])

    print("ok")


if __name__ == "__main__":
    main()
