"""VLM/OCR 결과 병합 및 한국어 단어 보정 모듈.

기본은 모델이 읽은 품목명을 유지한다. Kiwi 점수는 상품명보다 고빈도
일반명사를 선호해서, 점수만으로 갈아끼우면 구운란→구운난처럼 망가진다.

보정은 원본이 미등록(OOV) 덩어리일 때만 한다.
예: 잠치김밥(OOV) → 참치김밥(실존). 구운란·신라면소컵은 분석 가능하므로 그대로 둔다.
"""
import re
from dataclasses import dataclass, field

from kiwipiepy import Kiwi

# 저품질 영수증 인쇄에서 서로 헷갈리기 쉬운 자모 그룹
_ONSET_CONFUSION = [
    {"ㅈ", "ㅊ"}, {"ㅂ", "ㅍ"}, {"ㅁ", "ㅂ"}, {"ㄱ", "ㅋ"}, {"ㄷ", "ㅌ"},
    {"ㅅ", "ㅆ"}, {"ㅈ", "ㅉ"}, {"ㄱ", "ㄲ"}, {"ㅇ", "ㅎ"}, {"ㄴ", "ㄹ"},
]
_VOWEL_CONFUSION = [
    {"ㅏ", "ㅓ"}, {"ㅗ", "ㅜ"}, {"ㅐ", "ㅔ"}, {"ㅑ", "ㅕ"},
]
_CODA_CONFUSION = [
    {"ㅁ", "ㅂ"}, {"ㄴ", "ㄹ"}, {"ㄱ", "ㄲ"}, {"", "ㅇ"},
]

_ONSETS = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ",
           "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_VOWELS = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ",
           "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
_CODAS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
          "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ",
          "ㅌ", "ㅍ", "ㅎ"]

# 실제 OCR 오독(잠치김밥→참치김밥, 고추정→고추장)은 약 10점.
# 상품 표기 vs 고빈도 일반명사(구운란→구운난)는 약 3~4점이라 그 아래로 둔다.
_SCORE_MARGIN = 8.0

# 편의점 영수증에 자주 나오지만 일반 사전에는 약하게 잡히는 상품 형태
_USER_WORDS = [
    ("소컵", "NNG"),
    ("큰컵", "NNG"),
    ("대컵", "NNG"),
    ("미니컵", "NNG"),
    ("컵라면", "NNG"),
    ("구운란", "NNG"),
    ("훈제란", "NNG"),
    ("반숙란", "NNG"),
    ("염계란", "NNG"),
]


def _decompose(ch: str):
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:
        return None
    return _ONSETS[code // 588], _VOWELS[(code % 588) // 28], _CODAS[code % 28]


def _compose(onset: str, vowel: str, coda: str) -> str:
    return chr(0xAC00 + _ONSETS.index(onset) * 588 + _VOWELS.index(vowel) * 28 + _CODAS.index(coda))


def _swap_options(jamo: str, groups) -> set[str]:
    out = set()
    for group in groups:
        if jamo in group:
            out |= group - {jamo}
    return out


def _jamo_seq(text: str) -> list[str]:
    seq = []
    for ch in text:
        parts = _decompose(ch)
        seq.extend(parts if parts else [ch])
    return seq


def _jamo_distance(a: str, b: str) -> int:
    sa, sb = _jamo_seq(a), _jamo_seq(b)
    dp = list(range(len(sb) + 1))
    for i, ca in enumerate(sa, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(sb, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def _similar_length(a: str, b: str) -> bool:
    return abs(len(a) - len(b)) <= max(1, len(a) // 4)


@dataclass
class Correction:
    original: str
    corrected: str
    changed: bool
    reason: str
    candidates: list = field(default_factory=list)


class KoreanCorrector:
    def __init__(self):
        self._kiwi = Kiwi()
        for word, tag in _USER_WORDS:
            self._kiwi.add_user_word(word, tag)

    def _tokens(self, text: str):
        return self._kiwi.tokenize(text)

    def _has_oov(self, text: str) -> bool:
        return any(token.oov for token in self._tokens(text))

    def _one_char_nouns(self, text: str) -> set[str]:
        return {
            token.form
            for token in self._tokens(text)
            if token.len == 1 and token.tag.startswith("N")
        }

    def _is_safe_fix(self, original: str, candidate: str) -> bool:
        """OOV 오독만 실존 단어로 되돌린다. 빈도 높은 일반명사로 바꾸는 건 거부."""
        if original == candidate or len(original) != len(candidate):
            return False
        if not self._has_oov(original) or self._has_oov(candidate):
            return False
        extra_glue = self._one_char_nouns(candidate) - self._one_char_nouns(original)
        if extra_glue:
            return False
        return self.score(candidate) - self.score(original) >= _SCORE_MARGIN

    def score(self, text: str) -> float:
        """Kiwi 분석 점수. 높을수록 자연스러운(실존하는) 한국어."""
        korean = re.sub(r"[^가-힣]", "", text)
        if not korean:
            return 0.0
        return self._kiwi.analyze(korean)[0][1]

    def _confusion_candidates(self, word: str) -> set[str]:
        """혼동 자모를 한 글자씩 치환한 후보 생성 (자모 편집거리 1)."""
        candidates = set()
        chars = list(word)
        for i, ch in enumerate(chars):
            parts = _decompose(ch)
            if not parts:
                continue
            onset, vowel, coda = parts
            for alt in _swap_options(onset, _ONSET_CONFUSION):
                candidates.add("".join(chars[:i] + [_compose(alt, vowel, coda)] + chars[i + 1:]))
            for alt in _swap_options(vowel, _VOWEL_CONFUSION):
                candidates.add("".join(chars[:i] + [_compose(onset, alt, coda)] + chars[i + 1:]))
            for alt in _swap_options(coda, _CODA_CONFUSION):
                candidates.add("".join(chars[:i] + [_compose(onset, vowel, alt)] + chars[i + 1:]))
        candidates.discard(word)
        return candidates

    def _find_similar_ocr_token(self, word: str, ocr_texts: list[str]) -> str | None:
        """OCR 결과에서 word와 가장 가까운 토큰 탐색 (자모 거리 기준)."""
        korean = re.sub(r"[^가-힣]", "", word)
        if len(korean) < 2:
            return None
        best, best_dist = None, 10 ** 9
        limit = max(2, len(_jamo_seq(korean)) // 3)
        for line in ocr_texts:
            for token in re.findall(r"[가-힣]{2,}", line):
                if not _similar_length(korean, token):
                    continue
                dist = _jamo_distance(korean, token)
                if dist < best_dist:
                    best, best_dist = token, dist
        return best if best is not None and best_dist <= limit else None

    def correct(self, word: str, ocr_texts: list[str] | None = None) -> Correction:
        """품목명 내 한글 구간을 각각 검증/보정하고 결과를 합친다."""
        runs = list(re.finditer(r"[가-힣]{2,}", word))
        if not runs:
            return Correction(word, word, False, "too_short")

        rebuilt = word
        changed = False
        reasons = []
        all_candidates = []
        for m in reversed(runs):  # 뒤에서부터 치환해야 인덱스가 유지됨
            res = self._correct_run(m.group(), ocr_texts or [])
            reasons.append(res.reason)
            all_candidates.extend(res.candidates)
            if res.changed:
                changed = True
                rebuilt = rebuilt[: m.start()] + res.corrected + rebuilt[m.end():]
        return Correction(word, rebuilt, changed, ",".join(reversed(reasons)), all_candidates)

    def _correct_run(self, korean: str, ocr_texts: list[str]) -> Correction:
        ocr_token = self._find_similar_ocr_token(korean, ocr_texts)
        if ocr_token == korean:
            return Correction(korean, korean, False, "vlm_ocr_agree")

        # 분석 가능한 상품명(구운란, 소컵)은 Kiwi 점수가 높아도 손대지 않는다.
        if not self._has_oov(korean):
            return Correction(korean, korean, False, "original_plausible")

        candidates: dict[str, str] = {}
        if ocr_token:
            candidates[ocr_token] = "ocr"
        for cand in self._confusion_candidates(korean):
            candidates.setdefault(cand, "confusion_swap")
        if not candidates:
            return Correction(korean, korean, False, "no_candidates")

        safe = [
            (cand, src, self.score(cand))
            for cand, src in candidates.items()
            if self._is_safe_fix(korean, cand)
        ]
        if not safe:
            return Correction(korean, korean, False, "original_plausible")

        safe.sort(key=lambda item: item[2], reverse=True)
        best_cand, best_src, _ = safe[0]
        return Correction(
            korean, best_cand, True, best_src,
            candidates=[(cand, round(score, 1)) for cand, _, score in safe[:5]],
        )
