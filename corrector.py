"""VLM/OCR 결과 병합 및 한국어 단어 보정 모듈.

전략:
1. VLM 품목명이 OCR 텍스트와 (거의) 일치하면 그대로 채택 (두 모델 합의)
2. 불일치하면 Kiwi 형태소 분석 점수로 "실존 단어" 쪽을 선택
3. 둘 다 미등록 단어면 시각적 혼동 자모(ㅊ↔ㅈ 등)를 치환한 후보 중
   점수가 가장 좋은 실존 단어로 보정
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

# 보정 채택에 필요한 최소 점수 개선 폭 (참치김밥 vs 잠치김밥은 약 10점 차이)
_SCORE_MARGIN = 3.0


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
        base_score = self.score(korean)
        candidates: dict[str, str] = {}  # 후보 -> 출처

        # 1) OCR 교차 검증: 비슷한 토큰이 있으면 후보에 추가
        ocr_token = self._find_similar_ocr_token(korean, ocr_texts)
        if ocr_token == korean:
            return Correction(korean, korean, False, "vlm_ocr_agree")
        if ocr_token:
            candidates[ocr_token] = "ocr"

        # 2) 혼동 자모 치환 후보
        for cand in self._confusion_candidates(korean):
            candidates.setdefault(cand, "confusion_swap")

        if not candidates:
            return Correction(korean, korean, False, "no_candidates")

        scored = sorted(
            ((cand, src, self.score(cand)) for cand, src in candidates.items()),
            key=lambda x: x[2],
            reverse=True,
        )
        best_cand, best_src, best_score = scored[0]

        if best_score - base_score >= _SCORE_MARGIN:
            return Correction(
                korean, best_cand, True, best_src,
                candidates=[(c, round(s, 1)) for c, _, s in scored[:5]],
            )
        return Correction(korean, korean, False, "original_plausible")
