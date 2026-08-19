# 영수증 OCR: 이중 모델 병렬 처리 + 병합 로직

영수증 사진 한 장을 **비전 모델(VLM)과 텍스트 OCR 모델 두 개로 동시에** 분석하고,
두 결과를 비교해서 더 신뢰할 수 있는 값을 채택하는 파이프라인입니다.

저품질 영수증에서 `참치김밥`이 `잠치김밥`(ㅊ→ㅈ 오독)으로 나오는 문제를 잡기 위해 도입했습니다.

## 전체 구조

```text
                      ┌─ Qwen2.5-VL 3B (4bit, MLX) ──> 구조화 JSON (상호/일시/품목/합계)   ~17초
영수증 이미지 ─┤                                                              (병렬 실행)
                      └─ PP-OCRv5 korean (RapidOCR) ──> 텍스트 라인 + 신뢰도            ~1.5초
                                        │
                                        ▼
                              병합 (_merge)
              1. 요약 줄 필터: "과세물품가액/부가세/소계" 등은 품목에서 제거
              2. 품목명 보정: VLM 품목명 vs OCR 텍스트 교차 검증 (아래 상세)
              3. 합계 검증: VLM 합계가 OCR 숫자에 존재하면 total_verified=true
```

- 파일 구성: `server.py`(FastAPI + 병렬 실행 + 병합), `corrector.py`(단어 검증/보정)
- 역할 분담 근거(실측): VLM은 구조화와 한글에 강하고, OCR은 숫자가 신뢰도 1.00 수준으로 정확하지만
  한글 인식이 거칠다. 그래서 구조는 VLM, 검증은 OCR이 맡는다.

## 1. 병렬 실행 (server.py)

`ThreadPoolExecutor`로 두 모델을 동시에 던지고 둘 다 끝나면 병합한다.
OCR이 1.5초라 전체 지연은 사실상 VLM 시간과 동일하다.

```python
vlm_future = _executor.submit(_run_vlm, tmp_path)   # 구조화 JSON
ocr_future = _executor.submit(_run_text_ocr, tmp_path)  # 텍스트 라인
raw_text = vlm_future.result()
ocr_lines = ocr_future.result()

parsed = _parse_json(raw_text)
parsed, corrections = _merge(parsed, ocr_lines)
```

각 모델은 자체 락(`_vlm_lock`, `_ocr_lock`)으로 동시 추론을 1건씩 직렬화한다(로컬 단일 GPU 환경 기준).

## 2. 품목명 병합/보정 규칙 (corrector.py)

품목명마다 `KoreanCorrector.correct(name, ocr_texts)`를 호출한다. 판단 순서:

### 1) 두 모델 합의 → 그대로 채택

OCR 라인(신뢰도 0.8 이상만 사용)에서 자모 편집거리가 가장 가까운 한글 토큰을 찾는다.
완전히 같으면 두 모델이 합의한 것이므로 보정 없이 통과 (`vlm_ocr_agree`).

- 자모 편집거리: 글자를 초성/중성/종성으로 분해한 뒤 레벤슈타인 거리.
  `참치`와 `잠치`는 글자 단위로는 거리 1이지만 자모 단위로도 1이라서 미세한 오독 탐지에 유리.

### 2) 불일치 → "실존 단어" 쪽 선택

**Kiwi 형태소 분석기의 분석 점수(로그 확률)를 실존 단어 판별기로 사용한다.**
사전에 있는 형태소로 자연스럽게 분해되면 점수가 높고, 미등록 단어는 점수가 낮다.

```text
참치김밥 → 참치/NNG + 김밥/NNG  → 점수 -20.5  (실존)
잠치김밥 → 잠치김밥/NNG (미등록)  → 점수 -30.9  (가짜)
```

VLM 단어와 OCR 후보를 모두 점수화해서 높은 쪽을 채택한다.

### 3) 둘 다 이상함 → 혼동 자모 치환 보정

저품질 인쇄에서 시각적으로 헷갈리는 자모 쌍(ㅈ↔ㅊ, ㅂ↔ㅍ, ㅏ↔ㅓ, ㅁ↔ㅂ 등)을
한 글자씩 치환한 후보를 전부 생성하고, 그중 Kiwi 점수가 가장 좋은 후보로 보정한다.

### 오보정 방지 장치

- 후보 점수가 원본보다 **3.0점 이상 좋아질 때만** 교체 (`_SCORE_MARGIN`).
  참치김밥/잠치김밥은 10점 차이라 여유 있게 잡히고, 애매한 건 원본 유지.
- OCR 후보는 자모 거리가 단어 길이의 1/3 이내일 때만 인정 (엉뚱한 토큰 매칭 방지).
- 고유명사(브랜드 메뉴명)는 사전에 없어 점수가 낮게 나오므로, 마진 조건 덕에 보수적으로 원본이 유지된다.

## 3. 합계 교차 검증 (server.py)

OCR이 숫자에 강한 점을 이용해서, VLM이 뽑은 `total_amount`가
OCR 라인에서 추출한 숫자 집합에 존재하는지 확인한다. 존재하면 `total_verified: true`.

```python
parsed["total_verified"] = int(total) in _ocr_numbers(ocr_lines)
```

## 검증 결과

단위 테스트:

```text
잠치김밥          => 참치김밥        (confusion_swap)   # OCR 없이도 복원
잠치김밥 + OCR    => 참치김밥        (ocr)              # OCR 교차검증으로 복원
참치김밥          == 참치김밥        (original_plausible) # 실존 단어는 미변경
고추정 떡볶이     => 고추장 떡볶이    (confusion_swap)
꼬마김밥A + OCR   == 꼬마김밥A       (vlm_ocr_agree)    # 두 모델 합의
```

종단 테스트(더현대 영수증): 합계 13,000원 OCR 검증 통과, 요약 줄("과세물품가액" 등) 제거 확인.

## 알려진 한계

- 사전 기반이라 `아이수 → 아이스` 같은 케이스는 못 잡는다 ("아이+수"가 사전상 그럴듯해서 통과).
  → 음식명 사전을 Kiwi 사용자 사전(`add_user_word`)에 추가하면 개선 가능.
- 혼동 자모 치환은 편집거리 1까지만 생성 (두 글자 이상 동시 오독은 OCR 교차검증 경로에 의존).

## 사용 스택

| 구성 | 선택 | 비고 |
|---|---|---|
| VLM | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | Apple Silicon MLX, 약 2GB |
| OCR | `PP-OCRv5 korean mobile` (RapidOCR/onnxruntime) | CPU로 1.5초 |
| 단어 검증 | `kiwipiepy` (Kiwi 형태소 분석기) | 분석 점수를 실존 단어 판별에 사용 |
