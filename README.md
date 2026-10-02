# howmuch-ai-server

영수증 OCR AI 서버. 영수증 사진 한 장을 **비전 언어 모델(VLM)과 텍스트 OCR 모델로 병렬 분석**하고,
두 결과를 병합·교차검증해서 구조화된 정산 데이터(상호/일시/품목/합계)를 반환한다.

## 구조

```text
                ┌─ VLM (Qwen2.5-VL) ──> 구조화 JSON (상호/일시/품목/합계)
영수증 이미지 ─┤                                        (병렬 실행)
                └─ OCR (PP-OCRv5 korean) ──> 텍스트 라인 + 신뢰도
                              │
                              ▼
                          병합/보정
        · 품목명: 두 모델 합의 → 채택 / 불일치 → Kiwi 사전 점수로 실존 단어 선택
          / 둘 다 미등록 → 혼동 자모(ㅊ↔ㅈ 등) 치환 보정
        · 합계: OCR 숫자와 대조해 total_verified 플래그
        · 요약 줄(부가세/과세물품가액 등) 품목 오분류 필터
                              │
                              ▼
                          합계 검산 (reconcile.py)
        · 실제 결제액 = Σ품목 − 할인 + 배달비 가 맞도록 새는 줄을 빼고
          OCR 에 찍힌 할인·배달비를 채운다 → items_verified 플래그
```

상세 설계: [docs/dual-model-ocr.md](docs/dual-model-ocr.md)

## 실행

### 운영 (GPU 서버, Docker)

vLLM(Qwen2.5-VL-7B AWQ) + API 컨테이너 구성. [DEPLOY.md](DEPLOY.md) 참고.

```bash
docker compose up -d --build
```

### 로컬 개발 (Apple Silicon 맥)

MLX 백엔드(Qwen2.5-VL-3B 4bit)로 자동 전환된다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
```

## API

`POST /ocr/receipt` — multipart 필드 `file`에 영수증 이미지

```json
{
  "ok": true,
  "elapsed_sec": 3.2,
  "result": {
    "store_name": "...", "purchased_at": "YYYY-MM-DD HH:MM",
    "items": [{"name": "...", "quantity": 1, "price": 0, "discount": 0,
               "sub_items": [{"name": "...", "price": 0}]}],
    "discount": 0, "delivery_fee": 0,
    "total_amount": 0, "payment_method": "...", "total_verified": true, "items_verified": true
  },
  "corrections": [{"field": "item.name", "before": "잠치김밥", "after": "참치김밥", "reason": "ocr"}],
  "ocr_lines": [{"text": "...", "confidence": 0.98}]
}
```

### 금액 필드

| 필드 | 뜻 |
|---|---|
| `items[].price` | 옵션 제외, 할인 **전**, 수량 반영된 줄 금액 |
| `items[].sub_items[].price` | 옵션 추가금 (줄 금액에 더한다) |
| `items[].discount` | 그 품목 줄에 귀속되는 할인액 (0 이상, 없으면 0) |
| `discount` | 특정 품목에 귀속되지 않는 영수증 전체 단위 할인액 (0 이상, 없으면 0) |
| `delivery_fee` | 배달비 순액. 배달팁 할인·무료배달을 상계한 값 (0 이상, 없으면 0) |

줄 최종 금액은 `price + Σsub_items.price − items[].discount` 이다.

| 플래그 | 뜻 |
|---|---|
| `total_verified` | `total_amount` 가 영수증에 찍힌 결제액·합계로 확인됐다 |
| `items_verified` | `total_verified` 이고, 검증식 `Σ(줄 최종 금액) − discount + delivery_fee == total_amount` 도 성립한다 |

`items_verified: false` 면 품목·할인 중 놓치거나 잘못 읽은 줄이 있다는 뜻이다. 앱에서 품목 확인을
받는 UX 를 권장한다. 실측 63장(정답지 대조)에서 `items_verified: true` 인 결과는 모두 정답이었다.

`total_amount` 는 손님이 **실제로 낸 돈**이다. 배민 주문서의 `합계금액 23,300` 아래 `결제 금액 상세`
에 `21,300`(채널할인 1,000 + 배달앱할인 1,000)이 찍혀 있으면 21,300 이 합계이고 2,000 은
`discount` 로 들어간다. 카드 + 상품권·포인트 분할 결제는 합계가 그대로다.

할인은 항상 **양수**로 내려간다. 영수증이 `-1,510`으로 찍든 `3,000`으로 찍든
서버가 부호를 정규화한다. 포인트 사용·상품권 사용은 결제수단 분할이라 총액이 바뀌지
않으므로 할인이 아니다.

배달비는 **`items` 에 들어가지 않는다.** 모델이 `배달비 2,000` 을 품목이나 옵션으로 올려도
서버가 떼어내 `delivery_fee` 로 옮긴다. 배달팁 할인·무료배달(`기본배달팁 4,100` 바로 아래
`-4,100`)은 여기서 상계되고 `discount` 에는 들어가지 않는다. 배민 주문전표처럼 `배달팁`
합계 줄과 `ㄴ기본배달팁` 내역 줄이 같이 찍히면 합계 줄만 센다.

`sub_items` 는 품목에 붙는 하위 옵션(추가선택·사이즈업 등)이고 **할인이 아니다**.
할인 줄은 `sub_items` 가 아니라 `discount` 로 간다.

### 합계 검산

병합 마지막에 `reconcile.py` 가 영수증에 찍힌 결제액을 검산 기준으로 결과를 고친다.

1. 이름만 봐도 품목이 아닌 줄(`합할 계`, `가세(VAT):`, `카드/간편결제`, `적립포인트`)을 지운다.
2. OCR 에서 결제액 후보를 찾는다: 결제 라벨(`결제금액`·`카드결제`·배민 `결제 금액 상세`) >
   합계 라벨·두 번 이상 찍힌 금액·`합계 − 할인` > VLM 합계(OCR 근거가 하나도 없을 때만).
3. 그 결제액을 맞추는 **가장 작은 수정**을 찾는다(최대 2개 + 번진 배달팁 1개).
   - 뺄 수 있는 것: 좌표로 되살린 품목, 합계·세금 금액과 같은 품목·옵션, 다른 품목 금액을 끌어온 옵션,
     요약줄이 한 번 더 붙은 품목 할인
   - 더할 수 있는 것: OCR 에 찍힌 할인(음수 줄, 할인 어휘 줄, 결제 상세 할인), 번진 배달팁(차액)
   - 고칠 수 있는 것: 단가를 금액으로 적은 품목(곱한 금액이 영수증에 찍혔을 때), 옵션 금액이 이미
     들어간 줄 금액
   - 되살릴 수 있는 것: VLM 이 빠뜨렸고 OCR 이 이름을 흐리게 읽은 품목 줄. 그 줄을 넣어야만 결제액이
     **정확히** 맞을 때만 쓴다. 이름이 합계·헤더·배달·결제 줄 같거나 금액이 합계·부가세와 같은 줄,
     서로 다른 줄을 되살려도 맞는 경우는 쓰지 않는다. 이름은 OCR 이 읽은 글자 그대로라 틀릴 수 있다.

수정은 OCR 에 실제로 찍힌 금액에서만 고르고, 같은 비용으로 다른 결제액이 맞으면 아무것도 바꾸지
않는다. 고친 내용은 `corrections` 에 남는다.

| `corrections[].reason` | 뜻 |
|---|---|
| `label_dropped` | 품목이 아닌 라벨 줄을 지움 |
| `reconciled` | 검산으로 고침(지운 줄, 더한 할인, 고친 금액, 채운 배달비, 바꾼 합계) |
| `ocr_recovered` | VLM 이 빠뜨린 품목을 OCR 줄로 되살림(이름이 확실한 줄, 또는 합계 검산이 요구한 줄) |
| `date_from_ocr` | VLM 날짜가 영수증 어디에도 없어서 OCR 이 읽은 날짜로 바꿈 |
| `negative_item` | 음수 금액 품목(`(카드쿠폰) -12,000`)을 할인으로 바꿈 |
| `option_demoted` | 품목으로 올라온 옵션 줄(`ㄴ타피오카펄 추가`, 배달 영수증의 `보통맛 0`)을 옵션으로 내림 |

### 기울어 찍힌 사진과 날짜

- 사진이 기울면 줄 묶기가 오른쪽 금액을 다음 줄 이름과 짝지어 품목·할인이 어긋난다. `layout.py` 는
  같은 줄의 토큰 쌍 기울기로 기울어진 각도(0.8°~12°)를 추정하고, **품목 영역만** 펴서 줄을 묶는다
  (표 머리글을 찾았을 때만. 합계 영역은 건드리지 않는다).
- VLM 이 읽은 날짜(`2023-07-26`)가 OCR 줄 어디에도 없으면 OCR 이 읽은 날짜(`26-07-01`)로 바꾼다.
  라벨(`판매일` 등)이 있는 줄을 우선하고, 후보가 갈리거나 요일이 안 맞거나 월·일이 한 글자만 다르고
  요일 확인이 없으면 바꾸지 않는다.

### 실험용 쿼리

인식률 개선을 실측 영수증으로 비교하기 위한 옵션이다. 백엔드는 보내지 않으며, 생략하면
서버 기본값(환경변수)을 쓴다. 응답의 `pipeline`에 실제로 적용된 설정이 남는다.

| 쿼리 | 값 | 기본값(환경변수) |
|---|---|---|
| `prompt` | `v1`(기존) · `v2`(배달비 칸, `ㄴ` 옵션 기호, 시각 없으면 날짜만, 상호명 규칙) | `VLM_PROMPT_VERSION`, 없으면 `v1` |
| `preprocess` | 쉼표로 `crop`(영수증 영역 잘라내기) · `hires`(OCR 입력만 긴 변 2560) · `clahe`(OCR 입력만 대비 보정). 빈 문자열이면 끔 | `OCR_PREPROCESS`, 없으면 끔 |

예: `POST /ocr/receipt?prompt=v2&preprocess=crop,hires`

`GET /` — 폰 카메라 촬영 테스트 페이지

## 주요 파일

| 파일 | 역할 |
|---|---|
| `server.py` | FastAPI 서버, 이중 모델 병렬 실행 및 병합 |
| `vlm.py` | VLM 백엔드 추상화 (MLX 로컬 / vLLM OpenAI 호환) |
| `corrector.py` | 한국어 단어 검증(Kiwi) 및 혼동 자모 보정 |
| `layout.py` | OCR 좌표로 줄·품목·할인·배달비·합계 복원 |
| `reconcile.py` | 결제액 기준 합계 검산 |
| `preprocess.py` | 실험용 이미지 전처리(crop·hires·clahe) |
| `docker-compose.yml` | vLLM + API 운영 구성 |
| `DEPLOY.md` | GPU 서버 배포, Funnel, systemd 자동 기동 |
| `docs/dual-model-ocr.md` | 이중 모델 파이프라인 설계 |
| `docs/portfolio-*.svg` | 포트폴리오용 다이어그램 (PNG 동봉) |
