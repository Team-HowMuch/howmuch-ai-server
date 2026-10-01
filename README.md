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
    "total_amount": 0, "payment_method": "...", "total_verified": true
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

줄 최종 금액은 `price + Σsub_items.price − items[].discount` 이고, 검증식

`Σ(줄 최종 금액) − discount + delivery_fee == total_amount`

이 성립하면 `total_verified: true` 가 된다.

할인은 항상 **양수**로 내려간다. 영수증이 `-1,510`으로 찍든 `3,000`으로 찍든
서버가 부호를 정규화한다. 포인트 사용·상품권 사용은 결제수단 분할이라 총액이 바뀌지
않으므로 할인이 아니다.

배달비는 **`items` 에 들어가지 않는다.** 모델이 `배달비 2,000` 을 품목이나 옵션으로 올려도
서버가 떼어내 `delivery_fee` 로 옮긴다. 배달팁 할인·무료배달(`기본배달팁 4,100` 바로 아래
`-4,100`)은 여기서 상계되고 `discount` 에는 들어가지 않는다. 배민 주문전표처럼 `배달팁`
합계 줄과 `ㄴ기본배달팁` 내역 줄이 같이 찍히면 합계 줄만 센다.

`sub_items` 는 품목에 붙는 하위 옵션(추가선택·사이즈업 등)이고 **할인이 아니다**.
할인 줄은 `sub_items` 가 아니라 `discount` 로 간다.

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
| `docker-compose.yml` | vLLM + API 운영 구성 |
| `DEPLOY.md` | GPU 서버 배포, Funnel, systemd 자동 기동 |
| `docs/dual-model-ocr.md` | 이중 모델 파이프라인 설계 |
| `docs/portfolio-*.svg` | 포트폴리오용 다이어그램 (PNG 동봉) |
