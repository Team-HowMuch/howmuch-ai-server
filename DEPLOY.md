# 홈서버(RTX 3090) 배포 가이드

구성: vLLM 컨테이너(Qwen2.5-VL-7B AWQ, GPU) + API 컨테이너(FastAPI + PP-OCRv5 + 보정기, CPU)

## 사전 조건

- Docker + docker compose v2
- nvidia-container-toolkit (docker에서 GPU 사용)

```bash
# 설치 여부 확인 — GPU 정보가 출력되면 준비된 것
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

안 되어 있으면:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 1. 코드 전송 (맥북에서 실행)

```bash
rsync -av --exclude .venv --exclude server.log --exclude '*.zip' \
  ~/Projects/howmuch-ai-server/ <유저>@<홈서버IP>:~/howmuch-ai-server/
```

## 2. 기동 (홈서버에서 실행)

```bash
cd ~/howmuch-ai-server
docker compose up -d --build
```

- 최초 기동 시 vLLM이 모델(~7GB)을 다운로드하므로 5~10분 걸립니다.
- 모델은 named volume(hf-cache)에 캐시되어 재기동 시 바로 뜹니다.
- 진행 상황: `docker compose logs -f vllm`
- API는 vLLM healthcheck 통과 후 자동으로 시작됩니다.

## 3. 확인

```bash
# 서버 상태
docker compose ps

# OCR 테스트 (영수증 이미지로)
curl -F "file=@receipt.jpg" http://localhost:8600/ocr/receipt
```

폰/외부에서: `http://<홈서버IP>:8600` (촬영 테스트 페이지)

## 운영 명령어

```bash
docker compose logs -f api      # API 로그
docker compose restart api      # 코드 수정 후 재빌드: up -d --build api
docker compose down             # 전체 종료 (모델 캐시는 유지)
```

## 환경변수 (docker-compose.yml)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLM_BACKEND` | openai | vLLM API 호출 모드 |
| `VLM_API_BASE` | http://vllm:8000/v1 | vLLM 주소 |
| `VLM_MODEL` | Qwen/Qwen2.5-VL-7B-Instruct-AWQ | 모델 교체 시 vllm 서비스 command도 함께 변경 |

## Spring 백엔드 연동 (추후)

adapter.out에 OCR Output Port 구현체를 만들어 `POST http://<홈서버IP>:8600/ocr/receipt`
(multipart, 필드명 `file`)를 호출하면 됩니다. 응답 스키마:

```json
{
  "ok": true,
  "elapsed_sec": 3.2,
  "result": {
    "store_name": "...", "purchased_at": "YYYY-MM-DD HH:MM",
    "items": [{"name": "...", "quantity": 1, "price": 0}],
    "total_amount": 0, "payment_method": "...", "total_verified": true
  },
  "corrections": [{"field": "item.name", "before": "잠치김밥", "after": "참치김밥", "reason": "ocr"}],
  "ocr_lines": [{"text": "...", "confidence": 0.98}]
}
```

## 참고: 맥북 로컬 개발

맥에서는 환경변수 없이 `.venv/bin/python server.py`를 실행하면
자동으로 MLX 백엔드(Qwen2.5-VL-3B 4bit)로 동작합니다.
