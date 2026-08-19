#!/usr/bin/env python3
"""Qwen2.5-VL(4bit, MLX)로 한국 영수증에서 구조화 데이터 추출을 실측하는 스크립트."""
import sys
import time
from pathlib import Path

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
SAMPLE_DIR = Path(__file__).parent / "samples"

PROMPT = """이 영수증 이미지를 읽고 아래 JSON 형식으로만 답해줘. 다른 설명은 쓰지 마.
{
  "store_name": "상호명",
  "purchased_at": "YYYY-MM-DD HH:MM",
  "items": [{"name": "품목명", "quantity": 1, "price": 0}],
  "total_amount": 0,
  "payment_method": "카드 또는 현금"
}
금액은 숫자만(콤마 없이) 적어줘. 읽을 수 없는 값은 null로 해줘."""

print(f"[1/2] 모델 로드: {MODEL_ID}")
model, processor = load(MODEL_ID)
config = load_config(MODEL_ID)

images = sorted(SAMPLE_DIR.glob("*.JPEG")) + sorted(SAMPLE_DIR.glob("*.jpg"))
if not images:
    sys.exit("samples/ 폴더에 이미지가 없습니다.")

print(f"[2/2] 추론 시작: 이미지 {len(images)}장")
for path in images:
    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
    start = time.perf_counter()
    output = generate(
        model,
        processor,
        formatted,
        image=[str(path)],
        max_tokens=800,
        temperature=0.0,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    text = output.text if hasattr(output, "text") else output
    print(f"\n===== {path.name} ({elapsed:.1f}s) =====")
    print(text)

print("\n완료")
