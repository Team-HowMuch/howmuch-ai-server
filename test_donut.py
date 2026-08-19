#!/usr/bin/env python3
"""Donut(cord-v2)로 한국 영수증 샘플에서 구조화 데이터 추출을 실측하는 스크립트."""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
SAMPLE_DIR = Path(__file__).parent / "samples"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[1/3] 모델 로드: {MODEL_ID} (device={device})")
processor = DonutProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(device).eval()
params = sum(p.numel() for p in model.parameters())
print(f"      파라미터: {params / 1e6:.0f}M")

images = sorted(SAMPLE_DIR.glob("*.JPEG")) + sorted(SAMPLE_DIR.glob("*.jpg"))
if not images:
    sys.exit("samples/ 폴더에 이미지가 없습니다.")

print(f"[2/3] 추론 시작: 이미지 {len(images)}장")
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids.to(device)

for path in images:
    image = Image.open(path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    elapsed = time.perf_counter() - start

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(task_prompt, "")
    result = processor.token2json(sequence)

    print(f"\n===== {path.name} ({elapsed:.1f}s) =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))

print("\n[3/3] 완료")
