"""VLM 백엔드 추상화.

- MlxVlm: 맥북(Apple Silicon) 로컬 개발용. mlx-vlm으로 직접 추론.
- OpenAIVlm: 홈서버(3090) 운영용. vLLM의 OpenAI 호환 API를 호출.

선택은 환경변수 VLM_BACKEND(mlx|openai)로 하고, 미설정 시 플랫폼 자동 감지.
"""
import base64
import os
import sys
import threading

import httpx



# 품목이 많은 영수증은 800 토큰 안에 JSON 이 안 끝난다. 실측에서 품목 20개짜리
# 하나로마트 영수증이 정확히 여기서 잘려 JSON 파싱에 실패했고, 응답이 통째로
# ok=false 로 떨어졌다. 그 영수증의 출력이 약 800 토큰이라 여유를 둬 2048 로 올린다.
# vLLM 의 --max-model-len 8192 안이라 컨텍스트에는 영향이 없다.
_MAX_TOKENS = 2048

class MlxVlm:
    def __init__(self, model_id: str):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"VLM 로드 중 (MLX): {model_id}")
        self._model, self._processor = load(model_id)
        self._config = load_config(model_id)
        self._lock = threading.Lock()

    def generate(self, image_path: str, prompt: str, max_tokens: int = _MAX_TOKENS) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(self._processor, self._config, prompt, num_images=1)
        with self._lock:
            output = generate(
                self._model,
                self._processor,
                formatted,
                image=[image_path],
                max_tokens=max_tokens,
                temperature=0.0,
                verbose=False,
            )
        return output.text if hasattr(output, "text") else output


class OpenAIVlm:
    """vLLM 등 OpenAI 호환 서버 호출. 동시 요청 처리는 서버(vLLM 배칭)에 맡긴다."""

    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=timeout)
        print(f"VLM 백엔드 (OpenAI 호환): {self._base_url} / {model}")

    def generate(self, image_path: str, prompt: str, max_tokens: int = _MAX_TOKENS) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        resp = self._client.post(f"{self._base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def create_vlm():
    backend = os.getenv("VLM_BACKEND")
    if not backend:
        backend = "mlx" if sys.platform == "darwin" else "openai"

    if backend == "mlx":
        model_id = os.getenv("VLM_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
        return MlxVlm(model_id)
    if backend == "openai":
        base_url = os.getenv("VLM_API_BASE", "http://localhost:8000/v1")
        model = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        return OpenAIVlm(base_url, model)
    raise ValueError(f"지원하지 않는 VLM_BACKEND: {backend}")
