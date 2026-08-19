FROM python:3.12-slim

# rapidocr(opencv)가 요구하는 시스템 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY server.py vlm.py corrector.py ./
COPY static ./static

ENV VLM_BACKEND=openai
EXPOSE 8600

CMD ["python", "server.py"]
