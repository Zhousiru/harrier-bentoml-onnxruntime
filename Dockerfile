FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY service.py ./

EXPOSE 3000

CMD ["bentoml", "serve", "service:HarrierEmbeddingService", "--host", "0.0.0.0", "--port", "3000"]
