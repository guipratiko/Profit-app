FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROFIT_APP_STORAGE_DIR=/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY storage ./storage_seed
COPY scripts/start_backend.sh ./scripts/start_backend.sh

RUN chmod +x ./scripts/start_backend.sh

EXPOSE 8000

CMD ["./scripts/start_backend.sh"]