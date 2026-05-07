FROM python:3.11-slim

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="profit-app-api" \
      org.opencontainers.image.revision="$GIT_SHA"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PORT=8000 \
    PROFIT_APP_STORAGE_DIR=/data \
    PROFIT_APP_SQLALCHEMY_NULLPOOL=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.runtime.txt ./requirements.runtime.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.runtime.txt

COPY app ./app
COPY storage ./storage_seed
COPY --chmod=755 scripts/start_backend.sh ./scripts/start_backend.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import json,os,sys,urllib.request;p=os.environ.get('PORT','8000');r=urllib.request.urlopen('http://127.0.0.1:'+p+'/health');d=json.load(r);sys.exit(0 if d.get('status')=='ok' else 1)"

CMD ["./scripts/start_backend.sh"]
