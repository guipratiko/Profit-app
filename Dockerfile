# syntax=docker/dockerfile:1
# Easypanel pode enviar contexto de build vazio. O código vem de clone Git
# (build args opcionais: GIT_REPO, GIT_REF, GIT_SHA).

FROM alpine:3.20 AS src
ARG GIT_REPO=https://github.com/guipratiko/Profit-app.git
ARG GIT_REF=main
ARG GIT_SHA=undefined
RUN apk add --no-cache git ca-certificates
WORKDIR /repo
RUN set -eux \
    && if [ "${GIT_SHA}" != "undefined" ] && [ -n "${GIT_SHA}" ]; then \
         git clone -- "${GIT_REPO}" . \
         && git checkout --detach "${GIT_SHA}"; \
       else \
         git clone --depth 1 --branch "${GIT_REF}" -- "${GIT_REPO}" .; \
       fi

FROM python:3.11-slim

# Runtime (Easypanel → Environment): define DATABASE_URL ou PGHOST+PGUSER+PGPASSWORD+PGDATABASE
# (não coloques segredos aqui). Exemplos: ficheiro .env.example no repositório e em /app/.env.example na imagem.

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="profit-app-api" \
      org.opencontainers.image.revision="${GIT_SHA}"

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

COPY --from=src /repo/requirements.runtime.txt ./requirements.runtime.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.runtime.txt

COPY --from=src /repo/app ./app
COPY --from=src /repo/storage ./storage_seed
COPY --from=src /repo/.env.example ./.env.example

RUN mkdir -p scripts \
    && printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
        '' \
        'if [ -n "${PROFIT_APP_STORAGE_DIR:-}" ]; then' \
        '  mkdir -p "${PROFIT_APP_STORAGE_DIR}"' \
        '  if [ -d "/app/storage_seed" ]; then' \
        '    cp -an /app/storage_seed/. "${PROFIT_APP_STORAGE_DIR}/" 2>/dev/null || true' \
        '  fi' \
        'fi' \
        '' \
        'exec uvicorn app.api:app --host 0.0.0.0 --port "${PORT:-8000}"' \
        > scripts/start_backend.sh \
    && chmod +x scripts/start_backend.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import sys,urllib.request,os;p=os.environ.get('PORT','8000');urllib.request.urlopen('http://127.0.0.1:'+p+'/ready',timeout=5);sys.exit(0)"

CMD ["./scripts/start_backend.sh"]
