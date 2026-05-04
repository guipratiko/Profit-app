#!/bin/sh
set -eu

if [ -n "${PROFIT_APP_STORAGE_DIR:-}" ]; then
  mkdir -p "${PROFIT_APP_STORAGE_DIR}"
  if [ -d "/app/storage_seed" ]; then
    cp -an /app/storage_seed/. "${PROFIT_APP_STORAGE_DIR}/" 2>/dev/null || true
  fi
fi

exec uvicorn app.api:app --host 0.0.0.0 --port "${PORT:-8000}"