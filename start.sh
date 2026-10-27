#!/usr/bin/env bash
# Arranque del servicio web. Render inyecta PORT.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-10000}"
exec python -m uvicorn servidor_nhl:app --host 0.0.0.0 --port "$PORT"
