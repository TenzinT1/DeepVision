#!/usr/bin/env bash
# Start DeepVision: backend (FastAPI :8000) + frontend (Vite :5173).
# Usage: ./start.sh   then open http://localhost:5173   (stop with ./stop.sh)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "ERROR: no .venv found. Create it and install deps first (see README.md)." >&2
  exit 1
fi

echo "Starting DeepVision backend (:8000)..."
nohup "$ROOT/.venv/bin/python" -m uvicorn deepvision.api.main:app \
  --host 127.0.0.1 --port 8000 > "$ROOT/backend.log" 2>&1 &
echo $! > "$ROOT/.backend.pid"

echo "Starting DeepVision frontend (:5173)..."
cd "$ROOT/frontend"
[ -d node_modules ] || npm install
nohup npm run dev -- --port 5173 --host 127.0.0.1 > "$ROOT/frontend.log" 2>&1 &
echo $! > "$ROOT/.frontend.pid"
cd "$ROOT"

echo ""
echo "  Backend:  http://localhost:8000/api/health   (logs: backend.log)"
echo "  Frontend: http://localhost:5173              (logs: frontend.log)"
echo ""
echo "Open http://localhost:5173 in your browser. Run ./stop.sh to stop."
