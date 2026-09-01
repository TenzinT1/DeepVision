#!/usr/bin/env bash
# Stop the DeepVision backend + frontend started by ./start.sh
cd "$(dirname "$0")"
for f in .backend.pid .frontend.pid; do
  if [ -f "$f" ]; then
    kill "$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
  fi
done
# Belt-and-suspenders in case PIDs are stale:
pkill -f "uvicorn deepvision.api.main" 2>/dev/null || true
pkill -f "vite.*5173" 2>/dev/null || true
echo "DeepVision stopped."
