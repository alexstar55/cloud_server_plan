#!/usr/bin/env bash
set -euo pipefail

NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"

echo "==== Morning Summary ===="

echo "[1] Recent status files"
ls -lt "$NIGHTLY_DIR/reports" | head -n 20 || true

printf "\n[2] Latest convert status\n"
LATEST_STATUS=$(ls -t "$NIGHTLY_DIR"/reports/convert_status_*.csv 2>/dev/null | head -n 1 || true)
if [[ -n "${LATEST_STATUS:-}" ]]; then
  echo "status_file=$LATEST_STATUS"
  tail -n +1 "$LATEST_STATUS"
else
  echo "no convert status file"
fi

printf "\n[3] Container state\n"
if command -v docker >/dev/null 2>&1; then
  docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" | head -n 30 || true
else
  echo "docker not found"
fi

printf "\n[4] GPU snapshot\n"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
else
  echo "nvidia-smi not found"
fi

printf "\n[5] Artifact count\n"
find "$NIGHTLY_DIR/artifacts" -type f 2>/dev/null | wc -l

printf "\n[6] Error keywords in recent logs\n"
find "$NIGHTLY_DIR/logs" -type f -mtime -1 2>/dev/null | xargs -r grep -E "ERROR|Traceback|OOM|Killed|AssertionError" -n | tail -n 80 || true

echo "==== Summary Done ===="
