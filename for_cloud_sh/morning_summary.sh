#!/usr/bin/env bash
set -euo pipefail

NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"

echo "==== Morning Summary ===="

echo "[1] Recent status files"
ls -lt "$NIGHTLY_DIR/reports" | head -n 20 || true

echo "\n[2] Latest convert status"
LATEST_STATUS=$(ls -t "$NIGHTLY_DIR"/reports/convert_status_*.csv 2>/dev/null | head -n 1 || true)
if [[ -n "${LATEST_STATUS:-}" ]]; then
  echo "status_file=$LATEST_STATUS"
  tail -n +1 "$LATEST_STATUS"
else
  echo "no convert status file"
fi

echo "\n[3] Container state"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" | head -n 30

echo "\n[4] GPU snapshot"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "\n[5] Artifact count"
find "$NIGHTLY_DIR/artifacts" -type f 2>/dev/null | wc -l

echo "\n[6] Error keywords in recent logs"
find "$NIGHTLY_DIR/logs" -type f -mtime -1 2>/dev/null | xargs -r grep -E "ERROR|Traceback|OOM|Killed|AssertionError" -n | tail -n 80 || true

echo "==== Summary Done ===="
