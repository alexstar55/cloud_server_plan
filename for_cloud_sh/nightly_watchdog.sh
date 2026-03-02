#!/usr/bin/env bash
set -euo pipefail

NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
MAX_LOOPS="${MAX_LOOPS:-0}"  # 0 = infinite

mkdir -p "$NIGHTLY_DIR/reports"

echo "[INFO] watchdog started: interval=${INTERVAL_SEC}s, max_loops=${MAX_LOOPS}" >&2

loop_count=0
while true; do
  ts="$(date '+%F %T')"
  echo "$ts" >> "$NIGHTLY_DIR/reports/heartbeat.log"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader \
      >> "$NIGHTLY_DIR/reports/gpu_heartbeat.log" 2>/dev/null || true
  fi

  if command -v docker >/dev/null 2>&1; then
    docker ps --format "{{.Names}}\t{{.Status}}" \
      | sed "s/^/$ts\t/" >> "$NIGHTLY_DIR/reports/container_heartbeat.log" 2>/dev/null || true
  fi

  loop_count=$((loop_count + 1))
  if [[ "$MAX_LOOPS" != "0" && "$loop_count" -ge "$MAX_LOOPS" ]]; then
    echo "[INFO] watchdog finished after $loop_count loops" >&2
    break
  fi

  sleep "$INTERVAL_SEC"
done
