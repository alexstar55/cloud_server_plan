#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"

DO_PRECHECK="${DO_PRECHECK:-1}"
DO_SUMMARY="${DO_SUMMARY:-1}"
DO_WATCHDOG="${DO_WATCHDOG:-1}"
WATCHDOG_INTERVAL_SEC="${WATCHDOG_INTERVAL_SEC:-600}"
WATCHDOG_MAX_LOOPS="${WATCHDOG_MAX_LOOPS:-0}"

RUN_CONVERT_BATCH="${RUN_CONVERT_BATCH:-0}"
RUN_MERGE="${RUN_MERGE:-0}"
RUN_PSEUDOLABEL="${RUN_PSEUDOLABEL:-0}"

BATCH_FILE="${BATCH_FILE:-}"
DATASETS_LIST="${DATASETS_LIST:-}"
OUT_DIR="${OUT_DIR:-}"
PSEUDOLABEL_CMD="${PSEUDOLABEL_CMD:-}"

mkdir -p "$NIGHTLY_DIR/logs/run_all" "$NIGHTLY_DIR/reports"
LOG_FILE="$NIGHTLY_DIR/logs/run_all/run_all_$(date +%F_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

retry_once() {
  local cmd="$1"
  if bash -lc "$cmd" >>"$LOG_FILE" 2>&1; then
    return 0
  fi
  log "WARN first attempt failed, retry once: $cmd"
  sleep 30
  bash -lc "$cmd" >>"$LOG_FILE" 2>&1
}

WATCHDOG_PID=""
cleanup() {
  if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    log "INFO stopping watchdog pid=$WATCHDOG_PID"
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "INFO nightly_run_all start"

if [[ "$DO_PRECHECK" == "1" ]]; then
  log "INFO run precheck"
  bash "$SCRIPT_DIR/nightly_precheck.sh" >>"$LOG_FILE" 2>&1
fi

if [[ "$DO_WATCHDOG" == "1" ]]; then
  log "INFO start watchdog"
  INTERVAL_SEC="$WATCHDOG_INTERVAL_SEC" MAX_LOOPS="$WATCHDOG_MAX_LOOPS" \
    bash "$SCRIPT_DIR/nightly_watchdog.sh" >>"$LOG_FILE" 2>&1 &
  WATCHDOG_PID="$!"
fi

if [[ "$RUN_CONVERT_BATCH" == "1" ]]; then
  [[ -n "$BATCH_FILE" ]] || { log "ERROR RUN_CONVERT_BATCH=1 but BATCH_FILE is empty"; exit 1; }
  log "INFO run nightly_convert_nuscenes_batch"
  retry_once "bash '$SCRIPT_DIR/nightly_convert_nuscenes_batch.sh' '$BATCH_FILE'"
fi

if [[ "$RUN_MERGE" == "1" ]]; then
  [[ -n "$DATASETS_LIST" && -n "$OUT_DIR" ]] || { log "ERROR RUN_MERGE=1 but DATASETS_LIST/OUT_DIR is empty"; exit 1; }
  log "INFO run nightly_merge_nuscenes"
  retry_once "bash '$SCRIPT_DIR/nightly_merge_nuscenes.sh' '$DATASETS_LIST' '$OUT_DIR'"
fi

if [[ "$RUN_PSEUDOLABEL" == "1" ]]; then
  [[ -n "$PSEUDOLABEL_CMD" ]] || { log "ERROR RUN_PSEUDOLABEL=1 but PSEUDOLABEL_CMD is empty"; exit 1; }
  log "INFO run pseudo-label extension command"
  retry_once "$PSEUDOLABEL_CMD"
fi

if [[ "$DO_SUMMARY" == "1" ]]; then
  log "INFO run morning summary"
  bash "$SCRIPT_DIR/morning_summary.sh" >>"$LOG_FILE" 2>&1 || true
fi

log "INFO nightly_run_all done"
log "INFO log file: $LOG_FILE"
