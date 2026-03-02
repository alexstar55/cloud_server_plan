#!/usr/bin/env bash
set -euo pipefail

# 用途：用 nuScenes mini 做 AlpaSim 数据生成冒烟（先打通容器/挂载/日志/产物链路）
# 说明：
# 1) 默认优先尝试 alpasim_wizard（真实 AlpaSim 冒烟）
# 2) 若容器内无 alpasim_wizard，则自动回退到本目录的 Python 冒烟脚本
# 3) 如需完全自定义，可直接传入 ALPASIM_SMOKE_CMD

IMAGE="${IMAGE:-alpasim:v1}"
GPU_DEVICE="${GPU_DEVICE:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-alpasim_smoke}"

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/workspace/alpasim}"
NUSC_MINI_DIR="${NUSC_MINI_DIR:-/data/nu_mini}"
NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="${OUT_DIR:-$NIGHTLY_DIR/artifacts/alpasim_smoke}"
LOG_DIR="${LOG_DIR:-$NIGHTLY_DIR/logs/alpasim_smoke}"
SMOKE_MODE="${SMOKE_MODE:-wizard}"
WIZARD_LOG_DIR="${WIZARD_LOG_DIR:-/workspace/output/tutorial_smoke}"

# 可选：容器内执行的真实生成命令
# 若为空，将按 SMOKE_MODE 自动生成默认命令
# 例：ALPASIM_SMOKE_CMD='python tools/your_generate.py --data_root /data/nu_mini --version v1.0-mini --out /workspace/output'
ALPASIM_SMOKE_CMD="${ALPASIM_SMOKE_CMD:-}"

if [[ -z "$ALPASIM_SMOKE_CMD" ]]; then
  if [[ "$SMOKE_MODE" == "wizard" ]]; then
    ALPASIM_SMOKE_CMD="source /workspace/alpasim/setup_local_env.sh >/workspace/logs/setup_local_env.log 2>&1 || true; if command -v alpasim_wizard >/dev/null 2>&1; then alpasim_wizard +deploy=local wizard.log_dir=${WIZARD_LOG_DIR} runtime.default_scenario_parameters.n_rollouts=1; else python /workspace/smoke_tools/alpasim_nuscenes_mini_smoke.py --nuscenes-root /data/nu_mini --output-dir /workspace/output --tag fallback_no_wizard; fi"
  else
    ALPASIM_SMOKE_CMD="python /workspace/smoke_tools/alpasim_nuscenes_mini_smoke.py --nuscenes-root /data/nu_mini --output-dir /workspace/output --tag nuscenes_mini_only"
  fi
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
HOST_LOG="$LOG_DIR/smoke_$(date +%F_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$HOST_LOG"
}

log "start alpasim smoke with image=$IMAGE gpu=$GPU_DEVICE"

[[ -d "$WORKSPACE_DIR" ]] || { log "ERROR missing WORKSPACE_DIR: $WORKSPACE_DIR"; exit 1; }
[[ -d "$NUSC_MINI_DIR" ]] || { log "ERROR missing NUSC_MINI_DIR: $NUSC_MINI_DIR"; exit 1; }
[[ -f "$SCRIPT_DIR/alpasim_nuscenes_mini_smoke.py" ]] || { log "ERROR missing helper: $SCRIPT_DIR/alpasim_nuscenes_mini_smoke.py"; exit 1; }

command -v docker >/dev/null 2>&1 || { log "ERROR docker not found"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { log "ERROR nvidia-smi not found"; exit 1; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ERROR image not found: $IMAGE"; exit 1; }

# 防止重名容器冲突
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "INFO remove existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

log "GPU snapshot"
nvidia-smi | tee -a "$HOST_LOG" >/dev/null

log "run container smoke"
docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  --shm-size 32g \
  -v "$WORKSPACE_DIR":/workspace/alpasim \
  -v "$NUSC_MINI_DIR":/data/nu_mini \
  -v "$OUT_DIR":/workspace/output \
  -v "$LOG_DIR":/workspace/logs \
  -v "$SCRIPT_DIR":/workspace/smoke_tools \
  "$IMAGE" \
  bash -lc "set -euo pipefail; cd /workspace/alpasim; $ALPASIM_SMOKE_CMD" \
  2>&1 | tee -a "$HOST_LOG"

log "smoke done"
log "host log: $HOST_LOG"
log "output dir: $OUT_DIR"
