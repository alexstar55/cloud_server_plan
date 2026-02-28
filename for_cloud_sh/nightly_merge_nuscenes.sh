#!/usr/bin/env bash
set -euo pipefail

# 用法：
# bash nightly_merge_nuscenes.sh /path/to/datasets.list /path/to/output_dir
# datasets.list: 每行一个待合并目录（该目录内含 *.json）

DATASETS_LIST="${1:-}"
OUT_DIR="${2:-}"
[[ -n "$DATASETS_LIST" && -n "$OUT_DIR" ]] || { echo "Usage: $0 <datasets.list> <output_dir>"; exit 1; }
[[ -f "$DATASETS_LIST" ]] || { echo "[ERROR] not found: $DATASETS_LIST"; exit 1; }

ROOT_DIR="${ROOT_DIR:-$HOME/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/custom_data_to_nuscenes_trans_scripts}"
NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"

mkdir -p "$NIGHTLY_DIR/logs/merge" "$NIGHTLY_DIR/reports" "$OUT_DIR"
LOG_PREFIX="$NIGHTLY_DIR/logs/merge/merge_$(date +%F_%H%M%S)"

mapfile -t DATASETS < <(grep -v '^#' "$DATASETS_LIST" | sed '/^\s*$/d')
[[ ${#DATASETS[@]} -gt 0 ]] || { echo "[ERROR] datasets list empty"; exit 1; }

echo "[INFO] merging ${#DATASETS[@]} datasets"
python3 "$PROJECT_DIR/merge_nuscenes.py" --datasets "${DATASETS[@]}" --output "$OUT_DIR" \
  > "${LOG_PREFIX}_merge.log" 2>&1

# 可选：重写 sample_data.json 的绝对路径前缀
if [[ -n "${OLD_PREFIX:-}" && -n "${NEW_PREFIX:-}" ]]; then
  python3 "/mnt/ElementSE/AI_proj/plans/for_cloud_sh/rewrite_sample_data_paths.py" \
    --json "$OUT_DIR/sample_data.json" \
    --old-prefix "$OLD_PREFIX" \
    --new-prefix "$NEW_PREFIX" \
    --backup \
    > "${LOG_PREFIX}_rewrite.log" 2>&1
fi

# 合并后检查
DROOT="$(dirname "$OUT_DIR")"
VER="$(basename "$OUT_DIR")"
python3 "$PROJECT_DIR/check_scripts/nuscenes_data_check_v0.2static_check.py" \
  --dataroot "$DROOT" --version "$VER" --static_speed_check \
  > "${LOG_PREFIX}_check.log" 2>&1

echo "[OK] merge completed: $OUT_DIR"
echo "logs: ${LOG_PREFIX}_*.log"
