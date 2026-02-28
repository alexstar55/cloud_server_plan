#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash nightly_convert_nuscenes_batch.sh /path/to/nightly_batches.tsv
# TSV列：
# batch_name<TAB>input_json<TAB>output_dir<TAB>camera_calib<TAB>localization<TAB>seq_data<TAB>pathmap<TAB>sweep_root(optional)

BATCH_FILE="${1:-}"
[[ -n "$BATCH_FILE" ]] || { echo "Usage: $0 <nightly_batches.tsv>"; exit 1; }
[[ -f "$BATCH_FILE" ]] || { echo "[ERROR] batch file not found: $BATCH_FILE"; exit 1; }

ROOT_DIR="${ROOT_DIR:-$HOME/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/custom_data_to_nuscenes_trans_scripts}"
NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"
USE_ZERO_POSE="${USE_ZERO_POSE:-1}"
USE_SWEEPS="${USE_SWEEPS:-0}"
SWEEP_WINDOW="${SWEEP_WINDOW:-2}"
SWEEP_MAX_DT_US="${SWEEP_MAX_DT_US:-500000}"

mkdir -p "$NIGHTLY_DIR/logs/convert" "$NIGHTLY_DIR/reports"
STATUS_CSV="$NIGHTLY_DIR/reports/convert_status_$(date +%F_%H%M%S).csv"
echo "batch_name,stage,status,message" > "$STATUS_CSV"

run_one_batch() {
  local batch_name="$1"
  local input_json="$2"
  local output_dir="$3"
  local camera_calib="$4"
  local localization="$5"
  local seq_data="$6"
  local pathmap="$7"
  local sweep_root="${8:-}"

  local log_prefix="$NIGHTLY_DIR/logs/convert/${batch_name}_$(date +%F_%H%M%S)"

  echo "==== [${batch_name}] start ===="

  # Stage A: localization检查
  if python3 "$PROJECT_DIR/check_scripts/check_localization_csv_standalone.py" "$localization" \
      > "${log_prefix}_check_localization.log" 2>&1; then
    echo "${batch_name},check_localization,ok,passed" >> "$STATUS_CSV"
  else
    echo "${batch_name},check_localization,failed,see_log" >> "$STATUS_CSV"
    return 1
  fi

  # Stage B: 转NuScenes
  local cmd=(python3 "$PROJECT_DIR/convert_custom_to_nuscenes.py"
    --input "$input_json"
    --output "$output_dir"
    --camera_calib "$camera_calib"
    --localization "$localization"
    --seq_data "$seq_data"
    --pathmap "$pathmap")

  if [[ "$USE_ZERO_POSE" == "1" ]]; then
    cmd+=(--zero_pose)
  fi

  if [[ "$USE_SWEEPS" == "1" ]]; then
    cmd+=(--use_sweeps --sweep_window "$SWEEP_WINDOW" --sweep_max_dt_us "$SWEEP_MAX_DT_US")
    [[ -n "$sweep_root" ]] && cmd+=(--sweep_root "$sweep_root")
  fi

  if "${cmd[@]}" > "${log_prefix}_convert.log" 2>&1; then
    echo "${batch_name},convert,ok,passed" >> "$STATUS_CSV"
  else
    echo "${batch_name},convert,failed,see_log" >> "$STATUS_CSV"
    return 1
  fi

  # Stage C: 转后检查
  local dataroot version
  dataroot="$(dirname "$output_dir")"
  version="$(basename "$output_dir")"

  if python3 "$PROJECT_DIR/check_scripts/nuscenes_data_check_v0.2static_check.py" \
      --dataroot "$dataroot" --version "$version" --static_speed_check \
      > "${log_prefix}_check_nuscenes.log" 2>&1; then
    echo "${batch_name},check_nuscenes,ok,passed" >> "$STATUS_CSV"
  else
    echo "${batch_name},check_nuscenes,failed,see_log" >> "$STATUS_CSV"
    return 1
  fi

  echo "==== [${batch_name}] done ===="
  return 0
}

# 跳过空行和注释行
while IFS=$'\t' read -r batch_name input_json output_dir camera_calib localization seq_data pathmap sweep_root; do
  [[ -z "${batch_name:-}" ]] && continue
  [[ "${batch_name:0:1}" == "#" ]] && continue

  if run_one_batch "$batch_name" "$input_json" "$output_dir" "$camera_calib" "$localization" "$seq_data" "$pathmap" "$sweep_root"; then
    :
  else
    echo "[WARN] batch failed: $batch_name (继续下一个)"
  fi
done < "$BATCH_FILE"

echo "Status file: $STATUS_CSV"
