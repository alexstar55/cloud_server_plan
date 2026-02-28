#!/usr/bin/env bash
set -euo pipefail

echo "==== Nightly Precheck ===="

ROOT_DIR="${ROOT_DIR:-$HOME/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR/custom_data_to_nuscenes_trans_scripts}"
NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/nightly}"

mkdir -p "$NIGHTLY_DIR" "$NIGHTLY_DIR/logs" "$NIGHTLY_DIR/reports" "$NIGHTLY_DIR/artifacts"

echo "[1/6] GPU & Driver"
nvidia-smi

echo "[2/6] Docker"
docker ps >/dev/null
docker images | grep -E "alpasim|sparsedrive|diffusion_policy" || true

echo "[3/6] Disk/Memory"
df -h
free -h

echo "[4/6] Converter scripts"
for f in \
  "$PROJECT_DIR/convert_custom_to_nuscenes.py" \
  "$PROJECT_DIR/merge_nuscenes.py" \
  "$PROJECT_DIR/check_scripts/check_localization_csv_standalone.py" \
  "$PROJECT_DIR/check_scripts/nuscenes_data_check_v0.2static_check.py"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing file: $f"; exit 1; }
  echo "  - OK: $f"
done

echo "[5/6] Python"
python3 -V

echo "[6/6] Write heartbeat"
date '+%F %T precheck ok' >> "$NIGHTLY_DIR/reports/heartbeat.log"

echo "==== Precheck Passed ===="
