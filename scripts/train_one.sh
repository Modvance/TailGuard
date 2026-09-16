#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 <mvtec|visa> <data-path> <injection-manifest> <save-dir> <run-name> [gpu]" >&2
  exit 2
fi

profile="$1"
data_path="$2"
injection_manifest="$3"
save_dir="$4"
run_name="$5"
gpu="${6:-0}"

case "$profile" in
  mvtec|visa) ;;
  *) echo "profile must be mvtec or visa" >&2; exit 2 ;;
esac

python -m tailguard.cli.train \
  --dataset_profile "$profile" \
  --data_path "$data_path" \
  --diag_manifest_path "$injection_manifest" \
  --save_dir "$save_dir" \
  --save_name "$run_name" \
  --tg_method_mode full \
  --gpus "$gpu"
