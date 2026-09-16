#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <clean-data-root> <output-root> <mvtecad-nlt|visa-nlt> <pareto|step_k4|step_k1> <seed01..seed05>" >&2
  exit 2
fi

clean_root="$1"
output_root="$2"
benchmark="$3"
setting="$4"
seed="$5"

case "$benchmark" in
  mvtecad-nlt) dataset_prefix="mvtecad" ;;
  visa-nlt) dataset_prefix="visa" ;;
  *) echo "benchmark must be mvtecad-nlt or visa-nlt" >&2; exit 2 ;;
esac

manifest_root="manifests/${benchmark}/${setting}/${seed}"
python tools/make_mvtecad_nlt.py \
  --source-dir "$clean_root" \
  --dest-dir "${output_root}/${dataset_prefix}-${setting}-${seed}" \
  --prune-manifest "${manifest_root}/prune_good.txt" \
  --noisy-manifest "${manifest_root}/inject_defects.txt" \
  --symlink-all
