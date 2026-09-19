#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 && $# -ne 5 ]]; then
  echo "Usage: $0 <clean-data-root> <output-root> <mvtecad-nlt|visa-nlt> [<pareto|step_k4|step_k1> <seed01..seed05>]" >&2
  echo "With three arguments, all three settings and all five data constructions are built." >&2
  exit 2
fi

clean_root="$1"
output_root="$2"
benchmark="$3"

case "$benchmark" in
  mvtecad-nlt) dataset_prefix="mvtecad" ;;
  visa-nlt) dataset_prefix="visa" ;;
  *) echo "benchmark must be mvtecad-nlt or visa-nlt" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 5 ]]; then
  settings=("$4")
  seeds=("$5")
else
  settings=(pareto step_k4 step_k1)
  seeds=(seed01 seed02 seed03 seed04 seed05)
fi

for setting in "${settings[@]}"; do
  case "$setting" in
    pareto|step_k4|step_k1) ;;
    *) echo "setting must be pareto, step_k4, or step_k1" >&2; exit 2 ;;
  esac
done

for seed in "${seeds[@]}"; do
  case "$seed" in
    seed01|seed02|seed03|seed04|seed05) ;;
    *) echo "seed must be seed01, seed02, seed03, seed04, or seed05" >&2; exit 2 ;;
  esac
done

total=$(( ${#settings[@]} * ${#seeds[@]} ))
current=0

for setting in "${settings[@]}"; do
  for seed in "${seeds[@]}"; do
    current=$((current + 1))
    manifest_root="${repo_root}/manifests/${benchmark}/${setting}/${seed}"
    prune_manifest="${manifest_root}/prune_good.txt"
    noisy_manifest="${manifest_root}/inject_defects.txt"
    dest_dir="${output_root}/${dataset_prefix}-${setting}-${seed}"

    if [[ ! -f "$prune_manifest" || ! -f "$noisy_manifest" ]]; then
      echo "Missing manifests under: $manifest_root" >&2
      exit 1
    fi

    echo "[$current/$total] Building ${dataset_prefix}-${setting}-${seed}"
    python "${repo_root}/tools/make_mvtecad_nlt.py" \
      --source-dir "$clean_root" \
      --dest-dir "$dest_dir" \
      --prune-manifest "$prune_manifest" \
      --noisy-manifest "$noisy_manifest" \
      --symlink-all
  done
done

echo "Built $total dataset construction(s) under: $output_root"
