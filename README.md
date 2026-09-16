# TailGuard

Official implementation of **TailGuard: Progressive Reference Eligibility
Determination for Long-Tailed Noisy Industrial Anomaly Detection**.

TailGuard addresses unified industrial anomaly detection when an unlabeled
training set is both class imbalanced and contaminated. Instead of treating an
initial tail partition as a final decision, TailGuard progressively determines
whether uncertain samples may supervise reconstruction, whether their patch
features may enter normal memory, and whether that memory may support a test
query.

![TailGuard pipeline](assets/pipeline.png)

## Method

TailGuard consists of three dependent stages:

- **Dynamic Head Purification (DHP)** uses early reconstruction dynamics to identify and remove contamination from head candidates.
- **Tail Reconciliation and Preservation (TRP)** protects initial tail candidates and reconciles their assignments against the purified head structure.
- **Conditional Tail Enhancement (CTE)** constructs memories from the reconciled tail groups and applies their evidence only to matched queries.

The paper configuration uses a frozen DINOv2 ViT-B/14 encoder and the Dinomaly
reconstruction core. The canonical TailGuard parameters are centralized in
`tailguard/config.py`; selecting `--tg_method_mode full` runs the complete
method.

## Repository structure

```text
.
├── assets/                     # Pipeline and repository illustrations
├── scripts/                    # Convenience launchers
├── tailguard/
│   ├── cli/                    # Training and no-retraining replay entry points
│   ├── data/                   # Dataset readers, profiles, and manifest parsing
│   ├── engine/                 # Reconstruction training and evaluation
│   ├── method/                 # Tail sampling, DHP, TRP, and CTE
│   ├── models/                 # Encoder, decoder, and reconstruction model
│   ├── optimizers/             # Optimizer used by the paper configuration
│   ├── reporting/              # Artifact writers and post-hoc diagnostics
│   └── vendor/dinov2/          # Required DINOv2 architecture definitions
├── tools/                      # Dataset conversion, construction, and score fusion
└── requirements.txt
```

Datasets, pretrained weights, checkpoints, and generated results are not
stored in the repository. The DINOv2 checkpoint is downloaded from the
upstream host on first use. At that time, `backbones/weights/` is created
automatically as a local cache.

## Environment

The paper experiments used Python 3.8.12, PyTorch 1.12.0, and CUDA 11.3.

```bash
conda create -n tailguard python=3.8.12
conda activate tailguard
pip install -r requirements.txt
```

Run all commands below from the repository root.

## Data preparation

TailGuard expects an MVTec AD compatible directory layout:

```text
dataset_root/
└── class_name/
    ├── train/good/
    ├── test/good/
    ├── test/defect_type/
    └── ground_truth/defect_type/
```

### MVTec AD

To construct a long-tailed noisy dataset, prepare two text manifests:

- `prune_good.txt` lists normal training images to remove;
- `inject_defects.txt` lists anomalous images to insert into `train/good`.

Each entry is a path relative to the constructed dataset root. Build the
dataset with:

```bash
python tools/make_mvtecad_nlt.py \
  --source-dir /path/to/mvtec_anomaly_detection \
  --dest-dir /path/to/mvtecad-step_k4-seed01 \
  --prune-manifest /path/to/prune_good.txt \
  --noisy-manifest /path/to/inject_defects.txt \
  --symlink-all
```

`--symlink-all` avoids duplicating the source images. The clean source dataset
must remain available while the constructed dataset is in use. Omit this flag
to materialize a copy instead.

The convenience wrapper `scripts/build_one_dataset.sh` can also be used when
the manifests are organized as:

```text
manifests/<mvtecad-nlt|visa-nlt>/<pareto|step_k4|step_k1>/<seed01..seed05>/
├── prune_good.txt
└── inject_defects.txt
```

### VisA

Convert the original VisA release to the same directory layout first:

```bash
python tools/convert_visa_to_mvtec_format.py \
  --source_dir /path/to/visa \
  --target_dir /path/to/visa_mvtec
```

Then apply `tools/make_mvtecad_nlt.py` with the corresponding VisA manifests.

## Training

The complete MVTec AD configuration can be launched with:

```bash
python -m tailguard.cli.train \
  --dataset_profile mvtec \
  --data_path /path/to/mvtecad-step_k4-seed01 \
  --save_dir ./saved_results \
  --save_name mvtec_step_k4_seed01_full \
  --tg_method_mode full \
  --gpus 0
```

Use `--dataset_profile visa` for VisA. Training defaults to 10,000 iterations
and writes checkpoints, resolved configuration, intermediate assignments, and
evaluation results under `save_dir/save_name`.

If the injected-anomaly manifest is available, it may be supplied for
post-hoc auditing:

```bash
--diag_manifest_path /path/to/inject_defects.txt
```

The contamination labels loaded from this manifest are used only by reporting
and diagnostic code; they do not affect TailGuard decisions. With a manifest,
the equivalent convenience launcher is:

```bash
bash scripts/train_one.sh \
  mvtec \
  /path/to/mvtecad-step_k4-seed01 \
  /path/to/inject_defects.txt \
  ./saved_results \
  mvtec_step_k4_seed01_full \
  0
```

## Dual-reference image evaluation

The complete run writes reconciled CTE scores to
`tailguard/memory/memory_eval_scores.csv` inside the run directory. A coverage
reference can be rebuilt from the same trained checkpoint without updating
model parameters:

```bash
python -m tailguard.cli.coverage_replay \
  --full_run_dir ./saved_results/mvtec_step_k4_seed01_full \
  --data_path /path/to/mvtecad-step_k4-seed01 \
  --output_dir ./saved_results/mvtec_step_k4_seed01_full_coverage \
  --feature_cache_dir ./feature_cache/mvtec \
  --gpu 0
```

Fuse the coverage and reconciled image scores with:

```bash
python tools/fuse_dual_scores.py \
  --coverage-scores ./saved_results/mvtec_step_k4_seed01_full_coverage/coverage_eval_scores.csv \
  --reconciled-scores ./saved_results/mvtec_step_k4_seed01_full/tailguard/memory/memory_eval_scores.csv \
  --output-dir ./saved_results/mvtec_step_k4_seed01_full_dual
```

To additionally report head and tail subsets, pass the optional
`--class-roles` argument with a CSV containing `class_name` and `is_gt_tail`.
Pixel localization uses the reconciled final anomaly map, whereas the two
reference views are averaged for the final image score.

Full training and coverage replay require a CUDA GPU.

## Citation

The citation will be added after the paper becomes publicly available.

## Acknowledgments

TailGuard is implemented on top of
[Dinomaly](https://github.com/guojiajeremy/Dinomaly) and uses architecture
definitions from [DINOv2](https://github.com/facebookresearch/dinov2).
