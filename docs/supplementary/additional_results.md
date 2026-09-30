# Additional results

All entries are percentages. The labels **head** and **tail** are used only for
post-training evaluation and are not available to TailGuard.

## 1. Head, tail, and overall results

The main paper reports overall image-level area under the receiver operating
characteristic curve (I-AUROC) and pixel-level AUROC (P-AUROC). The following
tables separate the same evaluation into tail, head, and all-class subsets.

### I-AUROC

| Method | MVTec Pareto | MVTec K4 | MVTec K1 | VisA Pareto | VisA K4 | VisA K1 |
|---|---:|---:|---:|---:|---:|---:|
| TailedCore | 96.55 / 95.24 / 96.13 | 95.82 / 95.34 / 95.72 | 93.54 / 95.77 / 94.44 | 87.55 / 93.06 / 90.86 | 85.16 / 95.91 / 89.65 | **82.97** / 94.11 / **87.61** |
| **TailGuard** | **97.57** / **95.55** / **96.66** | **97.11** / 94.88 / **96.22** | **95.65** / 94.87 / **95.34** | **93.62** / 92.67 / **92.87** | **88.07** / **95.25** / **91.06** | 81.02 / **95.91** / 87.23 |

Each cell is ordered as **tail / head / all**. Across the six settings,
TailGuard obtains 92.17, 94.86, and 93.23 I-AUROC on tail, head, and all
classes. The tail mean is 1.91 points higher than TailedCore.

### P-AUROC

| Method | MVTec Pareto | MVTec K4 | MVTec K1 | VisA Pareto | VisA K4 | VisA K1 |
|---|---:|---:|---:|---:|---:|---:|
| TailedCore | 96.08 / 95.01 / 95.30 | 95.56 / 93.20 / 94.75 | 94.19 / 93.70 / 94.00 | **97.98** / **97.25** / **97.49** | **96.80** / 97.02 / 96.90 | **96.12** / 97.39 / **96.65** |
| **TailGuard** | **97.76** / **95.08** / **96.58** | **96.17** / **97.27** / **96.61** | **94.42** / **97.40** / **95.61** | 97.96 / 97.20 / 97.45 | 95.77 / **98.83** / **97.05** | 94.05 / **98.69** / 95.99 |

TailGuard obtains mean P-AUROC values of 96.02, 97.41, and 96.55 on tail,
head, and all classes. Its corresponding pixel-level area under the
per-region-overlap curve (P-AUPRO) values are 88.63, 92.47, and 90.11.

Machine-readable values: [`data/head_tail_results.csv`](data/head_tail_results.csv).

## 2. Complete prediction-distinct component comparison

The component comparison follows the method dependencies. TRP changes the
role of retained candidates but neither removes training samples nor changes
the reconstruction score. Consequently, DHP followed by TRP without CTE is
prediction-equivalent to DHP alone. CTE without TRP uses the initial tail
candidates directly; the full method uses the reconciled references. Rows that
violate a dependency or duplicate another prediction are not treated as
separate ablations.

### I-AUROC

| Variant | MVTec P | MVTec K4 | MVTec K1 | VisA P | VisA K4 | VisA K1 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reconstruction core | 93.32 | 89.08 | 88.08 | 91.71 | 90.84 | 89.01 | 90.34 |
| + DHP | 96.86 | 95.82 | 93.18 | **93.83** | 91.43 | 89.81 | 93.49 |
| + DHP + CTE, without TRP | 96.96 | **95.86** | **96.39** | 93.70 | 91.89 | **92.26** | 94.51 |
| **TailGuard** | **97.08** | **95.86** | 96.36 | 93.72 | **91.97** | 92.21 | **94.53** |

### P-AUPRO

| Variant | MVTec P | MVTec K4 | MVTec K1 | VisA P | VisA K4 | VisA K1 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reconstruction core | 90.08 | 88.86 | 86.07 | 92.31 | 90.21 | 84.04 | 88.60 |
| + DHP | **92.44** | 91.53 | 88.42 | **92.77** | 89.43 | 84.74 | 89.89 |
| + DHP + CTE, without TRP | 89.84 | 88.70 | 89.56 | 92.39 | 90.13 | **89.26** | 89.98 |
| **TailGuard** | 92.32 | **92.05** | **90.63** | 92.69 | **90.21** | 87.62 | **90.92** |

DHP raises the mean I-AUROC by 3.15 points and P-AUPRO by 1.29 points. CTE
applied to the initial candidates improves image ranking, while TRP raises the
complete method's mean P-AUPRO from 89.98 to 90.92.

Machine-readable values: [`data/component_ablation.csv`](data/component_ablation.csv).

## 3. Prediction equivalence before enhancement

The following check explains why DHP with TRP but without CTE is not displayed
as a separate prediction row. The initial partitions, head groups, selected
checkpoints, and removed sample identities are identical. The maximum
difference over seven reconstruction metrics is at numerical precision.

| Setting | Selected iteration | Initial head candidates | Removed | Maximum metric difference |
|---|---:|---:|---:|---:|
| MVTec Pareto | 980 | 730 | 67 | 5.25e-8 |
| MVTec K4 | 1140 | 1,570 | 150 | 2.53e-9 |
| MVTec K1 | 1140 | 1,857 | 180 | 1.92e-7 |
| VisA Pareto | 200 | 2,022 | 191 | 5.15e-9 |
| VisA K4 | 1500 | 4,263 | 383 | 1.20e-8 |
| VisA K1 | 1360 | 4,261 | 393 | 1.21e-9 |

TRP affects detection through subsequent tail grouping and memory construction,
not by changing the reconstruction model.

## 4. Internal reference views in CTE

The coverage and reconciled views are internal configurations of CTE rather
than separate proposed modules.

| Configuration | I-AUROC | P-AUPRO |
|---|---:|---:|
| Coverage view only | 94.51 | 89.98 |
| Reconciled view only | 94.16 | **90.92** |
| Combined views | **94.53** | **90.92** |

The coverage view preserves broad candidate evidence for image ranking. The
reconciled view supplies the stricter local reference set used by the final
anomaly map. Their fixed combination improves the mean image score without
changing the reconciled localization result.

## 5. CTE storage cost

The following estimates count float32 feature payload only. The fixed limit of
20,000 patch features per tail group bounds the retrieval state.

| Setting | Coverage / reconciled banks | Coverage / reconciled patch features | Total MiB |
|---|---:|---:|---:|
| MVTec Pareto | 15 / 2 | 63,504 / 9,408 | 213.6 |
| MVTec K4 | 24 / 17 | 42,336 / 28,224 | 206.7 |
| MVTec K1 | 17 / 8 | 21,952 / 6,272 | 82.7 |
| VisA Pareto | 5 / 4 | 57,632 / 37,632 | 279.1 |
| VisA K4 | 9 / 5 | 21,168 / 12,544 | 98.8 |
| VisA K1 | 7 / 4 | 6,272 / 3,136 | 27.6 |

Machine-readable values: [`data/cte_storage.csv`](data/cte_storage.csv).

## 6. Qualitative sample scores

The main paper shows representative anomaly maps. The numerical values below
are pixel average precision computed from the original score maps, not from
display-normalized heatmaps.

| Dataset | Example | SoftPatch | TailedCore | TailGuard |
|---|---|---:|---:|---:|
| MVTec K4 | Bottle | 10.10 | 65.04 | **95.92** |
| MVTec K4 | Grid, bent | 1.79 | 47.18 | **49.38** |
| MVTec K4 | Grid, thread | 1.43 | 50.45 | **52.43** |
| MVTec K4 | Metal Nut | 13.82 | 79.62 | **95.30** |
| MVTec K4 | Pill | 44.77 | 45.87 | **72.12** |
| MVTec K4 | Tile | 73.11 | 70.87 | **96.85** |
| VisA K4 | Capsules | — | 26.90 | **92.60** |
| VisA K4 | Chewing Gum | — | 46.00 | **69.20** |

Machine-readable values: [`data/qualitative_pixel_ap.csv`](data/qualitative_pixel_ap.csv).

## 7. Variability across independent data constructions

This table first averages over classes inside each construction, then reports
the mean and sample standard deviation of the five construction-level values.
It therefore measures construction-to-construction stability. This differs
from the category-level dispersion displayed in the main comparison, which
also contains differences in class difficulty.

| Metric | MVTec P | MVTec K4 | MVTec K1 | VisA P | VisA K4 | VisA K1 | Six-setting mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| I-AUROC | 96.66 ± 0.72 | 96.22 ± 0.42 | 95.34 ± 1.21 | 92.87 ± 0.63 | 91.06 ± 2.12 | 87.23 ± 4.81 | 93.23 ± 1.08 |
| P-AUROC | 96.58 ± 0.52 | 96.61 ± 0.61 | 95.61 ± 1.34 | 97.45 ± 0.90 | 97.05 ± 1.15 | 95.99 ± 1.27 | 96.55 ± 0.29 |
| P-AUPRO | 92.47 ± 0.44 | 91.21 ± 1.71 | 90.80 ± 1.61 | 90.61 ± 2.86 | 89.26 ± 3.00 | 86.34 ± 1.68 | 90.11 ± 0.63 |

Machine-readable values: [`data/construction_variability.csv`](data/construction_variability.csv).
