# Mechanism analysis

This page examines the decisions behind TailGuard rather than repeating the
final detection comparison. Ground-truth tail membership and injected anomaly
labels are introduced only after training for audit purposes.

## 1. Why imbalance and contamination must be studied jointly

We evaluate the same reconstruction detector under four training conditions:
balanced and clean, balanced and contaminated, long tailed and clean, and long
tailed and contaminated. For metric $Y$, we report the descriptive contrast

$$
\Delta_Y=Y_{\mathrm{LT,noise}}-Y_{\mathrm{LT,clean}}
-Y_{\mathrm{balanced,noise}}+Y_{\mathrm{balanced,clean}}.
$$

A negative value means that the degradation observed when imbalance and
contamination coexist is larger than the sum suggested by the two isolated
changes in this construction. Because long-tail construction also changes the
effective prevalence of injected samples, $\Delta_Y$ is a benchmark contrast,
not a causal estimate.

| Metric | Balanced clean | Balanced contaminated | Long-tail clean | Long-tail contaminated | $\Delta_Y$ | Negative settings |
|---|---:|---:|---:|---:|---:|---:|
| I-AUROC | 99.17 | 96.74 | 96.04 | 90.35 | -3.25 | 6 / 6 |
| Image AP | 99.33 | 97.87 | 97.04 | 93.89 | -1.68 | 6 / 6 |
| Image F1 | 97.60 | 95.04 | 94.36 | 89.93 | -1.87 | 6 / 6 |
| P-AUROC | 98.55 | 98.12 | 97.58 | 96.53 | -0.63 | 6 / 6 |
| Pixel AP | 61.15 | 56.29 | 54.83 | 49.30 | -0.67 | 4 / 6 |
| Pixel F1 | 62.41 | 58.11 | 56.95 | 51.67 | -0.97 | 4 / 6 |
| P-AUPRO | 94.59 | 93.47 | 91.83 | 88.57 | -2.14 | 6 / 6 |

For I-AUROC, contamination causes a 2.43-point loss under balanced sampling
and a 5.69-point loss under long-tailed sampling. The resulting contrast is
-3.25 points and is negative in all six settings. The same repeated pattern in
image AP, image F1, P-AUROC, and P-AUPRO supports evaluating the joint setting
rather than treating imbalance and contamination as independent benchmarks.

Machine-readable values: [`data/joint_setting_contrast.csv`](data/joint_setting_contrast.csv).

## 2. What the TailSampler score identifies

TailSampler estimates the relative frequency of the visual group surrounding a
sample. We separately test whether this score ranks benchmark tail membership
and injected contamination.

| Setting | Training samples | True tail / contamination | Tail AUROC | Tail AP | Contamination AUROC | Contamination AP |
|---|---:|---:|---:|---:|---:|---:|
| MVTec Pareto | 811 | 56 / 74 | 99.87 | 97.00 | 41.21 | 8.77 |
| MVTec K4 | 1,624 | 36 / 144 | 99.76 | 84.86 | 52.36 | 13.17 |
| MVTec K1 | 1,885 | 9 / 170 | 99.95 | 81.82 | 52.06 | 16.09 |
| VisA Pareto | 2,175 | 48 / 104 | 100.00 | 100.00 | 50.47 | 4.69 |
| VisA K4 | 4,290 | 28 / 202 | 95.22 | 76.47 | 50.20 | 5.06 |
| VisA K1 | 4,269 | 7 / 202 | 95.25 | 85.78 | 49.55 | 5.60 |
| **Mean / total** | **15,054** | **184 / 896** | **98.34** | **87.66** | **49.31** | **8.90** |

The score ranks tail membership at 98.34 AUROC but contamination at 49.31
AUROC. An initial tail assignment is therefore useful as a frequency proposal,
but it is not evidence that the image or all its patches are normal. This
distinction motivates protecting initial tail candidates while postponing
their eligibility for local memory.

Machine-readable values: [`data/tailsampler_audit.csv`](data/tailsampler_audit.csv).

## 3. Early reconstruction evidence in DHP

We compare three signals on the same initial head candidates at the checkpoint
selected by DHP:

- raw image reconstruction error;
- the error normalized by the median and median absolute deviation of its
  appearance group;
- the posterior probability of the higher-error Gaussian-mixture component.

AUROC and average precision measure retrospective contamination ranking. The
contamination labels do not select the checkpoint, mixture, or threshold.

| Setting | Selected iteration | Samples / contamination | Raw error AUROC / AP | Group-normalized AUROC / AP | GMM posterior AUROC / AP |
|---|---:|---:|---:|---:|---:|
| MVTec Pareto | 980 | 730 / 69 | 98.62 / 95.25 | 97.69 / 93.04 | 97.40 / 85.49 |
| MVTec K4 | 1140 | 1,570 / 126 | 91.16 / 68.20 | 96.22 / 81.04 | 94.58 / 79.92 |
| MVTec K1 | 1140 | 1,857 / 151 | 99.05 / 92.62 | 98.74 / 89.19 | 98.78 / 86.84 |
| VisA Pareto | 200 | 2,022 / 99 | 79.66 / 12.79 | 91.78 / 34.95 | 87.30 / 32.80 |
| VisA K4 | 1500 | 4,263 / 199 | 94.02 / 70.76 | 97.57 / 73.28 | 96.83 / 70.35 |
| VisA K1 | 1360 | 4,261 / 200 | 93.54 / 70.20 | 97.53 / 73.88 | 97.28 / 72.07 |
| **Mean / total** | — | **14,703 / 844** | **92.68 / 68.30** | **96.59 / 74.23** | **95.36 / 71.24** |

The group-normalized error gives the strongest mean ranking. The gain is most
visible on VisA, where appearance groups have heterogeneous reconstruction
scales. The mixture posterior is used to decide whether a distinct high-error
component exists; the normalized error provides the strongest ranking signal
on average.

Machine-readable values: [`data/dhp_early_evidence.csv`](data/dhp_early_evidence.csv).

## 4. Equal-budget purification

Removing more samples can trivially improve contamination recall. We therefore
compare DHP with a global early-error filter that removes exactly the same
number of samples in every setting.

- **Removal precision:** anomalous fraction of removed samples.
- **Contamination recall:** removed fraction of all injected anomalies.
- **Clean retention:** retained fraction of clean training samples.
- **Clean-tail removal:** removed fraction of clean ground-truth tail samples.
- **Remaining contamination:** anomalous fraction after removal.

| Setting | Strategy | Removed | Precision | Contamination recall | Clean retention | Clean-tail removal | Remaining contamination |
|---|---|---:|---:|---:|---:|---:|---:|
| MVTec P | Global | 67 | 68.66 | 62.16 | 97.15 | 33.93 | 3.76 |
|  | DHP | 67 | 89.55 | 81.08 | 99.05 | 0.00 | 1.88 |
| MVTec K4 | Global | 150 | 57.33 | 59.72 | 95.68 | 97.22 | 3.93 |
|  | DHP | 150 | 68.67 | 71.53 | 96.82 | 0.00 | 2.78 |
| MVTec K1 | Global | 180 | 83.33 | 88.24 | 98.25 | 100.00 | 1.17 |
|  | DHP | 180 | 79.44 | 84.12 | 97.84 | 0.00 | 1.58 |
| VisA P | Global | 191 | 11.52 | 21.15 | 91.84 | 100.00 | 4.13 |
|  | DHP | 191 | 32.98 | 60.58 | 93.82 | 0.00 | 2.07 |
| VisA K4 | Global | 383 | 42.04 | 79.70 | 94.57 | 100.00 | 1.05 |
|  | DHP | 383 | 44.65 | 84.65 | 94.81 | 14.29 | 0.79 |
| VisA K1 | Global | 393 | 41.98 | 81.68 | 94.39 | 100.00 | 0.95 |
|  | DHP | 393 | 43.51 | 84.65 | 94.54 | 14.29 | 0.80 |
| **Mean / total** | **Global** | **1,364** | **50.81** | **65.44** | **95.31** | **88.53** | **2.50** |
|  | **DHP** | **1,364** | **59.80** | **77.77** | **96.15** | **4.76** | **1.65** |

At the same 1,364-sample removal budget, DHP improves mean removal precision
by 8.99 points and contamination recall by 12.33 points. Clean-tail removal
falls from 88.53% to 4.76%. The comparison therefore isolates which samples
are removed rather than rewarding DHP for a larger deletion budget.

Machine-readable values: [`data/dhp_equal_budget.csv`](data/dhp_equal_budget.csv).

## 5. Reference quality after TRP

Initial candidate discovery favors tail coverage. TRP constructs a stricter
reference set for local comparison. Reference purity is one minus the
contamination rate; contaminant retention is the fraction of contaminated
initial candidates that remain eligible for reconciled memory.

| Metric | Initial candidates | Reconciled references |
|---|---:|---:|
| Reference purity | 91.44 | **97.88** |
| Contamination rate ↓ | 8.56 | **2.12** |
| Contaminant retention ↓ | 100.00 | **14.91** |
| Genuine-tail coverage | **94.67** | 66.10 |

TRP rejects 85.09% of contaminated candidates and lowers their rate from
8.56% to 2.12%. CTE retains the broad candidate view for image-level coverage
but admits only the reconciled view as dense localization evidence. Candidate
discovery and normal-reference construction consequently optimize different
requirements instead of forcing one set to serve both purposes.

Machine-readable values: [`data/trp_reference_quality.csv`](data/trp_reference_quality.csv).
