# Supplementary table data

These CSV files contain the values displayed in the TailGuard web supplement.
Percentage-valued columns use the 0–100 scale. The files are publication
aggregates, not replacements for the per-run artifacts written by `main.py`.

| File | Contents |
|---|---|
| `head_tail_results.csv` | TailedCore and TailGuard results separated into head, tail, and all classes. |
| `component_ablation.csv` | Prediction-distinct DHP, TRP, and CTE variants. |
| `joint_setting_contrast.csv` | Balanced/long-tail and clean/contaminated diagnostic. |
| `tailsampler_audit.csv` | TailSampler ranking of true tail membership and injected contamination. |
| `dhp_early_evidence.csv` | Retrospective contamination ranking of early reconstruction signals. |
| `dhp_equal_budget.csv` | DHP and global filtering at identical removal counts. |
| `trp_reference_quality.csv` | Candidate and reconciled-reference quality. |
| `cte_storage.csv` | Coverage and reconciled memory sizes. |
| `qualitative_pixel_ap.csv` | Pixel AP of the representative qualitative examples. |
| `construction_variability.csv` | Mean and sample standard deviation across five data constructions. |
| `protocol.json` | Aggregation rules and the intended scope of each evidence group. |

The main comparison aggregates five independent data constructions per
setting. Component and mechanism analyses use one matched construction per
setting. Ground-truth tail and contamination labels are used only for audit.
