# TailGuard supplementary material

This web supplement records experimental and implementation details that do
not fit in the four-page ICASSP technical paper. It complements the paper; it
does not introduce a different TailGuard configuration.

The supplement is organized as follows:

1. [Method and implementation details](method_details.md) gives the complete
   protocol and the operational details of Dynamic Head Purification (DHP),
   Tail Reconciliation and Preservation (TRP), and Conditional Tail
   Enhancement (CTE).
2. [Additional results](additional_results.md) reports disaggregated head and
   tail results, the complete prediction-distinct component comparison,
   internal CTE configurations, storage cost, qualitative sample scores, and
   variability across data constructions.
3. [Mechanism analysis](mechanism_analysis.md) provides quantitative evidence
   for the long-tailed contaminated setting, audits what TailSampler identifies,
   and evaluates the decisions made by DHP and TRP.
4. [`data/`](data/) contains the displayed tables in machine-readable CSV
   format.

## Evidence scope

The main comparison follows the public long-tailed noisy protocol and
aggregates five independent data constructions for each of six settings. The
component and mechanism analyses use one matched construction for each setting
so that compared decisions act on exactly the same training samples. Ground
truth class roles and injected-contamination manifests are used only for
post-training analysis; TailGuard does not use them during training or
inference.

All values are percentages unless otherwise stated. “Mean” denotes an
unweighted arithmetic mean over the six dataset and imbalance settings.

## Reproducibility map

A successful run writes seven compact artifacts under
`saved_results/<run>/tailguard/`:

| Artifact | Purpose in the supplement |
|---|---|
| `train_audit.csv` | Reconstructs candidate membership, DHP decisions, TRP assignments, and contamination audits when a manifest is available. |
| `training_dynamics.csv` | Records group-level early reconstruction statistics and the checkpoint selected by DHP. |
| `test_scores.csv` | Contains per-image coverage, reconciled, and final scores. |
| `per_class_metrics.csv` | Supports overall and class-disaggregated evaluation. |
| `run_summary.json` | Records the resolved configuration, aggregate metrics, timing, and SHA-256 hashes of the run artifacts. |
| `reconciled_memory.pt` | Stores the reconciled tail reference system used for localization. |
| `coverage_memory.pt` | Stores the broader candidate reference system used for image-level coverage. |

The CSV files in this directory are publication tables aggregated from the
corresponding run artifacts. They are included to make the reported numbers
easy to inspect without parsing the manuscript.

## Frozen scope

Only the final TailGuard method is documented here. Rejected calibration
prototypes, alternative routing rules, and exploratory module definitions are
not part of the released method and are intentionally excluded.
