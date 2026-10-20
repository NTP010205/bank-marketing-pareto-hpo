# Frozen experiment protocol (Weeks 1–2)

## Data gate

1. Require the exact semicolon-delimited `bank-full.csv`.
2. Validate shape `(45211, 17)`, ordered schema, and labels `{yes, no}`.
3. Record SHA-256, file bytes, nulls, `unknown` categories, class distribution, and duplicates.
4. Remove `duration` from features before any split.

Failure at this gate stops the real experiment; synthetic data is permitted only inside tests.

## Holdout and CV isolation

Use `train_test_split(test_size=0.20, stratify=y, random_state=42)`. Store positional train/test indices plus SHA-256 hashes. The 20% test partition is not used for model choice, EDA-driven choice, objective selection, feature choice, threshold tuning, or hyperparameter tuning.

On the 80% training partition, use `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`. The complete preprocessing and estimator pipeline is fit separately on each fold's training rows. One-hot categories are therefore never learned from validation or test rows.

The baseline configuration was fixed before evaluation. After CV, it was fit once on the full 80% training partition and the holdout was evaluated once during Week 2. Therefore the holdout must **not** be described as never observed. From Week 3 onward it is not read, printed, plotted, diagnosed, tuned against, or used for candidate selection. A future paper should use a new independent or temporal external test set for a genuinely unseen final estimate.

## Metrics

- Primary: Average Precision from positive-class `predict_proba`.
- Supplementary ranking metric: ROC-AUC from probabilities.
- Threshold metrics: F1, precision, recall at `p >= 0.5`.
- CV output: every fold plus arithmetic mean and sample standard deviation.

## Latency

Latency includes `ColumnTransformer` preprocessing and `predict_proba`. Select a deterministic batch of 1,000 holdout rows, run 2 warm-ups, then 7 timed repeats under a one-thread limit. Record every repeat and normalized milliseconds per 1,000 rows; summarize median, sample standard deviation, IQR, min, and max. If a test fixture contains fewer rows, record its actual batch size and normalize; never present that fixture value as a UCI benchmark.

## Artifact size and environment

Serialize the fitted pipeline with Joblib compression level 3. Report bytes, decimal MB, MiB, and SHA-256, explicitly distinguishing disk size from RAM usage. The registered 50 MB constraint uses decimal MB. Record fit time, platform, processor/machine fields, logical CPU count, Python executable/version, core package versions, detected thread pools, estimator `n_jobs`, and latency thread limit.

## Outputs

`results/baseline/` contains fold CSV, CV summary JSON, holdout metrics JSON, latency JSON, metadata JSON, split indices, model size metadata, the ignored serialized model, and a held-out precision-recall figure. Every number must be produced by code in the same run.

## Week 3 train-only HPO extension

The frozen `train_indices` from `results/baseline/split_indices.npz` are authoritative. HPO selects only those rows before constructing features/targets. The same `StratifiedKFold(3, shuffle=True, random_state=42)` split is materialized once in `results/hpo/cv_folds_seed42.npz`; fold hashes are logged.

Both samplers use the same five-dimensional domain, objectives and folds. Optuna runs trials sequentially (`n_jobs=1`), while Random Forest and timing thread limits remain one. Studies use separate resumable SQLite files for pilot/full. The pilot is hard-limited to three trials per sampler; full mode additionally requires `--confirm-full`.

Each fold records validation AP from positive-class probabilities, all seven timing repeats, fit time and compressed Joblib size. The HPO latency objective/constraint uses the maximum of the three fold median latencies; size uses the maximum fold file size. This is conservative but differs from Week 2, which measured one full-train model on one holdout batch. The mechanics match, but the data source and aggregation do not; the 35.53 ms cap remains frozen and must not be silently recalibrated.

The pilot found substantial timing variation for identical startup configurations and in the train-only baseline calibration. Full 40+40 execution is blocked until the aggregation and timing-stabilization procedure are approved. Pilot feasibility labels remain an auditable record of the measurements observed, not a claim of stable deployment feasibility.

### Latency stabilization gate v1

Before any official 40+40 study, run `configs/latency_calibration_v1.yaml`. It defines one deterministic 1,000-row batch drawn exclusively from the frozen training indices (seed 20261020; labels unused), shared across every baseline/pilot configuration and CV-fold model. Each config/fold is measured in five sessions, with five warm-ups and 15 `perf_counter_ns` repeats per session under estimator/threadpool limit 1. Configuration order is deterministically shuffled within every fold/session; raw repeats, order, thread metadata, contention signals and failures are retained.

Aggregation is median across the five session medians for each fold, followed by maximum across the three fold values. The engineering gate requires relative range across session medians <=15% for every config/fold, one shared batch hash, no failed measurement and no major contention. Passing this gate does not itself authorize full HPO: 35.53 ms remains unchanged, is not a like-for-like `1.5 ×` CV baseline, and needs an explicitly approved interpretation. See `protocol_amendment_latency_v1.md`.

The official study directory is reserved as `results/hpo/official_latency_v1/`; study names are versioned. The historical `results/hpo/pilot/` study is never resumed into it. The CLI hard-blocks full mode while `full.full_run_blocked: true`.
