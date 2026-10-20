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

The baseline configuration is fixed before evaluation. After CV, fit it once on the full 80% training partition and evaluate the untouched holdout once.

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
