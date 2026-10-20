# Research notes

## Decisions frozen before HPO

- Dataset identity: UCI Bank Marketing `bank-full.csv` only.
- Positive class: `yes`.
- Leakage exclusion: `duration`.
- Holdout: stratified 20%, seed 42.
- Inner evaluation: stratified 3-fold CV on train only, seed 42.
- Primary score: Average Precision from probabilities.
- Threshold metrics: threshold 0.5, not optimized.
- Baseline threads: one; latency thread limit: one.
- Serialization: Joblib, compression level 3.

## Interpretation cautions

- Literal `unknown` entries may represent missingness, non-disclosure, or a meaningful operational state; report them separately and do not silently impute them away.
- Duplicate rows are audited but not automatically deleted. Without a stable record identifier, removal could change the estimand; any future policy requires a documented sensitivity analysis.
- A random stratified split assumes rows are exchangeable. A future paper should discuss temporal and repeated-client risks if richer identifiers/timestamps become available.
- The test set is a final estimate, not a feedback channel. Do not iterate on it during Weeks 3–5.

## Run log convention

Keep each consequential run in a separate output directory or archive its metadata before rerunning. Compare runs only when dataset SHA, split hashes, config, environment, and metric definitions are accounted for.
