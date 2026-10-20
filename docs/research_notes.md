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
- More precisely, Week 2 already evaluated and reported the holdout once. It is no longer genuinely unseen, and Week 3–4 must not read or reuse it. A paper-quality final estimate needs a new external/temporal test set.

## Run log convention

Keep each consequential run in a separate output directory or archive its metadata before rerunning. Compare runs only when dataset SHA, split hashes, config, environment, and metric definitions are accounted for.

## Week 3 pilot observations and open questions

- RandomSampler and TPESampler receive identical first three configurations because TPE has ten startup trials and both use seed 42. This is useful for plumbing/timing reproducibility checks but says nothing about adaptive TPE performance.
- AP is deterministic for identical configurations/folds; observed latency is not. Some identical configurations changed enough to cross the frozen feasibility boundary.
- The max-fold aggregation is intentionally conservative, but it is not identical to the Week 2 one-model holdout measurement. Before full experiments, decide whether to keep it, add repeated timing sessions/interleaving, or define a train-only calibration workload—without changing the registered 35.53 ms cap silently.
- Model sizes were well below 50 MB in the pilot; this does not guarantee the full domain is always feasible.
- The v1 stabilization benchmark profiles one fixed train-only batch for all fold-fitted models. It does not use labels and is not a validation metric.
- The historic 35.53 ms cap remains numerically unchanged. It can be retained as an absolute target (Option A), but cannot be called a like-for-like `1.5 ×` calibrated CV baseline. A relative recalibration (Option B) requires an approved, versioned amendment.
- Historical pilot studies are not resumable into official results. The reserved official path and study names use `official_latency_v1`; full execution remains hard-blocked.

## Latency calibration v1 result (2026-10-20)

- The targeted real run completed in 83.424 seconds and preserved 900/900 timed repeats, 60 session summaries, one benchmark-batch hash, and zero holdout overlap.
- No measurement failed and no session crossed the registered CPU-contention thresholds. Effective BLAS/OpenMP thread counts were one during every timing block.
- Gate result: **FAIL**. Baseline fold 3 session medians `[21.1248, 21.4055, 18.8868, 19.4575, 18.0182]` ms/1,000 have relative range 17.126%, above the 15% engineering threshold. No observation was removed.
- The calibrated baseline max-fold value is 19.7269 ms/1,000. The unchanged 35.5317 ms cap is therefore 1.801× this value, not a like-for-like 1.5× CV baseline. This is evidence about protocol comparability, not authorization to recalculate the cap.
- All historical baseline and pilot file hashes matched before/after. Full 40+40 remains blocked.

## Latency calibration v2 result (2026-10-20)

- Option A is approved: 35.531700000092314 ms/1,000 is a fixed absolute deployment target, not a `1.5 ×` claim relative to the calibrated CV baseline. Size remains capped at 50 decimal MB.
- V2 pre-registered two independent refit rounds, five sessions per round, 10 warm-ups and 20 timed repeats. It interleaved all 12 config/fold pairs, used the exact v1 train-only batch, disabled cyclic GC only inside every equal timing block, and did not change power plan, priority or affinity.
- The baseline-only diagnostic passed on all folds: maximum all-session relative range 11.502%. Official round 1 also passed all 12 config/fold pairs, maximum 12.536%.
- Official round 2 exhibited a run-wide slowdown: median session latency increased from 18.2895 to 48.1037 ms/1,000 (2.630×), and all 12 config/fold pairs failed the 15% within-round criterion. Maximum within-round range was 99.908%; maximum range over all sessions was 174.007%.
- Model prediction hashes remained exactly identical across pre/post measurements and independent refits. The batch hash, frozen folds, priority class 32, `n_jobs=1`, effective thread limit 1 and artifact sizes were stable. The failure is timing/environmental rather than a change in predictions.
- Round-2 aggregate fitting time was 77.418 s versus 57.829 s in round 1 (+33.875%). Median system CPU rose from 8.61% to 14.70%, while median process CPU efficiency fell from 98.94% to 91.76%. The registered coarse contention thresholds did not fire.
- Sustained-load power/thermal behavior, Windows scheduling across heterogeneous cores, memory/cache pressure and background workloads are plausible. Available MHz snapshots do not expose executing-core identity, turbo state or temperatures, so a single causal mechanism is not proven.
- All 3,000 raw timings were retained. V2 gate is **FAIL**; apparent constraint-feasibility labels are diagnostic only. Full 40+40 remains blocked.

Open research questions, not novelty claims:

1. Does constrained multi-objective TPE improve sample efficiency over seeded random search after its startup phase under identical compute budgets?
2. How stable is the feasible Pareto set across independent timing sessions and random seeds?
3. How much do deployment-time feature-availability restrictions change the AP/latency trade-off?

## 2026-10-20 — AP–model-size v1 smoke

- A separate AP–model-size v1 protocol was approved for a 3+3 smoke only.
- Objectives were mean train-only three-fold CV AP (maximize) and maximum fold
  serialized full-pipeline size in decimal MB (minimize). Only the 50 MB size
  constraint was active; latency was excluded from objectives, feasibility,
  Pareto ranking and tie-breaking.
- RandomSampler and TPESampler each completed exactly three trials. All six were
  feasible, all 18 repeated serializations were byte/SHA exact, and paired
  startup configurations reproduced AP and fold sizes exactly.
- The smoke acceptance gate passed. TPE remained within its 10 startup trials,
  so no sampler-performance inference is permitted.
- Baseline, historical pilot and calibration v1/v2 integrity snapshots were
  unchanged. Both AP–latency full HPO and AP–model-size full 40+40 remain blocked.
4. Would an independent temporal/external test set preserve the ranking selected solely from training CV?

## 2026-10-20 — AP–model-size v1 official full_run_01

- Mandatory preflight passed before execution. Dataset SHA-256
  `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`,
  frozen split SHA-256
  `583acb9dbb4ee92852b1d0a9b584cffb2f7b1fc6409dfdb495a46df4cb3d7d1a`,
  train-index SHA-256
  `b078ac348228f8e49c30af463dc1198fb5e8e11d6379e5d842caf580fc728469`
  and CV-fold SHA-256
  `149fca7f07d85b8afabde2ed23c1eba22927d16c30b506b3b6ba1902c000b769`
  matched the smoke run. `duration` was excluded and no holdout rows were
  materialized.
- Fresh isolated RandomSampler and multi-objective TPESampler studies each
  completed exactly 40/40 attempts with zero failures. Ten configurations were
  duplicated across samplers because both studies used the same seed for the
  ten registered TPE startup trials; smoke trials were not imported.
- Best feasible mean CV AP was 0.4461627 for RandomSampler (trial 31,
  4.836080 MB max-fold size) and 0.4461862 for TPESampler (trial 38,
  3.693059 MB). These single-seed results are descriptive and do not establish
  TPE superiority.
- The combined feasible Pareto front contained 13 entries and 13 distinct
  configuration hashes. All 80 completed trials were below the 50 decimal MB
  constraint; the cap was therefore non-binding in this run.
- The pre-registered near-best rule retained configurations no more than 0.005
  AP below the best and selected the smallest serialized pipeline in that set:
  TPE trial 35 (`4cdd6dd35f1d`), AP 0.4434175 ± 0.0054617 and CV max-fold
  size 1.034871 MB. Its full-frozen-train pipeline serialized to 1.289340 MB,
  passed the 50 MB deployment audit, and produced an exact repeated artifact
  SHA-256 `d63cf98c067067f9dbd421d9569a44b8909a5740ac2b102f1a52f9a36f2175c1`.
- Initial wall time was 1,489.131 seconds. Trial-duration sums were 907.722
  seconds for RandomSampler and 571.027 seconds for TPESampler. Resume checks
  reopened both studies without adding trial 41.
- Official acceptance is **PASS / COMPLETED_AWAITING_USER_ACCEPTANCE**. The
  AP–latency full-run path remains blocked, no holdout evaluation was performed,
  and no Git commit or push was made.

## 2026-10-20 — Week 4 read-only experiment analysis

- Week 4 used only the immutable AP–model-size `full_run_01`, frozen Week 2
  baseline records and frozen training indices. It ran no training, Optuna trial
  or holdout evaluation.
- CSV, JSON and read-only SQLite objective records agreed. JSON AP values carry
  the expected 10-digit export rounding, with maximum absolute difference
  `4.9634e-11` from CSV; CSV and SQLite matched at the stricter objective audit.
- The official 80 trial entries represent 70 distinct hashes because of ten
  shared startup configurations. Thirteen distinct hashes are Pareto-optimal;
  67 trial entries and 57 distinct configurations are dominated.
- The registered near-best rule retained 18 distinct configurations. Selected
  trial `4cdd6dd35f1d` loses 0.0027687 absolute AP versus the best feasible AP
  while reducing CV max-fold size by 71.978% relative to the best-AP model.
- The 50 MB cap was non-binding: the largest observed CV max-fold artifact was
  13.594303 MB. This cannot be reframed as evidence about latency, RAM or energy.
- Five notebooks executed successfully under the project `.venv`. Seven 320-DPI
  PNG figures and seven PDF counterparts were generated; a second build produced
  bit-exact PNG hashes.
- Verification: 47 tests passed, compileall passed and all 31 installed packages
  were compatible. Week 5 remains unstarted pending user acceptance.
