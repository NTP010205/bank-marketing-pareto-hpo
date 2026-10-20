# Proposed experiment matrix: AP–model-size v1

Status: **FULL_RUN_01 COMPLETED; AWAITING USER ACCEPTANCE**

## Study matrix

| Stage | Sampler | Trials | Purpose | Storage/study isolation | Current status |
|---|---|---:|---|---|---|
| Historical Week 3 pilot | Random | 3 | AP–latency smoke diagnostic | Existing `results/hpo/pilot/` | Immutable; not reusable |
| Historical Week 3 pilot | TPE | 3 | AP–latency smoke diagnostic | Existing `results/hpo/pilot/` | Immutable; not reusable |
| Latency calibration v1/v2 | N/A | N/A | Timing methodology diagnostics | Existing versioned calibration directories | Immutable; both gates failed |
| Approved AP–size smoke | Random | 3 | Integration and reproducibility only | `results/hpo/ap_model_size_v1/smoke/`; fresh DB/name | Completed; acceptance PASS |
| Approved AP–size smoke | TPE | 3 | Integration and reproducibility only | `results/hpo/ap_model_size_v1/smoke/`; fresh DB/name | Completed; acceptance PASS; startup only |
| Official AP–size full_run_01 | Random | 40 | Fixed-budget search | `results/hpo/ap_model_size_v1/full_run_01/`; fresh DB/name | Completed 40/40; 0 failed |
| Official AP–size full_run_01 | TPE | 40, including 10 startup | Fixed-budget search | `results/hpo/ap_model_size_v1/full_run_01/`; fresh DB/name | Completed 40/40; 0 failed |
| Selected-candidate audit | N/A | Pre-registered candidates within 0.005 AP of best | Full-train artifact size in `full_run_01` | Completed; selected artifact 1.289340 MB and feasible |
| Holdout evaluation | N/A | Once per pre-registered final candidate policy | Final generalization report only | Existing protocol-controlled output | Never used for tuning/selection |

## Common frozen matrix dimensions

| Dimension | Registered value |
|---|---|
| Dataset | `bank-full.csv`, 45,211 x 17, registered SHA-256 |
| Optimization data | Frozen training indices only |
| Holdout | Excluded from objectives, constraints, Pareto and configuration choice |
| Feature exclusion | `duration` |
| CV | Frozen shuffled stratified 3-fold protocol |
| Pipeline | Fold-fitted imputation, one-hot encoding, Random Forest |
| Search domain | Existing Week 3 Random Forest domain, unchanged |
| Seed | 42 |
| Objective directions | `[maximize, minimize]` |
| Objective names | `[average_precision_cv_mean, model_size_max_mb]` |
| AP aggregation | Mean across three validation folds |
| Size aggregation | Maximum of three compressed fitted-pipeline sizes |
| Size unit | Decimal MB (`bytes / 1,000,000`) |
| Constraint | `model_size_max_mb - 50.0 <= 0` |
| Latency role | Supplementary deployment metric after selection only |
| Parallelism | Optuna `n_jobs=1`; estimator `n_jobs=1` |
| TPE startup | 10 trials in the future 40-trial study |

## Proposed 3+3 smoke design

The smoke test is not a research result and cannot compare sampler quality.
All three TPE trials occur within its 10-trial startup phase.

Each sampler receives exactly three trials. Acceptance requires:

1. exactly three terminal trials per fresh study with no accidental resume;
2. no holdout feature, label or metric read by the objective;
3. unchanged dataset, split, fold, search-space and environment fingerprints;
4. the complete preprocessing-plus-estimator pipeline serialized for every fold;
5. all three fold sizes retained in bytes, decimal MB and MiB;
6. `model_size_max_mb == max(fold_bytes) / 1,000,000` exactly;
7. size-only constraint values equal `model_size_max_mb - 50.0`, with
   feasibility iff the value is nonpositive;
8. objective values ordered as mean AP then maximum fold size;
9. AP–size Pareto output excludes failed/infeasible/dominated trials;
10. paired duplicate configurations, if produced by startup sampling, agree in
    AP and exact serialized byte sizes; any mismatch blocks full execution;
11. no latency value is passed into objective, constraints, feasibility,
    candidate selection or Pareto construction;
12. CSV/JSON manifests contain raw fold results, hashes, versions, timestamps,
    runtime, errors and study identity.

Smoke failure keeps all full paths blocked. Smoke success also keeps full paths
blocked until the user reviews the artifacts and explicitly approves 40+40.

## Proposed official comparison

At trial budgets 10, 20 and 40, report the feasible non-dominated set, best
feasible AP, smallest size at pre-registered AP thresholds, duplicate/failure
counts and elapsed time. A hypervolume comparison is permitted only if the
reference point and implementation are fixed in the approved config before the
official run. Do not choose a favorable reference point after viewing results.

Random and TPE receive equal 40-trial budgets. Results should be described as
outcomes for seed 42 on this dataset/protocol. Claims of general sampler
superiority require future independent sampler seeds and an approved budget
amendment.

## Registered execution sequence

1. User approves the objective, size definition, aggregation and latency role.
2. Implement and review the new config/runner/reporting namespace.
3. Run unit tests, compile checks, dependency/hash/split checks and hard-block
   tests.
4. Run fresh 3+3 smoke studies only.
5. Produce a smoke acceptance report and stop.
6. After a second explicit approval, run fresh official 40+40 studies.

Steps 1–6 completed on 2026-10-20. The run is now stopped at the required user
acceptance gate; holdout evaluation and any additional trials remain unauthorized.

The AP–latency full-run block and its historical databases remain untouched
throughout this sequence.

## Smoke acceptance result — 2026-10-20

- Exactly three RandomSampler and three TPE trials were attempted and completed.
- All six were feasible under `g_size <= 0`.
- All 18 fold artifacts passed repeated byte/SHA serialization checks.
- The three paired startup configurations reproduced AP and fold sizes exactly.
- Immutable baseline, pilot and latency-calibration snapshots were unchanged.
- Full 40+40 was blocked at this checkpoint pending explicit post-smoke
  acceptance. That later approval and preflight were recorded before execution.

## Official full_run_01 result — 2026-10-20

- RandomSampler: 40/40 COMPLETE, 0 FAIL, best AP 0.4461627.
- TPESampler: 40/40 COMPLETE, 0 FAIL, best AP 0.4461862.
- All 80 trials were feasible; the 50 MB constraint was non-binding.
- Combined Pareto: 13 entries, 13 distinct hashes; 10 configurations were
  shared across samplers because of the registered startup sequence.
- Locked selection rule chose TPE trial 35 (`4cdd6dd35f1d`), AP 0.4434175 ±
  0.0054617, CV max-fold size 1.034871 MB. Full-train size audit: 1.289340 MB,
  feasible.
- Status: **COMPLETED_AWAITING_USER_ACCEPTANCE**. No holdout evaluation ran.
