# Proposed protocol: AP–serialized-model-size v1

Status: **APPROVED AND LOCKED FOR FULL_RUN_01 (40+40)**

Protocol version: `ap_model_size_v1`

Technical approval date: 2026-10-20

Full-run approval date: 2026-10-20

Execution authorization: exactly 40 fresh RandomSampler + 40 fresh TPESampler trials

This document registers a separate replacement experiment for the unstable
AP–latency optimization protocol. After a passing 3+3 smoke and mandatory
preflight, one fresh `full_run_01` with 40 trials per sampler is approved. It
does not unblock the AP–latency path or authorize any additional trials.

## Research question

Within the frozen Random Forest search domain, what train-only configurations
form the best trade-off between cross-validated minority-class Average Precision
and the disk size of the compressed, serialized inference pipeline?

The proposed objectives are:

1. maximize mean three-fold validation Average Precision;
2. minimize the maximum compressed serialized pipeline size across the same
   three fitted CV-fold pipelines, measured in decimal MB.

Latency is a supplementary deployment measurement only. It is not an objective,
constraint, feasibility input, tie-breaker, Pareto coordinate, or reason to
change the selected set under this proposal.

## Frozen elements

The following remain unchanged:

- `bank-full.csv`, expected shape 45,211 x 17 and SHA-256
  `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`;
- excluded feature `duration`;
- frozen 80/20 train/holdout indices; the holdout is not read for optimization;
- shuffled, stratified three-fold CV generated from the frozen training set;
- leakage-safe preprocessing fitted inside each fold;
- Random Forest family, `random_state=42`, `n_jobs=1`, and the registered search
  space;
- Random Search and Optuna TPE, 40 trials per method in a future full study;
- `TPESampler(n_startup_trials=10)`, sequential Optuna execution;
- Joblib serialization with compression level 3;
- the 50.0 decimal-MB model-size limit and the no-relaxation policy.

Historical baseline, pilot, calibration v1/v2 and SQLite files are immutable.
The AP–latency hard block remains active.

## Objective and feasibility definitions

For trial `h` and fold `k in {1,2,3}`:

```text
AP_k(h) = average_precision(y_validation_k, p_positive_k)
S_k(h)  = byte length of joblib-compressed fitted Pipeline_k

f_AP(h)   = mean_k AP_k(h)                 (maximize)
f_size(h) = max_k S_k(h) / 1,000,000       (minimize)
g_size(h) = f_size(h) - 50.0                (feasible iff g_size <= 0)
```

The maximum fold size is retained from the existing protocol. It is conservative
with respect to the three CV fits and avoids choosing a trial from an unusually
small fold artifact. It is not guaranteed to upper-bound the size of a later
model refitted on all frozen training rows. Therefore every ultimately selected
candidate must also be refitted on the full frozen training set and have its
deployment artifact size reported and checked against 50.0 MB before holdout
evaluation. That post-selection audit must not be used to search the holdout or
silently substitute an unregistered candidate.

Failed/pruned trials are never feasible and never enter the Pareto front. A
serialization failure fails the trial; it must not be replaced by a guessed or
synthetic size. Optuna constraints use the signed value `g_size`, following the
required `g <= 0` convention.

## Serialization measurement contract

For each fold, fit the complete sklearn `Pipeline` containing both named steps:

```text
preprocess: ColumnTransformer with imputation and one-hot encoding
model:      RandomForestClassifier
```

Serialize that fitted `Pipeline`, not only `named_steps["model"]`, to a unique
temporary file using the pinned Joblib version and `compress=3`. Record the exact
file length in bytes before removing the temporary file. Derive:

```text
decimal MB = bytes / 1,000,000
MiB        = bytes / 1,048,576
```

The objective and 50 MB constraint use decimal MB only. MiB is descriptive and
must always be labelled. The trial log must retain all three fold byte counts,
the maximum bytes, decimal MB, MiB, compression setting, Python/scikit-learn/
Joblib versions, parameters, fold fingerprints, and pipeline schema. A SHA-256
digest may be recorded for audit but is not an objective.

Serialized disk size is deterministic only relative to the frozen data, code,
random seeds, library versions and serialization settings. It is not RAM usage,
latency, energy, or a cross-version portable model-complexity measure.

## Evidence supporting the measurement

The current implementation already serializes the fitted estimator returned by
`build_hpo_pipeline`; this estimator is the complete preprocessing-plus-model
pipeline. It records each fold's byte length and aggregates with `max`.

In the existing real v1/v2 calibration artifacts, all 12 configuration–fold
pairs had one unique byte length across sessions and two independent v2 refits.
The v1-to-v2 difference was exactly 0 bytes for every pair:

| Configuration | Fold 1 bytes | Fold 2 bytes | Fold 3 bytes | Max decimal MB |
|---|---:|---:|---:|---:|
| `5107cee7dd58` | 541,604 | 549,682 | 532,332 | 0.549682 |
| `114e7ccecdee` | 4,375,165 | 4,436,483 | 4,547,198 | 4.547198 |
| `48e74e2e02b3` | 5,791,055 | 5,682,181 | 5,952,823 | 5.952823 |
| baseline `f8d997bc1901` | 9,473,882 | 9,460,269 | 9,392,069 | 9.473882 |

This is strong engineering evidence under the pinned environment, not a general
statistical guarantee. The observed pilot configurations are far below 50 MB,
so the size constraint may be inactive over much of the search space. The Week 2
baseline refitted on all frozen training rows is 14.008766 MB, versus 9.473882 MB
for the largest baseline CV-fold artifact. This observed gap confirms that the
CV maximum is not an upper bound on the final deployment artifact and motivates
the required post-selection full-train size audit.

## Approved smoke study isolation and configuration

The approved runnable configuration is `configs/hpo_ap_model_size_v1.yaml`, with
a distinct output tree and fresh study names. It must not resume any AP–latency
SQLite database.

Registered design:

```yaml
protocol_version: ap_model_size_v1
status: approved_for_smoke_only
base_config: configs/experiment.yaml
split_indices_reference: results/baseline/split_indices.npz
output_root: results/hpo/ap_model_size_v1
seed: 42
samplers: [random, tpe]
directions: [maximize, minimize]
objective_names: [average_precision_cv_mean, model_size_max_mb]
aggregation:
  average_precision: mean_of_three_validation_folds
  model_size: max_of_three_fold_serialized_pipeline_sizes
constraints:
  model_size_cap_mb_decimal: 50.0
  latency_used: false
serialization:
  format: joblib
  compression: 3
pilot:
  trials_per_sampler: 3
  maximum_trials_per_sampler: 3
  approved: true
full:
  trials_per_sampler: 40
  blocked_until_smoke_acceptance_and_explicit_approval: true
tpe:
  n_startup_trials: 10
execution:
  optuna_n_jobs: 1
  estimator_n_jobs: 1
```

Proposed names include the protocol version, phase, sampler and seed, for
example `bank_rf_ap_size_v1_smoke_random_seed42`. Official names must replace
`smoke` with `official` and use separate SQLite files/directories.

## Pareto construction and reporting

Filter to `COMPLETE` trials satisfying `g_size <= 0`. Trial A dominates B iff:

```text
AP_A >= AP_B and size_A <= size_B
```

with at least one strict inequality. The historical AP–latency Pareto function
and reports retain their original meaning; AP–size uses separate labels and
versioned outputs.

Required future outputs include raw fold AP/size values, complete trial history,
feasibility values, AP–size Pareto CSV/JSON/figure, environment and protocol
fingerprints, and checksums. Candidate summaries may include quality-first,
size-first and a pre-registered normalized compromise, but no holdout or latency
value may choose among them.

## Comparing Random Search and TPE

Both methods receive the same domain, folds and 40-trial budget. Compare:

- feasible Pareto coverage and non-dominated points at fixed budgets 10/20/40;
- best feasible AP found by trial count;
- smallest size at registered AP thresholds;
- dominated hypervolume only if its reference point and implementation are
  fixed before seeing official results;
- failures, duplicate configurations and wall-clock cost as diagnostics.

Because TPE's first 10 trials are startup trials, a 3-trial smoke test checks
plumbing only and cannot demonstrate TPE superiority. A single sampler seed per
method supports a study-specific comparison, not a strong statistical claim
about algorithm superiority. A paper-grade claim would require an approved
multi-seed replication amendment and additional budget.

## Approval and stopping gates

The 2026-10-20 smoke approval covers exactly:

1. replacing latency with maximum fold serialized size as objective 2;
2. removing latency from optimization feasibility and Pareto selection;
3. the decimal-MB definition and `max` fold aggregation;
4. fresh study names/storage and the 3+3 smoke acceptance criteria.

The second explicit approval was recorded on 2026-10-20 for exactly one fresh
40+40 `full_run_01`. No objective, constraint, search-space, fold or budget
change is approved. The existing AP–latency hard block is not removed.

## Locked official selection rule

Before viewing official results, the primary solution rule is fixed as follows:

1. identify the highest mean CV AP among completed feasible trials;
2. retain completed feasible configurations no more than 0.005 absolute AP below
   that value;
3. select the distinct configuration with the smallest max-fold serialized byte
   size;
4. for an exact size tie, sort by higher mean AP, lower AP standard deviation,
   lexicographically smaller canonical config hash, Random before TPE, then lower
   trial number.

The chosen configuration is refit on all frozen training rows and its complete
pipeline is serialized with Joblib compression 3. If it exceeds 50 decimal MB,
the next ranked distinct configuration within the same AP-tolerance set is
audited. The tolerance is never expanded. Holdout data is not involved.

## Smoke execution outcome — 2026-10-20

The approved 3+3 smoke completed in 283.264 seconds. All six trials completed
and satisfied the 50.0 decimal-MB constraint. Each of the 18 fold pipelines was
serialized twice with identical bytes and SHA-256. RandomSampler and TPE emitted
the same three startup configurations; paired AP and fold-size results matched
exactly. This validates implementation reproducibility but is not evidence that
one sampler is better.

The three distinct non-dominated smoke points were:

| Config hash | Mean CV AP | CV AP std | Max fold size (MB) |
|---|---:|---:|---:|
| `5107cee7dd58` | 0.4153387044 | 0.0089621 | 0.549682 |
| `114e7ccecdee` | 0.4432024968 | 0.0066632 | 4.547198 |
| `b4e97ab5ded9` | 0.4458863530 | 0.0053475 | 5.952813 |

Smoke acceptance status was **PASS**. At that checkpoint the Full 40+40 remained
`BLOCKED_PENDING_USER_ACCEPTANCE`; the later explicit approval and mandatory
preflight, rather than the smoke result alone, authorized `full_run_01`.

## Official full_run_01 outcome — 2026-10-20

Preflight and final acceptance both passed. The fresh isolated studies completed
exactly 40 RandomSampler and 40 TPESampler attempts with zero failures; all 80
trials were feasible under the 50 decimal-MB cap. The combined feasible Pareto
front contains 13 entries with 13 distinct configuration hashes. Ten full-study
configurations occur in both samplers because their shared seed produces the
same registered TPE startup sequence.

The best feasible AP values were 0.4461627 for Random trial 31 (4.836080 MB)
and 0.4461862 for TPE trial 38 (3.693059 MB). Applying the locked selection rule
selected TPE trial 35, hash `4cdd6dd35f1d`: AP 0.4434175 ± 0.0054617 and
CV max-fold size 1.034871 MB. Its full-frozen-train pipeline was 1.289340 MB,
repeated serialization was exact, and the deployment-size audit passed.

The execution status is **COMPLETED_AWAITING_USER_ACCEPTANCE**. No holdout
evaluation was performed. The AP–latency path remains blocked, and this run does
not authorize extra trials, a second seed, commit/push, or holdout evaluation.
