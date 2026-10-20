# Draft protocol amendment: latency calibration v1

Status: **DRAFT — NOT APPROVED**  
Full HPO status: **BLOCKED**

## Scope and unchanged research design

This draft changes no data split, CV fold, search domain, seed, objective direction,
sampler budget, or test-set policy. The frozen 20% holdout remains excluded from HPO,
latency calibration used for selection, constraints, and Pareto analysis. Average
Precision is still evaluated by the frozen train-only stratified 3-fold CV protocol.
Random Search and TPE remain fixed at 40 trials each; TPE retains 10 startup trials.

## Registered historical rule

Week 2 fitted one baseline pipeline on the full 80% training partition and timed one
deterministic 1,000-row holdout batch. Its median was approximately 23.69 ms/1,000.
The registered latency cap was `1.5 × historic holdout baseline`, yielding
35.531700000092314 ms/1,000. These files and their labels remain historical and are
not rewritten.

## Reason for amendment

The Week 3 pilot timed fold-fitted models on fold-specific validation samples and
used the maximum of three fold medians. That differs from Week 2 in batch composition,
fitted model, training-set size, aggregation, run order, warm/cold history, and local
machine contention. Identical pilot hyperparameters consequently produced materially
different latency values. Therefore 35.53 ms is not a like-for-like `1.5 ×` cap for
the CV-fold workload.

## Proposed measurement protocol

Version `latency_calibration_v1` uses one fixed 1,000-row feature batch sampled only
from the frozen training indices with seed 20261020. Labels are not used. Every
baseline/pilot configuration is fit on each frozen fold-training partition, while
the same profiling batch is passed through full preprocessing plus `predict_proba`.
For each config/fold there are five sessions, five warm-up calls and 15 timed calls
using `perf_counter_ns`, `RandomForestClassifier(n_jobs=1)`, and threadpool limit 1.
Configuration order is deterministically shuffled for each fold/session. All repeats,
session medians, environment/thread metadata, failures and contention flags are saved.

The reported fold value is the median of its five session medians. The configuration
latency is the maximum over its three fold values. This maximum is deliberately
conservative and remains sensitive to a noisy fold. The engineering stability gate
requires each config/fold relative range across session medians to be at most 15%, a
single shared batch hash, no failed measurement, and no major CPU-contention flag.
This is not a formal statistical confidence guarantee.

## Constraint decision requiring approval

- **Option A — recommended for continuity:** retain 35.531700000092314 ms/1,000 as a
  predefined absolute deployment target. Do not describe it as `1.5 ×` the calibrated
  CV baseline. This preserves the numeric constraint while correcting its interpretation.
- **Option B:** define a new relative cap from the stable train-only calibrated baseline.
  This changes the registered constraint and requires explicit approval, a new protocol
  version, and a fresh official study. The multiplier and rounding rule must be stated
  before any official trial.

This draft does not select Option B and does not change the cap automatically. Even if
the stability gate passes, full HPO remains blocked until the user explicitly approves
the cap interpretation and the official objective is wired to the approved latency
protocol.

## Study isolation and risks

The existing `results/hpo/pilot/` CSV/JSON/SQLite files are diagnostic history only.
They must never be resumed into the official run. A future approved run uses
`results/hpo/official_latency_v1/` and versioned study names. Risks that remain include
hardware- and workload-dependence, Windows scheduling/power-state variation, the
conservatism of max-fold aggregation, and the possibility that an absolute local target
does not represent production hardware.

## Approval checklist

1. Calibration artifacts are complete and the stability gate result is reviewed.
2. Choose Option A or explicitly version and approve Option B.
3. Verify the official HPO latency objective implements the approved protocol.
4. Set `full.full_run_blocked: false` only after items 1–3 are documented.
5. Start fresh 40+40 studies; do not reuse pilot SQLite files.
