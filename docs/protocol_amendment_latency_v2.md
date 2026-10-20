# Protocol amendment: latency calibration v2

Status before execution: **PRE-REGISTERED FOR TARGETED CALIBRATION ONLY**  
Full HPO status: **BLOCKED REGARDLESS OF V2 OUTCOME**

## Approved constraint interpretation

Option A is approved. The latency cap remains exactly
`35.531700000092314 ms / 1,000 samples` and is a fixed absolute deployment
target. It must not be described as `1.5 ×` the calibrated CV baseline. The
model-size cap remains 50 decimal MB. Objectives remain maximize train-only CV
Average Precision and minimize inference latency. No cap is recalculated by v2.

## Preserved design

Dataset fingerprint, frozen train/test indices, frozen stratified three-fold CV,
search space, seed, sampler budgets and Optuna TPE startup count remain unchanged.
The holdout contributes no features, labels or metrics to timing, constraints,
selection or Pareto analysis. Its positional indices may be read only to assert
that the profiling batch is disjoint.

The v1 directory is immutable historical evidence. V2 writes only to
`results/hpo/latency_calibration_v2/` and refuses to overwrite a non-empty
versioned directory.

## V1 diagnosis motivating v2

V1 baseline fold 3 session medians drifted from 21.1248 and 21.4055 to
18.0182 ms/1,000. Its latency did not show a meaningful monotonic association
with recorded system CPU, process efficiency or configuration order in the small
sample. Prediction timing, rather than model scoring, varied. V1 also measured
folds in fixed order: every fold-3 block followed all fold-1 and fold-2 work,
confounding fold identity with elapsed time, cache, power and thermal state.

These observations identify plausible scheduling/power/cache drift but do not
establish a single causal mechanism. V2 is designed to separate fold identity,
run order and model-object lifetime more clearly.

## Pre-registered diagnostic

Before official v2 calibration, baseline folds 1–3 are measured in two separate
refit rounds. Each round constructs fresh deterministic model objects, then runs
five sessions. Fold order is deterministically shuffled within every session.
Each block uses 10 warm-up calls and 20 timed calls. Probability-output hashes
are recorded before and after timing to detect any same-object prediction change.

Processor power information (current/max MHz when exposed by Windows), process
priority, process thread count, system/process CPU, effective threadpool limits,
background-process snapshots, timestamps, order, GC policy and batch hashes are
logged. This diagnostic does not select or tune a model.

## Official v2 calibration protocol

- One fixed 1,000-row feature batch from frozen training indices, seed 20261020.
- The selected batch must exactly match the v1 batch artifact and have zero
  overlap with frozen holdout indices. Labels are unused.
- Baseline and all three distinct pilot configurations are fit on each of the
  three frozen fold-training partitions.
- Two independent refit rounds; five timing sessions per round; 10 session
  medians per config/fold in total.
- Each timing block has 10 warm-ups and 20 `perf_counter_ns` repeats.
- `RandomForestClassifier(n_jobs=1)` and effective BLAS/OpenMP limit 1.
- All 12 configuration/fold pairs are deterministically shuffled within each
  session, removing fixed fold-order confounding.
- `gc.collect()` runs before each block. Cyclic GC is disabled only during the
  warm-up/timed calls and restored immediately afterward. This applies equally
  to every configuration and fold.
- A fixed one-second boundary separates sessions; a fixed five-second settling
  interval separates diagnostic and official calibration.
- Power plan, process priority and CPU affinity are observed but not modified.

No raw observation is removed. Training and serialization remain outside the
latency timer. Timing still includes the complete preprocessing pipeline plus
`predict_proba`.

## Aggregation and stability gate

Aggregation is unchanged in substance: for each config/fold, take the median of
all 10 session medians, then take the maximum of the three fold values for the
configuration. V2 additionally reports round-specific medians and dispersion.

The 15% relative-range threshold is unchanged. Every baseline/HPO config-fold
must satisfy it across all 10 sessions **and independently within each of the two
five-session rounds**. One shared batch hash, no measurement failure, no major
contention flag and stable prediction digests are also required. This remains an
engineering gate, not a statistical confidence guarantee.

Even if v2 passes, `full.full_run_blocked` remains true. Full 40+40 requires user
acceptance of the calibration report and a separate verification that the
official HPO objective implements this approved absolute-target protocol in a
fresh versioned study.

## Execution outcome — 2026-10-20

V2 completed in 347.039 seconds and retained 2,400 official plus 600 diagnostic
raw timings. The preliminary baseline diagnostic passed for all folds (maximum
relative range 11.502%). Official round 1 also passed for all 12 config/fold
pairs (maximum 12.536%). Official round 2 then slowed across every configuration
and fold: the median session latency rose from 18.2895 to 48.1037 ms/1,000 and
within-round relative range reached 99.908%. Consequently the overall v2 gate is
**FAIL** and full HPO remains blocked.

Prediction hashes were identical before/after timing and across both refits.
Batch/fold fingerprints, priority class, effective thread limits and model sizes
were consistent. Round-2 total fitting time also rose 33.875%, supporting a
run-wide time/environment effect rather than a baseline-fold-3-only defect.
Available telemetry cannot distinguish thermal/power behavior, heterogeneous-core
scheduling, cache/memory effects and background workload causally. No slow
observation was removed and no threshold or cap was changed.

