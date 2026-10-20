# Research decision record: latency versus serialized model size

Decision status: **AP–MODEL-SIZE FULL_RUN_01 COMPLETED; AWAITING USER ACCEPTANCE**

Approval date: 2026-10-20

Active protocol remains: AP–latency, full run blocked.

## Why latency cannot currently be an official objective

Calibration v2 retained 2,400 official and 600 diagnostic raw measurements.
The preliminary diagnostic and official round 1 were stable, but official round
2 exhibited a system-wide slowdown. The median session latency rose from
18.2895 to 48.1037 ms/1,000 samples, a 2.630x ratio, and all 12
configuration–fold pairs failed the 15% within-round criterion. Prediction
digests, batch/fold fingerprints, effective thread limits, priority and model
sizes remained consistent. Round-2 fitting time also rose by 33.875%.

The evidence points to a run-wide environment effect rather than a prediction
correctness error or a baseline-fold-3-only problem. Available telemetry cannot
causally separate Windows scheduling, power/thermal state, heterogeneous-core
placement, cache/memory behavior and background workload. Because latency can
change feasibility and dominance, using these measurements would make the
official Pareto result dependent on an unstable nuisance variable.

No observations were removed, no threshold was relaxed and the fixed deployment
target `35.531700000092314 ms / 1,000` was not changed.

## Is AP–model-size a valid multi-objective problem?

Yes, provided its claim is narrow. AP measures predictive ranking quality while
compressed serialized size measures storage/distribution cost and partly
reflects tree-ensemble complexity. Increasing tree count and complexity can
improve AP while increasing artifact size, so neither objective universally
subsumes the other. A max-AP/min-size Pareto front is mathematically valid.

It is not a replacement measurement for latency. Smaller artifacts do not
guarantee lower inference time, RAM, energy use or better deployment behavior.
The proposed paper must describe the second objective as compressed serialized
pipeline size, not as computational efficiency in general.

## Evidence on reproducibility

The current code serializes the complete fitted sklearn pipeline, including the
fold-fitted `ColumnTransformer` and `RandomForestClassifier`, using pinned
Joblib 1.4.2 with compression level 3. It records exact bytes plus both decimal
MB and MiB; the 50 MB rule uses decimal MB.

Across calibration v1 and v2, every one of the 12 config–fold combinations had
the same exact byte length across all recorded sessions and both v2 refit rounds;
all v1→v2 byte deltas were zero. This is materially more stable than latency in
the current environment. Cross-version stability remains unproven, so software
versions and compression settings must stay part of the protocol fingerprint.

## Comparison

| Dimension | AP–latency | Proposed AP–model-size |
|---|---|---|
| Objective 1 | Mean train-only CV AP, maximize | Same |
| Objective 2 | Preprocess + `predict_proba` time, minimize | Compressed full-pipeline disk size, minimize |
| Current reproducibility | Failed v1 and v2 stability gates | Exact byte repeatability in observed v1/v2 fits |
| Environment sensitivity | High: scheduling, power, contention, cache | Low within pinned data/code/library environment |
| Aggregation | Max of fold timing medians | Max of fold serialized sizes |
| Constraint | Fixed 35.531700000092314 ms/1,000 plus 50 MB | 50.0 decimal MB only |
| Pareto feasibility | Unreliable on current machine | Deterministic under pinned environment |
| Deployment meaning | Direct time target on matching hardware/protocol | Storage/distribution and artifact complexity |
| Key limitation | Hardware-specific and presently unstable | Not a proxy for latency/RAM/energy |
| Historical study reuse | Forbidden | Forbidden; fresh studies required |

## Minimal code changes after approval

The existing AP–latency runner and reports should remain untouched as a
historical path. Implement the proposed protocol in versioned components:

1. add a new config and fresh output/storage namespace;
2. make the objective return `(mean_cv_ap, max_fold_size_mb)` and omit timing
   calls from the trial objective;
3. make `constraints_func` return only `[max_fold_size_mb - 50.0]`;
4. retain each fold's byte count, decimal MB/MiB, compression/version metadata
   and serialization failures;
5. add AP–size-specific reporting, labels, scenario names and Pareto outputs;
6. use separate study names and refuse any old database/objective signature;
7. retain two hard gates: approval before 3+3 smoke, then acceptance plus
   approval before 40+40.

An isolated `pareto_front_ap_model_size` utility and size-only feasibility helper
may be tested without creating or running a study. They do not activate the new
protocol or alter historical AP–latency semantics.

## Risks and limitations

- The 50 MB constraint may be non-binding: observed pilot maxima were only
  0.549682–5.952823 MB and the baseline was 9.473882 MB.
- Joblib output is version- and implementation-dependent; an environment upgrade
  can change bytes even when predictions agree.
- Maximum CV-fold size may not upper-bound a model refit on the full frozen
  training set; the observed baseline is 9.473882 MB at maximum CV-fold size but
  14.008766 MB after full-train refit. Selected deployment artifacts therefore
  need a separate size audit.
- One search seed and 40 trials per method are insufficient for a strong general
  claim that TPE outperforms Random Search.
- AP and size may be strongly correlated or yield a sparse Pareto front.
- Removing latency from selection improves reproducibility but narrows the
  research claim; latency can only be reported descriptively after selection.
- Any supplementary latency values on this machine inherit the v2 limitation
  and must carry a failed-stability warning.

## Approved smoke decision

AP–model-size v1 is approved as a new, deliberately changed research protocol.
The 3+3 smoke passed, and one fresh 40+40 `full_run_01` is authorized after the
mandatory preflight. Preserve AP–latency pilot/calibration results as
methodological diagnostics, not as results of the new study. The historical
AP–latency path remains blocked.

## Official execution outcome

The separately approved `full_run_01` completed on 2026-10-20 with 40/40
RandomSampler and 40/40 TPESampler trials, zero failures, and no holdout use.
All 80 configurations satisfied the 50 decimal-MB limit, so the constraint was
non-binding. The combined feasible Pareto set has 13 distinct configurations.
The locked near-best rule selected TPE trial 35 (`4cdd6dd35f1d`), whose full
frozen-training pipeline serialized to 1.289340 MB and passed the deployment
size audit. This single-seed outcome does not establish sampler superiority or
make model size a proxy for latency, RAM, or energy.
