# Week 4 research limitations and future experiment plan

Status: **DESIGN ONLY — NO FUTURE EXPERIMENTS EXECUTED**

## Current evidential limits

### One sampler seed

Random Search and TPE were each run once with seed 42. Their first ten
configurations are shared because TPE used ten startup trials. The observed AP,
size, convergence and runtime differences therefore describe this execution;
they do not estimate a sampling distribution or support a superiority test.
Three CV folds measure data-partition dispersion within a trial, not variability
between independent search runs.

### One historical dataset

UCI Bank Marketing is a single historical direct-marketing dataset. Results may
not transfer across countries, institutions, campaign policies, time periods,
contact channels or prevalence regimes. The current random stratified split also
cannot test temporal drift.

### Holdout already observed in Week 2

Week 2 evaluated the registered baseline on the frozen holdout once. Week 3–4
correctly avoided it, but the holdout is no longer a genuinely unseen final
estimate for a paper. Reusing it for model choice would compound leakage. A
paper-quality study should reserve a new external, temporal or institutional
test set before further model development.

### Serialization is environment-dependent

Joblib compression level 3 is deterministic in the pinned environment for the
observed pipelines, but byte size can change with Joblib, Python, NumPy,
scikit-learn, pickle implementation, object layout and compression-library
versions. Report bytes together with the full environment and never treat them
as an invariant property of the abstract estimator.

### Size is not runtime resource use

Compressed disk size is a valid storage/distribution objective. It is not a
measurement of inference latency, peak RAM, CPU energy, carbon emissions or
throughput. Latency remains supplementary because calibration v2 failed its
stability gate. Claims must preserve this narrow interpretation.

### Non-binding constraint

Every official trial was below 50 MB and the maximum was 13.594303 MB. The
experiment therefore characterizes an unconstrained AP–size trade-off inside
the sampled domain more than behavior at a binding deployment boundary. A lower
cap cannot be introduced retroactively.

## Registered multi-seed extension proposal

Before execution, create a versioned amendment specifying:

1. at least 10 independent sampler seeds per method, chosen before results;
2. equal 40-trial budgets, identical folds/search domain and fresh studies;
3. primary method-level outcomes such as best feasible AP, selected size,
   Pareto hypervolume with a pre-registered reference point, and time-to-target;
4. paired comparisons by seed where startup/search randomness permits;
5. uncertainty intervals and effect sizes, not only null-hypothesis tests;
6. multiplicity policy for several outcomes;
7. failed-run handling and fixed compute budget;
8. no holdout access during method comparison.

This plan requires new approval and compute accounting. Week 4 does not execute
it.

## Paper-stage model benchmark proposal

Benchmark 4–6 model families with comparable train-only evaluation and explicit
serialization contracts. A candidate set is:

- regularized logistic regression as a sparse linear reference;
- Random Forest as the current registered ensemble;
- histogram gradient boosting;
- a CPU-compatible gradient-boosted tree implementation, if dependency and
  licensing choices are frozen;
- Explainable Boosting Machine or another interpretable additive model;
- a compact multilayer perceptron only if preprocessing, calibration and
  serialization are specified reproducibly.

Budgets should be normalized by trial count and reported compute, with model-
family-specific spaces registered before execution. Cross-family serialized
size comparisons require one artifact contract and environment. Claims should
separate predictive performance, disk size, latency, RAM and interpretability.

## Ablation plan

Pre-register separate ablations for:

- feature availability: pre-call-only subset versus the current published set;
- literal `unknown` handling;
- class weighting;
- tree count, depth and minimum leaf size;
- preprocessing choices and one-hot representation;
- max-fold versus median-fold size aggregation;
- near-best tolerance values, evaluated as sensitivity analysis rather than
  post-hoc selection;
- compression level and serialization format, reported as artifact engineering
  rather than model-quality changes.

Each ablation must preserve an untouched evaluation set and distinguish
confirmatory from exploratory analysis.

## Related work and novelty discipline

The paper stage should systematically review multi-objective HPO, constrained
Bayesian optimization, model compression/storage-aware learning, cost-sensitive
bank marketing prediction, and reproducible benchmarking. Novelty must not be
claimed merely from combining Optuna, Random Forest and a size objective. A
defensible contribution could instead focus on a carefully evaluated protocol,
stability diagnostics, selection-rule behavior, or a new method that is tested
against strong baselines and ablations.

## Proposed-model gate

Do not introduce a proposed model until related work and multi-model baselines
identify a concrete gap. Before execution, register its mechanism, hypothesis,
parameter budget, comparison methods, ablations, statistical analysis and
failure criteria. Week 4 supplies descriptive evidence and a reproducible
analysis layer; it does not yet justify a novel-model claim.

