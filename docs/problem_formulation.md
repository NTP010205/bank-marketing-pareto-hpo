# Problem formulation

## Research question

Can hyperparameter search find Random Forest configurations that improve cross-validated Average Precision for the minority `yes` class while preserving practical inference latency and model size?

Weeks 1–2 establish only the data contract and reference baseline. They do not answer the multi-objective question.

## Prediction unit and target

Each row describes one direct-marketing contact. The binary target is whether the client subscribed to a term deposit (`y=yes`). Because the positive class is relatively rare, Average Precision is the primary quality metric; ROC-AUC is supplementary, while F1, precision, and recall are reported at a declared threshold of 0.5.

## Feature policy

`duration` is excluded before splitting because its value is only fully known after the call. Including it would answer an easier post-call question rather than the intended pre-call decision problem.

Availability is not binary for every remaining field. In particular:

- `campaign` depends on contacts already made in the current campaign;
- `contact` may depend on a channel assignment decision;
- `day` and `month` require a scheduled contact date.

The baseline retains these published fields to provide a reproducible point of comparison, but a deployment paper must define prediction time and perform a restricted-feature sensitivity analysis. No unlisted features are assumed.

## Week 3 registered multi-objective formulation

For hyperparameters `h`, minimize:

```text
f1(h) = -mean_stratified_3fold_AP(h)
f2(h) = median_latency_per_1000_samples(h)
```

subject to:

```text
compressed_serialized_model_size(h) <= 50 MB
median_latency(h) <= 1.5 * measured_baseline_latency
```

If no sampled configuration satisfies both constraints, the result is logged as infeasible. Constraints must not be relaxed silently. Random Search and TPE will each receive 40 trials over the same domain and the same folds.
