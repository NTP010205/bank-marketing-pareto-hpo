# Week 4 experiment analysis — AP–model-size v1

Status: **COMPLETE / READ-ONLY ANALYSIS**  
Date: 2026-10-20  
Official source: `results/hpo/ap_model_size_v1/full_run_01/`

## Scope and integrity

Week 4 analyzes the accepted 40+40 experiment without fitting a model, creating
an Optuna trial, changing the locked protocol, or evaluating the holdout. The
primary evidence is the official trial/fold history, SQLite studies, Pareto
tables, selected-candidate record and full-train size audit. Week 2 baseline
artifacts are used only for the explicitly labeled baseline comparison.

The source audit passed all checks:

- 80 CSV rows equal 80 JSON records and 40+40 SQLite trials;
- SQLite objective values equal CSV AP and size values;
- JSON AP differs from CSV by at most `4.9634e-11`, explained by the locked
  runner's 10-digit `DataFrame.to_json` representation;
- all 240 fold rows are present, three per completed trial;
- all official artifact and figure hashes match their manifests;
- recomputed and official Pareto hash sets both contain 13 configurations;
- no trial column contains latency, holdout or test metrics;
- selected hash `4cdd6dd35f1d` agrees across selection and deployment audit.

The train-only EDA reader verifies the registered dataset SHA-256, reads only
the 36,168 frozen training row numbers with `skiprows`, and excludes `duration`
at parse time. It observes 4,231 positive and 31,937 negative training labels
(11.6982% positive). Ten rows are duplicated after removing `duration`; this is
a feature-space observation, not authorization to delete records.

## Main results

| Model/reference | CV AP mean ± std | Serialized size | Measurement scope |
|---|---:|---:|---|
| Week 2 baseline | 0.4148607 ± 0.0074945 | 14.008766 MB | Full-train pipeline |
| Best Random, trial 31 | 0.4461627 ± 0.0043370 | 4.836080 MB | Maximum of three CV folds |
| Best TPE, trial 38 | 0.4461862 ± 0.0045716 | 3.693059 MB | Maximum of three CV folds |
| Selected TPE, trial 35 | 0.4434175 ± 0.0054617 | 1.034871 MB | Maximum of three CV folds |
| Selected TPE deployment audit | same CV selection record | 1.289340 MB | Full frozen-train pipeline |

CV fold standard deviation describes dispersion across the three fixed folds.
It is not an estimate of variability across independent HPO runs.

All 80 trial attempts completed and were feasible. There are 70 distinct
configuration hashes because the two samplers share 10 registered startup
configurations. The maximum observed CV max-fold size was 13.594303 MB, leaving
36.405697 MB below the 50 MB cap. The size constraint was therefore non-binding.

## Random Search versus TPE

| Sampler | Best AP | Size at best AP | Median size | Total trial duration |
|---|---:|---:|---:|---:|
| Random | 0.4461627 | 4.836080 MB | 3.381405 MB | 907.722 s |
| TPE | 0.4461862 | 3.693059 MB | 1.587451 MB | 571.027 s |

At budgets 10 and 20, both studies had best AP 0.4458864. At budget 40,
Random reached 0.4461627 and TPE reached 0.4461862. The terminal difference is
only 0.0000235 AP. TPE sampled smaller models in this run, which also reduced
observed evaluation cost. With one sampler seed and shared startup samples,
neither observation supports a statistical claim that TPE is superior.

## Pareto findings

The feasible front has 13 distinct configurations. At trial-entry level, 67 of
80 entries are dominated; at configuration level, 57 of 70 distinct hashes are
dominated. The front spans:

- 0.3847900 AP at 0.055190 MB at the smallest end;
- 0.4461862 AP at 3.693059 MB at the maximum-AP end.

The front shows a steep useful region near 0.5–1.1 MB and a flatter quality
region above roughly 3.5 MB. This description is specific to the registered
Random Forest search domain and serialization environment.

## Effect of the near-best rule

The registered threshold was `best AP - 0.005 = 0.4411862`. Eighteen distinct
configurations qualified. The deterministic rule selected TPE trial 35,
`4cdd6dd35f1d`, because it had the smallest max-fold serialized size among the
eligible configurations.

Relative to the maximum-AP configuration, the selected model:

- loses 0.0027687 absolute CV AP, remaining inside the 0.005 tolerance;
- reduces CV max-fold serialized size by 71.978%;
- passes the separate full-train audit at 1.289340 MB;
- requires no fallback candidate.

This is a deliberate quality–storage trade-off. It is not the “best AP model,”
and its full-train size must not be substituted for its CV objective value.

## Figures

All figures are generated from logged data by `scripts/build_week4_figures.py`,
saved as 320-DPI PNG and vector PDF under `figures/week4/`:

1. `01_best_so_far_cv_ap` — best feasible AP versus attempts;
2. `02_ap_vs_model_size` — all completed trials;
3. `03_pareto_selected` — dominated trials, Pareto front and selected point;
4. `04_model_size_distribution` — sampler size distributions;
5. `05_cumulative_search_time` — cumulative trial duration;
6. `06_key_models_comparison` — baseline, best and selected references;
7. `07_ap_loss_vs_size_reduction` — near-best decision trade-off.

The figure manifest records source hashes, output hashes and byte sizes in
`results/analysis/week4/analysis_manifest.json`.

## Reproduction

From the repository root with the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_week4_figures.py
.\.venv\Scripts\python.exe .\scripts\build_week4_notebooks.py --execute
```

These commands are read-only with respect to historical experiment artifacts.
They overwrite only derived Week 4 analysis tables, figures and notebooks.

