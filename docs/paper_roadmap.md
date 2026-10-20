# Paper-oriented roadmap

## Completed scope: Weeks 1–2

- Research-ready structure and deterministic environment specification.
- Dataset attribution, license, exact-file validation, audit, and fingerprinting.
- Leakage-aware feature construction and protected holdout.
- Fold-local preprocessing with a fixed Random Forest baseline.
- Quality, timing, model artifact, hardware, and provenance outputs.
- Unit/integration tests using synthetic fixtures only.

## Week 3 — controlled search (TODO)

Implement Random Search and Optuna TPE only after baseline tests and real-data run pass. Each gets exactly 40 trials, the same five-dimensional domain, identical precomputed folds, and consistent timing protocol. Persist trial-level parameters, fold metrics, failures, constraints, package versions, and study storage. Do not use the holdout.

## Week 4 — Pareto and robustness (TODO)

Apply the frozen 50 MB and `1.5 × baseline latency` constraints. Report infeasibility without changing limits. Compute non-dominated feasible configurations, repeat timing robustly, examine seed sensitivity, and perform the planned feature-availability sensitivity check without selecting from test outcomes.

## Week 5 — reporting (TODO)

Select the final configuration using training/CV evidence only, evaluate the holdout once under the final protocol, and prepare reproducible tables/figures. Discuss class imbalance, leakage prevention, operational availability, compute budget, external validity, uncertainty, AI assistance, and threats to validity.

## Suggested paper structure

1. Motivation and research question
2. Dataset and ethical/license considerations
3. Leakage-aware experimental design
4. Multi-objective optimization methods
5. Results and Pareto trade-offs
6. Robustness and limitations
7. Reproducibility statement
