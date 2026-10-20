"""Feasibility and two-objective Pareto utilities for Week 3 HPO."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def constraint_status(
    model_size_mb: float,
    latency_ms: float,
    *,
    model_size_cap_mb: float,
    latency_cap_ms: float,
) -> dict[str, float | bool]:
    """Return signed violations; boundaries are feasible (violation <= 0)."""
    size_violation = float(model_size_mb - model_size_cap_mb)
    latency_violation = float(latency_ms - latency_cap_ms)
    return {
        "size_violation_mb": size_violation,
        "latency_violation_ms": latency_violation,
        "feasible": size_violation <= 0.0 and latency_violation <= 0.0,
    }


def model_size_constraint_status(
    model_size_mb: float,
    *,
    model_size_cap_mb: float,
) -> dict[str, float | bool]:
    """Return the proposed size-only constraint using Optuna's ``g <= 0`` rule."""
    size_violation = float(model_size_mb - model_size_cap_mb)
    return {
        "size_violation_mb": size_violation,
        "feasible": size_violation <= 0.0,
    }


def _pareto_front_max_ap_min_resource(
    trials: pd.DataFrame,
    *,
    resource_column: str,
) -> pd.DataFrame:
    """Filter the max-AP/min-resource front without changing feasibility rules."""
    if trials.empty:
        return trials.copy()
    candidates = trials.loc[
        trials["state"].eq("COMPLETE") & trials["feasible"].fillna(False)
    ].copy()
    if candidates.empty:
        return candidates

    ap = candidates["average_precision_cv_mean"].to_numpy(dtype=float)
    resource = candidates[resource_column].to_numpy(dtype=float)
    keep = np.ones(len(candidates), dtype=bool)
    for index in range(len(candidates)):
        no_worse = (ap >= ap[index]) & (resource <= resource[index])
        strictly_better = (ap > ap[index]) | (resource < resource[index])
        if np.any(no_worse & strictly_better):
            keep[index] = False
    return candidates.loc[keep].sort_values(
        [resource_column, "average_precision_cv_mean"],
        ascending=[True, False],
    )


def pareto_front(trials: pd.DataFrame) -> pd.DataFrame:
    """Return completed feasible points not dominated in max-AP/min-latency space."""
    return _pareto_front_max_ap_min_resource(
        trials,
        resource_column="latency_per_1000_samples_ms",
    )


def pareto_front_ap_model_size(trials: pd.DataFrame) -> pd.DataFrame:
    """Return the proposed max-AP/min-serialized-size front.

    This utility does not authorize or start an AP–model-size Optuna study.  It is
    kept separate from :func:`pareto_front` so historical AP–latency reports retain
    their original meaning.
    """
    return _pareto_front_max_ap_min_resource(
        trials,
        resource_column="model_size_max_mb",
    )


def select_scenario_candidates(front: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    """Select transparent candidates without consulting the holdout set."""
    if front.empty:
        return {"quality_first": None, "speed_first": None, "balanced_normalized": None}

    quality_index = front["average_precision_cv_mean"].idxmax()
    speed_index = front["latency_per_1000_samples_ms"].idxmin()
    ap = front["average_precision_cv_mean"].astype(float)
    latency = front["latency_per_1000_samples_ms"].astype(float)
    ap_range = float(ap.max() - ap.min())
    latency_range = float(latency.max() - latency.min())
    ap_loss = (ap.max() - ap) / ap_range if ap_range else pd.Series(0.0, index=front.index)
    latency_loss = (
        (latency - latency.min()) / latency_range
        if latency_range
        else pd.Series(0.0, index=front.index)
    )
    balanced_index = (ap_loss.pow(2) + latency_loss.pow(2)).pow(0.5).idxmin()

    def summarize(index: Any) -> dict[str, Any]:
        row = front.loc[index]
        return {
            "sampler": str(row["sampler"]),
            "trial_number": int(row["trial_number"]),
            "average_precision_cv_mean": float(row["average_precision_cv_mean"]),
            "latency_per_1000_samples_ms": float(row["latency_per_1000_samples_ms"]),
            "model_size_max_mb": float(row["model_size_max_mb"]),
        }

    return {
        "quality_first": summarize(quality_index),
        "speed_first": summarize(speed_index),
        "balanced_normalized": summarize(balanced_index),
    }
