"""Export auditable HPO histories, summaries, Pareto tables, and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import optuna
import pandas as pd

from .data import write_json
from .pareto import pareto_front, select_scenario_candidates

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def study_to_dataframe(study: optuna.Study, *, sampler_name: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        values = list(trial.values) if trial.values is not None else [None, None]
        attributes = trial.user_attrs
        rows.append(
            {
                "sampler": sampler_name,
                "trial_number": int(trial.number),
                "state": trial.state.name,
                "average_precision_cv_mean": values[0] if len(values) > 0 else None,
                "latency_per_1000_samples_ms": values[1] if len(values) > 1 else None,
                "average_precision_cv_std": attributes.get("average_precision_cv_std"),
                "model_size_max_bytes": attributes.get("model_size_max_bytes"),
                "model_size_max_mb": attributes.get("model_size_max_mb"),
                "model_size_max_mib": attributes.get("model_size_max_mib"),
                "feasible": attributes.get("feasible"),
                "size_violation_mb": attributes.get("size_violation_mb"),
                "latency_violation_ms": attributes.get("latency_violation_ms"),
                "fold_average_precision_json": json.dumps(
                    attributes.get("fold_average_precision", []), separators=(",", ":")
                ),
                "fold_latency_medians_ms_json": json.dumps(
                    attributes.get("fold_latency_medians_ms", []), separators=(",", ":")
                ),
                "fold_latency_repeats_ms_json": json.dumps(
                    attributes.get("fold_latency_repeats_ms", []), separators=(",", ":")
                ),
                "fold_model_sizes_bytes_json": json.dumps(
                    attributes.get("fold_model_sizes_bytes", []), separators=(",", ":")
                ),
                "fold_fit_seconds_json": json.dumps(
                    attributes.get("fold_fit_seconds", []), separators=(",", ":")
                ),
                "runtime_seconds": (
                    trial.duration.total_seconds() if trial.duration is not None else None
                ),
                "seed": seed,
                "error": attributes.get("error"),
                "started_at": _iso(trial.datetime_start),
                "completed_at": _iso(trial.datetime_complete),
                "param_n_estimators": trial.params.get("n_estimators"),
                "param_max_depth": trial.params.get("max_depth"),
                "param_min_samples_leaf": trial.params.get("min_samples_leaf"),
                "param_max_features": trial.params.get("max_features"),
                "param_class_weight": trial.params.get("class_weight"),
            }
        )
    return pd.DataFrame(rows)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def export_study(study: optuna.Study, output_dir: Path, *, sampler_name: str, seed: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = study_to_dataframe(study, sampler_name=sampler_name, seed=seed)
    frame.to_csv(output_dir / f"{sampler_name}_trials.csv", index=False, encoding="utf-8")
    write_json({"trials": _records(frame)}, output_dir / f"{sampler_name}_trials.json")
    return frame


def _plot_best_feasible(trials: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for sampler, group in trials.sort_values("trial_number").groupby("sampler"):
        running: list[float] = []
        best = -np.inf
        for _, row in group.iterrows():
            if row["state"] == "COMPLETE" and bool(row["feasible"]):
                best = max(best, float(row["average_precision_cv_mean"]))
            running.append(np.nan if best == -np.inf else best)
        axis.plot(group["trial_number"], running, marker="o", label=sampler)
    axis.set_xlabel("Completed trial number")
    axis.set_ylabel("Best feasible CV Average Precision")
    axis.set_title("Pilot: best feasible AP versus trials")
    axis.grid(alpha=0.25)
    if not trials.empty:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_pareto(trials: pd.DataFrame, front: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    completed = trials.loc[trials["state"].eq("COMPLETE")]
    feasible = completed.loc[completed["feasible"].fillna(False)]
    infeasible = completed.loc[~completed["feasible"].fillna(False)]
    if not infeasible.empty:
        axis.scatter(
            infeasible["latency_per_1000_samples_ms"],
            infeasible["average_precision_cv_mean"],
            marker="x",
            color="tab:red",
            label="Infeasible",
        )
    if not feasible.empty:
        axis.scatter(
            feasible["latency_per_1000_samples_ms"],
            feasible["average_precision_cv_mean"],
            color="0.6",
            label="Feasible dominated",
        )
    if not front.empty:
        ordered = front.sort_values("latency_per_1000_samples_ms")
        axis.plot(
            ordered["latency_per_1000_samples_ms"],
            ordered["average_precision_cv_mean"],
            marker="o",
            color="tab:green",
            linewidth=1.5,
            label="Feasible Pareto front",
        )
    axis.set_xlabel("Conservative latency: max fold median (ms / 1,000)")
    axis.set_ylabel("Mean 3-fold validation AP")
    axis.set_title("Pilot AP–latency feasibility and Pareto front")
    axis.grid(alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_constraint_counts(trials: pd.DataFrame, path: Path) -> None:
    samplers = sorted(trials["sampler"].unique()) if not trials.empty else []
    categories = ["size", "latency", "both"]
    counts: dict[str, list[int]] = {category: [] for category in categories}
    for sampler in samplers:
        group = trials.loc[(trials["sampler"] == sampler) & trials["state"].eq("COMPLETE")]
        size = group["size_violation_mb"].fillna(np.inf) > 0
        latency = group["latency_violation_ms"].fillna(np.inf) > 0
        counts["size"].append(int((size & ~latency).sum()))
        counts["latency"].append(int((latency & ~size).sum()))
        counts["both"].append(int((size & latency).sum()))
    figure, axis = plt.subplots(figsize=(7, 5))
    x = np.arange(len(samplers))
    bottom = np.zeros(len(samplers), dtype=int)
    for category in categories:
        values = np.asarray(counts[category], dtype=int)
        axis.bar(x, values, bottom=bottom, label=category)
        bottom += values
    axis.set_xticks(x, samplers)
    axis.set_ylabel("Completed trials violating constraints")
    axis.set_title("Pilot constraint violations")
    if samplers:
        axis.legend(title="Violation")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_combined_report(frames: Iterable[pd.DataFrame], output_dir: Path) -> dict[str, Any]:
    nonempty = [frame for frame in frames if not frame.empty]
    trials = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_dir / "trial_history.csv", index=False, encoding="utf-8")
    write_json({"trials": _records(trials)}, output_dir / "trial_history.json")

    front = pareto_front(trials) if not trials.empty else trials.copy()
    front.to_csv(output_dir / "pareto_front.csv", index=False, encoding="utf-8")
    write_json({"trials": _records(front)}, output_dir / "pareto_front.json")
    scenarios = select_scenario_candidates(front) if not trials.empty else {}
    write_json(scenarios, output_dir / "scenario_candidates.json")

    summary: dict[str, Any] = {
        "attempted": int(len(trials)),
        "completed": int(trials["state"].eq("COMPLETE").sum()) if not trials.empty else 0,
        "failed": int(trials["state"].eq("FAIL").sum()) if not trials.empty else 0,
        "feasible": int(
            (trials["state"].eq("COMPLETE") & trials["feasible"].fillna(False)).sum()
        ) if not trials.empty else 0,
        "pareto_points": int(len(front)),
        "by_sampler": {},
    }
    if not trials.empty:
        for sampler, group in trials.groupby("sampler"):
            summary["by_sampler"][sampler] = {
                "attempted": int(len(group)),
                "completed": int(group["state"].eq("COMPLETE").sum()),
                "failed": int(group["state"].eq("FAIL").sum()),
                "feasible": int(
                    (group["state"].eq("COMPLETE") & group["feasible"].fillna(False)).sum()
                ),
                "runtime_seconds": float(group["runtime_seconds"].fillna(0).sum()),
            }
    write_json(summary, output_dir / "summary.json")

    parameter_columns = [
        "param_n_estimators", "param_max_depth", "param_min_samples_leaf",
        "param_max_features", "param_class_weight",
    ]
    duplicate_diagnostics: list[dict[str, Any]] = []

    def native_parameter(value: Any) -> Any:
        if pd.isna(value):
            return None
        return value.item() if isinstance(value, np.generic) else value

    if not trials.empty and all(column in trials for column in parameter_columns):
        completed = trials.loc[trials["state"].eq("COMPLETE")]
        for parameters, group in completed.groupby(parameter_columns, dropna=False):
            if len(group) < 2 or group["sampler"].nunique() < 2:
                continue
            latencies = group["latency_per_1000_samples_ms"].astype(float)
            median = float(latencies.median())
            duplicate_diagnostics.append({
                "parameters": {
                    column.removeprefix("param_"): native_parameter(value)
                    for column, value in zip(parameter_columns, parameters)
                },
                "observations": [
                    {
                        "sampler": str(row["sampler"]),
                        "trial_number": int(row["trial_number"]),
                        "latency_ms": float(row["latency_per_1000_samples_ms"]),
                    }
                    for _, row in group.iterrows()
                ],
                "absolute_range_ms": float(latencies.max() - latencies.min()),
                "relative_range_over_median": (
                    float((latencies.max() - latencies.min()) / median) if median else None
                ),
            })
    latency_diagnostics = {
        "purpose": "Repeated identical startup configurations expose timing reproducibility only.",
        "matched_configurations": len(duplicate_diagnostics),
        "comparisons": duplicate_diagnostics,
        "max_relative_range_over_median": max(
            (
                item["relative_range_over_median"]
                for item in duplicate_diagnostics
                if item["relative_range_over_median"] is not None
            ),
            default=None,
        ),
    }
    write_json(latency_diagnostics, output_dir / "latency_reproducibility.json")
    _plot_best_feasible(trials, output_dir / "best_feasible_ap_vs_trials.png")
    _plot_pareto(trials, front, output_dir / "ap_vs_latency_pareto.png")
    _plot_constraint_counts(trials, output_dir / "constraint_violations.png")
    return {
        "summary": summary,
        "scenarios": scenarios,
        "latency_reproducibility": latency_diagnostics,
    }
