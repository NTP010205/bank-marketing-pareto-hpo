"""Read-only Week 4 analysis for the official AP–model-size experiment.

This module never fits a model, creates an Optuna trial, or materializes the
frozen holdout partition.  It derives tables and figures from immutable logs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .data import EXPECTED_COLUMNS, LEAKAGE_COLUMN, TARGET_COLUMN, sha256_file
from .pareto import pareto_front_ap_model_size

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_RELATIVE = Path("results/hpo/ap_model_size_v1/full_run_01")
BASELINE_RELATIVE = Path("results/baseline")
ANALYSIS_RELATIVE = Path("results/analysis/week4")
FIGURES_RELATIVE = Path("figures/week4")
DATASET_RELATIVE = Path("data/raw/bank-full.csv")
SPLIT_RELATIVE = BASELINE_RELATIVE / "split_indices.npz"


@dataclass(frozen=True)
class Week4Sources:
    root: Path
    official_dir: Path
    baseline_dir: Path
    trials: pd.DataFrame
    folds: pd.DataFrame
    pareto: pd.DataFrame
    manifest: dict[str, Any]
    figure_manifest: dict[str, Any]
    selected: dict[str, Any]
    full_train_audit: dict[str, Any]
    baseline_cv: dict[str, Any]
    baseline_model: dict[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_sources(root: Path = PROJECT_ROOT) -> Week4Sources:
    root = root.resolve()
    official = root / OFFICIAL_RELATIVE
    baseline = root / BASELINE_RELATIVE
    return Week4Sources(
        root=root,
        official_dir=official,
        baseline_dir=baseline,
        trials=pd.read_csv(official / "trial_history.csv"),
        folds=pd.read_csv(official / "fold_metrics.csv"),
        pareto=pd.read_csv(official / "pareto_unique.csv"),
        manifest=read_json(official / "run_manifest.json"),
        figure_manifest=read_json(official / "figure_manifest.json"),
        selected=read_json(official / "selected_candidate.json"),
        full_train_audit=read_json(official / "full_train_size_audit.json"),
        baseline_cv=read_json(baseline / "cv_summary.json"),
        baseline_model=read_json(baseline / "model_artifact.json"),
    )


def _json_trial_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("trials", payload.get("records", []))
    if not isinstance(payload, list):
        raise ValueError("trial_history.json must contain a JSON list of trials")
    return payload


def _sqlite_objectives(db_path: Path) -> pd.DataFrame:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT trials.number, trials.state, trial_values.objective,
                   trial_values.value
            FROM trials
            JOIN trial_values ON trial_values.trial_id = trials.trial_id
            ORDER BY trials.number, trial_values.objective
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=["trial_number", "state", "objective", "value"])


def validate_sources(sources: Week4Sources) -> dict[str, Any]:
    required = [
        "run_manifest.json", "trial_history.csv", "trial_history.json",
        "fold_metrics.csv", "pareto_front.csv", "pareto_unique.csv",
        "selected_candidate.json", "full_train_size_audit.json",
        "figure_manifest.json", "acceptance_summary.json",
    ]
    missing = [name for name in required if not (sources.official_dir / name).is_file()]

    artifact_mismatches: list[dict[str, str]] = []
    for name, metadata in sources.manifest["artifacts"].items():
        path = sources.official_dir / name
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != metadata["sha256"]:
            artifact_mismatches.append(
                {"file": name, "expected": metadata["sha256"], "actual": actual}
            )

    figure_mismatches: list[dict[str, str]] = []
    for name, metadata in sources.figure_manifest["figures"].items():
        path = sources.official_dir / name
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != metadata["sha256"]:
            figure_mismatches.append(
                {"file": name, "expected": metadata["sha256"], "actual": actual}
            )

    json_records = _json_trial_records(sources.official_dir / "trial_history.json")
    json_frame = pd.DataFrame(json_records)
    csv_ap = sources.trials["average_precision_cv_mean"].astype(float).to_numpy()
    json_ap = json_frame["average_precision_cv_mean"].astype(float).to_numpy()
    csv_json_checks = {
        "row_count_equal": len(sources.trials) == len(json_frame) == 80,
        "config_hash_sequence_equal": (
            sources.trials["config_hash"].astype(str).tolist()
            == json_frame["config_hash"].astype(str).tolist()
        ),
        # pandas.DataFrame.to_json used by the locked runner records 10 decimal
        # digits by default. CSV and SQLite retain more digits, so JSON equality
        # is audited at a documented tolerance while SQLite remains the exact
        # objective-value cross-check.
        "ap_values_close_at_1e_9": bool(
            np.allclose(csv_ap, json_ap, rtol=0.0, atol=1e-9)
        ),
        "ap_max_absolute_json_rounding_delta": float(np.max(np.abs(csv_ap - json_ap))),
        "size_bytes_equal": (
            sources.trials["model_size_max_bytes"].astype(int).tolist()
            == json_frame["model_size_max_bytes"].astype(int).tolist()
        ),
    }

    sqlite_checks: dict[str, Any] = {}
    for sampler in ("random", "tpe"):
        study = sources.manifest["studies"][sampler]
        objective_rows = _sqlite_objectives(sources.official_dir / study["storage_file"])
        csv_rows = sources.trials.loc[sources.trials["sampler"].eq(sampler)].sort_values(
            "trial_number"
        )
        ap = objective_rows.loc[objective_rows["objective"].eq(0), "value"].to_numpy(float)
        size = objective_rows.loc[objective_rows["objective"].eq(1), "value"].to_numpy(float)
        sqlite_checks[sampler] = {
            "trial_count": int(objective_rows["trial_number"].nunique()),
            "all_states_complete": bool(objective_rows["state"].eq("COMPLETE").all()),
            "ap_matches_csv": bool(
                np.allclose(ap, csv_rows["average_precision_cv_mean"], rtol=0, atol=1e-15)
            ),
            "size_matches_csv": bool(
                np.allclose(size, csv_rows["model_size_max_mb"], rtol=0, atol=1e-12)
            ),
        }

    recomputed = pareto_front_ap_model_size(sources.trials)
    recomputed_hashes = set(recomputed["config_hash"].astype(str))
    official_hashes = set(sources.pareto["config_hash"].astype(str))
    selected = sources.selected["selected"]
    audit_selected = sources.full_train_audit["audit"]["selected"]

    checks = {
        "required_files_present": not missing,
        "manifest_artifact_hashes_match": not artifact_mismatches,
        "figure_manifest_hashes_match": not figure_mismatches,
        "csv_json_consistent": all(
            value for key, value in csv_json_checks.items() if key != "ap_max_absolute_json_rounding_delta"
        ),
        "sqlite_consistent": all(
            item["trial_count"] == 40
            and item["all_states_complete"]
            and item["ap_matches_csv"]
            and item["size_matches_csv"]
            for item in sqlite_checks.values()
        ),
        "fold_row_count_is_240": len(sources.folds) == 240,
        "three_folds_per_trial": bool(
            sources.folds.groupby(["sampler", "trial_number"]).size().eq(3).all()
        ),
        "pareto_hash_set_recomputed": recomputed_hashes == official_hashes,
        "selected_consistent_with_audit": (
            selected["config_hash"] == audit_selected["config_hash"] == "4cdd6dd35f1d"
        ),
        "selected_full_train_feasible": bool(audit_selected["deployment_feasible"]),
        "latency_absent_from_trial_columns": not any(
            "latency" in column.lower() for column in sources.trials.columns
        ),
        "holdout_absent_from_trial_columns": not any(
            token in column.lower()
            for column in sources.trials.columns
            for token in ("holdout", "test_")
        ),
        "manifest_says_holdout_unused": not bool(
            sources.manifest["full_run"]["holdout_evaluation_performed"]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_files": missing,
        "artifact_hash_mismatches": artifact_mismatches,
        "figure_hash_mismatches": figure_mismatches,
        "csv_json_checks": csv_json_checks,
        "sqlite_checks": sqlite_checks,
        "official_pareto_hash_count": len(official_hashes),
        "recomputed_pareto_hash_count": len(recomputed_hashes),
    }


def load_frozen_training_frame(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Materialize only frozen training rows and never expose holdout indices."""
    root = root.resolve()
    dataset = root / DATASET_RELATIVE
    split = root / SPLIT_RELATIVE
    manifest = read_json(root / OFFICIAL_RELATIVE / "run_manifest.json")
    actual_dataset_sha = sha256_file(dataset)
    if actual_dataset_sha != manifest["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 differs from the official full run")
    with np.load(split, allow_pickle=False) as arrays:
        train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    train_set = set(train_indices.tolist())
    safe_columns = [name for name in EXPECTED_COLUMNS if name != LEAKAGE_COLUMN]
    frame = pd.read_csv(
        dataset,
        sep=";",
        usecols=safe_columns,
        skiprows=lambda line_number: line_number != 0 and (line_number - 1) not in train_set,
        encoding="utf-8",
    )
    if len(frame) != int(manifest["training_samples"]):
        raise ValueError("Frozen training row count does not match the official manifest")
    if LEAKAGE_COLUMN in frame.columns:
        raise AssertionError("duration must not be materialized for Week 4 EDA")
    return frame


def frozen_train_eda_tables(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    categorical_columns = [
        column for column in frame.columns if column not in numeric_columns + [TARGET_COLUMN]
    ]
    target_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    numeric = frame[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
    numeric = numeric.rename(columns={"index": "feature"})
    category_rows: list[dict[str, Any]] = []
    for column in categorical_columns:
        counts = frame[column].value_counts(dropna=False)
        category_rows.append(
            {
                "feature": column,
                "unique_levels": int(frame[column].nunique(dropna=False)),
                "most_frequent": str(counts.index[0]),
                "most_frequent_count": int(counts.iloc[0]),
                "unknown_count": int(frame[column].eq("unknown").sum()),
            }
        )
    categorical = pd.DataFrame(category_rows)
    summary = {
        "scope": "frozen_training_rows_only",
        "rows": int(len(frame)),
        "columns_materialized": list(frame.columns),
        "duration_materialized": LEAKAGE_COLUMN in frame.columns,
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
        "positive_rate": float(frame[TARGET_COLUMN].eq("yes").mean()),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows_within_train": int(frame.duplicated().sum()),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "interpretation_policy": (
            "Descriptive train-only EDA; it cannot alter the locked model, features, "
            "search space, selected configuration, or constraints."
        ),
    }
    return summary, numeric, categorical


def dominance_table(sources: Week4Sources) -> pd.DataFrame:
    trials = sources.trials.loc[
        sources.trials["state"].eq("COMPLETE") & sources.trials["feasible"].fillna(False)
    ].copy()
    ap = trials["average_precision_cv_mean"].to_numpy(float)
    size = trials["model_size_max_mb"].to_numpy(float)
    dominated = np.zeros(len(trials), dtype=bool)
    for index in range(len(trials)):
        no_worse = (ap >= ap[index]) & (size <= size[index])
        strictly_better = (ap > ap[index]) | (size < size[index])
        dominated[index] = bool(np.any(no_worse & strictly_better))
    trials["dominated"] = dominated
    best_ap = float(ap.max())
    best_ap_size = float(trials.loc[trials["average_precision_cv_mean"].idxmax(), "model_size_max_mb"])
    trials["ap_loss_from_best"] = best_ap - trials["average_precision_cv_mean"].astype(float)
    trials["size_reduction_vs_best_ap_pct"] = (
        100.0 * (best_ap_size - trials["model_size_max_mb"].astype(float)) / best_ap_size
    )
    return trials


def analysis_tables(sources: Week4Sources) -> dict[str, pd.DataFrame]:
    trials = sources.trials.copy()
    complete = trials.loc[trials["state"].eq("COMPLETE")].copy()
    sampler_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    for sampler, group in complete.groupby("sampler", sort=True):
        group = group.sort_values("trial_number")
        sizes = group["model_size_max_mb"].astype(float)
        best = group.loc[group["average_precision_cv_mean"].astype(float).idxmax()]
        sampler_rows.append(
            {
                "sampler": sampler,
                "attempted": int(len(group)),
                "completed": int(group["state"].eq("COMPLETE").sum()),
                "feasible": int(group["feasible"].fillna(False).sum()),
                "best_trial": int(best["trial_number"]),
                "best_config_hash": str(best["config_hash"]),
                "best_ap_mean": float(best["average_precision_cv_mean"]),
                "best_ap_std": float(best["average_precision_cv_std"]),
                "best_ap_model_size_mb": float(best["model_size_max_mb"]),
                "model_size_min_mb": float(sizes.min()),
                "model_size_q1_mb": float(sizes.quantile(0.25)),
                "model_size_median_mb": float(sizes.median()),
                "model_size_q3_mb": float(sizes.quantile(0.75)),
                "model_size_max_mb": float(sizes.max()),
                "runtime_seconds_sum": float(group["runtime_seconds"].astype(float).sum()),
                "runtime_seconds_median": float(group["runtime_seconds"].astype(float).median()),
            }
        )
        for budget in (10, 20, 40):
            subset = group.loc[group["trial_number"].astype(int).lt(budget)]
            convergence_rows.append(
                {
                    "sampler": sampler,
                    "budget": budget,
                    "best_feasible_ap": float(
                        subset.loc[subset["feasible"].fillna(False), "average_precision_cv_mean"].max()
                    ),
                    "cumulative_runtime_seconds": float(subset["runtime_seconds"].sum()),
                }
            )

    selected = sources.selected["selected"]
    full_selected = sources.full_train_audit["audit"]["selected"]
    best_rows = []
    for sampler in ("random", "tpe"):
        group = complete.loc[complete["sampler"].eq(sampler)]
        row = group.loc[group["average_precision_cv_mean"].idxmax()]
        best_rows.append(
            {
                "model": f"Best {sampler.title()}",
                "role": "best_ap",
                "sampler": sampler,
                "trial_number": int(row["trial_number"]),
                "config_hash": str(row["config_hash"]),
                "cv_ap_mean": float(row["average_precision_cv_mean"]),
                "cv_ap_std": float(row["average_precision_cv_std"]),
                "cv_max_fold_size_mb": float(row["model_size_max_mb"]),
                "full_train_size_mb": np.nan,
            }
        )
    key_models = pd.DataFrame(
        [
            {
                "model": "Baseline",
                "role": "registered_baseline",
                "sampler": "baseline",
                "trial_number": np.nan,
                "config_hash": "baseline_week2",
                "cv_ap_mean": float(sources.baseline_cv["average_precision"]["mean"]),
                "cv_ap_std": float(sources.baseline_cv["average_precision"]["std"]),
                "cv_max_fold_size_mb": np.nan,
                "full_train_size_mb": float(sources.baseline_model["file_size_mb"]),
            },
            *best_rows,
            {
                "model": "Selected TPE",
                "role": "near_best_smallest_size",
                "sampler": str(selected["sampler"]),
                "trial_number": int(selected["trial_number"]),
                "config_hash": str(selected["config_hash"]),
                "cv_ap_mean": float(selected["cv_average_precision_mean"]),
                "cv_ap_std": float(selected["cv_average_precision_std"]),
                "cv_max_fold_size_mb": float(selected["cv_model_size_max_mb"]),
                "full_train_size_mb": float(full_selected["full_train_model_size_mb"]),
            },
        ]
    )
    dominance = dominance_table(sources)
    best_ap = float(complete["average_precision_cv_mean"].max())
    near_best = dominance.loc[dominance["average_precision_cv_mean"].ge(best_ap - 0.005)].copy()
    near_best["_sampler_rank"] = near_best["sampler"].map({"random": 0, "tpe": 1}).fillna(2)
    near_best = near_best.sort_values(
        [
            "model_size_max_bytes", "average_precision_cv_mean",
            "average_precision_cv_std", "config_hash", "_sampler_rank", "trial_number",
        ],
        ascending=[True, False, True, True, True, True],
    ).drop_duplicates("config_hash", keep="first")
    near_best = near_best.drop(columns="_sampler_rank")
    near_best["selection_rank"] = np.arange(1, len(near_best) + 1)
    return {
        "sampler_summary": pd.DataFrame(sampler_rows),
        "convergence_summary": pd.DataFrame(convergence_rows),
        "key_models": key_models,
        "dominance": dominance,
        "near_best": near_best,
    }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, Any]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.tight_layout()
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {
        "png": str(png.relative_to(PROJECT_ROOT)),
        "png_sha256": sha256_file(png),
        "png_bytes": png.stat().st_size,
        "pdf": str(pdf.relative_to(PROJECT_ROOT)),
        "pdf_sha256": sha256_file(pdf),
        "pdf_bytes": pdf.stat().st_size,
    }


def generate_figures(
    sources: Week4Sources, tables: dict[str, pd.DataFrame], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    colors = {"random": "#4472C4", "tpe": "#ED7D31"}
    records: dict[str, Any] = {}
    trials = sources.trials.sort_values(["sampler", "trial_number"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for sampler, group in trials.groupby("sampler"):
        feasible_ap = group["average_precision_cv_mean"].where(group["feasible"].fillna(False))
        ax.step(
            group["trial_number"].astype(int) + 1,
            feasible_ap.cummax(),
            where="post",
            linewidth=2,
            color=colors[sampler],
            label=sampler.title(),
        )
    ax.set(xlabel="Trial attempts", ylabel="Best feasible mean CV AP", title="Best-so-far CV Average Precision")
    ax.grid(alpha=0.25)
    ax.legend()
    records["01_best_so_far_cv_ap"] = _save_figure(fig, output_dir, "01_best_so_far_cv_ap")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for sampler, group in trials.groupby("sampler"):
        ax.scatter(
            group["model_size_max_mb"], group["average_precision_cv_mean"],
            s=34, alpha=0.7, color=colors[sampler], label=sampler.title(),
        )
    ax.set(xlabel="Maximum CV-fold serialized pipeline size (decimal MB)", ylabel="Mean 3-fold CV AP", title="Average Precision versus serialized model size")
    ax.grid(alpha=0.25)
    ax.legend()
    records["02_ap_vs_model_size"] = _save_figure(fig, output_dir, "02_ap_vs_model_size")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    dominance = tables["dominance"]
    dominated = dominance.loc[dominance["dominated"]]
    ax.scatter(dominated["model_size_max_mb"], dominated["average_precision_cv_mean"], s=26, color="0.75", alpha=0.65, label="Dominated trial")
    ordered = sources.pareto.sort_values("model_size_max_mb")
    ax.plot(ordered["model_size_max_mb"], ordered["average_precision_cv_mean"], "o-", color="#2E8B57", linewidth=1.7, markersize=5, label="Distinct Pareto front")
    selected = sources.selected["selected"]
    ax.scatter([selected["cv_model_size_max_mb"]], [selected["cv_average_precision_mean"]], marker="*", s=230, color="#C00000", edgecolor="white", linewidth=0.8, zorder=5, label="Selected trade-off")
    ax.set(xlabel="Maximum CV-fold serialized pipeline size (decimal MB)", ylabel="Mean 3-fold CV AP", title="Feasible Pareto front and selected configuration")
    ax.grid(alpha=0.25)
    ax.legend()
    records["03_pareto_selected"] = _save_figure(fig, output_dir, "03_pareto_selected")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    samples = [trials.loc[trials["sampler"].eq(name), "model_size_max_mb"].to_numpy(float) for name in ("random", "tpe")]
    box = ax.boxplot(samples, tick_labels=["Random", "TPE"], patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], (colors["random"], colors["tpe"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set(ylabel="Maximum CV-fold size (decimal MB)", title="Distribution of serialized model sizes")
    ax.grid(axis="y", alpha=0.25)
    records["04_model_size_distribution"] = _save_figure(fig, output_dir, "04_model_size_distribution")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for sampler, group in trials.groupby("sampler"):
        group = group.sort_values("trial_number")
        ax.plot(group["trial_number"].astype(int) + 1, group["runtime_seconds"].astype(float).cumsum(), linewidth=2, color=colors[sampler], label=sampler.title())
    ax.set(xlabel="Trial attempts", ylabel="Cumulative trial duration (seconds)", title="Cumulative search evaluation cost")
    ax.grid(alpha=0.25)
    ax.legend()
    records["05_cumulative_search_time"] = _save_figure(fig, output_dir, "05_cumulative_search_time")

    key = tables["key_models"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    axes[0].bar(key["model"], key["cv_ap_mean"], yerr=key["cv_ap_std"], color=["#70AD47", colors["random"], colors["tpe"], "#C00000"], alpha=0.82, capsize=3)
    axes[0].set(ylabel="Mean 3-fold CV AP", title="Predictive ranking quality")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(axis="y", alpha=0.25)
    size_labels = ["Baseline\nfull-train", "Best Random\nCV max-fold", "Best TPE\nCV max-fold", "Selected\nCV max-fold", "Selected\nfull-train"]
    size_values = [float(key.loc[key["model"].eq("Baseline"), "full_train_size_mb"].iloc[0]), float(key.loc[key["model"].eq("Best Random"), "cv_max_fold_size_mb"].iloc[0]), float(key.loc[key["model"].eq("Best Tpe"), "cv_max_fold_size_mb"].iloc[0]), float(selected["cv_model_size_max_mb"]), float(sources.full_train_audit["audit"]["selected"]["full_train_model_size_mb"])]
    axes[1].bar(size_labels, size_values, color=["#70AD47", colors["random"], colors["tpe"], "#C00000", "#A00000"], alpha=0.82)
    axes[1].set(ylabel="Serialized pipeline size (decimal MB)", title="Size audit (scope shown in labels)")
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].grid(axis="y", alpha=0.25)
    records["06_key_models_comparison"] = _save_figure(fig, output_dir, "06_key_models_comparison")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    pareto = dominance.loc[~dominance["dominated"]].copy()
    ax.scatter(pareto["ap_loss_from_best"], pareto["size_reduction_vs_best_ap_pct"], s=52, color="#2E8B57", alpha=0.8, label="Pareto trial")
    selected_row = dominance.loc[dominance["config_hash"].eq(selected["config_hash"])].iloc[0]
    ax.scatter([selected_row["ap_loss_from_best"]], [selected_row["size_reduction_vs_best_ap_pct"]], marker="*", s=240, color="#C00000", edgecolor="white", linewidth=0.8, label="Selected trade-off")
    ax.axvline(0.005, color="0.35", linestyle="--", linewidth=1.2, label="Near-best AP tolerance")
    ax.set(xlabel="Absolute CV AP loss from best feasible AP", ylabel="CV max-fold size reduction vs best-AP model (%)", title="AP loss versus serialized-size reduction")
    ax.grid(alpha=0.25)
    ax.legend()
    records["07_ap_loss_vs_size_reduction"] = _save_figure(fig, output_dir, "07_ap_loss_vs_size_reduction")
    return records


def build_week4_outputs(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    sources = load_sources(root)
    integrity = validate_sources(sources)
    if integrity["status"] != "PASS":
        raise RuntimeError(f"Week 4 source integrity failed: {integrity}")
    analysis_dir = root / ANALYSIS_RELATIVE
    figures_dir = root / FIGURES_RELATIVE
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train = load_frozen_training_frame(root)
    eda_summary, numeric, categorical = frozen_train_eda_tables(train)
    tables = analysis_tables(sources)
    figures = generate_figures(sources, tables, figures_dir)

    tables["sampler_summary"].to_csv(analysis_dir / "sampler_summary.csv", index=False)
    tables["convergence_summary"].to_csv(analysis_dir / "convergence_summary.csv", index=False)
    tables["key_models"].to_csv(analysis_dir / "key_models.csv", index=False)
    tables["dominance"].to_csv(analysis_dir / "trial_dominance.csv", index=False)
    tables["near_best"].to_csv(analysis_dir / "near_best_candidates.csv", index=False)
    numeric.to_csv(analysis_dir / "frozen_train_numeric_summary.csv", index=False)
    categorical.to_csv(analysis_dir / "frozen_train_categorical_summary.csv", index=False)
    write_json(eda_summary, analysis_dir / "frozen_train_eda_summary.json")
    write_json(integrity, analysis_dir / "source_integrity_report.json")

    dominance = tables["dominance"]
    distinct_trials = dominance.drop_duplicates("config_hash")
    selected = sources.selected["selected"]
    selected_row = dominance.loc[dominance["config_hash"].eq(selected["config_hash"])].iloc[0]
    main_results = {
        "scope": "read_only_week4_analysis_of_ap_model_size_v1_full_run_01",
        "trial_attempts": int(len(sources.trials)),
        "completed": int(sources.trials["state"].eq("COMPLETE").sum()),
        "feasible": int(sources.trials["feasible"].fillna(False).sum()),
        "distinct_configurations": int(sources.trials["config_hash"].nunique()),
        "cross_sampler_duplicate_configurations": int(
            len(
                set(sources.trials.loc[sources.trials["sampler"].eq("random"), "config_hash"])
                & set(sources.trials.loc[sources.trials["sampler"].eq("tpe"), "config_hash"])
            )
        ),
        "pareto_entries": int(len(sources.pareto)),
        "dominated_trial_entries": int(dominance["dominated"].sum()),
        "dominated_distinct_configurations": int(distinct_trials["dominated"].sum()),
        "near_best_distinct_configurations": int(sources.selected["eligible_distinct_configurations"]),
        "selected_config_hash": selected["config_hash"],
        "selected_cv_ap_mean": selected["cv_average_precision_mean"],
        "selected_cv_ap_std": selected["cv_average_precision_std"],
        "selected_cv_max_fold_size_mb": selected["cv_model_size_max_mb"],
        "selected_full_train_size_mb": sources.full_train_audit["audit"]["selected"]["full_train_model_size_mb"],
        "selected_ap_loss_from_best": float(selected_row["ap_loss_from_best"]),
        "selected_size_reduction_vs_best_ap_pct": float(selected_row["size_reduction_vs_best_ap_pct"]),
        "size_cap_mb": 50.0,
        "size_cap_binding": False,
        "largest_observed_cv_max_fold_size_mb": float(sources.trials["model_size_max_mb"].max()),
        "holdout_used": False,
        "new_training_or_hpo_performed": False,
        "interpretation_warning": (
            "Single sampler seed: fold-level CV dispersion is not between-search-run uncertainty. "
            "Serialized size is not latency, RAM, or energy."
        ),
    }
    write_json(main_results, analysis_dir / "main_results.json")

    source_files = [
        sources.official_dir / "run_manifest.json",
        sources.official_dir / "trial_history.csv",
        sources.official_dir / "trial_history.json",
        sources.official_dir / "fold_metrics.csv",
        sources.official_dir / "pareto_unique.csv",
        sources.official_dir / "selected_candidate.json",
        sources.official_dir / "full_train_size_audit.json",
        sources.baseline_dir / "cv_summary.json",
        sources.baseline_dir / "model_artifact.json",
        root / SPLIT_RELATIVE,
        root / DATASET_RELATIVE,
    ]
    analysis_manifest = {
        "status": "PASS",
        "source_policy": "official full_run_01 plus frozen Week 1-2 baseline and frozen train rows only",
        "source_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in source_files
        },
        "script_sha256": sha256_file(root / "scripts/build_week4_figures.py"),
        "analysis_module_sha256": sha256_file(Path(__file__)),
        "figures": figures,
        "holdout_materialized": False,
        "training_performed": False,
        "hpo_performed": False,
    }
    write_json(analysis_manifest, analysis_dir / "analysis_manifest.json")
    return {"integrity": integrity, "main_results": main_results, "manifest": analysis_manifest}

