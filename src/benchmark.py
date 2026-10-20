"""Run the fixed Week 2 baseline and write reproducible artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_info, threadpool_limits

from .data import (
    DatasetNotFoundError,
    audit_dataframe,
    load_bank_data,
    make_holdout_split,
    prepare_features,
    sha256_file,
    write_json,
)
from .metrics import evaluate_estimator
from .paths import resolve_project_path
from .pipeline import build_baseline_pipeline, infer_column_groups

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _summarize_folds(fold_frame: pd.DataFrame) -> dict[str, Any]:
    metric_names = ["average_precision", "roc_auc", "f1", "precision", "recall"]
    return {
        metric: {
            "mean": float(fold_frame[metric].mean()),
            "std": float(fold_frame[metric].std(ddof=1)),
        }
        for metric in metric_names
    }


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    split_cfg = config["split"]
    threshold = float(config["baseline"]["threshold"])
    cv = StratifiedKFold(
        n_splits=int(split_cfg["cv_folds"]),
        shuffle=bool(split_cfg["cv_shuffle"]),
        random_state=int(split_cfg["random_state"]),
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold, (fit_positions, validation_positions) in enumerate(cv.split(X_train, y_train), start=1):
        estimator = build_baseline_pipeline(X_train.iloc[fit_positions], config)
        fit_started = time.perf_counter()
        estimator.fit(X_train.iloc[fit_positions], y_train.iloc[fit_positions])
        fit_seconds = time.perf_counter() - fit_started
        metrics = evaluate_estimator(
            estimator,
            X_train.iloc[validation_positions],
            y_train.iloc[validation_positions],
            threshold=threshold,
        )
        rows.append(
            {
                "fold": fold,
                "fit_samples": len(fit_positions),
                "validation_samples": len(validation_positions),
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )
    return pd.DataFrame(rows), time.perf_counter() - started


def measure_latency(
    estimator: Any,
    X: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    latency_cfg = config["latency"]
    requested_batch_size = int(latency_cfg["batch_size"])
    sample_size = min(requested_batch_size, len(X))
    if sample_size < 1:
        raise ValueError("Latency measurement requires at least one sample.")
    sample = X.sample(n=sample_size, random_state=int(config["split"]["random_state"]))
    warmups = int(latency_cfg["warmup_runs"])
    repeats = int(latency_cfg["repeats"])
    if repeats < 5:
        raise ValueError("Latency protocol requires at least 5 repeats.")
    durations_ms: list[float] = []
    with threadpool_limits(limits=int(latency_cfg["thread_limit"])):
        for _ in range(warmups):
            estimator.predict_proba(sample)
        for _ in range(repeats):
            started = time.perf_counter()
            estimator.predict_proba(sample)
            durations_ms.append((time.perf_counter() - started) * 1000.0)
    normalized = [duration * 1000.0 / sample_size for duration in durations_ms]
    q1, q3 = np.percentile(normalized, [25, 75])
    return {
        "includes": "pipeline preprocessing + predict_proba",
        "requested_batch_size": requested_batch_size,
        "measured_batch_size": sample_size,
        "warmup_runs": warmups,
        "repeats": repeats,
        "thread_limit": int(latency_cfg["thread_limit"]),
        "raw_repeat_ms": durations_ms,
        "latency_per_1000_samples_ms": {
            "median": float(statistics.median(normalized)),
            "std": float(statistics.stdev(normalized)) if len(normalized) > 1 else 0.0,
            "iqr": float(q3 - q1),
            "min": float(min(normalized)),
            "max": float(max(normalized)),
        },
    }


def _save_precision_recall_plot(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series, path: Path) -> None:
    probability = estimator.predict_proba(X_test)[:, 1]
    figure, axis = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probability, name="Random Forest baseline", ax=axis)
    axis.axhline(float(y_test.mean()), color="grey", linestyle="--", label="Positive prevalence")
    axis.set_title("Held-out test precision-recall curve")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _environment_metadata() -> dict[str, Any]:
    import scipy

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "threadpools_detected": threadpool_info(),
    }


def run_experiment(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    data_path: Path | None = None,
) -> dict[str, Any]:
    output = resolve_project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(config, output / "resolved_config.json")
    X, y = prepare_features(frame)
    split_cfg = config["split"]
    split = make_holdout_split(
        X,
        y,
        test_size=float(split_cfg["test_fraction"]),
        random_state=int(split_cfg["random_state"]),
    )
    if set(split.train_indices).intersection(split.test_indices):
        raise RuntimeError("Train/test indices overlap.")

    np.savez_compressed(
        output / "split_indices.npz",
        train_indices=split.train_indices,
        test_indices=split.test_indices,
    )
    fold_frame, cv_seconds = run_cross_validation(split.X_train, split.y_train, config)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8")
    cv_summary = _summarize_folds(fold_frame)
    write_json(cv_summary, output / "cv_summary.json")

    estimator = build_baseline_pipeline(split.X_train, config)
    fit_started = time.perf_counter()
    estimator.fit(split.X_train, split.y_train)
    final_fit_seconds = time.perf_counter() - fit_started
    test_metrics = evaluate_estimator(
        estimator,
        split.X_test,
        split.y_test,
        threshold=float(config["baseline"]["threshold"]),
    )
    write_json(test_metrics, output / "test_metrics.json")

    latency = measure_latency(estimator, split.X_test, config)
    write_json(latency, output / "latency.json")
    model_path = output / "model.joblib"
    compression = int(config["artifact"]["compression"])
    joblib.dump(estimator, model_path, compress=compression)
    artifact = {
        "path": str(model_path),
        "serialization": "joblib",
        "compression": compression,
        "file_size_bytes": model_path.stat().st_size,
        "file_size_mb": model_path.stat().st_size / 1_000_000,
        "file_size_mib": model_path.stat().st_size / (1024**2),
        "sha256": sha256_file(model_path),
        "note": "Serialized compressed file size; this is not in-memory RAM usage.",
    }
    write_json(artifact, output / "model_artifact.json")
    optimization_cfg = config["optimization"]
    measured_baseline_latency = latency["latency_per_1000_samples_ms"]["median"]
    week3_constraints = {
        "status": "registered_not_executed",
        "model_size_mb_max": float(optimization_cfg["constraints"]["model_size_mb_max"]),
        "baseline_model_size_mb": artifact["file_size_mb"],
        "baseline_measured_latency_per_1000_samples_ms": measured_baseline_latency,
        "latency_multiplier_max": float(
            optimization_cfg["constraints"]["latency_multiplier_of_measured_baseline_max"]
        ),
        "derived_latency_per_1000_samples_ms_max": measured_baseline_latency
        * float(optimization_cfg["constraints"]["latency_multiplier_of_measured_baseline_max"]),
        "infeasible_policy": optimization_cfg["constraints"]["infeasible_policy"],
    }
    write_json(week3_constraints, output / "week3_constraints.json")
    _save_precision_recall_plot(estimator, split.X_test, split.y_test, output / "precision_recall_test.png")

    numeric_columns, categorical_columns = infer_column_groups(X)
    metadata: dict[str, Any] = {
        "phase": "weeks_1_2_baseline",
        "data_audit": audit_dataframe(frame, file_path=data_path) if data_path else audit_dataframe(frame),
        "feature_policy": {
            "target": "y",
            "duration_removed_before_split": True,
            "feature_count": X.shape[1],
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        "split": {
            "test_fraction": float(split_cfg["test_fraction"]),
            "random_state": int(split_cfg["random_state"]),
            "stratified": True,
            "train_samples": len(split.train_indices),
            "test_samples": len(split.test_indices),
            "train_index_sha256": _hash_array(split.train_indices),
            "test_index_sha256": _hash_array(split.test_indices),
            "indices_artifact": "split_indices.npz",
            "indices_artifact_sha256": sha256_file(output / "split_indices.npz"),
            "test_access_policy": "Held untouched during CV; evaluated once after the fixed baseline was fit.",
            "train_class_counts": {str(k): int(v) for k, v in split.y_train.value_counts().items()},
            "test_class_counts": {str(k): int(v) for k, v in split.y_test.value_counts().items()},
        },
        "cross_validation": {
            "folds": int(split_cfg["cv_folds"]),
            "type": "StratifiedKFold",
            "shuffle": bool(split_cfg["cv_shuffle"]),
            "preprocessing_fit_inside_each_fold": True,
        },
        "baseline": config["baseline"],
        "timings_seconds": {
            "cross_validation_total": cv_seconds,
            "final_train_fit": final_fit_seconds,
            "fold_fit": fold_frame[["fold", "fit_seconds"]].to_dict(orient="records"),
        },
        "model_artifact": artifact,
        "environment": _environment_metadata(),
    }
    write_json(metadata, output / "metadata.json")
    return {
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
        "latency": latency,
        "model_artifact": artifact,
        "output_dir": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed Random Forest baseline.")
    parser.add_argument("--data", default="data/raw/bank-full.csv", help="Dataset path.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Experiment YAML path.")
    parser.add_argument("--output", default="results/baseline", help="Artifact directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    data_path = resolve_project_path(args.data)
    try:
        frame = load_bank_data(data_path)
        config = load_config(args.config)
        result = run_experiment(frame, config, args.output, data_path=data_path)
    except (DatasetNotFoundError, FileNotFoundError, KeyError, ValueError) as exc:
        print(exc)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
