"""Leakage-free, resumable Week 3 multi-objective Random Forest HPO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_info, threadpool_limits

from .benchmark import load_config, measure_latency
from .data import load_bank_data, prepare_features, sha256_file, write_json
from .pareto import constraint_status
from .paths import PROJECT_ROOT, resolve_project_path
from .pipeline import build_baseline_pipeline, build_hpo_pipeline
from .reporting import export_study, generate_combined_report


def load_hpo_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"HPO configuration must be a mapping: {config_path}")
    return config


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_baseline_reference(directory: Path) -> dict[str, Any]:
    """Load only pre-HPO artifacts; intentionally never reads test_metrics.json."""
    cv_summary_path = directory / "cv_summary.json"
    latency_path = directory / "latency.json"
    artifact_path = directory / "model_artifact.json"
    constraints_path = directory / "week3_constraints.json"
    cv_summary = _json_file(cv_summary_path)
    latency = _json_file(latency_path)
    artifact = _json_file(artifact_path)
    constraints = _json_file(constraints_path)

    baseline_latency = float(latency["latency_per_1000_samples_ms"]["median"])
    if not np.isclose(
        baseline_latency,
        float(constraints["baseline_measured_latency_per_1000_samples_ms"]),
    ):
        raise ValueError("Baseline latency and frozen constraint artifact disagree.")
    if not np.isclose(float(artifact["file_size_mb"]), float(constraints["baseline_model_size_mb"])):
        raise ValueError("Baseline model size and frozen constraint artifact disagree.")
    return {
        "cv_average_precision_mean": float(cv_summary["average_precision"]["mean"]),
        "cv_average_precision_std": float(cv_summary["average_precision"]["std"]),
        "baseline_latency_per_1000_samples_ms": baseline_latency,
        "baseline_model_size_mb": float(artifact["file_size_mb"]),
        "model_size_cap_mb": float(constraints["model_size_mb_max"]),
        "latency_cap_ms": float(constraints["derived_latency_per_1000_samples_ms_max"]),
        "infeasible_policy": constraints["infeasible_policy"],
        "artifact_hashes": {
            "cv_summary.json": sha256_file(cv_summary_path),
            "latency.json": sha256_file(latency_path),
            "model_artifact.json": sha256_file(artifact_path),
            "week3_constraints.json": sha256_file(constraints_path),
        },
    }


def build_fixed_folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_config: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    split = base_config["split"]
    cv = StratifiedKFold(
        n_splits=int(split["cv_folds"]),
        shuffle=bool(split["cv_shuffle"]),
        random_state=int(split["random_state"]),
    )
    return [(fit.astype(np.int64), validation.astype(np.int64)) for fit, validation in cv.split(X_train, y_train)]


def ensure_fold_artifact(
    path: Path,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train_indices: np.ndarray,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    fold_metadata: list[dict[str, Any]] = []
    for fold_number, (fit, validation) in enumerate(folds, start=1):
        arrays[f"fold_{fold_number}_fit_positions"] = fit
        arrays[f"fold_{fold_number}_validation_positions"] = validation
        fold_metadata.append(
            {
                "fold": fold_number,
                "fit_samples": int(len(fit)),
                "validation_samples": int(len(validation)),
                "fit_original_indices_sha256": _sha256_array(train_indices[fit]),
                "validation_original_indices_sha256": _sha256_array(train_indices[validation]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as stored:
            if set(stored.files) != set(arrays):
                raise ValueError(f"Existing CV fold artifact has incompatible keys: {path}")
            for key, expected in arrays.items():
                if not np.array_equal(stored[key], expected):
                    raise ValueError(f"Existing CV fold artifact differs for {key}: {path}")
    else:
        np.savez_compressed(path, **arrays)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
        "folds": fold_metadata,
    }


def suggest_parameters(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    n_estimators = search_space["n_estimators"]
    max_depth = search_space["max_depth"]
    min_samples_leaf = search_space["min_samples_leaf"]
    max_features = search_space["max_features"]
    class_weight = search_space["class_weight"]
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            int(n_estimators["low"]),
            int(n_estimators["high"]),
            step=int(n_estimators["step"]),
        ),
        "max_depth": trial.suggest_int(
            "max_depth", int(max_depth["low"]), int(max_depth["high"])
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", int(min_samples_leaf["low"]), int(min_samples_leaf["high"])
        ),
        "max_features": trial.suggest_float(
            "max_features", float(max_features["low"]), float(max_features["high"])
        ),
        "class_weight": trial.suggest_categorical("class_weight", class_weight["choices"]),
    }


def optuna_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, float]:
    values = trial.user_attrs.get("constraint_values")
    if values is None:
        return (float("inf"), float("inf"))
    return (float(values[0]), float(values[1]))


def create_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_config: dict[str, Any],
    baseline_reference: dict[str, Any],
    trial_artifact_dir: Path,
) -> Callable[[optuna.Trial], tuple[float, float]]:
    search_space = base_config["optimization"]["search_space"]
    compression = int(base_config["artifact"]["compression"])
    thread_limit = int(base_config["latency"]["thread_limit"])
    trial_artifact_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = suggest_parameters(trial, search_space)
        fold_ap: list[float] = []
        fold_latency: list[float] = []
        fold_latency_repeats: list[list[float]] = []
        fold_sizes: list[int] = []
        fold_fit_seconds: list[float] = []
        try:
            for fold_number, (fit_positions, validation_positions) in enumerate(folds, start=1):
                X_fit = X_train.iloc[fit_positions]
                y_fit = y_train.iloc[fit_positions]
                X_validation = X_train.iloc[validation_positions]
                y_validation = y_train.iloc[validation_positions]
                estimator = build_hpo_pipeline(X_fit, base_config, params)
                fit_started = time.perf_counter()
                with threadpool_limits(limits=thread_limit):
                    estimator.fit(X_fit, y_fit)
                fold_fit_seconds.append(float(time.perf_counter() - fit_started))
                with threadpool_limits(limits=thread_limit):
                    probability = estimator.predict_proba(X_validation)[:, 1]
                fold_ap.append(float(average_precision_score(y_validation, probability)))

                latency = measure_latency(estimator, X_validation, base_config)
                fold_latency.append(
                    float(latency["latency_per_1000_samples_ms"]["median"])
                )
                fold_latency_repeats.append(
                    [float(value) for value in latency["raw_repeat_ms"]]
                )

                model_path = trial_artifact_dir / f"trial_{trial.number}_fold_{fold_number}.joblib"
                try:
                    joblib.dump(estimator, model_path, compress=compression)
                    fold_sizes.append(int(model_path.stat().st_size))
                finally:
                    model_path.unlink(missing_ok=True)

            ap_mean = float(np.mean(fold_ap))
            ap_std = float(np.std(fold_ap, ddof=1))
            latency_max = float(max(fold_latency))
            size_max_bytes = int(max(fold_sizes))
            size_max_mb = size_max_bytes / 1_000_000
            status = constraint_status(
                size_max_mb,
                latency_max,
                model_size_cap_mb=float(baseline_reference["model_size_cap_mb"]),
                latency_cap_ms=float(baseline_reference["latency_cap_ms"]),
            )
            trial.set_user_attr("average_precision_cv_std", ap_std)
            trial.set_user_attr("fold_average_precision", fold_ap)
            trial.set_user_attr("fold_latency_medians_ms", fold_latency)
            trial.set_user_attr("fold_latency_repeats_ms", fold_latency_repeats)
            trial.set_user_attr("fold_model_sizes_bytes", fold_sizes)
            trial.set_user_attr("fold_fit_seconds", fold_fit_seconds)
            trial.set_user_attr("model_size_max_bytes", size_max_bytes)
            trial.set_user_attr("model_size_max_mb", size_max_mb)
            trial.set_user_attr("model_size_max_mib", size_max_bytes / (1024**2))
            trial.set_user_attr("size_violation_mb", status["size_violation_mb"])
            trial.set_user_attr("latency_violation_ms", status["latency_violation_ms"])
            trial.set_user_attr("constraint_values", [
                status["size_violation_mb"], status["latency_violation_ms"]
            ])
            trial.set_user_attr("feasible", bool(status["feasible"]))
            return ap_mean, latency_max
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    return objective


def calibrate_baseline_on_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_config: dict[str, Any],
    baseline_reference: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Measure the fixed baseline on train-only folds without counting it as an HPO trial."""
    output_path = run_dir / "baseline_cv_latency_calibration.json"
    if output_path.exists():
        return _json_file(output_path)
    artifact_dir = run_dir / ".trial_models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    compression = int(base_config["artifact"]["compression"])
    thread_limit = int(base_config["latency"]["thread_limit"])
    rows: list[dict[str, Any]] = []
    for fold_number, (fit_positions, validation_positions) in enumerate(folds, start=1):
        X_fit = X_train.iloc[fit_positions]
        y_fit = y_train.iloc[fit_positions]
        X_validation = X_train.iloc[validation_positions]
        y_validation = y_train.iloc[validation_positions]
        estimator = build_baseline_pipeline(X_fit, base_config)
        started = time.perf_counter()
        with threadpool_limits(limits=thread_limit):
            estimator.fit(X_fit, y_fit)
        fit_seconds = float(time.perf_counter() - started)
        with threadpool_limits(limits=thread_limit):
            probability = estimator.predict_proba(X_validation)[:, 1]
        latency = measure_latency(estimator, X_validation, base_config)
        model_path = artifact_dir / f"baseline_fold_{fold_number}.joblib"
        try:
            joblib.dump(estimator, model_path, compress=compression)
            size_bytes = int(model_path.stat().st_size)
        finally:
            model_path.unlink(missing_ok=True)
        rows.append({
            "fold": fold_number,
            "average_precision": float(average_precision_score(y_validation, probability)),
            "fit_seconds": fit_seconds,
            "latency_median_ms": float(latency["latency_per_1000_samples_ms"]["median"]),
            "latency_raw_repeats_ms": [float(value) for value in latency["raw_repeat_ms"]],
            "model_size_bytes": size_bytes,
            "model_size_mb": size_bytes / 1_000_000,
        })
    ap_values = [row["average_precision"] for row in rows]
    latency_values = [row["latency_median_ms"] for row in rows]
    size_values = [row["model_size_mb"] for row in rows]
    payload = {
        "role": "train-only baseline calibration; not an HPO trial",
        "holdout_used": False,
        "folds": rows,
        "average_precision_cv_mean": float(np.mean(ap_values)),
        "average_precision_cv_std": float(np.std(ap_values, ddof=1)),
        "average_precision_absolute_delta_from_week2": abs(
            float(np.mean(ap_values)) - float(baseline_reference["cv_average_precision_mean"])
        ),
        "latency_max_fold_median_ms": float(max(latency_values)),
        "latency_week2_single_model_median_ms": float(
            baseline_reference["baseline_latency_per_1000_samples_ms"]
        ),
        "latency_ratio_cv_max_to_week2_median": float(
            max(latency_values) / baseline_reference["baseline_latency_per_1000_samples_ms"]
        ),
        "latency_cap_ms_unchanged": float(baseline_reference["latency_cap_ms"]),
        "calibration_passes_frozen_latency_cap": bool(
            max(latency_values) <= baseline_reference["latency_cap_ms"]
        ),
        "model_size_max_mb": float(max(size_values)),
        "measurement_mechanics_match_week2": True,
        "comparability_caveat": (
            "Week 3 source is CV validation and aggregate is max fold median; Week 2 source was "
            "one holdout batch for one full-train model. The frozen cap is not changed."
        ),
    }
    write_json(payload, output_path)
    return payload


def make_sampler(name: str, seed: int, hpo_config: dict[str, Any]) -> optuna.samplers.BaseSampler:
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=int(hpo_config["tpe"]["n_startup_trials"]),
            constraints_func=optuna_constraints,
        )
    raise ValueError(f"Unsupported sampler: {name}")


def _study_storage_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def study_name_for(
    mode: str,
    sampler_name: str,
    seed: int,
    hpo_config: dict[str, Any],
) -> str:
    """Return a versioned study name without changing the historical pilot name."""
    base = f"bank_rf_{mode}_{sampler_name}_seed{seed}"
    version = hpo_config[mode].get("study_version")
    return f"{base}_{version}" if version else base


def run_study(
    *,
    sampler_name: str,
    mode: str,
    target_trials: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_config: dict[str, Any],
    hpo_config: dict[str, Any],
    baseline_reference: dict[str, Any],
    run_dir: Path,
    protocol_fingerprint: str,
) -> optuna.Study:
    seed = int(hpo_config["seed"])
    study_name = study_name_for(mode, sampler_name, seed, hpo_config)
    database_path = run_dir / f"{sampler_name}_seed{seed}.db"
    sampler = make_sampler(sampler_name, seed, hpo_config)
    study = optuna.create_study(
        study_name=study_name,
        storage=_study_storage_url(database_path),
        sampler=sampler,
        directions=list(hpo_config["directions"]),
        load_if_exists=True,
    )
    existing_fingerprint = study.user_attrs.get("protocol_fingerprint")
    if existing_fingerprint is not None and existing_fingerprint != protocol_fingerprint:
        raise ValueError(
            f"Study {study_name} exists with a different protocol fingerprint; refusing to mix runs."
        )
    study.set_user_attr("protocol_fingerprint", protocol_fingerprint)
    study.set_user_attr("sampler", sampler_name)
    study.set_user_attr("seed", seed)
    study.set_user_attr("mode", mode)
    study.set_user_attr("directions", list(hpo_config["directions"]))
    study.set_user_attr("holdout_used", False)

    already_attempted = len(study.trials)
    remaining = max(0, target_trials - already_attempted)
    if remaining:
        objective = create_objective(
            X_train,
            y_train,
            folds,
            base_config,
            baseline_reference,
            run_dir / ".trial_models",
        )
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=int(hpo_config["execution"]["optuna_n_jobs"]),
            catch=(Exception,),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    return study


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    return {"commit": commit or None, "dirty": bool(status)}


def _package_versions() -> dict[str, str]:
    names = [
        "joblib", "matplotlib", "numpy", "optuna", "pandas", "PyYAML",
        "scikit-learn", "scipy", "SQLAlchemy", "threadpoolctl",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def _sanitized_threadpools() -> list[dict[str, Any]]:
    allowed = {
        "user_api", "internal_api", "num_threads", "prefix", "version",
        "threading_layer", "architecture",
    }
    return [
        {key: value for key, value in pool.items() if key in allowed}
        for pool in threadpool_info()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable train-only Week 3 HPO.")
    parser.add_argument("--config", default="configs/hpo_week3.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--sampler", choices=["random", "tpe", "both"], default="both")
    parser.add_argument("--trials", type=int, default=None, help="Total target trials per sampler.")
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="Required safety gate for the pre-registered 40+40 full run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        hpo_config = load_hpo_config(args.config)
        if args.mode == "full" and not args.confirm_full:
            raise ValueError("Full mode requires --confirm-full after explicit user approval.")
        mode_config = hpo_config[args.mode]
        if args.mode == "full" and bool(mode_config.get("full_run_blocked", True)):
            raise ValueError(
                "Full mode remains blocked pending a passing latency gate, explicit user acceptance, "
                "and verification of the official timing objective."
            )
        target_trials = int(
            args.trials if args.trials is not None else mode_config["trials_per_sampler"]
        )
        if target_trials < 1:
            raise ValueError("--trials must be positive.")
        if args.mode == "pilot" and target_trials > int(mode_config["maximum_trials_per_sampler"]):
            raise ValueError("Pilot is hard-limited to at most 3 trials per sampler.")
        if args.mode == "full" and target_trials > int(mode_config["trials_per_sampler"]):
            raise ValueError("Full run cannot exceed the pre-registered 40 trials per sampler.")

        base_config = load_config(hpo_config["base_config"])
        if int(base_config["baseline"]["n_jobs"]) != int(
            hpo_config["execution"]["estimator_n_jobs"]
        ):
            raise ValueError("Estimator n_jobs differs from the frozen baseline protocol.")
        data_path = resolve_project_path(hpo_config["data_path"])
        data_audit = _json_file(resolve_project_path(hpo_config["data_audit_reference"]))
        actual_dataset_sha = sha256_file(data_path)
        if actual_dataset_sha != data_audit["sha256"]:
            raise ValueError("Dataset SHA-256 differs from the frozen Week 1 audit.")

        frame = load_bank_data(data_path)
        split_path = resolve_project_path(hpo_config["split_indices_reference"])
        with np.load(split_path) as split_artifact:
            train_indices = np.asarray(split_artifact["train_indices"], dtype=np.int64)
        train_frame = frame.iloc[train_indices].copy()
        del frame
        X_train, y_train = prepare_features(train_frame)
        del train_frame
        if "duration" in X_train.columns:
            raise RuntimeError("Leakage guard failed: duration is present in HPO features.")

        baseline_dir = resolve_project_path(hpo_config["baseline_reference_dir"])
        baseline_reference = load_baseline_reference(baseline_dir)
        folds = build_fixed_folds(X_train, y_train, base_config)
        output_root = resolve_project_path(hpo_config["output_root"])
        fold_artifact = ensure_fold_artifact(
            output_root / f"cv_folds_seed{int(hpo_config['seed'])}.npz",
            folds,
            train_indices,
        )
        run_dir = output_root / mode_config["storage_subdir"]
        run_dir.mkdir(parents=True, exist_ok=True)

        baseline_cv_calibration = calibrate_baseline_on_cv(
            X_train, y_train, folds, base_config, baseline_reference, run_dir
        )

        fingerprinted_hpo_config = {
            key: value for key, value in hpo_config.items() if key != "diagnostics"
        }
        protocol_payload = {
            "base_config": base_config,
            "hpo_config": fingerprinted_hpo_config,
            "dataset_sha256": actual_dataset_sha,
            "train_indices_sha256": _sha256_array(train_indices),
            "split_artifact_sha256": sha256_file(split_path),
            "folds": fold_artifact,
            "baseline_reference": baseline_reference,
        }
        protocol_fingerprint = _config_fingerprint(protocol_payload)
        protocol_snapshot = {
            **protocol_payload,
            "diagnostics_config": hpo_config.get("diagnostics", {}),
        }
        write_json(protocol_snapshot, run_dir / "protocol_snapshot.json")

        selected = ["random", "tpe"] if args.sampler == "both" else [args.sampler]
        for sampler_name in selected:
            run_study(
                sampler_name=sampler_name,
                mode=args.mode,
                target_trials=target_trials,
                X_train=X_train,
                y_train=y_train,
                folds=folds,
                base_config=base_config,
                hpo_config=hpo_config,
                baseline_reference=baseline_reference,
                run_dir=run_dir,
                protocol_fingerprint=protocol_fingerprint,
            )

        frames: list[pd.DataFrame] = []
        study_references: dict[str, Any] = {}
        seed = int(hpo_config["seed"])
        for sampler_name in hpo_config["samplers"]:
            database_path = run_dir / f"{sampler_name}_seed{seed}.db"
            if not database_path.exists():
                continue
            study_name = study_name_for(args.mode, sampler_name, seed, hpo_config)
            study = optuna.load_study(
                study_name=study_name, storage=_study_storage_url(database_path)
            )
            frames.append(export_study(study, run_dir, sampler_name=sampler_name, seed=seed))
            study_references[sampler_name] = {
                "study_name": study_name,
                "storage": database_path.name,
                "attempted": len(study.trials),
            }
        report = generate_combined_report(frames, run_dir)

        manifest = {
            "phase": f"week_3_{args.mode}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": float(time.perf_counter() - started),
            "protocol_fingerprint": protocol_fingerprint,
            "dataset_sha256": actual_dataset_sha,
            "dataset_shape_validated": [45_211, 17],
            "holdout_policy": {
                "week_2_baseline_holdout_was_evaluated_once": True,
                "week_3_holdout_features_or_labels_used": False,
                "training_rows": int(len(X_train)),
                "source": "train_indices only from frozen split_indices.npz",
            },
            "train_indices_sha256": _sha256_array(train_indices),
            "split_artifact_sha256": sha256_file(split_path),
            "cv_fold_artifact": fold_artifact,
            "baseline_reference": baseline_reference,
            "baseline_cv_calibration": baseline_cv_calibration,
            "measurement_protocol": {
                "includes": "pipeline preprocessing + predict_proba",
                "batch_size": int(base_config["latency"]["batch_size"]),
                "warmup_runs": int(base_config["latency"]["warmup_runs"]),
                "repeats": int(base_config["latency"]["repeats"]),
                "thread_limit": int(base_config["latency"]["thread_limit"]),
                "hpo_data_source": "each CV validation fold; no holdout",
                "hpo_latency_aggregation": hpo_config["aggregation"]["latency"],
                "hpo_model_size_aggregation": hpo_config["aggregation"]["model_size"],
                "comparability_caveat": (
                    "Measurement mechanics match baseline, but HPO uses CV validation folds and "
                    "a conservative max across folds whereas baseline used one final model and one holdout batch."
                ),
            },
            "studies": study_references,
            "result_summary": report["summary"],
            "scenario_candidates": report["scenarios"],
            "latency_reproducibility": report["latency_reproducibility"],
            "full_run_readiness": {
                "status": "BLOCKED_PENDING_PROTOCOL_APPROVAL_AND_TIMING_STABILIZATION",
                "reason": (
                    "Identical pilot configurations showed material latency variation; "
                    "CV max-fold aggregation also differs from the Week 2 single-model protocol."
                ),
                "diagnostic_relative_range_warning_threshold": float(
                    hpo_config["diagnostics"]["duplicate_latency_relative_range_warning"]
                ),
            },
            "git": _git_state(),
            "environment": {
                "python": platform.python_version(),
                "os": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "packages": _package_versions(),
                "threadpools_detected": _sanitized_threadpools(),
            },
        }
        write_json(manifest, run_dir / "run_manifest.json")
        print(json.dumps({"manifest": str(run_dir / 'run_manifest.json'), **report}, indent=2))
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"HPO aborted: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
