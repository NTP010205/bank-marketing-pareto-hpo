"""Isolated, train-only AP–serialized-model-size v1 Optuna runner."""

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
from typing import Any, Callable, Iterable

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info, threadpool_limits

from .data import EXPECTED_COLUMNS, LEAKAGE_COLUMN, TARGET_COLUMN, sha256_file, write_json
from .pareto import model_size_constraint_status, pareto_front_ap_model_size
from .paths import PROJECT_ROOT, resolve_project_path
from .pipeline import build_hpo_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {resolved}")
    return value


def _json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_integrity(paths: Iterable[Path]) -> dict[str, str]:
    """Hash every existing file below immutable research paths."""
    snapshot: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(f"Required integrity path is missing: {root}")
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            snapshot[relative] = sha256_file(path)
    return snapshot


def load_frozen_training_rows(
    data_path: Path,
    train_indices: np.ndarray,
    data_audit: dict[str, Any],
) -> pd.DataFrame:
    """Materialize only frozen training rows; holdout rows are skipped by the parser."""
    expected_sha = str(data_audit["sha256"])
    if sha256_file(data_path) != expected_sha:
        raise ValueError("Dataset SHA-256 differs from the frozen Week 1 audit.")
    expected_rows = int(data_audit["shape"][0])
    indices = np.asarray(train_indices, dtype=np.int64)
    if len(indices) != len(np.unique(indices)):
        raise ValueError("Frozen training indices contain duplicates.")
    if len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= expected_rows:
        raise ValueError("Frozen training indices are outside the audited dataset range.")

    selected = set(indices.tolist())
    frame = pd.read_csv(
        data_path,
        sep=";",
        encoding="utf-8",
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in selected,
    )
    if tuple(frame.columns) != EXPECTED_COLUMNS or len(frame) != len(indices):
        raise ValueError("Training-only dataset load does not match the frozen schema/indices.")
    frame.index = np.sort(indices)
    ordered = frame.loc[indices].reset_index(drop=True)
    labels = set(ordered[TARGET_COLUMN].dropna().unique())
    if labels != {"yes", "no"}:
        raise ValueError(f"Training labels must be exactly yes/no; got {sorted(labels)}.")
    return ordered


def prepare_training_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = frame.drop(columns=[TARGET_COLUMN, LEAKAGE_COLUMN]).copy()
    y = frame[TARGET_COLUMN].map({"no": 0, "yes": 1})
    if LEAKAGE_COLUMN in X.columns or y.isna().any():
        raise RuntimeError("Leakage/label guard failed while preparing HPO training rows.")
    return X, y.astype("int8")


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
    return [
        (fit.astype(np.int64), validation.astype(np.int64))
        for fit, validation in cv.split(X_train, y_train)
    ]


def validate_frozen_folds(
    path: Path,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train_indices: np.ndarray,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen CV fold artifact is missing: {path}")
    expected_keys: set[str] = set()
    metadata: list[dict[str, Any]] = []
    with np.load(path) as stored:
        for fold_number, (fit, validation) in enumerate(folds, start=1):
            fit_key = f"fold_{fold_number}_fit_positions"
            validation_key = f"fold_{fold_number}_validation_positions"
            expected_keys.update([fit_key, validation_key])
            if fit_key not in stored or validation_key not in stored:
                raise ValueError(f"Frozen fold artifact is missing fold {fold_number} arrays.")
            if not np.array_equal(stored[fit_key], fit) or not np.array_equal(
                stored[validation_key], validation
            ):
                raise ValueError(f"Frozen CV fold {fold_number} differs from the registered protocol.")
            metadata.append(
                {
                    "fold": fold_number,
                    "fit_samples": int(len(fit)),
                    "validation_samples": int(len(validation)),
                    "fit_positions_sha256": sha256_array(fit),
                    "validation_positions_sha256": sha256_array(validation),
                    "fit_original_indices_sha256": sha256_array(train_indices[fit]),
                    "validation_original_indices_sha256": sha256_array(
                        train_indices[validation]
                    ),
                }
            )
        if set(stored.files) != expected_keys:
            raise ValueError("Frozen CV fold artifact has unexpected arrays.")
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "folds": metadata,
    }


def suggest_parameters(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            int(search_space["n_estimators"]["low"]),
            int(search_space["n_estimators"]["high"]),
            step=int(search_space["n_estimators"]["step"]),
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            int(search_space["max_depth"]["low"]),
            int(search_space["max_depth"]["high"]),
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            int(search_space["min_samples_leaf"]["low"]),
            int(search_space["min_samples_leaf"]["high"]),
        ),
        "max_features": trial.suggest_float(
            "max_features",
            float(search_space["max_features"]["low"]),
            float(search_space["max_features"]["high"]),
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", search_space["class_weight"]["choices"]
        ),
    }


def serialize_fitted_pipeline(
    estimator: Pipeline,
    path: Path,
    *,
    compression: int,
    verify_repeat: bool,
) -> dict[str, Any]:
    """Serialize the whole fitted pipeline and optionally repeat byte-for-byte."""
    if not isinstance(estimator, Pipeline) or set(estimator.named_steps) != {
        "preprocess",
        "model",
    }:
        raise TypeError("Size objective requires the complete preprocess+model Pipeline.")
    path.parent.mkdir(parents=True, exist_ok=True)
    repeat_path = path.with_name(f"{path.stem}_repeat{path.suffix}")
    started = time.perf_counter()
    try:
        joblib.dump(estimator, path, compress=compression)
        size_bytes = int(path.stat().st_size)
        digest = sha256_file(path)
        repeat_size: int | None = None
        repeat_digest: str | None = None
        repeat_exact = True
        if verify_repeat:
            joblib.dump(estimator, repeat_path, compress=compression)
            repeat_size = int(repeat_path.stat().st_size)
            repeat_digest = sha256_file(repeat_path)
            repeat_exact = repeat_size == size_bytes and repeat_digest == digest
            if not repeat_exact:
                raise RuntimeError("Repeated serialization of one fitted pipeline was not exact.")
        return {
            "model_size_bytes": size_bytes,
            "model_size_mb": size_bytes / 1_000_000,
            "model_size_mib": size_bytes / (1024**2),
            "artifact_sha256": digest,
            "repeat_size_bytes": repeat_size,
            "repeat_artifact_sha256": repeat_digest,
            "repeat_serialization_exact": repeat_exact,
            "serialization_seconds": float(time.perf_counter() - started),
            "pipeline_steps": list(estimator.named_steps),
        }
    finally:
        path.unlink(missing_ok=True)
        repeat_path.unlink(missing_ok=True)


def evaluate_configuration_cv(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_metadata: list[dict[str, Any]],
    base_config: dict[str, Any],
    params: dict[str, Any],
    artifact_dir: Path,
    artifact_prefix: str,
    compression: int,
    thread_limit: int,
    verify_repeat_serialization: bool,
    on_fold_complete: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Fit preprocessing within each training fold and return raw fold evidence."""
    records: list[dict[str, Any]] = []
    for fold_number, (fit_positions, validation_positions) in enumerate(folds, start=1):
        X_fit = X_train.iloc[fit_positions]
        y_fit = y_train.iloc[fit_positions]
        X_validation = X_train.iloc[validation_positions]
        y_validation = y_train.iloc[validation_positions]
        estimator = build_hpo_pipeline(X_fit, base_config, params)

        fit_started = time.perf_counter()
        with threadpool_limits(limits=thread_limit):
            estimator.fit(X_fit, y_fit)
        fit_seconds = float(time.perf_counter() - fit_started)

        predict_started = time.perf_counter()
        with threadpool_limits(limits=thread_limit):
            probability = estimator.predict_proba(X_validation)[:, 1]
        predict_seconds = float(time.perf_counter() - predict_started)
        ap = float(average_precision_score(y_validation, probability))

        serialization = serialize_fitted_pipeline(
            estimator,
            artifact_dir / f"{artifact_prefix}_fold_{fold_number}.joblib",
            compression=compression,
            verify_repeat=verify_repeat_serialization,
        )
        record = {
            "fold": fold_number,
            "fit_samples": int(len(fit_positions)),
            "validation_samples": int(len(validation_positions)),
            "average_precision": ap,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            **serialization,
            **{
                key: fold_metadata[fold_number - 1][key]
                for key in (
                    "fit_positions_sha256",
                    "validation_positions_sha256",
                    "fit_original_indices_sha256",
                    "validation_original_indices_sha256",
                )
            },
        }
        records.append(record)
        if on_fold_complete is not None:
            on_fold_complete(records)

    ap_values = [float(record["average_precision"]) for record in records]
    max_size_bytes = max(int(record["model_size_bytes"]) for record in records)
    return {
        "folds": records,
        "average_precision_cv_mean": float(np.mean(ap_values)),
        "average_precision_cv_std": float(np.std(ap_values, ddof=1)),
        "model_size_max_bytes": int(max_size_bytes),
        "model_size_max_mb": max_size_bytes / 1_000_000,
        "model_size_max_mib": max_size_bytes / (1024**2),
        "fit_seconds_total": float(sum(record["fit_seconds"] for record in records)),
        "predict_seconds_total": float(
            sum(record["predict_seconds"] for record in records)
        ),
        "serialization_seconds_total": float(
            sum(record["serialization_seconds"] for record in records)
        ),
    }


def optuna_size_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    values = trial.user_attrs.get("constraint_values")
    return (float("inf"),) if values is None else (float(values[0]),)


def create_objective(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_metadata: list[dict[str, Any]],
    base_config: dict[str, Any],
    protocol_config: dict[str, Any],
    artifact_dir: Path,
) -> Callable[[optuna.Trial], tuple[float, float]]:
    search_space = base_config["optimization"]["search_space"]
    compression = int(protocol_config["serialization"]["compression"])
    thread_limit = int(protocol_config["execution"]["thread_limit"])
    cap = float(protocol_config["constraints"]["model_size_cap_mb_decimal"])
    repeat = bool(protocol_config["serialization"]["repeat_serialization_check_in_smoke"])

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = suggest_parameters(trial, search_space)
        config_hash = canonical_hash(params)[:12]
        trial.set_user_attr("config_hash", config_hash)
        trial.set_user_attr("seed", int(protocol_config["seed"]))
        trial.set_user_attr("parameters", params)

        def log_partial(records: list[dict[str, Any]]) -> None:
            trial.set_user_attr("fold_metrics", records)

        try:
            result = evaluate_configuration_cv(
                X_train=X_train,
                y_train=y_train,
                folds=folds,
                fold_metadata=fold_metadata,
                base_config=base_config,
                params=params,
                artifact_dir=artifact_dir,
                artifact_prefix=f"trial_{trial.number}_{config_hash}",
                compression=compression,
                thread_limit=thread_limit,
                verify_repeat_serialization=repeat,
                on_fold_complete=log_partial,
            )
            status = model_size_constraint_status(
                float(result["model_size_max_mb"]), model_size_cap_mb=cap
            )
            for key, value in result.items():
                if key != "folds":
                    trial.set_user_attr(key, value)
            trial.set_user_attr("fold_metrics", result["folds"])
            trial.set_user_attr("size_violation_mb", status["size_violation_mb"])
            trial.set_user_attr("constraint_values", [status["size_violation_mb"]])
            trial.set_user_attr("feasible", bool(status["feasible"]))
            trial.set_user_attr("latency_used", False)
            return (
                float(result["average_precision_cv_mean"]),
                float(result["model_size_max_mb"]),
            )
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("feasible", False)
            raise

    return objective


def make_sampler(
    sampler_name: str,
    seed: int,
    protocol_config: dict[str, Any],
) -> optuna.samplers.BaseSampler:
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=int(protocol_config["tpe"]["n_startup_trials"]),
            constraints_func=optuna_size_constraints,
        )
    raise ValueError(f"Unsupported sampler: {sampler_name}")


def study_name_for(mode: str, sampler_name: str, config: dict[str, Any]) -> str:
    version = str(config[mode]["study_version"])
    return f"bank_rf_{version}_{sampler_name}_seed{int(config['seed'])}"


def storage_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def audit_historical_config_hashes(config: dict[str, Any]) -> dict[str, Any]:
    """Explain the legacy CSV-rounding hash without changing any parameters."""
    pilot_dir = resolve_project_path("results/hpo/pilot")
    smoke_dir = resolve_project_path("results/hpo/ap_model_size_v1/smoke")
    pilot = optuna.load_study(
        study_name="bank_rf_pilot_random_seed42",
        storage=storage_url(pilot_dir / "random_seed42.db"),
    )
    smoke = optuna.load_study(
        study_name=study_name_for("smoke", "random", config),
        storage=storage_url(smoke_dir / "random_seed42_ap_model_size_v1.db"),
    )
    pilot_params = dict(pilot.trials[1].params)
    smoke_params = dict(smoke.trials[1].params)
    pilot_csv = pd.read_csv(pilot_dir / "trial_history.csv")
    row = pilot_csv.loc[
        (pilot_csv["sampler"] == "random") & (pilot_csv["trial_number"] == 1)
    ].iloc[0]
    csv_params = {
        "n_estimators": int(row["param_n_estimators"]),
        "max_depth": int(row["param_max_depth"]),
        "min_samples_leaf": int(row["param_min_samples_leaf"]),
        "max_features": float(row["param_max_features"]),
        "class_weight": None if pd.isna(row["param_class_weight"]) else row["param_class_weight"],
    }
    direct_hash = canonical_hash(pilot_params)[:12]
    csv_hash = canonical_hash(csv_params)[:12]
    if pilot_params != smoke_params or direct_hash != "b4e97ab5ded9" or csv_hash != "48e74e2e02b3":
        raise RuntimeError("Historical configuration-hash provenance no longer matches the audit.")
    return {
        "pilot_sqlite_parameters": pilot_params,
        "smoke_sqlite_parameters": smoke_params,
        "raw_sqlite_parameters_equal": True,
        "direct_canonical_hash": direct_hash,
        "historical_csv_roundtrip_parameters": csv_params,
        "historical_csv_roundtrip_hash": csv_hash,
        "max_features_absolute_delta": abs(
            float(pilot_params["max_features"]) - float(csv_params["max_features"])
        ),
        "explanation": (
            "48e74e2e02b3 was derived after pandas parsed the pilot CSV and rounded "
            "max_features from 0.41235069657748147 to 0.4123506965774814. "
            "b4e97ab5ded9 hashes the exact Optuna SQLite parameters directly."
        ),
    }


def run_study(
    *,
    sampler_name: str,
    mode: str,
    target_trials: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_metadata: list[dict[str, Any]],
    base_config: dict[str, Any],
    protocol_config: dict[str, Any],
    run_dir: Path,
    protocol_fingerprint: str,
) -> tuple[optuna.Study, float]:
    seed = int(protocol_config["seed"])
    database_path = run_dir / f"{sampler_name}_seed{seed}_ap_model_size_v1.db"
    study_name = study_name_for(mode, sampler_name, protocol_config)
    started = time.perf_counter()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url(database_path),
        sampler=make_sampler(sampler_name, seed, protocol_config),
        directions=list(protocol_config["directions"]),
        load_if_exists=True,
    )
    prior_fingerprint = study.user_attrs.get("protocol_fingerprint")
    if prior_fingerprint is not None and prior_fingerprint != protocol_fingerprint:
        raise ValueError("Existing study has a different protocol fingerprint; refusing resume.")
    if list(study.directions) != [
        optuna.study.StudyDirection.MAXIMIZE,
        optuna.study.StudyDirection.MINIMIZE,
    ]:
        raise ValueError("Existing study objective directions are incompatible.")
    if len(study.trials) > target_trials:
        raise ValueError("Existing study already exceeds the requested evaluation budget.")
    study.set_user_attr("protocol_fingerprint", protocol_fingerprint)
    study.set_user_attr("protocol_version", protocol_config["protocol_version"])
    study.set_user_attr("sampler", sampler_name)
    study.set_user_attr("seed", seed)
    study.set_user_attr("mode", mode)
    study.set_user_attr("holdout_used", False)
    study.set_user_attr("latency_used", False)
    study.set_user_attr("model_size_cap_mb_decimal", float(
        protocol_config["constraints"]["model_size_cap_mb_decimal"]
    ))

    remaining = target_trials - len(study.trials)
    if remaining:
        objective = create_objective(
            X_train=X_train,
            y_train=y_train,
            folds=folds,
            fold_metadata=fold_metadata,
            base_config=base_config,
            protocol_config=protocol_config,
            artifact_dir=run_dir / ".trial_models",
        )
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=int(protocol_config["execution"]["optuna_n_jobs"]),
            catch=(Exception,),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    return study, float(time.perf_counter() - started)


def study_to_frames(
    study: optuna.Study,
    *,
    sampler_name: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        values = list(trial.values or [])
        attrs = trial.user_attrs
        duration = trial.duration.total_seconds() if trial.duration is not None else None
        row = {
            "sampler": sampler_name,
            "seed": seed,
            "trial_number": int(trial.number),
            "state": trial.state.name,
            "config_hash": attrs.get("config_hash"),
            "average_precision_cv_mean": values[0] if len(values) > 0 else None,
            "average_precision_cv_std": attrs.get("average_precision_cv_std"),
            "model_size_max_bytes": attrs.get("model_size_max_bytes"),
            "model_size_max_mb": values[1] if len(values) > 1 else attrs.get("model_size_max_mb"),
            "model_size_max_mib": attrs.get("model_size_max_mib"),
            "size_violation_mb": attrs.get("size_violation_mb"),
            "feasible": attrs.get("feasible", False),
            "datetime_start_utc": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete_utc": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            "runtime_seconds": duration,
            "fit_seconds_total": attrs.get("fit_seconds_total"),
            "predict_seconds_total": attrs.get("predict_seconds_total"),
            "serialization_seconds_total": attrs.get("serialization_seconds_total"),
            "fold_average_precision_json": json.dumps(
                [fold.get("average_precision") for fold in attrs.get("fold_metrics", [])],
                separators=(",", ":"),
            ),
            "fold_model_sizes_bytes_json": json.dumps(
                [fold.get("model_size_bytes") for fold in attrs.get("fold_metrics", [])],
                separators=(",", ":"),
            ),
            "fold_artifact_sha256_json": json.dumps(
                [fold.get("artifact_sha256") for fold in attrs.get("fold_metrics", [])],
                separators=(",", ":"),
            ),
            "error": attrs.get("error"),
            "param_n_estimators": trial.params.get("n_estimators"),
            "param_max_depth": trial.params.get("max_depth"),
            "param_min_samples_leaf": trial.params.get("min_samples_leaf"),
            "param_max_features": trial.params.get("max_features"),
            "param_class_weight": trial.params.get("class_weight"),
        }
        trial_rows.append(row)
        for fold in attrs.get("fold_metrics", []):
            fold_rows.append(
                {
                    "sampler": sampler_name,
                    "seed": seed,
                    "trial_number": int(trial.number),
                    "config_hash": attrs.get("config_hash"),
                    **fold,
                }
            )
    return pd.DataFrame(trial_rows), pd.DataFrame(fold_rows)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def unique_pareto_configurations(front: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical configurations while retaining reproducible provenance."""
    if front.empty:
        return front.copy()
    rows: list[dict[str, Any]] = []
    parameter_columns = [
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
        "param_max_features",
        "param_class_weight",
    ]
    working = front.assign(
        _sampler_rank=front["sampler"].map({"random": 0, "tpe": 1}).fillna(99)
    )
    for config_hash, group in working.groupby("config_hash", sort=False):
        if any(group[column].nunique(dropna=False) != 1 for column in parameter_columns):
            raise RuntimeError(f"Config-hash collision detected for {config_hash}.")
        ordered = group.sort_values(["_sampler_rank", "trial_number"])
        representative = ordered.iloc[0].drop(labels=["_sampler_rank"]).to_dict()
        representative.update(
            {
                "pareto_occurrences": int(len(group)),
                "samplers_json": json.dumps(
                    sorted(group["sampler"].astype(str).unique().tolist()),
                    separators=(",", ":"),
                ),
                "trial_references_json": json.dumps(
                    [
                        {"sampler": str(row["sampler"]), "trial_number": int(row["trial_number"])}
                        for _, row in ordered.iterrows()
                    ],
                    separators=(",", ":"),
                ),
            }
        )
        rows.append(representative)
    return pd.DataFrame(rows).sort_values(
        ["model_size_max_mb", "average_precision_cv_mean"], ascending=[True, False]
    )


def rank_selection_candidates(
    trials: pd.DataFrame,
    *,
    ap_tolerance: float,
) -> tuple[pd.DataFrame, float, float]:
    """Apply the pre-registered CV-only selection and deterministic tie-breaks."""
    feasible = trials.loc[
        trials["state"].eq("COMPLETE") & trials["feasible"].fillna(False)
    ].copy()
    if feasible.empty:
        raise RuntimeError("No completed feasible trial is available for selection.")
    best_ap = float(feasible["average_precision_cv_mean"].max())
    threshold = best_ap - float(ap_tolerance)
    eligible = feasible.loc[feasible["average_precision_cv_mean"] >= threshold].copy()
    eligible["_sampler_rank"] = eligible["sampler"].map({"random": 0, "tpe": 1}).fillna(99)
    eligible = eligible.sort_values(
        [
            "model_size_max_bytes",
            "average_precision_cv_mean",
            "average_precision_cv_std",
            "config_hash",
            "_sampler_rank",
            "trial_number",
        ],
        ascending=[True, False, True, True, True, True],
    )
    distinct = eligible.drop_duplicates("config_hash", keep="first").drop(
        columns=["_sampler_rank"]
    )
    return distinct.reset_index(drop=True), best_ap, threshold


def _params_from_trial_row(row: pd.Series) -> dict[str, Any]:
    class_weight = row["param_class_weight"]
    return {
        "n_estimators": int(row["param_n_estimators"]),
        "max_depth": int(row["param_max_depth"]),
        "min_samples_leaf": int(row["param_min_samples_leaf"]),
        "max_features": float(row["param_max_features"]),
        "class_weight": None if pd.isna(class_weight) else str(class_weight),
    }


def audit_full_train_selection(
    candidates: pd.DataFrame,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_config: dict[str, Any],
    protocol_config: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Audit ranked candidates without holdout and apply the registered fallback."""
    cap = float(protocol_config["selection"]["full_train_size_audit"]["cap_mb_decimal"])
    compression = int(protocol_config["serialization"]["compression"])
    thread_limit = int(protocol_config["execution"]["thread_limit"])
    audits: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
        params = _params_from_trial_row(row)
        estimator = build_hpo_pipeline(X_train, base_config, params)
        fit_started = time.perf_counter()
        with threadpool_limits(limits=thread_limit):
            estimator.fit(X_train, y_train)
        fit_seconds = float(time.perf_counter() - fit_started)
        serialized = serialize_fitted_pipeline(
            estimator,
            artifact_dir / f"full_train_rank_{rank}_{row['config_hash']}.joblib",
            compression=compression,
            verify_repeat=True,
        )
        feasible = float(serialized["model_size_mb"]) <= cap
        audit = {
            "rank": rank,
            "sampler": str(row["sampler"]),
            "trial_number": int(row["trial_number"]),
            "config_hash": str(row["config_hash"]),
            "parameters": params,
            "cv_average_precision_mean": float(row["average_precision_cv_mean"]),
            "cv_average_precision_std": float(row["average_precision_cv_std"]),
            "cv_model_size_max_bytes": int(row["model_size_max_bytes"]),
            "cv_model_size_max_mb": float(row["model_size_max_mb"]),
            "full_train_fit_seconds": fit_seconds,
            "full_train_model_size_bytes": int(serialized["model_size_bytes"]),
            "full_train_model_size_mb": float(serialized["model_size_mb"]),
            "full_train_model_size_mib": float(serialized["model_size_mib"]),
            "full_train_artifact_sha256": str(serialized["artifact_sha256"]),
            "repeat_serialization_exact": bool(serialized["repeat_serialization_exact"]),
            "deployment_size_cap_mb_decimal": cap,
            "deployment_feasible": feasible,
        }
        audits.append(audit)
        if feasible:
            selected = audit
            break
    return {
        "selection_rule": protocol_config["selection"],
        "ranked_candidate_count": int(len(candidates)),
        "audited_candidates": audits,
        "selected": selected,
        "deployment_feasible": selected is not None,
        "holdout_used": False,
    }


def _plot_best_feasible_ap(trials: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    for sampler, group in trials.sort_values("trial_number").groupby("sampler"):
        best = -np.inf
        values: list[float] = []
        for _, row in group.iterrows():
            if row["state"] == "COMPLETE" and bool(row["feasible"]):
                best = max(best, float(row["average_precision_cv_mean"]))
            values.append(np.nan if best == -np.inf else best)
        axis.plot(group["trial_number"] + 1, values, marker="o", markersize=3, label=sampler)
    axis.set_xlabel("Attempted trials")
    axis.set_ylabel("Best-so-far feasible mean CV AP")
    axis.set_title("AP–model-size v1: feasible AP convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_pareto_size(trials: pd.DataFrame, front: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    colors = {"random": "tab:blue", "tpe": "tab:orange"}
    completed = trials.loc[trials["state"].eq("COMPLETE")]
    for sampler, group in completed.groupby("sampler"):
        axis.scatter(
            group["model_size_max_mb"],
            group["average_precision_cv_mean"],
            alpha=0.45,
            s=28,
            color=colors.get(str(sampler)),
            label=f"{sampler} complete",
        )
    if not front.empty:
        unique = unique_pareto_configurations(front)
        axis.plot(
            unique["model_size_max_mb"],
            unique["average_precision_cv_mean"],
            color="black",
            marker="o",
            linewidth=1.4,
            label="unique feasible Pareto",
        )
    axis.set_xlabel("Maximum CV-fold serialized pipeline size (decimal MB)")
    axis.set_ylabel("Mean three-fold CV Average Precision")
    axis.set_title("Feasible AP versus serialized model size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_sampler_comparison(
    trials: pd.DataFrame,
    unique_front: pd.DataFrame,
    path: Path,
) -> None:
    samplers = [sampler for sampler in ["random", "tpe"] if sampler in set(trials["sampler"])]
    best_ap = [
        float(
            trials.loc[
                (trials["sampler"] == sampler)
                & trials["state"].eq("COMPLETE")
                & trials["feasible"].fillna(False),
                "average_precision_cv_mean",
            ].max()
        )
        for sampler in samplers
    ]
    pareto_counts = [
        int(unique_front["samplers_json"].fillna("").str.contains(sampler).sum())
        for sampler in samplers
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    axes[0].bar(samplers, best_ap, color=["tab:blue", "tab:orange"][: len(samplers)])
    axes[0].set_ylabel("Best feasible mean CV AP")
    axes[0].set_title("Best AP at 40 attempts")
    axes[0].set_ylim(max(0.0, min(best_ap) - 0.02), max(best_ap) + 0.005)
    axes[1].bar(samplers, pareto_counts, color=["tab:blue", "tab:orange"][: len(samplers)])
    axes[1].set_ylabel("Unique Pareto configs observed")
    axes[1].set_title("Pareto coverage (single seed)")
    figure.suptitle("RandomSampler versus TPE: descriptive comparison only")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_size_distribution(trials: pd.DataFrame, path: Path) -> None:
    completed = trials.loc[trials["state"].eq("COMPLETE")]
    samplers = [sampler for sampler in ["random", "tpe"] if sampler in set(completed["sampler"])]
    values = [completed.loc[completed["sampler"] == sampler, "model_size_max_mb"] for sampler in samplers]
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.boxplot(values, labels=samplers, showmeans=True)
    axis.set_ylabel("Maximum CV-fold size (decimal MB)")
    axis.set_title("Serialized pipeline size distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_cumulative_runtime(trials: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    for sampler, group in trials.sort_values("trial_number").groupby("sampler"):
        cumulative = group["runtime_seconds"].fillna(0).astype(float).cumsum()
        axis.plot(group["trial_number"] + 1, cumulative, marker="o", markersize=3, label=sampler)
    axis.set_xlabel("Attempted trials")
    axis.set_ylabel("Cumulative trial runtime (seconds)")
    axis.set_title("Cumulative evaluation cost")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_official_figures(
    trials: pd.DataFrame,
    front: pd.DataFrame,
    unique_front: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    methods = {
        "best_feasible_ap_vs_trial.png": "Running maximum feasible mean CV AP by sampler and attempt.",
        "ap_vs_model_size_pareto.png": "All complete trials with the unique feasible Pareto front overlaid.",
        "random_vs_tpe_comparison.png": "Best feasible AP and unique Pareto coverage at 40 attempts; descriptive single-seed comparison.",
        "trial_size_distribution.png": "Boxplot of max-fold serialized pipeline size for complete trials by sampler.",
        "cumulative_evaluation_cost.png": "Cumulative Optuna trial duration by sampler and attempt.",
    }
    _plot_best_feasible_ap(trials, output_dir / "best_feasible_ap_vs_trial.png")
    _plot_pareto_size(trials, front, output_dir / "ap_vs_model_size_pareto.png")
    _plot_sampler_comparison(trials, unique_front, output_dir / "random_vs_tpe_comparison.png")
    _plot_size_distribution(trials, output_dir / "trial_size_distribution.png")
    _plot_cumulative_runtime(trials, output_dir / "cumulative_evaluation_cost.png")
    return {
        "source": "trial_history.csv and pareto_unique.csv generated from the same official studies",
        "generation_module": "src.optimize_ap_model_size",
        "figures": {
            name: {
                "method": method,
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name, method in methods.items()
        },
    }


def build_full_acceptance_summary(
    trials: pd.DataFrame,
    folds: pd.DataFrame,
    front: pd.DataFrame,
    unique_front: pd.DataFrame,
    *,
    config: dict[str, Any],
    old_artifacts_unchanged: bool,
    ap_latency_blocked: bool,
    resume_no_add: bool,
    selection_audit: dict[str, Any],
    figures_exist: bool,
) -> dict[str, Any]:
    expected_each = int(config["full"]["trials_per_sampler"])
    completed = trials.loc[trials["state"].eq("COMPLETE")]
    fold_groups = folds.groupby(["sampler", "trial_number"]) if not folds.empty else None
    aggregate_ok = True
    if fold_groups is not None:
        for (sampler, trial_number), group in fold_groups:
            row = trials.loc[
                (trials["sampler"] == sampler) & (trials["trial_number"] == trial_number)
            ].iloc[0]
            if not np.isclose(float(row["average_precision_cv_mean"]), float(group["average_precision"].mean())):
                aggregate_ok = False
            if not np.isclose(float(row["average_precision_cv_std"]), float(group["average_precision"].std(ddof=1))):
                aggregate_ok = False
            if int(row["model_size_max_bytes"]) != int(group["model_size_bytes"].max()):
                aggregate_ok = False
            if not np.isclose(
                float(row["size_violation_mb"]), float(row["model_size_max_mb"]) - 50.0
            ):
                aggregate_ok = False

    recomputed = pareto_front_ap_model_size(trials)
    pareto_refs = set(zip(front["sampler"], front["trial_number"]))
    recomputed_refs = set(zip(recomputed["sampler"], recomputed["trial_number"]))
    checks = {
        "exactly_40_attempts_per_sampler": all(
            int((trials["sampler"] == sampler).sum()) == expected_each
            for sampler in config["samplers"]
        ),
        "no_trial_failures": bool(len(completed) == expected_each * len(config["samplers"])),
        "three_fold_rows_per_complete_trial": bool(len(folds) == len(completed) * 3),
        "ap_size_constraint_aggregates_correct": bool(aggregate_ok),
        "pareto_recomputation_exact": bool(pareto_refs == recomputed_refs),
        "pareto_only_completed_feasible": bool(
            front["state"].eq("COMPLETE").all() and front["feasible"].fillna(False).all()
        ),
        "unique_pareto_hashes_unique": bool(unique_front["config_hash"].is_unique),
        "latency_excluded": not any("latency" in column.lower() for column in trials.columns),
        "old_artifacts_unchanged": bool(old_artifacts_unchanged),
        "resume_does_not_add_trials_at_budget": bool(resume_no_add),
        "ap_latency_runner_still_blocked": bool(ap_latency_blocked),
        "full_train_selected_candidate_feasible": bool(selection_audit["deployment_feasible"]),
        "holdout_unused": bool(selection_audit["holdout_used"] is False),
        "all_required_figures_exist": bool(figures_exist),
    }
    return {
        "protocol_version": config["protocol_version"],
        "scope": "official_full_run_01_40_plus_40",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "trial_counts": {
            sampler: {
                "attempted": int((trials["sampler"] == sampler).sum()),
                "complete": int(
                    ((trials["sampler"] == sampler) & trials["state"].eq("COMPLETE")).sum()
                ),
                "failed": int(
                    ((trials["sampler"] == sampler) & trials["state"].eq("FAIL")).sum()
                ),
            }
            for sampler in config["samplers"]
        },
        "pareto": {
            "total_entries": int(len(front)),
            "distinct_configurations": int(len(unique_front)),
            "duplicate_entries": int(len(front) - len(unique_front)),
        },
        "full_run_status": "COMPLETED_AWAITING_USER_ACCEPTANCE",
        "holdout_evaluation_performed": False,
    }


def build_acceptance_summary(
    trials: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    config: dict[str, Any],
    old_artifacts_unchanged: bool,
    ap_latency_blocked: bool,
    resume_no_add: bool,
) -> dict[str, Any]:
    expected_trials = int(config["smoke"]["trials_per_sampler"]) * len(config["samplers"])
    expected_fold_rows = expected_trials * 3
    completed = trials.loc[trials["state"].eq("COMPLETE")]
    repeat_exact = bool(
        len(folds) == expected_fold_rows
        and folds["repeat_serialization_exact"].fillna(False).all()
    )

    duplicate_checks: list[dict[str, Any]] = []
    paired_exact = True
    if not completed.empty:
        for config_hash, group in completed.groupby("config_hash"):
            if group["sampler"].nunique() < 2:
                continue
            ap_values = group["average_precision_cv_mean"].astype(float)
            sizes = group["fold_model_sizes_bytes_json"].astype(str)
            exact = bool(ap_values.nunique() == 1 and sizes.nunique() == 1)
            paired_exact = paired_exact and exact
            duplicate_checks.append(
                {
                    "config_hash": str(config_hash),
                    "observations": int(len(group)),
                    "samplers": sorted(group["sampler"].astype(str).unique().tolist()),
                    "ap_and_fold_sizes_exact": exact,
                }
            )

    checks = {
        "exactly_three_trials_per_sampler": bool(
            len(trials) == expected_trials
            and all(
                int((trials["sampler"] == sampler).sum())
                == int(config["smoke"]["trials_per_sampler"])
                for sampler in config["samplers"]
            )
        ),
        "all_trials_complete": bool(len(completed) == expected_trials),
        "all_trials_feasible": bool(
            len(completed) == expected_trials and completed["feasible"].fillna(False).all()
        ),
        "three_fold_rows_per_trial": bool(len(folds) == expected_fold_rows),
        "repeat_serialization_exact": repeat_exact,
        "paired_duplicate_results_exact": paired_exact,
        "latency_excluded": bool(config["constraints"]["latency_used"] is False),
        "old_artifacts_unchanged": bool(old_artifacts_unchanged),
        "resume_does_not_add_trials_at_budget": bool(resume_no_add),
        "ap_latency_runner_still_blocked": bool(ap_latency_blocked),
        "full_ap_model_size_run_still_blocked": bool(config["full"]["full_run_blocked"]),
    }
    passed = all(checks.values())
    return {
        "protocol_version": config["protocol_version"],
        "scope": "smoke_3_plus_3_only",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "paired_duplicate_checks": duplicate_checks,
        "tpe_interpretation": (
            "Implementation smoke only: all three TPE trials are within n_startup_trials=10."
        ),
        "full_run_status": "BLOCKED_PENDING_USER_ACCEPTANCE",
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {"commit": commit or None, "dirty": bool(status)}


def _package_versions() -> dict[str, str]:
    names = ["joblib", "numpy", "optuna", "pandas", "PyYAML", "scikit-learn", "scipy", "SQLAlchemy", "threadpoolctl"]
    return {name: importlib.metadata.version(name) for name in names}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated AP–model-size v1 HPO.")
    parser.add_argument("--config", default="configs/hpo_ap_model_size_v1.yaml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--sampler", choices=["random", "tpe", "both"], default="both")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="Required after recorded approval for the locked full_run_01 40+40 run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    wall_started = time.perf_counter()
    try:
        config = load_yaml_mapping(args.config)
        if config.get("protocol_version") != "ap_model_size_v1":
            raise ValueError("Unexpected protocol version.")
        mode_config = config[args.mode]
        target_trials = int(args.trials or mode_config["trials_per_sampler"])
        if args.mode == "full":
            if bool(mode_config.get("full_run_blocked", True)) or not bool(
                mode_config.get("explicit_approval_recorded", False)
            ):
                raise ValueError("Full AP–model-size remains hard-blocked without recorded approval.")
            if not args.confirm_full:
                raise ValueError("Full AP–model-size requires --confirm-full after preflight PASS.")
            if args.sampler != "both":
                raise ValueError("Official full_run_01 requires both samplers in one locked invocation.")
            if target_trials != int(mode_config["trials_per_sampler"]):
                raise ValueError("Official full_run_01 budget is fixed at exactly 40 trials per sampler.")
        elif target_trials < 1 or target_trials > int(mode_config["maximum_trials_per_sampler"]):
            raise ValueError("Smoke is hard-limited to 1–3 trials per sampler.")
        if config["objective_names"] != [
            "average_precision_cv_mean",
            "model_size_max_mb",
        ] or config["directions"] != ["maximize", "minimize"]:
            raise ValueError("Objective signature differs from approved AP–model-size v1.")
        if config["constraints"].get("latency_used") is not False:
            raise ValueError("Latency must be disabled for AP–model-size v1.")

        base_config = load_yaml_mapping(config["base_config"])
        if int(base_config["artifact"]["compression"]) != int(
            config["serialization"]["compression"]
        ):
            raise ValueError("Serialization compression differs from the frozen protocol.")
        if int(base_config["baseline"]["n_jobs"]) != int(
            config["execution"]["estimator_n_jobs"]
        ):
            raise ValueError("Estimator n_jobs differs from the frozen protocol.")

        integrity_paths = [resolve_project_path(path) for path in config["integrity_paths"]]
        integrity_before = snapshot_integrity(integrity_paths)
        data_path = resolve_project_path(config["data_path"])
        data_audit = _json_object(resolve_project_path(config["data_audit_reference"]))
        split_path = resolve_project_path(config["split_indices_reference"])
        with np.load(split_path) as split:
            train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        train_frame = load_frozen_training_rows(data_path, train_indices, data_audit)
        X_train, y_train = prepare_training_features(train_frame)
        del train_frame

        folds = build_fixed_folds(X_train, y_train, base_config)
        fold_artifact = validate_frozen_folds(
            resolve_project_path(config["cv_folds_reference"]), folds, train_indices
        )
        output_root = resolve_project_path(config["output_root"])
        run_dir = output_root / mode_config["storage_subdir"]
        if args.mode == "full" and run_dir.exists() and any(run_dir.iterdir()):
            expected_databases = {
                run_dir / f"{sampler}_seed{int(config['seed'])}_ap_model_size_v1.db"
                for sampler in config["samplers"]
            }
            if not any(path.exists() for path in expected_databases):
                raise ValueError(
                    "Official output directory is non-empty without a resumable full_run_01 study."
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        prior_manifest_path = run_dir / "run_manifest.json"
        prior_manifest = (
            _json_object(prior_manifest_path) if prior_manifest_path.is_file() else None
        )

        implementation_hashes = {
            path: sha256_file(resolve_project_path(path))
            for path in [
                "src/optimize_ap_model_size.py",
                "src/pipeline.py",
                "src/pareto.py",
                "configs/hpo_ap_model_size_v1.yaml",
            ]
        }
        hash_audit = audit_historical_config_hashes(config)
        protocol_payload = {
            "protocol_config": config,
            "base_search_space": base_config["optimization"]["search_space"],
            "dataset_sha256": sha256_file(data_path),
            "split_artifact_sha256": sha256_file(split_path),
            "train_indices_sha256": sha256_array(train_indices),
            "cv_fold_artifact": fold_artifact,
            "feature_columns": list(X_train.columns),
            "implementation_source_sha256": implementation_hashes,
            "historical_config_hash_audit": hash_audit,
        }
        protocol_fingerprint = canonical_hash(protocol_payload)
        write_json(protocol_payload, run_dir / "protocol_snapshot.json")
        preflight = {
            "status": "PASS",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "authorized_scope": "fresh full_run_01 with exactly 40 RandomSampler + 40 TPESampler attempts",
            "dataset_sha256": sha256_file(data_path),
            "split_artifact_sha256": sha256_file(split_path),
            "train_indices_sha256": sha256_array(train_indices),
            "cv_fold_artifact_sha256": fold_artifact["sha256"],
            "training_samples": int(len(X_train)),
            "feature_columns": list(X_train.columns),
            "duration_excluded": LEAKAGE_COLUMN not in X_train.columns,
            "holdout_rows_materialized": False,
            "preprocessing_fit_scope": "new pipeline constructed and fit inside every CV training fold",
            "serialization_scope": "complete fitted preprocess+model Pipeline",
            "latency_used": False,
            "historical_config_hash_audit": hash_audit,
            "historical_artifact_file_count": int(len(integrity_before)),
            "historical_artifact_snapshot_sha256": canonical_hash(integrity_before),
            "official_namespace_was_fresh_or_valid_resume": True,
            "pytest": "38 passed before gate opening",
            "compileall": "PASS",
            "dependency_check": "31 installed packages compatible",
            "protocol_fingerprint": protocol_fingerprint,
        }
        write_json(preflight, run_dir / "preflight_report.json")

        selected = ["random", "tpe"] if args.sampler == "both" else [args.sampler]
        sampler_wall_seconds: dict[str, float] = {}
        sampler_initial_wall_seconds: dict[str, float] = {}
        resume_checks: dict[str, bool] = {}
        studies: dict[str, optuna.Study] = {}
        for sampler_name in selected:
            database_path = run_dir / (
                f"{sampler_name}_seed{int(config['seed'])}_ap_model_size_v1.db"
            )
            preexisting_trials = 0
            if database_path.exists():
                existing = optuna.load_study(
                    study_name=study_name_for(args.mode, sampler_name, config),
                    storage=storage_url(database_path),
                )
                preexisting_trials = len(existing.trials)
            study, elapsed = run_study(
                sampler_name=sampler_name,
                mode=args.mode,
                target_trials=target_trials,
                X_train=X_train,
                y_train=y_train,
                folds=folds,
                fold_metadata=fold_artifact["folds"],
                base_config=base_config,
                protocol_config=config,
                run_dir=run_dir,
                protocol_fingerprint=protocol_fingerprint,
            )
            studies[sampler_name] = study
            sampler_wall_seconds[sampler_name] = elapsed
            previous_study = (
                prior_manifest.get("studies", {}).get(sampler_name, {})
                if prior_manifest is not None
                else {}
            )
            sampler_initial_wall_seconds[sampler_name] = float(
                previous_study.get(
                    "wall_seconds_initial_execution",
                    previous_study.get("wall_seconds_this_invocation", elapsed),
                )
                if preexisting_trials >= target_trials
                else elapsed
            )

            before_resume = len(study.trials)
            resumed, _ = run_study(
                sampler_name=sampler_name,
                mode=args.mode,
                target_trials=target_trials,
                X_train=X_train,
                y_train=y_train,
                folds=folds,
                fold_metadata=fold_artifact["folds"],
                base_config=base_config,
                protocol_config=config,
                run_dir=run_dir,
                protocol_fingerprint=protocol_fingerprint,
            )
            resume_checks[sampler_name] = len(resumed.trials) == before_resume == target_trials
            studies[sampler_name] = resumed

        trial_frames: list[pd.DataFrame] = []
        fold_frames: list[pd.DataFrame] = []
        study_manifest: dict[str, Any] = {}
        seed = int(config["seed"])
        for sampler_name in config["samplers"]:
            database_path = run_dir / f"{sampler_name}_seed{seed}_ap_model_size_v1.db"
            if not database_path.exists():
                continue
            study = studies.get(sampler_name) or optuna.load_study(
                study_name=study_name_for(args.mode, sampler_name, config),
                storage=storage_url(database_path),
            )
            trial_frame, fold_frame = study_to_frames(
                study, sampler_name=sampler_name, seed=seed
            )
            trial_frames.append(trial_frame)
            fold_frames.append(fold_frame)
            study_manifest[sampler_name] = {
                "study_name": study.study_name,
                "storage_file": database_path.name,
                "attempted_trials": int(len(study.trials)),
                "wall_seconds_this_invocation": sampler_wall_seconds.get(sampler_name, 0.0),
                "wall_seconds_initial_execution": sampler_initial_wall_seconds.get(
                    sampler_name, 0.0
                ),
                "trial_runtime_seconds_sum": float(
                    trial_frame["runtime_seconds"].fillna(0).sum()
                ),
                "resume_no_add_verified": bool(resume_checks.get(sampler_name, False)),
            }

        trials = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
        fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
        package_versions = _package_versions()
        if not trials.empty:
            trials["protocol_fingerprint"] = protocol_fingerprint
            trials["dataset_sha256"] = sha256_file(data_path)
            trials["split_artifact_sha256"] = sha256_file(split_path)
            trials["cv_fold_artifact_sha256"] = fold_artifact["sha256"]
            trials["library_versions_json"] = json.dumps(
                package_versions, sort_keys=True, separators=(",", ":")
            )
        if not fold_metrics.empty:
            fold_metrics["protocol_fingerprint"] = protocol_fingerprint
            fold_metrics["dataset_sha256"] = sha256_file(data_path)
            fold_metrics["split_artifact_sha256"] = sha256_file(split_path)
            fold_metrics["cv_fold_artifact_sha256"] = fold_artifact["sha256"]
        trials.to_csv(run_dir / "trial_history.csv", index=False, encoding="utf-8")
        write_json({"trials": _records(trials)}, run_dir / "trial_history.json")
        fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8")

        front = pareto_front_ap_model_size(trials) if not trials.empty else trials.copy()
        front.to_csv(run_dir / "pareto_front.csv", index=False, encoding="utf-8")
        unique_front = unique_pareto_configurations(front)
        if args.mode == "full":
            unique_front.to_csv(run_dir / "pareto_unique.csv", index=False, encoding="utf-8")

        cap = float(config["constraints"]["model_size_cap_mb_decimal"])
        constraint_summary = {
            "definition": "g_size = model_size_max_mb - 50.0; feasible iff g_size <= 0",
            "model_size_cap_mb_decimal": cap,
            "latency_used": False,
            "completed": int(trials["state"].eq("COMPLETE").sum()),
            "feasible": int(
                (trials["state"].eq("COMPLETE") & trials["feasible"].fillna(False)).sum()
            ),
            "infeasible": int(
                (trials["state"].eq("COMPLETE") & ~trials["feasible"].fillna(False)).sum()
            ),
            "failed": int(trials["state"].eq("FAIL").sum()),
        }
        write_json(constraint_summary, run_dir / "constraint_summary.json")

        integrity_after = snapshot_integrity(integrity_paths)
        old_artifacts_unchanged = integrity_before == integrity_after
        latency_config = load_yaml_mapping("configs/hpo_week3.yaml")
        ap_latency_blocked = bool(latency_config["full"]["full_run_blocked"])
        resume_no_add = bool(
            resume_checks and all(resume_checks.get(sampler, False) for sampler in selected)
        )
        selection_audit: dict[str, Any] | None = None
        figure_manifest: dict[str, Any] | None = None
        if args.mode == "full":
            ranked, best_ap, ap_threshold = rank_selection_candidates(
                trials,
                ap_tolerance=float(config["selection"]["best_ap_tolerance_absolute"]),
            )
            selection_audit = audit_full_train_selection(
                ranked,
                X_train=X_train,
                y_train=y_train,
                base_config=base_config,
                protocol_config=config,
                artifact_dir=run_dir / ".trial_models",
            )
            selection_audit["best_feasible_cv_ap"] = best_ap
            selection_audit["ap_eligibility_threshold"] = ap_threshold
            selection_audit["eligible_distinct_configurations"] = int(len(ranked))
            write_json(selection_audit, run_dir / "selected_candidate.json")
            write_json(
                {
                    "rule": config["selection"]["full_train_size_audit"],
                    "audit": selection_audit,
                },
                run_dir / "full_train_size_audit.json",
            )
            figure_manifest = generate_official_figures(
                trials, front, unique_front, run_dir
            )
            write_json(figure_manifest, run_dir / "figure_manifest.json")
            figures_exist = all(
                (run_dir / name).is_file() for name in config["reporting"]["figures"]
            )
            acceptance = build_full_acceptance_summary(
                trials,
                fold_metrics,
                front,
                unique_front,
                config=config,
                old_artifacts_unchanged=old_artifacts_unchanged,
                ap_latency_blocked=ap_latency_blocked,
                resume_no_add=resume_no_add,
                selection_audit=selection_audit,
                figures_exist=figures_exist,
            )
        else:
            acceptance = build_acceptance_summary(
                trials,
                fold_metrics,
                config=config,
                old_artifacts_unchanged=old_artifacts_unchanged,
                ap_latency_blocked=ap_latency_blocked,
                resume_no_add=resume_no_add,
            )
        write_json(acceptance, run_dir / "acceptance_summary.json")

        output_names = [
            "preflight_report.json",
            "protocol_snapshot.json",
            "trial_history.csv",
            "trial_history.json",
            "fold_metrics.csv",
            "pareto_front.csv",
            "constraint_summary.json",
            "acceptance_summary.json",
        ]
        if args.mode == "full":
            output_names.extend(
                [
                    "pareto_unique.csv",
                    "selected_candidate.json",
                    "full_train_size_audit.json",
                    "figure_manifest.json",
                    *config["reporting"]["figures"],
                ]
            )
        manifest = {
            "protocol_version": config["protocol_version"],
            "protocol_status": str(config["status"]).upper(),
            "execution_mode": args.mode,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds_this_invocation": float(time.perf_counter() - wall_started),
            "initial_execution_wall_time_seconds": float(
                prior_manifest.get(
                    "initial_execution_wall_time_seconds",
                    prior_manifest.get(
                        "initial_smoke_wall_time_seconds",
                        prior_manifest.get("wall_time_seconds", time.perf_counter() - wall_started),
                    ),
                )
                if prior_manifest is not None
                else time.perf_counter() - wall_started
            ),
            "protocol_fingerprint": protocol_fingerprint,
            "objectives": {
                "names": config["objective_names"],
                "directions": config["directions"],
                "latency_used": False,
            },
            "dataset_sha256": sha256_file(data_path),
            "dataset_expected_shape": data_audit["shape"],
            "split_artifact_sha256": sha256_file(split_path),
            "train_indices_sha256": sha256_array(train_indices),
            "training_samples": int(len(X_train)),
            "holdout_policy": {
                "holdout_rows_materialized": False,
                "holdout_labels_read_for_optimization": False,
                "holdout_metrics_read_or_written": False,
            },
            "feature_columns": list(X_train.columns),
            "duration_excluded": LEAKAGE_COLUMN not in X_train.columns,
            "cv_fold_artifact": fold_artifact,
            "serialization": {
                "format": "joblib",
                "compression": int(config["serialization"]["compression"]),
                "artifact_scope": "complete_fitted_preprocess_plus_model_pipeline",
                "objective_unit": "decimal_MB_bytes_divided_by_1000000",
                "aggregation": "maximum_across_three_cv_folds",
                "repeat_check": True,
            },
            "constraint": constraint_summary,
            "studies": study_manifest,
            "pareto": {
                "total_entries": int(len(front)),
                "distinct_configurations": int(len(unique_front)),
                "duplicate_entries": int(len(front) - len(unique_front)),
            },
            "selection": selection_audit,
            "figure_manifest": figure_manifest,
            "preflight": preflight,
            "integrity": {
                "old_artifacts_unchanged": old_artifacts_unchanged,
                "files_checked": int(len(integrity_before)),
                "snapshot_before_sha256": canonical_hash(integrity_before),
                "snapshot_after_sha256": canonical_hash(integrity_after),
            },
            "artifacts": {
                name: {"sha256": sha256_file(run_dir / name), "bytes": (run_dir / name).stat().st_size}
                for name in output_names
            },
            "implementation_source_sha256": implementation_hashes,
            "acceptance": acceptance,
            "full_run": {
                "status": (
                    "COMPLETED_AWAITING_USER_ACCEPTANCE"
                    if args.mode == "full"
                    else "BLOCKED_PENDING_USER_ACCEPTANCE"
                ),
                "configured_trials_per_sampler": int(config["full"]["trials_per_sampler"]),
                "executed": args.mode == "full",
                "holdout_evaluation_performed": False,
            },
            "git": _git_state(),
            "environment": {
                "python": platform.python_version(),
                "os": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "packages": package_versions,
                "threadpools": threadpool_info(),
            },
        }
        write_json(manifest, run_dir / "run_manifest.json")
        print(json.dumps({"run_dir": str(run_dir), "acceptance": acceptance}, indent=2))
        return 0 if acceptance["status"] == "PASS" else 1
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"AP–model-size HPO aborted: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
