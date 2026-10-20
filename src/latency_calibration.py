"""Versioned, train-only latency calibration for the Week 3 safety gate."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from .benchmark import load_config
from .data import load_bank_data, prepare_features, sha256_file, write_json
from .optimize import build_fixed_folds, ensure_fold_artifact
from .paths import PROJECT_ROOT, resolve_project_path
from .pipeline import build_baseline_pipeline, build_hpo_pipeline


@dataclass(frozen=True)
class CalibrationConfiguration:
    config_id: str
    label: str
    params: dict[str, Any] | None
    source: str


def load_calibration_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Calibration config must be a mapping: {config_path}")
    return value


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def canonical_config_id(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def select_benchmark_positions(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    if batch_size < 1 or batch_size > len(train_indices):
        raise ValueError("Benchmark batch size must be within the frozen training partition.")
    rng = np.random.default_rng(seed)
    train_positions = rng.choice(len(train_indices), size=batch_size, replace=False).astype(np.int64)
    original_indices = train_indices[train_positions]
    if np.intersect1d(original_indices, test_indices).size:
        raise RuntimeError("Train-only benchmark batch overlaps the frozen holdout indices.")
    return train_positions, original_indices


def duration_ns_to_ms_per_1000(duration_ns: int, batch_size: int) -> float:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return float(duration_ns / 1_000_000.0 * 1000.0 / batch_size)


def summarize_repeats(durations_ns: Iterable[int], batch_size: int) -> dict[str, Any]:
    raw_ns = [int(value) for value in durations_ns]
    if not raw_ns:
        raise ValueError("At least one timed repeat is required.")
    raw_ms = [duration_ns_to_ms_per_1000(value, batch_size) for value in raw_ns]
    q1, q3 = np.percentile(raw_ms, [25, 75])
    return {
        "raw_duration_ns": raw_ns,
        "raw_ms_per_1000": raw_ms,
        "median_ms_per_1000": float(statistics.median(raw_ms)),
        "mean_ms_per_1000": float(statistics.mean(raw_ms)),
        "std_ms_per_1000": float(statistics.stdev(raw_ms)) if len(raw_ms) > 1 else 0.0,
        "iqr_ms_per_1000": float(q3 - q1),
        "min_ms_per_1000": float(min(raw_ms)),
        "max_ms_per_1000": float(max(raw_ms)),
    }


def relative_range(values: Iterable[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("relative_range requires values")
    mean = statistics.mean(numeric)
    return float((max(numeric) - min(numeric)) / mean) if mean else 0.0


def aggregate_session_summaries(session_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "config_id", "fold", "session", "session_median_ms_per_1000",
        "model_size_bytes", "measurement_failed", "contention_detected",
        "batch_original_indices_sha256",
    }
    missing = required.difference(session_frame.columns)
    if missing:
        raise ValueError(f"Session frame missing columns: {sorted(missing)}")
    fold_rows: list[dict[str, Any]] = []
    for (config_id, fold), group in session_frame.groupby(["config_id", "fold"]):
        medians = group["session_median_ms_per_1000"].astype(float).tolist()
        fold_rows.append({
            "config_id": config_id,
            "fold": int(fold),
            "session_medians_ms_per_1000_json": json.dumps(medians, separators=(",", ":")),
            "fold_median_of_session_medians_ms_per_1000": float(statistics.median(medians)),
            "session_median_relative_range": relative_range(medians),
            "session_median_iqr_ms_per_1000": float(
                np.percentile(medians, 75) - np.percentile(medians, 25)
            ),
            "model_size_bytes": int(group["model_size_bytes"].iloc[0]),
            "model_size_mb": int(group["model_size_bytes"].iloc[0]) / 1_000_000,
            "measurement_failed": bool(group["measurement_failed"].any()),
            "contention_detected": bool(group["contention_detected"].any()),
            "batch_original_indices_sha256": str(group["batch_original_indices_sha256"].iloc[0]),
        })
    fold_frame = pd.DataFrame(fold_rows)

    config_rows: list[dict[str, Any]] = []
    for config_id, group in fold_frame.groupby("config_id"):
        fold_medians = group["fold_median_of_session_medians_ms_per_1000"].astype(float).tolist()
        config_rows.append({
            "config_id": config_id,
            "fold_medians_ms_per_1000_json": json.dumps(fold_medians, separators=(",", ":")),
            "aggregate_max_fold_median_ms_per_1000": float(max(fold_medians)),
            "median_across_fold_medians_ms_per_1000": float(statistics.median(fold_medians)),
            "fold_median_relative_range": relative_range(fold_medians),
            "fold_median_iqr_ms_per_1000": float(
                np.percentile(fold_medians, 75) - np.percentile(fold_medians, 25)
            ),
            "model_size_max_bytes": int(group["model_size_bytes"].max()),
            "model_size_max_mb": int(group["model_size_bytes"].max()) / 1_000_000,
            "maximum_session_median_relative_range": float(
                group["session_median_relative_range"].max()
            ),
            "measurement_failed": bool(group["measurement_failed"].any()),
            "contention_detected": bool(group["contention_detected"].any()),
            "batch_original_indices_sha256": str(group["batch_original_indices_sha256"].iloc[0]),
        })
    return fold_frame, pd.DataFrame(config_rows)


def evaluate_stability_gate(
    session_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    config_frame: pd.DataFrame,
    *,
    expected_batch_hash: str,
    maximum_relative_range: float,
    expected_config_count: int,
    expected_folds: int,
    expected_sessions: int,
) -> dict[str, Any]:
    observed_hashes = sorted(session_frame["batch_original_indices_sha256"].dropna().unique())
    expected_rows = expected_config_count * expected_folds * expected_sessions
    failures: list[str] = []
    if len(session_frame) != expected_rows:
        failures.append(f"expected {expected_rows} session rows, observed {len(session_frame)}")
    if observed_hashes != [expected_batch_hash]:
        failures.append("not every measurement used the identical benchmark batch")
    if session_frame["measurement_failed"].any():
        failures.append("one or more measurements failed")
    if session_frame["contention_detected"].any():
        failures.append("one or more sessions detected major CPU contention")
    unstable = fold_frame.loc[
        fold_frame["session_median_relative_range"] > maximum_relative_range,
        ["config_id", "fold", "session_median_relative_range"],
    ]
    if not unstable.empty:
        failures.append("one or more config/fold session ranges exceed the stability criterion")
    if config_frame["config_id"].nunique() != expected_config_count:
        failures.append("configuration count is incomplete")
    return {
        "status": "PASS" if not failures else "FAIL",
        "full_run_blocked": True,
        "criterion": (
            "Every config/fold relative range across session medians <= "
            f"{maximum_relative_range:.2%}, one shared batch, no failures, no major contention."
        ),
        "criterion_is_engineering_gate_not_confidence_guarantee": True,
        "observed_batch_hashes": observed_hashes,
        "maximum_observed_session_median_relative_range": float(
            fold_frame["session_median_relative_range"].max()
        ),
        "unstable_config_folds": json.loads(unstable.to_json(orient="records")),
        "failure_reasons": failures,
        "cap_approval_required_even_if_stability_passes": True,
    }


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    def integer(self) -> int:
        return (int(self.high) << 32) | int(self.low)


def _windows_cpu_snapshot() -> tuple[int, int, int] | None:
    if os.name != "nt":
        return None
    idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
    ok = ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    )
    return (idle.integer(), kernel.integer(), user.integer()) if ok else None


def _system_cpu_percent(
    before: tuple[int, int, int] | None,
    after: tuple[int, int, int] | None,
) -> float | None:
    if before is None or after is None:
        return None
    idle = after[0] - before[0]
    total = (after[1] - before[1]) + (after[2] - before[2])
    return float(100.0 * (total - idle) / total) if total > 0 else None


def _sanitized_threadpools() -> list[dict[str, Any]]:
    allowed = {
        "user_api", "internal_api", "num_threads", "prefix", "version",
        "threading_layer", "architecture",
    }
    return [
        {key: value for key, value in pool.items() if key in allowed}
        for pool in threadpool_info()
    ]


def _priority_class() -> int | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.GetPriorityClass.restype = ctypes.c_uint
    value = int(kernel32.GetPriorityClass(kernel32.GetCurrentProcess()))
    return value or None


def _power_plan() -> str | None:
    if os.name != "nt":
        return None
    result = subprocess.run(
        ["powercfg", "/getactivescheme"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or result.stderr.strip() or None


def _directory_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def load_distinct_pilot_configurations(path: Path) -> tuple[list[CalibrationConfiguration], list[dict[str, Any]]]:
    frame = pd.read_csv(path)
    columns = [
        "param_n_estimators", "param_max_depth", "param_min_samples_leaf",
        "param_max_features", "param_class_weight",
    ]
    configurations: dict[str, CalibrationConfiguration] = {}
    observations: dict[str, list[float]] = {}
    for _, row in frame.loc[frame["state"].eq("COMPLETE")].iterrows():
        params = {
            "n_estimators": int(row["param_n_estimators"]),
            "max_depth": int(row["param_max_depth"]),
            "min_samples_leaf": int(row["param_min_samples_leaf"]),
            "max_features": float(row["param_max_features"]),
            "class_weight": None if pd.isna(row["param_class_weight"]) else str(row["param_class_weight"]),
        }
        config_id = canonical_config_id(params)
        configurations.setdefault(
            config_id,
            CalibrationConfiguration(config_id, f"pilot_{config_id}", params, "historical_pilot"),
        )
        observations.setdefault(config_id, []).append(float(row["latency_per_1000_samples_ms"]))
    if len(configurations) != 3:
        raise ValueError(f"Expected exactly 3 distinct pilot configurations; found {len(configurations)}")
    paired = [
        {
            "config_id": config_id,
            "parameters": configurations[config_id].params,
            "historical_latencies_ms_per_1000": values,
            "relative_range_over_mean": relative_range(values),
        }
        for config_id, values in sorted(observations.items())
    ]
    if any(len(item["historical_latencies_ms_per_1000"]) != 2 for item in paired):
        raise ValueError("Each pilot configuration must have paired Random/TPE observations.")
    baseline_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    }
    baseline_id = canonical_config_id(baseline_params)
    result = [
        CalibrationConfiguration(
            baseline_id, "week2_baseline_rf", None, "frozen_week2_baseline_config"
        ),
        *[configurations[key] for key in sorted(configurations)],
    ]
    return result, paired


def validate_versioned_output_path(path: Path) -> None:
    if not re.fullmatch(r"latency_calibration_v\d+", path.name):
        raise ValueError("Calibration output directory must be versioned as latency_calibration_vN.")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing calibration output: {path}")


def measure_session(
    estimator: Any,
    X_batch: pd.DataFrame,
    *,
    warmups: int,
    repeats: int,
    thread_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if estimator.named_steps["model"].n_jobs != 1:
        raise ValueError("Calibration requires RandomForestClassifier(n_jobs=1).")
    system_before = _windows_cpu_snapshot()
    process_before = time.process_time_ns()
    block_before = time.perf_counter_ns()
    raw: list[int] = []
    with threadpool_limits(limits=thread_limit):
        effective_threadpools = _sanitized_threadpools()
        for _ in range(warmups):
            estimator.predict_proba(X_batch)
        for _ in range(repeats):
            started = time.perf_counter_ns()
            estimator.predict_proba(X_batch)
            raw.append(time.perf_counter_ns() - started)
    block_after = time.perf_counter_ns()
    process_after = time.process_time_ns()
    system_after = _windows_cpu_snapshot()
    wall_ms = (block_after - block_before) / 1_000_000.0
    process_ms = (process_after - process_before) / 1_000_000.0
    environment = {
        "system_cpu_percent_during_block": _system_cpu_percent(system_before, system_after),
        "process_cpu_time_ms": process_ms,
        "wall_block_ms": wall_ms,
        "process_cpu_efficiency_percent": float(100.0 * process_ms / wall_ms) if wall_ms else None,
        "effective_threadpools": effective_threadpools,
    }
    return summarize_repeats(raw, len(X_batch)), environment


def _write_frames(
    raw_rows: list[dict[str, Any]],
    session_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    config_frame: pd.DataFrame,
    output: Path,
) -> None:
    pd.DataFrame(raw_rows).to_csv(output / "raw_measurements.csv", index=False, encoding="utf-8")
    session_frame.to_csv(output / "session_summary.csv", index=False, encoding="utf-8")
    fold_frame.to_csv(output / "fold_summary.csv", index=False, encoding="utf-8")
    config_frame.to_csv(output / "config_summary.csv", index=False, encoding="utf-8")
    write_json(
        {"measurements": json.loads(pd.DataFrame(raw_rows).to_json(orient="records"))},
        output / "raw_measurements.json",
    )
    write_json(
        {"sessions": json.loads(session_frame.to_json(orient="records"))},
        output / "session_summary.json",
    )
    write_json(
        {"folds": json.loads(fold_frame.to_json(orient="records"))},
        output / "fold_summary.json",
    )
    write_json(
        {"configurations": json.loads(config_frame.to_json(orient="records"))},
        output / "config_summary.json",
    )


def run_calibration(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve_project_path(config["output_dir"])
    validate_versioned_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(config, output / "resolved_calibration_config.json")
    started_wall = time.perf_counter()
    baseline_dir = resolve_project_path(config["baseline_reference_dir"])
    pilot_dir = resolve_project_path(config["pilot_reference_dir"])
    baseline_hashes_before = _directory_hashes(baseline_dir)
    pilot_hashes_before = _directory_hashes(pilot_dir)
    try:
        data_path = resolve_project_path(config["data_path"])
        audit = json.loads(resolve_project_path(config["data_audit_reference"]).read_text(encoding="utf-8"))
        if sha256_file(data_path) != audit["sha256"]:
            raise ValueError("Dataset differs from the frozen data audit fingerprint.")
        frame = load_bank_data(data_path)
        split_path = resolve_project_path(config["split_indices_reference"])
        with np.load(split_path) as split:
            train_indices = np.asarray(split["train_indices"], dtype=np.int64)
            test_indices = np.asarray(split["test_indices"], dtype=np.int64)
        batch_cfg = config["benchmark_batch"]
        batch_positions, batch_original = select_benchmark_positions(
            train_indices,
            test_indices,
            batch_size=int(batch_cfg["size"]),
            seed=int(batch_cfg["deterministic_seed"]),
        )
        batch_hash = sha256_array(batch_original)
        np.savez_compressed(
            output / "benchmark_batch_indices.npz",
            train_positions=batch_positions,
            original_indices=batch_original,
        )
        write_json({
            "source": "frozen training indices only",
            "labels_used": False,
            "batch_size": len(batch_original),
            "seed": int(batch_cfg["deterministic_seed"]),
            "train_positions_sha256": sha256_array(batch_positions),
            "original_indices_sha256": batch_hash,
            "holdout_overlap_count": int(np.intersect1d(batch_original, test_indices).size),
            "artifact_sha256": sha256_file(output / "benchmark_batch_indices.npz"),
        }, output / "benchmark_batch.json")

        train_frame = frame.iloc[train_indices].copy()
        del frame
        X_train, y_train = prepare_features(train_frame)
        del train_frame
        X_batch = X_train.iloc[batch_positions].copy()
        if "duration" in X_batch.columns:
            raise RuntimeError("Leakage guard failed: duration is present in benchmark features.")

        base_config = load_config("configs/experiment.yaml")
        folds = build_fixed_folds(X_train, y_train, base_config)
        fold_reference_path = resolve_project_path(config["cv_folds_reference"])
        if not fold_reference_path.is_file():
            raise FileNotFoundError(f"Frozen CV fold artifact is missing: {fold_reference_path}")
        fold_reference = ensure_fold_artifact(fold_reference_path, folds, train_indices)
        configurations, historical_pairs = load_distinct_pilot_configurations(
            resolve_project_path(config["pilot_history_reference"])
        )
        timing = config["timing"]
        gate_cfg = config["stability_gate"]
        if int(base_config["baseline"]["n_jobs"]) != int(timing["estimator_n_jobs"]):
            raise ValueError("Calibration estimator_n_jobs differs from the frozen baseline.")

        configuration_payload = {
            item.config_id: {
                "label": item.label,
                "params": item.params,
                "source": item.source,
            }
            for item in configurations
        }
        write_json({
            "configurations": configuration_payload,
            "historical_paired_latency_audit": historical_pairs,
        }, output / "configurations.json")

        raw_rows: list[dict[str, Any]] = []
        session_rows: list[dict[str, Any]] = []
        model_sizes: dict[tuple[str, int], int] = {}
        fit_metadata: list[dict[str, Any]] = []
        run_orders: list[dict[str, Any]] = []
        model_dir = output / ".models"
        model_dir.mkdir()

        for fold_number, (fit_positions, _) in enumerate(folds, start=1):
            X_fit = X_train.iloc[fit_positions]
            y_fit = y_train.iloc[fit_positions]
            fitted: dict[str, Any] = {}
            fit_order = list(configurations)
            random.Random(int(timing["interleave_seed"]) + fold_number).shuffle(fit_order)
            for item in fit_order:
                estimator = (
                    build_baseline_pipeline(X_fit, base_config)
                    if item.params is None
                    else build_hpo_pipeline(X_fit, base_config, item.params)
                )
                fit_started = time.perf_counter_ns()
                with threadpool_limits(limits=int(timing["thread_limit"])):
                    estimator.fit(X_fit, y_fit)
                fit_ns = time.perf_counter_ns() - fit_started
                fitted[item.config_id] = estimator
                fit_metadata.append({
                    "config_id": item.config_id,
                    "fold": fold_number,
                    "fit_duration_ns": int(fit_ns),
                    "fit_seconds": fit_ns / 1_000_000_000.0,
                    "fit_samples": int(len(fit_positions)),
                    "random_state": int(base_config["baseline"]["random_state"]),
                    "estimator_n_jobs": int(estimator.named_steps["model"].n_jobs),
                })

            for session in range(1, int(timing["sessions"]) + 1):
                gc.collect()
                order = list(configurations)
                order_seed = int(timing["interleave_seed"]) + fold_number * 1000 + session
                random.Random(order_seed).shuffle(order)
                run_orders.append({
                    "fold": fold_number,
                    "session": session,
                    "order_seed": order_seed,
                    "configuration_order": [item.config_id for item in order],
                })
                for order_position, item in enumerate(order, start=1):
                    timestamp_start = datetime.now(timezone.utc).isoformat()
                    failed = False
                    error: str | None = None
                    try:
                        summary, environment = measure_session(
                            fitted[item.config_id],
                            X_batch,
                            warmups=int(timing["warmup_calls_per_session"]),
                            repeats=int(timing["timed_repeats_per_session"]),
                            thread_limit=int(timing["thread_limit"]),
                        )
                    except Exception as exc:
                        failed = True
                        error = f"{type(exc).__name__}: {exc}"
                        summary = {
                            "raw_duration_ns": [], "raw_ms_per_1000": [],
                            "median_ms_per_1000": np.nan, "mean_ms_per_1000": np.nan,
                            "std_ms_per_1000": np.nan, "iqr_ms_per_1000": np.nan,
                            "min_ms_per_1000": np.nan, "max_ms_per_1000": np.nan,
                        }
                        environment = {
                            "system_cpu_percent_during_block": None,
                            "process_cpu_time_ms": None,
                            "wall_block_ms": None,
                            "process_cpu_efficiency_percent": None,
                            "effective_threadpools": [],
                        }
                    timestamp_end = datetime.now(timezone.utc).isoformat()
                    system_cpu = environment["system_cpu_percent_during_block"]
                    efficiency = environment["process_cpu_efficiency_percent"]
                    contention = bool(
                        (system_cpu is not None and system_cpu > float(gate_cfg["maximum_system_cpu_percent"]))
                        or (efficiency is not None and efficiency < float(gate_cfg["minimum_process_cpu_efficiency_percent"]))
                    )
                    for repeat_number, (duration_ns, duration_ms) in enumerate(
                        zip(summary["raw_duration_ns"], summary["raw_ms_per_1000"]), start=1
                    ):
                        raw_rows.append({
                            "config_id": item.config_id,
                            "fold": fold_number,
                            "session": session,
                            "order_position": order_position,
                            "repeat": repeat_number,
                            "duration_ns": duration_ns,
                            "duration_ms_per_1000": duration_ms,
                            "batch_size": len(X_batch),
                            "batch_original_indices_sha256": batch_hash,
                        })
                    session_rows.append({
                        "config_id": item.config_id,
                        "fold": fold_number,
                        "session": session,
                        "order_position": order_position,
                        "order_seed": order_seed,
                        "timestamp_start_utc": timestamp_start,
                        "timestamp_end_utc": timestamp_end,
                        "warmup_calls": int(timing["warmup_calls_per_session"]),
                        "timed_repeats": int(timing["timed_repeats_per_session"]),
                        "session_median_ms_per_1000": summary["median_ms_per_1000"],
                        "session_mean_ms_per_1000": summary["mean_ms_per_1000"],
                        "session_std_ms_per_1000": summary["std_ms_per_1000"],
                        "session_iqr_ms_per_1000": summary["iqr_ms_per_1000"],
                        "session_min_ms_per_1000": summary["min_ms_per_1000"],
                        "session_max_ms_per_1000": summary["max_ms_per_1000"],
                        "raw_duration_ns_json": json.dumps(summary["raw_duration_ns"], separators=(",", ":")),
                        "raw_ms_per_1000_json": json.dumps(summary["raw_ms_per_1000"], separators=(",", ":")),
                        "system_cpu_percent": system_cpu,
                        "process_cpu_efficiency_percent": efficiency,
                        "contention_detected": contention,
                        "measurement_failed": failed,
                        "error": error,
                        "batch_size": len(X_batch),
                        "batch_original_indices_sha256": batch_hash,
                        "model_size_bytes": 0,
                        "threadpools_json": json.dumps(environment["effective_threadpools"], separators=(",", ":")),
                    })

            for item in configurations:
                model_path = model_dir / f"{item.config_id}_fold_{fold_number}.joblib"
                try:
                    joblib.dump(
                        fitted[item.config_id],
                        model_path,
                        compress=int(base_config["artifact"]["compression"]),
                    )
                    model_sizes[(item.config_id, fold_number)] = int(model_path.stat().st_size)
                finally:
                    model_path.unlink(missing_ok=True)
            del fitted
            gc.collect()

        session_frame = pd.DataFrame(session_rows)
        for index, row in session_frame.iterrows():
            session_frame.at[index, "model_size_bytes"] = model_sizes[
                (str(row["config_id"]), int(row["fold"]))
            ]
        fold_frame, config_frame = aggregate_session_summaries(session_frame)
        labels = {item.config_id: item.label for item in configurations}
        params_json = {
            item.config_id: json.dumps(item.params, sort_keys=True, separators=(",", ":"))
            for item in configurations
        }
        for frame_to_label in (fold_frame, config_frame):
            frame_to_label.insert(1, "label", frame_to_label["config_id"].map(labels))
            frame_to_label.insert(2, "parameters_json", frame_to_label["config_id"].map(params_json))

        gate = evaluate_stability_gate(
            session_frame,
            fold_frame,
            config_frame,
            expected_batch_hash=batch_hash,
            maximum_relative_range=float(gate_cfg["maximum_session_median_relative_range"]),
            expected_config_count=len(configurations),
            expected_folds=len(folds),
            expected_sessions=int(timing["sessions"]),
        )
        cap = float(config["constraints"]["historical_latency_cap_ms_per_1000"])
        config_frame["historical_cap_ms_per_1000"] = cap
        config_frame["meets_historical_absolute_cap"] = (
            config_frame["aggregate_max_fold_median_ms_per_1000"] <= cap
        )
        _write_frames(raw_rows, session_frame, fold_frame, config_frame, output)
        write_json({"fit_metadata": fit_metadata}, output / "fit_metadata.json")
        write_json({"run_orders": run_orders}, output / "run_order.json")
        write_json(gate, output / "gate_decision.json")

        baseline_hashes_after = _directory_hashes(baseline_dir)
        pilot_hashes_after = _directory_hashes(pilot_dir)
        manifest = {
            "protocol_version": config["protocol_version"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": float(time.perf_counter() - started_wall),
            "dataset_sha256": sha256_file(data_path),
            "split_artifact_sha256": sha256_file(split_path),
            "cv_fold_artifact": fold_reference,
            "base_config_sha256": sha256_file(resolve_project_path("configs/experiment.yaml")),
            "train_indices_sha256": sha256_array(train_indices),
            "test_indices_used_only_for_overlap_assertion": True,
            "holdout_feature_rows_loaded_for_benchmark": False,
            "holdout_labels_used": False,
            "benchmark_batch": json.loads((output / "benchmark_batch.json").read_text(encoding="utf-8")),
            "timing_protocol": timing,
            "stability_gate": gate,
            "constraint_decision": {
                "historical_cap_ms_per_1000_unchanged": cap,
                "automatic_cap_change": False,
                "full_run_blocked": True,
                "approval_required": True,
            },
            "baseline_hashes_before": baseline_hashes_before,
            "baseline_hashes_after": baseline_hashes_after,
            "baseline_unchanged": baseline_hashes_before == baseline_hashes_after,
            "pilot_hashes_before": pilot_hashes_before,
            "pilot_hashes_after": pilot_hashes_after,
            "pilot_unchanged": pilot_hashes_before == pilot_hashes_after,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "process_priority_class": _priority_class(),
                "active_power_plan": _power_plan(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ["joblib", "numpy", "optuna", "pandas", "scikit-learn", "threadpoolctl"]
                },
                "threadpools_before_limits": _sanitized_threadpools(),
                "affinity_policy": "not changed",
            },
            "config_summary": json.loads(config_frame.to_json(orient="records")),
        }
        write_json(manifest, output / "manifest.json")
        return manifest
    except Exception as exc:
        failure = {
            "protocol_version": config.get("protocol_version"),
            "status": "FAIL",
            "full_run_blocked": True,
            "error": f"{type(exc).__name__}: {exc}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(failure, output / "calibration_failure.json")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the train-only latency calibration gate.")
    parser.add_argument("--config", default="configs/latency_calibration_v1.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        manifest = run_calibration(load_calibration_config(args.config))
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Calibration aborted: {exc}")
        return 2
    print(json.dumps({
        "output": str(resolve_project_path(load_calibration_config(args.config)["output_dir"])),
        "gate": manifest["stability_gate"],
        "full_run_blocked": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
