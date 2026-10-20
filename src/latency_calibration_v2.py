"""Independent-round, train-only latency calibration v2."""

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
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .benchmark import load_config
from .data import load_bank_data, prepare_features, sha256_file, write_json
from .latency_calibration import (
    CalibrationConfiguration,
    _directory_hashes,
    _power_plan,
    _priority_class,
    _sanitized_threadpools,
    _system_cpu_percent,
    _windows_cpu_snapshot,
    load_calibration_config,
    load_distinct_pilot_configurations,
    relative_range,
    select_benchmark_positions,
    sha256_array,
    summarize_repeats,
    validate_versioned_output_path,
)
from .optimize import build_fixed_folds, ensure_fold_artifact, load_hpo_config
from .pareto import constraint_status, pareto_front
from .paths import resolve_project_path
from .pipeline import build_baseline_pipeline, build_hpo_pipeline


class _ProcessorPowerInformation(ctypes.Structure):
    _fields_ = [
        ("number", ctypes.c_ulong),
        ("max_mhz", ctypes.c_ulong),
        ("current_mhz", ctypes.c_ulong),
        ("mhz_limit", ctypes.c_ulong),
        ("max_idle_state", ctypes.c_ulong),
        ("current_idle_state", ctypes.c_ulong),
    ]


class _ThreadEntry32(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("cntUsage", ctypes.c_ulong),
        ("th32ThreadID", ctypes.c_ulong),
        ("th32OwnerProcessID", ctypes.c_ulong),
        ("tpBasePri", ctypes.c_long),
        ("tpDeltaPri", ctypes.c_long),
        ("dwFlags", ctypes.c_ulong),
    ]


def processor_power_snapshot() -> dict[str, Any] | None:
    """Read Windows processor MHz without changing the power policy."""
    if os.name != "nt":
        return None
    count = os.cpu_count() or 1
    values = (_ProcessorPowerInformation * count)()
    function = ctypes.WinDLL("PowrProf").CallNtPowerInformation
    function.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong
    ]
    function.restype = ctypes.c_ulong
    status = int(function(11, None, 0, ctypes.byref(values), ctypes.sizeof(values)))
    if status != 0:
        return {"status": status, "available": False}
    current = [int(item.current_mhz) for item in values]
    maximum = [int(item.max_mhz) for item in values]
    limits = [int(item.mhz_limit) for item in values]
    return {
        "status": status,
        "available": True,
        "logical_processors": count,
        "current_mhz_min": min(current),
        "current_mhz_median": float(statistics.median(current)),
        "current_mhz_max": max(current),
        "max_mhz_min": min(maximum),
        "max_mhz_max": max(maximum),
        "mhz_limit_min": min(limits),
        "mhz_limit_max": max(limits),
        "current_mhz_by_processor": current,
    }


def process_thread_count() -> int | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Thread32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ThreadEntry32)]
    kernel32.Thread32First.restype = ctypes.c_int
    kernel32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ThreadEntry32)]
    kernel32.Thread32Next.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
    if snapshot in (None, ctypes.c_void_p(-1).value):
        return None
    entry = _ThreadEntry32()
    entry.dwSize = ctypes.sizeof(_ThreadEntry32)
    owner = os.getpid()
    count = 0
    try:
        ok = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if int(entry.th32OwnerProcessID) == owner:
                count += 1
            ok = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return count


def _power_setting(alias: str) -> dict[str, Any]:
    result = subprocess.run(
        ["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", alias],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "alias": alias,
        "returncode": result.returncode,
        "output": result.stdout.strip() or result.stderr.strip(),
    }


def _background_process_snapshot() -> dict[str, Any]:
    if os.name != "nt":
        return {"available": False}
    command = (
        "Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 "
        "ProcessName,Id,CPU,@{n='ThreadCount';e={$_.Threads.Count}},WorkingSet64 | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    payload: Any = None
    if result.returncode == 0 and result.stdout.strip():
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = result.stdout.strip()
    return {
        "available": result.returncode == 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "top_by_cumulative_cpu": payload,
        "error": result.stderr.strip() or None,
    }


def prediction_digest(estimator: Any, X_batch: pd.DataFrame, thread_limit: int) -> str:
    with threadpool_limits(limits=thread_limit):
        probability = np.ascontiguousarray(estimator.predict_proba(X_batch), dtype=np.float64)
    return hashlib.sha256(probability.tobytes()).hexdigest()


def measure_session_v2(
    estimator: Any,
    X_batch: pd.DataFrame,
    *,
    warmups: int,
    repeats: int,
    thread_limit: int,
    disable_cyclic_gc: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if estimator.named_steps["model"].n_jobs != 1:
        raise ValueError("Calibration v2 requires RandomForestClassifier(n_jobs=1).")
    gc.collect()
    gc_enabled_before = gc.isenabled()
    system_before = _windows_cpu_snapshot()
    power_before = processor_power_snapshot()
    threads_before = process_thread_count()
    process_before = time.process_time_ns()
    block_before = time.perf_counter_ns()
    raw: list[int] = []
    effective_threadpools: list[dict[str, Any]] = []
    with threadpool_limits(limits=thread_limit):
        effective_threadpools = _sanitized_threadpools()
        if disable_cyclic_gc and gc_enabled_before:
            gc.disable()
        try:
            for _ in range(warmups):
                estimator.predict_proba(X_batch)
            for _ in range(repeats):
                started = time.perf_counter_ns()
                estimator.predict_proba(X_batch)
                raw.append(time.perf_counter_ns() - started)
        finally:
            if gc_enabled_before and not gc.isenabled():
                gc.enable()
    block_after = time.perf_counter_ns()
    process_after = time.process_time_ns()
    system_after = _windows_cpu_snapshot()
    power_after = processor_power_snapshot()
    threads_after = process_thread_count()
    wall_ms = (block_after - block_before) / 1_000_000.0
    process_ms = (process_after - process_before) / 1_000_000.0
    environment = {
        "system_cpu_percent_during_block": _system_cpu_percent(system_before, system_after),
        "process_cpu_time_ms": process_ms,
        "wall_block_ms": wall_ms,
        "process_cpu_efficiency_percent": float(100.0 * process_ms / wall_ms) if wall_ms else None,
        "processor_power_before": power_before,
        "processor_power_after": power_after,
        "process_threads_before": threads_before,
        "process_threads_after": threads_after,
        "process_priority_class": _priority_class(),
        "gc_enabled_before": gc_enabled_before,
        "gc_disabled_during_block": disable_cyclic_gc,
        "gc_enabled_after": gc.isenabled(),
        "effective_threadpools": effective_threadpools,
    }
    return summarize_repeats(raw, len(X_batch)), environment


def _resolved_params(item: CalibrationConfiguration, base_config: dict[str, Any]) -> dict[str, Any]:
    if item.params is not None:
        return item.params
    baseline = base_config["baseline"]
    return {
        "n_estimators": int(baseline["n_estimators"]),
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
        "random_state": int(baseline["random_state"]),
        "n_jobs": int(baseline["n_jobs"]),
    }


def _build_estimator(
    item: CalibrationConfiguration,
    X_fit: pd.DataFrame,
    base_config: dict[str, Any],
) -> Any:
    if item.source == "frozen_week2_baseline_config":
        return build_baseline_pipeline(X_fit, base_config)
    if item.params is None:
        raise ValueError(f"Missing HPO parameters for {item.config_id}")
    return build_hpo_pipeline(X_fit, base_config, item.params)


def _measurement_rows(
    *,
    phase: str,
    item: CalibrationConfiguration,
    fold: int,
    round_number: int,
    session_in_round: int,
    session_global: int,
    order_position: int,
    order_seed: int,
    batch_hash: str,
    X_batch: pd.DataFrame,
    estimator: Any,
    timing: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timestamp_start = datetime.now(timezone.utc).isoformat()
    failed = False
    error: str | None = None
    try:
        summary, environment = measure_session_v2(
            estimator,
            X_batch,
            warmups=int(timing["warmup_calls_per_session"]),
            repeats=int(timing["timed_repeats_per_session"]),
            thread_limit=int(timing["thread_limit"]),
            disable_cyclic_gc=bool(timing["cyclic_gc_disabled_during_measurement_block"]),
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
            "process_cpu_efficiency_percent": None,
            "processor_power_before": None,
            "processor_power_after": None,
            "process_threads_before": None,
            "process_threads_after": None,
            "process_priority_class": None,
            "gc_enabled_before": None,
            "gc_disabled_during_block": None,
            "gc_enabled_after": None,
            "effective_threadpools": [],
        }
    timestamp_end = datetime.now(timezone.utc).isoformat()
    gate_cfg = timing["gate"]
    system_cpu = environment["system_cpu_percent_during_block"]
    efficiency = environment["process_cpu_efficiency_percent"]
    contention = bool(
        (system_cpu is not None and system_cpu > float(gate_cfg["maximum_system_cpu_percent"]))
        or (
            efficiency is not None
            and efficiency < float(gate_cfg["minimum_process_cpu_efficiency_percent"])
        )
    )
    power_before = environment["processor_power_before"] or {}
    power_after = environment["processor_power_after"] or {}
    raw_rows = [
        {
            "phase": phase,
            "config_id": item.config_id,
            "fold": fold,
            "round": round_number,
            "session_in_round": session_in_round,
            "session_global": session_global,
            "order_position": order_position,
            "order_seed": order_seed,
            "repeat": repeat_number,
            "duration_ns": duration_ns,
            "duration_ms_per_1000": duration_ms,
            "batch_size": len(X_batch),
            "batch_original_indices_sha256": batch_hash,
            "timestamp_start_utc": timestamp_start,
            "timestamp_end_utc": timestamp_end,
        }
        for repeat_number, (duration_ns, duration_ms) in enumerate(
            zip(summary["raw_duration_ns"], summary["raw_ms_per_1000"]), start=1
        )
    ]
    session_row = {
        "phase": phase,
        "config_id": item.config_id,
        "label": item.label,
        "fold": fold,
        "round": round_number,
        "session_in_round": session_in_round,
        "session_global": session_global,
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
        "processor_current_mhz_median_before": power_before.get("current_mhz_median"),
        "processor_current_mhz_median_after": power_after.get("current_mhz_median"),
        "processor_current_mhz_min_before": power_before.get("current_mhz_min"),
        "processor_current_mhz_max_before": power_before.get("current_mhz_max"),
        "processor_power_before_json": json.dumps(power_before, separators=(",", ":")),
        "processor_power_after_json": json.dumps(power_after, separators=(",", ":")),
        "process_threads_before": environment["process_threads_before"],
        "process_threads_after": environment["process_threads_after"],
        "process_priority_class": environment["process_priority_class"],
        "gc_enabled_before": environment["gc_enabled_before"],
        "gc_disabled_during_block": environment["gc_disabled_during_block"],
        "gc_enabled_after": environment["gc_enabled_after"],
        "threadpools_json": json.dumps(environment["effective_threadpools"], separators=(",", ":")),
        "contention_detected": contention,
        "measurement_failed": failed,
        "error": error,
        "batch_size": len(X_batch),
        "batch_original_indices_sha256": batch_hash,
        "model_size_bytes": 0,
    }
    return raw_rows, session_row


def aggregate_v2(session_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    for (config_id, label, fold), group in session_frame.groupby(["config_id", "label", "fold"]):
        ordered = group.sort_values(["round", "session_in_round"])
        medians = ordered["session_median_ms_per_1000"].astype(float).tolist()
        round_ranges: dict[str, float] = {}
        round_medians: dict[str, float] = {}
        for round_number, round_group in ordered.groupby("round"):
            values = round_group["session_median_ms_per_1000"].astype(float).tolist()
            round_ranges[str(int(round_number))] = relative_range(values)
            round_medians[str(int(round_number))] = float(statistics.median(values))
        q1, q3 = np.percentile(medians, [25, 75])
        fold_rows.append({
            "config_id": config_id,
            "label": label,
            "fold": int(fold),
            "session_medians_ms_per_1000_json": json.dumps(medians, separators=(",", ":")),
            "round_medians_ms_per_1000_json": json.dumps(round_medians, separators=(",", ":")),
            "round_relative_ranges_json": json.dumps(round_ranges, separators=(",", ":")),
            "fold_median_of_all_session_medians_ms_per_1000": float(statistics.median(medians)),
            "all_session_median_relative_range": relative_range(medians),
            "maximum_round_relative_range": max(round_ranges.values()),
            "session_median_iqr_ms_per_1000": float(q3 - q1),
            "session_median_std_ms_per_1000": float(statistics.stdev(medians)),
            "model_size_max_bytes": int(group["model_size_bytes"].max()),
            "model_size_max_mb": int(group["model_size_bytes"].max()) / 1_000_000,
            "measurement_failed": bool(group["measurement_failed"].any()),
            "contention_detected": bool(group["contention_detected"].any()),
            "batch_original_indices_sha256": str(group["batch_original_indices_sha256"].iloc[0]),
        })
    fold_frame = pd.DataFrame(fold_rows)
    config_rows: list[dict[str, Any]] = []
    for (config_id, label), group in fold_frame.groupby(["config_id", "label"]):
        fold_medians = group["fold_median_of_all_session_medians_ms_per_1000"].astype(float).tolist()
        q1, q3 = np.percentile(fold_medians, [25, 75])
        config_rows.append({
            "config_id": config_id,
            "label": label,
            "fold_medians_ms_per_1000_json": json.dumps(fold_medians, separators=(",", ":")),
            "aggregate_max_fold_median_ms_per_1000": float(max(fold_medians)),
            "median_across_fold_medians_ms_per_1000": float(statistics.median(fold_medians)),
            "fold_median_relative_range": relative_range(fold_medians),
            "fold_median_iqr_ms_per_1000": float(q3 - q1),
            "maximum_all_session_relative_range": float(group["all_session_median_relative_range"].max()),
            "maximum_round_relative_range": float(group["maximum_round_relative_range"].max()),
            "model_size_max_bytes": int(group["model_size_max_bytes"].max()),
            "model_size_max_mb": float(group["model_size_max_mb"].max()),
            "measurement_failed": bool(group["measurement_failed"].any()),
            "contention_detected": bool(group["contention_detected"].any()),
            "batch_original_indices_sha256": str(group["batch_original_indices_sha256"].iloc[0]),
        })
    return fold_frame.sort_values(["label", "fold"]), pd.DataFrame(config_rows).sort_values("label")


def evaluate_gate_v2(
    session_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    *,
    expected_batch_hash: str,
    expected_config_count: int,
    expected_folds: int,
    expected_rounds: int,
    sessions_per_round: int,
    maximum_relative_range: float,
    prediction_digest_consistent: bool,
) -> dict[str, Any]:
    expected_rows = expected_config_count * expected_folds * expected_rounds * sessions_per_round
    failures: list[str] = []
    hashes = sorted(session_frame["batch_original_indices_sha256"].dropna().unique())
    overall_unstable = fold_frame.loc[
        fold_frame["all_session_median_relative_range"] > maximum_relative_range,
        ["config_id", "fold", "all_session_median_relative_range"],
    ]
    round_unstable = fold_frame.loc[
        fold_frame["maximum_round_relative_range"] > maximum_relative_range,
        ["config_id", "fold", "maximum_round_relative_range", "round_relative_ranges_json"],
    ]
    if len(session_frame) != expected_rows:
        failures.append(f"expected {expected_rows} session rows, observed {len(session_frame)}")
    if hashes != [expected_batch_hash]:
        failures.append("not every session used the identical benchmark batch")
    if session_frame["measurement_failed"].any():
        failures.append("one or more measurements failed")
    if session_frame["contention_detected"].any():
        failures.append("one or more sessions detected major CPU contention")
    if not overall_unstable.empty:
        failures.append("one or more config/folds exceed 15% across all session medians")
    if not round_unstable.empty:
        failures.append("one or more config/folds exceed 15% within an independent round")
    if not prediction_digest_consistent:
        failures.append("prediction digest changed across sessions/refit rounds")
    return {
        "status": "PASS" if not failures else "FAIL",
        "full_run_blocked": True,
        "criterion": (
            "Every config/fold relative range across all sessions and within every independent "
            f"round <= {maximum_relative_range:.2%}; shared batch; stable prediction digests; "
            "no measurement failures or major contention."
        ),
        "criterion_is_engineering_gate_not_confidence_guarantee": True,
        "expected_session_rows": expected_rows,
        "observed_session_rows": len(session_frame),
        "observed_batch_hashes": hashes,
        "maximum_observed_all_session_relative_range": float(
            fold_frame["all_session_median_relative_range"].max()
        ),
        "maximum_observed_round_relative_range": float(
            fold_frame["maximum_round_relative_range"].max()
        ),
        "overall_unstable_config_folds": json.loads(overall_unstable.to_json(orient="records")),
        "round_unstable_config_folds": json.loads(round_unstable.to_json(orient="records")),
        "prediction_digest_consistent": prediction_digest_consistent,
        "failure_reasons": failures,
        "user_acceptance_required_even_if_passes": True,
    }


def _analyze_v1(v1_dir: Path) -> dict[str, Any]:
    artifacts = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(v1_dir.iterdir()) if path.is_file()
    }
    for path in v1_dir.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    raw = pd.read_csv(v1_dir / "raw_measurements.csv")
    sessions = pd.read_csv(v1_dir / "session_summary.csv")
    folds = pd.read_csv(v1_dir / "fold_summary.csv")
    configs = pd.read_csv(v1_dir / "config_summary.csv")
    baseline_id = str(
        configs.loc[configs["label"].eq("week2_baseline_rf"), "config_id"].iloc[0]
    )
    baseline = sessions.loc[sessions["config_id"].eq(baseline_id)].copy()
    baseline_f3 = baseline.loc[baseline["fold"].eq(3)].sort_values("session")
    return {
        "all_artifacts_read_and_hashed": artifacts,
        "csv_rows": {
            "raw_measurements": len(raw),
            "session_summary": len(sessions),
            "fold_summary": len(folds),
            "config_summary": len(configs),
        },
        "baseline_fold3_session_medians_ms_per_1000": baseline_f3[
            "session_median_ms_per_1000"
        ].astype(float).tolist(),
        "baseline_fold3_order_positions": baseline_f3["order_position"].astype(int).tolist(),
        "baseline_order_latency_spearman": float(
            baseline["order_position"].corr(baseline["session_median_ms_per_1000"], method="spearman")
        ),
        "baseline_system_cpu_latency_spearman": float(
            baseline["system_cpu_percent"].corr(
                baseline["session_median_ms_per_1000"], method="spearman"
            )
        ),
        "baseline_process_efficiency_latency_spearman": float(
            baseline["process_cpu_efficiency_percent"].corr(
                baseline["session_median_ms_per_1000"], method="spearman"
            )
        ),
        "fixed_fold_order_confounding_present": True,
        "finding": (
            "V1 fold 3 was always measured after folds 1 and 2. Baseline F3 drifted downward "
            "across sessions without a strong observed CPU/order association; cache, Windows "
            "scheduling, power/thermal state and elapsed-time effects remain plausible, not proven."
        ),
    }


def _constraint_and_pareto_audit(
    pilot_path: Path,
    hpo_config: dict[str, Any],
    latency_cap: float,
    size_cap: float,
) -> dict[str, Any]:
    frame = pd.read_csv(pilot_path)
    completed = frame.loc[frame["state"].eq("COMPLETE")].copy()
    mismatches: list[dict[str, Any]] = []
    for _, row in completed.iterrows():
        status = constraint_status(
            float(row["model_size_max_mb"]),
            float(row["latency_per_1000_samples_ms"]),
            model_size_cap_mb=size_cap,
            latency_cap_ms=latency_cap,
        )
        observed = str(row["feasible"]).lower() == "true"
        if observed != bool(status["feasible"]):
            mismatches.append({"sampler": row["sampler"], "trial_number": int(row["trial_number"])})
    front = pareto_front(frame)
    invalid_front_rows = front.loc[
        ~front["state"].eq("COMPLETE") | ~front["feasible"].astype(bool)
    ]
    return {
        "objective_directions": list(hpo_config["directions"]),
        "constraint_convention": "g_i <= 0 is feasible",
        "completed_trials_checked": len(completed),
        "feasibility_mismatches": mismatches,
        "pareto_rows": len(front),
        "invalid_pareto_rows": len(invalid_front_rows),
        "pass": not mismatches and invalid_front_rows.empty,
    }


def _serialize_size(estimator: Any, path: Path, compression: int) -> int:
    try:
        joblib.dump(estimator, path, compress=compression)
        return int(path.stat().st_size)
    finally:
        path.unlink(missing_ok=True)


def run_calibration_v2(config: dict[str, Any]) -> dict[str, Any]:
    output = resolve_project_path(config["output_dir"])
    validate_versioned_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(config, output / "resolved_calibration_config.json")
    started_wall = time.perf_counter()
    baseline_dir = resolve_project_path(config["baseline_reference_dir"])
    pilot_dir = resolve_project_path(config["pilot_reference_dir"])
    v1_dir = resolve_project_path(config["v1_reference_dir"])
    hashes_before = {
        "baseline": _directory_hashes(baseline_dir),
        "pilot": _directory_hashes(pilot_dir),
        "v1": _directory_hashes(v1_dir),
    }
    try:
        v1_analysis = _analyze_v1(v1_dir)
        data_path = resolve_project_path(config["data_path"])
        audit = json.loads(resolve_project_path(config["data_audit_reference"]).read_text(encoding="utf-8"))
        if sha256_file(data_path) != audit["sha256"]:
            raise ValueError("Dataset differs from the frozen data-audit fingerprint.")
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
        with np.load(resolve_project_path(config["v1_batch_reference"])) as v1_batch:
            if not np.array_equal(batch_positions, v1_batch["train_positions"]):
                raise ValueError("V2 train positions differ from the immutable v1 benchmark batch.")
            if not np.array_equal(batch_original, v1_batch["original_indices"]):
                raise ValueError("V2 original indices differ from the immutable v1 benchmark batch.")
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
            "original_indices_sha256": batch_hash,
            "train_positions_sha256": sha256_array(batch_positions),
            "holdout_overlap_count": int(np.intersect1d(batch_original, test_indices).size),
            "exact_match_to_v1_batch": True,
        }, output / "benchmark_batch.json")

        train_frame = frame.iloc[train_indices].copy()
        del frame
        X_train, y_train = prepare_features(train_frame)
        del train_frame
        X_batch = X_train.iloc[batch_positions].copy()
        if "duration" in X_batch.columns:
            raise RuntimeError("Leakage guard failed: duration is present in benchmark features.")
        base_config = load_config(config["base_config"])
        hpo_config = load_hpo_config(config["hpo_config"])
        folds = build_fixed_folds(X_train, y_train, base_config)
        fold_path = resolve_project_path(config["cv_folds_reference"])
        if not fold_path.is_file():
            raise FileNotFoundError(f"Frozen CV folds are missing: {fold_path}")
        fold_reference = ensure_fold_artifact(fold_path, folds, train_indices)
        configurations, historical_pairs = load_distinct_pilot_configurations(
            resolve_project_path(config["pilot_history_reference"])
        )
        labels = {item.config_id: item.label for item in configurations}
        configuration_payload = {
            item.config_id: {
                "label": item.label,
                "source": item.source,
                "parameters": _resolved_params(item, base_config),
            }
            for item in configurations
        }
        write_json({
            "configurations": configuration_payload,
            "historical_paired_latency_audit": historical_pairs,
        }, output / "configurations.json")

        shared_timing = dict(config["timing"])
        shared_timing["gate"] = config["stability_gate"]
        diagnostic_cfg = config["diagnostic"]
        diagnostic_timing = dict(shared_timing)
        diagnostic_timing.update({
            "warmup_calls_per_session": diagnostic_cfg["warmup_calls_per_session"],
            "timed_repeats_per_session": diagnostic_cfg["timed_repeats_per_session"],
        })
        baseline_item = next(
            item for item in configurations if item.source == "frozen_week2_baseline_config"
        )
        compression = int(base_config["artifact"]["compression"])
        thread_limit = int(shared_timing["thread_limit"])
        if int(base_config["baseline"]["n_jobs"]) != int(shared_timing["estimator_n_jobs"]):
            raise ValueError("Estimator n_jobs differs from the frozen baseline protocol.")

        environment_start = {
            "background_processes": _background_process_snapshot(),
            "processor_power": processor_power_snapshot(),
            "process_priority_class": _priority_class(),
            "process_thread_count": process_thread_count(),
            "active_power_plan": _power_plan(),
            "processor_minimum_state": _power_setting("PROCTHROTTLEMIN"),
            "processor_maximum_state": _power_setting("PROCTHROTTLEMAX"),
        }

        diagnostic_raw: list[dict[str, Any]] = []
        diagnostic_sessions: list[dict[str, Any]] = []
        diagnostic_digests: list[dict[str, Any]] = []
        diagnostic_orders: list[dict[str, Any]] = []
        diagnostic_rounds = int(diagnostic_cfg["independent_refit_rounds"])
        diagnostic_sessions_per_round = int(diagnostic_cfg["sessions_per_round"])
        for round_number in range(1, diagnostic_rounds + 1):
            models: dict[int, Any] = {}
            fit_order = list(range(1, len(folds) + 1))
            random.Random(int(diagnostic_cfg["interleave_seed"]) + round_number).shuffle(fit_order)
            for fold_number in fit_order:
                fit_positions, _ = folds[fold_number - 1]
                estimator = _build_estimator(
                    baseline_item, X_train.iloc[fit_positions], base_config
                )
                with threadpool_limits(limits=thread_limit):
                    estimator.fit(X_train.iloc[fit_positions], y_train.iloc[fit_positions])
                models[fold_number] = estimator
                diagnostic_digests.append({
                    "fold": fold_number,
                    "round": round_number,
                    "stage": "before_timing",
                    "prediction_sha256": prediction_digest(estimator, X_batch, thread_limit),
                })
            for session_in_round in range(1, diagnostic_sessions_per_round + 1):
                session_global = (round_number - 1) * diagnostic_sessions_per_round + session_in_round
                order_seed = (
                    int(diagnostic_cfg["interleave_seed"])
                    + round_number * 1000
                    + session_in_round
                )
                order = list(models)
                random.Random(order_seed).shuffle(order)
                diagnostic_orders.append({
                    "round": round_number,
                    "session_in_round": session_in_round,
                    "session_global": session_global,
                    "order_seed": order_seed,
                    "fold_order": order,
                })
                for order_position, fold_number in enumerate(order, start=1):
                    raw_rows, session_row = _measurement_rows(
                        phase="diagnostic",
                        item=baseline_item,
                        fold=fold_number,
                        round_number=round_number,
                        session_in_round=session_in_round,
                        session_global=session_global,
                        order_position=order_position,
                        order_seed=order_seed,
                        batch_hash=batch_hash,
                        X_batch=X_batch,
                        estimator=models[fold_number],
                        timing=diagnostic_timing,
                    )
                    diagnostic_raw.extend(raw_rows)
                    diagnostic_sessions.append(session_row)
                time.sleep(float(shared_timing["session_pause_seconds"]))
            for fold_number, estimator in models.items():
                diagnostic_digests.append({
                    "fold": fold_number,
                    "round": round_number,
                    "stage": "after_timing",
                    "prediction_sha256": prediction_digest(estimator, X_batch, thread_limit),
                })
            del models
            gc.collect()

        diagnostic_session_frame = pd.DataFrame(diagnostic_sessions)
        diagnostic_raw_frame = pd.DataFrame(diagnostic_raw)
        diagnostic_session_frame.to_csv(
            output / "diagnostic_session_summary.csv", index=False, encoding="utf-8"
        )
        diagnostic_raw_frame.to_csv(
            output / "diagnostic_raw_measurements.csv", index=False, encoding="utf-8"
        )
        time.sleep(float(shared_timing["diagnostic_to_official_settle_seconds"]))

        raw_rows: list[dict[str, Any]] = []
        session_rows: list[dict[str, Any]] = []
        fit_metadata: list[dict[str, Any]] = []
        prediction_digests: list[dict[str, Any]] = []
        run_orders: list[dict[str, Any]] = []
        model_sizes: dict[tuple[str, int, int], int] = {}
        model_dir = output / ".models"
        model_dir.mkdir()
        rounds = int(shared_timing["independent_refit_rounds"])
        sessions_per_round = int(shared_timing["sessions_per_round"])
        model_keys = [
            (item.config_id, fold_number)
            for item in configurations
            for fold_number in range(1, len(folds) + 1)
        ]
        item_by_id = {item.config_id: item for item in configurations}
        for round_number in range(1, rounds + 1):
            models: dict[tuple[str, int], Any] = {}
            fit_order = list(model_keys)
            random.Random(int(shared_timing["interleave_seed"]) + round_number).shuffle(fit_order)
            for config_id, fold_number in fit_order:
                item = item_by_id[config_id]
                fit_positions, _ = folds[fold_number - 1]
                estimator = _build_estimator(item, X_train.iloc[fit_positions], base_config)
                fit_started = time.perf_counter_ns()
                with threadpool_limits(limits=thread_limit):
                    estimator.fit(X_train.iloc[fit_positions], y_train.iloc[fit_positions])
                fit_ns = time.perf_counter_ns() - fit_started
                models[(config_id, fold_number)] = estimator
                digest = prediction_digest(estimator, X_batch, thread_limit)
                prediction_digests.append({
                    "config_id": config_id,
                    "fold": fold_number,
                    "round": round_number,
                    "stage": "before_timing",
                    "prediction_sha256": digest,
                })
                fit_metadata.append({
                    "config_id": config_id,
                    "fold": fold_number,
                    "round": round_number,
                    "fit_duration_ns": int(fit_ns),
                    "fit_seconds": fit_ns / 1_000_000_000.0,
                    "fit_samples": int(len(fit_positions)),
                    "random_state": int(base_config["baseline"]["random_state"]),
                    "estimator_n_jobs": int(estimator.named_steps["model"].n_jobs),
                    "prediction_sha256_before_timing": digest,
                })
            for session_in_round in range(1, sessions_per_round + 1):
                session_global = (round_number - 1) * sessions_per_round + session_in_round
                order_seed = (
                    int(shared_timing["interleave_seed"])
                    + round_number * 1000
                    + session_in_round
                )
                order = list(model_keys)
                random.Random(order_seed).shuffle(order)
                run_orders.append({
                    "round": round_number,
                    "session_in_round": session_in_round,
                    "session_global": session_global,
                    "order_seed": order_seed,
                    "configuration_fold_order": [
                        {"config_id": config_id, "fold": fold_number}
                        for config_id, fold_number in order
                    ],
                })
                for order_position, (config_id, fold_number) in enumerate(order, start=1):
                    raw, session = _measurement_rows(
                        phase="official_v2_calibration",
                        item=item_by_id[config_id],
                        fold=fold_number,
                        round_number=round_number,
                        session_in_round=session_in_round,
                        session_global=session_global,
                        order_position=order_position,
                        order_seed=order_seed,
                        batch_hash=batch_hash,
                        X_batch=X_batch,
                        estimator=models[(config_id, fold_number)],
                        timing=shared_timing,
                    )
                    raw_rows.extend(raw)
                    session_rows.append(session)
                time.sleep(float(shared_timing["session_pause_seconds"]))
            for (config_id, fold_number), estimator in models.items():
                after_digest = prediction_digest(estimator, X_batch, thread_limit)
                prediction_digests.append({
                    "config_id": config_id,
                    "fold": fold_number,
                    "round": round_number,
                    "stage": "after_timing",
                    "prediction_sha256": after_digest,
                })
                model_sizes[(config_id, fold_number, round_number)] = _serialize_size(
                    estimator,
                    model_dir / f"{config_id}_fold_{fold_number}_round_{round_number}.joblib",
                    compression,
                )
            del models
            gc.collect()
        model_dir.rmdir()

        session_frame = pd.DataFrame(session_rows)
        for index, row in session_frame.iterrows():
            session_frame.at[index, "model_size_bytes"] = model_sizes[
                (str(row["config_id"]), int(row["fold"]), int(row["round"]))
            ]
        fold_frame, config_frame = aggregate_v2(session_frame)
        config_frame.insert(
            2,
            "parameters_json",
            config_frame["config_id"].map({
                item.config_id: json.dumps(
                    _resolved_params(item, base_config), sort_keys=True, separators=(",", ":")
                )
                for item in configurations
            }),
        )
        digest_frame = pd.DataFrame(prediction_digests)
        digest_consistent_by_model = (
            digest_frame.groupby(["config_id", "fold"])["prediction_sha256"].nunique() == 1
        )
        prediction_digest_consistent = bool(digest_consistent_by_model.all())
        gate = evaluate_gate_v2(
            session_frame,
            fold_frame,
            expected_batch_hash=batch_hash,
            expected_config_count=len(configurations),
            expected_folds=len(folds),
            expected_rounds=rounds,
            sessions_per_round=sessions_per_round,
            maximum_relative_range=float(
                config["stability_gate"]["maximum_session_median_relative_range"]
            ),
            prediction_digest_consistent=prediction_digest_consistent,
        )
        latency_cap = float(config["constraints"]["latency_cap_ms_per_1000"])
        size_cap = float(config["constraints"]["model_size_cap_mb"])
        constraint_rows: list[dict[str, Any]] = []
        for index, row in config_frame.iterrows():
            status = constraint_status(
                float(row["model_size_max_mb"]),
                float(row["aggregate_max_fold_median_ms_per_1000"]),
                model_size_cap_mb=size_cap,
                latency_cap_ms=latency_cap,
            )
            config_frame.at[index, "latency_cap_ms_per_1000"] = latency_cap
            config_frame.at[index, "model_size_cap_mb"] = size_cap
            config_frame.at[index, "latency_violation_ms"] = status["latency_violation_ms"]
            config_frame.at[index, "size_violation_mb"] = status["size_violation_mb"]
            config_frame.at[index, "constraint_feasible"] = bool(status["feasible"])
            constraint_rows.append({"config_id": row["config_id"], **status})
        pareto_audit = _constraint_and_pareto_audit(
            resolve_project_path(config["pilot_history_reference"]),
            hpo_config,
            latency_cap,
            size_cap,
        )

        pd.DataFrame(raw_rows).to_csv(output / "raw_measurements.csv", index=False, encoding="utf-8")
        session_frame.to_csv(output / "session_summary.csv", index=False, encoding="utf-8")
        fold_frame.to_csv(output / "fold_summary.csv", index=False, encoding="utf-8")
        config_frame.to_csv(output / "config_summary.csv", index=False, encoding="utf-8")
        write_json({"fit_metadata": fit_metadata}, output / "fit_metadata.json")
        write_json({"run_orders": run_orders, "diagnostic_orders": diagnostic_orders}, output / "run_order.json")
        write_json({"prediction_digests": prediction_digests}, output / "prediction_digests.json")
        write_json(gate, output / "gate_decision.json")

        diagnostic_fold_rows: list[dict[str, Any]] = []
        for fold_number, group in diagnostic_session_frame.groupby("fold"):
            values = group.sort_values(["round", "session_in_round"])[
                "session_median_ms_per_1000"
            ].astype(float).tolist()
            round_ranges = {
                str(int(round_number)): relative_range(
                    round_group["session_median_ms_per_1000"].astype(float).tolist()
                )
                for round_number, round_group in group.groupby("round")
            }
            diagnostic_fold_rows.append({
                "fold": int(fold_number),
                "session_medians_ms_per_1000": values,
                "relative_range_all_sessions": relative_range(values),
                "relative_ranges_by_round": round_ranges,
                "median_ms_per_1000": float(statistics.median(values)),
            })
        diagnostic_digest_frame = pd.DataFrame(diagnostic_digests)
        diagnostic_digest_consistent = bool(
            (
                diagnostic_digest_frame.groupby("fold")["prediction_sha256"].nunique() == 1
            ).all()
        )
        environment_end = {
            "background_processes": _background_process_snapshot(),
            "processor_power": processor_power_snapshot(),
            "process_priority_class": _priority_class(),
            "process_thread_count": process_thread_count(),
            "active_power_plan": _power_plan(),
        }
        stability_diagnostics = {
            "v1_analysis": v1_analysis,
            "v2_diagnostic_protocol": diagnostic_cfg,
            "v2_diagnostic_fold_results": diagnostic_fold_rows,
            "v2_diagnostic_prediction_digest_consistent": diagnostic_digest_consistent,
            "official_prediction_digest_consistent": prediction_digest_consistent,
            "official_prediction_digest_unique_counts": {
                f"{config_id}_fold_{fold}": int(value)
                for (config_id, fold), value in digest_frame.groupby(
                    ["config_id", "fold"]
                )["prediction_sha256"].nunique().items()
            },
            "frequency_observation": {
                "available_session_rows": int(
                    session_frame["processor_current_mhz_median_before"].notna().sum()
                ),
                "minimum_observed_current_mhz": float(
                    session_frame["processor_current_mhz_min_before"].min()
                ),
                "maximum_observed_current_mhz": float(
                    session_frame["processor_current_mhz_max_before"].max()
                ),
                "median_of_pre_block_medians_mhz": float(
                    session_frame["processor_current_mhz_median_before"].median()
                ),
            },
            "system_cpu_percent_range": [
                float(session_frame["system_cpu_percent"].min()),
                float(session_frame["system_cpu_percent"].max()),
            ],
            "process_cpu_efficiency_percent_range": [
                float(session_frame["process_cpu_efficiency_percent"].min()),
                float(session_frame["process_cpu_efficiency_percent"].max()),
            ],
            "process_thread_count_range": [
                int(session_frame["process_threads_before"].min()),
                int(session_frame["process_threads_before"].max()),
            ],
            "process_priority_classes": sorted(
                int(value) for value in session_frame["process_priority_class"].dropna().unique()
            ),
            "contention_flags": int(session_frame["contention_detected"].sum()),
            "measurement_failures": int(session_frame["measurement_failed"].sum()),
            "environment_start": environment_start,
            "environment_end": environment_end,
            "constraint_and_pareto_audit": pareto_audit,
            "causal_conclusion": (
                "The measurements can test whether v1 instability reproduces after removing fixed "
                "fold order and adding independent refits. They cannot uniquely attribute residual "
                "variation to frequency, scheduler, cache, thermal state or background processes."
            ),
        }
        write_json(stability_diagnostics, output / "stability_diagnostics.json")

        hashes_after = {
            "baseline": _directory_hashes(baseline_dir),
            "pilot": _directory_hashes(pilot_dir),
            "v1": _directory_hashes(v1_dir),
        }
        all_immutable = hashes_before == hashes_after
        if not all_immutable:
            raise RuntimeError("An immutable baseline, pilot or v1 artifact changed during v2.")
        wall_seconds = float(time.perf_counter() - started_wall)
        acceptance = {
            "protocol_version": config["protocol_version"],
            "status": gate["status"],
            "full_run_blocked": True,
            "wall_time_seconds": wall_seconds,
            "evidence_counts": {
                "configurations": len(configurations),
                "folds_per_configuration": len(folds),
                "independent_refit_rounds": rounds,
                "sessions_per_round": sessions_per_round,
                "official_session_rows": len(session_frame),
                "official_raw_measurement_rows": len(raw_rows),
                "diagnostic_session_rows": len(diagnostic_session_frame),
                "diagnostic_raw_measurement_rows": len(diagnostic_raw_frame),
                "measurement_failures": int(session_frame["measurement_failed"].sum()),
                "contention_flags": int(session_frame["contention_detected"].sum()),
            },
            "gate": gate,
            "constraints": {
                "latency_cap_ms_per_1000": latency_cap,
                "latency_cap_interpretation": "fixed_absolute_deployment_target",
                "model_size_cap_mb": size_cap,
                "configurations": constraint_rows,
            },
            "immutable_artifacts_unchanged": all_immutable,
            "holdout_overlap_count": 0,
            "holdout_labels_used": False,
            "full_hpo_executed": False,
            "git_commit_or_push_performed": False,
            "user_acceptance_required": True,
        }
        write_json(acceptance, output / "acceptance_summary.json")
        manifest = {
            "protocol_version": config["protocol_version"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": wall_seconds,
            "dataset_sha256": sha256_file(data_path),
            "split_artifact_sha256": sha256_file(split_path),
            "train_indices_sha256": sha256_array(train_indices),
            "test_indices_used_only_for_overlap_assertion": True,
            "holdout_features_or_labels_used_for_benchmark": False,
            "cv_fold_artifact": fold_reference,
            "benchmark_batch": json.loads(
                (output / "benchmark_batch.json").read_text(encoding="utf-8")
            ),
            "timing_protocol": shared_timing,
            "diagnostic_protocol": diagnostic_cfg,
            "constraint_decision": {
                "approved_option": "A",
                "latency_cap_ms_per_1000": latency_cap,
                "interpretation": "fixed absolute deployment target",
                "not_claimed_as_1_5x_calibrated_cv_baseline": True,
                "model_size_cap_mb": size_cap,
                "full_run_blocked": True,
            },
            "stability_gate": gate,
            "baseline_hashes_before": hashes_before["baseline"],
            "baseline_hashes_after": hashes_after["baseline"],
            "pilot_hashes_before": hashes_before["pilot"],
            "pilot_hashes_after": hashes_after["pilot"],
            "v1_hashes_before": hashes_before["v1"],
            "v1_hashes_after": hashes_after["v1"],
            "immutable_artifacts_unchanged": all_immutable,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "joblib", "numpy", "optuna", "pandas", "scikit-learn", "threadpoolctl"
                    ]
                },
                "power_plan_changed": False,
                "process_priority_changed": False,
                "cpu_affinity_changed": False,
                "start": environment_start,
                "end": environment_end,
            },
            "config_summary": json.loads(config_frame.to_json(orient="records")),
            "artifact_files": sorted(path.name for path in output.iterdir() if path.is_file()),
        }
        write_json(manifest, output / "manifest.json")
        return manifest
    except Exception as exc:
        write_json({
            "protocol_version": config.get("protocol_version"),
            "status": "FAIL",
            "full_run_blocked": True,
            "error": f"{type(exc).__name__}: {exc}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, output / "calibration_failure.json")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run train-only latency calibration v2.")
    parser.add_argument("--config", default="configs/latency_calibration_v2.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        config = load_calibration_config(args.config)
        manifest = run_calibration_v2(config)
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Calibration v2 aborted: {exc}")
        return 2
    print(json.dumps({
        "output": str(resolve_project_path(config["output_dir"])),
        "gate": manifest["stability_gate"],
        "full_run_blocked": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
