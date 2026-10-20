from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.latency_calibration import (
    aggregate_session_summaries,
    duration_ns_to_ms_per_1000,
    evaluate_stability_gate,
    load_calibration_config,
    select_benchmark_positions,
    sha256_array,
    summarize_repeats,
    validate_versioned_output_path,
)
from src.optimize import load_hpo_config, main, study_name_for
from src.paths import resolve_project_path


def _sessions(*, failed: bool = False, second_batch: bool = False) -> pd.DataFrame:
    rows = []
    values = {
        ("a", 1): [10.0, 11.0, 12.0],
        ("a", 2): [20.0, 21.0, 22.0],
        ("b", 1): [30.0, 31.0, 32.0],
        ("b", 2): [40.0, 41.0, 42.0],
    }
    for (config_id, fold), medians in values.items():
        for session, median in enumerate(medians, start=1):
            rows.append({
                "config_id": config_id,
                "fold": fold,
                "session": session,
                "session_median_ms_per_1000": median,
                "model_size_bytes": 1_000_000 + fold,
                "measurement_failed": failed and config_id == "a" and fold == 1 and session == 1,
                "contention_detected": False,
                "batch_original_indices_sha256": (
                    "other" if second_batch and config_id == "b" else "shared"
                ),
            })
    return pd.DataFrame(rows)


def test_real_benchmark_indices_are_deterministic_train_only_and_same_size():
    split_path = resolve_project_path("results/baseline/split_indices.npz")
    with np.load(split_path) as split:
        train = np.asarray(split["train_indices"], dtype=np.int64)
        test = np.asarray(split["test_indices"], dtype=np.int64)
    first_positions, first_original = select_benchmark_positions(
        train, test, batch_size=1000, seed=20261020
    )
    second_positions, second_original = select_benchmark_positions(
        train, test, batch_size=1000, seed=20261020
    )
    assert len(first_positions) == len(first_original) == 1000
    assert np.array_equal(first_positions, second_positions)
    assert np.array_equal(first_original, second_original)
    assert np.intersect1d(first_original, test).size == 0
    assert sha256_array(first_original) == sha256_array(second_original)


def test_latency_units_are_ms_per_1000_and_raw_values_are_preserved():
    assert duration_ns_to_ms_per_1000(2_000_000, 1000) == 2.0
    assert duration_ns_to_ms_per_1000(1_000_000, 500) == 2.0
    summary = summarize_repeats([1_000_000, 2_000_000, 3_000_000], 1000)
    assert summary["raw_duration_ns"] == [1_000_000, 2_000_000, 3_000_000]
    assert summary["raw_ms_per_1000"] == [1.0, 2.0, 3.0]
    assert summary["median_ms_per_1000"] == 2.0


def test_aggregation_is_median_across_sessions_then_max_across_folds():
    fold_frame, config_frame = aggregate_session_summaries(_sessions())
    a_folds = fold_frame.loc[fold_frame["config_id"].eq("a")].sort_values("fold")
    assert a_folds["fold_median_of_session_medians_ms_per_1000"].tolist() == [11.0, 21.0]
    a_config = config_frame.loc[config_frame["config_id"].eq("a")].iloc[0]
    assert a_config["aggregate_max_fold_median_ms_per_1000"] == 21.0
    assert json.loads(a_config["fold_medians_ms_per_1000_json"]) == [11.0, 21.0]


def test_gate_requires_one_shared_batch_and_blocks_failed_measurements():
    sessions = _sessions(failed=True, second_batch=True)
    folds, configs = aggregate_session_summaries(sessions)
    gate = evaluate_stability_gate(
        sessions,
        folds,
        configs,
        expected_batch_hash="shared",
        maximum_relative_range=0.15,
        expected_config_count=2,
        expected_folds=2,
        expected_sessions=3,
    )
    assert gate["status"] == "FAIL"
    assert gate["full_run_blocked"] is True
    assert "one or more measurements failed" in gate["failure_reasons"]
    assert "not every measurement used the identical benchmark batch" in gate["failure_reasons"]


def test_constraint_is_unchanged_and_full_study_is_versioned_and_blocked(capsys):
    calibration = load_calibration_config("configs/latency_calibration_v1.yaml")
    hpo = load_hpo_config("configs/hpo_week3.yaml")
    assert calibration["constraints"]["historical_latency_cap_ms_per_1000"] == pytest.approx(
        35.531700000092314
    )
    assert calibration["constraints"]["change_cap_automatically"] is False
    assert hpo["tpe"]["n_startup_trials"] == 10
    assert hpo["full"]["trials_per_sampler"] == 40
    assert hpo["full"]["storage_subdir"] == "official_latency_v1"
    assert study_name_for("full", "random", 42, hpo).endswith("official_latency_v1")
    assert main(["--mode", "full", "--sampler", "both", "--trials", "40", "--confirm-full"]) == 2
    assert "remains blocked" in capsys.readouterr().out


def test_output_path_must_be_versioned_and_existing_artifacts_are_not_overwritten(tmp_path: Path):
    with pytest.raises(ValueError):
        validate_versioned_output_path(tmp_path / "latency_calibration")
    versioned = tmp_path / "latency_calibration_v7"
    validate_versioned_output_path(versioned)
    versioned.mkdir()
    (versioned / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        validate_versioned_output_path(versioned)
