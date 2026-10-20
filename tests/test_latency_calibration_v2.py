from __future__ import annotations

import json

import numpy as np

import pandas as pd

from src.latency_calibration import load_calibration_config
from src.latency_calibration_v2 import (
    _analyze_v1,
    _constraint_and_pareto_audit,
    aggregate_v2,
    evaluate_gate_v2,
)
from src.optimize import load_hpo_config
from src.paths import resolve_project_path


def _v2_sessions(*, unstable_round: bool = False) -> pd.DataFrame:
    rows = []
    for config_id, label, base in [("a", "baseline", 10.0), ("b", "pilot", 20.0)]:
        for fold in [1, 2]:
            for round_number in [1, 2]:
                for session in [1, 2, 3]:
                    value = base + fold + (round_number - 1) * 0.1 + session * 0.01
                    if unstable_round and config_id == "a" and fold == 2 and round_number == 2:
                        value = [10.0, 12.0, 14.0][session - 1]
                    rows.append({
                        "config_id": config_id,
                        "label": label,
                        "fold": fold,
                        "round": round_number,
                        "session_in_round": session,
                        "session_global": (round_number - 1) * 3 + session,
                        "session_median_ms_per_1000": value,
                        "model_size_bytes": 1_000_000,
                        "measurement_failed": False,
                        "contention_detected": False,
                        "batch_original_indices_sha256": "shared",
                    })
    return pd.DataFrame(rows)


def test_v2_protocol_is_preregistered_without_changing_constraints_or_aggregation():
    config = load_calibration_config("configs/latency_calibration_v2.yaml")
    assert config["output_dir"].endswith("latency_calibration_v2")
    assert config["timing"]["independent_refit_rounds"] == 2
    assert config["timing"]["sessions_per_round"] == 5
    assert config["timing"]["warmup_calls_per_session"] == 10
    assert config["timing"]["timed_repeats_per_session"] == 20
    assert config["timing"]["aggregation"] == (
        "median_all_session_medians_per_fold_then_max_across_folds"
    )
    assert config["stability_gate"]["maximum_session_median_relative_range"] == 0.15
    assert config["constraints"]["latency_cap_ms_per_1000"] == 35.531700000092314
    assert config["constraints"]["latency_cap_interpretation"] == (
        "fixed_absolute_deployment_target"
    )
    assert config["constraints"]["model_size_cap_mb"] == 50.0
    assert config["official_study_plan"]["full_run_blocked"] is True


def test_v1_diagnostics_read_real_historical_schema_without_mutation():
    analysis = _analyze_v1(resolve_project_path("results/hpo/latency_calibration_v1"))
    assert analysis["csv_rows"] == {
        "raw_measurements": 900,
        "session_summary": 60,
        "fold_summary": 12,
        "config_summary": 4,
    }
    assert analysis["baseline_fold3_session_medians_ms_per_1000"] == [
        21.1248, 21.4055, 18.8868, 19.4575, 18.0182
    ]


def test_v2_aggregation_preserves_rounds_and_uses_all_session_medians():
    folds, configs = aggregate_v2(_v2_sessions())
    row = folds.loc[(folds["config_id"].eq("a")) & (folds["fold"].eq(1))].iloc[0]
    assert len(json.loads(row["session_medians_ms_per_1000_json"])) == 6
    assert set(json.loads(row["round_relative_ranges_json"])) == {"1", "2"}
    config = configs.loc[configs["config_id"].eq("a")].iloc[0]
    expected = folds.loc[folds["config_id"].eq("a")][
        "fold_median_of_all_session_medians_ms_per_1000"
    ].max()
    assert config["aggregate_max_fold_median_ms_per_1000"] == expected


def test_v2_gate_requires_each_independent_round_to_pass():
    sessions = _v2_sessions(unstable_round=True)
    folds, _ = aggregate_v2(sessions)
    gate = evaluate_gate_v2(
        sessions,
        folds,
        expected_batch_hash="shared",
        expected_config_count=2,
        expected_folds=2,
        expected_rounds=2,
        sessions_per_round=3,
        maximum_relative_range=0.15,
        prediction_digest_consistent=True,
    )
    assert gate["status"] == "FAIL"
    assert gate["full_run_blocked"] is True
    assert "one or more config/folds exceed 15% within an independent round" in gate[
        "failure_reasons"
    ]


def test_v2_gate_blocks_prediction_changes_and_pilot_constraints_remain_consistent():
    sessions = _v2_sessions()
    folds, _ = aggregate_v2(sessions)
    gate = evaluate_gate_v2(
        sessions,
        folds,
        expected_batch_hash="shared",
        expected_config_count=2,
        expected_folds=2,
        expected_rounds=2,
        sessions_per_round=3,
        maximum_relative_range=0.15,
        prediction_digest_consistent=False,
    )
    assert gate["status"] == "FAIL"
    assert "prediction digest changed across sessions/refit rounds" in gate["failure_reasons"]

    hpo = load_hpo_config("configs/hpo_week3.yaml")
    audit = _constraint_and_pareto_audit(
        resolve_project_path("results/hpo/pilot/trial_history.csv"),
        hpo,
        latency_cap=35.531700000092314,
        size_cap=50.0,
    )
    assert audit["objective_directions"] == ["maximize", "minimize"]
    assert audit["constraint_convention"] == "g_i <= 0 is feasible"
    assert audit["pass"] is True


def test_completed_v2_artifacts_are_complete_train_only_and_still_block_full():
    root = resolve_project_path("results/hpo/latency_calibration_v2")
    required = {
        "manifest.json", "raw_measurements.csv", "session_summary.csv",
        "fold_summary.csv", "config_summary.csv", "stability_diagnostics.json",
        "gate_decision.json", "acceptance_summary.json",
    }
    assert required.issubset({path.name for path in root.iterdir()})
    raw = pd.read_csv(root / "raw_measurements.csv")
    sessions = pd.read_csv(root / "session_summary.csv")
    diagnostic_raw = pd.read_csv(root / "diagnostic_raw_measurements.csv")
    assert len(raw) == len(raw.drop_duplicates(["config_id", "fold", "round", "session_in_round", "repeat"])) == 2400
    assert len(sessions) == 120
    assert len(diagnostic_raw) == 600
    assert sessions["batch_original_indices_sha256"].nunique() == 1
    with np.load(root / "benchmark_batch_indices.npz") as batch, np.load(
        resolve_project_path("results/baseline/split_indices.npz")
    ) as split:
        assert np.intersect1d(batch["original_indices"], split["test_indices"]).size == 0
    gate = json.loads((root / "gate_decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert gate["status"] == "FAIL"
    assert gate["full_run_blocked"] is True
    assert manifest["immutable_artifacts_unchanged"] is True

