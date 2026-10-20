from __future__ import annotations

import copy

import numpy as np
import optuna
import pandas as pd

from src.benchmark import load_config
from src.data import prepare_features
from src.optimize import (
    build_fixed_folds,
    load_hpo_config,
    main,
    optuna_constraints,
    run_study,
    suggest_parameters,
)
from src.pareto import constraint_status, pareto_front
from src.reporting import generate_combined_report


def test_week3_search_space_and_objective_directions_are_frozen():
    base = load_config("configs/experiment.yaml")
    hpo = load_hpo_config("configs/hpo_week3.yaml")
    space = base["optimization"]["search_space"]
    assert (space["n_estimators"]["low"], space["n_estimators"]["high"], space["n_estimators"]["step"]) == (50, 300, 25)
    assert (space["max_depth"]["low"], space["max_depth"]["high"]) == (4, 24)
    assert (space["min_samples_leaf"]["low"], space["min_samples_leaf"]["high"]) == (1, 12)
    assert (space["max_features"]["low"], space["max_features"]["high"]) == (0.4, 1.0)
    assert space["class_weight"]["choices"] == [None, "balanced", "balanced_subsample"]
    assert hpo["directions"] == ["maximize", "minimize"]

    fixed = optuna.trial.FixedTrial({
        "n_estimators": 300,
        "max_depth": 24,
        "min_samples_leaf": 12,
        "max_features": 1.0,
        "class_weight": "balanced_subsample",
    })
    assert suggest_parameters(fixed, space)["n_estimators"] == 300


def test_constraint_boundary_is_feasible_and_positive_excess_is_not():
    boundary = constraint_status(
        50.0, 35.53, model_size_cap_mb=50.0, latency_cap_ms=35.53
    )
    assert boundary["feasible"] is True
    assert boundary["size_violation_mb"] == 0.0
    assert boundary["latency_violation_ms"] == 0.0
    assert constraint_status(
        50.0001, 35.53, model_size_cap_mb=50.0, latency_cap_ms=35.53
    )["feasible"] is False

    feasible = optuna.trial.create_trial(
        values=[0.4, 20.0],
        params={},
        distributions={},
        user_attrs={"constraint_values": [-1.0, 0.0]},
    )
    rejected = optuna.trial.create_trial(
        values=[0.5, 10.0],
        params={},
        distributions={},
        user_attrs={"constraint_values": [0.001, -2.0]},
    )
    assert all(value <= 0 for value in optuna_constraints(feasible))
    assert any(value > 0 for value in optuna_constraints(rejected))


def test_fixed_folds_are_reproducible_disjoint_and_cover_training_data(synthetic_bank_frame):
    X, y = prepare_features(synthetic_bank_frame)
    config = load_config("configs/experiment.yaml")
    first = build_fixed_folds(X, y, config)
    second = build_fixed_folds(X, y, config)
    validation_positions: list[int] = []
    for (fit_a, validation_a), (fit_b, validation_b) in zip(first, second):
        assert np.array_equal(fit_a, fit_b)
        assert np.array_equal(validation_a, validation_b)
        assert set(fit_a).isdisjoint(validation_a)
        validation_positions.extend(validation_a.tolist())
    assert sorted(validation_positions) == list(range(len(X)))


def test_pareto_uses_only_completed_feasible_trials():
    trials = pd.DataFrame([
        {"sampler": "random", "trial_number": 0, "state": "COMPLETE", "feasible": True, "average_precision_cv_mean": 0.40, "latency_per_1000_samples_ms": 20.0, "model_size_max_mb": 10.0},
        {"sampler": "random", "trial_number": 1, "state": "COMPLETE", "feasible": True, "average_precision_cv_mean": 0.39, "latency_per_1000_samples_ms": 22.0, "model_size_max_mb": 10.0},
        {"sampler": "tpe", "trial_number": 0, "state": "COMPLETE", "feasible": True, "average_precision_cv_mean": 0.42, "latency_per_1000_samples_ms": 25.0, "model_size_max_mb": 12.0},
        {"sampler": "tpe", "trial_number": 1, "state": "COMPLETE", "feasible": False, "average_precision_cv_mean": 0.50, "latency_per_1000_samples_ms": 18.0, "model_size_max_mb": 60.0},
        {"sampler": "tpe", "trial_number": 2, "state": "FAIL", "feasible": False, "average_precision_cv_mean": np.nan, "latency_per_1000_samples_ms": np.nan, "model_size_max_mb": np.nan},
    ])
    front = pareto_front(trials)
    assert set(zip(front["sampler"], front["trial_number"])) == {("random", 0), ("tpe", 0)}


def test_reporting_writes_rebuildable_csv_json_and_figures(tmp_path):
    frame = pd.DataFrame([
        {
            "sampler": "random", "trial_number": 0, "state": "COMPLETE",
            "average_precision_cv_mean": 0.4, "latency_per_1000_samples_ms": 20.0,
            "model_size_max_mb": 10.0, "feasible": True,
            "size_violation_mb": -40.0, "latency_violation_ms": -15.0,
            "runtime_seconds": 1.0,
        }
    ])
    report = generate_combined_report([frame], tmp_path)
    assert report["summary"]["completed"] == 1
    assert report["summary"]["feasible"] == 1
    for name in [
        "trial_history.csv", "trial_history.json", "pareto_front.csv",
        "best_feasible_ap_vs_trials.png", "ap_vs_latency_pareto.png",
        "constraint_violations.png",
    ]:
        assert (tmp_path / name).is_file()


def test_resumable_study_does_not_add_trials_after_target(tmp_path, synthetic_bank_frame):
    X, y = prepare_features(synthetic_bank_frame)
    base = copy.deepcopy(load_config("configs/experiment.yaml"))
    base["optimization"]["search_space"].update({
        "n_estimators": {"type": "int", "low": 5, "high": 5, "step": 1},
        "max_depth": {"type": "int", "low": 4, "high": 4},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 1},
        "max_features": {"type": "float", "low": 0.5, "high": 0.5},
        "class_weight": {"type": "categorical", "choices": [None]},
    })
    base["latency"].update({"batch_size": 20, "warmup_runs": 1, "repeats": 5})
    hpo = copy.deepcopy(load_hpo_config("configs/hpo_week3.yaml"))
    folds = build_fixed_folds(X, y, base)
    reference = {"model_size_cap_mb": 50.0, "latency_cap_ms": 10_000.0}
    kwargs = dict(
        sampler_name="random", mode="pilot", target_trials=1,
        X_train=X, y_train=y, folds=folds, base_config=base, hpo_config=hpo,
        baseline_reference=reference, run_dir=tmp_path, protocol_fingerprint="test",
    )
    first = run_study(**kwargs)
    second = run_study(**kwargs)
    assert len(first.trials) == 1
    assert len(second.trials) == 1


def test_full_mode_is_blocked_without_explicit_confirmation(capsys):
    assert main(["--mode", "full", "--sampler", "both", "--trials", "40"]) == 2
    assert "requires --confirm-full" in capsys.readouterr().out
