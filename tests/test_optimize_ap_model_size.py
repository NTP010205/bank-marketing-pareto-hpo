from __future__ import annotations

import copy

import numpy as np
import optuna
import pandas as pd

from src.optimize_ap_model_size import (
    audit_historical_config_hashes,
    build_fixed_folds,
    create_objective,
    evaluate_configuration_cv,
    generate_official_figures,
    load_yaml_mapping,
    main,
    optuna_size_constraints,
    prepare_training_features,
    rank_selection_candidates,
    run_study,
    serialize_fitted_pipeline,
    sha256_array,
    study_to_frames,
    unique_pareto_configurations,
)
from src.pipeline import build_baseline_pipeline


def _fold_metadata(folds, original_indices):
    rows = []
    for fold_number, (fit, validation) in enumerate(folds, start=1):
        rows.append(
            {
                "fold": fold_number,
                "fit_positions_sha256": sha256_array(fit),
                "validation_positions_sha256": sha256_array(validation),
                "fit_original_indices_sha256": sha256_array(original_indices[fit]),
                "validation_original_indices_sha256": sha256_array(
                    original_indices[validation]
                ),
            }
        )
    return rows


def _small_base_config():
    base = copy.deepcopy(load_yaml_mapping("configs/experiment.yaml"))
    base["optimization"]["search_space"].update(
        {
            "n_estimators": {"type": "int", "low": 5, "high": 5, "step": 1},
            "max_depth": {"type": "int", "low": 4, "high": 4},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 1},
            "max_features": {"type": "float", "low": 0.5, "high": 0.5},
            "class_weight": {"type": "categorical", "choices": [None]},
        }
    )
    return base


def test_approved_config_has_exact_objectives_budget_and_full_run_01_lock():
    config = load_yaml_mapping("configs/hpo_ap_model_size_v1.yaml")
    assert config["status"] == "approved_for_full_run_01"
    assert config["directions"] == ["maximize", "minimize"]
    assert config["objective_names"] == [
        "average_precision_cv_mean",
        "model_size_max_mb",
    ]
    assert config["constraints"] == {
        "model_size_cap_mb_decimal": 50.0,
        "feasibility_rule": "g_le_zero",
        "latency_used": False,
    }
    assert config["smoke"]["trials_per_sampler"] == 3
    assert config["tpe"]["n_startup_trials"] == 10
    assert config["full"]["trials_per_sampler"] == 40
    assert config["full"]["storage_subdir"] == "full_run_01"
    assert config["full"]["full_run_blocked"] is False
    assert config["full"]["explicit_approval_recorded"] is True
    assert config["selection"]["best_ap_tolerance_absolute"] == 0.005


def test_cv_evaluation_excludes_duration_and_uses_max_fold_size(
    tmp_path,
    synthetic_bank_frame,
):
    X, y = prepare_training_features(synthetic_bank_frame)
    assert "duration" not in X.columns
    base = _small_base_config()
    folds = build_fixed_folds(X, y, base)
    metadata = _fold_metadata(folds, np.arange(len(X), dtype=np.int64))
    params = {
        "n_estimators": 5,
        "max_depth": 4,
        "min_samples_leaf": 1,
        "max_features": 0.5,
        "class_weight": None,
    }
    result = evaluate_configuration_cv(
        X_train=X,
        y_train=y,
        folds=folds,
        fold_metadata=metadata,
        base_config=base,
        params=params,
        artifact_dir=tmp_path,
        artifact_prefix="test",
        compression=3,
        thread_limit=1,
        verify_repeat_serialization=True,
    )
    assert len(result["folds"]) == 3
    assert result["model_size_max_bytes"] == max(
        fold["model_size_bytes"] for fold in result["folds"]
    )
    assert result["model_size_max_mb"] == result["model_size_max_bytes"] / 1_000_000
    assert all(fold["repeat_serialization_exact"] for fold in result["folds"])
    assert all(fold["pipeline_steps"] == ["preprocess", "model"] for fold in result["folds"])


def test_serialization_is_complete_and_byte_reproducible(tmp_path, synthetic_bank_frame):
    X, y = prepare_training_features(synthetic_bank_frame)
    estimator = build_baseline_pipeline(X, load_yaml_mapping("configs/experiment.yaml"))
    estimator.fit(X, y)
    result = serialize_fitted_pipeline(
        estimator,
        tmp_path / "whole_pipeline.joblib",
        compression=3,
        verify_repeat=True,
    )
    assert result["pipeline_steps"] == ["preprocess", "model"]
    assert result["repeat_serialization_exact"] is True
    assert result["model_size_bytes"] == result["repeat_size_bytes"]


def test_objective_and_constraint_are_ap_then_size_only(tmp_path, synthetic_bank_frame):
    X, y = prepare_training_features(synthetic_bank_frame)
    base = _small_base_config()
    protocol = copy.deepcopy(load_yaml_mapping("configs/hpo_ap_model_size_v1.yaml"))
    folds = build_fixed_folds(X, y, base)
    metadata = _fold_metadata(folds, np.arange(len(X), dtype=np.int64))
    objective = create_objective(
        X_train=X,
        y_train=y,
        folds=folds,
        fold_metadata=metadata,
        base_config=base,
        protocol_config=protocol,
        artifact_dir=tmp_path,
    )
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 5,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "max_features": 0.5,
            "class_weight": None,
        }
    )
    ap, size = objective(trial)
    assert ap == trial.user_attrs["average_precision_cv_mean"]
    assert size == trial.user_attrs["model_size_max_mb"]
    assert trial.user_attrs["constraint_values"] == [size - 50.0]
    assert trial.user_attrs["latency_used"] is False


def test_optuna_size_constraint_boundary_and_missing_value():
    boundary = optuna.trial.create_trial(
        values=[0.4, 50.0],
        params={},
        distributions={},
        user_attrs={"constraint_values": [0.0]},
    )
    missing = optuna.trial.create_trial(
        values=[0.4, 10.0], params={}, distributions={}, user_attrs={}
    )
    assert optuna_size_constraints(boundary) == (0.0,)
    assert optuna_size_constraints(missing) == (float("inf"),)


def test_study_resume_stops_at_target_budget(tmp_path, synthetic_bank_frame):
    X, y = prepare_training_features(synthetic_bank_frame)
    base = _small_base_config()
    protocol = copy.deepcopy(load_yaml_mapping("configs/hpo_ap_model_size_v1.yaml"))
    protocol["serialization"]["repeat_serialization_check_in_smoke"] = False
    folds = build_fixed_folds(X, y, base)
    metadata = _fold_metadata(folds, np.arange(len(X), dtype=np.int64))
    kwargs = dict(
        sampler_name="random",
        mode="smoke",
        target_trials=1,
        X_train=X,
        y_train=y,
        folds=folds,
        fold_metadata=metadata,
        base_config=base,
        protocol_config=protocol,
        run_dir=tmp_path,
        protocol_fingerprint="test-protocol",
    )
    first, _ = run_study(**kwargs)
    second, _ = run_study(**kwargs)
    assert len(first.trials) == 1
    assert len(second.trials) == 1
    trials, fold_metrics = study_to_frames(second, sampler_name="random", seed=42)
    assert len(trials) == 1
    assert len(fold_metrics) == 3


def test_full_mode_requires_explicit_confirmation(capsys):
    assert main(["--mode", "full", "--sampler", "both", "--trials", "40"]) == 2
    assert "requires --confirm-full" in capsys.readouterr().out


def test_historical_hash_difference_is_only_csv_float_rounding():
    audit = audit_historical_config_hashes(load_yaml_mapping("configs/hpo_ap_model_size_v1.yaml"))
    assert audit["raw_sqlite_parameters_equal"] is True
    assert audit["direct_canonical_hash"] == "b4e97ab5ded9"
    assert audit["historical_csv_roundtrip_hash"] == "48e74e2e02b3"
    assert audit["max_features_absolute_delta"] == 2**-54


def test_selection_rule_and_unique_pareto_are_deterministic(tmp_path):
    trials = pd.DataFrame(
        [
            {
                "sampler": "random", "trial_number": 0, "state": "COMPLETE",
                "feasible": True, "config_hash": "a", "average_precision_cv_mean": 0.450,
                "average_precision_cv_std": 0.01, "model_size_max_bytes": 5_000_000,
                "model_size_max_mb": 5.0, "runtime_seconds": 1.0,
                "param_n_estimators": 100, "param_max_depth": 10,
                "param_min_samples_leaf": 2, "param_max_features": 0.5,
                "param_class_weight": None,
            },
            {
                "sampler": "tpe", "trial_number": 0, "state": "COMPLETE",
                "feasible": True, "config_hash": "b", "average_precision_cv_mean": 0.447,
                "average_precision_cv_std": 0.01, "model_size_max_bytes": 3_000_000,
                "model_size_max_mb": 3.0, "runtime_seconds": 1.0,
                "param_n_estimators": 75, "param_max_depth": 8,
                "param_min_samples_leaf": 2, "param_max_features": 0.5,
                "param_class_weight": None,
            },
            {
                "sampler": "random", "trial_number": 1, "state": "COMPLETE",
                "feasible": True, "config_hash": "b", "average_precision_cv_mean": 0.447,
                "average_precision_cv_std": 0.01, "model_size_max_bytes": 3_000_000,
                "model_size_max_mb": 3.0, "runtime_seconds": 1.0,
                "param_n_estimators": 75, "param_max_depth": 8,
                "param_min_samples_leaf": 2, "param_max_features": 0.5,
                "param_class_weight": None,
            },
        ]
    )
    ranked, best, threshold = rank_selection_candidates(trials, ap_tolerance=0.005)
    assert best == 0.450
    assert threshold == 0.445
    assert ranked.iloc[0]["config_hash"] == "b"

    front = trials.copy()
    unique = unique_pareto_configurations(front)
    assert len(unique) == 2
    assert int(unique.loc[unique["config_hash"] == "b", "pareto_occurrences"].iloc[0]) == 2
    figure_manifest = generate_official_figures(trials, front, unique, tmp_path)
    assert len(figure_manifest["figures"]) == 5
    assert all((tmp_path / name).is_file() for name in figure_manifest["figures"])
