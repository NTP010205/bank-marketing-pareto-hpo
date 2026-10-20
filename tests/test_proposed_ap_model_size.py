from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.benchmark import load_config
from src.data import prepare_features, sha256_file
from src.pareto import model_size_constraint_status, pareto_front_ap_model_size
from src.pipeline import build_baseline_pipeline


def test_serialized_size_covers_the_fitted_preprocessor_and_estimator(
    tmp_path,
    synthetic_bank_frame,
):
    X, y = prepare_features(synthetic_bank_frame)
    config = load_config("configs/experiment.yaml")
    estimator = build_baseline_pipeline(X, config)
    estimator.fit(X, y)

    first_path = tmp_path / "pipeline_first.joblib"
    second_path = tmp_path / "pipeline_second.joblib"
    compression = int(config["artifact"]["compression"])
    joblib.dump(estimator, first_path, compress=compression)
    joblib.dump(estimator, second_path, compress=compression)

    restored = joblib.load(first_path)
    assert list(restored.named_steps) == ["preprocess", "model"]
    assert restored.predict_proba(X).shape == (len(X), 2)
    assert first_path.stat().st_size == second_path.stat().st_size
    assert sha256_file(first_path) == sha256_file(second_path)


def test_model_size_units_and_max_fold_aggregation_are_explicit():
    size_bytes = 5_952_823
    assert size_bytes / 1_000_000 == 5.952823
    assert size_bytes / (1024**2) == pytest.approx(5.677054405212402)
    assert max([5_791_055, 5_682_181, 5_952_823]) == size_bytes


def test_size_only_constraint_uses_nonpositive_feasibility_boundary():
    assert model_size_constraint_status(
        50.0,
        model_size_cap_mb=50.0,
    ) == {"size_violation_mb": 0.0, "feasible": True}
    assert model_size_constraint_status(
        50.000001,
        model_size_cap_mb=50.0,
    )["feasible"] is False


def test_ap_model_size_pareto_excludes_dominated_infeasible_and_failed_trials():
    trials = pd.DataFrame(
        [
            {
                "sampler": "random",
                "trial_number": 0,
                "state": "COMPLETE",
                "feasible": True,
                "average_precision_cv_mean": 0.40,
                "model_size_max_mb": 4.0,
            },
            {
                "sampler": "random",
                "trial_number": 1,
                "state": "COMPLETE",
                "feasible": True,
                "average_precision_cv_mean": 0.39,
                "model_size_max_mb": 5.0,
            },
            {
                "sampler": "tpe",
                "trial_number": 0,
                "state": "COMPLETE",
                "feasible": True,
                "average_precision_cv_mean": 0.42,
                "model_size_max_mb": 6.0,
            },
            {
                "sampler": "tpe",
                "trial_number": 1,
                "state": "COMPLETE",
                "feasible": False,
                "average_precision_cv_mean": 0.50,
                "model_size_max_mb": 60.0,
            },
            {
                "sampler": "tpe",
                "trial_number": 2,
                "state": "FAIL",
                "feasible": False,
                "average_precision_cv_mean": np.nan,
                "model_size_max_mb": np.nan,
            },
        ]
    )

    front = pareto_front_ap_model_size(trials)
    assert set(zip(front["sampler"], front["trial_number"])) == {
        ("random", 0),
        ("tpe", 0),
    }
