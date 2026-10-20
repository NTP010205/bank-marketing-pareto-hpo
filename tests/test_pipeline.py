from __future__ import annotations

import copy

from src.benchmark import load_config, run_experiment
from src.data import prepare_features
from src.pipeline import build_baseline_pipeline


def test_column_transformer_is_fit_only_on_fold_training_rows(synthetic_bank_frame):
    X, y = prepare_features(synthetic_bank_frame)
    X = X.copy()
    X.loc[X.index[-1], "job"] = "validation_only_job"
    config = load_config("configs/experiment.yaml")
    pipeline = build_baseline_pipeline(X.iloc[:-1], config)
    pipeline.fit(X.iloc[:-1], y.iloc[:-1])

    categorical = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    learned = {str(value) for values in categorical.named_steps["onehot"].categories_ for value in values}
    assert "validation_only_job" not in learned
    transformed = pipeline.named_steps["preprocess"].transform(X.iloc[[-1]])
    assert transformed.shape[0] == 1


def test_baseline_smoke_writes_real_artifacts_from_test_fixture(tmp_path, synthetic_bank_frame):
    config = copy.deepcopy(load_config("configs/experiment.yaml"))
    config["baseline"]["n_estimators"] = 5
    config["latency"].update({"batch_size": 20, "warmup_runs": 1, "repeats": 5})
    result = run_experiment(synthetic_bank_frame, config, tmp_path)

    assert (tmp_path / "fold_metrics.csv").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "split_indices.npz").is_file()
    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "week3_constraints.json").is_file()
    assert result["latency"]["repeats"] == 5
