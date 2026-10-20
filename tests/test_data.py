from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DatasetNotFoundError,
    EXPECTED_COLUMNS,
    load_bank_data,
    make_holdout_split,
    prepare_features,
)


def test_missing_file_has_actionable_message(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="bank-full.csv"):
        load_bank_data(tmp_path / "bank-full.csv")


def test_schema_validation_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bank-full.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError, match="Unexpected schema"):
        load_bank_data(path, validate_shape=False)


def test_duration_is_never_a_feature(synthetic_bank_frame):
    X, y = prepare_features(synthetic_bank_frame)
    assert "duration" not in X.columns
    assert "y" not in X.columns
    assert set(y.unique()) == {0, 1}


def test_split_is_reproducible_stratified_and_non_overlapping(synthetic_bank_frame):
    X, y = prepare_features(synthetic_bank_frame)
    first = make_holdout_split(X, y, test_size=0.2, random_state=42)
    second = make_holdout_split(X, y, test_size=0.2, random_state=42)
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.test_indices, second.test_indices)
    assert set(first.train_indices).isdisjoint(first.test_indices)
    assert len(first.train_indices) + len(first.test_indices) == len(X)
    assert abs(int(first.y_train.sum()) - round(float(y.mean()) * len(first.y_train))) <= 1
    assert abs(int(first.y_test.sum()) - round(float(y.mean()) * len(first.y_test))) <= 1


def test_expected_schema_fixture_is_complete(synthetic_bank_frame):
    assert tuple(synthetic_bank_frame.columns) == EXPECTED_COLUMNS
