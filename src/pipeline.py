"""Leakage-safe preprocessing and baseline estimator construction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def infer_column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = infer_column_groups(X)
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def build_baseline_pipeline(X: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    baseline = config["baseline"]
    model = RandomForestClassifier(
        n_estimators=int(baseline["n_estimators"]),
        random_state=int(baseline["random_state"]),
        n_jobs=int(baseline["n_jobs"]),
    )
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", model)])
