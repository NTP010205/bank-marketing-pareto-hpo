"""Classification metrics; Average Precision always consumes probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_from_probabilities(
    y_true: Any, positive_probability: Any, *, threshold: float = 0.5
) -> dict[str, float]:
    probability = np.asarray(positive_probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
    }


def evaluate_estimator(estimator: Any, X: Any, y: Any, *, threshold: float = 0.5) -> dict[str, float]:
    probability = estimator.predict_proba(X)[:, 1]
    return metrics_from_probabilities(y, probability, threshold=threshold)
