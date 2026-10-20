from __future__ import annotations

import numpy as np

from src.metrics import evaluate_estimator, metrics_from_probabilities


class ProbabilityOnlyEstimator:
    def predict_proba(self, X):
        positive = np.array([0.05, 0.4, 0.8, 0.95])
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        raise AssertionError("Average Precision evaluation must not call predict().")


def test_average_precision_uses_probability_not_hard_prediction():
    y = np.array([0, 1, 0, 1])
    probability = np.array([0.05, 0.4, 0.8, 0.95])
    direct = metrics_from_probabilities(y, probability)
    through_estimator = evaluate_estimator(ProbabilityOnlyEstimator(), np.zeros((4, 1)), y)
    assert through_estimator["average_precision"] == direct["average_precision"]
    assert 0.0 < direct["average_precision"] < 1.0
