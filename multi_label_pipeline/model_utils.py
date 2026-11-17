"""Utilities for independent binary classifiers and positive probabilities."""

import numpy as np


def positive_probabilities(model, X):
    """Keep the fixed target order and select class 1 explicitly."""
    return np.column_stack([
        probabilities[:, list(estimator.classes_).index(1)]
        for estimator, probabilities in zip(model.estimators_, model.predict_proba(X))
    ])


class BinaryRelevancePredictor:
    """Persist individually fitted estimators, including per-label parameters."""

    def __init__(self, estimators):
        self.estimators_ = estimators
        self.classes_ = [estimator.classes_ for estimator in estimators]

    def predict_proba(self, X):
        return [estimator.predict_proba(X) for estimator in self.estimators_]

    def predict(self, X):
        return np.column_stack([estimator.predict(X) for estimator in self.estimators_])
