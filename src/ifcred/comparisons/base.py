from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ComparisonCase:
    dataset_id: str
    condition: str
    variant: str
    ratio: float
    repetition: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    protected_index: int | None
    random_seed: int


def predict_probability(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(model.classes_)
        return probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    return np.asarray(model.predict(X), dtype=float)
