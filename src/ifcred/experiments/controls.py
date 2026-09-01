"""Interpretation controls separating prediction consistency from usefulness."""

from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from ifcred.core.fairness import assess_ifcred


def prediction_control_assessments(context, graph, coverage_score: float, distance_score: float, X_test, y_test) -> Mapping[str, dict]:
    """Evaluate constant, random, and smooth-uninformative probability controls."""

    X = np.asarray(X_test, dtype=float); y = np.asarray(y_test, dtype=int)
    prevalence = float(np.mean(y))
    rng = np.random.default_rng(context.repetition.seed_for("model", "prediction_controls"))
    smooth = 1.0 / (1.0 + np.exp(-np.clip(X[:, 0], -10, 10)))
    controls = {
        "constant_prevalence": {f"control_{i}": np.full(len(X), prevalence) for i in range(5)},
        "random_uniform": {f"control_{i}": rng.random(len(X)) for i in range(5)},
        "smooth_uninformative": {f"control_{i}": smooth.copy() for i in range(5)},
    }
    output = {}
    for name, probabilities in controls.items():
        fairness = context.setup.fairness.evaluate(probabilities, graph)
        assessment = assess_ifcred(C=coverage_score, D=distance_score, fairness=fairness)
        mean_probability = np.mean(np.column_stack(tuple(probabilities.values())), axis=1)
        output[name] = {
            "C": assessment.C, "D": assessment.D, "F": assessment.F,
            "M": assessment.M, "V": assessment.V,
            "accuracy": float(accuracy_score(y, mean_probability >= 0.5)),
            "roc_auc": float(roc_auc_score(y, mean_probability)),
            "brier_score": float(brier_score_loss(y, mean_probability)),
        }
    return output
