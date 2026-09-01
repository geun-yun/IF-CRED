"""Reference weighted local prediction-consistency metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from numpy.typing import ArrayLike

from ifcred.core.components import weighted_prediction_consistency
from ifcred.core.fairness import FairnessEvaluation


@dataclass(frozen=True)
class WeightedPredictionConsistencyInputs:
    """Inputs required by the study's reference fairness metric."""

    probabilities_by_model: Mapping[str, ArrayLike]
    neighbour_indices: ArrayLike
    weights: ArrayLike


@dataclass(frozen=True)
class WeightedPredictionConsistency:
    """Study-selected metric based on weighted prediction differences."""

    epsilon: float = 1e-12
    name: str = "weighted_prediction_consistency"

    def evaluate(self, context: WeightedPredictionConsistencyInputs) -> FairnessEvaluation:
        model_scores, local_scores = weighted_prediction_consistency(
            context.probabilities_by_model,
            context.neighbour_indices,
            context.weights,
            epsilon=self.epsilon,
        )
        return FairnessEvaluation(
            metric_name=self.name,
            model_scores=model_scores,
            local_scores=local_scores,
        )

