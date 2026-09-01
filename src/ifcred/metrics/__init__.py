"""Individual-fairness metric adapters shipped with IF-CRED."""

from ifcred.metrics.weighted_prediction_consistency import (
    WeightedPredictionConsistency,
    WeightedPredictionConsistencyInputs,
)

__all__ = ["WeightedPredictionConsistency", "WeightedPredictionConsistencyInputs"]

