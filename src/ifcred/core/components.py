"""Fixed IF-CRED component definitions.

This module operates on already-constructed audit evidence. It intentionally
contains no dataset, preprocessing, distance-metric, or model choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoverageResult:
    """Coverage score and its two diagnostic terms for every individual."""

    score: float
    balance_by_individual: FloatArray
    strength_by_individual: FloatArray
    local_coverage: FloatArray


def _as_finite_2d(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_unit_interval(array: FloatArray, *, name: str) -> None:
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain values in [0, 1]")


def coverage(weights: ArrayLike, intended_k: int | ArrayLike) -> CoverageResult:
    """Calculate C from an ``(individual, neighbour)`` weight matrix.

    Each row represents the declared neighbour slots for one individual; a
    zero is therefore weak or absent evidence within that declared set, not
    padding outside it. An individual with no positive evidence receives zero
    balance, strength, and local coverage.
    """

    weight_array = _as_finite_2d(weights, name="weights")
    _validate_unit_interval(weight_array, name="weights")

    if np.isscalar(intended_k):
        k = np.full(weight_array.shape[0], float(intended_k))
    else:
        k = np.asarray(intended_k, dtype=float)
        if k.shape != (weight_array.shape[0],):
            raise ValueError("intended_k must be scalar or have one value per individual")
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("intended_k values must be finite and positive")

    weight_sum = weight_array.sum(axis=1)
    squared_sum = np.square(weight_array).sum(axis=1)
    ess = np.divide(
        np.square(weight_sum),
        squared_sum,
        out=np.zeros_like(weight_sum),
        where=squared_sum > 0,
    )
    balance = np.minimum(1.0, ess / k)
    strength = weight_array.mean(axis=1)
    local_coverage = balance * strength

    return CoverageResult(
        score=float(local_coverage.mean()),
        balance_by_individual=balance,
        strength_by_individual=strength,
        local_coverage=local_coverage,
    )


def distance_stability(
    weights: ArrayLike,
    similarities: ArrayLike,
) -> tuple[float, FloatArray]:
    """Calculate D and pair reliabilities.

    ``similarities`` has shape ``(definition, individual, neighbour)`` and
    uses the declared primary population-standard-deviation definition.
    """

    weight_array = _as_finite_2d(weights, name="weights")
    _validate_unit_interval(weight_array, name="weights")
    similarity_array = np.asarray(similarities, dtype=float)
    expected_tail = weight_array.shape
    if similarity_array.ndim != 3 or similarity_array.shape[1:] != expected_tail:
        raise ValueError(
            "similarities must have shape (definition, individual, neighbour) "
            "matching weights"
        )
    if similarity_array.shape[0] == 0 or not np.all(np.isfinite(similarity_array)):
        raise ValueError("similarities must include finite values for at least one definition")
    _validate_unit_interval(similarity_array, name="similarities")

    reliability = np.clip(1.0 - 2.0 * similarity_array.std(axis=0, ddof=0), 0.0, 1.0)
    total_weight = float(weight_array.sum())
    if total_weight == 0.0:
        return 0.0, reliability
    score = float(np.sum(weight_array * reliability) / total_weight)
    return score, reliability


def weighted_prediction_consistency(
    probabilities_by_model: Mapping[str, ArrayLike],
    neighbour_indices: ArrayLike,
    weights: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> tuple[dict[str, float], dict[str, FloatArray]]:
    """Calculate weighted local prediction consistency.

    This is the reference individual-fairness metric selected for the initial
    IF-CRED study, not a fixed requirement of the validation framework.
    Probabilities contain one value per audited individual. Neighbour indices
    and weights have shape ``(individual, neighbour)``.
    """

    weight_array = _as_finite_2d(weights, name="weights")
    _validate_unit_interval(weight_array, name="weights")
    indices = np.asarray(neighbour_indices)
    if indices.shape != weight_array.shape or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("neighbour_indices must be an integer array matching weights")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if not probabilities_by_model:
        raise ValueError("probabilities_by_model must contain at least one model")

    n_individuals = weight_array.shape[0]
    if np.any((indices < 0) | (indices >= n_individuals)):
        raise ValueError("neighbour_indices contain an out-of-range individual index")

    weight_sum = weight_array.sum(axis=1)
    scores: dict[str, float] = {}
    local_scores: dict[str, FloatArray] = {}
    for model_name, probabilities in probabilities_by_model.items():
        probability_array = np.asarray(probabilities, dtype=float)
        if probability_array.shape != (n_individuals,):
            raise ValueError(f"probabilities for {model_name!r} must have one value per individual")
        if not np.all(np.isfinite(probability_array)):
            raise ValueError(f"probabilities for {model_name!r} must be finite")
        _validate_unit_interval(probability_array, name=f"probabilities for {model_name!r}")

        differences = np.abs(probability_array[:, None] - probability_array[indices])
        weighted_difference = np.sum(weight_array * differences, axis=1)
        local = np.clip(1.0 - weighted_difference / (weight_sum + epsilon), 0.0, 1.0)
        local_scores[model_name] = local
        scores[model_name] = float(local.mean())

    return scores, local_scores


def model_stability(fairness_by_model: Mapping[str, float] | ArrayLike) -> float:
    """Calculate M using population SD across the declared model family."""

    if isinstance(fairness_by_model, Mapping):
        values = np.asarray(list(fairness_by_model.values()), dtype=float)
    else:
        values = np.asarray(fairness_by_model, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("fairness values must be a non-empty, finite one-dimensional collection")
    _validate_unit_interval(values, name="fairness values")
    return float(max(0.0, 1.0 - 2.0 * values.std(ddof=0)))
