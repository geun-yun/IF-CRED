import numpy as np
import pytest

from ifcred.core.components import (
    coverage,
    distance_stability,
    model_stability,
    weighted_prediction_consistency,
)
from ifcred.core.composite import composite


def test_composite_is_product_and_cannot_exceed_a_factor():
    factors = {"C": 0.8, "D": 0.9, "F": 0.7, "M": 0.95}
    score = composite(**factors)

    assert score == pytest.approx(np.prod(list(factors.values())))
    assert all(score <= factor for factor in factors.values())


def test_uniform_strong_weights_have_full_coverage():
    result = coverage(np.ones((3, 4)), intended_k=4)

    assert result.score == pytest.approx(1.0)
    np.testing.assert_allclose(result.balance_by_individual, 1.0)
    np.testing.assert_allclose(result.strength_by_individual, 1.0)


def test_concentrating_fixed_weight_reduces_ess_balance():
    uniform = coverage([[0.25, 0.25, 0.25, 0.25]], intended_k=4)
    concentrated = coverage([[1.0, 0.0, 0.0, 0.0]], intended_k=4)

    assert concentrated.balance_by_individual[0] < uniform.balance_by_individual[0]


def test_zero_evidence_has_zero_coverage():
    result = coverage(np.zeros((2, 3)), intended_k=3)

    assert result.score == 0.0
    np.testing.assert_allclose(result.local_coverage, 0.0)


def test_identical_similarity_definitions_have_perfect_stability():
    similarities = np.full((3, 2, 2), 0.75)
    score, reliability = distance_stability(np.ones((2, 2)), similarities)

    assert score == pytest.approx(1.0)
    np.testing.assert_allclose(reliability, 1.0)


def test_more_similarity_dispersion_reduces_reliability():
    weights = [[1.0]]
    low_dispersion = np.array([[[0.4]], [[0.5]], [[0.6]]])
    high_dispersion = np.array([[[0.0]], [[0.5]], [[1.0]]])

    _, low_reliability = distance_stability(weights, low_dispersion)
    _, high_reliability = distance_stability(weights, high_dispersion)

    assert high_reliability[0, 0] < low_reliability[0, 0]


def test_equal_local_probabilities_give_perfect_fairness():
    probabilities = {"constant": np.full(3, 0.4)}
    neighbours = np.array([[1, 2], [0, 2], [0, 1]])
    weights = np.ones((3, 2))

    model_scores, local_scores = weighted_prediction_consistency(
        probabilities, neighbours, weights
    )

    assert model_scores["constant"] == pytest.approx(1.0)
    np.testing.assert_allclose(local_scores["constant"], 1.0)


def test_larger_prediction_differences_cannot_increase_fairness():
    neighbours = np.array([[1], [0]])
    weights = np.ones((2, 1))
    small, _ = weighted_prediction_consistency({"model": [0.4, 0.6]}, neighbours, weights)
    large, _ = weighted_prediction_consistency({"model": [0.1, 0.9]}, neighbours, weights)

    assert large["model"] < small["model"]


def test_equal_model_fairness_has_perfect_model_stability():
    assert model_stability({"a": 0.8, "b": 0.8, "c": 0.8}) == pytest.approx(1.0)


def test_more_model_dispersion_reduces_model_stability():
    assert model_stability([0.2, 0.8]) < model_stability([0.45, 0.55])


@pytest.mark.parametrize(
    "calculation",
    [
        lambda: coverage([[1.1]], intended_k=1),
        lambda: distance_stability([[1.0]], [[[1.2]]]),
        lambda: model_stability([1.1]),
        lambda: composite(C=-0.1, D=1.0, F=1.0, M=1.0),
    ],
)
def test_out_of_range_inputs_are_rejected(calculation):
    with pytest.raises(ValueError):
        calculation()
