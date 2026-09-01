import numpy as np
import pytest

from ifcred.core.components import coverage, distance_stability
from ifcred.core.graph import (
    AuditGraphSpec,
    BandwidthPolicy,
    DistanceMetricSpec,
    WeightingRule,
    build_audit_graph,
)
from ifcred.metrics import (
    WeightedPredictionConsistency,
    WeightedPredictionConsistencyInputs,
)


def graph_spec(k=2, *, pair_batch_size=100_000):
    return AuditGraphSpec(
        k=k,
        metrics=(
            DistanceMetricSpec("euclidean", "euclidean"),
            DistanceMetricSpec("manhattan", "manhattan"),
            DistanceMetricSpec("cosine", "cosine"),
        ),
        primary_metric="euclidean",
        bandwidth_policy=BandwidthPolicy.MEDIAN_RETAINED_MAX,
        weighting_rule=WeightingRule.GAUSSIAN,
        pair_batch_size=pair_batch_size,
    )


def test_graph_has_fixed_k_unique_nonself_neighbours():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 1.0]])
    graph = build_audit_graph(X, graph_spec())

    assert graph.neighbour_indices.shape == (4, 2)
    for row, neighbours in enumerate(graph.neighbour_indices):
        assert row not in neighbours
        assert len(np.unique(neighbours)) == 2
    assert graph.weights.shape == (4, 2)
    assert np.all((graph.weights >= 0.0) & (graph.weights <= 1.0))


def test_alternative_distances_are_computed_on_retained_pairs():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 1.0], [6.0, 2.0]])
    graph = build_audit_graph(X, graph_spec(k=1))

    for row, neighbour in enumerate(graph.neighbour_indices[:, 0]):
        difference = X[row] - X[neighbour]
        assert graph.distances["euclidean"][row, 0] == pytest.approx(
            np.linalg.norm(difference)
        )
        assert graph.distances["manhattan"][row, 0] == pytest.approx(
            np.abs(difference).sum()
        )


def test_similarity_tensor_connects_directly_to_c_and_d():
    rng = np.random.default_rng(4)
    graph = build_audit_graph(rng.normal(size=(20, 3)), graph_spec(k=3))

    C = coverage(graph.weights, intended_k=graph.spec.k)
    D, reliability = distance_stability(graph.weights, graph.similarity_tensor())

    assert 0.0 <= C.score <= 1.0
    assert 0.0 <= D <= 1.0
    assert reliability.shape == graph.weights.shape


def test_graph_connects_to_the_current_selectable_f_metric():
    rng = np.random.default_rng(12)
    graph = build_audit_graph(rng.normal(size=(16, 4)), graph_spec(k=3))
    probabilities = {
        "linear": np.linspace(0.1, 0.9, 16),
        "tree": np.linspace(0.9, 0.1, 16),
    }

    evaluation = WeightedPredictionConsistency().evaluate(
        WeightedPredictionConsistencyInputs(
            probabilities_by_model=probabilities,
            neighbour_indices=graph.neighbour_indices,
            weights=graph.weights,
        )
    )

    assert evaluation.metric_name == "weighted_prediction_consistency"
    assert set(evaluation.model_scores) == set(probabilities)
    assert all(0.0 <= score <= 1.0 for score in evaluation.model_scores.values())
    assert all(local.shape == (16,) for local in evaluation.local_scores.values())


def test_graph_construction_is_reproducible_with_ties_and_duplicates():
    X = np.array([[0.0], [0.0], [1.0], [1.0], [2.0]])
    first = build_audit_graph(X, graph_spec(k=2))
    second = build_audit_graph(X, graph_spec(k=2))

    np.testing.assert_array_equal(first.neighbour_indices, second.neighbour_indices)
    np.testing.assert_allclose(first.weights, second.weights)


def test_custom_row_ids_are_retained_for_pair_provenance():
    X = np.arange(10, dtype=float).reshape(5, 2)
    row_ids = np.array(["real-1", "real-2", "syn-a", "real-3", "syn-b"])
    graph = build_audit_graph(X, graph_spec(k=1), row_ids=row_ids)

    np.testing.assert_array_equal(graph.row_ids, row_ids)
    assert graph.neighbour_row_ids.shape == (5, 1)


def test_pair_distance_calculation_respects_small_batches():
    calls = 0

    def counted_distance(left, right):
        nonlocal calls
        calls += 1
        return float(np.linalg.norm(left - right))

    spec = AuditGraphSpec(
        k=2,
        metrics=(
            DistanceMetricSpec("euclidean", "euclidean"),
            DistanceMetricSpec("counted", counted_distance),
        ),
        primary_metric="euclidean",
        bandwidth_policy=BandwidthPolicy.MEDIAN_POSITIVE_PAIR,
        pair_batch_size=3,
    )
    graph = build_audit_graph(np.arange(18, dtype=float).reshape(9, 2), spec)

    assert calls == 9 * 2
    assert graph.distances["counted"].shape == (9, 2)


def test_fixed_bandwidths_are_explicit_and_used():
    spec = AuditGraphSpec(
        k=1,
        metrics=(DistanceMetricSpec("euclidean", "euclidean"),),
        primary_metric="euclidean",
        bandwidth_policy=BandwidthPolicy.FIXED,
        fixed_bandwidths={"euclidean": 2.0},
    )
    graph = build_audit_graph(np.array([[0.0], [1.0], [4.0]]), spec)

    assert graph.bandwidths["euclidean"] == 2.0
    expected = np.exp(-np.square(graph.distances["euclidean"]) / 8.0)
    np.testing.assert_allclose(graph.weights, expected)


def test_invalid_graph_requests_are_rejected():
    with pytest.raises(ValueError, match="more rows than k"):
        build_audit_graph(np.ones((2, 2)), graph_spec(k=2))
