import inspect
import json

import numpy as np
import pytest

from ifcred.core.graph import (
    AuditGraphSpec,
    BandwidthPolicy,
    DistanceMetricSpec,
)
from ifcred.experiments import fit_similarity_calibration


def fitting_spec():
    return AuditGraphSpec(
        k=2,
        metrics=(
            DistanceMetricSpec("euclidean", "euclidean"),
            DistanceMetricSpec("manhattan", "manhattan"),
        ),
        primary_metric="euclidean",
        bandwidth_policy=BandwidthPolicy.MEDIAN_RETAINED_MAX,
        pair_batch_size=3,
    )


def test_calibration_fits_each_metric_from_training_features():
    X_train = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [4.0, 1.0], [5.0, 2.0]]
    )
    calibration = fit_similarity_calibration(X_train, fitting_spec())

    assert set(calibration.bandwidths) == {"euclidean", "manhattan"}
    assert all(value > 0.0 for value in calibration.bandwidths.values())
    assert calibration.n_training_rows == 5
    assert calibration.n_features == 2
    assert len(calibration.training_matrix_sha256) == 64
    assert "y_train" not in inspect.signature(fit_similarity_calibration).parameters


def test_evaluation_graph_uses_frozen_training_bandwidths():
    rng = np.random.default_rng(8)
    calibration = fit_similarity_calibration(rng.normal(size=(30, 3)), fitting_spec())
    clean_test = rng.normal(size=(12, 3))
    injected_test = np.vstack([clean_test, np.full((1, 3), 100.0)])

    clean_graph = calibration.build_evaluation_graph(clean_test)
    injected_graph = calibration.build_evaluation_graph(injected_test)

    assert clean_graph.bandwidths == calibration.bandwidths
    assert injected_graph.bandwidths == calibration.bandwidths
    assert clean_graph.spec.bandwidth_policy == BandwidthPolicy.FIXED
    assert injected_graph.spec.bandwidth_policy == BandwidthPolicy.FIXED


def test_pair_evaluator_uses_frozen_bandwidths():
    calibration = fit_similarity_calibration(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]]),
        fitting_spec(),
    )
    left = np.array([[0.0, 0.0], [1.0, 0.0]])
    right = np.array([[1.0, 0.0], [1.0, 2.0]])
    similarities = calibration.evaluate_pairs(left, right)

    assert set(similarities) == {"euclidean", "manhattan"}
    for values in similarities.values():
        assert values.shape == (2,)
        assert np.all((values >= 0.0) & (values <= 1.0))


def test_frozen_similarity_matches_declared_gaussian_conversion():
    X_train = np.arange(18, dtype=float).reshape(9, 2)
    X_test = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 1.0], [6.0, 2.0]])
    calibration = fit_similarity_calibration(X_train, fitting_spec())
    graph = calibration.build_evaluation_graph(X_test)

    for name, distances in graph.distances.items():
        bandwidth = calibration.bandwidths[name]
        expected = np.exp(-np.square(distances) / (2.0 * bandwidth**2))
        np.testing.assert_allclose(graph.similarities[name], expected)


def test_calibration_is_reproducible_and_manifest_is_serializable():
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(20, 4))
    first = fit_similarity_calibration(X_train, fitting_spec())
    second = fit_similarity_calibration(X_train.copy(), fitting_spec())

    assert first.bandwidths == second.bandwidths
    assert first.training_matrix_sha256 == second.training_matrix_sha256
    json.dumps(first.to_manifest())


def test_evaluation_must_reuse_calibrated_feature_columns():
    calibration = fit_similarity_calibration(np.ones((8, 2)), fitting_spec())
    with pytest.raises(ValueError, match="same feature columns"):
        calibration.build_evaluation_graph(np.ones((8, 3)))


def test_fixed_bandwidth_spec_cannot_be_refitted():
    adaptive = fitting_spec()
    fixed = AuditGraphSpec(
        k=adaptive.k,
        metrics=adaptive.metrics,
        primary_metric=adaptive.primary_metric,
        bandwidth_policy=BandwidthPolicy.FIXED,
        fixed_bandwidths={"euclidean": 1.0, "manhattan": 1.0},
    )
    with pytest.raises(ValueError, match="cannot fit"):
        fit_similarity_calibration(np.ones((8, 2)), fixed)
