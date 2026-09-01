import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ifcred.core.graph import (
    AuditGraphSpec,
    BandwidthPolicy,
    DistanceMetricSpec,
    build_audit_graph,
)
from ifcred.metrics import (
    WeightedPredictionConsistency,
    WeightedPredictionConsistencyInputs,
)
from ifcred.models import (
    CalibrationSpec,
    DECLARED_MODEL_FAMILIES,
    ModelFamily,
    ModelSpec,
    run_model_family,
    validate_declared_family,
)


def tiny_declared_specs():
    return (
        ModelSpec(
            ModelFamily.LOGISTIC_REGRESSION,
            {"max_iter": 300, "solver": "liblinear"},
        ),
        ModelSpec(
            ModelFamily.MLP,
            {"hidden_layer_sizes": (8,), "max_iter": 80, "early_stopping": True},
        ),
        ModelSpec(ModelFamily.GAUSSIAN_NAIVE_BAYES, {"var_smoothing": 1e-9}),
        ModelSpec(
            ModelFamily.RANDOM_FOREST,
            {"n_estimators": 12, "max_depth": 3},
        ),
        ModelSpec(ModelFamily.DECISION_TREE, {"max_depth": 3}),
    )


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=140,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=5,
    )
    return train_test_split(X, y, test_size=40, random_state=11, stratify=y)


@pytest.fixture(scope="module")
def family_run(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    return run_model_family(
        X_train,
        y_train,
        X_test,
        y_test,
        model_specs=tiny_declared_specs(),
        calibration=CalibrationSpec(folds=2),
        random_state=19,
        calibration_bins=5,
    )


def test_runner_fits_exactly_the_five_declared_models(family_run):
    assert set(family_run.models) == {family.value for family in DECLARED_MODEL_FAMILIES}
    assert family_run.n_train == 100
    assert family_run.n_evaluation == 40
    assert family_run.n_features == 8
    for result in family_run.models.values():
        assert result.native_probabilities.shape == (40,)
        assert result.calibrated_probabilities.shape == (40,)
        assert np.all((result.calibrated_probabilities >= 0.0) & (result.calibrated_probabilities <= 1.0))
        for utility in (result.native_utility, result.calibrated_utility):
            assert 0.0 <= utility.accuracy <= 1.0
            assert 0.0 <= utility.roc_auc <= 1.0
            assert 0.0 <= utility.brier_score <= 1.0
            assert 0.0 <= utility.expected_calibration_error <= 1.0


def test_evaluation_outcomes_cannot_change_predictions(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    kwargs = dict(
        model_specs=(tiny_declared_specs()[0],),
        calibration=CalibrationSpec(folds=2),
        random_state=23,
        require_declared_family=False,
    )
    original = run_model_family(X_train, y_train, X_test, y_test, **kwargs)
    flipped = run_model_family(X_train, y_train, X_test, 1 - y_test, **kwargs)

    np.testing.assert_allclose(
        original.models["logistic_regression"].calibrated_probabilities,
        flipped.models["logistic_regression"].calibrated_probabilities,
    )
    np.testing.assert_allclose(
        original.models["logistic_regression"].native_probabilities,
        flipped.models["logistic_regression"].native_probabilities,
    )


def test_same_seed_reproduces_scientific_predictions(binary_data, family_run):
    X_train, X_test, y_train, y_test = binary_data
    repeated = run_model_family(
        X_train,
        y_train,
        X_test,
        y_test,
        model_specs=tiny_declared_specs(),
        calibration=CalibrationSpec(folds=2),
        random_state=19,
        calibration_bins=5,
    )

    for name in family_run.models:
        np.testing.assert_allclose(
            family_run.models[name].native_probabilities,
            repeated.models[name].native_probabilities,
        )
        np.testing.assert_allclose(
            family_run.models[name].calibrated_probabilities,
            repeated.models[name].calibrated_probabilities,
        )


def test_calibrated_predictions_feed_the_selected_f_metric(binary_data, family_run):
    _, X_test, _, _ = binary_data
    graph = build_audit_graph(
        X_test,
        AuditGraphSpec(
            k=3,
            metrics=(DistanceMetricSpec("euclidean", "euclidean"),),
            primary_metric="euclidean",
            bandwidth_policy=BandwidthPolicy.MEDIAN_RETAINED_MAX,
        ),
    )
    evaluation = WeightedPredictionConsistency().evaluate(
        WeightedPredictionConsistencyInputs(
            probabilities_by_model=family_run.calibrated_probabilities_by_model,
            neighbour_indices=graph.neighbour_indices,
            weights=graph.weights,
        )
    )

    assert set(evaluation.model_scores) == set(family_run.models)
    assert all(0.0 <= score <= 1.0 for score in evaluation.model_scores.values())


def test_manifest_records_resolved_model_and_calibration_settings(family_run):
    manifest = family_run.to_manifest()

    assert manifest["calibration"] == {
        "method": "sigmoid",
        "folds": 2,
        "ensemble": True,
    }
    assert manifest["calibration_bins"] == 5
    assert manifest["models"]["gaussian_naive_bayes"]["resolved_estimator_parameters"]["var_smoothing"] == 1e-9
    assert manifest["models"]["random_forest"]["resolved_estimator_parameters"]["n_estimators"] == 12
    json.dumps(manifest)


def test_primary_family_requires_every_declared_model_exactly_once():
    with pytest.raises(ValueError, match="exactly the five"):
        validate_declared_family((tiny_declared_specs()[0],))
    with pytest.raises(ValueError, match="must not repeat"):
        validate_declared_family(
            (tiny_declared_specs()[0], tiny_declared_specs()[0])
        )


def test_runner_rejects_too_few_training_examples_for_calibration(binary_data):
    X_train, X_test, y_train, y_test = binary_data
    with pytest.raises(ValueError, match="support every calibration fold"):
        run_model_family(
            X_train[:6],
            y_train[:6],
            X_test,
            y_test,
            model_specs=(tiny_declared_specs()[0],),
            calibration=CalibrationSpec(folds=4),
            random_state=2,
            require_declared_family=False,
        )
