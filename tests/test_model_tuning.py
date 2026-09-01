import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ifcred.models import (
    CalibrationSpec,
    CrossValidationSpec,
    ModelFamily,
    ModelSearchSpace,
    ModelSpec,
    run_model_family,
    stratified_kfold_indices,
    tune_model_family,
)


def small_search_spaces():
    return (
        ModelSearchSpace(
            ModelSpec(ModelFamily.LOGISTIC_REGRESSION, {"max_iter": 250}),
            {"C": (0.5, 1.0)},
        ),
        ModelSearchSpace(
            ModelSpec(
                ModelFamily.MLP,
                {"max_iter": 60, "early_stopping": True},
            ),
            {"hidden_layer_sizes": ((6,),)},
        ),
        ModelSearchSpace(
            ModelSpec(ModelFamily.GAUSSIAN_NAIVE_BAYES),
            {"var_smoothing": (1e-9,)},
        ),
        ModelSearchSpace(
            ModelSpec(
                ModelFamily.RANDOM_FOREST,
                {"n_estimators": 10},
            ),
            {"max_depth": (2,)},
        ),
        ModelSearchSpace(
            ModelSpec(ModelFamily.DECISION_TREE),
            {"max_depth": (2, 4)},
        ),
    )


@pytest.fixture(scope="module")
def tuning_data():
    X, y = make_classification(
        n_samples=150,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        random_state=7,
    )
    return train_test_split(X, y, test_size=40, random_state=13, stratify=y)


@pytest.fixture(scope="module")
def tuning_run(tuning_data):
    X_train, _, y_train, _ = tuning_data
    return tune_model_family(
        X_train,
        y_train,
        search_spaces=small_search_spaces(),
        cross_validation=CrossValidationSpec(folds=3),
        random_state=29,
        n_jobs=2,
    )


def test_stratified_kfold_partitions_training_rows_once(tuning_data):
    _, _, y_train, _ = tuning_data
    folds = stratified_kfold_indices(
        y_train,
        cross_validation=CrossValidationSpec(folds=5),
        random_state=17,
    )

    validation_rows = np.concatenate([validation for _, validation in folds])
    np.testing.assert_array_equal(np.sort(validation_rows), np.arange(len(y_train)))
    assert len(np.unique(validation_rows)) == len(y_train)
    for train, validation in folds:
        assert not set(train) & set(validation)
        assert set(np.unique(y_train[train])) == {0, 1}
        assert set(np.unique(y_train[validation])) == {0, 1}


def test_tuning_selects_one_frozen_spec_for_each_model(tuning_run):
    assert set(tuning_run.results) == {
        "logistic_regression",
        "mlp",
        "gaussian_naive_bayes",
        "random_forest",
        "decision_tree",
    }
    assert len(tuning_run.selected_model_specs) == 5
    assert len(tuning_run.results["logistic_regression"].candidates) == 2
    assert len(tuning_run.results["decision_tree"].candidates) == 2
    for result in tuning_run.results.values():
        assert np.isfinite(result.best_mean_validation_score)
        assert min(candidate.rank for candidate in result.candidates) == 1
        assert all(len(candidate.fold_validation_scores) == 3 for candidate in result.candidates)


def test_selected_specs_feed_existing_calibrated_runner(tuning_data, tuning_run):
    X_train, X_test, y_train, y_test = tuning_data
    model_run = run_model_family(
        X_train,
        y_train,
        X_test,
        y_test,
        model_specs=tuning_run.selected_model_specs,
        calibration=CalibrationSpec(folds=2),
        random_state=29,
        n_jobs=2,
    )

    assert set(model_run.models) == set(tuning_run.results)
    assert all(
        prediction.calibrated_probabilities.shape == (len(X_test),)
        for prediction in model_run.models.values()
    )


def test_tuning_is_reproducible_from_training_data_only(tuning_data):
    X_train, _, y_train, _ = tuning_data
    space = (
        ModelSearchSpace(
            ModelSpec(ModelFamily.DECISION_TREE),
            {"max_depth": (1, 2, 4), "min_samples_leaf": (1, 3)},
        ),
    )
    kwargs = dict(
        search_spaces=space,
        cross_validation=CrossValidationSpec(folds=4),
        random_state=41,
        require_declared_family=False,
    )
    first = tune_model_family(X_train, y_train, **kwargs)
    second = tune_model_family(X_train, y_train, **kwargs)

    assert (
        first.results["decision_tree"].selected_spec.to_manifest()
        == second.results["decision_tree"].selected_spec.to_manifest()
    )
    assert (
        first.results["decision_tree"].best_mean_validation_score
        == second.results["decision_tree"].best_mean_validation_score
    )


def test_tuning_manifest_contains_every_fold_score(tuning_run):
    manifest = tuning_run.to_manifest()

    assert manifest["cross_validation"]["splitter"] == "StratifiedKFold"
    assert manifest["cross_validation"]["folds"] == 3
    for model in manifest["models"].values():
        assert all(
            len(candidate["fold_validation_scores"]) == 3
            for candidate in model["candidates"]
        )
    json.dumps(manifest)


def test_search_space_rejects_runner_controlled_parameters():
    with pytest.raises(ValueError, match="controlled"):
        ModelSearchSpace(
            ModelSpec(ModelFamily.RANDOM_FOREST),
            {"n_jobs": (1, 2)},
        )


def test_kfold_rejects_insufficient_minority_class_support():
    with pytest.raises(ValueError, match="support every cross-validation fold"):
        stratified_kfold_indices(
            np.array([0, 0, 0, 1, 1]),
            cross_validation=CrossValidationSpec(folds=3),
            random_state=1,
        )
