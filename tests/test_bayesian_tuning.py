import json

import numpy as np
import pytest
from sklearn.datasets import make_classification

from ifcred.models import (
    BayesianOptimizationSpec,
    BayesianSearchSpace,
    CategoricalDistribution,
    CrossValidationSpec,
    FloatDistribution,
    IntDistribution,
    ModelFamily,
    ModelSpec,
    recommended_bayesian_search_spaces,
    tune_model_family_bayesian,
)


def tiny_bayesian_spaces():
    return (
        BayesianSearchSpace(
            ModelSpec(
                ModelFamily.LOGISTIC_REGRESSION,
                {"solver": "liblinear", "max_iter": 200},
            ),
            {"C": FloatDistribution(0.1, 2.0, log=True)},
            n_trials=2,
        ),
        BayesianSearchSpace(
            ModelSpec(
                ModelFamily.MLP,
                {"max_iter": 50, "early_stopping": True},
            ),
            {
                "hidden_layer_sizes": CategoricalDistribution(
                    {"four": (4,), "six": (6,)}
                ),
                "alpha": FloatDistribution(1e-5, 1e-2, log=True),
            },
            n_trials=2,
        ),
        BayesianSearchSpace(
            ModelSpec(ModelFamily.GAUSSIAN_NAIVE_BAYES),
            {
                "var_smoothing": FloatDistribution(1e-12, 1e-7, log=True),
            },
            n_trials=2,
        ),
        BayesianSearchSpace(
            ModelSpec(
                ModelFamily.RANDOM_FOREST,
                {"n_estimators": 10},
            ),
            {
                "max_depth": IntDistribution(1, 3),
            },
            n_trials=2,
        ),
        BayesianSearchSpace(
            ModelSpec(ModelFamily.DECISION_TREE),
            {
                "max_depth": IntDistribution(1, 5),
                "criterion": CategoricalDistribution(
                    {"gini": "gini", "entropy": "entropy"}
                ),
            },
            n_trials=2,
        ),
    )


@pytest.fixture(scope="module")
def bayesian_data():
    return make_classification(
        n_samples=110,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        random_state=9,
    )


@pytest.fixture(scope="module")
def bayesian_run(bayesian_data):
    X, y = bayesian_data
    return tune_model_family_bayesian(
        X,
        y,
        search_spaces=tiny_bayesian_spaces(),
        optimization=BayesianOptimizationSpec(
            cross_validation=CrossValidationSpec(folds=3),
            startup_trials=1,
        ),
        random_state=43,
    )


def test_recommended_spaces_cover_declared_family_with_sensible_budgets():
    spaces = recommended_bayesian_search_spaces()

    assert {space.model.family for space in spaces} == set(ModelFamily)
    assert {space.model.family: space.n_trials for space in spaces} == {
        ModelFamily.LOGISTIC_REGRESSION: 30,
        ModelFamily.MLP: 60,
        ModelFamily.GAUSSIAN_NAIVE_BAYES: 20,
        ModelFamily.RANDOM_FOREST: 50,
        ModelFamily.DECISION_TREE: 40,
    }
    assert all(space.distributions for space in spaces)


def test_bayesian_cv_returns_every_trial_and_fold(bayesian_run):
    assert set(bayesian_run.results) == {family.value for family in ModelFamily}
    assert len(bayesian_run.selected_model_specs) == 5
    for result in bayesian_run.results.values():
        assert len(result.trials) == 2
        assert all(len(trial.fold_validation_scores) == 3 for trial in result.trials)
        assert result.best_mean_validation_score == max(
            trial.mean_validation_score for trial in result.trials
        )
        assert result.selected_spec.name == result.model_name


def test_bayesian_tuning_is_seed_reproducible(bayesian_data):
    X, y = bayesian_data
    spaces = (
        BayesianSearchSpace(
            ModelSpec(ModelFamily.DECISION_TREE),
            {
                "max_depth": IntDistribution(1, 8),
                "min_samples_leaf": IntDistribution(1, 10),
                "ccp_alpha": FloatDistribution(1e-8, 1e-2, log=True),
            },
            n_trials=5,
        ),
    )
    kwargs = dict(
        search_spaces=spaces,
        optimization=BayesianOptimizationSpec(
            cross_validation=CrossValidationSpec(folds=3),
            startup_trials=2,
        ),
        random_state=47,
        require_declared_family=False,
    )
    first = tune_model_family_bayesian(X, y, **kwargs)
    second = tune_model_family_bayesian(X, y, **kwargs)

    first_result = first.results["decision_tree"]
    second_result = second.results["decision_tree"]
    assert first_result.selected_spec.to_manifest() == second_result.selected_spec.to_manifest()
    assert [trial.mean_validation_score for trial in first_result.trials] == [
        trial.mean_validation_score for trial in second_result.trials
    ]


def test_bayesian_manifest_records_algorithm_spaces_and_trials(bayesian_run):
    manifest = bayesian_run.to_manifest()

    assert manifest["optimization"]["algorithm"] == "Optuna TPESampler"
    assert manifest["optimization"]["cross_validation"]["scoring"] == "roc_auc"
    assert len(manifest["search_spaces"]) == 5
    assert all(len(model["trials"]) == 2 for model in manifest["models"].values())
    json.dumps(manifest)


def test_categorical_labels_decode_to_estimator_values(bayesian_run):
    selected = bayesian_run.results["mlp"].selected_spec.hyperparameters
    assert selected["hidden_layer_sizes"] in {(4,), (6,)}


def test_bayesian_space_rejects_controlled_parameters():
    with pytest.raises(ValueError, match="controlled"):
        BayesianSearchSpace(
            ModelSpec(ModelFamily.RANDOM_FOREST),
            {"n_jobs": CategoricalDistribution({"one": 1, "two": 2})},
            n_trials=2,
        )


def test_bayesian_cross_validation_supports_multiple_thread_workers(bayesian_data):
    X, y = bayesian_data
    run = tune_model_family_bayesian(
        X,
        y,
        search_spaces=(
            BayesianSearchSpace(
                ModelSpec(ModelFamily.DECISION_TREE),
                {"max_depth": IntDistribution(1, 3)},
                n_trials=1,
            ),
        ),
        optimization=BayesianOptimizationSpec(
            cross_validation=CrossValidationSpec(folds=2),
            startup_trials=1,
        ),
        random_state=53,
        n_jobs=2,
        require_declared_family=False,
    )

    assert len(run.results["decision_tree"].trials) == 1


def test_invalid_distributions_are_rejected():
    with pytest.raises(ValueError, match="low < high"):
        FloatDistribution(1.0, 1.0)
    with pytest.raises(ValueError, match="low > 0"):
        FloatDistribution(0.0, 1.0, log=True)
    with pytest.raises(ValueError, match="low < high"):
        IntDistribution(2, 2)
    with pytest.raises(ValueError, match="non-empty labels"):
        CategoricalDistribution({})
