"""Seeded Bayesian hyperparameter optimization over training-only CV."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np
import optuna
from numpy.typing import ArrayLike
from sklearn.model_selection import cross_val_score

from ifcred.models.registry import ModelFamily, ModelSpec, build_classifier, validate_declared_family
from ifcred.models.tuning import CrossValidationSpec, stratified_kfold_indices
from ifcred.parallel import sklearn_parallelism


class ParameterDistribution(Protocol):
    """One serializable hyperparameter distribution."""

    def suggest(self, trial: optuna.Trial, name: str) -> Any: ...

    def to_manifest(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FloatDistribution:
    low: float
    high: float
    log: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high) or self.low >= self.high:
            raise ValueError("float distribution requires finite low < high")
        if self.log and self.low <= 0:
            raise ValueError("a log float distribution requires low > 0")
        if self.log and self.step is not None:
            raise ValueError("a log float distribution cannot also use a step")
        if self.step is not None and (not np.isfinite(self.step) or self.step <= 0):
            raise ValueError("float distribution step must be finite and positive")

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        return float(
            trial.suggest_float(name, self.low, self.high, log=self.log, step=self.step)
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "type": "float",
            "low": self.low,
            "high": self.high,
            "log": self.log,
            "step": self.step,
        }


@dataclass(frozen=True)
class IntDistribution:
    low: int
    high: int
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError("integer distribution requires low < high")
        if self.step < 1:
            raise ValueError("integer distribution step must be positive")
        if self.log and self.low <= 0:
            raise ValueError("a log integer distribution requires low > 0")
        if self.log and self.step != 1:
            raise ValueError("a log integer distribution must use step=1")

    def suggest(self, trial: optuna.Trial, name: str) -> int:
        return int(
            trial.suggest_int(name, self.low, self.high, log=self.log, step=self.step)
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "type": "integer",
            "low": self.low,
            "high": self.high,
            "log": self.log,
            "step": self.step,
        }


@dataclass(frozen=True)
class CategoricalDistribution:
    """Label-to-value choices; labels remain stable in Optuna storage."""

    choices: Mapping[str, Any]

    def __post_init__(self) -> None:
        choices = dict(self.choices)
        if not choices or any(not str(label).strip() for label in choices):
            raise ValueError("categorical distribution requires non-empty labels")
        object.__setattr__(self, "choices", MappingProxyType(choices))

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        label = trial.suggest_categorical(name, tuple(self.choices))
        return self.choices[str(label)]

    def to_manifest(self) -> dict[str, Any]:
        return {"type": "categorical", "choices": dict(self.choices)}


@dataclass(frozen=True)
class BayesianSearchSpace:
    """One model, explicit parameter distributions, and fixed trial budget."""

    model: ModelSpec
    distributions: Mapping[str, ParameterDistribution]
    n_trials: int

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("Bayesian trial budget must be positive")
        distributions = dict(self.distributions)
        if not distributions:
            raise ValueError("Bayesian search requires at least one distribution")
        if any(not name.strip() for name in distributions):
            raise ValueError("hyperparameter names must be non-empty")
        # Probe values validate centrally controlled names and estimator kwargs.
        # The actual estimator validates value compatibility during each trial.
        controlled_probe = dict(self.model.hyperparameters)
        for name, distribution in distributions.items():
            if isinstance(distribution, FloatDistribution):
                controlled_probe[name] = distribution.low
            elif isinstance(distribution, IntDistribution):
                controlled_probe[name] = distribution.low
            elif isinstance(distribution, CategoricalDistribution):
                controlled_probe[name] = next(iter(distribution.choices.values()))
            else:
                raise TypeError(f"unsupported distribution for {name!r}")
        ModelSpec(self.model.family, controlled_probe)
        object.__setattr__(self, "distributions", MappingProxyType(distributions))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model": self.model.to_manifest(),
            "n_trials": self.n_trials,
            "distributions": {
                name: distribution.to_manifest()
                for name, distribution in self.distributions.items()
            },
        }


@dataclass(frozen=True)
class BayesianOptimizationSpec:
    """Shared sampler and cross-validation protocol."""

    cross_validation: CrossValidationSpec = field(default_factory=CrossValidationSpec)
    startup_trials: int = 10

    def __post_init__(self) -> None:
        if self.startup_trials < 1:
            raise ValueError("startup_trials must be positive")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "algorithm": "Optuna TPESampler",
            "optuna_version": optuna.__version__,
            "startup_trials": self.startup_trials,
            "cross_validation": self.cross_validation.to_manifest(),
        }


@dataclass(frozen=True)
class BayesianTrialResult:
    trial_number: int
    hyperparameters: Mapping[str, Any]
    mean_validation_score: float
    std_validation_score: float
    fold_validation_scores: tuple[float, ...]


@dataclass(frozen=True)
class BayesianModelResult:
    model_name: str
    selected_spec: ModelSpec
    best_mean_validation_score: float
    trials: tuple[BayesianTrialResult, ...]
    random_state: int


@dataclass(frozen=True)
class BayesianFamilyTuningRun:
    results: Mapping[str, BayesianModelResult]
    optimization: BayesianOptimizationSpec
    search_spaces: tuple[BayesianSearchSpace, ...]
    base_random_state: int
    n_train: int
    n_features: int

    @property
    def selected_model_specs(self) -> tuple[ModelSpec, ...]:
        return tuple(result.selected_spec for result in self.results.values())

    def to_manifest(self) -> dict[str, Any]:
        return {
            "base_random_state": self.base_random_state,
            "n_train": self.n_train,
            "n_features": self.n_features,
            "optimization": self.optimization.to_manifest(),
            "search_spaces": [space.to_manifest() for space in self.search_spaces],
            "models": {
                name: {
                    "random_state": result.random_state,
                    "selected_spec": result.selected_spec.to_manifest(),
                    "best_mean_validation_score": result.best_mean_validation_score,
                    "trials": [
                        {
                            "trial_number": trial.trial_number,
                            "hyperparameters": dict(trial.hyperparameters),
                            "mean_validation_score": trial.mean_validation_score,
                            "std_validation_score": trial.std_validation_score,
                            "fold_validation_scores": list(trial.fold_validation_scores),
                        }
                        for trial in result.trials
                    ],
                }
                for name, result in self.results.items()
            },
        }


def recommended_bayesian_search_spaces() -> tuple[BayesianSearchSpace, ...]:
    """Reviewable exploratory spaces chosen for an initial compute pilot.

    These ranges are intentionally returned by a named function rather than
    installed as hidden runner defaults. They are not a frozen confirmatory
    configuration.
    """

    return (
        BayesianSearchSpace(
            model=ModelSpec(
                ModelFamily.LOGISTIC_REGRESSION,
                {"solver": "liblinear", "max_iter": 2_000},
            ),
            distributions={
                "C": FloatDistribution(1e-4, 1e4, log=True),
                "penalty": CategoricalDistribution({"l1": "l1", "l2": "l2"}),
                "class_weight": CategoricalDistribution(
                    {"none": None, "balanced": "balanced"}
                ),
            },
            n_trials=30,
        ),
        BayesianSearchSpace(
            model=ModelSpec(
                ModelFamily.MLP,
                {
                    "max_iter": 500,
                    "early_stopping": True,
                    "n_iter_no_change": 20,
                },
            ),
            distributions={
                "hidden_layer_sizes": CategoricalDistribution(
                    {
                        "32": (32,),
                        "64": (64,),
                        "64x32": (64, 32),
                        "128x64": (128, 64),
                    }
                ),
                "activation": CategoricalDistribution(
                    {"relu": "relu", "tanh": "tanh"}
                ),
                "alpha": FloatDistribution(1e-6, 1e-2, log=True),
                "learning_rate_init": FloatDistribution(1e-4, 1e-2, log=True),
                "batch_size": CategoricalDistribution(
                    {"32": 32, "64": 64, "128": 128}
                ),
            },
            n_trials=60,
        ),
        BayesianSearchSpace(
            model=ModelSpec(ModelFamily.GAUSSIAN_NAIVE_BAYES),
            distributions={
                "var_smoothing": FloatDistribution(1e-12, 1e-6, log=True),
            },
            n_trials=20,
        ),
        BayesianSearchSpace(
            model=ModelSpec(ModelFamily.RANDOM_FOREST),
            distributions={
                "n_estimators": IntDistribution(100, 600, step=50),
                "max_depth": CategoricalDistribution(
                    {"none": None, "5": 5, "10": 10, "20": 20, "30": 30}
                ),
                "min_samples_split": IntDistribution(2, 20),
                "min_samples_leaf": IntDistribution(1, 10),
                "max_features": CategoricalDistribution(
                    {"sqrt": "sqrt", "log2": "log2", "half": 0.5}
                ),
                "class_weight": CategoricalDistribution(
                    {
                        "none": None,
                        "balanced": "balanced",
                        "balanced_subsample": "balanced_subsample",
                    }
                ),
            },
            n_trials=50,
        ),
        BayesianSearchSpace(
            model=ModelSpec(ModelFamily.DECISION_TREE),
            distributions={
                "criterion": CategoricalDistribution(
                    {"gini": "gini", "entropy": "entropy", "log_loss": "log_loss"}
                ),
                "max_depth": IntDistribution(2, 30),
                "min_samples_split": IntDistribution(2, 30),
                "min_samples_leaf": IntDistribution(1, 20),
                "max_features": CategoricalDistribution(
                    {"none": None, "sqrt": "sqrt", "log2": "log2"}
                ),
                "ccp_alpha": FloatDistribution(1e-8, 1e-2, log=True),
            },
            n_trials=40,
        ),
    )


def _training_data(X_train: ArrayLike, y_train: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(X_train, dtype=float)
    target = np.asarray(y_train)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X_train must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X_train must contain only finite values")
    if target.shape != (len(matrix),) or not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("y_train must contain one binary value per training row")
    return matrix, target.astype(np.int8)


def tune_model_family_bayesian(
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    search_spaces: tuple[BayesianSearchSpace, ...],
    optimization: BayesianOptimizationSpec,
    random_state: int,
    n_jobs: int = 1,
    require_declared_family: bool = True,
) -> BayesianFamilyTuningRun:
    """Optimize mean stratified-K-fold score without any test-set input."""

    if not search_spaces:
        raise ValueError("search_spaces must contain at least one model")
    base_specs = tuple(space.model for space in search_spaces)
    if require_declared_family:
        validate_declared_family(base_specs)
    elif len({spec.family for spec in base_specs}) != len(base_specs):
        raise ValueError("search spaces must not repeat a model family")
    if random_state < 0:
        raise ValueError("random_state must be non-negative")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    matrix, target = _training_data(X_train, y_train)
    folds = stratified_kfold_indices(
        target,
        cross_validation=optimization.cross_validation,
        random_state=random_state,
    )
    seeds = np.random.SeedSequence(random_state).generate_state(
        len(search_spaces), dtype=np.uint32
    )

    results: dict[str, BayesianModelResult] = {}
    for space, generated_seed in zip(search_spaces, seeds):
        model_seed = int(generated_seed)
        sampler = optuna.samplers.TPESampler(
            seed=model_seed,
            n_startup_trials=min(optimization.startup_trials, space.n_trials),
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            parameters = dict(space.model.hyperparameters)
            for name, distribution in space.distributions.items():
                parameters[name] = distribution.suggest(trial, name)
            candidate = ModelSpec(space.model.family, parameters)
            estimator = build_classifier(
                candidate,
                random_state=model_seed,
                n_jobs=1,
                native_probability=False,
            )
            with sklearn_parallelism(n_jobs):
                scores = cross_val_score(
                    estimator,
                    matrix,
                    target,
                    scoring=optimization.cross_validation.scoring,
                    cv=folds,
                    n_jobs=n_jobs,
                    error_score="raise",
                )
            fold_scores = tuple(float(score) for score in scores)
            trial.set_user_attr("resolved_hyperparameters", parameters)
            trial.set_user_attr("fold_validation_scores", fold_scores)
            trial.set_user_attr("std_validation_score", float(np.std(scores, ddof=0)))
            return float(np.mean(scores))

        study.optimize(objective, n_trials=space.n_trials, n_jobs=1, show_progress_bar=False)
        complete_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(complete_trials) != space.n_trials:
            raise RuntimeError(f"not every Bayesian trial completed for {space.model.name!r}")
        best = study.best_trial
        selected_parameters = dict(best.user_attrs["resolved_hyperparameters"])
        selected_spec = ModelSpec(space.model.family, selected_parameters)
        trials = tuple(
            BayesianTrialResult(
                trial_number=trial.number,
                hyperparameters=MappingProxyType(
                    dict(trial.user_attrs["resolved_hyperparameters"])
                ),
                mean_validation_score=float(trial.value),
                std_validation_score=float(trial.user_attrs["std_validation_score"]),
                fold_validation_scores=tuple(
                    float(score)
                    for score in trial.user_attrs["fold_validation_scores"]
                ),
            )
            for trial in complete_trials
        )
        results[space.model.name] = BayesianModelResult(
            model_name=space.model.name,
            selected_spec=selected_spec,
            best_mean_validation_score=float(best.value),
            trials=trials,
            random_state=model_seed,
        )
    return BayesianFamilyTuningRun(
        results=MappingProxyType(results),
        optimization=optimization,
        search_spaces=search_spaces,
        base_random_state=random_state,
        n_train=len(matrix),
        n_features=matrix.shape[1],
    )
