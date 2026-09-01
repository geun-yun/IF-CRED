"""Training-only stratified K-fold hyperparameter selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ifcred.models.registry import ModelSpec, build_classifier, validate_declared_family
from ifcred.parallel import sklearn_parallelism

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CrossValidationSpec:
    """Reproducible hyperparameter-selection protocol."""

    folds: int = 5
    scoring: str = "roc_auc"

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("cross-validation folds must be at least two")
        if not self.scoring.strip():
            raise ValueError("cross-validation scoring must be non-empty")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "splitter": "StratifiedKFold",
            "folds": self.folds,
            "shuffle": True,
            "scoring": self.scoring,
            "selection_rule": "highest_mean_validation_score",
        }


@dataclass(frozen=True)
class ModelSearchSpace:
    """A base estimator specification plus its declared finite search grid."""

    model: ModelSpec
    parameter_grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid: dict[str, tuple[Any, ...]] = {}
        for name, values in self.parameter_grid.items():
            if isinstance(values, (str, bytes)):
                raise TypeError("each hyperparameter grid value must be a sequence")
            candidates = tuple(values)
            if not candidates:
                raise ValueError(f"hyperparameter {name!r} has no candidate values")
            grid[name] = candidates
        # ModelSpec performs the central controlled-parameter check. Combining
        # one candidate from each dimension validates names before GridSearchCV.
        probe = dict(self.model.hyperparameters)
        probe.update({name: values[0] for name, values in grid.items()})
        ModelSpec(self.model.family, probe)
        object.__setattr__(self, "parameter_grid", MappingProxyType(grid))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model": self.model.to_manifest(),
            "parameter_grid": {
                name: list(values) for name, values in self.parameter_grid.items()
            },
        }


@dataclass(frozen=True)
class CandidateCVResult:
    """Cross-validation evidence for one hyperparameter candidate."""

    hyperparameters: Mapping[str, Any]
    mean_validation_score: float
    std_validation_score: float
    fold_validation_scores: tuple[float, ...]
    rank: int


@dataclass(frozen=True)
class ModelTuningResult:
    """Selected immutable specification and complete candidate evidence."""

    model_name: str
    selected_spec: ModelSpec
    best_mean_validation_score: float
    candidates: tuple[CandidateCVResult, ...]
    random_state: int


@dataclass(frozen=True)
class ModelFamilyTuningRun:
    """Training-only hyperparameter selections for a model family."""

    results: Mapping[str, ModelTuningResult]
    cross_validation: CrossValidationSpec
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
            "cross_validation": self.cross_validation.to_manifest(),
            "models": {
                name: {
                    "random_state": result.random_state,
                    "selected_spec": result.selected_spec.to_manifest(),
                    "best_mean_validation_score": result.best_mean_validation_score,
                    "candidates": [
                        {
                            "hyperparameters": dict(candidate.hyperparameters),
                            "mean_validation_score": candidate.mean_validation_score,
                            "std_validation_score": candidate.std_validation_score,
                            "fold_validation_scores": list(
                                candidate.fold_validation_scores
                            ),
                            "rank": candidate.rank,
                        }
                        for candidate in result.candidates
                    ],
                }
                for name, result in self.results.items()
            },
        }


def _training_data(
    X_train: ArrayLike, y_train: ArrayLike
) -> tuple[FloatArray, NDArray[np.int8]]:
    matrix = np.asarray(X_train, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X_train must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X_train must contain only finite values")
    target = np.asarray(y_train)
    if target.shape != (len(matrix),):
        raise ValueError("y_train must have one value per training row")
    try:
        target_is_finite = np.all(np.isfinite(target.astype(float)))
    except (TypeError, ValueError) as exc:
        raise ValueError("y_train must contain finite binary values 0/1") from exc
    if not target_is_finite or not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("y_train must contain finite binary values 0/1")
    return matrix, target.astype(np.int8)


def stratified_kfold_indices(
    y_train: ArrayLike,
    *,
    cross_validation: CrossValidationSpec,
    random_state: int,
) -> tuple[tuple[IntArray, IntArray], ...]:
    """Expose the exact training folds for audit and reproducibility tests."""

    target = np.asarray(y_train)
    if target.ndim != 1 or not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("y_train must be a one-dimensional binary target")
    counts = np.bincount(target.astype(np.int8), minlength=2)
    if np.any(counts < cross_validation.folds):
        raise ValueError("each training class must support every cross-validation fold")
    splitter = StratifiedKFold(
        n_splits=cross_validation.folds,
        shuffle=True,
        random_state=random_state,
    )
    dummy = np.zeros((len(target), 1), dtype=float)
    return tuple(
        (train.astype(np.int64), validation.astype(np.int64))
        for train, validation in splitter.split(dummy, target)
    )


def tune_model_family(
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    search_spaces: tuple[ModelSearchSpace, ...],
    cross_validation: CrossValidationSpec,
    random_state: int,
    n_jobs: int = 1,
    require_declared_family: bool = True,
) -> ModelFamilyTuningRun:
    """Select hyperparameters using only stratified folds of training data."""

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
        cross_validation=cross_validation,
        random_state=random_state,
    )

    seeds = np.random.SeedSequence(random_state).generate_state(
        len(search_spaces), dtype=np.uint32
    )
    results: dict[str, ModelTuningResult] = {}
    for space, generated_seed in zip(search_spaces, seeds):
        model_seed = int(generated_seed)
        estimator = build_classifier(
            space.model,
            random_state=model_seed,
            n_jobs=1,
            native_probability=False,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=dict(space.parameter_grid),
            scoring=cross_validation.scoring,
            cv=folds,
            refit=False,
            n_jobs=n_jobs,
            error_score="raise",
            return_train_score=False,
        )
        with sklearn_parallelism(n_jobs):
            search.fit(matrix, target)
        cv_results = search.cv_results_
        best_index = int(search.best_index_)
        best_grid_parameters = dict(cv_results["params"][best_index])
        selected_parameters = dict(space.model.hyperparameters)
        selected_parameters.update(best_grid_parameters)
        selected_spec = ModelSpec(space.model.family, selected_parameters)

        candidates: list[CandidateCVResult] = []
        for index, grid_parameters in enumerate(cv_results["params"]):
            parameters = dict(space.model.hyperparameters)
            parameters.update(dict(grid_parameters))
            fold_scores = tuple(
                float(cv_results[f"split{fold}_test_score"][index])
                for fold in range(cross_validation.folds)
            )
            candidates.append(
                CandidateCVResult(
                    hyperparameters=MappingProxyType(parameters),
                    mean_validation_score=float(cv_results["mean_test_score"][index]),
                    std_validation_score=float(cv_results["std_test_score"][index]),
                    fold_validation_scores=fold_scores,
                    rank=int(cv_results["rank_test_score"][index]),
                )
            )
        best_score = float(search.best_score_)
        if not np.isfinite(best_score):
            raise RuntimeError(f"tuning {space.model.name!r} produced a non-finite score")
        results[space.model.name] = ModelTuningResult(
            model_name=space.model.name,
            selected_spec=selected_spec,
            best_mean_validation_score=best_score,
            candidates=tuple(candidates),
            random_state=model_seed,
        )
    return ModelFamilyTuningRun(
        results=MappingProxyType(results),
        cross_validation=cross_validation,
        base_random_state=random_state,
        n_train=len(matrix),
        n_features=matrix.shape[1],
    )
