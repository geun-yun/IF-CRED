"""Leakage-safe fitting, calibration, prediction, and utility reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ifcred.models.registry import ModelSpec, build_classifier, validate_declared_family
from ifcred.parallel import sklearn_parallelism

FloatArray = NDArray[np.float64]


class CalibrationMethod(StrEnum):
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


@dataclass(frozen=True)
class CalibrationSpec:
    """Common post-hoc probability-calibration protocol."""

    method: CalibrationMethod = CalibrationMethod.SIGMOID
    folds: int = 5
    ensemble: bool = True

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("calibration folds must be at least two")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "folds": self.folds,
            "ensemble": self.ensemble,
        }


@dataclass(frozen=True)
class PredictiveUtility:
    """Predictive diagnostics reported separately from IF-CRED."""

    accuracy: float
    roc_auc: float
    brier_score: float
    expected_calibration_error: float


@dataclass(frozen=True)
class ModelPredictions:
    """Native and commonly calibrated outputs for one model family."""

    model_name: str
    native_probabilities: FloatArray
    calibrated_probabilities: FloatArray
    native_utility: PredictiveUtility
    calibrated_utility: PredictiveUtility
    random_state: int
    resolved_estimator_parameters: Mapping[str, Any]
    native_estimator: Any = field(repr=False, compare=False)
    calibrated_estimator: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class ModelTrainingData:
    """One model's training partition for differential-exposure conditions."""

    X_train: ArrayLike
    y_train: ArrayLike


@dataclass(frozen=True)
class ModelFamilyRun:
    """Outputs for one shared train/evaluation matrix pair."""

    models: Mapping[str, ModelPredictions]
    calibration: CalibrationSpec
    base_random_state: int
    n_train: int
    n_evaluation: int
    n_features: int
    calibration_bins: int

    @property
    def native_probabilities_by_model(self) -> dict[str, FloatArray]:
        return {
            name: result.native_probabilities.copy()
            for name, result in self.models.items()
        }

    @property
    def calibrated_probabilities_by_model(self) -> dict[str, FloatArray]:
        return {
            name: result.calibrated_probabilities.copy()
            for name, result in self.models.items()
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "base_random_state": self.base_random_state,
            "n_train": self.n_train,
            "n_evaluation": self.n_evaluation,
            "n_features": self.n_features,
            "calibration_bins": self.calibration_bins,
            "calibration": self.calibration.to_manifest(),
            "models": {
                name: {
                    "random_state": result.random_state,
                    "resolved_estimator_parameters": dict(
                        result.resolved_estimator_parameters
                    ),
                }
                for name, result in self.models.items()
            },
        }


def _matrix(values: ArrayLike, *, name: str) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _binary_target(values: ArrayLike, *, n_rows: int, name: str) -> NDArray[np.int8]:
    result = np.asarray(values)
    if result.shape != (n_rows,) or not np.all(np.isfinite(result.astype(float))):
        raise ValueError(f"{name} must have one finite value per matrix row")
    if not set(np.unique(result)).issubset({0, 1}):
        raise ValueError(f"{name} must contain binary values 0/1")
    return result.astype(np.int8)


def _positive_probabilities(estimator: Any, X: FloatArray) -> FloatArray:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_)
    positions = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(X), len(classes)) or len(positions) != 1:
        raise RuntimeError("classifier did not return an identifiable positive-class probability")
    result = probabilities[:, int(positions[0])]
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("classifier produced invalid probabilities")
    return result


def _expected_calibration_error(
    y_true: NDArray[np.int8], probabilities: FloatArray, *, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bins - 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if np.any(selected):
            error += float(selected.sum()) / total * abs(
                float(y_true[selected].mean()) - float(probabilities[selected].mean())
            )
    return float(error)


def _utility(
    y_true: NDArray[np.int8], probabilities: FloatArray, *, calibration_bins: int
) -> PredictiveUtility:
    if len(np.unique(y_true)) != 2:
        raise ValueError("evaluation outcomes must contain both classes for AUROC")
    return PredictiveUtility(
        accuracy=float(accuracy_score(y_true, probabilities >= 0.5)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        expected_calibration_error=_expected_calibration_error(
            y_true, probabilities, bins=calibration_bins
        ),
    )


def _fit_one_model(
    model_spec: ModelSpec,
    train: FloatArray,
    train_target: NDArray[np.int8],
    evaluation: FloatArray,
    evaluation_target: NDArray[np.int8],
    *,
    calibration: CalibrationSpec,
    model_seed: int,
    n_jobs: int,
    calibration_bins: int,
) -> ModelPredictions:
    class_counts = np.bincount(train_target, minlength=2)
    if np.any(class_counts < calibration.folds):
        raise ValueError("each training class must support every calibration fold")
    native_estimator = build_classifier(
        model_spec,
        random_state=model_seed,
        n_jobs=n_jobs,
        native_probability=True,
    )
    with sklearn_parallelism(n_jobs):
        native_estimator.fit(train, train_target)
    native_probabilities = _positive_probabilities(native_estimator, evaluation)

    calibration_cv = StratifiedKFold(
        n_splits=calibration.folds,
        shuffle=True,
        random_state=model_seed,
    )
    calibrated_estimator = CalibratedClassifierCV(
        estimator=build_classifier(
            model_spec,
            random_state=model_seed,
            # CalibratedClassifierCV owns fold-level parallelism. Keeping the
            # base estimator single-worker prevents nested oversubscription.
            n_jobs=1,
            native_probability=False,
        ),
        method=calibration.method.value,
        cv=calibration_cv,
        n_jobs=n_jobs,
        ensemble=calibration.ensemble,
    )
    with warnings.catch_warnings(), sklearn_parallelism(n_jobs):
        # Extreme injected cases can briefly overflow inside sklearn's sigmoid
        # optimizer even when the fitted calibrator returns valid probabilities.
        # Silence only those known internal matmul warnings; all other warnings
        # and the explicit probability validation below remain active.
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
            module=r"sklearn\.calibration",
        )
        calibrated_estimator.fit(train, train_target)
    calibrated_probabilities = _positive_probabilities(
        calibrated_estimator, evaluation
    )
    return ModelPredictions(
        model_name=model_spec.name,
        native_probabilities=native_probabilities,
        calibrated_probabilities=calibrated_probabilities,
        native_utility=_utility(
            evaluation_target,
            native_probabilities,
            calibration_bins=calibration_bins,
        ),
        calibrated_utility=_utility(
            evaluation_target,
            calibrated_probabilities,
            calibration_bins=calibration_bins,
        ),
        random_state=model_seed,
        resolved_estimator_parameters=native_estimator.get_params(deep=False),
        native_estimator=native_estimator,
        calibrated_estimator=calibrated_estimator,
    )


def _validate_specs(
    model_specs: tuple[ModelSpec, ...], *, require_declared_family: bool
) -> None:
    if not model_specs:
        raise ValueError("model_specs must contain at least one model")
    if require_declared_family:
        validate_declared_family(model_specs)
    elif len({spec.family for spec in model_specs}) != len(model_specs):
        raise ValueError("model specifications must not repeat a family")


def run_model_family_partitioned(
    training_by_model: Mapping[str, ModelTrainingData],
    X_evaluation: ArrayLike,
    y_evaluation: ArrayLike,
    *,
    model_specs: tuple[ModelSpec, ...],
    calibration: CalibrationSpec,
    random_state: int,
    n_jobs: int = 1,
    calibration_bins: int = 10,
    require_declared_family: bool = True,
) -> ModelFamilyRun:
    """Fit a family when systems receive model-specific training partitions."""

    _validate_specs(model_specs, require_declared_family=require_declared_family)
    expected_names = {spec.name for spec in model_specs}
    if set(training_by_model) != expected_names:
        raise ValueError("training_by_model must contain exactly the selected models")
    if random_state < 0:
        raise ValueError("random_state must be non-negative")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")

    evaluation = _matrix(X_evaluation, name="X_evaluation")
    evaluation_target = _binary_target(
        y_evaluation, n_rows=len(evaluation), name="y_evaluation"
    )
    seeds = np.random.SeedSequence(random_state).generate_state(
        len(model_specs), dtype=np.uint32
    )
    results: dict[str, ModelPredictions] = {}
    train_sizes: set[int] = set()
    for model_spec, generated_seed in zip(model_specs, seeds):
        supplied = training_by_model[model_spec.name]
        train = _matrix(supplied.X_train, name=f"X_train[{model_spec.name}]")
        if train.shape[1] != evaluation.shape[1]:
            raise ValueError("every training matrix must match evaluation columns")
        train_target = _binary_target(
            supplied.y_train,
            n_rows=len(train),
            name=f"y_train[{model_spec.name}]",
        )
        train_sizes.add(len(train))
        results[model_spec.name] = _fit_one_model(
            model_spec,
            train,
            train_target,
            evaluation,
            evaluation_target,
            calibration=calibration,
            model_seed=int(generated_seed),
            n_jobs=n_jobs,
            calibration_bins=calibration_bins,
        )
    if len(train_sizes) != 1:
        raise ValueError("model-specific training partitions must have equal row counts")
    return ModelFamilyRun(
        models=results,
        calibration=calibration,
        base_random_state=random_state,
        n_train=train_sizes.pop(),
        n_evaluation=len(evaluation),
        n_features=evaluation.shape[1],
        calibration_bins=calibration_bins,
    )


def run_model_family(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_evaluation: ArrayLike,
    y_evaluation: ArrayLike,
    *,
    model_specs: tuple[ModelSpec, ...],
    calibration: CalibrationSpec,
    random_state: int,
    n_jobs: int = 1,
    calibration_bins: int = 10,
    require_declared_family: bool = True,
) -> ModelFamilyRun:
    """Fit all models and calibrators without accessing evaluation outcomes.

    Evaluation outcomes are used only after prediction to calculate utility.
    The returned calibrated probabilities are the primary inputs to ``F``;
    native probabilities are retained for the planned sensitivity analysis.
    """

    train = _matrix(X_train, name="X_train")
    train_target = _binary_target(y_train, n_rows=len(train), name="y_train")
    return run_model_family_partitioned(
        {
            spec.name: ModelTrainingData(X_train=train, y_train=train_target)
            for spec in model_specs
        },
        X_evaluation,
        y_evaluation,
        model_specs=model_specs,
        calibration=calibration,
        random_state=random_state,
        n_jobs=n_jobs,
        calibration_bins=calibration_bins,
        require_declared_family=require_declared_family,
    )
