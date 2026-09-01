"""Explicit construction of the five declared IF-CRED model families."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelFamily(StrEnum):
    """Model families declared for the primary model-stability analysis."""

    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"
    GAUSSIAN_NAIVE_BAYES = "gaussian_naive_bayes"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"


DECLARED_MODEL_FAMILIES = tuple(ModelFamily)


@dataclass(frozen=True)
class ModelSpec:
    """One model family and its reviewable hyperparameter choices."""

    family: ModelFamily
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, ModelFamily):
            raise TypeError("family must be a ModelFamily value")
        parameters = dict(self.hyperparameters)
        controlled = {"random_state", "n_jobs"} & set(parameters)
        if controlled:
            raise ValueError(
                f"parameters controlled by the registry/runner cannot appear in "
                f"ModelSpec: {sorted(controlled)}"
            )
        object.__setattr__(self, "hyperparameters", MappingProxyType(parameters))

    @property
    def name(self) -> str:
        return self.family.value

    def to_manifest(self) -> dict[str, Any]:
        return {"family": self.family.value, "hyperparameters": dict(self.hyperparameters)}


def build_classifier(
    spec: ModelSpec,
    *,
    random_state: int,
    n_jobs: int = 1,
    native_probability: bool = True,
) -> ClassifierMixin:
    """Build one unfitted estimator while centrally controlling randomness."""

    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    parameters = dict(spec.hyperparameters)
    if spec.family == ModelFamily.LOGISTIC_REGRESSION:
        # LogisticRegression's selected liblinear solver does not use n_jobs;
        # pinning it to one prevents the surrounding joblib context from
        # resolving its default to every worker and emitting a no-effect
        # warning for every fit and calibration fold.
        return LogisticRegression(random_state=random_state, n_jobs=1, **parameters)
    if spec.family == ModelFamily.MLP:
        return MLPClassifier(random_state=random_state, **parameters)
    if spec.family == ModelFamily.GAUSSIAN_NAIVE_BAYES:
        return GaussianNB(**parameters)
    if spec.family == ModelFamily.RANDOM_FOREST:
        return RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs, **parameters
        )
    if spec.family == ModelFamily.DECISION_TREE:
        return DecisionTreeClassifier(random_state=random_state, **parameters)
    raise ValueError(f"unsupported model family: {spec.family}")


def validate_declared_family(specs: tuple[ModelSpec, ...]) -> None:
    """Require exactly one specification for each declared primary family."""

    families = tuple(spec.family for spec in specs)
    if len(set(families)) != len(families):
        raise ValueError("model specifications must not repeat a family")
    if set(families) != set(DECLARED_MODEL_FAMILIES):
        missing = sorted(set(DECLARED_MODEL_FAMILIES) - set(families))
        extra = sorted(set(families) - set(DECLARED_MODEL_FAMILIES))
        raise ValueError(
            f"primary model family must contain exactly the five declared models; "
            f"missing={missing}, extra={extra}"
        )
