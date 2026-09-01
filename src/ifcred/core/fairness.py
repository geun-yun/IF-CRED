"""Plug-in contract for individual-fairness metrics and IF-CRED assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ifcred.core.components import model_stability
from ifcred.core.composite import composite, worst_case_composite

FloatArray = NDArray[np.float64]
ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class FairnessEvaluation:
    """Standardized output supplied by any chosen fairness metric.

    Native metric values must already be oriented and scaled to ``[0, 1]``,
    where higher means better individual fairness. A metric adapter is
    responsible for any transformation and for retaining the raw native output.
    """

    metric_name: str
    model_scores: Mapping[str, float]
    local_scores: Mapping[str, ArrayLike] | None = None
    native_outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric_name.strip():
            raise ValueError("metric_name must be non-empty")
        if not self.model_scores:
            raise ValueError("model_scores must contain at least one model")

        normalized_scores = {name: float(value) for name, value in self.model_scores.items()}
        if any(not name for name in normalized_scores):
            raise ValueError("model names must be non-empty")
        values = np.asarray(list(normalized_scores.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("model_scores must be finite and in [0, 1]")
        object.__setattr__(self, "model_scores", normalized_scores)

        if self.local_scores is None:
            return
        if set(self.local_scores) != set(normalized_scores):
            raise ValueError("local_scores must have the same model names as model_scores")
        normalized_local: dict[str, FloatArray] = {}
        expected_size: int | None = None
        for name, scores in self.local_scores.items():
            array = np.asarray(scores, dtype=float)
            if array.ndim != 1 or array.size == 0:
                raise ValueError("each local_scores value must be a non-empty one-dimensional array")
            if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
                raise ValueError("local_scores must be finite and in [0, 1]")
            if expected_size is None:
                expected_size = array.size
            elif array.size != expected_size:
                raise ValueError("all local_scores arrays must contain the same individuals")
            normalized_local[name] = array
        object.__setattr__(self, "local_scores", normalized_local)

    @property
    def mean_score(self) -> float:
        """Mean selected-metric fairness across the declared model family."""

        return float(np.mean(list(self.model_scores.values())))

    @property
    def minimum_score(self) -> float:
        """Worst selected-metric fairness across the declared model family."""

        return float(min(self.model_scores.values()))


class IndividualFairnessMetric(Protocol, Generic[ContextT]):
    """Interface implemented by an individual-fairness metric adapter.

    ``ContextT`` is metric-specific, so a metric may declare exactly the inputs
    it requires rather than being forced into the reference metric's signature.
    """

    name: str

    def evaluate(self, context: ContextT) -> FairnessEvaluation:
        """Evaluate the metric and return standardized plus native outputs."""


def evaluate_metric(
    metric: IndividualFairnessMetric[ContextT], context: ContextT
) -> FairnessEvaluation:
    """Run the selected metric through the common IF-CRED contract."""

    result = metric.evaluate(context)
    if not isinstance(result, FairnessEvaluation):
        raise TypeError("a metric adapter must return FairnessEvaluation")
    if result.metric_name != metric.name:
        raise ValueError("metric result name must match the selected metric name")
    return result


@dataclass(frozen=True)
class IFCredAssessment:
    """IF-CRED scores conditional on one explicitly selected fairness metric."""

    metric_name: str
    C: float
    D: float
    F: float
    M: float
    V: float
    F_min: float
    V_worst: float
    model_fairness: Mapping[str, float]
    local_fairness: Mapping[str, ArrayLike] | None


def assess_ifcred(*, C: float, D: float, fairness: FairnessEvaluation) -> IFCredAssessment:
    """Qualify a chosen fairness metric with coverage and stability evidence."""

    F = fairness.mean_score
    F_min = fairness.minimum_score
    M = model_stability(fairness.model_scores)
    V = composite(C=C, D=D, F=F, M=M)
    V_worst = worst_case_composite(C=C, D=D, F_min=F_min, M=M)
    return IFCredAssessment(
        metric_name=fairness.metric_name,
        C=float(C),
        D=float(D),
        F=F,
        M=M,
        V=V,
        F_min=F_min,
        V_worst=V_worst,
        model_fairness=fairness.model_scores,
        local_fairness=fairness.local_scores,
    )

