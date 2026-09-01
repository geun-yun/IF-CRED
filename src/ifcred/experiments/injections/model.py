"""Controlled model-family disagreement using differential training exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ifcred.experiments.injections.base import (
    InjectedPair,
    assemble_real_anchored_injection,
)
from ifcred.experiments.injections.near_duplicate import PairedNearDuplicateConditions


@dataclass(frozen=True)
class ModelFamilyInjectionPlan:
    """Nested allocation of model systems to contradictory training exposure."""

    assignments: Mapping[str, str]
    requested_affected_fraction: float
    realized_affected_fraction: float
    allocation_order: tuple[str, ...]

    @property
    def affected_models(self) -> tuple[str, ...]:
        return tuple(name for name, condition in self.assignments.items() if condition == "contradictory")

    @property
    def unaffected_models(self) -> tuple[str, ...]:
        return tuple(name for name, condition in self.assignments.items() if condition == "benign")


@dataclass(frozen=True)
class ModelTrainingPartition:
    """One model's controlled augmented training inputs."""

    X_train: NDArray[np.float64]
    y_train: NDArray
    exposure: str
    train_pairs: tuple[InjectedPair, ...]


@dataclass(frozen=True)
class ModelFamilyDisagreementCase:
    """Model-specific training data and a common evaluation partition."""

    training_by_model: Mapping[str, ModelTrainingPartition]
    X_test: NDArray[np.float64]
    y_test: NDArray
    test_pairs: tuple[InjectedPair, ...]
    plan: ModelFamilyInjectionPlan
    injection_ratio: float


def prepare_model_family_injection_plan(
    model_names: Sequence[str],
    *,
    affected_fraction: float,
    random_state: int,
) -> ModelFamilyInjectionPlan:
    """Select a reproducible nested prefix of systems for contradictory exposure."""

    names = tuple(model_names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("model_names must contain at least two unique, non-empty names")
    if not np.isfinite(affected_fraction) or not 0.0 <= affected_fraction <= 1.0:
        raise ValueError("affected_fraction must be finite and in [0, 1]")
    rng = np.random.default_rng(random_state)
    order = tuple(names[position] for position in rng.permutation(len(names)))
    affected_count = int(round(affected_fraction * len(names)))
    affected = set(order[:affected_count])
    assignments = {
        name: "contradictory" if name in affected else "benign" for name in names
    }
    return ModelFamilyInjectionPlan(
        assignments=assignments,
        requested_affected_fraction=float(affected_fraction),
        realized_affected_fraction=affected_count / len(names),
        allocation_order=order,
    )


def assemble_model_family_disagreement_case(
    X: ArrayLike,
    y: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    paired_conditions: PairedNearDuplicateConditions,
    plan: ModelFamilyInjectionPlan,
    *,
    injection_ratio: float,
) -> ModelFamilyDisagreementCase:
    """Give affected systems contradictory training and all systems one test set.

    Benign and contradictory candidate features must be identical. The common
    test partition uses benign labels so utility is not defined differently for
    different systems; IF-CRED fairness remains based on model predictions.
    """

    benign = assemble_real_anchored_injection(
        X,
        y,
        train_indices,
        test_indices,
        paired_conditions.benign,
        injection_ratio=injection_ratio,
    )
    contradictory = assemble_real_anchored_injection(
        X,
        y,
        train_indices,
        test_indices,
        paired_conditions.contradictory,
        injection_ratio=injection_ratio,
    )
    if not np.array_equal(
        benign.selected_candidate_indices, contradictory.selected_candidate_indices
    ):
        raise ValueError("paired conditions must select identical candidates")
    if not np.array_equal(benign.X_train, contradictory.X_train) or not np.array_equal(
        benign.X_test, contradictory.X_test
    ):
        raise ValueError("paired conditions must have identical synthetic geometry")

    training: dict[str, ModelTrainingPartition] = {}
    for model_name, exposure in plan.assignments.items():
        source = contradictory if exposure == "contradictory" else benign
        training[model_name] = ModelTrainingPartition(
            X_train=source.X_train,
            y_train=source.y_train,
            exposure=exposure,
            train_pairs=source.train_pairs,
        )
    return ModelFamilyDisagreementCase(
        training_by_model=training,
        X_test=benign.X_test,
        y_test=benign.y_test,
        test_pairs=benign.test_pairs,
        plan=plan,
        injection_ratio=float(injection_ratio),
    )
