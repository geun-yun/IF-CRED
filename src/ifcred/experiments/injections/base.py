"""Shared machinery for real-anchored synthetic-instance experiments.

Injection mechanisms prepare candidate synthetic rows. This module selects a
nested prefix of those candidates and places each one in the same partition as
its real anchor. It does not define how a synthetic row is generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class InjectionTarget(StrEnum):
    """Primary theoretical target declared by an injection mechanism."""

    COVERAGE = "C"
    DISTANCE_STABILITY = "D"
    FAIRNESS = "F"
    MODEL_STABILITY = "M"
    BENIGN_CONTROL = "control"
    MULTIPLE = "multiple"


@dataclass(frozen=True)
class PreparedInjection:
    """Candidate synthetic rows produced from real anchor rows.

    ``anchor_indices`` refer to rows in the original, unsplit dataset.
    ``priority_order`` is a permutation of candidate positions. Selecting a
    prefix creates nested injection-ratio conditions.
    """

    condition: str
    target: InjectionTarget
    synthetic_features: ArrayLike
    synthetic_labels: ArrayLike
    anchor_indices: ArrayLike
    priority_order: ArrayLike
    known_unfair: ArrayLike
    candidate_metadata: Mapping[str, ArrayLike] = field(default_factory=dict)
    selection_group_ids: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not self.condition.strip():
            raise ValueError("condition must be non-empty")
        features = np.asarray(self.synthetic_features, dtype=float)
        labels = np.asarray(self.synthetic_labels)
        anchors = np.asarray(self.anchor_indices)
        order = np.asarray(self.priority_order)
        unfair = np.asarray(self.known_unfair)

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("synthetic_features must be a non-empty two-dimensional array")
        n_candidates = features.shape[0]
        if labels.shape != (n_candidates,):
            raise ValueError("synthetic_labels must contain one value per candidate")
        if anchors.shape != (n_candidates,) or not np.issubdtype(anchors.dtype, np.integer):
            raise ValueError("anchor_indices must be an integer vector with one value per candidate")
        if order.shape != (n_candidates,) or not np.issubdtype(order.dtype, np.integer):
            raise ValueError("priority_order must be an integer permutation of candidate positions")
        if set(order.tolist()) != set(range(n_candidates)):
            raise ValueError("priority_order must contain every candidate position exactly once")
        if unfair.shape != (n_candidates,) or unfair.dtype != np.bool_:
            raise ValueError("known_unfair must be a boolean vector with one value per candidate")
        if not np.all(np.isfinite(features)):
            raise ValueError("synthetic_features must contain only finite values")
        if not np.all(np.isfinite(np.asarray(labels, dtype=float))):
            raise ValueError("synthetic_labels must contain only finite values")
        normalized_metadata: dict[str, NDArray] = {}
        for name, values in self.candidate_metadata.items():
            array = np.asarray(values)
            if not name or array.ndim == 0 or array.shape[0] != n_candidates:
                raise ValueError(
                    "each candidate_metadata value must begin with the candidate dimension"
                )
            normalized_metadata[name] = array
        if self.selection_group_ids is None:
            groups = np.arange(n_candidates, dtype=np.int64)
        else:
            groups = np.asarray(self.selection_group_ids)
            if groups.shape != (n_candidates,) or not np.issubdtype(groups.dtype, np.integer):
                raise ValueError("selection_group_ids must be an integer vector per candidate")
            if np.any(groups < 0):
                raise ValueError("selection_group_ids must be non-negative")
            groups = groups.astype(np.int64, copy=False)

        object.__setattr__(self, "synthetic_features", features)
        object.__setattr__(self, "synthetic_labels", labels)
        object.__setattr__(self, "anchor_indices", anchors.astype(np.int64, copy=False))
        object.__setattr__(self, "priority_order", order.astype(np.int64, copy=False))
        object.__setattr__(self, "known_unfair", unfair)
        object.__setattr__(self, "candidate_metadata", normalized_metadata)
        object.__setattr__(self, "selection_group_ids", groups)


@dataclass(frozen=True)
class InjectedPair:
    """Traceable relationship between an original row and one synthetic row."""

    condition: str
    target: InjectionTarget
    partition: str
    candidate_index: int
    anchor_original_index: int
    anchor_partition_index: int
    synthetic_partition_index: int
    known_unfair: bool


@dataclass(frozen=True)
class AugmentedPartitions:
    """Augmented train/test arrays plus pair-level provenance."""

    X_train: FloatArray
    y_train: NDArray
    X_test: FloatArray
    y_test: NDArray
    selected_candidate_indices: IntArray
    injected_pairs: tuple[InjectedPair, ...]
    requested_injection_ratio: float
    realized_injection_ratio: float

    @property
    def train_pairs(self) -> tuple[InjectedPair, ...]:
        return tuple(pair for pair in self.injected_pairs if pair.partition == "train")

    @property
    def test_pairs(self) -> tuple[InjectedPair, ...]:
        return tuple(pair for pair in self.injected_pairs if pair.partition == "test")


def _validated_partition(
    values: ArrayLike, *, name: str, n_rows: int
) -> IntArray:
    indices = np.asarray(values)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicate rows")
    if np.any((indices < 0) | (indices >= n_rows)):
        raise ValueError(f"{name} contains an out-of-range row")
    return indices.astype(np.int64, copy=False)


def assemble_real_anchored_injection(
    X: ArrayLike,
    y: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    prepared: PreparedInjection,
    *,
    injection_ratio: float,
) -> AugmentedPartitions:
    """Add selected synthetic rows to the same partitions as their anchors.

    The ratio is defined relative to the number of original rows, matching the
    prior experiment's interpretation. The function never modifies model
    predictions and never moves an anchor or its synthetic counterpart across
    the train/test boundary.
    """

    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if labels.shape != (features.shape[0],):
        raise ValueError("y must contain one label per row in X")
    if prepared.synthetic_features.shape[1] != features.shape[1]:
        raise ValueError("synthetic and original rows must have the same number of features")
    if np.any((prepared.anchor_indices < 0) | (prepared.anchor_indices >= len(features))):
        raise ValueError("prepared injection contains an out-of-range anchor")
    if not np.isfinite(injection_ratio) or not 0.0 <= injection_ratio <= 1.0:
        raise ValueError("injection_ratio must be finite and in [0, 1]")

    train = _validated_partition(train_indices, name="train_indices", n_rows=len(features))
    test = _validated_partition(test_indices, name="test_indices", n_rows=len(features))
    train_set = set(train.tolist())
    test_set = set(test.tolist())
    if train_set & test_set:
        raise ValueError("train_indices and test_indices must be disjoint")
    if train_set | test_set != set(range(len(features))):
        raise ValueError("train_indices and test_indices must partition every original row")

    requested_count = int(round(injection_ratio * len(features)))
    if requested_count > len(prepared.priority_order):
        raise ValueError("prepared injection does not contain enough candidates for this ratio")
    if requested_count == 0:
        selected = np.empty(0, dtype=np.int64)
    else:
        ordered_groups: list[int] = []
        for candidate in prepared.priority_order:
            group = int(prepared.selection_group_ids[candidate])
            if group not in ordered_groups:
                ordered_groups.append(group)
        selected_list: list[int] = []
        priority_rank = {
            int(candidate): rank for rank, candidate in enumerate(prepared.priority_order)
        }
        for group in ordered_groups:
            members = np.flatnonzero(prepared.selection_group_ids == group).tolist()
            members.sort(key=priority_rank.__getitem__)
            selected_list.extend(members)
            if len(selected_list) >= requested_count:
                break
        selected = np.asarray(selected_list, dtype=np.int64)
    selected_train = [int(i) for i in selected if int(prepared.anchor_indices[i]) in train_set]
    selected_test = [int(i) for i in selected if int(prepared.anchor_indices[i]) in test_set]

    X_train = np.vstack((features[train], prepared.synthetic_features[selected_train]))
    y_train = np.concatenate((labels[train], prepared.synthetic_labels[selected_train]))
    X_test = np.vstack((features[test], prepared.synthetic_features[selected_test]))
    y_test = np.concatenate((labels[test], prepared.synthetic_labels[selected_test]))

    train_positions = {int(original): position for position, original in enumerate(train)}
    test_positions = {int(original): position for position, original in enumerate(test)}
    pairs: list[InjectedPair] = []
    for appended_position, candidate in enumerate(selected_train):
        anchor = int(prepared.anchor_indices[candidate])
        pairs.append(
            InjectedPair(
                condition=prepared.condition,
                target=prepared.target,
                partition="train",
                candidate_index=candidate,
                anchor_original_index=anchor,
                anchor_partition_index=train_positions[anchor],
                synthetic_partition_index=len(train) + appended_position,
                known_unfair=bool(prepared.known_unfair[candidate]),
            )
        )
    for appended_position, candidate in enumerate(selected_test):
        anchor = int(prepared.anchor_indices[candidate])
        pairs.append(
            InjectedPair(
                condition=prepared.condition,
                target=prepared.target,
                partition="test",
                candidate_index=candidate,
                anchor_original_index=anchor,
                anchor_partition_index=test_positions[anchor],
                synthetic_partition_index=len(test) + appended_position,
                known_unfair=bool(prepared.known_unfair[candidate]),
            )
        )

    return AugmentedPartitions(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        selected_candidate_indices=np.asarray(selected, dtype=np.int64),
        injected_pairs=tuple(pairs),
        requested_injection_ratio=float(injection_ratio),
        realized_injection_ratio=len(selected) / len(features),
    )
