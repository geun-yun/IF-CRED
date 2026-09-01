"""Paired benign and contradictory near-duplicate injections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

from ifcred.experiments.injections.base import InjectionTarget, PreparedInjection


@dataclass(frozen=True)
class PairedNearDuplicateConditions:
    """Geometry-matched control and contradiction candidate pools."""

    benign: PreparedInjection
    contradictory: PreparedInjection


def _validated_indices(
    values: ArrayLike, *, name: str, upper_bound: int, require_nonempty: bool = True
) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if require_nonempty and indices.size == 0:
        raise ValueError(f"{name} must not be empty")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicates")
    if np.any((indices < 0) | (indices >= upper_bound)):
        raise ValueError(f"{name} contains an out-of-range index")
    return indices.astype(np.int64, copy=False)


def _local_kth_distances(
    features: np.ndarray,
    anchors: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    n_rows = features.shape[0]
    if not 1 <= k < n_rows:
        raise ValueError("k must be at least 1 and smaller than the number of rows")
    search = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(features)
    distances, indices = search.kneighbors(features[anchors])
    kth = np.empty(len(anchors), dtype=float)
    for position, anchor in enumerate(anchors):
        non_self = distances[position][indices[position] != anchor]
        if len(non_self) < k:
            raise RuntimeError("nearest-neighbour search did not return enough non-self rows")
        kth[position] = non_self[k - 1]

    positive = kth[kth > 0.0]
    if positive.size == 0:
        raise ValueError("near-duplicates require variation in legitimate continuous features")
    kth[kth == 0.0] = float(np.median(positive))
    return kth


def prepare_paired_near_duplicates(
    X: ArrayLike,
    y: ArrayLike,
    *,
    legitimate_continuous_indices: ArrayLike,
    k: int,
    radius_fraction: float,
    random_state: int,
    eligible_anchor_indices: ArrayLike | None = None,
) -> PairedNearDuplicateConditions:
    """Create geometry-identical benign and contradictory candidates.

    Only declared legitimate continuous features are perturbed. Every other
    column—including protected attributes, categorical encodings, and missing
    indicators—is copied exactly from the real anchor.

    The candidate radius is ``radius_fraction * d_k(anchor)``, where ``d_k`` is
    measured among original rows in the declared continuous feature subspace.
    """

    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("X must be a two-dimensional array with at least two rows")
    if labels.shape != (len(features),):
        raise ValueError("y must contain one label per row in X")
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}) or len(unique_labels) != 2:
        raise ValueError("contradictory near-duplicates require binary labels encoded as 0 and 1")
    if not np.isfinite(radius_fraction) or radius_fraction <= 0.0:
        raise ValueError("radius_fraction must be finite and positive")

    continuous = _validated_indices(
        legitimate_continuous_indices,
        name="legitimate_continuous_indices",
        upper_bound=features.shape[1],
    )
    anchors = (
        np.arange(len(features), dtype=np.int64)
        if eligible_anchor_indices is None
        else _validated_indices(
            eligible_anchor_indices,
            name="eligible_anchor_indices",
            upper_bound=len(features),
        )
    )
    kth_distances = _local_kth_distances(features[:, continuous], anchors, k=k)
    radii = radius_fraction * kth_distances

    rng = np.random.default_rng(random_state)
    directions = rng.normal(size=(len(anchors), len(continuous)))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    while np.any(norms == 0.0):
        zero = np.flatnonzero(norms[:, 0] == 0.0)
        directions[zero] = rng.normal(size=(len(zero), len(continuous)))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms

    synthetic = features[anchors].copy()
    synthetic[:, continuous] += directions * radii[:, None]
    order = rng.permutation(len(anchors)).astype(np.int64)
    shared_metadata = {
        "anchor_kth_distance": kth_distances,
        "perturbation_radius": radii,
        "perturbation_direction": directions,
    }
    benign = PreparedInjection(
        condition="benign_near_duplicate",
        target=InjectionTarget.BENIGN_CONTROL,
        synthetic_features=synthetic,
        synthetic_labels=labels[anchors].copy(),
        anchor_indices=anchors,
        priority_order=order,
        known_unfair=np.zeros(len(anchors), dtype=bool),
        candidate_metadata={
            **shared_metadata,
            "label_contradiction": np.zeros(len(anchors), dtype=bool),
        },
    )
    contradictory = PreparedInjection(
        condition="contradictory_near_duplicate",
        target=InjectionTarget.FAIRNESS,
        synthetic_features=synthetic.copy(),
        synthetic_labels=1 - labels[anchors],
        anchor_indices=anchors.copy(),
        priority_order=order.copy(),
        # A contradictory label is a known stressor, but it is not yet a known
        # unfair prediction. That status must be derived from model outputs.
        known_unfair=np.zeros(len(anchors), dtype=bool),
        candidate_metadata={
            **shared_metadata,
            "label_contradiction": np.ones(len(anchors), dtype=bool),
        },
    )
    return PairedNearDuplicateConditions(benign=benign, contradictory=contradictory)


def prepare_localized_subgroup_contradiction(
    X: ArrayLike,
    y: ArrayLike,
    *,
    subgroup_mask: ArrayLike,
    legitimate_continuous_indices: ArrayLike,
    k: int,
    radius_fraction: float,
    random_state: int,
    minimum_subgroup_size: int = 2,
) -> PreparedInjection:
    """Create contradictory near-duplicates only for a declared subgroup.

    The mask is constructed by the dataset-specific protocol, allowing a
    protected group or a supported intersection to be selected without making
    the injection module depend on dataset column names or encodings.
    """

    features = np.asarray(X)
    mask = np.asarray(subgroup_mask)
    if mask.shape != (len(features),) or mask.dtype != np.bool_:
        raise ValueError("subgroup_mask must be a boolean vector with one value per row")
    if minimum_subgroup_size < 1:
        raise ValueError("minimum_subgroup_size must be positive")
    anchors = np.flatnonzero(mask)
    if len(anchors) < minimum_subgroup_size:
        raise ValueError("declared subgroup does not meet minimum_subgroup_size")

    paired = prepare_paired_near_duplicates(
        X,
        y,
        legitimate_continuous_indices=legitimate_continuous_indices,
        k=k,
        radius_fraction=radius_fraction,
        random_state=random_state,
        eligible_anchor_indices=anchors,
    )
    contradiction = paired.contradictory
    return PreparedInjection(
        condition="localized_subgroup_contradiction",
        target=InjectionTarget.FAIRNESS,
        synthetic_features=contradiction.synthetic_features,
        synthetic_labels=contradiction.synthetic_labels,
        anchor_indices=contradiction.anchor_indices,
        priority_order=contradiction.priority_order,
        known_unfair=contradiction.known_unfair,
        candidate_metadata={
            **contradiction.candidate_metadata,
            "localized_subgroup": np.ones(len(anchors), dtype=bool),
        },
    )
