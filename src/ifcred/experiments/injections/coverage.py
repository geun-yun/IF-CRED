"""Controlled weak-evidence injections targeting IF-CRED coverage C."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

from ifcred.experiments.injections.base import InjectionTarget, PreparedInjection


def _indices(values: ArrayLike, *, name: str, upper: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer) or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional integer array")
    if len(np.unique(result)) != len(result) or np.any((result < 0) | (result >= upper)):
        raise ValueError(f"{name} must contain unique in-range indices")
    return result.astype(np.int64, copy=False)


def _inputs(
    X: ArrayLike,
    y: ArrayLike,
    continuous_indices: ArrayLike,
    eligible_anchor_indices: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if features.ndim != 2 or len(features) < 2 or not np.all(np.isfinite(features)):
        raise ValueError("X must be a finite two-dimensional array with at least two rows")
    if labels.shape != (len(features),):
        raise ValueError("y must contain one label per row")
    continuous = _indices(
        continuous_indices, name="legitimate_continuous_indices", upper=features.shape[1]
    )
    anchors = (
        np.arange(len(features), dtype=np.int64)
        if eligible_anchor_indices is None
        else _indices(eligible_anchor_indices, name="eligible_anchor_indices", upper=len(features))
    )
    return features, labels, continuous, anchors


def _bandwidth(features: np.ndarray, *, k: int) -> float:
    if not 1 <= k < len(features):
        raise ValueError("k must be at least 1 and smaller than the number of rows")
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(features)
    kth = np.empty(len(features), dtype=float)
    for row in range(len(features)):
        non_self = distances[row][indices[row] != row]
        kth[row] = non_self[k - 1]
    positive = kth[kth > 0.0]
    if positive.size == 0:
        raise ValueError("coverage injections require variation in continuous features")
    return float(np.median(positive))


def _distance_at_similarity(similarity: float, bandwidth: float) -> float:
    if not np.isfinite(similarity) or not 0.0 < similarity < 1.0:
        raise ValueError("similarity thresholds must be finite and strictly between 0 and 1")
    return float(bandwidth * np.sqrt(-2.0 * np.log(similarity)))


def _random_unit_vectors(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    vectors = rng.normal(size=(rows, columns))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    while np.any(norms == 0.0):
        zero = np.flatnonzero(norms[:, 0] == 0.0)
        vectors[zero] = rng.normal(size=(len(zero), columns))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def _move_until_isolated(
    original: np.ndarray,
    anchors: np.ndarray,
    directions: np.ndarray,
    required_nearest_distance: float,
    *,
    extra_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    search = NearestNeighbors(n_neighbors=1).fit(original)
    radii = np.full(len(anchors), required_nearest_distance + extra_radius)
    candidates = original[anchors] + directions * radii[:, None]
    nearest = search.kneighbors(candidates, return_distance=True)[0][:, 0]
    for _ in range(40):
        unresolved = nearest < required_nearest_distance
        if not np.any(unresolved):
            return candidates, radii, nearest
        radii[unresolved] *= 1.5
        candidates[unresolved] = (
            original[anchors[unresolved]] + directions[unresolved] * radii[unresolved, None]
        )
        nearest[unresolved] = search.kneighbors(
            candidates[unresolved], return_distance=True
        )[0][:, 0]
    raise RuntimeError("could not construct isolated candidates within 40 expansions")


def prepare_isolated_instances(
    X: ArrayLike,
    y: ArrayLike,
    *,
    legitimate_continuous_indices: ArrayLike,
    k: int,
    maximum_background_similarity: float,
    random_state: int,
    eligible_anchor_indices: ArrayLike | None = None,
) -> PreparedInjection:
    """Generate same-label rows whose strongest background similarity is bounded."""

    features, labels, continuous, anchors = _inputs(
        X, y, legitimate_continuous_indices, eligible_anchor_indices
    )
    original = features[:, continuous]
    bandwidth = _bandwidth(original, k=k)
    required = _distance_at_similarity(maximum_background_similarity, bandwidth)
    rng = np.random.default_rng(random_state)
    directions = _random_unit_vectors(rng, len(anchors), len(continuous))
    moved, radii, nearest = _move_until_isolated(original, anchors, directions, required)
    synthetic = features[anchors].copy()
    synthetic[:, continuous] = moved
    achieved_similarity = np.exp(-np.square(nearest) / (2.0 * bandwidth**2))
    return PreparedInjection(
        condition="isolated_instance",
        target=InjectionTarget.COVERAGE,
        synthetic_features=synthetic,
        synthetic_labels=labels[anchors].copy(),
        anchor_indices=anchors,
        priority_order=rng.permutation(len(anchors)),
        known_unfair=np.zeros(len(anchors), dtype=bool),
        candidate_metadata={
            "displacement_radius": radii,
            "nearest_background_distance": nearest,
            "maximum_background_similarity": achieved_similarity,
            "similarity_bandwidth": np.full(len(anchors), bandwidth),
        },
    )


def prepare_dominant_neighbour_pairs(
    X: ArrayLike,
    y: ArrayLike,
    *,
    legitimate_continuous_indices: ArrayLike,
    k: int,
    minimum_pair_similarity: float,
    maximum_background_similarity: float,
    random_state: int,
    eligible_anchor_indices: ArrayLike | None = None,
) -> PreparedInjection:
    """Generate remote two-row groups with one strong within-pair neighbour."""

    if minimum_pair_similarity <= maximum_background_similarity:
        raise ValueError("minimum_pair_similarity must exceed maximum_background_similarity")
    features, labels, continuous, anchors = _inputs(
        X, y, legitimate_continuous_indices, eligible_anchor_indices
    )
    original = features[:, continuous]
    bandwidth = _bandwidth(original, k=k)
    far_distance = _distance_at_similarity(maximum_background_similarity, bandwidth)
    pair_distance = _distance_at_similarity(minimum_pair_similarity, bandwidth)
    rng = np.random.default_rng(random_state)
    center_directions = _random_unit_vectors(rng, len(anchors), len(continuous))
    pair_directions = _random_unit_vectors(rng, len(anchors), len(continuous))
    centers, center_radii, _ = _move_until_isolated(
        original,
        anchors,
        center_directions,
        far_distance + pair_distance / 2.0,
        extra_radius=pair_distance / 2.0,
    )

    first = centers - pair_directions * (pair_distance / 2.0)
    second = centers + pair_directions * (pair_distance / 2.0)
    search = NearestNeighbors(n_neighbors=1).fit(original)
    nearest_first = search.kneighbors(first, return_distance=True)[0][:, 0]
    nearest_second = search.kneighbors(second, return_distance=True)[0][:, 0]
    # If an oblique pair direction moved an endpoint toward the data cloud,
    # move the entire centre farther along its calibrated escape direction.
    for _ in range(40):
        unresolved = np.minimum(nearest_first, nearest_second) < far_distance
        if not np.any(unresolved):
            break
        center_radii[unresolved] *= 1.5
        centers[unresolved] = (
            original[anchors[unresolved]]
            + center_directions[unresolved] * center_radii[unresolved, None]
        )
        first[unresolved] = centers[unresolved] - pair_directions[unresolved] * (
            pair_distance / 2.0
        )
        second[unresolved] = centers[unresolved] + pair_directions[unresolved] * (
            pair_distance / 2.0
        )
        nearest_first[unresolved] = search.kneighbors(
            first[unresolved], return_distance=True
        )[0][:, 0]
        nearest_second[unresolved] = search.kneighbors(
            second[unresolved], return_distance=True
        )[0][:, 0]
    else:
        raise RuntimeError("could not construct dominant-neighbour pairs")

    n_groups = len(anchors)
    repeated_anchors = np.repeat(anchors, 2)
    synthetic = features[repeated_anchors].copy()
    paired_continuous = np.empty((2 * n_groups, len(continuous)))
    paired_continuous[0::2] = first
    paired_continuous[1::2] = second
    synthetic[:, continuous] = paired_continuous
    group_order = rng.permutation(n_groups)
    priority_order = np.asarray(
        [candidate for group in group_order for candidate in (2 * group, 2 * group + 1)],
        dtype=np.int64,
    )
    background_distance = np.repeat(np.minimum(nearest_first, nearest_second), 2)
    return PreparedInjection(
        condition="dominant_neighbour_pair",
        target=InjectionTarget.COVERAGE,
        synthetic_features=synthetic,
        synthetic_labels=np.repeat(labels[anchors], 2),
        anchor_indices=repeated_anchors,
        priority_order=priority_order,
        known_unfair=np.zeros(2 * n_groups, dtype=bool),
        selection_group_ids=np.repeat(np.arange(n_groups), 2),
        candidate_metadata={
            "pair_group": np.repeat(np.arange(n_groups), 2),
            "pair_member": np.tile([0, 1], n_groups),
            "within_pair_similarity": np.full(
                2 * n_groups, np.exp(-(pair_distance**2) / (2.0 * bandwidth**2))
            ),
            "maximum_background_similarity": np.exp(
                -np.square(background_distance) / (2.0 * bandwidth**2)
            ),
            "similarity_bandwidth": np.full(2 * n_groups, bandwidth),
        },
    )

