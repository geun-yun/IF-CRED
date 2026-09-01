"""Similarity-disagreement injection targeting IF-CRED distance stability D."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ifcred.experiments.injections.base import InjectionTarget, PreparedInjection
from ifcred.experiments.injections.near_duplicate import (
    _local_kth_distances,
    _validated_indices,
)

PairSimilarityEvaluator = Callable[
    [np.ndarray, np.ndarray], Mapping[str, ArrayLike]
]


def prepare_metric_disagreement_instances(
    X: ArrayLike,
    y: ArrayLike,
    *,
    legitimate_continuous_indices: ArrayLike,
    k: int,
    radius_fractions: Sequence[float],
    directions_per_radius: int,
    similarity_evaluator: PairSimilarityEvaluator,
    primary_metric: str,
    minimum_primary_similarity: float,
    target_reliability: float,
    random_state: int,
    eligible_anchor_indices: ArrayLike | None = None,
    require_target: bool = True,
) -> PreparedInjection:
    """Search local candidates for retained pairs with metric disagreement.

    The evaluator receives equal-length matrices of anchor and candidate rows
    and returns one ``[0, 1]`` similarity vector per declared metric. This keeps
    metric choice outside the injection mechanism.
    """

    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if features.ndim != 2 or len(features) < 2 or not np.all(np.isfinite(features)):
        raise ValueError("X must be a finite two-dimensional array with at least two rows")
    if labels.shape != (len(features),):
        raise ValueError("y must contain one label per row")
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
    fractions = np.asarray(radius_fractions, dtype=float)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("radius_fractions must be a non-empty sequence")
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("radius_fractions must be finite and positive")
    if directions_per_radius < 1:
        raise ValueError("directions_per_radius must be positive")
    if not 0.0 <= minimum_primary_similarity <= 1.0:
        raise ValueError("minimum_primary_similarity must be in [0, 1]")
    if not 0.0 <= target_reliability <= 1.0:
        raise ValueError("target_reliability must be in [0, 1]")

    local_scale = _local_kth_distances(features[:, continuous], anchors, k=k)
    candidates_per_anchor = len(fractions) * directions_per_radius
    rng = np.random.default_rng(random_state)
    candidate_bank = np.repeat(features[anchors, None, :], candidates_per_anchor, axis=1)
    candidate_fraction = np.tile(np.repeat(fractions, directions_per_radius), len(anchors))
    directions = rng.normal(
        size=(len(anchors), candidates_per_anchor, len(continuous))
    )
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    while np.any(norms == 0.0):
        zero_rows, zero_candidates = np.where(norms[:, :, 0] == 0.0)
        directions[zero_rows, zero_candidates] = rng.normal(
            size=(len(zero_rows), len(continuous))
        )
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
    directions /= norms
    radii = local_scale[:, None] * np.repeat(
        fractions, directions_per_radius
    )[None, :]
    candidate_bank[:, :, continuous] += directions * radii[:, :, None]

    flat_candidates = candidate_bank.reshape(-1, features.shape[1])
    flat_anchors = np.repeat(features[anchors], candidates_per_anchor, axis=0)
    evaluated = dict(similarity_evaluator(flat_anchors, flat_candidates))
    if len(evaluated) < 2:
        raise ValueError("distance disagreement requires at least two similarity definitions")
    if primary_metric not in evaluated:
        raise ValueError("primary_metric is missing from similarity_evaluator output")
    metric_names = tuple(evaluated)
    similarity_columns = []
    for name in metric_names:
        values = np.asarray(evaluated[name], dtype=float)
        if values.shape != (len(flat_candidates),):
            raise ValueError("each evaluated similarity must contain one value per candidate pair")
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("evaluated similarities must be finite and in [0, 1]")
        similarity_columns.append(values)
    similarity_matrix = np.column_stack(similarity_columns)
    reliability = np.clip(1.0 - 2.0 * similarity_matrix.std(axis=1, ddof=0), 0.0, 1.0)
    primary = similarity_matrix[:, metric_names.index(primary_metric)]

    selected_flat: list[int] = []
    selected_anchor_positions: list[int] = []
    target_met: list[bool] = []
    for anchor_position in range(len(anchors)):
        start = anchor_position * candidates_per_anchor
        positions = np.arange(start, start + candidates_per_anchor)
        retained = positions[primary[positions] >= minimum_primary_similarity]
        if retained.size == 0:
            # The threshold is a hard scientific constraint. Exclude this
            # anchor from the candidate pool instead of silently selecting a
            # below-threshold perturbation. Large real datasets provide many
            # more feasible anchors than any requested injection ratio needs.
            continue
        meeting = retained[reliability[retained] <= target_reliability]
        if meeting.size:
            # Among candidates that satisfy the target, retain the one with
            # the lowest cross-metric reliability (i.e. strongest metric
            # disagreement). Choosing the maximum would select the weakest
            # qualifying stressor and dilute the intended S5 response.
            chosen = int(meeting[np.argmin(reliability[meeting])])
            target_met.append(True)
        else:
            if require_target:
                # Strict confirmatory injections exclude an infeasible anchor.
                # They must never substitute a high-reliability pair under a
                # condition labelled as meeting the declared target.
                continue
            chosen = int(retained[np.argmin(reliability[retained])])
            target_met.append(False)
        selected_flat.append(chosen)
        selected_anchor_positions.append(anchor_position)

    if not selected_flat:
        raise ValueError(
            "no eligible anchor has a candidate meeting primary similarity"
        )

    chosen = np.asarray(selected_flat, dtype=np.int64)
    selected_anchors = anchors[np.asarray(selected_anchor_positions, dtype=np.int64)]
    chosen_similarities = similarity_matrix[chosen]
    metadata: dict[str, ArrayLike] = {
        "pair_reliability": reliability[chosen],
        "target_reliability": np.full(len(chosen), target_reliability),
        "target_reliability_met": np.asarray(target_met, dtype=bool),
        "primary_similarity": primary[chosen],
        "radius_fraction": candidate_fraction[chosen],
    }
    for metric_position, name in enumerate(metric_names):
        metadata[f"similarity__{name}"] = chosen_similarities[:, metric_position]

    selected_reliability = reliability[chosen]
    tie_breaker = rng.random(len(chosen))
    priority_order = np.lexsort((tie_breaker, selected_reliability))
    return PreparedInjection(
        condition="metric_disagreement_instance",
        target=InjectionTarget.DISTANCE_STABILITY,
        synthetic_features=flat_candidates[chosen],
        synthetic_labels=labels[selected_anchors].copy(),
        anchor_indices=selected_anchors,
        priority_order=priority_order,
        known_unfair=np.zeros(len(selected_anchors), dtype=bool),
        candidate_metadata=metadata,
    )
