"""Memory-bounded construction of an IF-CRED audit neighbourhood graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import NearestNeighbors

from ifcred.parallel import sklearn_parallelism

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ScalarDistance = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


class BandwidthPolicy(StrEnum):
    """Declared conversion scale from distances to similarities."""

    MEDIAN_RETAINED_MAX = "median_retained_max"
    MEDIAN_POSITIVE_PAIR = "median_positive_pair"
    FIXED = "fixed"


class WeightingRule(StrEnum):
    """Supported distance-to-similarity rules."""

    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class DistanceMetricSpec:
    """One declared distance definition."""

    name: str
    metric: str | ScalarDistance
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("distance metric name must be non-empty")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise TypeError("metric must be a scikit-learn metric name or scalar callable")

    @property
    def implementation_name(self) -> str:
        if isinstance(self.metric, str):
            return self.metric
        return f"{self.metric.__module__}.{self.metric.__qualname__}"


@dataclass(frozen=True)
class AuditGraphSpec:
    """Complete graph construction configuration.

    Scientific choices are required arguments. Search/batching settings affect
    engineering performance and have reproducible defaults.
    """

    k: int
    metrics: tuple[DistanceMetricSpec, ...]
    primary_metric: str
    bandwidth_policy: BandwidthPolicy
    weighting_rule: WeightingRule = WeightingRule.GAUSSIAN
    fixed_bandwidths: Mapping[str, float] = field(default_factory=dict)
    search_algorithm: str = "auto"
    n_jobs: int = 1
    pair_batch_size: int = 100_000

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        if not self.metrics:
            raise ValueError("at least one distance metric must be declared")
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("distance metric names must be unique")
        if self.primary_metric not in names:
            raise ValueError("primary_metric must name a declared distance metric")
        if self.search_algorithm not in {"auto", "ball_tree", "kd_tree", "brute"}:
            raise ValueError("unsupported nearest-neighbour search algorithm")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        if self.pair_batch_size < 1:
            raise ValueError("pair_batch_size must be positive")
        if self.bandwidth_policy == BandwidthPolicy.FIXED:
            if set(self.fixed_bandwidths) != set(names):
                raise ValueError("fixed_bandwidths must contain every declared metric")
            values = np.asarray(list(self.fixed_bandwidths.values()), dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("fixed bandwidths must be finite and positive")
        elif self.fixed_bandwidths:
            raise ValueError("fixed_bandwidths are only valid with the fixed policy")

    def metric(self, name: str) -> DistanceMetricSpec:
        return next(metric for metric in self.metrics if metric.name == name)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "primary_metric": self.primary_metric,
            "bandwidth_policy": self.bandwidth_policy.value,
            "weighting_rule": self.weighting_rule.value,
            "metrics": [
                {
                    "name": metric.name,
                    "implementation": metric.implementation_name,
                    "params": dict(metric.params),
                }
                for metric in self.metrics
            ],
            "fixed_bandwidths": dict(self.fixed_bandwidths),
            "search_algorithm": self.search_algorithm,
            "n_jobs": self.n_jobs,
            "pair_batch_size": self.pair_batch_size,
        }


@dataclass(frozen=True)
class AuditGraph:
    """Directed fixed-K graph and retained-pair evidence for IF-CRED."""

    spec: AuditGraphSpec
    row_ids: NDArray
    neighbour_indices: IntArray
    weights: FloatArray
    distances: Mapping[str, FloatArray]
    similarities: Mapping[str, FloatArray]
    bandwidths: Mapping[str, float]

    def __post_init__(self) -> None:
        indices = np.asarray(self.neighbour_indices)
        weights = np.asarray(self.weights, dtype=float)
        row_ids = np.asarray(self.row_ids)
        if indices.ndim != 2 or indices.shape[1] != self.spec.k:
            raise ValueError("neighbour_indices must have shape (individual, k)")
        n_rows = indices.shape[0]
        if row_ids.shape != (n_rows,) or len(np.unique(row_ids)) != n_rows:
            raise ValueError("row_ids must contain one unique ID per individual")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("neighbour_indices must be integer-valued")
        if np.any((indices < 0) | (indices >= n_rows)):
            raise ValueError("neighbour_indices contain an out-of-range row")
        if any(i in indices[i] for i in range(n_rows)):
            raise ValueError("self-neighbours are not permitted")
        if any(len(np.unique(indices[i])) != self.spec.k for i in range(n_rows)):
            raise ValueError("neighbours must be unique within each row")
        if weights.shape != indices.shape or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite and match neighbour_indices")
        if np.any((weights < 0.0) | (weights > 1.0)):
            raise ValueError("weights must lie in [0, 1]")
        metric_names = {metric.name for metric in self.spec.metrics}
        if set(self.distances) != metric_names or set(self.similarities) != metric_names:
            raise ValueError("distance and similarity mappings must match declared metrics")
        if set(self.bandwidths) != metric_names:
            raise ValueError("bandwidths must match declared metrics")
        for name in metric_names:
            distance = np.asarray(self.distances[name], dtype=float)
            similarity = np.asarray(self.similarities[name], dtype=float)
            if distance.shape != indices.shape or not np.all(np.isfinite(distance)):
                raise ValueError(f"distances for {name!r} are invalid")
            if np.any(distance < 0.0):
                raise ValueError(f"distances for {name!r} must be non-negative")
            if similarity.shape != indices.shape or not np.all(np.isfinite(similarity)):
                raise ValueError(f"similarities for {name!r} are invalid")
            if np.any((similarity < 0.0) | (similarity > 1.0)):
                raise ValueError(f"similarities for {name!r} must lie in [0, 1]")
        if not np.allclose(weights, self.similarities[self.spec.primary_metric]):
            raise ValueError("weights must equal primary-metric similarities")

    @property
    def neighbour_row_ids(self) -> NDArray:
        return self.row_ids[self.neighbour_indices]

    @property
    def metric_order(self) -> tuple[str, ...]:
        return tuple(metric.name for metric in self.spec.metrics)

    def similarity_tensor(self) -> FloatArray:
        """Return ``(metric, individual, neighbour)`` similarities for D."""

        return np.stack([self.similarities[name] for name in self.metric_order], axis=0)


def _primary_neighbours(X: FloatArray, spec: AuditGraphSpec) -> IntArray:
    if len(X) <= spec.k:
        raise ValueError("audit matrix must contain more rows than k")
    primary = spec.metric(spec.primary_metric)
    with sklearn_parallelism(spec.n_jobs):
        search = NearestNeighbors(
            n_neighbors=spec.k + 1,
            algorithm=spec.search_algorithm,
            metric=primary.metric,
            metric_params=dict(primary.params) or None,
            n_jobs=spec.n_jobs,
        ).fit(X)
        returned_distances, returned_indices = search.kneighbors(X, return_distance=True)
    neighbours = np.empty((len(X), spec.k), dtype=np.int64)
    for row in range(len(X)):
        keep = returned_indices[row] != row
        candidate_indices = returned_indices[row][keep]
        candidate_distances = returned_distances[row][keep]
        if len(candidate_indices) < spec.k:
            raise RuntimeError("nearest-neighbour search did not return k non-self rows")
        order = np.lexsort((candidate_indices, candidate_distances))
        neighbours[row] = candidate_indices[order[: spec.k]]
    return neighbours


def _retained_pair_distances(
    X: FloatArray,
    indices: IntArray,
    metric: DistanceMetricSpec,
    *,
    batch_size: int,
) -> FloatArray:
    flat_neighbours = indices.reshape(-1)
    result = np.empty(len(flat_neighbours), dtype=float)
    k = indices.shape[1]
    for start in range(0, len(flat_neighbours), batch_size):
        stop = min(start + batch_size, len(flat_neighbours))
        anchors = np.arange(start, stop, dtype=np.int64) // k
        neighbours = flat_neighbours[start:stop]
        values = paired_distances(
            X[anchors],
            X[neighbours],
            metric=metric.metric,
            **dict(metric.params),
        )
        result[start:stop] = values
    matrix = result.reshape(indices.shape)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"metric {metric.name!r} produced invalid retained-pair distances")
    return matrix


def _bandwidth(
    distances: FloatArray,
    metric_name: str,
    spec: AuditGraphSpec,
) -> float:
    if spec.bandwidth_policy == BandwidthPolicy.FIXED:
        return float(spec.fixed_bandwidths[metric_name])
    if spec.bandwidth_policy == BandwidthPolicy.MEDIAN_RETAINED_MAX:
        values = np.max(distances, axis=1)
        positive = values[values > 0.0]
    else:
        positive = distances[distances > 0.0]
    if positive.size:
        return float(np.median(positive))
    return 1.0


def _similarity(distances: FloatArray, bandwidth: float, rule: WeightingRule) -> FloatArray:
    if rule != WeightingRule.GAUSSIAN:
        raise ValueError(f"unsupported weighting rule: {rule}")
    return np.clip(
        np.exp(-np.square(distances) / (2.0 * bandwidth**2)), 0.0, 1.0
    )


def build_audit_graph(
    X: NDArray,
    spec: AuditGraphSpec,
    *,
    row_ids: NDArray | None = None,
) -> AuditGraph:
    """Construct a directed graph without allocating full distance matrices."""

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("X must be a non-empty two-dimensional audit matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values")
    resolved_row_ids = (
        np.arange(len(matrix), dtype=np.int64)
        if row_ids is None
        else np.asarray(row_ids)
    )
    indices = _primary_neighbours(matrix, spec)
    distances: dict[str, FloatArray] = {}
    similarities: dict[str, FloatArray] = {}
    bandwidths: dict[str, float] = {}
    for metric in spec.metrics:
        distance = _retained_pair_distances(
            matrix, indices, metric, batch_size=spec.pair_batch_size
        )
        bandwidth = _bandwidth(distance, metric.name, spec)
        distances[metric.name] = distance
        bandwidths[metric.name] = bandwidth
        similarities[metric.name] = _similarity(
            distance, bandwidth, spec.weighting_rule
        )
    return AuditGraph(
        spec=spec,
        row_ids=resolved_row_ids,
        neighbour_indices=indices,
        weights=similarities[spec.primary_metric].copy(),
        distances=distances,
        similarities=similarities,
        bandwidths=bandwidths,
    )
