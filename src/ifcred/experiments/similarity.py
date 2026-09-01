"""Training-fitted similarity scaling for fixed evaluation semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics.pairwise import paired_distances

from ifcred.core.graph import AuditGraph, AuditGraphSpec, BandwidthPolicy, build_audit_graph

FloatArray = NDArray[np.float64]


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SimilarityCalibration:
    """Per-metric bandwidths fitted without outcomes or predictions."""

    fitting_spec: AuditGraphSpec
    bandwidths: Mapping[str, float]
    n_training_rows: int
    n_features: int
    training_matrix_sha256: str

    def __post_init__(self) -> None:
        if self.fitting_spec.bandwidth_policy == BandwidthPolicy.FIXED:
            raise ValueError("fitting_spec must estimate rather than fix bandwidths")
        expected = {metric.name for metric in self.fitting_spec.metrics}
        bandwidths = {name: float(value) for name, value in self.bandwidths.items()}
        if set(bandwidths) != expected:
            raise ValueError("calibrated bandwidths must match every declared metric")
        values = np.asarray(list(bandwidths.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("calibrated bandwidths must be finite and positive")
        if self.n_training_rows <= self.fitting_spec.k or self.n_features < 1:
            raise ValueError("calibration dimensions are inconsistent with the graph")
        if len(self.training_matrix_sha256) != 64:
            raise ValueError("training_matrix_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "bandwidths", MappingProxyType(bandwidths))

    def fixed_audit_spec(self) -> AuditGraphSpec:
        """Create the graph specification applied unchanged to evaluation sets."""

        return replace(
            self.fitting_spec,
            bandwidth_policy=BandwidthPolicy.FIXED,
            fixed_bandwidths=dict(self.bandwidths),
        )

    def build_evaluation_graph(
        self,
        X_evaluation: ArrayLike,
        *,
        row_ids: ArrayLike | None = None,
    ) -> AuditGraph:
        """Build an evaluation graph using only the frozen training scales."""

        matrix = np.asarray(X_evaluation, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features:
            raise ValueError(
                "X_evaluation must use the same feature columns as calibration"
            )
        resolved_ids = None if row_ids is None else np.asarray(row_ids)
        return build_audit_graph(
            matrix,
            self.fixed_audit_spec(),
            row_ids=resolved_ids,
        )

    def evaluate_pairs(
        self, left: ArrayLike, right: ArrayLike
    ) -> Mapping[str, FloatArray]:
        """Evaluate aligned candidate pairs with the frozen training scales."""

        first = np.asarray(left, dtype=float)
        second = np.asarray(right, dtype=float)
        if (
            first.ndim != 2
            or first.shape != second.shape
            or first.shape[1] != self.n_features
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
        ):
            raise ValueError("left and right must be equal finite runtime matrices")
        output: dict[str, FloatArray] = {}
        for metric in self.fitting_spec.metrics:
            distance = paired_distances(
                first,
                second,
                metric=metric.metric,
                **dict(metric.params),
            )
            bandwidth = self.bandwidths[metric.name]
            output[metric.name] = np.exp(
                -np.square(distance) / (2.0 * bandwidth**2)
            )
        return MappingProxyType(output)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "fitting_spec": self.fitting_spec.to_manifest(),
            "application_policy": "fixed_training_fitted_bandwidths",
            "bandwidths": dict(self.bandwidths),
            "n_training_rows": self.n_training_rows,
            "n_features": self.n_features,
            "training_matrix_sha256": self.training_matrix_sha256,
        }


def fit_similarity_calibration(
    X_train: ArrayLike,
    fitting_spec: AuditGraphSpec,
) -> SimilarityCalibration:
    """Fit distance-to-similarity scales from clean training features only."""

    matrix = np.asarray(X_train, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X_train must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X_train must contain only finite values")
    if fitting_spec.bandwidth_policy == BandwidthPolicy.FIXED:
        raise ValueError("cannot fit similarity calibration from a fixed-bandwidth spec")
    training_graph = build_audit_graph(matrix, fitting_spec)
    return SimilarityCalibration(
        fitting_spec=fitting_spec,
        bandwidths=training_graph.bandwidths,
        n_training_rows=len(matrix),
        n_features=matrix.shape[1],
        training_matrix_sha256=_matrix_fingerprint(matrix),
    )
