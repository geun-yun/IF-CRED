"""VF1: bounded observed-neighbour approximation of formal verification."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.spatial.distance import cdist
from ifcred.comparisons.base import ComparisonCase


@dataclass(frozen=True)
class VF1Config:
    max_samples: int = 1000
    neighbours_per_sample: int = 10
    max_nonprotected_distance: float = 0.35


def run_vf1(model: object, case: ComparisonCase, config: VF1Config = VF1Config()) -> dict:
    started = perf_counter()
    if case.protected_index is None:
        return {"framework": "VF1", "applicable": False, "reason": "protected feature excluded", "runtime_seconds": perf_counter()-started}
    X = case.X_test[:config.max_samples]; protected = case.protected_index
    allowed = [j for j in range(X.shape[1]) if j != protected]
    scale = np.maximum(np.std(case.X_train[:, allowed], axis=0), 1e-12)
    distances = cdist(X[:, allowed]/scale, X[:, allowed]/scale); np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :min(config.neighbours_per_sample, len(X)-1)]
    pairs = {tuple(sorted((i, int(j)))) for i, neighbours in enumerate(nearest) for j in neighbours if distances[i,j] <= config.max_nonprotected_distance and X[i,protected] != X[j,protected]}
    labels = np.asarray(model.predict(X), dtype=int)
    violations = sum(labels[i] != labels[j] for i,j in pairs)
    return {"framework": "VF1", "applicable": True, "bias_found": bool(violations), "pairs_checked": len(pairs), "violations": int(violations), "detection_rate": violations/len(pairs) if pairs else 0.0, "mode": "approximate_search", "neighbours_per_sample": config.neighbours_per_sample, "max_nonprotected_distance": config.max_nonprotected_distance, "runtime_seconds": perf_counter()-started}
