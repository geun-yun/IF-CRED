"""VF3 (IFT-V): validity-filtered protected-attribute counterfactual testing."""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
import numpy as np
from ifcred.comparisons.base import ComparisonCase


@dataclass(frozen=True)
class VF3Config:
    theta: int = 0
    n_bins: int = 10
    max_samples: int = 5000


def run_vf3(model: object, case: ComparisonCase, config: VF3Config = VF3Config()) -> dict:
    started = perf_counter()
    if case.protected_index is None:
        return {"framework": "VF3_IFT_V", "applicable": False, "reason": "protected feature excluded", "runtime_seconds": perf_counter()-started}
    X = case.X_test[:config.max_samples]; p = case.protected_index
    lows_highs = np.quantile(case.X_train[:, p], (0.10, 0.90))
    original = np.asarray(model.predict(X), dtype=int)
    variants = []
    for value in lows_highs:
        changed = X.copy(); changed[:, p] = value
        for i in np.flatnonzero(original != np.asarray(model.predict(changed), dtype=int)):
            variants.append((X[int(i)], changed[int(i)]))
    edges = [np.linspace(np.min(case.X_train[:,j]), np.max(case.X_train[:,j]), config.n_bins+1)[1:-1] if np.ptp(case.X_train[:,j]) > 0 else np.array([]) for j in range(case.X_train.shape[1])]
    discretise = lambda row: tuple(int(np.digitize(row[j], edges[j])) for j in range(len(edges)))
    train_binned = [discretise(row) for row in case.X_train]
    pairs = tuple(combinations(range(case.X_train.shape[1]), 2))
    observed = {pair: {} for pair in pairs}
    for row in train_binned:
        for a,b in pairs: observed[(a,b)][(row[a],row[b])] = observed[(a,b)].get((row[a],row[b]),0)+1
    def valid(row):
        bins = discretise(row)
        return all(observed[(a,b)].get((bins[a],bins[b]),0) > config.theta for a,b in pairs)
    valid_count = sum(valid(a) and valid(b) for a,b in variants)
    return {"framework": "VF3_IFT_V", "applicable": True, "raw_idis": len(variants), "valid_idis": int(valid_count), "detection_rate": valid_count/len(variants) if variants else 0.0, "t": 2, "theta": config.theta, "n_bins": config.n_bins, "runtime_seconds": perf_counter()-started}
