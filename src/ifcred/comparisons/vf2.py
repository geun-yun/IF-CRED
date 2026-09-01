"""VF2: model-agnostic approximation of the statistical loss-ratio audit."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.stats import norm
from ifcred.comparisons.base import ComparisonCase, predict_probability


@dataclass(frozen=True)
class VF2Config:
    n_attack_steps: int = 20
    step_size: float = 0.05
    n_random_directions: int = 20
    distance_penalty_lambda: float = 0.1
    delta_threshold: float = 1.25
    alpha: float = 0.05
    max_samples: int = 1000


def run_vf2(model: object, case: ComparisonCase, config: VF2Config = VF2Config()) -> dict:
    started = perf_counter()
    if case.protected_index is None:
        return {"framework": "VF2", "applicable": False, "reason": "protected feature excluded", "runtime_seconds": perf_counter() - started}
    rng = np.random.default_rng(case.random_seed)
    count = min(len(case.X_test), config.max_samples)
    chosen = rng.choice(len(case.X_test), count, replace=False)
    X, y = case.X_test[chosen].copy(), case.y_test[chosen]
    eps = 1e-6
    loss = lambda p: -(y * np.log(np.clip(p, eps, 1-eps)) + (1-y) * np.log(np.clip(1-p, eps, 1-eps)))
    original = loss(predict_probability(model, X))
    per = config.n_attack_steps * config.n_random_directions
    directions = rng.normal(size=(count, per, X.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=2, keepdims=True), 1e-12)
    for step in range(config.n_attack_steps):
        offset = step * config.n_random_directions
        directions[:, offset:offset+2] = 0.0
        directions[:, offset, case.protected_index] = 1.0
        directions[:, offset+1, case.protected_index] = -1.0
    radii = np.repeat(np.arange(1, config.n_attack_steps + 1), config.n_random_directions) * config.step_size
    candidates = X[:, None, :] + directions * radii[None, :, None]
    candidates = np.clip(candidates, np.min(case.X_train, axis=0), np.max(case.X_train, axis=0))
    repeated_y = np.repeat(y, per)
    flat_p = predict_probability(model, candidates.reshape(-1, X.shape[1]))
    candidate_loss = -(repeated_y*np.log(np.clip(flat_p,eps,1-eps)) + (1-repeated_y)*np.log(np.clip(1-flat_p,eps,1-eps))).reshape(count, per)
    delta = candidates - X[:, None, :]
    delta[:, :, case.protected_index] = 0.0
    objective = candidate_loss - config.distance_penalty_lambda * np.sum(delta**2, axis=2)
    best = np.argmax(objective, axis=1)
    adversarial = np.where(objective[np.arange(count), best] > original, candidate_loss[np.arange(count), best], original)
    ratios = adversarial / (original + eps)
    mean = float(np.mean(ratios)); std = float(np.std(ratios, ddof=1)) if count > 1 else 0.0
    lower = mean - float(norm.ppf(1-config.alpha)) * std / np.sqrt(max(count,1))
    return {"framework": "VF2", "applicable": True, "mean_loss_ratio": mean, "lower_confidence_statistic": lower, "reject_fairness": bool(lower > config.delta_threshold), "detection_rate": float(np.mean(ratios > config.delta_threshold)), "n_evaluated": count, "delta_threshold": config.delta_threshold, "alpha": config.alpha, "attack_steps": config.n_attack_steps, "random_directions": config.n_random_directions, "runtime_seconds": perf_counter()-started}
