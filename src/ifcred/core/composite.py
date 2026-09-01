"""IF-CRED primary and diagnostic composite calculations."""

from __future__ import annotations

import numpy as np


def _validated_factor(value: float, *, name: str) -> float:
    factor = float(value)
    if not np.isfinite(factor) or not 0.0 <= factor <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return factor


def composite(*, C: float, D: float, F: float, M: float) -> float:
    """Calculate the fixed primary composite ``V = C * D * F * M``."""

    factors = [
        _validated_factor(C, name="C"),
        _validated_factor(D, name="D"),
        _validated_factor(F, name="F"),
        _validated_factor(M, name="M"),
    ]
    return float(np.clip(np.prod(factors), 0.0, 1.0))


def worst_case_composite(*, C: float, D: float, F_min: float, M: float) -> float:
    """Calculate the declared diagnostic ``V_worst`` variant."""

    return composite(C=C, D=D, F=F_min, M=M)

