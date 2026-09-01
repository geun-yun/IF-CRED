"""Execute VF1, VF2, and VF3 on exactly matched IF-CRED cases."""

from __future__ import annotations
from dataclasses import replace
import hashlib
from typing import Any
from ifcred.comparisons.base import ComparisonCase
from ifcred.comparisons.vf1 import run_vf1
from ifcred.comparisons.vf2 import run_vf2
from ifcred.comparisons.vf3_iftv import run_vf3


def run_prior_frameworks(model: object, case: ComparisonCase) -> list[dict[str, Any]]:
    """Return native framework outputs with a shared identifying envelope."""

    common = {"dataset_id": case.dataset_id, "condition": case.condition, "variant": case.variant, "ratio": case.ratio, "repetition": case.repetition, "implementation_status": "documented_tabular_approximation", "native_scales_preserved": True, "comparison_label_version": "chronological_2020_2021_2024"}

    def framework_case(name: str) -> ComparisonCase:
        material = f"ifcred-comparator-v1|{case.random_seed}|{name}".encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        return replace(case, random_seed=seed)

    results = (
        # Retain the historical method-specific seed keys so relabelling does
        # not alter the numerical comparator outputs.
        run_vf1(model, framework_case("VF2")),
        run_vf2(model, framework_case("VF1")),
        run_vf3(model, framework_case("VF3_IFT_V")),
    )
    return [{**common, **result} for result in results]
