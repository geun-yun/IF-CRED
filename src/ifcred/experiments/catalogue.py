"""Reviewable catalogue of planned IF-CRED experiment branches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExperimentStatus(StrEnum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partially_implemented"
    PLANNED = "planned"


@dataclass(frozen=True)
class PlannedExperiment:
    experiment_id: str
    name: str
    purpose: str
    conditions: tuple[str, ...]
    primary_outcomes: tuple[str, ...]
    status: ExperimentStatus


EXPERIMENT_CATALOGUE = (
    PlannedExperiment(
        "E1",
        "CDFM inspection without synthetic injection",
        "Characterize IF-CRED on clean real datasets and test audit-design sensitivity.",
        (
            "D6 Adult, D7 Credit Default, D8 Cleveland",
            "primary protected attribute included",
            "primary protected attribute excluded",
            "30 paired outer repetitions (provisional)",
            "declared K, metric-set, bandwidth, and probability sensitivities",
        ),
        ("C", "D", "F", "M", "V", "utility", "individual-level profiles"),
        ExperimentStatus.IMPLEMENTED,
    ),
    PlannedExperiment(
        "E2",
        "CDFM inspection with synthetic injection",
        "Test component selectivity, dose response, and interpretation safeguards under controlled stressors.",
        (
            "benign and contradictory near duplicates",
            "isolated instances and dominant-neighbour pairs",
            "metric-disagreement instances",
            "model-family disagreement",
            "multiple injection ratios and mechanism-specific severities",
            "30 paired outer repetitions (provisional)",
            "constant/random/uninformative prediction controls",
        ),
        (
            "C balance and strength",
            "D pair reliability",
            "model and local F",
            "M fairness stability",
            "V and component cross-effects",
            "utility",
        ),
        ExperimentStatus.IMPLEMENTED,
    ),
    PlannedExperiment(
        "E3",
        "Prior-framework comparison",
        "Test the incremental diagnostic information provided by IF-CRED.",
        (
            "VF1 search verifier (John et al., 2020)",
            "VF2 loss-ratio audit (Maity et al., 2021)",
            "VF3 IFT-V testing",
            "clean E1 settings and applicable injected E2 settings",
            "30 matched outer repetitions (provisional)",
        ),
        (
            "native decisions and rates",
            "known-stressor detection",
            "agreement and complementary failure cases",
            "runtime and applicability",
        ),
        ExperimentStatus.IMPLEMENTED,
    ),
)


def get_experiment(experiment_id: str) -> PlannedExperiment:
    for experiment in EXPERIMENT_CATALOGUE:
        if experiment.experiment_id == experiment_id:
            return experiment
    raise KeyError(f"unknown experiment ID: {experiment_id}")
