"""Real-anchored synthetic-instance injection infrastructure."""

from ifcred.experiments.injections.base import (
    AugmentedPartitions,
    InjectedPair,
    InjectionTarget,
    PreparedInjection,
    assemble_real_anchored_injection,
)
from ifcred.experiments.injections.near_duplicate import (
    PairedNearDuplicateConditions,
    prepare_localized_subgroup_contradiction,
    prepare_paired_near_duplicates,
)
from ifcred.experiments.injections.coverage import (
    prepare_dominant_neighbour_pairs,
    prepare_isolated_instances,
)
from ifcred.experiments.injections.distance import (
    PairSimilarityEvaluator,
    prepare_metric_disagreement_instances,
)
from ifcred.experiments.injections.model import (
    ModelFamilyDisagreementCase,
    ModelFamilyInjectionPlan,
    ModelTrainingPartition,
    assemble_model_family_disagreement_case,
    prepare_model_family_injection_plan,
)

__all__ = [
    "AugmentedPartitions",
    "InjectedPair",
    "InjectionTarget",
    "ModelFamilyDisagreementCase",
    "ModelFamilyInjectionPlan",
    "ModelTrainingPartition",
    "PairedNearDuplicateConditions",
    "PairSimilarityEvaluator",
    "PreparedInjection",
    "assemble_real_anchored_injection",
    "assemble_model_family_disagreement_case",
    "prepare_dominant_neighbour_pairs",
    "prepare_isolated_instances",
    "prepare_localized_subgroup_contradiction",
    "prepare_metric_disagreement_instances",
    "prepare_model_family_injection_plan",
    "prepare_paired_near_duplicates",
]
