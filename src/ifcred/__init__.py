"""IF-CRED: graded evidence for local individual-fairness audits."""

from ifcred.core.components import (
    CoverageResult,
    coverage,
    distance_stability,
    model_stability,
    weighted_prediction_consistency,
)
from ifcred.core.composite import composite, worst_case_composite
from ifcred.core.fairness import (
    FairnessEvaluation,
    IFCredAssessment,
    IndividualFairnessMetric,
    assess_ifcred,
    evaluate_metric,
)
from ifcred.core.graph import (
    AuditGraph,
    AuditGraphSpec,
    BandwidthPolicy,
    DistanceMetricSpec,
    WeightingRule,
    build_audit_graph,
)
from ifcred.metrics import WeightedPredictionConsistency, WeightedPredictionConsistencyInputs

__all__ = [
    "CoverageResult",
    "AuditGraph",
    "AuditGraphSpec",
    "BandwidthPolicy",
    "DistanceMetricSpec",
    "FairnessEvaluation",
    "IFCredAssessment",
    "IndividualFairnessMetric",
    "WeightedPredictionConsistency",
    "WeightedPredictionConsistencyInputs",
    "WeightingRule",
    "assess_ifcred",
    "build_audit_graph",
    "composite",
    "coverage",
    "distance_stability",
    "evaluate_metric",
    "model_stability",
    "weighted_prediction_consistency",
    "worst_case_composite",
]
