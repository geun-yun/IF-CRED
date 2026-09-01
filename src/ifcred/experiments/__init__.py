"""Experimental implementations built around the theory-only core."""

from ifcred.experiments.catalogue import (
    EXPERIMENT_CATALOGUE,
    ExperimentStatus,
    PlannedExperiment,
    get_experiment,
)
from ifcred.experiments.orchestrator import (
    BayesianModelSelection,
    ConditionResult,
    ExperimentSetup,
    FairnessBinding,
    FixedModelSelection,
    PreparedConditionDefinition,
    PreparedRepetition,
    SharedPreparedRepetition,
    iter_repeated_experiment,
    prepare_experiment_repetition,
    prepare_policy_repetition,
    prepare_shared_repetition,
    run_clean_condition,
    run_model_disagreement_condition,
    run_prepared_condition_definitions,
    run_prepared_injection_condition,
    weighted_prediction_consistency_binding,
)
from ifcred.experiments.persistence import save_condition_result
from ifcred.experiments.repetitions import (
    RECOMMENDED_REPETITIONS,
    STANDARD_SEED_ROLES,
    ExperimentRepetition,
    RepetitionPlan,
)
from ifcred.experiments.similarity import (
    SimilarityCalibration,
    fit_similarity_calibration,
)

__all__ = [
    "ExperimentRepetition",
    "EXPERIMENT_CATALOGUE",
    "ExperimentSetup",
    "ExperimentStatus",
    "FairnessBinding",
    "FixedModelSelection",
    "BayesianModelSelection",
    "ConditionResult",
    "PlannedExperiment",
    "PreparedConditionDefinition",
    "PreparedRepetition",
    "SharedPreparedRepetition",
    "RECOMMENDED_REPETITIONS",
    "RepetitionPlan",
    "STANDARD_SEED_ROLES",
    "SimilarityCalibration",
    "fit_similarity_calibration",
    "get_experiment",
    "iter_repeated_experiment",
    "prepare_experiment_repetition",
    "prepare_policy_repetition",
    "prepare_shared_repetition",
    "run_clean_condition",
    "run_model_disagreement_condition",
    "run_prepared_condition_definitions",
    "run_prepared_injection_condition",
    "save_condition_result",
    "weighted_prediction_consistency_binding",
]
