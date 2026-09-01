"""End-to-end composition of IF-CRED experimental stages."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ifcred.core.components import CoverageResult, coverage, distance_stability
from ifcred.core.fairness import (
    FairnessEvaluation,
    IFCredAssessment,
    assess_ifcred,
    evaluate_metric,
)
from ifcred.core.graph import AuditGraph, AuditGraphSpec
from ifcred.data import (
    DatasetBundle,
    PreparedDataset,
    ProtectedAttributePolicy,
    make_stratified_split,
    preprocess_dataset,
)
from ifcred.experiments.injections import (
    AugmentedPartitions,
    InjectedPair,
    InjectionTarget,
    ModelFamilyDisagreementCase,
    PreparedInjection,
    assemble_real_anchored_injection,
)
from ifcred.experiments.repetitions import ExperimentRepetition, RepetitionPlan
from ifcred.experiments.similarity import SimilarityCalibration, fit_similarity_calibration
from ifcred.metrics import (
    WeightedPredictionConsistency,
    WeightedPredictionConsistencyInputs,
)
from ifcred.models import (
    BayesianFamilyTuningRun,
    BayesianOptimizationSpec,
    BayesianSearchSpace,
    CalibrationSpec,
    ModelFamilyRun,
    ModelSpec,
    ModelTrainingData,
    run_model_family,
    run_model_family_partitioned,
    tune_model_family_bayesian,
    validate_declared_family,
)
from ifcred.parallel import PARALLEL_BACKEND

FloatArray = NDArray[np.float64]
FairnessEvaluator = Callable[[Mapping[str, ArrayLike], AuditGraph], FairnessEvaluation]
InjectionFactory = Callable[["PreparedRepetition", int], PreparedInjection]


@dataclass(frozen=True)
class FairnessBinding:
    """Bind a selectable F plug-in to its graph/model context constructor."""

    name: str
    evaluator: FairnessEvaluator

    def __post_init__(self) -> None:
        if not self.name.strip() or not callable(self.evaluator):
            raise ValueError("fairness binding requires a name and evaluator")

    def evaluate(
        self, probabilities_by_model: Mapping[str, ArrayLike], graph: AuditGraph
    ) -> FairnessEvaluation:
        result = self.evaluator(probabilities_by_model, graph)
        if result.metric_name != self.name:
            raise ValueError("fairness binding name must match its evaluation")
        return result


def weighted_prediction_consistency_binding() -> FairnessBinding:
    metric = WeightedPredictionConsistency()

    def evaluator(
        probabilities_by_model: Mapping[str, ArrayLike], graph: AuditGraph
    ) -> FairnessEvaluation:
        return evaluate_metric(
            metric,
            WeightedPredictionConsistencyInputs(
                probabilities_by_model=probabilities_by_model,
                neighbour_indices=graph.neighbour_indices,
                weights=graph.weights,
            ),
        )

    return FairnessBinding(name=metric.name, evaluator=evaluator)


@dataclass(frozen=True)
class FixedModelSelection:
    """Preselected model settings, primarily for smoke/sensitivity runs."""

    model_specs: tuple[ModelSpec, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_declared_family(self.model_specs)
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "strategy": "fixed",
            "model_specs": [spec.to_manifest() for spec in self.model_specs],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class BayesianModelSelection:
    """Training-only Bayesian selection used by a full repetition."""

    search_spaces: tuple[BayesianSearchSpace, ...]
    optimization: BayesianOptimizationSpec

    def __post_init__(self) -> None:
        validate_declared_family(tuple(space.model for space in self.search_spaces))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "strategy": "bayesian",
            "optimization": self.optimization.to_manifest(),
            "search_spaces": [space.to_manifest() for space in self.search_spaces],
        }


ModelSelection = FixedModelSelection | BayesianModelSelection


@dataclass(frozen=True)
class ExperimentSetup:
    """Shared scientific and engineering choices for one dataset/policy branch."""

    protected_policy: ProtectedAttributePolicy
    test_size: float
    graph_fitting_spec: AuditGraphSpec
    model_selection: ModelSelection
    probability_calibration: CalibrationSpec
    fairness: FairnessBinding = field(
        default_factory=weighted_prediction_consistency_binding
    )
    stratify_protected: bool = True
    n_jobs: int = 1
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be strictly between zero and one")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least two")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "protected_policy": self.protected_policy.value,
            "test_size": self.test_size,
            "graph_fitting_spec": self.graph_fitting_spec.to_manifest(),
            "model_selection": self.model_selection.to_manifest(),
            "probability_calibration": self.probability_calibration.to_manifest(),
            "fairness_metric": self.fairness.name,
            "stratify_protected": self.stratify_protected,
            "n_jobs": self.n_jobs,
            "parallel_backend": PARALLEL_BACKEND,
            "calibration_bins": self.calibration_bins,
        }


@dataclass(frozen=True)
class PreparedRepetition:
    """Clean split-fitted state shared across paired conditions."""

    bundle: DatasetBundle
    repetition: ExperimentRepetition
    setup: ExperimentSetup
    prepared_dataset: PreparedDataset
    experiment_matrix: FloatArray
    continuous_indices: NDArray[np.int64]
    selected_model_specs: tuple[ModelSpec, ...]
    model_selection_manifest: Mapping[str, Any]
    similarity_calibration: SimilarityCalibration

    def to_manifest(self) -> dict[str, Any]:
        split = self.prepared_dataset.split
        return {
            "dataset": dict(self.bundle.manifest),
            "repetition": self.repetition.to_manifest(),
            "setup": self.setup.to_manifest(),
            "split": {
                "train_indices": split.train_indices.tolist(),
                "test_indices": split.test_indices.tolist(),
                "random_state": split.random_state,
                "test_size": split.test_size,
                "stratification_strategy": split.stratification_strategy,
            },
            "experiment_matrix_shape": list(self.experiment_matrix.shape),
            "continuous_indices": self.continuous_indices.tolist(),
            "primary_protected_indices": self.prepared_dataset.experiment_primary_protected_indices(
                self.setup.protected_policy
            ).tolist(),
            "selected_model_specs": [
                spec.to_manifest() for spec in self.selected_model_specs
            ],
            "model_selection_result": dict(self.model_selection_manifest),
            "similarity_calibration": self.similarity_calibration.to_manifest(),
        }


@dataclass(frozen=True)
class SharedPreparedRepetition:
    """One split and preprocessing fit reused by all feature policies."""

    bundle: DatasetBundle
    repetition: ExperimentRepetition
    prepared_dataset: PreparedDataset


def prepare_shared_repetition(
    bundle: DatasetBundle,
    repetition: ExperimentRepetition,
    *,
    test_size: float,
    stratify_protected: bool = True,
) -> SharedPreparedRepetition:
    """Split and preprocess once before branching into runtime policies."""

    split = make_stratified_split(
        bundle,
        test_size=test_size,
        random_state=repetition.seed_for("split"),
        stratify_protected=stratify_protected,
    )
    return SharedPreparedRepetition(
        bundle=bundle,
        repetition=repetition,
        prepared_dataset=preprocess_dataset(bundle, split),
    )


def prepare_policy_repetition(
    shared: SharedPreparedRepetition,
    setup: ExperimentSetup,
) -> PreparedRepetition:
    """Fit policy-specific tuning and similarity stages without preprocessing again."""

    prepared = shared.prepared_dataset
    split = prepared.split
    if setup.test_size != split.test_size:
        raise ValueError("setup test_size must match the shared prepared split")
    matrix = prepared.experiment_matrix(setup.protected_policy)
    continuous = prepared.experiment_continuous_indices(setup.protected_policy)
    if isinstance(setup.model_selection, FixedModelSelection):
        selected_specs = setup.model_selection.model_specs
        selection_manifest = setup.model_selection.to_manifest()
    else:
        tuning: BayesianFamilyTuningRun = tune_model_family_bayesian(
            matrix[split.train_indices],
            prepared.y[split.train_indices],
            search_spaces=setup.model_selection.search_spaces,
            optimization=setup.model_selection.optimization,
            random_state=shared.repetition.seed_for("tuning"),
            n_jobs=setup.n_jobs,
        )
        selected_specs = tuning.selected_model_specs
        selection_manifest = tuning.to_manifest()
    similarity = fit_similarity_calibration(
        matrix[split.train_indices], setup.graph_fitting_spec
    )
    return PreparedRepetition(
        bundle=shared.bundle,
        repetition=shared.repetition,
        setup=setup,
        prepared_dataset=prepared,
        experiment_matrix=matrix,
        continuous_indices=continuous,
        selected_model_specs=selected_specs,
        model_selection_manifest=MappingProxyType(selection_manifest),
        similarity_calibration=similarity,
    )


@dataclass(frozen=True)
class ConditionResult:
    """Complete test-population output for one condition and repetition."""

    condition: str
    condition_variant: str
    target: InjectionTarget
    requested_injection_ratio: float
    realized_injection_ratio: float
    assessment: IFCredAssessment
    coverage: CoverageResult
    distance_reliability: FloatArray
    graph: AuditGraph
    model_run: ModelFamilyRun
    evaluation_labels: NDArray
    injected_pairs: tuple[InjectedPair, ...] = ()
    condition_metadata: Mapping[str, Any] = field(default_factory=dict)
    prediction_controls: Mapping[str, Any] = field(default_factory=dict)
    training_features_by_model: Mapping[str, FloatArray] = field(
        default_factory=dict, repr=False, compare=False
    )
    training_labels_by_model: Mapping[str, NDArray] = field(
        default_factory=dict, repr=False, compare=False
    )
    evaluation_features: FloatArray | None = field(
        default=None, repr=False, compare=False
    )

    def to_manifest(self) -> dict[str, Any]:
        assessment = {
            "metric_name": self.assessment.metric_name,
            "C": self.assessment.C,
            "D": self.assessment.D,
            "F": self.assessment.F,
            "M": self.assessment.M,
            "V": self.assessment.V,
            "F_min": self.assessment.F_min,
            "V_worst": self.assessment.V_worst,
            "model_fairness": dict(self.assessment.model_fairness),
        }
        utility = {
            name: {
                "native": asdict(output.native_utility),
                "calibrated": asdict(output.calibrated_utility),
            }
            for name, output in self.model_run.models.items()
        }
        return {
            "condition": self.condition,
            "condition_variant": self.condition_variant,
            "target": self.target.value,
            "requested_injection_ratio": self.requested_injection_ratio,
            "realized_injection_ratio": self.realized_injection_ratio,
            "n_evaluation": len(self.evaluation_labels),
            "n_injected_train": len(
                [pair for pair in self.injected_pairs if pair.partition == "train"]
            ),
            "n_injected_test": len(
                [pair for pair in self.injected_pairs if pair.partition == "test"]
            ),
            "assessment": assessment,
            "predictive_utility": utility,
            "model_run": self.model_run.to_manifest(),
            "graph_spec": self.graph.spec.to_manifest(),
            "graph_bandwidths": dict(self.graph.bandwidths),
            "injected_pairs": [asdict(pair) for pair in self.injected_pairs],
            "condition_metadata": dict(self.condition_metadata),
            "prediction_controls": dict(self.prediction_controls),
        }


@dataclass(frozen=True)
class PreparedConditionDefinition:
    """Nested severity conditions generated once per repetition."""

    name: str
    target: InjectionTarget
    injection_ratios: tuple[float, ...]
    factory: InjectionFactory
    seed_stream: str | None = None
    variant: str = "default"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.variant.strip() or not callable(self.factory):
            raise ValueError("condition definition requires a name and factory")
        ratios = np.asarray(self.injection_ratios, dtype=float)
        if ratios.ndim != 1 or ratios.size == 0:
            raise ValueError("injection_ratios must be a non-empty sequence")
        if not np.all(np.isfinite(ratios)) or np.any((ratios <= 0.0) | (ratios > 1.0)):
            raise ValueError("injection ratios must lie in (0, 1]")
        if np.any(np.diff(ratios) <= 0.0):
            raise ValueError("injection ratios must be strictly increasing")

    @property
    def resolved_seed_stream(self) -> str:
        return self.name if self.seed_stream is None else self.seed_stream

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variant": self.variant,
            "target": self.target.value,
            "injection_ratios": list(self.injection_ratios),
            "seed_stream": self.resolved_seed_stream,
            "factory": f"{self.factory.__module__}.{self.factory.__qualname__}",
        }


def prepare_experiment_repetition(
    bundle: DatasetBundle,
    repetition: ExperimentRepetition,
    setup: ExperimentSetup,
) -> PreparedRepetition:
    """Fit every clean-training-only stage once for one outer repetition."""

    shared = prepare_shared_repetition(
        bundle,
        repetition,
        test_size=setup.test_size,
        stratify_protected=setup.stratify_protected,
    )
    return prepare_policy_repetition(shared, setup)


def _row_ids(
    test_indices: NDArray[np.int64], injected_pairs: tuple[InjectedPair, ...]
) -> NDArray:
    real = [f"real:{int(index)}" for index in test_indices]
    synthetic = [
        f"synthetic:{pair.condition}:{pair.candidate_index}"
        for pair in injected_pairs
        if pair.partition == "test"
    ]
    return np.asarray([*real, *synthetic], dtype=str)


def _evaluate(
    context: PreparedRepetition,
    *,
    condition: str,
    condition_variant: str,
    target: InjectionTarget,
    X_test: ArrayLike,
    y_test: ArrayLike,
    model_run: ModelFamilyRun,
    requested_ratio: float,
    realized_ratio: float,
    injected_pairs: tuple[InjectedPair, ...],
    condition_metadata: Mapping[str, Any] | None = None,
    training_features_by_model: Mapping[str, ArrayLike] | None = None,
    training_labels_by_model: Mapping[str, ArrayLike] | None = None,
    probability_source: str = "calibrated",
) -> ConditionResult:
    row_ids = _row_ids(context.prepared_dataset.split.test_indices, injected_pairs)
    graph = context.similarity_calibration.build_evaluation_graph(
        X_test, row_ids=row_ids
    )
    C = coverage(graph.weights, intended_k=graph.spec.k)
    D, reliability = distance_stability(graph.weights, graph.similarity_tensor())
    if probability_source == "calibrated":
        probabilities = model_run.calibrated_probabilities_by_model
    elif probability_source == "native":
        probabilities = model_run.native_probabilities_by_model
    else:
        raise ValueError("probability_source must be 'calibrated' or 'native'")
    fairness = context.setup.fairness.evaluate(probabilities, graph)
    assessment = assess_ifcred(C=C.score, D=D, fairness=fairness)
    return ConditionResult(
        condition=condition,
        condition_variant=condition_variant,
        target=target,
        requested_injection_ratio=float(requested_ratio),
        realized_injection_ratio=float(realized_ratio),
        assessment=assessment,
        coverage=C,
        distance_reliability=reliability,
        graph=graph,
        model_run=model_run,
        evaluation_labels=np.asarray(y_test).copy(),
        injected_pairs=injected_pairs,
        condition_metadata=MappingProxyType(
            {"probability_source": probability_source, **dict(condition_metadata or {})}
        ),
        training_features_by_model=MappingProxyType(
            {
                name: np.asarray(values, dtype=float).copy()
                for name, values in (training_features_by_model or {}).items()
            }
        ),
        training_labels_by_model=MappingProxyType(
            {
                name: np.asarray(values).copy()
                for name, values in (training_labels_by_model or {}).items()
            }
        ),
        evaluation_features=np.asarray(X_test, dtype=float).copy(),
    )


def _common_model_run(
    context: PreparedRepetition, partitions: AugmentedPartitions
) -> ModelFamilyRun:
    return run_model_family(
        partitions.X_train,
        partitions.y_train,
        partitions.X_test,
        partitions.y_test,
        model_specs=context.selected_model_specs,
        calibration=context.setup.probability_calibration,
        random_state=context.repetition.seed_for("model"),
        n_jobs=context.setup.n_jobs,
        calibration_bins=context.setup.calibration_bins,
    )


def run_clean_condition(
    context: PreparedRepetition,
    *,
    model_run: ModelFamilyRun | None = None,
    condition_variant: str = "primary",
    probability_source: str = "calibrated",
) -> ConditionResult:
    split = context.prepared_dataset.split
    partitions = AugmentedPartitions(
        X_train=context.experiment_matrix[split.train_indices],
        y_train=context.prepared_dataset.y[split.train_indices],
        X_test=context.experiment_matrix[split.test_indices],
        y_test=context.prepared_dataset.y[split.test_indices],
        selected_candidate_indices=np.empty(0, dtype=np.int64),
        injected_pairs=(),
        requested_injection_ratio=0.0,
        realized_injection_ratio=0.0,
    )
    if model_run is None:
        model_run = _common_model_run(context, partitions)
    training_features = {
        name: partitions.X_train for name in model_run.models
    }
    training_labels = {name: partitions.y_train for name in model_run.models}
    result = _evaluate(
        context,
        condition="clean_baseline",
        condition_variant=condition_variant,
        target=InjectionTarget.BENIGN_CONTROL,
        X_test=partitions.X_test,
        y_test=partitions.y_test,
        model_run=model_run,
        requested_ratio=0.0,
        realized_ratio=0.0,
        injected_pairs=(),
        training_features_by_model=training_features,
        training_labels_by_model=training_labels,
        probability_source=probability_source,
    )
    if probability_source == "calibrated":
        from ifcred.experiments.controls import prediction_control_assessments

        controls = prediction_control_assessments(
            context,
            result.graph,
            result.assessment.C,
            result.assessment.D,
            partitions.X_test,
            partitions.y_test,
        )
        result = replace(
            result, prediction_controls=MappingProxyType(dict(controls))
        )
    return result


def run_prepared_injection_condition(
    context: PreparedRepetition,
    prepared_injection: PreparedInjection,
    *,
    injection_ratio: float,
    condition_variant: str = "default",
) -> ConditionResult:
    split = context.prepared_dataset.split
    partitions = assemble_real_anchored_injection(
        context.experiment_matrix,
        context.prepared_dataset.y,
        split.train_indices,
        split.test_indices,
        prepared_injection,
        injection_ratio=injection_ratio,
    )
    model_run = _common_model_run(context, partitions)
    metadata = {
        "selected_candidate_indices": partitions.selected_candidate_indices.tolist(),
    }
    if prepared_injection.target == InjectionTarget.DISTANCE_STABILITY:
        selected = partitions.selected_candidate_indices
        reliability = np.asarray(
            prepared_injection.candidate_metadata["pair_reliability"], dtype=float
        )[selected]
        target_met = np.asarray(
            prepared_injection.candidate_metadata["target_reliability_met"],
            dtype=bool,
        )[selected]
        metadata.update(
            {
                "selected_pair_reliability_mean": float(np.mean(reliability)),
                "selected_pair_reliability_max": float(np.max(reliability)),
                "selected_target_reliability_met_fraction": float(
                    np.mean(target_met)
                ),
            }
        )
    training_features = {
        name: partitions.X_train for name in model_run.models
    }
    training_labels = {name: partitions.y_train for name in model_run.models}
    return _evaluate(
        context,
        condition=prepared_injection.condition,
        condition_variant=condition_variant,
        target=prepared_injection.target,
        X_test=partitions.X_test,
        y_test=partitions.y_test,
        model_run=model_run,
        requested_ratio=partitions.requested_injection_ratio,
        realized_ratio=partitions.realized_injection_ratio,
        injected_pairs=partitions.injected_pairs,
        condition_metadata=metadata,
        training_features_by_model=training_features,
        training_labels_by_model=training_labels,
    )


def run_model_disagreement_condition(
    context: PreparedRepetition,
    case: ModelFamilyDisagreementCase,
    *,
    condition_variant: str = "default",
) -> ConditionResult:
    training = {
        name: ModelTrainingData(partition.X_train, partition.y_train)
        for name, partition in case.training_by_model.items()
    }
    model_run = run_model_family_partitioned(
        training,
        case.X_test,
        case.y_test,
        model_specs=context.selected_model_specs,
        calibration=context.setup.probability_calibration,
        random_state=context.repetition.seed_for("model"),
        n_jobs=context.setup.n_jobs,
        calibration_bins=context.setup.calibration_bins,
    )
    all_pairs = tuple(
        pair
        for partition in case.training_by_model.values()
        for pair in partition.train_pairs
    ) + case.test_pairs
    metadata = {
        "model_exposure_assignments": dict(case.plan.assignments),
        "requested_affected_fraction": case.plan.requested_affected_fraction,
        "realized_affected_fraction": case.plan.realized_affected_fraction,
        "allocation_order": list(case.plan.allocation_order),
    }
    training_features = {
        name: case.training_by_model[name].X_train for name in model_run.models
    }
    training_labels = {
        name: case.training_by_model[name].y_train for name in model_run.models
    }
    return _evaluate(
        context,
        condition="model_family_disagreement",
        condition_variant=condition_variant,
        target=InjectionTarget.MODEL_STABILITY,
        X_test=case.X_test,
        y_test=case.y_test,
        model_run=model_run,
        requested_ratio=case.injection_ratio,
        realized_ratio=case.injection_ratio,
        injected_pairs=all_pairs,
        condition_metadata=metadata,
        training_features_by_model=training_features,
        training_labels_by_model=training_labels,
    )


def run_prepared_condition_definitions(
    context: PreparedRepetition,
    definitions: tuple[PreparedConditionDefinition, ...],
    *,
    include_clean: bool = True,
) -> tuple[ConditionResult, ...]:
    results: list[ConditionResult] = []
    if include_clean:
        results.append(run_clean_condition(context))
    for definition in definitions:
        seed = context.repetition.injection_seed(definition.resolved_seed_stream)
        prepared = definition.factory(context, seed)
        if prepared.condition != definition.name or prepared.target != definition.target:
            raise ValueError("condition factory output does not match its definition")
        for ratio in definition.injection_ratios:
            results.append(
                run_prepared_injection_condition(
                    context,
                    prepared,
                    injection_ratio=ratio,
                    condition_variant=definition.variant,
                )
            )
    return tuple(results)


def iter_repeated_experiment(
    bundle: DatasetBundle,
    plan: RepetitionPlan,
    setup: ExperimentSetup,
    definitions: tuple[PreparedConditionDefinition, ...],
) -> Iterator[tuple[PreparedRepetition, ConditionResult]]:
    """Yield bounded-memory outputs for every paired repeated condition."""

    for repetition in plan.repetitions:
        context = prepare_experiment_repetition(bundle, repetition, setup)
        for result in run_prepared_condition_definitions(context, definitions):
            yield context, result
