"""Explicit exploratory study profiles; no hidden confirmatory defaults."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ifcred.core.graph import (
    AuditGraphSpec,
    BandwidthPolicy,
    DistanceMetricSpec,
)
from ifcred.data import ProtectedAttributePolicy
from ifcred.experiments.orchestrator import (
    BayesianModelSelection,
    ExperimentSetup,
    FixedModelSelection,
)
from ifcred.models import (
    BayesianOptimizationSpec,
    CalibrationSpec,
    CrossValidationSpec,
    ModelFamily,
    ModelSpec,
    recommended_bayesian_search_spaces,
)


METRICS = (
    DistanceMetricSpec("euclidean", "euclidean"),
    DistanceMetricSpec("manhattan", "manhattan"),
    DistanceMetricSpec("cosine", "cosine"),
)


@dataclass(frozen=True)
class AuditVariant:
    """One E1 audit-design setting."""

    name: str
    graph_spec: AuditGraphSpec


@dataclass(frozen=True)
class StudyConfig:
    """Complete execution surface for E1--E3."""

    profile: str
    dataset_ids: tuple[str, ...]
    policies: tuple[ProtectedAttributePolicy, ...]
    n_repetitions: int
    root_seed: int
    test_size: float
    audit_variants: tuple[AuditVariant, ...]
    injection_ratios: tuple[float, ...]
    near_duplicate_radii: tuple[float, ...]
    isolation_similarities: tuple[float, ...]
    dominant_background_similarities: tuple[float, ...]
    disagreement_reliabilities: tuple[float, ...]
    model_affected_fractions: tuple[float, ...]
    calibration: CalibrationSpec
    model_selection: FixedModelSelection | BayesianModelSelection
    n_jobs: int = 1
    development_fraction: float | None = None
    frozen_model_config_root: Path | None = None
    repetition_start: int = 0

    @property
    def repetition_numbers(self) -> tuple[int, ...]:
        return tuple(range(self.repetition_start, self.repetition_start + self.n_repetitions))

    @property
    def primary_audit(self) -> AuditVariant:
        return self.audit_variants[0]

    def setup(
        self,
        policy: ProtectedAttributePolicy,
        variant: AuditVariant | None = None,
        *,
        model_selection: FixedModelSelection | BayesianModelSelection | None = None,
    ) -> ExperimentSetup:
        selected = self.primary_audit if variant is None else variant
        return ExperimentSetup(
            protected_policy=policy,
            test_size=self.test_size,
            graph_fitting_spec=replace(selected.graph_spec, n_jobs=self.n_jobs),
            model_selection=(
                self.model_selection if model_selection is None else model_selection
            ),
            probability_calibration=self.calibration,
            n_jobs=self.n_jobs,
        )


def _graph(k: int, bandwidth: BandwidthPolicy, *, metrics=METRICS) -> AuditGraphSpec:
    return AuditGraphSpec(
        k=k,
        metrics=tuple(metrics),
        primary_metric="euclidean",
        bandwidth_policy=bandwidth,
    )


def _fixed_models() -> FixedModelSelection:
    return FixedModelSelection(
        (
            ModelSpec(ModelFamily.LOGISTIC_REGRESSION, {"max_iter": 1000}),
            ModelSpec(
                ModelFamily.MLP,
                {"hidden_layer_sizes": (32,), "max_iter": 300, "early_stopping": True},
            ),
            ModelSpec(ModelFamily.GAUSSIAN_NAIVE_BAYES),
            ModelSpec(
                ModelFamily.RANDOM_FOREST,
                {"n_estimators": 60, "max_depth": 8},
            ),
            ModelSpec(ModelFamily.DECISION_TREE, {"max_depth": 8}),
        )
    )


def exploratory_config(
    *, root_seed: int = 20260825, n_jobs: int = 1
) -> StudyConfig:
    """Full exploratory protocol intended to precede a frozen confirmatory run."""

    spaces = recommended_bayesian_search_spaces()
    return StudyConfig(
        profile="exploratory",
        dataset_ids=("D6", "D7", "D8"),
        policies=(
            ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,
            ProtectedAttributePolicy.EXCLUDE_PROTECTED,
        ),
        n_repetitions=30,
        root_seed=root_seed,
        test_size=0.30,
        audit_variants=(
            AuditVariant("primary_k10", _graph(10, BandwidthPolicy.MEDIAN_RETAINED_MAX)),
            AuditVariant("k5", _graph(5, BandwidthPolicy.MEDIAN_RETAINED_MAX)),
            AuditVariant("k20", _graph(20, BandwidthPolicy.MEDIAN_RETAINED_MAX)),
            AuditVariant("alternate_bandwidth", _graph(10, BandwidthPolicy.MEDIAN_POSITIVE_PAIR)),
            AuditVariant("euclidean_manhattan", _graph(10, BandwidthPolicy.MEDIAN_RETAINED_MAX, metrics=METRICS[:2])),
        ),
        injection_ratios=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        near_duplicate_radii=(0.01, 0.05, 0.15),
        isolation_similarities=(0.20, 0.05),
        dominant_background_similarities=(0.20, 0.05),
        disagreement_reliabilities=(0.90, 0.75),
        model_affected_fractions=(0.20, 0.60, 1.0),
        calibration=CalibrationSpec(folds=5),
        model_selection=BayesianModelSelection(
            spaces,
            BayesianOptimizationSpec(
                cross_validation=CrossValidationSpec(folds=5), startup_trials=10
            ),
        ),
        n_jobs=n_jobs,
    )


def smoke_config(*, root_seed: int = 20260825, n_jobs: int = 1) -> StudyConfig:
    """Fast end-to-end profile that exercises every pipeline layer."""

    return StudyConfig(
        profile="smoke",
        dataset_ids=("D8",),
        policies=(ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,),
        n_repetitions=2,
        root_seed=root_seed,
        test_size=0.30,
        audit_variants=(
            AuditVariant("primary_k5", _graph(5, BandwidthPolicy.MEDIAN_RETAINED_MAX)),
            AuditVariant("k3", _graph(3, BandwidthPolicy.MEDIAN_RETAINED_MAX)),
        ),
        injection_ratios=(0.02,),
        near_duplicate_radii=(0.05,),
        isolation_similarities=(0.10,),
        dominant_background_similarities=(0.10,),
        disagreement_reliabilities=(0.75,),
        model_affected_fractions=(0.60,),
        calibration=CalibrationSpec(folds=2),
        model_selection=_fixed_models(),
        n_jobs=n_jobs,
    )


def frozen_config(
    model_config_root: str | Path,
    *,
    root_seed: int = 20260825,
    n_jobs: int = 1,
    development_fraction: float = 0.20,
) -> StudyConfig:
    """Repeated-study profile using model settings selected on disjoint rows."""

    return replace(
        exploratory_config(root_seed=root_seed, n_jobs=n_jobs),
        profile="frozen",
        model_selection=_fixed_models(),  # replaced per dataset-policy at runtime
        development_fraction=development_fraction,
        frozen_model_config_root=Path(model_config_root).resolve(),
    )


def override_config(
    config: StudyConfig,
    *,
    dataset_ids: tuple[str, ...] | None = None,
    n_repetitions: int | None = None,
) -> StudyConfig:
    """Apply narrow CLI overrides while retaining a manifestable profile."""

    return replace(
        config,
        dataset_ids=config.dataset_ids if dataset_ids is None else dataset_ids,
        n_repetitions=(
            config.n_repetitions if n_repetitions is None else n_repetitions
        ),
    )
