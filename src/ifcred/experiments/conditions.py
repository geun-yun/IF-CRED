"""Factories for the declared E2 controlled synthetic-instance mechanisms."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil

from ifcred.experiments.injections import (
    InjectionTarget,
    PreparedInjection,
    prepare_dominant_neighbour_pairs,
    prepare_isolated_instances,
    prepare_metric_disagreement_instances,
    prepare_paired_near_duplicates,
)
from ifcred.experiments.orchestrator import (
    PreparedConditionDefinition,
    PreparedRepetition,
)
from ifcred.experiments.study_config import StudyConfig


def _near_factory(radius: float, kind: str) -> Callable:
    def factory(context: PreparedRepetition, seed: int) -> PreparedInjection:
        paired = prepare_paired_near_duplicates(
            context.experiment_matrix,
            context.prepared_dataset.y,
            legitimate_continuous_indices=context.continuous_indices,
            k=context.setup.graph_fitting_spec.k,
            radius_fraction=radius,
            random_state=seed,
        )
        return paired.benign if kind == "benign" else paired.contradictory

    return factory


def _isolated_factory(maximum_similarity: float) -> Callable:
    def factory(context: PreparedRepetition, seed: int) -> PreparedInjection:
        return prepare_isolated_instances(
            context.experiment_matrix,
            context.prepared_dataset.y,
            legitimate_continuous_indices=context.continuous_indices,
            k=context.setup.graph_fitting_spec.k,
            maximum_background_similarity=maximum_similarity,
            random_state=seed,
        )

    return factory


def _dominant_factory(maximum_similarity: float) -> Callable:
    def factory(context: PreparedRepetition, seed: int) -> PreparedInjection:
        return prepare_dominant_neighbour_pairs(
            context.experiment_matrix,
            context.prepared_dataset.y,
            legitimate_continuous_indices=context.continuous_indices,
            k=context.setup.graph_fitting_spec.k,
            minimum_pair_similarity=0.95,
            maximum_background_similarity=maximum_similarity,
            random_state=seed,
        )

    return factory


def _distance_factory(
    target_reliability: float, *, minimum_candidate_fraction: float
) -> Callable:
    def factory(context: PreparedRepetition, seed: int) -> PreparedInjection:
        prepared = prepare_metric_disagreement_instances(
            context.experiment_matrix,
            context.prepared_dataset.y,
            legitimate_continuous_indices=context.continuous_indices,
            k=context.setup.graph_fitting_spec.k,
            radius_fractions=(0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.0, 3.0, 4.0),
            directions_per_radius=8,
            similarity_evaluator=context.similarity_calibration.evaluate_pairs,
            primary_metric=context.setup.graph_fitting_spec.primary_metric,
            minimum_primary_similarity=0.50,
            target_reliability=target_reliability,
            random_state=seed,
            require_target=True,
        )
        required = ceil(minimum_candidate_fraction * len(context.experiment_matrix))
        if len(prepared.synthetic_features) < required:
            raise ValueError(
                "strict metric-disagreement bank cannot support the largest "
                f"injection ratio: found {len(prepared.synthetic_features)} "
                f"target-meeting candidates, require {required}"
            )
        return prepared

    return factory


def standard_injection_definitions(
    config: StudyConfig,
) -> tuple[PreparedConditionDefinition, ...]:
    """Build E2 definitions; ratios are nested within every severity bank."""

    definitions: list[PreparedConditionDefinition] = []
    for radius in config.near_duplicate_radii:
        label = f"radius_{radius:g}"
        for kind, target in (
            ("benign", InjectionTarget.BENIGN_CONTROL),
            ("contradictory", InjectionTarget.FAIRNESS),
        ):
            definitions.append(
                PreparedConditionDefinition(
                    name=f"{kind}_near_duplicate",
                    variant=label,
                    target=target,
                    injection_ratios=config.injection_ratios,
                    factory=_near_factory(radius, kind),
                    seed_stream=f"paired_near_duplicate:{label}",
                )
            )
    for similarity in config.isolation_similarities:
        definitions.append(
            PreparedConditionDefinition(
                name="isolated_instance",
                variant=f"max_similarity_{similarity:g}",
                target=InjectionTarget.COVERAGE,
                injection_ratios=config.injection_ratios,
                factory=_isolated_factory(similarity),
            )
        )
    for similarity in config.dominant_background_similarities:
        definitions.append(
            PreparedConditionDefinition(
                name="dominant_neighbour_pair",
                variant=f"background_similarity_{similarity:g}",
                target=InjectionTarget.COVERAGE,
                injection_ratios=config.injection_ratios,
                factory=_dominant_factory(similarity),
            )
        )
    for reliability in config.disagreement_reliabilities:
        definitions.append(
            PreparedConditionDefinition(
                name="metric_disagreement_instance",
                # Version the corrected selection rule so immutable results
                # produced with the weaker pre-fix candidate choice are not
                # mistaken for corrected S5 outputs.
                variant=f"strict_target_reliability_v2_{reliability:g}",
                target=InjectionTarget.DISTANCE_STABILITY,
                injection_ratios=config.injection_ratios,
                factory=_distance_factory(
                    reliability,
                    minimum_candidate_fraction=max(config.injection_ratios),
                ),
            )
        )
    return tuple(definitions)
