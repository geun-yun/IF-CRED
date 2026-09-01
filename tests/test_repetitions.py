import json

import pytest

from ifcred.experiments import (
    RECOMMENDED_REPETITIONS,
    STANDARD_SEED_ROLES,
    ExperimentRepetition,
    RepetitionPlan,
)


def test_recommended_plan_contains_thirty_distinct_repetitions():
    plan = RepetitionPlan(root_seed=20260825)

    assert plan.n_repetitions == RECOMMENDED_REPETITIONS == 30
    assert len(plan.repetitions) == 30
    split_seeds = [run.seed_for("split") for run in plan.repetitions]
    assert len(set(split_seeds)) == 30


def test_named_seed_streams_are_distinct_and_order_independent():
    run = ExperimentRepetition(repetition=4, root_seed=91)
    reverse = [run.seed_for(role) for role in reversed(STANDARD_SEED_ROLES)]
    forward = [run.seed_for(role) for role in STANDARD_SEED_ROLES]

    assert reverse == list(reversed(forward))
    assert len(set(forward)) == len(STANDARD_SEED_ROLES)
    assert all(0 <= seed < 2**32 for seed in forward)


def test_injection_severities_can_share_paired_random_draws():
    run = ExperimentRepetition(repetition=2, root_seed=15)

    low_severity = run.injection_seed("contradictory_near_duplicate")
    high_severity = run.injection_seed("contradictory_near_duplicate")
    other_type = run.injection_seed("isolated_instance")

    assert low_severity == high_severity
    assert low_severity != other_type


def test_comparators_have_stable_distinct_streams_within_each_repetition():
    run = ExperimentRepetition(repetition=2, root_seed=15)

    assert run.comparator_seed("VF1") == run.comparator_seed("VF1")
    assert run.comparator_seed("VF1") != run.comparator_seed("VF2")
    assert run.comparator_seed("VF1") != run.seed_for("model")


def test_same_plan_is_exactly_reproducible_and_serializable():
    first = RepetitionPlan(root_seed=700, n_repetitions=5)
    second = RepetitionPlan(root_seed=700, n_repetitions=5)

    assert first.to_manifest() == second.to_manifest()
    json.dumps(first.to_manifest())


def test_different_root_seed_changes_every_standard_stream():
    first = ExperimentRepetition(repetition=0, root_seed=1).standard_seeds()
    second = ExperimentRepetition(repetition=0, root_seed=2).standard_seeds()

    assert all(first[role] != second[role] for role in STANDARD_SEED_ROLES)


def test_repeated_plan_requires_multiple_runs():
    with pytest.raises(ValueError, match="at least two"):
        RepetitionPlan(root_seed=1, n_repetitions=1)
