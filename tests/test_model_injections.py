import numpy as np

from ifcred.experiments.injections import (
    assemble_model_family_disagreement_case,
    prepare_model_family_injection_plan,
    prepare_paired_near_duplicates,
)


def data():
    X = np.column_stack((np.linspace(0, 9, 10), np.tile([0.0, 1.0], 5)))
    y = np.tile([0, 1], 5)
    return X, y


def test_model_allocations_are_nested_for_a_fixed_seed():
    names = ["linear", "tree", "kernel", "ensemble"]
    low = prepare_model_family_injection_plan(
        names, affected_fraction=0.25, random_state=5
    )
    high = prepare_model_family_injection_plan(
        names, affected_fraction=0.75, random_state=5
    )

    assert set(low.affected_models).issubset(set(high.affected_models))
    assert low.allocation_order == high.allocation_order


def test_affected_models_receive_contradictions_and_share_common_test_features():
    X, y = data()
    paired = prepare_paired_near_duplicates(
        X,
        y,
        legitimate_continuous_indices=[0],
        k=1,
        radius_fraction=0.1,
        random_state=2,
    )
    plan = prepare_model_family_injection_plan(
        ["linear", "tree", "kernel", "ensemble"],
        affected_fraction=0.5,
        random_state=3,
    )
    case = assemble_model_family_disagreement_case(
        X,
        y,
        train_indices=np.arange(7),
        test_indices=np.arange(7, 10),
        paired_conditions=paired,
        plan=plan,
        injection_ratio=0.4,
    )

    benign_labels = {
        tuple(partition.y_train)
        for partition in case.training_by_model.values()
        if partition.exposure == "benign"
    }
    contradictory_labels = {
        tuple(partition.y_train)
        for partition in case.training_by_model.values()
        if partition.exposure == "contradictory"
    }
    assert len(benign_labels) == 1
    assert len(contradictory_labels) == 1
    assert benign_labels != contradictory_labels
    feature_matrices = [partition.X_train for partition in case.training_by_model.values()]
    assert all(np.array_equal(feature_matrices[0], matrix) for matrix in feature_matrices[1:])
    assert case.X_test.shape[0] >= 3


def test_zero_and_full_exposure_are_valid_control_endpoints():
    names = ["a", "b", "c"]
    none = prepare_model_family_injection_plan(names, affected_fraction=0.0, random_state=1)
    all_models = prepare_model_family_injection_plan(
        names, affected_fraction=1.0, random_state=1
    )

    assert len(none.affected_models) == 0
    assert len(all_models.affected_models) == 3
