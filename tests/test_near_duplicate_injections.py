import numpy as np
import pytest

from ifcred.experiments.injections import (
    InjectionTarget,
    assemble_real_anchored_injection,
    prepare_localized_subgroup_contradiction,
    prepare_paired_near_duplicates,
)


def example_data():
    X = np.array(
        [
            [0.0, 10.0, 0.0],
            [1.0, 11.0, 1.0],
            [2.0, 12.0, 0.0],
            [3.0, 13.0, 1.0],
            [4.0, 14.0, 0.0],
            [5.0, 15.0, 1.0],
        ]
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    return X, y


def test_paired_conditions_have_identical_geometry_and_order():
    X, y = example_data()
    pair = prepare_paired_near_duplicates(
        X,
        y,
        legitimate_continuous_indices=[0, 1],
        k=2,
        radius_fraction=0.1,
        random_state=17,
    )

    np.testing.assert_allclose(
        pair.benign.synthetic_features, pair.contradictory.synthetic_features
    )
    np.testing.assert_array_equal(pair.benign.anchor_indices, pair.contradictory.anchor_indices)
    np.testing.assert_array_equal(pair.benign.priority_order, pair.contradictory.priority_order)
    np.testing.assert_array_equal(pair.benign.synthetic_labels, y)
    np.testing.assert_array_equal(pair.contradictory.synthetic_labels, 1 - y)
    assert pair.benign.target == InjectionTarget.BENIGN_CONTROL
    assert pair.contradictory.target == InjectionTarget.FAIRNESS


def test_only_declared_continuous_features_are_perturbed():
    X, y = example_data()
    pair = prepare_paired_near_duplicates(
        X,
        y,
        legitimate_continuous_indices=[0],
        k=1,
        radius_fraction=0.2,
        random_state=3,
    )

    np.testing.assert_array_equal(pair.benign.synthetic_features[:, 1:], X[:, 1:])
    assert np.all(pair.benign.synthetic_features[:, 0] != X[:, 0])


def test_perturbation_radius_is_calibrated_to_local_kth_distance():
    X, y = example_data()
    pair = prepare_paired_near_duplicates(
        X,
        y,
        legitimate_continuous_indices=[0, 1],
        k=2,
        radius_fraction=0.25,
        random_state=9,
    )

    metadata = pair.benign.candidate_metadata
    np.testing.assert_allclose(
        metadata["perturbation_radius"], 0.25 * metadata["anchor_kth_distance"]
    )
    actual = np.linalg.norm(pair.benign.synthetic_features[:, :2] - X[:, :2], axis=1)
    np.testing.assert_allclose(actual, metadata["perturbation_radius"])


def test_eligible_anchors_support_localized_conditions():
    X, y = example_data()
    pair = prepare_paired_near_duplicates(
        X,
        y,
        legitimate_continuous_indices=[0, 1],
        eligible_anchor_indices=[1, 3, 5],
        k=1,
        radius_fraction=0.1,
        random_state=4,
    )

    np.testing.assert_array_equal(pair.benign.anchor_indices, [1, 3, 5])
    np.testing.assert_array_equal(pair.contradictory.synthetic_labels, 1 - y[[1, 3, 5]])


def test_paired_conditions_assemble_with_the_same_selected_anchors():
    X, y = example_data()
    pair = prepare_paired_near_duplicates(
        X,
        y,
        legitimate_continuous_indices=[0, 1],
        k=1,
        radius_fraction=0.1,
        random_state=6,
    )
    benign = assemble_real_anchored_injection(
        X, y, [0, 1, 2, 3], [4, 5], pair.benign, injection_ratio=0.5
    )
    contradictory = assemble_real_anchored_injection(
        X, y, [0, 1, 2, 3], [4, 5], pair.contradictory, injection_ratio=0.5
    )

    np.testing.assert_array_equal(
        benign.selected_candidate_indices, contradictory.selected_candidate_indices
    )
    np.testing.assert_allclose(benign.X_train, contradictory.X_train)
    np.testing.assert_allclose(benign.X_test, contradictory.X_test)
    assert not np.array_equal(benign.y_train, contradictory.y_train)


def test_nonbinary_labels_are_rejected():
    X, _ = example_data()
    with pytest.raises(ValueError, match="binary labels"):
        prepare_paired_near_duplicates(
            X,
            np.arange(len(X)),
            legitimate_continuous_indices=[0],
            k=1,
            radius_fraction=0.1,
            random_state=2,
        )


def test_localized_condition_only_uses_declared_subgroup_anchors():
    X, y = example_data()
    subgroup = np.array([False, True, False, True, False, True])
    prepared = prepare_localized_subgroup_contradiction(
        X,
        y,
        subgroup_mask=subgroup,
        legitimate_continuous_indices=[0, 1],
        k=1,
        radius_fraction=0.1,
        random_state=8,
    )

    assert prepared.condition == "localized_subgroup_contradiction"
    np.testing.assert_array_equal(prepared.anchor_indices, [1, 3, 5])
    np.testing.assert_array_equal(prepared.synthetic_labels, 1 - y[[1, 3, 5]])


def test_localized_condition_rejects_unsupported_subgroup():
    X, y = example_data()
    with pytest.raises(ValueError, match="minimum_subgroup_size"):
        prepare_localized_subgroup_contradiction(
            X,
            y,
            subgroup_mask=np.array([True, False, False, False, False, False]),
            legitimate_continuous_indices=[0],
            k=1,
            radius_fraction=0.1,
            random_state=8,
            minimum_subgroup_size=2,
        )
