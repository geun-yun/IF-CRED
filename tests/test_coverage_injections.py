import numpy as np
import pytest

from ifcred.experiments.injections import (
    assemble_real_anchored_injection,
    prepare_dominant_neighbour_pairs,
    prepare_isolated_instances,
)


def data():
    rng = np.random.default_rng(12)
    continuous = rng.normal(size=(30, 2))
    protected = np.tile([0.0, 1.0], 15)[:, None]
    X = np.column_stack((continuous, protected))
    y = (continuous[:, 0] > 0).astype(int)
    return X, y


def test_isolated_instances_meet_similarity_bound_and_preserve_other_columns():
    X, y = data()
    prepared = prepare_isolated_instances(
        X,
        y,
        legitimate_continuous_indices=[0, 1],
        k=3,
        maximum_background_similarity=0.1,
        random_state=4,
    )

    assert np.all(prepared.candidate_metadata["maximum_background_similarity"] <= 0.1 + 1e-12)
    np.testing.assert_array_equal(prepared.synthetic_features[:, 2], X[:, 2])
    np.testing.assert_array_equal(prepared.synthetic_labels, y)


def test_dominant_pairs_are_strong_within_pair_and_weak_to_background():
    X, y = data()
    prepared = prepare_dominant_neighbour_pairs(
        X,
        y,
        legitimate_continuous_indices=[0, 1],
        k=3,
        minimum_pair_similarity=0.95,
        maximum_background_similarity=0.1,
        random_state=5,
        eligible_anchor_indices=[0, 1, 2, 3],
    )

    assert len(prepared.synthetic_features) == 8
    assert np.all(prepared.candidate_metadata["within_pair_similarity"] >= 0.95 - 1e-12)
    assert np.all(prepared.candidate_metadata["maximum_background_similarity"] <= 0.1 + 1e-12)
    np.testing.assert_array_equal(prepared.synthetic_features[:, 2], np.repeat(X[:4, 2], 2))


def test_dominant_pair_selection_never_splits_a_pair_group():
    X, y = data()
    prepared = prepare_dominant_neighbour_pairs(
        X,
        y,
        legitimate_continuous_indices=[0, 1],
        k=3,
        minimum_pair_similarity=0.9,
        maximum_background_similarity=0.2,
        random_state=7,
        eligible_anchor_indices=np.arange(10),
    )
    result = assemble_real_anchored_injection(
        X,
        y,
        train_indices=np.arange(20),
        test_indices=np.arange(20, 30),
        prepared=prepared,
        injection_ratio=0.1,
    )

    groups = prepared.selection_group_ids[result.selected_candidate_indices]
    _, counts = np.unique(groups, return_counts=True)
    np.testing.assert_array_equal(counts, 2)


def test_dominant_pair_requires_clear_similarity_separation():
    X, y = data()
    with pytest.raises(ValueError, match="must exceed"):
        prepare_dominant_neighbour_pairs(
            X,
            y,
            legitimate_continuous_indices=[0, 1],
            k=3,
            minimum_pair_similarity=0.2,
            maximum_background_similarity=0.5,
            random_state=1,
        )
