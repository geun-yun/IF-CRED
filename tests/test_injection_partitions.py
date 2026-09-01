import numpy as np
import pytest

from ifcred.experiments.injections import (
    InjectionTarget,
    PreparedInjection,
    assemble_real_anchored_injection,
)


def prepared_candidates() -> PreparedInjection:
    return PreparedInjection(
        condition="example",
        target=InjectionTarget.FAIRNESS,
        synthetic_features=np.arange(12, dtype=float).reshape(6, 2) + 100.0,
        synthetic_labels=np.array([1, 0, 1, 0, 1, 0]),
        anchor_indices=np.arange(6),
        priority_order=np.array([4, 1, 5, 0, 3, 2]),
        known_unfair=np.array([True, True, True, True, True, True]),
    )


def test_synthetic_rows_follow_anchor_partitions_and_keep_pair_provenance():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    result = assemble_real_anchored_injection(
        X,
        y,
        train_indices=[0, 1, 2, 3],
        test_indices=[4, 5],
        prepared=prepared_candidates(),
        injection_ratio=0.5,
    )

    assert result.selected_candidate_indices.tolist() == [4, 1, 5]
    assert result.X_train.shape == (5, 2)
    assert result.X_test.shape == (4, 2)
    assert len(result.train_pairs) == 1
    assert len(result.test_pairs) == 2
    assert result.train_pairs[0].anchor_original_index == 1
    assert result.test_pairs[0].anchor_original_index == 4
    assert result.test_pairs[0].synthetic_partition_index == 2


def test_injection_levels_are_nested_prefixes():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6) % 2
    low = assemble_real_anchored_injection(
        X, y, [0, 1, 2, 3], [4, 5], prepared_candidates(), injection_ratio=2 / 6
    )
    high = assemble_real_anchored_injection(
        X, y, [0, 1, 2, 3], [4, 5], prepared_candidates(), injection_ratio=4 / 6
    )

    assert set(low.selected_candidate_indices).issubset(set(high.selected_candidate_indices))


def test_anchor_and_synthetic_counterpart_cannot_cross_partitions():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6) % 2
    result = assemble_real_anchored_injection(
        X, y, [0, 1, 2], [3, 4, 5], prepared_candidates(), injection_ratio=1.0
    )

    train_anchors = {0, 1, 2}
    test_anchors = {3, 4, 5}
    assert all(pair.anchor_original_index in train_anchors for pair in result.train_pairs)
    assert all(pair.anchor_original_index in test_anchors for pair in result.test_pairs)


def test_overlapping_partitions_are_rejected():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6) % 2

    with pytest.raises(ValueError, match="disjoint"):
        assemble_real_anchored_injection(
            X, y, [0, 1, 2, 3], [3, 4, 5], prepared_candidates(), injection_ratio=0.5
        )


def test_ratio_requires_enough_prepared_candidates():
    prepared = PreparedInjection(
        condition="limited",
        target=InjectionTarget.BENIGN_CONTROL,
        synthetic_features=np.ones((2, 2)),
        synthetic_labels=np.array([0, 1]),
        anchor_indices=np.array([0, 1]),
        priority_order=np.array([0, 1]),
        known_unfair=np.array([False, False]),
    )
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6) % 2

    with pytest.raises(ValueError, match="enough candidates"):
        assemble_real_anchored_injection(
            X, y, [0, 1, 2], [3, 4, 5], prepared, injection_ratio=1.0
        )
