import numpy as np
import pytest

from ifcred.experiments.injections import prepare_metric_disagreement_instances


def data():
    X = np.column_stack((np.linspace(0.0, 5.0, 12), np.tile([0.0, 1.0], 6)))
    y = np.tile([0, 1], 6)
    return X, y


def opposing_similarities(anchors, candidates):
    displacement = np.abs(candidates[:, 0] - anchors[:, 0])
    primary = np.exp(-displacement)
    return {"primary": primary, "opposing": 1.0 - primary}


def test_metric_disagreement_meets_declared_primary_and_reliability_targets():
    X, y = data()
    prepared = prepare_metric_disagreement_instances(
        X,
        y,
        legitimate_continuous_indices=[0],
        k=2,
        radius_fractions=[0.1, 0.5, 1.0, 2.0],
        directions_per_radius=2,
        similarity_evaluator=opposing_similarities,
        primary_metric="primary",
        minimum_primary_similarity=0.55,
        target_reliability=0.7,
        random_state=3,
    )

    metadata = prepared.candidate_metadata
    assert np.all(metadata["primary_similarity"] >= 0.55)
    assert np.all(metadata["pair_reliability"] <= 0.7)
    assert np.all(metadata["target_reliability_met"])
    ordered_reliability = metadata["pair_reliability"][prepared.priority_order]
    assert np.all(np.diff(ordered_reliability) >= 0.0)
    np.testing.assert_array_equal(prepared.synthetic_features[:, 1], X[:, 1])
    np.testing.assert_array_equal(prepared.synthetic_labels, y)


def test_metric_disagreement_can_report_unmet_target_exploratorily():
    X, y = data()

    def agreeing(anchors, candidates):
        similarity = np.exp(-np.abs(candidates[:, 0] - anchors[:, 0]))
        return {"a": similarity, "b": similarity}

    prepared = prepare_metric_disagreement_instances(
        X,
        y,
        legitimate_continuous_indices=[0],
        k=1,
        radius_fractions=[0.1, 0.2],
        directions_per_radius=1,
        similarity_evaluator=agreeing,
        primary_metric="a",
        minimum_primary_similarity=0.5,
        target_reliability=0.2,
        random_state=4,
        require_target=False,
    )

    assert not np.any(prepared.candidate_metadata["target_reliability_met"])
    np.testing.assert_allclose(prepared.candidate_metadata["pair_reliability"], 1.0)


def test_strict_metric_disagreement_excludes_unmet_anchors_without_fallback():
    X, y = data()

    def partly_feasible(anchors, candidates):
        primary = np.full(len(anchors), 0.9)
        other = np.where(anchors[:, 0] < 2.5, 0.1, 0.9)
        return {"primary": primary, "other": other}

    prepared = prepare_metric_disagreement_instances(
        X,
        y,
        legitimate_continuous_indices=[0],
        k=1,
        radius_fractions=[0.1],
        directions_per_radius=1,
        similarity_evaluator=partly_feasible,
        primary_metric="primary",
        minimum_primary_similarity=0.5,
        target_reliability=0.5,
        random_state=4,
        require_target=True,
    )

    assert len(prepared.anchor_indices) == 6
    assert np.all(prepared.anchor_indices < 6)
    assert np.all(prepared.candidate_metadata["target_reliability_met"])
    assert np.all(prepared.candidate_metadata["pair_reliability"] <= 0.5)


def test_metric_disagreement_excludes_infeasible_anchors_without_relaxing_threshold():
    X, y = data()

    def one_infeasible_anchor(anchors, candidates):
        similarity = np.exp(-np.abs(candidates[:, 0] - anchors[:, 0]))
        similarity[anchors[:, 0] == 0.0] = 0.0
        return {"primary": similarity, "other": similarity}

    prepared = prepare_metric_disagreement_instances(
        X,
        y,
        legitimate_continuous_indices=[0],
        k=1,
        radius_fractions=[0.1, 0.2],
        directions_per_radius=1,
        similarity_evaluator=one_infeasible_anchor,
        primary_metric="primary",
        minimum_primary_similarity=0.5,
        target_reliability=0.2,
        random_state=4,
        require_target=False,
    )

    assert 0 not in prepared.anchor_indices
    assert len(prepared.anchor_indices) == len(X) - 1
    assert np.all(prepared.candidate_metadata["primary_similarity"] >= 0.5)


def test_metric_disagreement_requires_multiple_metrics():
    X, y = data()
    with pytest.raises(ValueError, match="at least two"):
        prepare_metric_disagreement_instances(
            X,
            y,
            legitimate_continuous_indices=[0],
            k=1,
            radius_fractions=[0.1],
            directions_per_radius=1,
            similarity_evaluator=lambda anchors, candidates: {
                "only": np.ones(len(anchors))
            },
            primary_metric="only",
            minimum_primary_similarity=0.5,
            target_reliability=0.5,
            random_state=1,
        )
