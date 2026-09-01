import numpy as np
import pandas as pd

from ifcred.comparisons.base import ComparisonCase
from ifcred.comparisons.runner import run_prior_frameworks
from ifcred.reporting.loader import load_comparison_outputs


class _ThresholdModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probability = 1.0 / (1.0 + np.exp(-np.asarray(X)[:, 0]))
        return np.column_stack((1.0 - probability, probability))

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= 0.0).astype(int)


def _case():
    X = np.array([[-1.0, 0.0], [-0.2, 1.0], [0.2, 0.0], [1.0, 1.0]])
    return ComparisonCase(
        dataset_id="D-test",
        condition="clean_baseline",
        variant="primary",
        ratio=0.0,
        repetition=0,
        X_train=X,
        y_train=np.array([0, 0, 1, 1]),
        X_test=X,
        y_test=np.array([0, 0, 1, 1]),
        protected_index=1,
        random_seed=7,
    )


def test_comparator_labels_follow_publication_chronology():
    rows = run_prior_frameworks(_ThresholdModel(), _case())
    by_framework = {row["framework"]: row for row in rows}

    assert list(by_framework) == ["VF1", "VF2", "VF3_IFT_V"]
    assert "bias_found" in by_framework["VF1"]
    assert "lower_confidence_statistic" in by_framework["VF2"]
    assert "valid_idis" in by_framework["VF3_IFT_V"]
    assert {
        row["comparison_label_version"] for row in rows
    } == {"chronological_2020_2021_2024"}


def test_legacy_saved_labels_are_swapped_only_when_unversioned(tmp_path):
    legacy = tmp_path / "legacy" / "comparisons.csv"
    legacy.parent.mkdir()
    pd.DataFrame(
        [
            {"framework": "VF1", "lower_confidence_statistic": 1.4},
            {"framework": "VF2", "bias_found": True},
            {"framework": "VF3_IFT_V", "valid_idis": 3},
        ]
    ).to_csv(legacy, index=False)

    loaded = load_comparison_outputs(tmp_path)

    assert loaded.loc[loaded.lower_confidence_statistic.notna(), "framework"].item() == "VF2"
    assert loaded.loc[loaded.bias_found.notna(), "framework"].item() == "VF1"
    assert loaded.loc[loaded.valid_idis.notna(), "framework"].item() == "VF3_IFT_V"


def test_versioned_saved_labels_are_not_swapped(tmp_path):
    current = tmp_path / "current" / "comparisons.csv"
    current.parent.mkdir()
    pd.DataFrame(
        [
            {
                "framework": "VF1",
                "bias_found": True,
                "comparison_label_version": "chronological_2020_2021_2024",
            },
            {
                "framework": "VF2",
                "lower_confidence_statistic": 1.4,
                "comparison_label_version": "chronological_2020_2021_2024",
            },
        ]
    ).to_csv(current, index=False)

    loaded = load_comparison_outputs(tmp_path)

    assert loaded.loc[loaded.bias_found.notna(), "framework"].item() == "VF1"
    assert loaded.loc[loaded.lower_confidence_statistic.notna(), "framework"].item() == "VF2"
