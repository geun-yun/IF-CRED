from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ifcred.data import fetch_uci_dataset, get_dataset_spec, load_cached_dataset


def fake_remote(features: pd.DataFrame, targets: pd.DataFrame):
    return SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))


def adult_frames(n=12):
    spec = get_dataset_spec("D6")
    values = {
        "age": np.arange(25, 25 + n),
        "workclass": np.tile(["Private", " ? "], n // 2),
        "fnlwgt": np.arange(100_000, 100_000 + n),
        "education": np.tile(["HS-grad", "Bachelors"], n // 2),
        "education-num": np.tile([9, 13], n // 2),
        "marital-status": np.tile(["Never-married", "Married-civ-spouse"], n // 2),
        "occupation": np.tile(["Sales", "Tech-support"], n // 2),
        "relationship": np.tile(["Not-in-family", "Husband"], n // 2),
        "race": np.tile(["Black", "White"], n // 2),
        "sex": np.tile(["Female", "Male"], n // 2),
        "capital-gain": np.zeros(n, dtype=int),
        "capital-loss": np.zeros(n, dtype=int),
        "hours-per-week": np.tile([35, 40], n // 2),
        "native-country": np.repeat("United-States", n),
    }
    features = pd.DataFrame(values).loc[:, spec.expected_features]
    targets = pd.DataFrame(
        {"income": np.tile(["<=50K", ">50K."], n // 2)}
    )
    return features, targets


def test_adult_source_values_are_retained_as_received():
    features, targets = adult_frames()
    bundle = fetch_uci_dataset(
        "D6", fetcher=lambda **kwargs: fake_remote(features, targets)
    )

    assert bundle.target.tolist() == ["<=50K", ">50K."] * 6
    assert bundle.features["workclass"].tolist()[1] == " ? "
    assert bundle.features["workclass"].isna().sum() == 0
    assert tuple(bundle.protected.columns) == ("sex", "race")
    assert bundle.manifest["uci_id"] == 2


def test_credit_and_cleveland_source_targets_are_not_binarized():
    credit_spec = get_dataset_spec("D7")
    credit_features = pd.DataFrame(
        {name: np.arange(8) + position for position, name in enumerate(credit_spec.expected_features)}
    )
    credit_features["X2"] = np.tile([1, 2], 4)
    credit = fetch_uci_dataset(
        "D7",
        fetcher=lambda **kwargs: fake_remote(
            credit_features, pd.DataFrame({"Y": [0, 1] * 4})
        ),
    )
    assert credit.target.tolist() == [0, 1] * 4
    assert set(credit.protected["X2"]) == {1, 2}

    heart_spec = get_dataset_spec("D8")
    heart_features = pd.DataFrame(
        {name: np.arange(8) + position for position, name in enumerate(heart_spec.expected_features)}
    )
    heart_features["sex"] = np.tile([0, 1], 4)
    heart_features.loc[0, "ca"] = np.nan
    heart = fetch_uci_dataset(
        "D8",
        fetcher=lambda **kwargs: fake_remote(
            heart_features, pd.DataFrame({"num": [0, 1, 2, 3, 4, 0, 2, 0]})
        ),
    )
    assert heart.target.tolist() == [0, 1, 2, 3, 4, 0, 2, 0]
    assert pd.isna(heart.features.loc[0, "ca"])


def test_immutable_cache_round_trip_verifies_checksum(tmp_path):
    features, targets = adult_frames()
    original = fetch_uci_dataset(
        "D6",
        cache_root=tmp_path,
        fetcher=lambda **kwargs: fake_remote(features, targets),
    )
    cached = load_cached_dataset("D6", tmp_path)

    pd.testing.assert_frame_equal(cached.features, original.features, check_dtype=False)
    pd.testing.assert_series_equal(cached.target, original.target, check_dtype=False)
    assert cached.manifest["source_sha256"] == original.manifest["source_sha256"]


def test_schema_drift_is_rejected():
    features, targets = adult_frames()
    features = features.drop(columns=["age"])
    with pytest.raises(ValueError, match="schema mismatch"):
        fetch_uci_dataset(
            "D6", fetcher=lambda **kwargs: fake_remote(features, targets)
        )
