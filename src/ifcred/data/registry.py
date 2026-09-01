"""Fixed UCI dataset registry and exploratory feature-role declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset identity plus explicit, reviewable feature roles."""

    dataset_id: str
    name: str
    domain: str
    uci_id: int
    uci_url: str
    doi: str
    outcome_source: str
    outcome_description: str
    primary_protected: str
    secondary_protected: tuple[str, ...]
    expected_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    shared_features: tuple[str, ...]
    shared_continuous_features: tuple[str, ...]
    excluded_feature_reasons: Mapping[str, str]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = set(self.expected_features)
        typed = set(self.numeric_features) | set(self.categorical_features)
        if typed != expected or set(self.numeric_features) & set(self.categorical_features):
            raise ValueError("numeric and categorical features must partition expected_features")
        for name, values in {
            "shared_features": self.shared_features,
            "shared_continuous_features": self.shared_continuous_features,
        }.items():
            if not set(values).issubset(expected):
                raise ValueError(f"{name} contains an unknown feature")
        if self.primary_protected not in expected:
            raise ValueError("primary_protected must be an expected feature")
        if self.primary_protected in self.shared_features:
            raise ValueError("shared_features must exclude the primary protected attribute")
        if not set(self.shared_continuous_features).issubset(self.numeric_features):
            raise ValueError("shared_continuous_features must be numeric")
        if not set(self.shared_continuous_features).issubset(self.shared_features):
            raise ValueError("shared_continuous_features must be shared features")

    @property
    def protected_attributes(self) -> tuple[str, ...]:
        return (self.primary_protected, *self.secondary_protected)

    @property
    def protected_included_shared_features(self) -> tuple[str, ...]:
        """Second experimental view adding the primary protected attribute."""

        return (*self.shared_features, self.primary_protected)

    @property
    def preprocessing_features(self) -> tuple[str, ...]:
        """Ordered technical union required by both shared experiments."""

        required = set(self.protected_included_shared_features)
        return tuple(name for name in self.expected_features if name in required)


@dataclass(frozen=True)
class DatasetBundle:
    """Original UCI dataset as received, plus source provenance."""

    spec: DatasetSpec
    features: pd.DataFrame
    target: pd.Series
    protected: pd.DataFrame
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(self.features.columns) != self.spec.expected_features:
            raise ValueError("feature columns do not match the registered schema and order")
        if len(self.features) == 0 or len(self.target) != len(self.features):
            raise ValueError("features and target must be non-empty and row-aligned")
        if not self.features.index.equals(self.target.index):
            raise ValueError("features and target must have identical indices")
        if tuple(self.protected.columns) != self.spec.protected_attributes:
            raise ValueError("protected columns do not match the registered attributes")
        if not self.features.index.equals(self.protected.index):
            raise ValueError("protected attributes must be row-aligned")

    @property
    def n_rows(self) -> int:
        return len(self.features)


ADULT_FEATURES = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
)

CREDIT_FEATURES = tuple(f"X{i}" for i in range(1, 24))

CLEVELAND_FEATURES = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)


DATASET_REGISTRY: Mapping[str, DatasetSpec] = {
    "D6": DatasetSpec(
        dataset_id="D6",
        name="Adult Census Income",
        domain="socioeconomic",
        uci_id=2,
        uci_url="https://archive.ics.uci.edu/dataset/2/adult",
        doi="10.24432/C5XW20",
        outcome_source="income",
        outcome_description="annual income above USD 50,000",
        primary_protected="sex",
        secondary_protected=("race",),
        expected_features=ADULT_FEATURES,
        numeric_features=(
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ),
        categorical_features=tuple(
            name
            for name in ADULT_FEATURES
            if name
            not in {
                "age",
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
            }
        ),
        shared_features=(
            "age",
            "workclass",
            "education-num",
            "occupation",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ),
        shared_continuous_features=(
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ),
        excluded_feature_reasons={
            "fnlwgt": "survey sampling weight, not an individual characteristic",
            "education": "redundant with education-num in the exploratory primary model",
            "race": "protected context; excluded from primary prediction and similarity",
            "sex": "primary protected attribute",
            "marital-status": "normative similarity exclusion",
            "relationship": "normative similarity exclusion and close proxy for family status",
            "native-country": "normative similarity exclusion and proxy risk",
        },
        limitations=(
            "1994 US Census benchmark; not evidence about a current population",
            "income threshold is an imperfect and historically situated outcome",
            "feature roles are normative exploratory choices requiring review",
        ),
    ),
    "D7": DatasetSpec(
        dataset_id="D7",
        name="Default of Credit Card Clients",
        domain="finance",
        uci_id=350,
        uci_url="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        doi="10.24432/C55S3H",
        outcome_source="Y",
        outcome_description="default payment in the following month",
        primary_protected="X2",
        secondary_protected=(),
        expected_features=CREDIT_FEATURES,
        numeric_features=tuple(name for name in CREDIT_FEATURES if name not in {"X2", "X3", "X4"}),
        categorical_features=("X2", "X3", "X4"),
        shared_features=("X1", "X5", *tuple(f"X{i}" for i in range(6, 24))),
        shared_continuous_features=("X1", "X5", *tuple(f"X{i}" for i in range(6, 24))),
        excluded_feature_reasons={
            "X2": "primary protected attribute (sex)",
            "X3": "education excluded from primary similarity as a socially structured attribute",
            "X4": "marital status excluded from primary similarity",
        },
        limitations=(
            "Taiwan credit-card clients observed in 2005",
            "repayment codes include undocumented or irregular values in common copies",
            "feature roles require domain review before confirmation",
        ),
    ),
    "D8": DatasetSpec(
        dataset_id="D8",
        name="Cleveland Heart Disease",
        domain="healthcare",
        uci_id=45,
        uci_url="https://archive.ics.uci.edu/dataset/45/heart+disease",
        doi="10.24432/C52P4X",
        outcome_source="num",
        outcome_description="angiographic heart-disease presence (num > 0)",
        primary_protected="sex",
        secondary_protected=(),
        expected_features=CLEVELAND_FEATURES,
        numeric_features=("age", "trestbps", "chol", "thalach", "oldpeak"),
        categorical_features=("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"),
        shared_features=tuple(name for name in CLEVELAND_FEATURES if name != "sex"),
        shared_continuous_features=("age", "trestbps", "chol", "thalach", "oldpeak"),
        excluded_feature_reasons={"sex": "primary protected attribute"},
        limitations=(
            "small historical benchmark with limited subgroup support",
            "six values are missing in ca/thal in the current UCI dataframe",
            "does not establish validity for a contemporary clinical population",
        ),
    ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """Return one fixed registered specification."""

    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unknown dataset ID: {dataset_id}") from exc
