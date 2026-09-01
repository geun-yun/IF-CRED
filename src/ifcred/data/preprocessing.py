"""Leakage-resistant splitting and reusable transformed feature views."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ifcred.data.registry import DatasetBundle


class ProtectedAttributePolicy(StrEnum):
    """The two symmetric prediction-and-similarity experiments."""

    EXCLUDE_PROTECTED = "exclude_protected"
    INCLUDE_PRIMARY_PROTECTED = "include_primary_protected"


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    random_state: int
    test_size: float
    stratification_strategy: str

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices)
        test = np.asarray(self.test_indices)
        if train.ndim != 1 or test.ndim != 1:
            raise ValueError("split indices must be one-dimensional")
        if set(train.tolist()) & set(test.tolist()):
            raise ValueError("train and test indices must be disjoint")


@dataclass(frozen=True)
class PreparedDataset:
    """The single train-fitted preprocessed version of one original dataset."""

    bundle: DatasetBundle
    split: DatasetSplit
    transformer: ColumnTransformer
    X_preprocessed: NDArray[np.float64]
    target_preprocessed: pd.Series
    protected_preprocessed: pd.DataFrame
    transformed_feature_names: tuple[str, ...]
    transformed_indices_by_source: Mapping[str, tuple[int, ...]]
    shared_indices: NDArray[np.int64]
    protected_included_indices: NDArray[np.int64]
    shared_continuous_indices: NDArray[np.int64]

    @property
    def y(self) -> NDArray[np.int8]:
        return self.target_preprocessed.to_numpy(dtype=np.int8, copy=True)

    def experiment_matrix(
        self,
        policy: ProtectedAttributePolicy = ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,
    ) -> NDArray[np.float64]:
        """Return one runtime view for both prediction and similarity.

        The stored preprocessed dataset includes the primary protected
        attribute. Exclusion is a run-time column projection, not another
        stored dataset.
        """

        return self.experiment_matrix_from(self.X_preprocessed, policy)

    def experiment_matrix_from(
        self,
        matrix: NDArray[np.float64],
        policy: ProtectedAttributePolicy = ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,
    ) -> NDArray[np.float64]:
        """Project original or augmented technical rows into one experiment."""

        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.X_preprocessed.shape[1]:
            raise ValueError("matrix must use the stored preprocessed columns")
        if policy == ProtectedAttributePolicy.EXCLUDE_PROTECTED:
            indices = self.shared_indices
        elif policy == ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED:
            indices = self.protected_included_indices
        else:
            raise ValueError(f"unsupported protected-attribute policy: {policy}")
        return values[:, indices]

    def experiment_continuous_indices(
        self,
        policy: ProtectedAttributePolicy = ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,
    ) -> NDArray[np.int64]:
        """Positions of legitimate continuous columns in the runtime matrix."""

        if policy == ProtectedAttributePolicy.EXCLUDE_PROTECTED:
            selected = self.shared_indices
        elif policy == ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED:
            selected = self.protected_included_indices
        else:
            raise ValueError(f"unsupported protected-attribute policy: {policy}")
        positions = {int(source): position for position, source in enumerate(selected)}
        try:
            return np.asarray(
                [positions[int(source)] for source in self.shared_continuous_indices],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise RuntimeError("continuous feature is absent from the experiment matrix") from exc

    def experiment_primary_protected_indices(
        self,
        policy: ProtectedAttributePolicy = ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,
    ) -> NDArray[np.int64]:
        """Positions occupied by the primary protected field in a runtime view."""

        if policy == ProtectedAttributePolicy.EXCLUDE_PROTECTED:
            return np.empty(0, dtype=np.int64)
        if policy != ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED:
            raise ValueError(f"unsupported protected-attribute policy: {policy}")
        source = self.transformed_indices_by_source[
            self.bundle.spec.primary_protected
        ]
        positions = {
            int(column): position
            for position, column in enumerate(self.protected_included_indices)
        }
        return np.asarray([positions[int(column)] for column in source], dtype=np.int64)

    @property
    def train_protected(self) -> pd.DataFrame:
        return self.protected_preprocessed.iloc[self.split.train_indices].copy()

    @property
    def test_protected(self) -> pd.DataFrame:
        return self.protected_preprocessed.iloc[self.split.test_indices].copy()


def _clean_features(bundle: DatasetBundle) -> pd.DataFrame:
    """Perform deterministic cleaning as the first preprocessing operation."""

    result = bundle.features.copy()
    for name in bundle.spec.categorical_features:
        def token(value: object) -> object:
            if pd.isna(value):
                return np.nan
            if isinstance(value, str):
                cleaned = value.strip()
                return np.nan if cleaned in {"", "?"} else cleaned
            numeric = float(value)
            return str(int(numeric)) if numeric.is_integer() else str(numeric)

        result[name] = result[name].map(token).astype(object)
    for name in bundle.spec.numeric_features:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    return result


def _binary_target(bundle: DatasetBundle) -> pd.Series:
    """Create the registered binary outcome during preprocessing."""

    source = bundle.target
    if bundle.spec.dataset_id == "D6":
        cleaned = source.astype(str).str.strip().str.rstrip(".")
        unknown = set(cleaned.unique()) - {"<=50K", ">50K"}
        if unknown:
            raise ValueError(f"unexpected Adult target values: {sorted(unknown)}")
        result = (cleaned == ">50K").astype(np.int8)
    elif bundle.spec.dataset_id == "D7":
        numeric = pd.to_numeric(source, errors="raise")
        if not set(numeric.unique()).issubset({0, 1}):
            raise ValueError("Credit target must be binary 0/1")
        result = numeric.astype(np.int8)
    elif bundle.spec.dataset_id == "D8":
        numeric = pd.to_numeric(source, errors="raise")
        if numeric.isna().any() or np.any((numeric < 0) | (numeric > 4)):
            raise ValueError("Cleveland num target must be complete and in 0..4")
        result = (numeric > 0).astype(np.int8)
    else:
        raise ValueError(f"no target preprocessing for {bundle.spec.dataset_id}")
    result.name = "target"
    return result


def _valid_joint_strata(values: pd.Series, *, test_size: float) -> bool:
    counts = values.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return False
    n_test = int(np.ceil(test_size * len(values)))
    n_train = len(values) - n_test
    return n_test >= len(counts) and n_train >= len(counts)


def make_stratified_split(
    bundle: DatasetBundle,
    *,
    test_size: float,
    random_state: int,
    stratify_protected: bool = True,
) -> DatasetSplit:
    """Split by target×primary-protected cells when support permits."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")
    cleaned = _clean_features(bundle)
    target = _binary_target(bundle).astype(str)
    protected = cleaned[bundle.spec.primary_protected].astype(str)
    joint = target + "|" + protected
    if stratify_protected and _valid_joint_strata(joint, test_size=test_size):
        strata = joint
        strategy = "target_x_primary_protected"
    else:
        strata = target
        strategy = "target"
    indices = np.arange(bundle.n_rows)
    train, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )
    return DatasetSplit(
        train_indices=np.sort(train).astype(np.int64),
        test_indices=np.sort(test).astype(np.int64),
        random_state=random_state,
        test_size=float(test_size),
        stratification_strategy=strategy,
    )


def _source_index_mapping(
    transformer: ColumnTransformer,
    numeric: list[str],
    categorical: list[str],
) -> dict[str, tuple[int, ...]]:
    mapping: dict[str, tuple[int, ...]] = {}
    for position, name in enumerate(numeric):
        mapping[name] = (position,)
    offset = len(numeric)
    if categorical:
        encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
        for name, categories in zip(categorical, encoder.categories_):
            width = len(categories)
            dropped = encoder.drop_idx_[categorical.index(name)]
            output_width = width - (0 if dropped is None else 1)
            mapping[name] = tuple(range(offset, offset + output_width))
            offset += output_width
    if offset != len(transformer.get_feature_names_out()):
        raise RuntimeError("transformed source-feature mapping is inconsistent")
    return mapping


def _flatten_indices(
    feature_names: tuple[str, ...], mapping: Mapping[str, tuple[int, ...]]
) -> NDArray[np.int64]:
    return np.asarray(
        [index for name in feature_names for index in mapping[name]], dtype=np.int64
    )


def preprocess_dataset(bundle: DatasetBundle, split: DatasetSplit) -> PreparedDataset:
    """Fit imputation/encoding/scaling on training rows and transform all rows."""

    all_indices = set(split.train_indices.tolist()) | set(split.test_indices.tolist())
    if all_indices != set(range(bundle.n_rows)):
        raise ValueError("split must partition every dataset row")
    spec = bundle.spec
    selected = list(spec.preprocessing_features)
    numeric = [name for name in selected if name in spec.numeric_features]
    categorical = [name for name in selected if name in spec.categorical_features]
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
    cleaned = _clean_features(bundle)
    target = _binary_target(bundle)
    protected = cleaned.loc[:, spec.protected_attributes].copy()
    frame = cleaned.loc[:, selected]
    transformer.fit(frame.iloc[split.train_indices])
    transformed = np.asarray(transformer.transform(frame), dtype=float)
    if transformed.ndim != 2 or not np.all(np.isfinite(transformed)):
        raise RuntimeError("preprocessing produced a non-finite or invalid matrix")
    names = tuple(str(name) for name in transformer.get_feature_names_out())
    mapping = _source_index_mapping(transformer, numeric, categorical)
    return PreparedDataset(
        bundle=bundle,
        split=split,
        transformer=transformer,
        X_preprocessed=transformed,
        target_preprocessed=target,
        protected_preprocessed=protected,
        transformed_feature_names=names,
        transformed_indices_by_source=mapping,
        shared_indices=_flatten_indices(spec.shared_features, mapping),
        protected_included_indices=_flatten_indices(
            spec.protected_included_shared_features, mapping
        ),
        shared_continuous_indices=_flatten_indices(spec.shared_continuous_features, mapping),
    )
