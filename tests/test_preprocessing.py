import numpy as np
import pandas as pd

from ifcred.data import (
    DatasetSplit,
    ProtectedAttributePolicy,
    fetch_uci_dataset,
    make_stratified_split,
    preprocess_dataset,
)
from test_data_acquisition import adult_frames, fake_remote


def adult_bundle(n=12):
    features, targets = adult_frames(n)
    return fetch_uci_dataset(
        "D6", fetcher=lambda **kwargs: fake_remote(features, targets)
    )


def test_split_is_reproducible_and_preserves_joint_cells():
    bundle = adult_bundle()
    first = make_stratified_split(bundle, test_size=1 / 3, random_state=7)
    second = make_stratified_split(bundle, test_size=1 / 3, random_state=7)

    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert first.stratification_strategy == "target_x_primary_protected"
    assert set(first.train_indices) | set(first.test_indices) == set(range(bundle.n_rows))


def test_target_and_missing_token_normalization_happens_during_preprocessing():
    bundle = adult_bundle()
    assert bundle.target.iloc[1] == ">50K."
    assert bundle.features.loc[1, "workclass"] == " ? "
    split = DatasetSplit(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        random_state=1,
        test_size=1 / 3,
        stratification_strategy="manual_test",
    )
    prepared = preprocess_dataset(bundle, split)

    assert prepared.y.tolist() == [0, 1] * 6
    assert prepared.protected_preprocessed["sex"].tolist() == ["Female", "Male"] * 6
    assert np.all(np.isfinite(prepared.X_preprocessed))


def test_preprocessing_is_fitted_only_on_training_rows():
    bundle = adult_bundle()
    split = DatasetSplit(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        random_state=1,
        test_size=1 / 3,
        stratification_strategy="manual_test",
    )
    original = preprocess_dataset(bundle, split)

    altered_features = bundle.features.copy()
    altered_features.loc[split.test_indices, "age"] = 1_000_000
    altered_bundle = type(bundle)(
        spec=bundle.spec,
        features=altered_features,
        target=bundle.target,
        protected=altered_features.loc[:, bundle.spec.protected_attributes],
        manifest=bundle.manifest,
    )
    altered = preprocess_dataset(altered_bundle, split)

    np.testing.assert_allclose(
        original.X_preprocessed[split.train_indices],
        altered.X_preprocessed[split.train_indices],
    )
    age_index = original.transformed_indices_by_source["age"][0]
    assert abs(float(original.X_preprocessed[split.train_indices, age_index].mean())) < 1e-12
    assert altered.X_preprocessed[split.test_indices, age_index].min() > 100_000


def test_prediction_and_similarity_use_the_same_policy_matrix():
    bundle = adult_bundle()
    split = DatasetSplit(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        random_state=1,
        test_size=1 / 3,
        stratification_strategy="manual_test",
    )
    prepared = preprocess_dataset(bundle, split)
    sex_indices = set(prepared.transformed_indices_by_source["sex"])

    assert sex_indices.isdisjoint(set(prepared.shared_indices))
    assert sex_indices.issubset(set(prepared.protected_included_indices))
    excluded = prepared.experiment_matrix(ProtectedAttributePolicy.EXCLUDE_PROTECTED)
    included = prepared.experiment_matrix(
        ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED
    )
    assert excluded.shape[0] == bundle.n_rows
    assert included.shape[1] == excluded.shape[1] + 1
    assert len(sex_indices) == 1
    np.testing.assert_array_equal(prepared.experiment_matrix(), included)


def test_missing_and_unseen_categories_produce_finite_values():
    bundle = adult_bundle()
    features = bundle.features.copy()
    features.loc[10, "occupation"] = "Never-seen-in-training"
    modified = type(bundle)(
        spec=bundle.spec,
        features=features,
        target=bundle.target,
        protected=features.loc[:, bundle.spec.protected_attributes],
        manifest=bundle.manifest,
    )
    split = DatasetSplit(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        random_state=1,
        test_size=1 / 3,
        stratification_strategy="manual_test",
    )
    prepared = preprocess_dataset(modified, split)

    assert np.all(np.isfinite(prepared.X_preprocessed))


def test_augmented_master_rows_can_be_projected_into_both_experiments():
    bundle = adult_bundle()
    split = DatasetSplit(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        random_state=1,
        test_size=1 / 3,
        stratification_strategy="manual_test",
    )
    prepared = preprocess_dataset(bundle, split)
    augmented = np.vstack(
        (prepared.X_preprocessed, prepared.X_preprocessed[[0]])
    )

    assert prepared.experiment_matrix_from(
        augmented, ProtectedAttributePolicy.EXCLUDE_PROTECTED
    ).shape[0] == bundle.n_rows + 1
    assert prepared.experiment_matrix_from(
        augmented, ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED
    ).shape[0] == bundle.n_rows + 1


def test_binary_categorical_features_use_one_transformed_column():
    bundle = adult_bundle()
    split = DatasetSplit(
        train_indices=np.arange(8),
        test_indices=np.arange(8, 12),
        random_state=1,
        test_size=1 / 3,
        stratification_strategy="manual_test",
    )
    prepared = preprocess_dataset(bundle, split)

    assert len(prepared.transformed_indices_by_source["sex"]) == 1
