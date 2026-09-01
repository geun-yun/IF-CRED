"""Dataset acquisition, contracts, and split-safe preprocessing."""

from ifcred.data.acquisition import fetch_uci_dataset, load_cached_dataset
from ifcred.data.preprocessing import (
    DatasetSplit,
    ProtectedAttributePolicy,
    PreparedDataset,
    make_stratified_split,
    preprocess_dataset,
)
from ifcred.data.registry import DATASET_REGISTRY, DatasetBundle, DatasetSpec, get_dataset_spec

__all__ = [
    "DATASET_REGISTRY",
    "DatasetBundle",
    "DatasetSpec",
    "DatasetSplit",
    "ProtectedAttributePolicy",
    "PreparedDataset",
    "fetch_uci_dataset",
    "get_dataset_spec",
    "load_cached_dataset",
    "make_stratified_split",
    "preprocess_dataset",
]
