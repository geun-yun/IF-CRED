"""One-time tuning on development rows excluded from repeated experiments."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ifcred.data import (
    DatasetBundle,
    DatasetSplit,
    ProtectedAttributePolicy,
    fetch_uci_dataset,
    load_cached_dataset,
    make_stratified_split,
    preprocess_dataset,
)
from ifcred.experiments.repetitions import ExperimentRepetition
from ifcred.models import (
    BayesianOptimizationSpec,
    BayesianSearchSpace,
    CrossValidationSpec,
    ModelFamily,
    ModelSpec,
    recommended_bayesian_search_spaces,
    tune_model_family_bayesian,
)


PILOT_TRIAL_BUDGETS = {
    "logistic_regression": 15,
    "mlp": 25,
    "gaussian_naive_bayes": 10,
    "random_forest": 25,
    "decision_tree": 20,
}


def partition_seed(root_seed: int, dataset_id: str) -> int:
    return ExperimentRepetition(0, root_seed).seed_for(
        "split", f"development_partition:{dataset_id}"
    )


def _subset_bundle(
    bundle: DatasetBundle, indices: np.ndarray, *, role: str, root_seed: int
) -> DatasetBundle:
    selected = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256(selected.tobytes()).hexdigest()
    features = bundle.features.iloc[selected].reset_index(drop=True)
    target = bundle.target.iloc[selected].reset_index(drop=True)
    protected = features.loc[:, bundle.spec.protected_attributes].copy()
    manifest = {
        **dict(bundle.manifest),
        "analysis_subset_role": role,
        "parent_n_rows": bundle.n_rows,
        "subset_n_rows": len(selected),
        "subset_original_indices_sha256": digest,
        "development_partition_root_seed": root_seed,
    }
    return DatasetBundle(
        spec=bundle.spec,
        features=features,
        target=target,
        protected=protected,
        manifest=manifest,
    )


def development_experiment_partition(
    bundle: DatasetBundle,
    *,
    development_fraction: float,
    root_seed: int,
) -> tuple[DatasetBundle, DatasetBundle, dict[str, Any]]:
    """Create fixed disjoint development and repeated-experiment populations."""

    if not 0.1 <= development_fraction <= 0.5:
        raise ValueError("development_fraction must lie between 0.1 and 0.5")
    seed = partition_seed(root_seed, bundle.spec.dataset_id)
    partition = make_stratified_split(
        bundle,
        test_size=development_fraction,
        random_state=seed,
        stratify_protected=True,
    )
    development_indices = partition.test_indices
    experiment_indices = partition.train_indices
    development = _subset_bundle(
        bundle, development_indices, role="hyperparameter_development", root_seed=root_seed
    )
    experiment = _subset_bundle(
        bundle, experiment_indices, role="repeated_experiment", root_seed=root_seed
    )
    evidence = {
        "random_state": seed,
        "development_fraction_requested": development_fraction,
        "development_fraction_realized": len(development_indices) / bundle.n_rows,
        "stratification_strategy": partition.stratification_strategy,
        "development_original_indices": development_indices.tolist(),
        "experiment_original_indices": experiment_indices.tolist(),
    }
    return development, experiment, evidence


def search_spaces_for_budget(budget: str) -> tuple[BayesianSearchSpace, ...]:
    spaces = recommended_bayesian_search_spaces()
    if budget == "full":
        return spaces
    if budget != "pilot":
        raise ValueError("budget must be 'pilot' or 'full'")
    return tuple(
        replace(space, n_trials=PILOT_TRIAL_BUDGETS[space.model.name])
        for space in spaces
    )


def tuning_bundle_path(
    output_root: str | Path,
    dataset_id: str,
    policy: ProtectedAttributePolicy,
) -> Path:
    return Path(output_root).resolve() / dataset_id / policy.value / "tuning.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.stem}-",
        suffix=".json.tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def tune_frozen_model_bundles(
    *,
    dataset_ids: tuple[str, ...],
    policies: tuple[ProtectedAttributePolicy, ...],
    cache_root: str | Path,
    output_root: str | Path,
    root_seed: int,
    development_fraction: float = 0.20,
    budget: str = "pilot",
    n_jobs: int = 1,
    folds: int = 5,
) -> dict[str, int]:
    """Tune five families once for each dataset-policy combination."""

    spaces = search_spaces_for_budget(budget)
    optimization = BayesianOptimizationSpec(
        cross_validation=CrossValidationSpec(folds=folds),
        startup_trials=10,
    )
    written = skipped = model_checkpoints_written = model_checkpoints_reused = 0
    for dataset_id in dataset_ids:
        try:
            original = load_cached_dataset(dataset_id, cache_root)
        except FileNotFoundError:
            original = fetch_uci_dataset(dataset_id, cache_root=cache_root)
        development, _, partition = development_experiment_partition(
            original,
            development_fraction=development_fraction,
            root_seed=root_seed,
        )
        all_rows = np.arange(development.n_rows, dtype=np.int64)
        preprocessing_split = DatasetSplit(
            train_indices=all_rows,
            test_indices=np.empty(0, dtype=np.int64),
            random_state=partition["random_state"],
            test_size=0.0,
            stratification_strategy="dedicated_development_population",
        )
        prepared = preprocess_dataset(development, preprocessing_split)
        for policy in policies:
            destination = tuning_bundle_path(output_root, dataset_id, policy)
            if destination.exists():
                skipped += 1
                continue
            matrix = prepared.experiment_matrix(policy)
            model_manifests: dict[str, Any] = {}
            selected_specs: list[dict[str, Any]] = []
            for space in spaces:
                checkpoint = destination.parent / f"{space.model.name}.json"
                if checkpoint.exists():
                    model_payload = json.loads(checkpoint.read_text())
                    expected = {
                        "dataset_id": dataset_id,
                        "protected_policy": policy.value,
                        "root_seed": root_seed,
                        "development_fraction": development_fraction,
                        "budget": budget,
                        "source_sha256": original.manifest["source_sha256"],
                    }
                    if any(model_payload.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"incompatible tuning checkpoint: {checkpoint}")
                    model_checkpoints_reused += 1
                else:
                    tuning_seed = ExperimentRepetition(0, root_seed).seed_for(
                        "tuning",
                        f"frozen:{dataset_id}:{policy.value}:{space.model.name}",
                    )
                    tuning = tune_model_family_bayesian(
                        matrix,
                        prepared.y,
                        search_spaces=(space,),
                        optimization=optimization,
                        random_state=tuning_seed,
                        n_jobs=n_jobs,
                        require_declared_family=False,
                    )
                    result = tuning.results[space.model.name]
                    model_payload = {
                        "schema_version": 1,
                        "dataset_id": dataset_id,
                        "protected_policy": policy.value,
                        "model_name": space.model.name,
                        "root_seed": root_seed,
                        "development_fraction": development_fraction,
                        "budget": budget,
                        "source_sha256": original.manifest["source_sha256"],
                        "selected_model_spec": result.selected_spec.to_manifest(),
                        "tuning": tuning.to_manifest(),
                    }
                    _atomic_json(checkpoint, model_payload)
                    model_checkpoints_written += 1
                selected_specs.append(model_payload["selected_model_spec"])
                model_manifests[space.model.name] = checkpoint.name
            payload = {
                "schema_version": 1,
                "strategy": "dedicated_development_partition_then_frozen",
                "dataset_id": dataset_id,
                "protected_policy": policy.value,
                "root_seed": root_seed,
                "development_fraction": development_fraction,
                "budget": budget,
                "partition": partition,
                "source_sha256": original.manifest["source_sha256"],
                "development_preprocessed_shape": list(matrix.shape),
                "selected_model_specs": selected_specs,
                "model_tuning_files": model_manifests,
            }
            _atomic_json(destination, payload)
            written += 1
    return {
        "bundles_written": written,
        "bundles_skipped": skipped,
        "model_checkpoints_written": model_checkpoints_written,
        "model_checkpoints_reused": model_checkpoints_reused,
    }


def load_frozen_model_specs(
    root: str | Path,
    dataset_id: str,
    policy: ProtectedAttributePolicy,
    *,
    expected_root_seed: int,
    expected_development_fraction: float,
    expected_source_sha256: str,
) -> tuple[ModelSpec, ...]:
    """Load and validate one immutable dataset-policy model bundle."""

    path = tuning_bundle_path(root, dataset_id, policy)
    if not path.exists():
        raise FileNotFoundError(f"missing frozen tuning bundle: {path}")
    payload = json.loads(path.read_text())
    if (
        payload["dataset_id"] != dataset_id
        or payload["protected_policy"] != policy.value
    ):
        raise ValueError(f"dataset or policy does not match tuning bundle: {path}")
    if payload["root_seed"] != expected_root_seed:
        raise ValueError(f"root seed does not match frozen tuning bundle: {path}")
    if not np.isclose(
        payload["development_fraction"], expected_development_fraction
    ):
        raise ValueError(f"development fraction does not match tuning bundle: {path}")
    if payload["source_sha256"] != expected_source_sha256:
        raise ValueError(f"dataset snapshot does not match tuning bundle: {path}")
    return tuple(
        ModelSpec(ModelFamily(item["family"]), item["hyperparameters"])
        for item in payload["selected_model_specs"]
    )
