"""Resumable E1/E2 execution with inline E3 and compact persistence."""

from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat
from typing import Iterable

from ifcred.data import fetch_uci_dataset, load_cached_dataset
from ifcred.experiments.conditions import standard_injection_definitions
from ifcred.experiments.frozen_tuning import (
    development_experiment_partition,
    load_frozen_model_specs,
    tuning_bundle_path,
)
from ifcred.experiments.injections import (
    assemble_model_family_disagreement_case,
    prepare_model_family_injection_plan,
    prepare_paired_near_duplicates,
)
from ifcred.experiments.orchestrator import (
    PreparedRepetition,
    prepare_policy_repetition,
    prepare_shared_repetition,
    run_clean_condition,
    run_model_disagreement_condition,
    run_prepared_injection_condition,
    FixedModelSelection,
)
from ifcred.experiments.persistence import (
    configuration_id_for_setup,
    expected_condition_path,
    save_condition_result,
)
from ifcred.experiments.repetitions import ExperimentRepetition
from ifcred.experiments.similarity import fit_similarity_calibration
from ifcred.experiments.similarity import SimilarityCalibration
from ifcred.experiments.study_config import StudyConfig
from ifcred.models import ModelFamily, ModelSpec


def _load_bundle(dataset_id: str, cache_root: Path):
    try:
        return load_cached_dataset(dataset_id, cache_root)
    except FileNotFoundError:
        return fetch_uci_dataset(dataset_id, cache_root=cache_root)


def _audit_context(primary: PreparedRepetition, setup) -> PreparedRepetition:
    split = primary.prepared_dataset.split
    similarity = fit_similarity_calibration(
        primary.experiment_matrix[split.train_indices], setup.graph_fitting_spec
    )
    return replace(primary, setup=setup, similarity_calibration=similarity)


def _resume_context(shared, setup, root: Path) -> PreparedRepetition | None:
    """Hydrate frozen tuning/similarity state from any existing condition."""

    base = (
        root
        / shared.bundle.spec.dataset_id
        / setup.protected_policy.value
        / f"repetition_{shared.repetition.repetition:03d}"
        / f"config_{configuration_id_for_setup(setup)}"
    )
    manifests = sorted(base.glob("**/manifest.json"))
    if not manifests:
        return None
    dataless_flag = getattr(stat, "SF_DATALESS", 0)
    stored = None
    for manifest_path in manifests:
        if dataless_flag and manifest_path.stat().st_flags & dataless_flag:
            continue
        try:
            stored = json.loads(manifest_path.read_text())["context"]
            break
        except TimeoutError:
            continue
    if stored is None:
        # Preprocessing, splitting, model selection, and similarity calibration
        # are deterministic for this repetition. Reconstruct them rather than
        # requiring iCloud to hydrate an old checkpoint.
        return None
    specs = tuple(
        ModelSpec(ModelFamily(item["family"]), item["hyperparameters"])
        for item in stored["selected_model_specs"]
    )
    calibration = stored["similarity_calibration"]
    similarity = SimilarityCalibration(
        fitting_spec=setup.graph_fitting_spec,
        bandwidths=calibration["bandwidths"],
        n_training_rows=calibration["n_training_rows"],
        n_features=calibration["n_features"],
        training_matrix_sha256=calibration["training_matrix_sha256"],
    )
    prepared = shared.prepared_dataset
    return PreparedRepetition(
        bundle=shared.bundle,
        repetition=shared.repetition,
        setup=setup,
        prepared_dataset=prepared,
        experiment_matrix=prepared.experiment_matrix(setup.protected_policy),
        continuous_indices=prepared.experiment_continuous_indices(setup.protected_policy),
        selected_model_specs=specs,
        model_selection_manifest=stored["model_selection_result"],
        similarity_calibration=similarity,
    )


def _missing(root: Path, context: PreparedRepetition, condition: str, variant: str, ratio: float) -> bool:
    return not expected_condition_path(root, context, condition=condition, variant=variant, ratio=ratio).exists()


def _setup_result_exists(root: Path, shared, setup, condition: str, variant: str) -> bool:
    base = (
        root / shared.bundle.spec.dataset_id / setup.protected_policy.value
        / f"repetition_{shared.repetition.repetition:03d}"
        / f"config_{configuration_id_for_setup(setup)}" / condition / variant
    )
    return any(base.glob("rho_*"))


def _validate_compact_results_root(root: Path) -> None:
    """Prevent mixing the former large deferred-E3 format with compact runs."""

    # Legacy packages always contain at least one of these large payloads.
    # Checking their names avoids hydrating thousands of compact manifests
    # merely to resume a run from an iCloud-backed directory.
    for legacy_name in ("arrays.npz", "estimators.joblib"):
        legacy_path = next(root.glob(f"**/{legacy_name}"), None)
        if legacy_path is not None:
            raise ValueError(
                "results directory contains legacy full-model artifacts. "
                "Keep them unchanged and resume the compact run in a new --results "
                f"directory. First legacy artifact: {legacy_path}"
            )

    dataless_flag = getattr(stat, "SF_DATALESS", 0)
    for manifest_path in root.glob("**/manifest.json"):
        if dataless_flag and manifest_path.stat().st_flags & dataless_flag:
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except TimeoutError:
            # A completed atomic result can be an iCloud placeholder. Its
            # directory still acts as the resume checkpoint; reporting can
            # hydrate it later when the user is online.
            continue
        if "result" in manifest and manifest.get("storage_mode") != "compact_inline_e3":
            raise ValueError(
                "results directory contains legacy full-model artifacts. "
                "Keep them unchanged and resume the compact run in a new --results directory. "
                f"First legacy artifact: {manifest_path}"
            )


def run_study(
    config: StudyConfig,
    *,
    results_root: str | Path,
    cache_root: str | Path = "data/original",
    experiments: Iterable[str] = ("E1", "E2"),
    conditions: Iterable[str] | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """Run requested model-fitting experiments and persist every condition atomically."""

    requested = set(experiments)
    if not requested or not requested.issubset({"E1", "E2"}):
        raise ValueError("run_study accepts E1 and/or E2; E3 is computed inline")
    condition_filter = None if conditions is None else set(conditions)
    valid_conditions = {
        "benign_near_duplicate",
        "contradictory_near_duplicate",
        "isolated_instance",
        "dominant_neighbour_pair",
        "metric_disagreement_instance",
        "model_family_disagreement",
    }
    if condition_filter is not None and not condition_filter.issubset(valid_conditions):
        raise ValueError(f"unknown condition filter: {sorted(condition_filter - valid_conditions)}")
    if config.frozen_model_config_root is not None:
        missing = [
            tuning_bundle_path(config.frozen_model_config_root, dataset_id, policy)
            for dataset_id in config.dataset_ids
            for policy in config.policies
            if not tuning_bundle_path(
                config.frozen_model_config_root, dataset_id, policy
            ).exists()
        ]
        if missing:
            listing = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "frozen study requires every dataset-policy tuning bundle; "
                f"run `ifcred tune` first. Missing:\n{listing}"
            )
    root = Path(results_root); cache = Path(cache_root); written = skipped = 0
    _validate_compact_results_root(root)
    repetitions = tuple(
        ExperimentRepetition(index, config.root_seed)
        for index in config.repetition_numbers
    )
    for dataset_id in config.dataset_ids:
        original_bundle = _load_bundle(dataset_id, cache)
        if config.development_fraction is None:
            bundle = original_bundle
        else:
            _, bundle, _ = development_experiment_partition(
                original_bundle,
                development_fraction=config.development_fraction,
                root_seed=config.root_seed,
            )
        for repetition in repetitions:
            shared = prepare_shared_repetition(bundle, repetition, test_size=config.test_size)
            for policy in config.policies:
                model_selection = config.model_selection
                if config.frozen_model_config_root is not None:
                    specs = load_frozen_model_specs(
                        config.frozen_model_config_root,
                        dataset_id,
                        policy,
                        expected_root_seed=config.root_seed,
                        expected_development_fraction=config.development_fraction,
                        expected_source_sha256=original_bundle.manifest["source_sha256"],
                    )
                    tuning_path = tuning_bundle_path(
                        config.frozen_model_config_root, dataset_id, policy
                    )
                    model_selection = FixedModelSelection(
                        specs,
                        provenance={
                            "strategy": "dedicated_development_partition_then_frozen",
                            "tuning_bundle_sha256": hashlib.sha256(
                                tuning_path.read_bytes()
                            ).hexdigest(),
                            "development_fraction": config.development_fraction,
                            "root_seed": config.root_seed,
                            "source_sha256": original_bundle.manifest["source_sha256"],
                        },
                    )
                setup = config.setup(policy, model_selection=model_selection)
                primary = _resume_context(shared, setup, root) if resume else None
                if primary is None:
                    primary = prepare_policy_repetition(shared, setup)
                primary_model_run = None
                primary_clean = None
                if "E1" in requested or "E2" in requested:
                    clean_missing = _missing(root, primary, "clean_baseline", "primary", 0.0)
                    native_missing = "E1" in requested and _missing(root, primary, "clean_baseline", "native_probability", 0.0)
                    if clean_missing or native_missing:
                        primary_clean = run_clean_condition(primary)
                        primary_model_run = primary_clean.model_run
                    if clean_missing:
                        save_condition_result(root, primary, primary_clean); written += 1
                    else:
                        skipped += 1
                    if "E1" in requested:
                        if native_missing:
                            native = run_clean_condition(primary, model_run=primary_model_run, condition_variant="native_probability", probability_source="native")
                            save_condition_result(root, primary, native); written += 1
                        else:
                            skipped += 1
                        for audit in config.audit_variants[1:]:
                            audit_setup = config.setup(
                                policy, audit, model_selection=model_selection
                            )
                            if _setup_result_exists(
                                root, shared, audit_setup, "clean_baseline", audit.name
                            ):
                                skipped += 1
                                continue
                            context = _audit_context(primary, audit_setup)
                            if not _missing(root, context, "clean_baseline", audit.name, 0.0):
                                skipped += 1; continue
                            if primary_model_run is None:
                                primary_clean = run_clean_condition(primary)
                                primary_model_run = primary_clean.model_run
                            result = run_clean_condition(context, model_run=primary_model_run, condition_variant=audit.name)
                            save_condition_result(root, context, result); written += 1
                if "E2" not in requested:
                    continue
                definitions = standard_injection_definitions(config)
                if condition_filter is not None:
                    definitions = tuple(
                        definition
                        for definition in definitions
                        if definition.name in condition_filter
                    )
                for definition in definitions:
                    missing_ratios = [ratio for ratio in definition.injection_ratios if _missing(root, primary, definition.name, definition.variant, ratio)]
                    skipped += len(definition.injection_ratios) - len(missing_ratios)
                    if not missing_ratios:
                        continue
                    prepared = definition.factory(primary, repetition.injection_seed(definition.resolved_seed_stream))
                    for ratio in missing_ratios:
                        result = run_prepared_injection_condition(primary, prepared, injection_ratio=ratio, condition_variant=definition.variant)
                        save_condition_result(root, primary, result); written += 1
                if condition_filter is not None and "model_family_disagreement" not in condition_filter:
                    continue
                paired = prepare_paired_near_duplicates(
                    primary.experiment_matrix, primary.prepared_dataset.y,
                    legitimate_continuous_indices=primary.continuous_indices,
                    k=primary.setup.graph_fitting_spec.k, radius_fraction=config.near_duplicate_radii[0],
                    random_state=repetition.injection_seed("model_family_disagreement"),
                )
                split = primary.prepared_dataset.split
                for affected in config.model_affected_fractions:
                    variant = f"affected_fraction_{affected:g}"
                    for ratio in config.injection_ratios:
                        if not _missing(root, primary, "model_family_disagreement", variant, ratio):
                            skipped += 1; continue
                        plan_for_models = prepare_model_family_injection_plan(
                            [spec.name for spec in primary.selected_model_specs], affected_fraction=affected,
                            random_state=repetition.injection_seed("model_family_allocation"),
                        )
                        case = assemble_model_family_disagreement_case(
                            primary.experiment_matrix, primary.prepared_dataset.y,
                            split.train_indices, split.test_indices, paired, plan_for_models,
                            injection_ratio=ratio,
                        )
                        result = run_model_disagreement_condition(primary, case, condition_variant=variant)
                        save_condition_result(root, primary, result); written += 1
    run_manifest = {
        "profile": config.profile,
        "datasets": list(config.dataset_ids),
        "repetition_numbers": list(config.repetition_numbers),
        "requested_experiments": sorted(requested),
        "written": written,
        "skipped_existing": skipped,
        "complete": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(
        json.dumps(run_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_directory = root / "run_manifests"
    run_directory.mkdir(exist_ok=True)
    (run_directory / f"run_{run_id}.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    return {"written": written, "skipped": skipped}
