import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import joblib

from ifcred.core.graph import AuditGraphSpec, BandwidthPolicy, DistanceMetricSpec
from ifcred.data import ProtectedAttributePolicy, fetch_uci_dataset
from ifcred.experiments import (
    EXPERIMENT_CATALOGUE,
    BayesianModelSelection,
    ExperimentRepetition,
    RepetitionPlan,
    ExperimentSetup,
    FixedModelSelection,
    PreparedConditionDefinition,
    prepare_experiment_repetition,
    iter_repeated_experiment,
    run_model_disagreement_condition,
    run_prepared_condition_definitions,
    save_condition_result,
)
from ifcred.experiments.persistence import configuration_id_for_setup
from ifcred.experiments.migration import migrate_legacy_results
from ifcred.experiments.persistence import condition_result_path
from ifcred.comparisons.artifacts import comparison_rows_for_saved_condition
from ifcred.experiments.injections import (
    InjectionTarget,
    assemble_model_family_disagreement_case,
    prepare_model_family_injection_plan,
    prepare_paired_near_duplicates,
)
from ifcred.models import (
    BayesianOptimizationSpec,
    BayesianSearchSpace,
    CalibrationSpec,
    CrossValidationSpec,
    FloatDistribution,
    IntDistribution,
    ModelFamily,
    ModelSpec,
)
from test_data_acquisition import adult_frames, fake_remote


def tiny_model_specs():
    return (
        ModelSpec(
            ModelFamily.LOGISTIC_REGRESSION,
            {"max_iter": 200, "solver": "liblinear"},
        ),
        ModelSpec(
            ModelFamily.MLP,
            {"hidden_layer_sizes": (6,), "max_iter": 60, "early_stopping": True},
        ),
        ModelSpec(ModelFamily.GAUSSIAN_NAIVE_BAYES),
        ModelSpec(
            ModelFamily.RANDOM_FOREST,
            {"n_estimators": 10, "max_depth": 2},
        ),
        ModelSpec(ModelFamily.DECISION_TREE, {"max_depth": 3}),
    )


@pytest.fixture(scope="module")
def prepared_context():
    features, targets = adult_frames(80)
    bundle = fetch_uci_dataset(
        "D6", fetcher=lambda **kwargs: fake_remote(features, targets)
    )
    setup = ExperimentSetup(
        protected_policy=ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,
        test_size=0.25,
        graph_fitting_spec=AuditGraphSpec(
            k=3,
            metrics=(
                DistanceMetricSpec("euclidean", "euclidean"),
                DistanceMetricSpec("manhattan", "manhattan"),
            ),
            primary_metric="euclidean",
            bandwidth_policy=BandwidthPolicy.MEDIAN_RETAINED_MAX,
        ),
        model_selection=FixedModelSelection(tiny_model_specs()),
        probability_calibration=CalibrationSpec(folds=2),
        calibration_bins=5,
    )
    return prepare_experiment_repetition(
        bundle,
        ExperimentRepetition(repetition=0, root_seed=101),
        setup,
    )


def contradictory_factory(context, seed):
    return prepare_paired_near_duplicates(
        context.experiment_matrix,
        context.prepared_dataset.y,
        legitimate_continuous_indices=context.continuous_indices,
        k=3,
        radius_fraction=0.05,
        random_state=seed,
    ).contradictory


@pytest.fixture(scope="module")
def paired_results(prepared_context):
    definition = PreparedConditionDefinition(
        name="contradictory_near_duplicate",
        target=InjectionTarget.FAIRNESS,
        injection_ratios=(0.05, 0.10),
        factory=contradictory_factory,
        seed_stream="paired_near_duplicate",
    )
    return run_prepared_condition_definitions(prepared_context, (definition,))


def test_end_to_end_clean_and_nested_injection_conditions(paired_results):
    assert [result.condition for result in paired_results] == [
        "clean_baseline",
        "contradictory_near_duplicate",
        "contradictory_near_duplicate",
    ]
    assert paired_results[1].requested_injection_ratio == 0.05
    assert paired_results[2].requested_injection_ratio == 0.10
    low_selected = set(
        paired_results[1].condition_metadata["selected_candidate_indices"]
    )
    high_selected = set(
        paired_results[2].condition_metadata["selected_candidate_indices"]
    )
    assert low_selected < high_selected


def test_every_score_uses_one_shared_evaluation_population(paired_results):
    for result in paired_results:
        n_rows = len(result.evaluation_labels)
        assert result.graph.neighbour_indices.shape == (n_rows, 3)
        assert len(np.unique(result.graph.row_ids)) == n_rows
        assert all(
            len(probabilities) == n_rows
            for probabilities in result.model_run.calibrated_probabilities_by_model.values()
        )
        assert all(
            len(local) == n_rows
            for local in result.assessment.local_fairness.values()
        )
        assert all(
            0.0 <= value <= 1.0
            for value in (
                result.assessment.C,
                result.assessment.D,
                result.assessment.F,
                result.assessment.M,
                result.assessment.V,
            )
        )


def test_similarity_scales_and_model_settings_are_frozen_across_conditions(
    prepared_context, paired_results
):
    expected_bandwidths = dict(prepared_context.similarity_calibration.bandwidths)
    expected_specs = [spec.to_manifest() for spec in prepared_context.selected_model_specs]

    for result in paired_results:
        assert result.graph.bandwidths == expected_bandwidths
        resolved_families = set(result.model_run.models)
        assert resolved_families == {spec["family"] for spec in expected_specs}


def test_model_family_disagreement_uses_common_test_graph(prepared_context):
    paired = prepare_paired_near_duplicates(
        prepared_context.experiment_matrix,
        prepared_context.prepared_dataset.y,
        legitimate_continuous_indices=prepared_context.continuous_indices,
        k=3,
        radius_fraction=0.05,
        random_state=prepared_context.repetition.injection_seed(
            "model_family_disagreement"
        ),
    )
    model_names = tuple(spec.name for spec in prepared_context.selected_model_specs)
    plan = prepare_model_family_injection_plan(
        model_names,
        affected_fraction=0.4,
        random_state=prepared_context.repetition.seed_for("injection", "model_allocation"),
    )
    split = prepared_context.prepared_dataset.split
    case = assemble_model_family_disagreement_case(
        prepared_context.experiment_matrix,
        prepared_context.prepared_dataset.y,
        split.train_indices,
        split.test_indices,
        paired,
        plan,
        injection_ratio=0.1,
    )
    result = run_model_disagreement_condition(prepared_context, case)

    assert result.target == InjectionTarget.MODEL_STABILITY
    assert len(result.graph.row_ids) == len(case.y_test)
    assert result.condition_metadata["realized_affected_fraction"] == 0.4
    assert set(result.condition_metadata["model_exposure_assignments"]) == set(model_names)


def test_result_storage_is_atomic_immutable_and_checksummed(
    tmp_path, prepared_context, paired_results
):
    path = save_condition_result(tmp_path, prepared_context, paired_results[1])
    manifest = json.loads((path / "manifest.json").read_text())
    comparisons = (path / "comparisons.csv").read_bytes()

    assert manifest["schema_version"] == 2
    assert manifest["storage_mode"] == "compact_inline_e3"
    assert manifest["result"]["condition"] == "contradictory_near_duplicate"
    assert manifest["comparisons"]["sha256"] == hashlib.sha256(comparisons).hexdigest()
    assert manifest["comparisons"]["rows"] == 15
    assert manifest["result"]["injection_trace"]["pair_count"] > 0
    assert "injected_pairs" not in manifest["result"]
    assert "train_indices" not in manifest["context"]["split"]
    assert not (path / "arrays.npz").exists()
    assert not (path / "estimators.joblib").exists()
    assert (path / "manifest.json").stat().st_size < 100_000
    assert sum(item.stat().st_size for item in path.iterdir()) < 150_000
    assert manifest["configuration_id"] in str(path)
    with pytest.raises(FileExistsError, match="immutable result"):
        save_condition_result(tmp_path, prepared_context, paired_results[1])


def test_configuration_id_ignores_worker_count(prepared_context):
    original = prepared_context.setup
    parallel = replace(
        original,
        n_jobs=5,
        graph_fitting_spec=replace(original.graph_fitting_spec, n_jobs=5),
    )

    assert configuration_id_for_setup(original) == configuration_id_for_setup(parallel)


def test_legacy_results_migrate_resumably_to_compact_checkpoints(
    tmp_path, prepared_context, paired_results
):
    result = paired_results[1]
    source = tmp_path / "legacy"
    legacy_path = source / "one_condition"
    legacy_path.mkdir(parents=True)
    arrays_payload = {
        "evaluation_features": result.evaluation_features,
        "evaluation_labels": result.evaluation_labels,
    }
    for name in result.model_run.models:
        arrays_payload[f"training_features__{name}"] = result.training_features_by_model[name]
        arrays_payload[f"training_labels__{name}"] = result.training_labels_by_model[name]
    np.savez_compressed(legacy_path / "arrays.npz", **arrays_payload)
    joblib.dump(
        {
            name: output.calibrated_estimator
            for name, output in result.model_run.models.items()
        },
        legacy_path / "estimators.joblib",
        compress=3,
    )
    arrays = (legacy_path / "arrays.npz").read_bytes()
    estimators = (legacy_path / "estimators.joblib").read_bytes()
    legacy_manifest = {
        "schema_version": 1,
        "configuration_id": "legacy-runtime-dependent-id",
        "context": prepared_context.to_manifest(),
        "result": result.to_manifest(),
        "arrays_file": "arrays.npz",
        "arrays_sha256": hashlib.sha256(arrays).hexdigest(),
        "estimators_file": "estimators.joblib",
        "estimators_sha256": hashlib.sha256(estimators).hexdigest(),
    }
    (legacy_path / "manifest.json").write_text(json.dumps(legacy_manifest))

    destination = tmp_path / "compact"
    first = migrate_legacy_results(source, destination, n_jobs=2)
    final = condition_result_path(destination, prepared_context, result)
    compact = json.loads((final / "manifest.json").read_text())

    assert first == {
        "conditions_migrated": 1,
        "conditions_skipped": 0,
        "conditions_cloud_unavailable": 0,
        "comparison_artifacts_written": 1,
    }
    assert compact["storage_mode"] == "compact_inline_e3"
    assert compact["migration"]["source_schema_version"] == 1
    assert (final / "comparisons.csv").exists()
    assert not (final / "arrays.npz").exists()
    assert not (final / "estimators.joblib").exists()
    assert (legacy_path / "arrays.npz").exists()
    assert migrate_legacy_results(source, destination, n_jobs=2)[
        "conditions_skipped"
    ] == 1


def test_legacy_excluded_policy_e3_does_not_load_model_files(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "context": {
            "dataset": {"dataset_id": "D6"},
            "experiment_matrix_shape": [100, 8],
            "primary_protected_indices": [],
            "repetition": {
                "repetition": 0,
                "standard_seeds": {"comparator": 123},
            },
        },
        "result": {
            "condition": "clean_baseline",
            "condition_variant": "primary",
            "requested_injection_ratio": 0.0,
            "model_run": {"models": {name: {} for name in (
                "logistic_regression", "mlp", "gaussian_naive_bayes",
                "random_forest", "decision_tree",
            )}},
        },
        "arrays_file": "does_not_exist.npz",
        "estimators_file": "does_not_exist.joblib",
    }
    manifest_path.write_text(json.dumps(manifest))

    rows = comparison_rows_for_saved_condition(manifest_path, manifest, n_jobs=5)

    assert len(rows) == 15
    assert all(row["applicable"] is False for row in rows)


def test_continuous_positions_refer_to_runtime_matrix(prepared_context):
    matrix = prepared_context.experiment_matrix
    indices = prepared_context.continuous_indices

    assert np.all((indices >= 0) & (indices < matrix.shape[1]))
    assert len(indices) == len(prepared_context.prepared_dataset.shared_continuous_indices)


def test_primary_protected_position_is_explicit(prepared_context):
    positions = prepared_context.prepared_dataset.experiment_primary_protected_indices(
        prepared_context.setup.protected_policy
    )

    assert positions.shape == (1,)
    assert 0 <= positions[0] < prepared_context.experiment_matrix.shape[1]


def test_catalogue_has_unique_reviewable_experiment_ids():
    ids = [experiment.experiment_id for experiment in EXPERIMENT_CATALOGUE]

    assert ids == ["E1", "E2", "E3"]
    assert len(set(ids)) == len(ids)
    assert all(experiment.conditions for experiment in EXPERIMENT_CATALOGUE)


def test_repetition_iterator_runs_same_clean_experiment_across_split_seeds(
    prepared_context,
):
    outputs = list(
        iter_repeated_experiment(
            prepared_context.bundle,
            RepetitionPlan(root_seed=303, n_repetitions=2),
            prepared_context.setup,
            (),
        )
    )

    assert len(outputs) == 2
    assert all(result.condition == "clean_baseline" for _, result in outputs)
    assert outputs[0][0].prepared_dataset.split.random_state != outputs[1][0].prepared_dataset.split.random_state
    assert outputs[0][1].model_run.base_random_state != outputs[1][1].model_run.base_random_state


def test_orchestrator_can_select_models_with_training_only_bayesian_cv(
    prepared_context,
):
    bases = tiny_model_specs()
    spaces = (
        BayesianSearchSpace(bases[0], {"C": FloatDistribution(0.5, 1.0)}, 1),
        BayesianSearchSpace(bases[1], {"alpha": FloatDistribution(1e-5, 1e-3, log=True)}, 1),
        BayesianSearchSpace(
            bases[2], {"var_smoothing": FloatDistribution(1e-12, 1e-7, log=True)}, 1
        ),
        BayesianSearchSpace(bases[3], {"n_estimators": IntDistribution(4, 6)}, 1),
        BayesianSearchSpace(bases[4], {"max_depth": IntDistribution(2, 4)}, 1),
    )
    setup = ExperimentSetup(
        protected_policy=prepared_context.setup.protected_policy,
        test_size=prepared_context.setup.test_size,
        graph_fitting_spec=prepared_context.setup.graph_fitting_spec,
        model_selection=BayesianModelSelection(
            spaces,
            BayesianOptimizationSpec(
                cross_validation=CrossValidationSpec(folds=2),
                startup_trials=1,
            ),
        ),
        probability_calibration=CalibrationSpec(folds=2),
    )
    tuned = prepare_experiment_repetition(
        prepared_context.bundle,
        ExperimentRepetition(repetition=0, root_seed=909),
        setup,
    )

    assert len(tuned.selected_model_specs) == 5
    assert tuned.model_selection_manifest["optimization"]["algorithm"] == "Optuna TPESampler"
    assert all(
        len(model["trials"]) == 1
        for model in tuned.model_selection_manifest["models"].values()
    )
