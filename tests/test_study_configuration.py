from ifcred.data import ProtectedAttributePolicy
import pytest
from ifcred.experiments.conditions import standard_injection_definitions
from ifcred.experiments.frozen_tuning import (
    PILOT_TRIAL_BUDGETS,
    development_experiment_partition,
    search_spaces_for_budget,
)
from ifcred.experiments.study_config import exploratory_config, frozen_config, smoke_config
from ifcred.experiments.study import run_study
from ifcred.data import fetch_uci_dataset
from test_data_acquisition import adult_frames, fake_remote


def test_exploratory_profile_covers_declared_datasets_policies_and_seeds():
    config = exploratory_config()

    assert config.dataset_ids == ("D6", "D7", "D8")
    assert config.n_repetitions == 30
    assert config.policies[0] == ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED
    assert set(config.policies) == set(ProtectedAttributePolicy)
    assert {variant.graph_spec.k for variant in config.audit_variants} >= {5, 10, 20}
    assert config.injection_ratios == (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    assert config.disagreement_reliabilities == (0.90, 0.75)


def test_e2_definitions_cover_component_targets_and_nested_ratios():
    config = smoke_config()
    definitions = standard_injection_definitions(config)

    assert {definition.target.value for definition in definitions} >= {
        "C", "D", "F", "control"
    }
    assert all(definition.injection_ratios == config.injection_ratios for definition in definitions)
    assert len({(definition.name, definition.variant) for definition in definitions}) == len(definitions)


def test_development_rows_are_disjoint_from_repeated_experiment_population():
    features, target = adult_frames(100)
    bundle = fetch_uci_dataset(
        "D6", fetcher=lambda **kwargs: fake_remote(features, target)
    )
    development, experiment, evidence = development_experiment_partition(
        bundle, development_fraction=0.20, root_seed=41
    )

    development_rows = set(evidence["development_original_indices"])
    experiment_rows = set(evidence["experiment_original_indices"])
    assert development_rows.isdisjoint(experiment_rows)
    assert development_rows | experiment_rows == set(range(bundle.n_rows))
    assert development.n_rows + experiment.n_rows == bundle.n_rows


def test_pilot_tuning_budget_is_explicit_and_smaller_than_full():
    pilot = search_spaces_for_budget("pilot")
    full = search_spaces_for_budget("full")

    assert {space.model.name: space.n_trials for space in pilot} == PILOT_TRIAL_BUDGETS
    assert all(a.n_trials < b.n_trials for a, b in zip(pilot, full))


def test_frozen_profile_disables_repetition_level_bayesian_selection(tmp_path):
    config = frozen_config(tmp_path)

    assert config.profile == "frozen"
    assert config.development_fraction == 0.20
    assert config.frozen_model_config_root == tmp_path.resolve()


def test_one_n_jobs_setting_propagates_to_models_tuning_and_graphs(tmp_path):
    config = frozen_config(tmp_path, n_jobs=4)
    setup = config.setup(config.policies[0])

    assert config.n_jobs == 4
    assert setup.n_jobs == 4
    assert setup.graph_fitting_spec.n_jobs == 4


def test_frozen_run_fails_before_data_loading_when_bundles_are_missing(tmp_path):
    config = frozen_config(tmp_path / "missing", n_jobs=2)

    with pytest.raises(FileNotFoundError, match="run `ifcred tune` first"):
        run_study(
            config,
            results_root=tmp_path / "results",
            cache_root=tmp_path / "data",
        )
