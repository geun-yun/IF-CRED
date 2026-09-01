import pandas as pd
import numpy as np
import json

from ifcred.reporting.loader import load_tuning_outputs
from ifcred.reporting.plots import (
    make_dataset_response_figures,
    make_e1_sensitivity_figure,
    make_figures,
    make_model_fairness_ratio_figures,
    make_tuning_figures,
)
from ifcred.reporting.report import generate_report
from ifcred.reporting.statistics import (
    e1_sensitivity_statistics,
    paired_clean_deltas,
    repetition_summary,
)


def sample_rows():
    rows = []
    for repetition in (0, 1, 2):
        common = dict(
            dataset_id="D8", policy="include_primary_protected",
            repetition=repetition, configuration_id="abc", target="F", k=5,
            primary_metric="euclidean", metric_set="euclidean+manhattan",
            bandwidth_policy="median_retained_max", F_min=0.7, V_worst=0.5,
        )
        rows.append({**common, "condition":"clean_baseline", "variant":"primary", "ratio":0.0, "C":0.8,"D":0.9,"F":0.8,"M":0.9,"V":0.52})
        rows.append({**common, "condition":"contradictory_near_duplicate", "variant":"radius_0.05", "ratio":0.1, "C":0.8,"D":0.9,"F":0.6,"M":0.8,"V":0.35})
    return pd.DataFrame(rows)


def test_repetition_statistics_and_paired_deltas_use_outer_repetitions():
    frame = sample_rows()
    native = frame[frame.condition.eq("clean_baseline")].copy()
    native["variant"] = "native_probability"
    native[["C", "D", "F", "M", "V"]] = 0.1
    frame = pd.concat([native, frame], ignore_index=True)
    summary = repetition_summary(frame, draws=50)
    deltas = paired_clean_deltas(frame)

    assert set(summary.n_repetitions) == {3}
    np.testing.assert_allclose(deltas.delta_F, -0.2)
    np.testing.assert_allclose(deltas.delta_V, -0.17)


def test_e1_sensitivity_statistics_and_plot_use_matched_repetitions(tmp_path):
    rows = []
    for dataset in ("D1", "D2", "D3"):
        for policy in ("exclude_protected", "include_primary_protected"):
            for repetition in range(4):
                common = {
                    "dataset_id": dataset,
                    "policy": policy,
                    "repetition": repetition,
                    "condition": "clean_baseline",
                }
                baseline = 0.8 + 0.001 * repetition
                rows.append(
                    {
                        **common,
                        "variant": "primary",
                        "C": baseline,
                        "D": baseline,
                        "F": baseline,
                        "M": baseline,
                        "V": baseline,
                    }
                )
                rows.append(
                    {
                        **common,
                        "variant": "k5",
                        "C": baseline + 0.02,
                        "D": baseline,
                        "F": baseline + 0.01,
                        "M": baseline + 0.005,
                        "V": baseline + 0.015,
                    }
                )

    sensitivity = e1_sensitivity_statistics(pd.DataFrame(rows))
    aggregate_c = sensitivity[
        (sensitivity.scope == "Aggregate")
        & (sensitivity.variant == "k5")
        & (sensitivity.metric == "C")
    ].iloc[0]

    assert aggregate_c.n_pairs == 4
    np.testing.assert_allclose(aggregate_c.mean_difference, 0.02)
    assert aggregate_c.p_holm < 0.05
    assert make_e1_sensitivity_figure(sensitivity, tmp_path) == [
        "e1_sensitivity_analysis"
    ]
    assert (
        tmp_path
        / "e1_sensitivity_analysis"
        / "e1_sensitivity_analysis.png"
    ).exists()


def test_figures_save_exact_source_data_without_experiment_inputs(tmp_path):
    generated = make_figures(sample_rows(), pd.DataFrame(), tmp_path)

    assert generated == ["e1_component_profile", "e2_dose_response"]
    for name in generated:
        assert (tmp_path / name / "source_data.csv").exists()
        assert (tmp_path / name / f"{name}.png").exists()
        assert (tmp_path / name / f"{name}.pdf").exists()


def test_tuning_convergence_is_rebuilt_from_saved_trials(tmp_path):
    checkpoint = tmp_path / "D8" / "include_primary_protected" / "gaussian_naive_bayes.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "dataset_id": "D8",
                "protected_policy": "include_primary_protected",
                "model_name": "gaussian_naive_bayes",
                "budget": "pilot",
                "tuning": {
                    "models": {
                        "gaussian_naive_bayes": {
                            "trials": [
                                {"trial_number": 0, "hyperparameters": {"C": 1}, "mean_validation_score": 0.7, "std_validation_score": 0.02},
                                {"trial_number": 1, "hyperparameters": {"C": 2}, "mean_validation_score": 0.8, "std_validation_score": 0.01},
                            ]
                        }
                    }
                },
            }
        )
    )

    trials = load_tuning_outputs(tmp_path)
    generated = make_tuning_figures(trials, tmp_path / "figures")

    assert trials.best_score_so_far.tolist() == [0.7, 0.8]
    assert generated == ["tuning_convergence"]
    assert (tmp_path / "figures" / "tuning_convergence" / "source_data.csv").exists()


def test_model_fairness_ratio_plot_includes_matched_zero_baseline(tmp_path):
    rows = []
    for repetition in (0, 1, 2):
        for model, baseline in (("logistic_regression", 0.9), ("random_forest", 0.85)):
            common = dict(
                dataset_id="D8", policy="include_primary_protected",
                repetition=repetition, configuration_id="abc", model=model,
            )
            rows.append({**common, "condition":"clean_baseline", "variant":"primary", "ratio":0.0, "model_F":baseline})
            for ratio in (0.05, 0.10):
                rows.append({**common, "condition":"contradictory_near_duplicate", "variant":"radius_0.05", "ratio":ratio, "model_F":baseline-ratio})

    generated = make_model_fairness_ratio_figures(pd.DataFrame(rows), tmp_path)

    assert len(generated) == 1
    source = pd.read_csv(tmp_path / generated[0] / "source_data.csv")
    assert sorted(source.ratio.unique()) == [0.0, 0.05, 0.1]
    assert set(source.model) == {"logistic_regression", "random_forest"}


def test_report_uses_publication_dataset_labels_in_tables_and_plots(
    tmp_path, monkeypatch
):
    summary = sample_rows()
    summary.loc[summary.index[:2], "dataset_id"] = "D6"
    summary.loc[summary.index[2:4], "dataset_id"] = "D7"
    models = pd.DataFrame({"dataset_id": ["D6", "D7", "D8"]})
    controls = pd.DataFrame({"dataset_id": ["D6", "D7", "D8"]})

    monkeypatch.setattr(
        "ifcred.reporting.report.load_condition_outputs",
        lambda _: (summary, models, controls),
    )
    monkeypatch.setattr(
        "ifcred.reporting.report.load_comparison_outputs",
        lambda _: pd.DataFrame({"dataset_id": ["D6", "D7", "D8"]}),
    )
    monkeypatch.setattr(
        "ifcred.reporting.report.load_tuning_outputs",
        lambda _: pd.DataFrame(),
    )

    output = tmp_path / "artifacts"
    generate_report(tmp_path / "results", output)

    for path in (output / "tables").glob("*.csv"):
        if not path.read_text().strip():
            continue
        frame = pd.read_csv(path)
        if "dataset_id" in frame:
            assert not set(frame.dataset_id.astype(str)) & {"D6", "D7", "D8"}
            assert set(frame.dataset_id.astype(str)) <= {"D1", "D2", "D3"}
    plot_source = pd.read_csv(
        output / "figures" / "e1_component_profile" / "source_data.csv"
    )
    assert set(plot_source.dataset_id) == {"D1", "D2", "D3"}


def test_report_reuses_cached_tables_when_saved_inputs_are_unchanged(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    results.mkdir()
    output = tmp_path / "artifacts"
    calls = {"conditions": 0}

    def load_conditions(_):
        calls["conditions"] += 1
        return sample_rows(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        "ifcred.reporting.report.load_condition_outputs", load_conditions
    )
    monkeypatch.setattr(
        "ifcred.reporting.report.load_comparison_outputs", lambda _: pd.DataFrame()
    )
    monkeypatch.setattr(
        "ifcred.reporting.report.load_tuning_outputs", lambda _: pd.DataFrame()
    )
    for function in (
        "make_tuning_figures",
        "make_figures",
        "make_dataset_response_figures",
        "make_model_fairness_ratio_figures",
        "make_model_fairness_vs_v_figure",
        "make_e1_predictive_performance_figure",
        "make_e1_predictive_performance_by_dataset_figure",
    ):
        monkeypatch.setattr(f"ifcred.reporting.report.{function}", lambda *a, **k: [])

    first = generate_report(results, output)
    second = generate_report(results, output)

    assert calls["conditions"] == 1
    assert first["report_cache_used"] is False
    assert second["report_cache_used"] is True
    assert (output / "tables" / "report_cache.json").exists()


def test_dataset_and_aggregate_response_figures_use_seed_level_confidence(
    tmp_path,
):
    scenarios = (
        ("model_family_disagreement", "affected_fraction_0.6"),
        ("benign_near_duplicate", "radius_0.05"),
        ("isolated_instance", "max_similarity_0.05"),
        ("dominant_neighbour_pair", "background_similarity_0.05"),
        ("metric_disagreement_instance", "strict_target_reliability_0.75"),
        ("contradictory_near_duplicate", "radius_0.05"),
    )
    summary_rows = []
    comparison_rows = []
    for dataset_offset, dataset in enumerate(("D1", "D2")):
        for repetition in (0, 1):
            baseline = 0.9 - 0.02 * dataset_offset + 0.01 * repetition
            summary_rows.append(
                {
                    "dataset_id": dataset,
                    "policy": "include_primary_protected",
                    "repetition": repetition,
                    "condition": "clean_baseline",
                    "variant": "primary",
                    "ratio": 0.0,
                    "C": baseline,
                    "D": baseline,
                    "F": baseline,
                    "M": baseline,
                    "V": baseline / 2,
                }
            )
            for condition, variant in scenarios:
                summary_rows.append(
                    {
                        "dataset_id": dataset,
                        "policy": "include_primary_protected",
                        "repetition": repetition,
                        "condition": condition,
                        "variant": variant,
                        "ratio": 0.1,
                        "C": baseline - 0.05,
                        "D": baseline - 0.04,
                        "F": baseline - 0.03,
                        "M": baseline - 0.02,
                        "V": baseline / 2 - 0.04,
                    }
                )
            for framework in ("VF1", "VF2", "VF3_IFT_V"):
                for model_offset, model in enumerate(("a", "b")):
                    common = {
                        "dataset_id": dataset,
                        "repetition": repetition,
                        "framework": framework,
                        "applicable": True,
                        "model": model,
                        "detection_rate": 0.1 + 0.01 * model_offset,
                        "lower_confidence_statistic": 1.2
                        + 0.01 * model_offset,
                    }
                    comparison_rows.append(
                        {
                            **common,
                            "condition": "clean_baseline",
                            "variant": "primary",
                            "ratio": 0.0,
                        }
                    )
                    for condition, variant in scenarios:
                        comparison_rows.append(
                            {
                                **common,
                                "condition": condition,
                                "variant": variant,
                                "ratio": 0.1,
                                "detection_rate": 0.2 + 0.01 * model_offset,
                                "lower_confidence_statistic": 1.4
                                + 0.01 * model_offset,
                            }
                        )

    summary = pd.DataFrame(summary_rows)
    excluded = summary.copy()
    excluded["policy"] = "exclude_protected"
    generated = make_dataset_response_figures(
        pd.concat([summary, excluded], ignore_index=True),
        pd.DataFrame(comparison_rows),
        tmp_path,
    )

    assert generated == [
        "e2_component_response_d1_include_protected",
        "e2_component_response_d2_include_protected",
        "e2_component_response_all_datasets_include_protected",
        "e2_component_response_d1_exclude_protected",
        "e2_component_response_d2_exclude_protected",
        "e2_component_response_all_datasets_exclude_protected",
        "e3_framework_comparison_d1",
        "e3_framework_comparison_d2",
        "e3_framework_comparison_all_datasets",
    ]
    component_source = pd.read_csv(
        tmp_path
        / "e2_component_response_all_datasets_include_protected"
        / "source_data.csv"
    )
    assert set(component_source["count"]) == {2}
    assert set(component_source["policy"]) == {"protected included"}
    framework_source = pd.read_csv(
        tmp_path / "e3_framework_comparison_all_datasets" / "source_data.csv"
    )
    assert set(framework_source["count"]) == {2}
    assert (framework_source.ci_high >= framework_source["mean"]).all()
    assert (framework_source.ci_low <= framework_source["mean"]).all()
