from __future__ import annotations
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from ifcred.reporting.statistics import separability_statistics


_RESPONSE_SCENARIOS = (
    (
        "benign_near_duplicate",
        ("radius_0.05", "radius_0.01", "radius_0.15"),
        "Clean dense clusters (S1)",
    ),
    (
        "contradictory_near_duplicate",
        ("radius_0.05", "radius_0.01", "radius_0.15"),
        "Contradictory near-duplicates (S2)",
    ),
    (
        "isolated_instance",
        ("max_similarity_0.05", "max_similarity_0.1", "max_similarity_0.2"),
        "Sparse isolated instances (S3)",
    ),
    (
        "dominant_neighbour_pair",
        (
            "background_similarity_0.05",
            "background_similarity_0.1",
            "background_similarity_0.2",
        ),
        "Dominant neighbour (S4)",
    ),
        (
            "metric_disagreement_instance",
            (
                "strict_target_reliability_v2_0.75",
                "strict_target_reliability_v2_0.9",
                "strict_target_reliability_0.75",
                "strict_target_reliability_0.9",
                "target_reliability_0.5",
            "target_reliability_0.75",
        ),
        "Metric disagreement geometry (S5)",
    ),
    (
        "model_family_disagreement",
        ("affected_fraction_0.6", "affected_fraction_0.2", "affected_fraction_1"),
        "Model-family disagreement (S6)",
    ),
)

_COMPONENT_LABELS = {
    "C": "Coverage C",
    "D": "Distance stability D",
    "F": "Individual fairness F",
    "M": "Model stability M",
}

_LINE_STYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1, 1, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
)
_MARKERS = ("o", "s", "^", "D", "P", "X")


def _selected_response_scenarios(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Select one declared severity per response mechanism."""

    available = set(zip(frame.condition.astype(str), frame.variant.astype(str)))
    selected = []
    for condition, variants, title in _RESPONSE_SCENARIOS:
        variant = next(
            (candidate for candidate in variants if (condition, candidate) in available),
            None,
        )
        if variant is not None:
            selected.append((condition, variant, title))
    return selected


def _seed_confidence_summary(
    frame: pd.DataFrame, value: str, *, group: list[str]
) -> pd.DataFrame:
    """Compute a normal 95% CI from independent outer-repetition means."""

    seed_means = (
        frame.groupby(group + ["repetition"], as_index=False, dropna=False)[value]
        .mean()
        .dropna(subset=[value])
    )
    if seed_means.empty:
        return pd.DataFrame()
    result = (
        seed_means.groupby(group, as_index=False, dropna=False)[value]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    result["sem"] = result["std"].fillna(0.0) / np.sqrt(result["count"])
    result["ci_low"] = result["mean"] - 1.96 * result["sem"]
    result["ci_high"] = result["mean"] + 1.96 * result["sem"]
    return result


def _response_scope(frame: pd.DataFrame, dataset: str | None) -> pd.DataFrame:
    if dataset is None:
        return frame.copy()
    return frame[frame.dataset_id == dataset].copy()


def _scenario_curve(
    frame: pd.DataFrame,
    condition: str,
    variant: str,
) -> pd.DataFrame:
    baseline = frame[
        (frame.condition == "clean_baseline") & (frame.variant == "primary")
    ].copy()
    baseline["ratio"] = 0.0
    injected = frame[
        (frame.condition == condition) & (frame.variant == variant)
    ].copy()
    return pd.concat([baseline, injected], ignore_index=True)


def _plot_confidence_curve(
    ax,
    source: pd.DataFrame,
    *,
    label: str,
    color,
    linestyle="-",
    marker: str = "o",
) -> None:
    source = source.sort_values("ratio")
    x = source.ratio.to_numpy(float)
    mean = source["mean"].to_numpy(float)
    low = source.ci_low.to_numpy(float)
    high = source.ci_high.to_numpy(float)
    ax.fill_between(
        x,
        low,
        high,
        color=color,
        alpha=0.10,
        edgecolor=color,
        linewidth=0.7,
    )
    ax.plot(
        x,
        mean,
        marker=marker,
        markersize=9,
        markerfacecolor="none",
        markeredgewidth=1.25,
        linewidth=2.0,
        linestyle=linestyle,
        color=color,
        alpha=0.82,
        label=label,
    )
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack((mean - low, high - mean)),
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        alpha=0.65,
    )


def _make_component_response_scope(
    scoped: pd.DataFrame,
    scenarios: list[tuple[str, str, str]],
    root: Path,
    *,
    scope_label: str,
    safe_scope: str,
    policy_label: str,
    policy_suffix: str,
) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    source_rows = []
    component_colors = plt.get_cmap("tab10").colors[:4]
    for ax, (condition, variant, title) in zip(axes.flat, scenarios):
        curve = _scenario_curve(scoped, condition, variant)
        for line_index, ((component, line_label), color) in enumerate(
            zip(_COMPONENT_LABELS.items(), component_colors)
        ):
            values = _seed_confidence_summary(curve, component, group=["ratio"])
            if values.empty:
                continue
            values["scope"] = scope_label
            values["policy"] = policy_label
            values["scenario"] = title
            values["condition"] = condition
            values["variant"] = variant
            values["component"] = component
            source_rows.append(values)
            _plot_confidence_curve(
                ax,
                values,
                label=line_label,
                color=color,
                linestyle=_LINE_STYLES[line_index],
                marker=_MARKERS[line_index],
            )
        ax.set_title(title, fontsize=15)
        ax.set_ylabel("Mean component score", fontsize=14)
        ax.set_xlabel("Synthetic instance ratio (ρ)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.985),
    )
    # Aggregate figures are intended to be embedded with a manuscript-level
    # caption, so leave out the redundant figure-level title.  Keep titles on
    # individual panels because they identify the response scenarios.
    if scope_label != "All datasets":
        fig.suptitle(
            f"IF-CRED component response · {scope_label} · {policy_label} · "
            "95% CI across seeds",
            y=1.02,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    source = pd.concat(source_rows, ignore_index=True)
    name = f"e2_component_response_{safe_scope}_{policy_suffix}"
    _save(fig, root, name, source)
    return name


def make_dataset_response_figures(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_root: str | Path,
) -> list[str]:
    """Create per-dataset and aggregate manuscript-style response figures."""

    if summary.empty:
        return []
    root = Path(output_root)
    datasets = sorted(summary.dataset_id.unique())
    scopes: list[tuple[str | None, str, str]] = [
        (dataset, dataset, dataset.lower()) for dataset in datasets
    ] + [(None, "All datasets", "all_datasets")]
    generated: list[str] = []

    policy_specs = (
        (
            "include_primary_protected",
            "protected included",
            "include_protected",
        ),
        ("exclude_protected", "protected excluded", "exclude_protected"),
    )
    for policy, policy_label, policy_suffix in policy_specs:
        if "policy" not in summary or not (summary.policy == policy).any():
            continue
        policy_frame = summary[summary.policy == policy].copy()
        scenarios = _selected_response_scenarios(policy_frame)
        if not scenarios:
            continue
        for dataset, scope_label, safe_scope in scopes:
            generated.append(
                _make_component_response_scope(
                    _response_scope(policy_frame, dataset),
                    scenarios,
                    root,
                    scope_label=scope_label,
                    safe_scope=safe_scope,
                    policy_label=policy_label,
                    policy_suffix=policy_suffix,
                )
            )

    primary = summary.copy()
    if "policy" in primary and (primary.policy == "include_primary_protected").any():
        primary = primary[primary.policy == "include_primary_protected"].copy()
    scenarios = _selected_response_scenarios(primary)
    comparison_columns = {
            "applicable",
            "condition",
            "variant",
            "framework",
            "ratio",
            "repetition",
    }
    if (
        not scenarios
        or comparisons.empty
        or not comparison_columns.issubset(comparisons.columns)
    ):
        return generated

    for dataset, scope_label, safe_scope in scopes:
        scoped = _response_scope(primary, dataset)
        applicable = comparisons[comparisons.applicable == True].copy()  # noqa: E712
        applicable = _response_scope(applicable, dataset)
        proposed = scoped.copy()
        panels = (
            ("IF-CRED", proposed, "V", "Composite score V", False),
            (
                "VF1",
                applicable[applicable.framework == "VF1"],
                "detection_rate",
                "Violation rate among checked pairs",
                True,
            ),
            (
                "VF2",
                applicable[applicable.framework == "VF2"],
                "lower_confidence_statistic",
                "Lower confidence statistic",
                True,
            ),
            (
                "VF3",
                applicable[applicable.framework == "VF3_IFT_V"],
                "detection_rate",
                "Valid discriminatory fraction",
                True,
            ),
        )
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
        source_rows = []
        scenario_colors = plt.get_cmap("tab10").colors[: len(scenarios)]
        for ax, (panel, panel_frame, value, ylabel, reverse_axis) in zip(
            axes.flat, panels
        ):
            for line_index, ((condition, variant, title), color) in enumerate(
                zip(scenarios, scenario_colors)
            ):
                curve = _scenario_curve(panel_frame, condition, variant)
                values = _seed_confidence_summary(curve, value, group=["ratio"])
                if values.empty:
                    continue
                values["scope"] = scope_label
                values["panel"] = panel
                values["scenario"] = title
                values["condition"] = condition
                values["variant"] = variant
                values["metric"] = value
                source_rows.append(values)
                _plot_confidence_curve(
                    ax,
                    values,
                    label=title,
                    color=color,
                    linestyle=_LINE_STYLES[line_index],
                    marker=_MARKERS[line_index],
                )
            ax.set_title(panel, fontsize=15)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_xlabel("Synthetic instance ratio (ρ)", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(alpha=0.25)
            if reverse_axis:
                ax.invert_yaxis()
        handles, labels = axes.flat[0].get_legend_handles_labels()
        # Matplotlib fills multi-column legends down columns. Reorder the
        # handles so the displayed reading order is row-wise: S1, S2, S3,
        # then S4, S5, S6 from left to right.
        legend_columns = 3
        ordered = sorted(
            zip(labels, handles),
            key=lambda item: int(re.search(r"S(\d+)", item[0]).group(1)),
        )
        ordered_labels, ordered_handles = zip(*ordered)
        legend_rows = int(np.ceil(len(ordered_handles) / legend_columns))
        display_order = [
            row * legend_columns + column
            for column in range(legend_columns)
            for row in range(legend_rows)
            if row * legend_columns + column < len(ordered_handles)
        ]
        handles = [ordered_handles[index] for index in display_order]
        labels = [ordered_labels[index] for index in display_order]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            fontsize=13,
            bbox_to_anchor=(0.5, 0.985),
        )
        if scope_label != "All datasets":
            fig.suptitle(
                f"Framework response comparison · {scope_label} · protected included · "
                "95% CI across seeds",
                y=1.02,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.92))
        source = pd.concat(source_rows, ignore_index=True)
        name = f"e3_framework_comparison_{safe_scope}"
        _save(fig, root, name, source)
        generated.append(name)
    return generated


def _save(fig, root: Path, name: str, source: pd.DataFrame) -> None:
    directory = root / name; directory.mkdir(parents=True, exist_ok=True)
    source.to_csv(directory / "source_data.csv", index=False)
    # 300 dpi is a standard minimum for raster figures in CS manuscripts.
    fig.savefig(directory / f"{name}.png", dpi=500, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def make_model_fairness_ratio_figures(
    models: pd.DataFrame, output_root: str | Path
) -> list[str]:
    """Create one repeated-seed model-F dose-response figure per E2 arm."""

    if models.empty or "model_F" not in models:
        return []
    injected = models[models.condition != "clean_baseline"].copy()
    clean = models[
        (models.condition == "clean_baseline") & (models.variant == "primary")
    ].copy()
    if injected.empty or clean.empty:
        return []
    match = ["dataset_id", "policy", "repetition", "configuration_id", "model"]
    clean = clean[match + ["model_F"]].rename(columns={"model_F": "clean_model_F"})
    root = Path(output_root)
    generated: list[str] = []
    for (condition, variant, policy), values in injected.groupby(
        ["condition", "variant", "policy"], dropna=False
    ):
        matched = values.merge(clean, on=match, how="left", validate="many_to_one")
        if matched.clean_model_F.isna().any():
            raise ValueError(
                "model-F ratio plot requires a matched primary clean baseline"
            )
        baseline = matched[match + ["clean_model_F"]].drop_duplicates(match)
        baseline = baseline.rename(columns={"clean_model_F": "model_F"})
        baseline["ratio"] = 0.0
        observed = matched[match + ["ratio", "model_F"]]
        repeated = pd.concat([baseline, observed], ignore_index=True)
        source = (
            repeated.groupby(["dataset_id", "model", "ratio"], as_index=False)
            .model_F.agg(["mean", "std", "count"])
            .reset_index()
        )
        source["sem"] = source["std"].fillna(0.0) / source["count"].pow(0.5)
        source["ci_low"] = (source["mean"] - 1.96 * source["sem"]).clip(0.0, 1.0)
        source["ci_high"] = (source["mean"] + 1.96 * source["sem"]).clip(0.0, 1.0)
        source["condition"] = condition
        source["variant"] = variant
        source["policy"] = policy
        datasets = sorted(source.dataset_id.unique())
        fig, axes = plt.subplots(
            1,
            len(datasets),
            figsize=(5.2 * len(datasets), 4.5),
            sharey=True,
            squeeze=False,
        )
        for ax, dataset in zip(axes[0], datasets):
            selected = source[source.dataset_id == dataset]
            for model, curve in selected.groupby("model"):
                curve = curve.sort_values("ratio")
                x = curve.ratio.to_numpy(float)
                mean = curve["mean"].to_numpy(float)
                ax.plot(x, mean, marker="o", label=model.replace("_", " ").title())
                if (curve["count"] > 1).any():
                    ax.fill_between(
                        x,
                        curve.ci_low.to_numpy(float),
                        curve.ci_high.to_numpy(float),
                        alpha=0.12,
                    )
            ax.set_title(dataset)
            ax.set_xlabel("Synthetic-to-real ratio (ρ)")
            ax.grid(alpha=0.25)
            ax.set_xticks(sorted(source.ratio.unique()))
        axes[0, 0].set_ylabel("Individual fairness F")
        axes[0, -1].legend(fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.suptitle(f"{condition} · {variant} · {policy}")
        name = _safe_name(f"e2_model_f_{condition}_{variant}_{policy}")
        _save(fig, root, name, source)
        generated.append(name)
    return generated


def make_model_fairness_vs_v_figure(
    models: pd.DataFrame,
    output_root: str | Path,
    *,
    separability: pd.DataFrame | None = None,
) -> list[str]:
    """Plot model-level fairness against composite V for clean E1 runs."""

    required = {"model_F", "V", "model", "dataset_id", "policy", "repetition"}
    if models.empty or not required.issubset(models.columns):
        return []
    source = models[
        (models.condition == "clean_baseline")
        & (models.variant == "primary")
    ].copy()
    if source.empty:
        return []
    if separability is None:
        separability = separability_statistics(source)
    source = source[
        ["dataset_id", "policy", "repetition", "model", "model_F", "V"]
    ].dropna()
    root = Path(output_root)
    datasets = sorted(source.dataset_id.unique())
    # Keep the datasets side by side so the figure can be compared at a glance
    # and inserted naturally as a landscape manuscript figure.
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(18.0, 7.733),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0, :]
    models_order = sorted(source.model.unique())
    colors = plt.get_cmap("tab10").colors
    model_labels = {
        "logistic_regression": "LR",
        "mlp": "MLP",
        "gaussian_naive_bayes": "NB",
        "random_forest": "RF",
        "decision_tree": "DT",
    }
    desired_model_order = [
        model for model in (
            "logistic_regression",
            "mlp",
            "gaussian_naive_bayes",
            "random_forest",
            "decision_tree",
        ) if model in models_order
    ]
    policy_markers = {
        "exclude_protected": "X",
        "include_primary_protected": "o",
    }
    for ax, dataset in zip(axes, datasets):
        subset = source[source.dataset_id == dataset]
        for index, model in enumerate(desired_model_order):
            for policy, marker in policy_markers.items():
                values = subset[(subset.model == model) & (subset.policy == policy)]
                if values.empty:
                    continue
                ax.scatter(
                    values.model_F,
                    values.V,
                    s=160,
                    alpha=0.52,
                    color=colors[index % len(colors)],
                    marker=marker,
                    edgecolors="none",
                )
        ax.set_title(dataset, fontsize=14, pad=8)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
        panel_stats = separability[separability.dataset_id == dataset]
        model_stats = panel_stats[panel_stats.comparison == "model"]
        policy_stats = panel_stats[panel_stats.comparison == "protected_policy"]
        if not model_stats.empty or not policy_stats.empty:
            lines = []
            if not model_stats.empty:
                value = model_stats.mahalanobis_distance.median()
                low = model_stats.ci_low.median(); high = model_stats.ci_high.median()
                lines.append(f"Model separation: {value:.2f} [{low:.2f}, {high:.2f}]")
            if not policy_stats.empty:
                value = policy_stats.mahalanobis_distance.median()
                low = policy_stats.ci_low.median(); high = policy_stats.ci_high.median()
                lines.append(f"A-policy separation: {value:.2f} [{low:.2f}, {high:.2f}]")
            ax.text(
                0.045, 0.045, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=14,
                bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.85, "pad": 4},
            )
    model_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=colors[index % len(colors)], markeredgecolor="none",
            markersize=10.5, label=model_labels.get(model, model),
        )
        for index, model in enumerate(desired_model_order)
    ]
    policy_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", color="black", markersize=10.5, label=label)
        for marker, label in (("X", "$A$ excluded"), ("o", "$A$ included"))
    ]
    # Matplotlib fills multi-column legends down columns. Arrange the handles
    # so the visible rows read: LR MLP NB / RF DT / A excluded A included.
    legend_handles = [
        model_handles[0], model_handles[3], policy_handles[0],
        model_handles[1], model_handles[4], policy_handles[1],
        model_handles[2],
        Line2D([], [], linestyle="none", label=""),
        Line2D([], [], linestyle="none", label=""),
    ]
    axes[0].legend(
        handles=legend_handles,
        fontsize=13,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.2,
        columnspacing=0.8,
        handletextpad=0.35,
        framealpha=0.9,
    )
    # Explicit figure labels with compact margins avoid the excessive gap
    # introduced by constrained placement of supxlabel/supylabel.
    fig.text(0.5, 0.020, "Model-level individual fairness ($F_m$)",
             ha="center", va="bottom", fontsize=16)
    fig.text(0.030, 0.5, "Composite validation score ($V$)",
             ha="center", va="center", rotation="vertical", fontsize=16)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.93, wspace=0.10)
    _save(fig, root, "e1_model_fairness_vs_v", source)
    return ["e1_model_fairness_vs_v"]


def make_e1_predictive_performance_figure(
    models: pd.DataFrame, output_root: str | Path
) -> list[str]:
    """Plot calibrated predictive performance for clean E1 runs."""
    required = {"dataset_id", "model", "policy", "condition", "variant"}
    metrics = {
        "calibrated_accuracy": "Accuracy ↑",
        "calibrated_roc_auc": "AUROC ↑",
        "calibrated_brier_score": "Brier score ↓",
        "calibrated_expected_calibration_error": "Expected calibration error ↓",
    }
    if models.empty or not required.issubset(models.columns):
        return []
    source = models[(models.condition == "clean_baseline") & (models.variant == "primary")].copy()
    if source.empty or not set(metrics).issubset(source.columns):
        return []
    source["dataset_id"] = source["dataset_id"].replace({"D6": "D1", "D7": "D2", "D8": "D3"})
    source = source.groupby(["dataset_id", "model", "policy"], as_index=False)[list(metrics)].mean()
    model_order = ["logistic_regression", "mlp", "gaussian_naive_bayes", "random_forest", "decision_tree"]
    model_labels = {"logistic_regression": "LR", "mlp": "MLP", "gaussian_naive_bayes": "NB", "random_forest": "RF", "decision_tree": "DT"}
    dataset_order = [d for d in ("D1", "D2", "D3") if d in set(source.dataset_id)]
    colors = dict(zip(model_order, plt.get_cmap("tab10").colors[:5]))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True)
    policies = [("exclude_protected", "A excluded", "//"), ("include_primary_protected", "A included", "")]
    width = 0.78 / (len(model_order) * len(policies))
    for ax, (metric, title) in zip(axes.flat, metrics.items()):
        for dataset_index, dataset in enumerate(dataset_order):
            for model_index, model in enumerate(model_order):
                for policy_index, (policy, _policy_label, hatch) in enumerate(policies):
                    row = source[(source.dataset_id == dataset) & (source.model == model) & (source.policy == policy)]
                    if row.empty:
                        continue
                    index = model_index * len(policies) + policy_index
                    position = dataset_index - 0.39 + width / 2 + index * width
                    ax.bar(position, row.iloc[0][metric], width=width,
                           color=colors[model], alpha=0.85, hatch=hatch,
                           edgecolor="black", linewidth=0.35)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_xticks(range(len(dataset_order)), dataset_order)
    legend_handles = [Patch(facecolor=colors[m], edgecolor="black", label=model_labels[m]) for m in model_order]
    legend_handles += [Patch(facecolor="white", edgecolor="black", hatch=h, label=label) for _, label, h in policies]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(legend_handles), fontsize=9, columnspacing=0.8, framealpha=0.9)
    fig.text(0.5, 0.03, "Dataset", ha="center", fontsize=16)
    fig.text(0.025, 0.5, "Performance metric", va="center", rotation="vertical", fontsize=16)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.86, wspace=0.18, hspace=0.28)
    _save(fig, Path(output_root), "e1_predictive_performance", source)
    return ["e1_predictive_performance"]


def make_e1_predictive_performance_by_dataset_figure(
    models: pd.DataFrame, output_root: str | Path
) -> list[str]:
    """Plot calibrated E1 performance as dataset-specific small multiples."""
    metrics = {
        "calibrated_accuracy": "Accuracy",
        "calibrated_roc_auc": "AUROC",
        "calibrated_brier_score": "Brier score",
        "calibrated_expected_calibration_error": "ECE",
    }
    if models.empty or not {"dataset_id", "model", "policy", "condition", "variant"}.issubset(models.columns):
        return []
    source = models[(models.condition == "clean_baseline") & (models.variant == "primary")].copy()
    if source.empty or not set(metrics).issubset(source.columns):
        return []
    source["dataset_id"] = source["dataset_id"].replace({"D6": "D1", "D7": "D2", "D8": "D3"})
    source = source.groupby(["dataset_id", "model", "policy"], as_index=False)[list(metrics)].mean()
    model_order = ["logistic_regression", "mlp", "gaussian_naive_bayes", "random_forest", "decision_tree"]
    model_labels = {"logistic_regression": "LR", "mlp": "MLP", "gaussian_naive_bayes": "NB", "random_forest": "RF", "decision_tree": "DT"}
    datasets = [d for d in ("D1", "D2", "D3") if d in set(source.dataset_id)]
    colors = dict(zip(model_order, plt.get_cmap("tab10").colors[:5]))
    policies = [("exclude_protected", "A excluded", "//"), ("include_primary_protected", "A included", "")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 10.5), sharey=True)
    block_size = len(model_order) * len(policies)
    y_positions = []
    for dataset_index in range(len(datasets)):
        start = dataset_index * (block_size + 2)
        y_positions.extend(range(start, start + block_size))
    for ax, (metric, title) in zip(axes, metrics.items()):
        for dataset_index, dataset in enumerate(datasets):
            data = source[source.dataset_id == dataset]
            start = dataset_index * (block_size + 2)
            for model_index, model in enumerate(model_order):
                for policy_index, (policy, _label, hatch) in enumerate(policies):
                    row = data[(data.model == model) & (data.policy == policy)]
                    if row.empty:
                        continue
                    y = start + model_index * len(policies) + policy_index
                    ax.barh(y, row.iloc[0][metric], height=0.78, color=colors[model],
                            hatch=hatch, edgecolor="black", linewidth=0.3, alpha=0.85)
            if dataset_index < len(datasets) - 1:
                ax.axhline(start + block_size + 0.75, color="0.6", linewidth=0.8)
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([model_labels[m] + ("−" if p == "exclude_protected" else "+")
                             for _d in datasets for m in model_order for p, _l, _h in policies])
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    model_handles = [Patch(facecolor=colors[m], edgecolor="black", label=model_labels[m]) for m in model_order]
    policy_handles = [Patch(facecolor="white", edgecolor="black", hatch=h, label=label) for _, label, h in policies]
    fig.legend(model_handles + policy_handles, [h.get_label() for h in model_handles + policy_handles], loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=7, fontsize=10, framealpha=0.9)
    for dataset_index, dataset in enumerate(datasets):
        center = dataset_index * (block_size + 2) + (block_size - 1) / 2
        axes[0].text(-0.19, center, dataset, transform=axes[0].get_yaxis_transform(),
                     ha="right", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.025, "Mean calibrated performance", ha="center", fontsize=16)
    fig.text(0.025, 0.5, "Dataset / model / policy", va="center", rotation="vertical", fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.09, top=0.88, wspace=0.18)
    _save(fig, Path(output_root), "e1_predictive_performance_by_dataset", source)
    return ["e1_predictive_performance_by_dataset"]


def make_e1_sensitivity_figure(
    sensitivity: pd.DataFrame, output_root: str | Path
) -> list[str]:
    """Plot dataset-specific and aggregate clean-E1 sensitivity effects."""

    required = {
        "scope",
        "variant",
        "setting",
        "metric",
        "mean_difference",
        "ci_low",
        "ci_high",
        "p_holm",
    }
    if sensitivity.empty or not required.issubset(sensitivity.columns):
        return []
    sensitivity = sensitivity.copy()
    sensitivity["scope"] = sensitivity["scope"].replace(
        {"D6": "D1", "D7": "D2", "D8": "D3"}
    )
    metrics = ("C", "D", "F", "M", "V")
    variants = (
        "k5",
        "k20",
        "euclidean_manhattan",
        "alternate_bandwidth",
        "native_probability",
    )
    setting_labels = {
        "k5": r"$k=5$",
        "k20": r"$k=20$",
        "euclidean_manhattan": r"$\mathcal{R} \setminus r_2$",
        "alternate_bandwidth": "Alternative bandwidth",
        "native_probability": "Native probabilities",
    }
    y_positions = {variant: index for index, variant in enumerate(variants)}
    dataset_specs = (
        ("D1", "o", "tab:blue", -0.18),
        ("D2", "o", "tab:orange", -0.06),
        ("D3", "o", "tab:green", 0.06),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), squeeze=False)
    metric_axes = list(zip(axes.flat, metrics))
    for panel_index, (ax, metric) in enumerate(metric_axes):
        selected = sensitivity[sensitivity.metric.eq(metric)]
        for scope, marker, color, offset in dataset_specs:
            values = selected[selected.scope.eq(scope)]
            for row in values.itertuples(index=False):
                ax.scatter(
                    row.mean_difference,
                    y_positions[row.variant] + offset,
                    s=96,
                    marker=marker,
                    color=color,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3,
                )
        aggregate = selected[selected.scope.eq("Aggregate")]
        for row in aggregate.itertuples(index=False):
            y = y_positions[row.variant] + 0.18
            significant = float(row.p_holm) < 0.05
            ax.errorbar(
                row.mean_difference,
                y,
                xerr=[
                    [row.mean_difference - row.ci_low],
                    [row.ci_high - row.mean_difference],
                ],
                fmt="D",
                markersize=6.5,
                markerfacecolor="black" if significant else "white",
                markeredgecolor="black",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
                zorder=4,
            )
        ax.axvline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel(r"Mean paired difference ($\Delta$)", fontsize=14)
        ax.set_yticks(
            list(y_positions.values()),
            [setting_labels.get(variant, variant) for variant in variants],
        )
        if panel_index not in (0, 3):
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.grid(axis="x", alpha=0.22)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=10,
            label=scope,
        )
        for scope, marker, color, _offset in dataset_specs
    ]
    handles.extend(
        [
            Line2D(
                [0], [0], marker="D", linestyle="-", color="black",
                markerfacecolor="black", markersize=11,
                label=r"Aggregate, $p<0.05$",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="-", color="black",
                markerfacecolor="white", markersize=11,
                label="Aggregate, not significant",
            ),
        ]
    )
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        framealpha=0.92,
        fontsize=14,
        handlelength=1.7,
    )
    legend_ax.text(
        0.5,
        0.24,
        "Horizontal bars show aggregate 95% confidence intervals.\n"
        "Differences are alternative minus primary specification.",
        ha="center",
        va="center",
        fontsize=12,
        transform=legend_ax.transAxes,
    )
    fig.subplots_adjust(
        left=0.17, right=0.985, bottom=0.09, top=0.95, wspace=0.2, hspace=0.3
    )
    _save(fig, Path(output_root), "e1_sensitivity_analysis", sensitivity)
    return ["e1_sensitivity_analysis"]


def make_figures(summary: pd.DataFrame, comparisons: pd.DataFrame, output_root: str | Path) -> list[str]:
    """Generate every figure solely from tabular outputs already on disk."""

    root = Path(output_root); generated = []
    if not summary.empty:
        clean = summary[(summary.condition == "clean_baseline") & (summary.variant == "primary")]
        if not clean.empty:
            source = clean.groupby(["dataset_id","policy"], as_index=False)[["C","D","F","M","V"]].mean()
            policy_order = {"exclude_protected": 0, "include_primary_protected": 1}
            source = source.assign(
                _policy_order=source.policy.map(policy_order).fillna(99),
                x_label=source.apply(
                    lambda row: f"{row.dataset_id}$_{{{'-' if row.policy == 'exclude_protected' else '+'}}}$",
                    axis=1,
                ),
            ).sort_values(["dataset_id", "_policy_order"])
            fig, ax = plt.subplots(figsize=(9,4.8))
            source.set_index("x_label")[["C","D","F","M","V"]].plot.bar(ax=ax)
            ax.set_ylim(0,1); ax.set_ylabel("Mean score"); ax.set_xlabel(""); ax.legend(ncol=5)
            source = source.drop(columns=["_policy_order", "x_label"])
            _save(fig, root, "e1_component_profile", source); generated.append("e1_component_profile")
        sensitivity = summary[
            (summary.condition == "clean_baseline")
            & (~summary.variant.isin(["primary", "native_probability"]))
        ]
        if not sensitivity.empty:
            source = sensitivity.groupby(["variant", "k", "metric_set", "bandwidth_policy"], as_index=False)[["C","D","F","M","V"]].mean()
            fig, ax = plt.subplots(figsize=(9,4.8)); source.set_index("variant")[["C","D","F","M","V"]].plot.bar(ax=ax)
            ax.set_ylim(0,1); ax.set_ylabel("Mean score"); ax.set_title("E1: audit-design sensitivity"); ax.legend(ncol=5)
            _save(fig, root, "e1_audit_sensitivity", source); generated.append("e1_audit_sensitivity")
        probability = summary[
            (summary.condition == "clean_baseline")
            & (summary.variant.isin(["primary", "native_probability"]))
        ]
        if not probability.empty and probability.variant.nunique() > 1:
            source = probability.groupby(["dataset_id","policy","variant"], as_index=False)[["F","M","V"]].mean()
            fig, ax = plt.subplots(figsize=(9,4.8)); source.set_index(["dataset_id","policy","variant"])[["F","M","V"]].plot.bar(ax=ax)
            ax.set_ylim(0,1); ax.set_ylabel("Mean score"); ax.set_title("E1: calibrated versus native probability sensitivity")
            _save(fig, root, "e1_probability_sensitivity", source); generated.append("e1_probability_sensitivity")
        injected = summary[summary.condition != "clean_baseline"]
        if not injected.empty:
            source = injected.groupby(["condition","variant","ratio"], as_index=False)[["C","D","F","M","V"]].mean()
            fig, axes = plt.subplots(1,5,figsize=(16,3.6),sharex=True)
            for ax, metric in zip(axes,("C","D","F","M","V")):
                for condition, values in source.groupby("condition"):
                    values.groupby("ratio")[metric].mean().plot(ax=ax, marker="o", label=condition)
                ax.set_title(metric); ax.set_ylim(0,1); ax.set_xlabel("Injection ratio")
            axes[0].set_ylabel("Mean score"); axes[-1].legend(fontsize=6, bbox_to_anchor=(1.04,1), loc="upper left")
            fig.suptitle("E2: component dose response")
            _save(fig, root, "e2_dose_response", source); generated.append("e2_dose_response")
    if not comparisons.empty and "detection_rate" in comparisons:
        source = comparisons.groupby(["framework","condition","ratio"], as_index=False)["detection_rate"].mean()
        fig, ax = plt.subplots(figsize=(9,4.8))
        comparator_order = ("VF1", "VF2", "VF3_IFT_V")
        comparator_labels = {
            "VF1": "VF1 (John et al., 2020)",
            "VF2": "VF2 (Maity et al., 2021)",
            "VF3_IFT_V": "VF3 (Kitamura et al., 2024; IFT-V)",
        }
        for framework in comparator_order:
            values = source[source.framework == framework]
            if not values.empty:
                values.groupby("ratio").detection_rate.mean().plot(
                    ax=ax,
                    marker="o",
                    label=comparator_labels[framework],
                )
        ax.set_ylim(0,1); ax.set_ylabel("Mean native detection rate"); ax.set_xlabel("Injection ratio"); ax.set_title("E3: prior-framework response")
        ax.legend(); _save(fig, root, "e3_framework_response", source); generated.append("e3_framework_response")
        if "runtime_seconds" in comparisons:
            runtime = comparisons.groupby(["framework","condition"], as_index=False).runtime_seconds.mean()
            runtime_plot = runtime.pivot(index="condition", columns="framework", values="runtime_seconds")
            runtime_plot = runtime_plot.reindex(columns=[name for name in comparator_order if name in runtime_plot]).rename(columns=comparator_labels)
            fig, ax = plt.subplots(figsize=(9,4.8)); runtime_plot.plot.bar(ax=ax)
            ax.set_yscale("log"); ax.set_ylabel("Mean runtime (seconds, log scale)"); ax.set_title("E3: prior-framework runtime")
            _save(fig, root, "e3_framework_runtime", runtime); generated.append("e3_framework_runtime")
    return generated


def make_tuning_figures(tuning: pd.DataFrame, output_root: str | Path) -> list[str]:
    """Plot cumulative optimization curves from saved trial checkpoints only."""

    if tuning.empty:
        return []
    root = Path(output_root)
    models = list(dict.fromkeys(tuning.model.tolist()))
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3 * len(models)), squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        selected = tuning[tuning.model == model]
        for (dataset, policy), values in selected.groupby(["dataset_id", "policy"]):
            values = values.sort_values("trial")
            ax.plot(values.trial, values.best_score_so_far, label=f"{dataset} · {policy}")
        ax.set_title(model); ax.set_ylabel("Best CV AUROC"); ax.set_xlabel("Trial")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Bayesian tuning convergence", y=1.0)
    fig.tight_layout()
    _save(fig, root, "tuning_convergence", tuning)
    return ["tuning_convergence"]
