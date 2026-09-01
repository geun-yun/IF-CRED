from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp, wilcoxon


def _mahalanobis_distance(first: np.ndarray, second: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return float(np.sqrt(max(0.0, delta @ np.linalg.pinv(covariance) @ delta)))


def _mahalanobis_distance_from_precision(
    first: np.ndarray, second: np.ndarray, precision: np.ndarray
) -> float:
    """Evaluate a Mahalanobis distance with a precomputed inverse covariance."""

    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return float(np.sqrt(max(0.0, delta @ precision @ delta)))


def separability_statistics(
    models: pd.DataFrame, *, bootstrap_seed: int = 104729, draws: int = 2000
) -> pd.DataFrame:
    """Estimate model and protected-policy separation in (F_m, V) space.

    Distances use a pooled, lightly regularized covariance per dataset. Bootstrap
    intervals quantify uncertainty from the finite seed sample.
    """
    required = {"dataset_id", "model", "policy", "repetition", "model_F", "V"}
    if models.empty or not required.issubset(models.columns):
        return pd.DataFrame()
    source = models[(models.condition == "clean_baseline") & (models.variant == "primary")].copy()
    source = source.dropna(subset=["dataset_id", "model", "policy", "repetition", "model_F", "V"])
    rows = []
    rng = np.random.default_rng(bootstrap_seed)
    for dataset, data in source.groupby("dataset_id"):
        points = data[["model_F", "V"]].to_numpy(float)
        covariance = np.cov(points, rowvar=False) if len(points) > 2 else np.eye(2)
        scale = max(float(np.trace(covariance)) / 2.0, 1e-8)
        covariance = covariance + np.eye(2) * (1e-6 * scale)
        # The covariance is fixed within a dataset.  Reusing its pseudoinverse
        # avoids performing the same matrix decomposition for every bootstrap
        # draw while leaving the bootstrap samples and resulting intervals
        # unchanged.
        precision = np.linalg.pinv(covariance)

        # Model separation is averaged over the two protected-attribute policies.
        model_groups = {m: g[["model_F", "V"]].to_numpy(float) for m, g in data.groupby("model")}
        model_names = sorted(model_groups)
        for index, first_name in enumerate(model_names):
            for second_name in model_names[index + 1:]:
                first, second = model_groups[first_name], model_groups[second_name]
                distance = _mahalanobis_distance_from_precision(
                    first.mean(0), second.mean(0), precision
                )
                boot = []
                for _ in range(draws):
                    boot.append(_mahalanobis_distance_from_precision(
                        first[rng.integers(len(first), size=len(first))].mean(0),
                        second[rng.integers(len(second), size=len(second))].mean(0),
                        precision,
                    ))
                rows.append({"dataset_id": dataset, "comparison": "model", "group_a": first_name,
                             "group_b": second_name, "policy": "all", "mahalanobis_distance": distance,
                             "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))})

        # Policy separation is computed within each model using paired repetitions.
        for model, group in data.groupby("model"):
            wide = group.pivot_table(index="repetition", columns="policy", values=["model_F", "V"], aggfunc="mean")
            if not {"exclude_protected", "include_primary_protected"}.issubset(wide.columns.get_level_values(1)):
                continue
            first = wide.xs("exclude_protected", level=1, axis=1)[["model_F", "V"]].dropna()
            second = wide.xs("include_primary_protected", level=1, axis=1)[["model_F", "V"]].dropna()
            paired = first.join(second, lsuffix="_a", rsuffix="_b", how="inner")
            if paired.empty:
                continue
            distance = _mahalanobis_distance_from_precision(
                paired[["model_F_a", "V_a"]].mean(0),
                paired[["model_F_b", "V_b"]].mean(0),
                precision,
            )
            boot = []
            for _ in range(draws):
                sample = paired.iloc[rng.integers(len(paired), size=len(paired))]
                boot.append(_mahalanobis_distance_from_precision(
                    sample[["model_F_a", "V_a"]].mean(0),
                    sample[["model_F_b", "V_b"]].mean(0),
                    precision,
                ))
            rows.append({"dataset_id": dataset, "comparison": "protected_policy", "group_a": "A excluded",
                         "group_b": "A included", "model": model, "policy": "paired",
                         "mahalanobis_distance": distance, "ci_low": float(np.quantile(boot, .025)),
                         "ci_high": float(np.quantile(boot, .975))})
    return pd.DataFrame(rows)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm family-wise adjusted p-values in their original order."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(
            running,
            min(1.0, (len(values) - rank) * float(values[position])),
        )
        adjusted[position] = running
    return adjusted


def e1_sensitivity_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched E1 audit-design and probability sensitivity effects.

    Protected-attribute policies are averaged within each dataset and repetition.
    The aggregate then averages the three datasets within each repetition, leaving
    20 repetition-level paired effects for inference in the frozen study.
    """

    metrics = ("C", "D", "F", "M", "V")
    required = {
        "dataset_id",
        "policy",
        "repetition",
        "condition",
        "variant",
        *metrics,
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    clean = frame[frame.condition.eq("clean_baseline")].copy()
    keys = ["dataset_id", "policy", "repetition"]
    primary = clean[clean.variant.eq("primary")].set_index(keys)[list(metrics)]
    if primary.empty:
        return pd.DataFrame()

    specifications = (
        ("k5", "$k=5$", "audit_design", metrics),
        ("k20", "$k=20$", "audit_design", metrics),
        (
            "euclidean_manhattan",
            "Euclidean + Manhattan",
            "audit_design",
            metrics,
        ),
        (
            "alternate_bandwidth",
            "Alternative bandwidth",
            "audit_design",
            metrics,
        ),
        (
            "native_probability",
            "Native probabilities",
            "probability",
            metrics,
        ),
    )
    rows = []
    for variant, setting_label, family, tested_metrics in specifications:
        alternative = clean[clean.variant.eq(variant)].set_index(keys)[list(metrics)]
        if alternative.empty:
            continue
        delta = (alternative - primary.loc[alternative.index]).reset_index()
        by_dataset_seed = delta.groupby(
            ["dataset_id", "repetition"], as_index=False
        )[list(metrics)].mean()
        scopes = {
            str(dataset): values.set_index("repetition")[list(metrics)]
            for dataset, values in by_dataset_seed.groupby("dataset_id")
        }
        scopes["Aggregate"] = by_dataset_seed.groupby("repetition")[
            list(metrics)
        ].mean()
        for scope, values in scopes.items():
            for metric in tested_metrics:
                sample = values[metric].dropna().to_numpy(float)
                if not len(sample):
                    continue
                mean = float(np.mean(sample))
                sd = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0
                if len(sample) < 2 or np.allclose(sample, 0.0):
                    ci_low = ci_high = mean
                    p_value = 1.0
                elif np.isclose(sd, 0.0):
                    ci_low = ci_high = mean
                    p_value = 0.0
                else:
                    sem = sd / np.sqrt(len(sample))
                    ci_low, ci_high = map(
                        float,
                        t.interval(
                            0.95,
                            len(sample) - 1,
                            loc=mean,
                            scale=sem,
                        ),
                    )
                    p_value = float(ttest_1samp(sample, 0.0).pvalue)
                rows.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "setting": setting_label,
                        "family": family,
                        "metric": metric,
                        "n_pairs": len(sample),
                        "mean_difference": mean,
                        "sd": sd,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_holm"] = np.nan
    for (_scope, _family), indexes in result.groupby(
        ["scope", "family"], sort=False
    ).groups.items():
        result.loc[indexes, "p_holm"] = _holm_adjust(
            result.loc[indexes, "p_value"].to_numpy(float)
        )
    return result


def repetition_summary(frame: pd.DataFrame, *, bootstrap_seed: int = 104729, draws: int = 2000) -> pd.DataFrame:
    """Summarize outer-repetition means with percentile bootstrap intervals."""

    if frame.empty:
        return frame.copy()
    groups = ["dataset_id","policy","configuration_id","condition","variant","target","ratio","k","primary_metric","metric_set","bandwidth_policy"]
    metrics = [name for name in ("C","D","F","M","V","F_min","V_worst") if name in frame]
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for keys, values in frame.groupby(groups, dropna=False):
        row = dict(zip(groups, keys)); row["n_repetitions"] = values["repetition"].nunique()
        for metric in metrics:
            sample = values[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(sample)); row[f"{metric}_sd"] = float(np.std(sample, ddof=1)) if len(sample)>1 else 0.0
            row[f"{metric}_median"] = float(np.median(sample))
            q25, q75 = np.quantile(sample, (0.25, 0.75))
            row[f"{metric}_q25"] = float(q25)
            row[f"{metric}_q75"] = float(q75)
            row[f"{metric}_iqr"] = float(q75 - q25)
            if len(sample) > 1:
                means = np.mean(rng.choice(sample, size=(draws,len(sample)), replace=True), axis=1)
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = map(float, np.quantile(means, (0.025,0.975)))
            else:
                row[f"{metric}_ci_low"] = row[f"{metric}_ci_high"] = float(sample[0])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_clean_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute injected-minus-clean deltas within matched outer repetitions."""

    if frame.empty:
        return frame.copy()
    keys = ["dataset_id","policy","repetition","configuration_id"]
    clean = frame[
        (frame.condition == "clean_baseline") & (frame.variant == "primary")
    ][keys + ["C","D","F","M","V"]].drop_duplicates(keys)
    injected = frame[frame.condition != "clean_baseline"].copy()
    merged = injected.merge(clean, on=keys, suffixes=("", "_clean"), how="left", validate="many_to_one")
    for metric in ("C","D","F","M","V"):
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_clean"]
    return merged


def component_response_tests(deltas: pd.DataFrame) -> pd.DataFrame:
    """One-sided paired stressor tests with Holm family-wise correction."""

    if deltas.empty:
        return deltas.copy()
    groups = ["dataset_id", "policy", "condition", "variant", "target", "ratio"]
    rows = []
    for keys, values in deltas.groupby(groups, dropna=False):
        target = str(keys[4])
        if target not in {"C", "D", "F", "M"}:
            continue
        sample = values[f"delta_{target}"].dropna().to_numpy(float)
        if len(sample) < 2 or np.allclose(sample, 0.0):
            statistic, p_value = np.nan, 1.0
        else:
            test = wilcoxon(sample, alternative="less", zero_method="wilcox")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append({**dict(zip(groups, keys)), "component_tested": target, "n_pairs": len(sample), "mean_delta": float(np.mean(sample)) if len(sample) else np.nan, "median_delta": float(np.median(sample)) if len(sample) else np.nan, "wilcoxon_statistic": statistic, "p_value": p_value})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = np.argsort(result.p_value.to_numpy())
    adjusted = np.empty(len(result), dtype=float)
    running = 0.0; total = len(result)
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (total-rank) * float(result.iloc[position].p_value)))
        adjusted[position] = running
    result["p_holm"] = adjusted
    return result
