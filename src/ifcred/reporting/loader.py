from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def load_condition_outputs(results_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flatten immutable manifests without loading models or rebuilding graphs."""

    summaries, models, controls = [], [], []
    for path in sorted(Path(results_root).glob("**/manifest.json")):
        manifest = json.loads(path.read_text())
        if "result" not in manifest or "context" not in manifest:
            continue
        context, result = manifest["context"], manifest["result"]
        assessment = result["assessment"]
        graph = result["graph_spec"]
        common = {
            "dataset_id": context["dataset"]["dataset_id"],
            "policy": context["setup"]["protected_policy"],
            "repetition": context["repetition"]["repetition"],
            "configuration_id": manifest["configuration_id"],
            "condition": result["condition"],
            "variant": result.get("condition_variant", "default"),
            "target": result["target"],
            "ratio": result["requested_injection_ratio"],
            "realized_ratio": result["realized_injection_ratio"],
            "k": graph["k"],
            "primary_metric": graph["primary_metric"],
            "metric_set": "+".join(item["name"] for item in graph["metrics"]),
            "bandwidth_policy": context["setup"]["graph_fitting_spec"]["bandwidth_policy"],
            "C": assessment["C"], "D": assessment["D"], "F": assessment["F"],
            "M": assessment["M"], "V": assessment["V"],
            "F_min": assessment["F_min"], "V_worst": assessment["V_worst"],
            "n_evaluation": result["n_evaluation"],
            "artifact_path": str(path.parent.resolve()),
        }
        summaries.append(common)
        for name, utility in result["predictive_utility"].items():
            models.append({**common, "model": name, **{f"native_{k}": v for k,v in utility["native"].items()}, **{f"calibrated_{k}": v for k,v in utility["calibrated"].items()}, "model_F": assessment["model_fairness"][name]})
        for name, values in result.get("prediction_controls", {}).items():
            controls.append({**common, "control": name, **values})
    return pd.DataFrame(summaries), pd.DataFrame(models), pd.DataFrame(controls)


def load_comparison_outputs(results_root: str | Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(results_root).glob("**/comparisons.csv")):
        frame = pd.read_csv(path)
        if "comparison_label_version" not in frame.columns:
            # Saved results created before the chronological convention used
            # VF1 for Maity et al. (2021) and VF2 for John et al. (2020).
            frame["framework"] = frame["framework"].replace(
                {"VF1": "__LEGACY_VF1__", "VF2": "VF1"}
            ).replace({"__LEGACY_VF1__": "VF2"})
            frame["comparison_label_version"] = "chronological_2020_2021_2024"
        frame["artifact_path"] = str(path.parent.resolve())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_tuning_outputs(tuning_root: str | Path | None) -> pd.DataFrame:
    """Load model-level Bayesian trial checkpoints for convergence inspection."""

    if tuning_root is None:
        return pd.DataFrame()
    rows = []
    for path in sorted(Path(tuning_root).glob("**/*.json")):
        if path.name == "tuning.json":
            continue
        payload = json.loads(path.read_text())
        if "tuning" not in payload or "model_name" not in payload:
            continue
        model = payload["tuning"]["models"][payload["model_name"]]
        best_so_far = float("-inf")
        for trial in model["trials"]:
            score = float(trial["mean_validation_score"])
            best_so_far = max(best_so_far, score)
            rows.append(
                {
                    "dataset_id": payload["dataset_id"],
                    "policy": payload["protected_policy"],
                    "model": payload["model_name"],
                    "budget": payload["budget"],
                    "trial": trial["trial_number"],
                    "mean_validation_score": score,
                    "std_validation_score": trial["std_validation_score"],
                    "best_score_so_far": best_so_far,
                    "hyperparameters": json.dumps(
                        trial["hyperparameters"], sort_keys=True
                    ),
                    "checkpoint_path": str(path.resolve()),
                }
            )
    return pd.DataFrame(rows)
