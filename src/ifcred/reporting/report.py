from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ifcred.reporting.loader import load_comparison_outputs, load_condition_outputs, load_tuning_outputs
from ifcred.reporting.plots import (
    make_dataset_response_figures,
    make_figures,
    make_model_fairness_ratio_figures,
    make_model_fairness_vs_v_figure,
    make_e1_predictive_performance_figure,
    make_e1_predictive_performance_by_dataset_figure,
    make_e1_sensitivity_figure,
    make_tuning_figures,
)
from ifcred.reporting.statistics import component_response_tests, e1_sensitivity_statistics, paired_clean_deltas, repetition_summary, separability_statistics


REPORT_DATASET_LABELS = {"D6": "D1", "D7": "D2", "D8": "D3"}
REPORT_CACHE_VERSION = 3


_TABLE_NAMES = (
    "condition_results",
    "model_results",
    "prediction_controls",
    "repetition_summary",
    "paired_clean_deltas",
    "component_response_tests",
    "separability_statistics",
    "e1_sensitivity_statistics",
    "prior_framework_results",
    "tuning_trials",
)


def _with_report_dataset_labels(frame):
    """Return a copy with internal dataset keys replaced by publication labels."""

    labelled = frame.copy()
    if "dataset_id" in labelled.columns:
        labelled["dataset_id"] = labelled["dataset_id"].replace(
            REPORT_DATASET_LABELS
        )
    if "scope" in labelled.columns:
        labelled["scope"] = labelled["scope"].replace(REPORT_DATASET_LABELS)
    return labelled


def _source_signature(root: str | Path | None, filenames: tuple[str, ...]) -> dict:
    """Return a cheap change signature without parsing experiment artifacts."""

    if root is None:
        return {"root": None, "count": 0, "size": 0, "mtime_ns_sum": 0}
    path = Path(root)
    files = [item for name in filenames for item in path.glob(f"**/{name}")]
    stats = [item.stat() for item in files]
    return {
        "root": str(path.resolve()),
        "count": len(stats),
        "size": sum(item.st_size for item in stats),
        "mtime_ns_sum": sum(item.st_mtime_ns for item in stats),
    }


def _cache_signature(
    results_root: str | Path, tuning_root: str | Path | None
) -> dict:
    return {
        "version": REPORT_CACHE_VERSION,
        "results": _source_signature(
            results_root, ("manifest.json", "comparisons.csv")
        ),
        "tuning": _source_signature(tuning_root, ("*.json",)),
    }


def _read_cached_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_report_cache(tables: Path, signature: dict):
    metadata_path = tables / "report_cache.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("signature") != signature:
        return None
    paths = {name: tables / f"{name}.csv" for name in _TABLE_NAMES}
    if not all(path.exists() for path in paths.values()):
        return None
    return {name: _read_cached_table(path) for name, path in paths.items()}


def _write_report_cache(tables: Path, signature: dict) -> None:
    (tables / "report_cache.json").write_text(
        json.dumps({"signature": signature}, indent=2) + "\n"
    )


def generate_report(
    results_root: str | Path,
    output_root: str | Path,
    *,
    tuning_root: str | Path | None = None,
) -> dict:
    """Rebuild tables and figures without loading a dataset or fitting a model."""

    output = Path(output_root); tables = output / "tables"; figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    signature = _cache_signature(results_root, tuning_root)
    cached = _load_report_cache(tables, signature)
    cache_used = cached is not None
    if cached is None:
        summary, models, controls = load_condition_outputs(results_root)
        comparisons = load_comparison_outputs(results_root)
        tuning = load_tuning_outputs(tuning_root)
        repeated = repetition_summary(summary)
        deltas = paired_clean_deltas(summary)
        tests = component_response_tests(deltas)
        separability = separability_statistics(models)
        sensitivity = e1_sensitivity_statistics(summary)
        (
            summary,
            models,
            controls,
            repeated,
            deltas,
            tests,
            separability,
            sensitivity,
            comparisons,
            tuning,
        ) = (
            _with_report_dataset_labels(frame)
            for frame in (
                summary,
                models,
                controls,
                repeated,
                deltas,
                tests,
                separability,
                sensitivity,
                comparisons,
                tuning,
            )
        )
        frames = {
            "condition_results": summary,
            "model_results": models,
            "prediction_controls": controls,
            "repetition_summary": repeated,
            "paired_clean_deltas": deltas,
            "component_response_tests": tests,
            "separability_statistics": separability,
            "e1_sensitivity_statistics": sensitivity,
            "prior_framework_results": comparisons,
            "tuning_trials": tuning,
        }
        for name, frame in frames.items():
            frame.to_csv(tables / f"{name}.csv", index=False)
        _write_report_cache(tables, signature)
    else:
        summary = cached["condition_results"]
        models = cached["model_results"]
        controls = cached["prediction_controls"]
        repeated = cached["repetition_summary"]
        deltas = cached["paired_clean_deltas"]
        tests = cached["component_response_tests"]
        separability = cached["separability_statistics"]
        sensitivity = cached["e1_sensitivity_statistics"]
        comparisons = cached["prior_framework_results"]
        tuning = cached["tuning_trials"]
    generated = (
        make_tuning_figures(tuning, figures)
        + make_figures(summary, comparisons, figures)
        + make_e1_sensitivity_figure(sensitivity, figures)
        + make_dataset_response_figures(summary, comparisons, figures)
        + make_model_fairness_ratio_figures(models, figures)
        + make_model_fairness_vs_v_figure(
            models, figures, separability=separability
        )
        + make_e1_predictive_performance_figure(models, figures)
        + make_e1_predictive_performance_by_dataset_figure(models, figures)
    )
    manifest = {"source_results": str(Path(results_root).resolve()), "source_tuning": None if tuning_root is None else str(Path(tuning_root).resolve()), "tables": [p.name for p in sorted(tables.glob("*.csv"))], "figures": generated, "experiment_execution": False, "report_cache_used": cache_used}
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
