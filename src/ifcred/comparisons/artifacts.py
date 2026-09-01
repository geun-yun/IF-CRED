"""Run E3 on live conditions or legacy saved E1/E2 cases."""

from __future__ import annotations
import hashlib
import json
import stat
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ifcred.comparisons.base import ComparisonCase
from ifcred.comparisons.runner import run_prior_frameworks
from ifcred.parallel import sklearn_parallelism


class LegacyArtifactCloudUnavailable(OSError):
    """A legacy artifact is an unhydrated macOS/iCloud placeholder."""


def _raise_if_dataless(path: Path) -> None:
    """Fail immediately instead of waiting for an iCloud read to time out."""

    dataless_flag = getattr(stat, "SF_DATALESS", 0)
    if dataless_flag and path.stat().st_flags & dataless_flag:
        raise LegacyArtifactCloudUnavailable(
            f"legacy artifact is not stored locally: {path}"
        )


def _should_compare(condition: str, variant: str) -> bool:
    """E1 sensitivities reuse the primary clean case and are not new E3 cases."""

    return not (condition == "clean_baseline" and variant != "primary")


def comparison_rows_for_result(context, result) -> list[dict]:
    """Compute E3 while a condition's fitted models and matrices are in memory."""

    if not _should_compare(result.condition, result.condition_variant):
        return []
    if result.evaluation_features is None:
        raise ValueError("E3 requires evaluation features while the condition is live")
    model_names = set(result.model_run.models)
    if set(result.training_features_by_model) != model_names:
        raise ValueError("E3 requires one training matrix for every fitted model")
    if set(result.training_labels_by_model) != model_names:
        raise ValueError("E3 requires one training target for every fitted model")

    protected = context.prepared_dataset.experiment_primary_protected_indices(
        context.setup.protected_policy
    )
    protected_index = int(protected[0]) if len(protected) == 1 else None
    def evaluate_model(model_name, output) -> list[dict]:
        case = ComparisonCase(
            dataset_id=context.bundle.spec.dataset_id,
            condition=result.condition,
            variant=result.condition_variant,
            ratio=float(result.requested_injection_ratio),
            repetition=int(context.repetition.repetition),
            X_train=np.asarray(result.training_features_by_model[model_name]),
            y_train=np.asarray(result.training_labels_by_model[model_name]),
            X_test=np.asarray(result.evaluation_features),
            y_test=np.asarray(result.evaluation_labels),
            protected_index=protected_index,
            random_seed=int(context.repetition.seed_for("comparator")),
        )
        return [
            {"model": model_name, **row}
            for row in run_prior_frameworks(output.calibrated_estimator, case)
        ]

    items = list(result.model_run.models.items())
    with sklearn_parallelism(context.setup.n_jobs):
        by_model = Parallel(n_jobs=context.setup.n_jobs)(
            delayed(evaluate_model)(model_name, output)
            for model_name, output in items
        )
    return [row for model_rows in by_model for row in model_rows]


def write_comparison_rows(destination: Path, rows: list[dict]) -> None:
    """Write one already-computed E3 table."""

    pd.DataFrame(rows).to_csv(destination, index=False)


def comparison_rows_for_saved_condition(
    manifest_path: Path,
    manifest: dict | None = None,
    *,
    n_jobs: int = 1,
) -> list[dict]:
    """Compute E3 from one verified legacy full-model condition package."""

    manifest = (
        json.loads(manifest_path.read_text()) if manifest is None else manifest
    )
    context, result = manifest["context"], manifest["result"]
    if not _should_compare(
        result["condition"], result.get("condition_variant", "primary")
    ):
        return []
    protected = context.get("primary_protected_indices", [])
    protected_index = int(protected[0]) if len(protected) == 1 else None
    if protected_index is None:
        # All three comparison frameworks declare protected-excluded cases
        # inapplicable before touching a model or matrix. Build those native
        # outputs from manifest metadata without hydrating legacy cloud files.
        n_features = int(context["experiment_matrix_shape"][1])
        empty_X = np.empty((0, n_features), dtype=float)
        empty_y = np.empty(0, dtype=np.int8)
        case = ComparisonCase(
            dataset_id=context["dataset"]["dataset_id"],
            condition=result["condition"],
            variant=result.get("condition_variant", "default"),
            ratio=float(result["requested_injection_ratio"]),
            repetition=int(context["repetition"]["repetition"]),
            X_train=empty_X,
            y_train=empty_y,
            X_test=empty_X,
            y_test=empty_y,
            protected_index=None,
            random_seed=int(context["repetition"]["standard_seeds"]["comparator"]),
        )
        return [
            {"model": model_name, **row}
            for model_name in result["model_run"]["models"]
            for row in run_prior_frameworks(object(), case)
        ]
    arrays_path = manifest_path.parent / manifest["arrays_file"]
    estimators_path = manifest_path.parent / manifest["estimators_file"]
    _raise_if_dataless(arrays_path)
    _raise_if_dataless(estimators_path)
    _verify(arrays_path, manifest["arrays_sha256"])
    _verify(estimators_path, manifest["estimators_sha256"])
    estimators = joblib.load(estimators_path)
    with np.load(arrays_path) as stored:
        evaluation_features = np.asarray(stored["evaluation_features"])
        evaluation_labels = np.asarray(stored["evaluation_labels"])
        training = {
            model_name: (
                np.asarray(stored[f"training_features__{model_name.replace('/', '_')}"]),
                np.asarray(stored[f"training_labels__{model_name.replace('/', '_')}"]),
            )
            for model_name in estimators
        }
    def evaluate_model(model_name, model) -> list[dict]:
        X_train, y_train = training[model_name]
        case = ComparisonCase(
            dataset_id=context["dataset"]["dataset_id"],
            condition=result["condition"],
            variant=result.get("condition_variant", "default"),
            ratio=float(result["requested_injection_ratio"]),
            repetition=int(context["repetition"]["repetition"]),
            X_train=X_train,
            y_train=y_train,
            X_test=evaluation_features,
            y_test=evaluation_labels,
            protected_index=protected_index,
            random_seed=int(context["repetition"]["standard_seeds"]["comparator"]),
        )
        return [
            {"model": model_name, **row}
            for row in run_prior_frameworks(model, case)
        ]

    items = list(estimators.items())
    with sklearn_parallelism(n_jobs):
        by_model = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_model)(model_name, model)
            for model_name, model in items
        )
    return [row for model_rows in by_model for row in model_rows]


def _verify(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"artifact checksum mismatch: {path}")


def run_comparisons_from_saved_results(
    results_root: str | Path, *, overwrite: bool = False
) -> int:
    """Add comparisons.csv beside each immutable matched condition artifact."""

    completed = 0
    for manifest_path in sorted(Path(results_root).glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if "result" not in manifest:
            continue
        destination = manifest_path.parent / "comparisons.csv"
        context, result = manifest["context"], manifest["result"]
        if not _should_compare(
            result["condition"], result.get("condition_variant", "primary")
        ):
            continue
        if manifest.get("storage_mode") == "compact_inline_e3":
            if destination.exists() and not overwrite:
                continue
            raise FileNotFoundError(
                "compact conditions calculate E3 before discarding fitted models; "
                f"comparisons cannot be rebuilt or overwritten at {manifest_path.parent}"
            )
        if destination.exists() and not overwrite:
            continue
        rows = comparison_rows_for_saved_condition(
            manifest_path, manifest, n_jobs=1
        )
        temporary = destination.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False); temporary.replace(destination)
        completed += 1
    return completed
