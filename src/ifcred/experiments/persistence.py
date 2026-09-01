"""Immutable, atomic persistence of condition-level experiment evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

import numpy as np
from ifcred.comparisons.artifacts import (
    comparison_rows_for_result,
    write_comparison_rows,
)
from ifcred.experiments.orchestrator import ConditionResult, ExperimentSetup, PreparedRepetition


def _safe_segment(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError("result path segment cannot be empty")
    return result


def _ratio_segment(value: float) -> str:
    return f"rho_{value:.8f}".replace(".", "p")


def configuration_id_for_setup_manifest(setup_manifest: Mapping[str, Any]) -> str:
    """Hash scientific settings while excluding hardware/runtime choices."""

    manifest = deepcopy(_json_safe(setup_manifest))
    # Worker counts and equivalent search/batching implementations affect
    # runtime, not the scientific condition. Excluding them lets CPU shards
    # with different hardware merge into the same deterministic result tree.
    manifest.pop("n_jobs", None)
    manifest.pop("parallel_backend", None)
    graph = manifest["graph_fitting_spec"]
    for name in ("n_jobs", "pair_batch_size", "search_algorithm"):
        graph.pop(name, None)
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def configuration_id_for_setup(setup: ExperimentSetup) -> str:
    """Stable identifier for scientific settings, excluding condition/severity."""

    return configuration_id_for_setup_manifest(setup.to_manifest())


def configuration_id(context: PreparedRepetition) -> str:
    return configuration_id_for_setup(context.setup)


def condition_result_path(
    output_root: str | Path,
    context: PreparedRepetition,
    result: ConditionResult,
) -> Path:
    """Resolve the immutable location used for one result."""

    return Path(output_root).resolve() / Path(
        _safe_segment(context.bundle.spec.dataset_id),
        _safe_segment(context.setup.protected_policy.value),
        f"repetition_{context.repetition.repetition:03d}",
        f"config_{configuration_id(context)}",
        _safe_segment(result.condition),
        _safe_segment(result.condition_variant),
        _ratio_segment(result.requested_injection_ratio),
    )


def expected_condition_path(
    output_root: str | Path,
    context: PreparedRepetition,
    *,
    condition: str,
    variant: str,
    ratio: float,
) -> Path:
    """Resolve a path before expensive model fitting, enabling true resume."""

    return Path(output_root).resolve() / Path(
        _safe_segment(context.bundle.spec.dataset_id),
        _safe_segment(context.setup.protected_policy.value),
        f"repetition_{context.repetition.repetition:03d}",
        f"config_{configuration_id(context)}",
        _safe_segment(condition),
        _safe_segment(variant),
        _ratio_segment(ratio),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    return value


def _array_digest(values: Any) -> str:
    array = np.ascontiguousarray(values)
    descriptor = json.dumps(
        {"shape": list(array.shape), "dtype": str(array.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + array.tobytes()).hexdigest()


def _json_digest(values: Any) -> str:
    encoded = json.dumps(
        _json_safe(values), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compact_context_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove repeated split vectors while retaining verifiable provenance."""

    payload = deepcopy(dict(payload))
    split = payload["split"]
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    payload["split"] = {
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "train_indices_sha256": _array_digest(train_indices),
        "test_indices_sha256": _array_digest(test_indices),
        "random_state": split["random_state"],
        "test_size": split["test_size"],
        "stratification_strategy": split["stratification_strategy"],
    }
    return payload


def compact_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Replace condition-scale trace lists with counts and stable hashes."""

    payload = deepcopy(dict(payload))
    pairs = payload.pop("injected_pairs")
    payload["injection_trace"] = {
        "pair_count": len(pairs),
        "pairs_sha256": _json_digest(pairs),
    }
    metadata = dict(payload["condition_metadata"])
    selected = metadata.pop("selected_candidate_indices", None)
    if selected is not None:
        metadata["selected_candidate_count"] = len(selected)
        metadata["selected_candidate_indices_sha256"] = _array_digest(
            np.asarray(selected, dtype=np.int64)
        )
    payload["condition_metadata"] = metadata
    return payload


def save_condition_result(
    output_root: str | Path,
    context: PreparedRepetition,
    result: ConditionResult,
) -> Path:
    """Compute E3 in memory, then atomically save compact analysis outputs."""

    final = condition_result_path(output_root, context, result)
    if final.exists():
        raise FileExistsError(f"immutable result already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ifcred-tmp-", dir=final.parent))
    try:
        comparison_rows = comparison_rows_for_result(context, result)
        comparison_metadata: dict[str, Any] = {"applicable": bool(comparison_rows)}
        if comparison_rows:
            comparisons_path = temporary / "comparisons.csv"
            write_comparison_rows(comparisons_path, comparison_rows)
            comparison_metadata.update(
                {
                    "file": comparisons_path.name,
                    "sha256": hashlib.sha256(comparisons_path.read_bytes()).hexdigest(),
                    "rows": len(comparison_rows),
                }
            )
        manifest = {
            "schema_version": 2,
            "storage_mode": "compact_inline_e3",
            "configuration_id": configuration_id(context),
            "context": compact_context_payload(context.to_manifest()),
            "result": compact_result_payload(result.to_manifest()),
            "comparisons": comparison_metadata,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(
                _json_safe(manifest),
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.rename(final)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final
