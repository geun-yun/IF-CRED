"""Resumable conversion of legacy full-model results into compact E3 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from collections.abc import Callable
from typing import Any

from ifcred.comparisons.artifacts import (
    LegacyArtifactCloudUnavailable,
    comparison_rows_for_saved_condition,
    write_comparison_rows,
)
from ifcred.experiments.persistence import (
    compact_context_payload,
    compact_result_payload,
    configuration_id_for_setup_manifest,
)


def _segment(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError("result path segment cannot be empty")
    return result


def _ratio(value: float) -> str:
    return f"rho_{value:.8f}".replace(".", "p")


def _destination_path(root: Path, manifest: dict[str, Any]) -> Path:
    context, result = manifest["context"], manifest["result"]
    return root / Path(
        _segment(context["dataset"]["dataset_id"]),
        _segment(context["setup"]["protected_policy"]),
        f"repetition_{int(context['repetition']['repetition']):03d}",
        f"config_{configuration_id_for_setup_manifest(context['setup'])}",
        _segment(result["condition"]),
        _segment(result.get("condition_variant", "default")),
        _ratio(float(result["requested_injection_ratio"])),
    )


def _verify_existing(path: Path, *, expected_configuration_id: str) -> None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"incomplete compact migration output: {path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("storage_mode") != "compact_inline_e3"
        or manifest.get("configuration_id") != expected_configuration_id
    ):
        raise ValueError(f"incompatible compact migration output: {path}")
    comparison = manifest.get("comparisons", {})
    if comparison.get("applicable"):
        comparison_path = path / comparison["file"]
        if not comparison_path.exists():
            raise RuntimeError(f"missing migrated comparison table: {comparison_path}")
        digest = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
        if digest != comparison["sha256"]:
            raise ValueError(f"migrated comparison checksum mismatch: {comparison_path}")


def migrate_legacy_results(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    n_jobs: int = 1,
    progress: Callable[[int, int, str, Path], None] | None = None,
) -> dict[str, int]:
    """Compute missing E3 outputs and compact legacy checkpoints without mutation."""

    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if source == destination:
        raise ValueError("legacy source and compact destination must be different")
    if not source.exists():
        raise FileNotFoundError(f"legacy results directory does not exist: {source}")
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    for existing in destination.glob("**/manifest.json"):
        payload = json.loads(existing.read_text())
        if "result" in payload and payload.get("storage_mode") != "compact_inline_e3":
            raise ValueError(f"compact destination contains a legacy artifact: {existing}")

    migrated = skipped = unavailable = comparisons_written = 0
    manifests = sorted(source.glob("**/manifest.json"))
    for position, source_manifest_path in enumerate(manifests, start=1):
        try:
            legacy = json.loads(source_manifest_path.read_text())
        except (TimeoutError, LegacyArtifactCloudUnavailable):
            unavailable += 1
            if progress is not None:
                progress(position, len(manifests), "cloud-unavailable", source_manifest_path)
            continue
        if "result" not in legacy or "context" not in legacy:
            continue
        if legacy.get("storage_mode") == "compact_inline_e3":
            raise ValueError(f"source contains a compact artifact: {source_manifest_path}")
        final = _destination_path(destination, legacy)
        configuration_id = final.parts[-4].removeprefix("config_")
        if final.exists():
            _verify_existing(final, expected_configuration_id=configuration_id)
            skipped += 1
            if progress is not None:
                progress(position, len(manifests), "skipped", source_manifest_path)
            continue

        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".ifcred-migrate-", dir=final.parent)
        )
        try:
            rows = comparison_rows_for_saved_condition(
                source_manifest_path, legacy, n_jobs=n_jobs
            )
            wrote_comparisons = bool(rows)
            comparison_metadata: dict[str, Any] = {"applicable": bool(rows)}
            if rows:
                comparison_path = temporary / "comparisons.csv"
                write_comparison_rows(comparison_path, rows)
                comparison_metadata.update(
                    {
                        "file": comparison_path.name,
                        "sha256": hashlib.sha256(
                            comparison_path.read_bytes()
                        ).hexdigest(),
                        "rows": len(rows),
                    }
                )
            compact = {
                "schema_version": 2,
                "storage_mode": "compact_inline_e3",
                "configuration_id": configuration_id,
                "context": compact_context_payload(legacy["context"]),
                "result": compact_result_payload(legacy["result"]),
                "comparisons": comparison_metadata,
                "migration": {
                    "source_relative_path": str(
                        source_manifest_path.relative_to(source)
                    ),
                    "source_manifest_sha256": hashlib.sha256(
                        source_manifest_path.read_bytes()
                    ).hexdigest(),
                    "source_schema_version": legacy.get("schema_version", 1),
                    "source_arrays_sha256": legacy.get("arrays_sha256"),
                    "source_estimators_sha256": legacy.get("estimators_sha256"),
                },
            }
            (temporary / "manifest.json").write_text(
                json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            temporary.rename(final)
            migrated += 1
            comparisons_written += int(wrote_comparisons)
            if progress is not None:
                progress(position, len(manifests), "migrated", source_manifest_path)
        except (TimeoutError, LegacyArtifactCloudUnavailable):
            if temporary.exists():
                shutil.rmtree(temporary)
            unavailable += 1
            if progress is not None:
                progress(position, len(manifests), "cloud-unavailable", source_manifest_path)
            continue
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    return {
        "conditions_migrated": migrated,
        "conditions_skipped": skipped,
        "conditions_cloud_unavailable": unavailable,
        "comparison_artifacts_written": comparisons_written,
    }
