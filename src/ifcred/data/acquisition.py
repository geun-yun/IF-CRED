"""Authoritative UCI acquisition and immutable original-data snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ifcred.data.registry import DatasetBundle, DatasetSpec, get_dataset_spec

UCIFetcher = Callable[..., Any]


def _validate_original(
    remote: Any, spec: DatasetSpec
) -> tuple[pd.DataFrame, pd.Series]:
    try:
        features = pd.DataFrame(remote.data.features).copy()
        targets = pd.DataFrame(remote.data.targets).copy()
    except AttributeError as exc:
        raise ValueError("UCI response does not expose data.features and data.targets") from exc
    if tuple(features.columns) != spec.expected_features:
        missing = sorted(set(spec.expected_features) - set(features.columns))
        extra = sorted(set(features.columns) - set(spec.expected_features))
        raise ValueError(f"UCI schema mismatch; missing={missing}, extra={extra}")
    if spec.outcome_source not in targets:
        raise ValueError(f"UCI target is missing {spec.outcome_source!r}")
    target = targets[spec.outcome_source].copy()
    if len(features) != len(target) or not features.index.equals(target.index):
        raise ValueError("UCI features and target are not row-aligned")
    return features, target


def _snapshot_digest(features: pd.DataFrame, target: pd.Series) -> str:
    feature_bytes = features.to_csv(index=False, lineterminator="\n").encode("utf-8")
    target_bytes = target.to_frame().to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(feature_bytes + b"\n--TARGET--\n" + target_bytes).hexdigest()


def _manifest(spec: DatasetSpec, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    literal_questions = {}
    for name in features:
        series = features[name]
        literal_questions[name] = int(
            series.map(lambda value: isinstance(value, str) and value.strip() == "?").sum()
        )
    return {
        "dataset_id": spec.dataset_id,
        "name": spec.name,
        "uci_id": spec.uci_id,
        "source_url": spec.uci_url,
        "doi": spec.doi,
        "license": "CC BY 4.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(features),
        "n_features": features.shape[1],
        "source_sha256": _snapshot_digest(features, target),
        "source_nulls_by_feature": {
            name: int(value) for name, value in features.isna().sum().items()
        },
        "source_literal_question_marks_by_feature": literal_questions,
        "source_target_value_counts": {
            str(name): int(value) for name, value in target.value_counts(dropna=False).items()
        },
    }


def _bundle(
    spec: DatasetSpec,
    features: pd.DataFrame,
    target: pd.Series,
    manifest: dict[str, Any],
) -> DatasetBundle:
    protected = features.loc[:, spec.protected_attributes].copy()
    return DatasetBundle(
        spec=spec,
        features=features,
        target=target,
        protected=protected,
        manifest=manifest,
    )


def _write_snapshot(
    cache_root: Path, features: pd.DataFrame, target: pd.Series, manifest: dict[str, Any]
) -> None:
    directory = cache_root / manifest["dataset_id"] / manifest["source_sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": directory / "features.csv",
        "target": directory / "target.csv",
        "manifest": directory / "manifest.json",
    }
    if all(path.exists() for path in paths.values()):
        return
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"incomplete immutable snapshot already exists at {directory}")
    features.to_csv(paths["features"], index=False)
    target.to_frame().to_csv(paths["target"], index=False)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def fetch_uci_dataset(
    dataset_id: str,
    *,
    cache_root: str | Path | None = None,
    fetcher: UCIFetcher | None = None,
) -> DatasetBundle:
    """Fetch, validate the schema, and optionally snapshot original UCI values."""

    spec = get_dataset_spec(dataset_id)
    if fetcher is None:
        from ucimlrepo import fetch_ucirepo

        fetcher = fetch_ucirepo
    remote = fetcher(id=spec.uci_id)
    features, target = _validate_original(remote, spec)
    manifest = _manifest(spec, features, target)
    if cache_root is not None:
        _write_snapshot(Path(cache_root), features, target, manifest)
    return _bundle(spec, features, target, manifest)


def load_cached_dataset(dataset_id: str, cache_root: str | Path) -> DatasetBundle:
    """Load the most recently retrieved valid immutable snapshot."""

    spec = get_dataset_spec(dataset_id)
    manifests = list((Path(cache_root) / dataset_id).glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no cached snapshot for {dataset_id}")
    candidates = []
    for path in manifests:
        data = json.loads(path.read_text())
        candidates.append((data["retrieved_at_utc"], path, data))
    _, manifest_path, manifest = max(candidates, key=lambda item: item[0])
    directory = manifest_path.parent
    features = pd.read_csv(directory / "features.csv")
    if tuple(features.columns) != spec.expected_features:
        raise ValueError("cached feature schema does not match the registry")
    target_frame = pd.read_csv(directory / "target.csv")
    if spec.outcome_source not in target_frame:
        raise ValueError("cached target file is invalid")
    target = target_frame[spec.outcome_source]
    digest = _snapshot_digest(features, target)
    if digest != manifest["source_sha256"] or digest != directory.name:
        raise ValueError("cached snapshot checksum does not match its manifest/path")
    return _bundle(spec, features, target, manifest)
