"""Command-line acquisition for content-addressed UCI snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ifcred.data.acquisition import fetch_uci_dataset
from ifcred.data.registry import DATASET_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_ids",
        nargs="*",
        choices=tuple(DATASET_REGISTRY),
        default=list(DATASET_REGISTRY),
        help="datasets to acquire; omit to acquire D6, D7, and D8",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="directory that will contain immutable content-addressed snapshots",
    )
    args = parser.parse_args()
    summaries = []
    for dataset_id in args.dataset_ids:
        bundle = fetch_uci_dataset(dataset_id, cache_root=args.cache_root)
        summaries.append(dict(bundle.manifest))
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
