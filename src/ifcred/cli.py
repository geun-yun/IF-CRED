"""Command-line entry points for separable, resumable experiment stages."""

from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from ifcred.comparisons.artifacts import run_comparisons_from_saved_results
from ifcred.data import ProtectedAttributePolicy
from ifcred.experiments.frozen_tuning import (
    tune_frozen_model_bundles,
)
from ifcred.experiments.migration import migrate_legacy_results
from ifcred.experiments.study import run_study
from ifcred.experiments.study_config import exploratory_config, frozen_config, smoke_config
from ifcred.reporting import generate_report


def _config(args):
    if args.profile == "smoke":
        config = smoke_config(root_seed=args.root_seed, n_jobs=args.n_jobs)
    elif args.profile == "exploratory":
        config = exploratory_config(root_seed=args.root_seed, n_jobs=args.n_jobs)
    else:
        if not args.model_config:
            raise ValueError("--model-config is required with --profile frozen")
        config = frozen_config(
            args.model_config,
            root_seed=args.root_seed,
            n_jobs=args.n_jobs,
            development_fraction=args.development_fraction,
        )
    if args.datasets:
        config = replace(config, dataset_ids=tuple(item.strip().upper() for item in args.datasets.split(",") if item.strip()))
    if args.repetitions is not None:
        config = replace(config, n_repetitions=args.repetitions)
    if args.repetition_start < 0:
        raise ValueError("--repetition-start must be non-negative")
    if args.repetition_start and args.repetitions is None:
        raise ValueError("--repetition-start requires --repetitions")
    config = replace(config, repetition_start=args.repetition_start)
    return config


def _add_profile(parser):
    parser.add_argument("--profile", choices=("smoke", "exploratory", "frozen"), default="frozen")
    parser.add_argument("--datasets", help="comma-separated subset, e.g. D6,D8")
    parser.add_argument("--repetitions", type=int, help="override outer repetitions (minimum 2)")
    parser.add_argument(
        "--repetition-start",
        type=int,
        default=0,
        help="first repetition index for a non-overlapping compute shard",
    )
    parser.add_argument("--root-seed", type=int, default=20260825)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--model-config", help="directory written by `ifcred tune`; required for frozen")
    parser.add_argument("--development-fraction", type=float, default=0.20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ifcred")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run E1 and/or E2 and save immutable outputs")
    _add_profile(run); run.add_argument("--experiments", default="E1,E2", help="E1, E2, or E1,E2")
    run.add_argument("--conditions", help="optional E2 condition filter, e.g. S5")
    run.add_argument("--results", default="results"); run.add_argument("--cache", default="data/original")
    compare = sub.add_parser("compare", help="run E3 for legacy deferred-E3 artifacts")
    compare.add_argument("--results", default="results"); compare.add_argument("--overwrite", action="store_true")
    report = sub.add_parser("report", help="regenerate tables/plots from saved outputs only")
    report.add_argument("--results", default="results"); report.add_argument("--output", default="artifacts"); report.add_argument("--model-config")
    all_parser = sub.add_parser("all", help="run compact E1+E2 with inline E3, then report")
    _add_profile(all_parser); all_parser.add_argument("--results", default="results"); all_parser.add_argument("--cache", default="data/original"); all_parser.add_argument("--output", default="artifacts")
    status = sub.add_parser(
        "status", help="show saved-artifact counts, throughput, and optional targets"
    )
    _add_profile(status)
    status.set_defaults(model_config="configs/tuned_models")
    status.add_argument("--results", default="results")
    migrate = sub.add_parser(
        "migrate",
        help="convert legacy full-model results to compact inline-E3 checkpoints",
    )
    migrate.add_argument("--source-results", required=True)
    migrate.add_argument("--results", required=True)
    migrate.add_argument("--n-jobs", type=int, default=1)
    plan = sub.add_parser("plan", help="show workload counts without loading data or running experiments")
    _add_profile(plan)
    tune = sub.add_parser("tune", help="tune five models once per dataset-policy on disjoint development rows")
    tune.add_argument("--datasets", default="D6,D7,D8")
    tune.add_argument("--cache", default="data/original")
    tune.add_argument("--output", default="configs/tuned_models")
    tune.add_argument("--budget", choices=("pilot", "full"), default="pilot")
    tune.add_argument("--development-fraction", type=float, default=0.20)
    tune.add_argument("--folds", type=int, default=5)
    tune.add_argument("--root-seed", type=int, default=20260825)
    tune.add_argument("--n-jobs", type=int, default=1)
    return parser


def _workload(config) -> dict:
    branches = len(config.dataset_ids) * len(config.policies) * len(config.repetition_numbers)
    e1_per_branch = len(config.audit_variants) + 1  # primary + native + sensitivities
    definition_arms = (
        2 * len(config.near_duplicate_radii)
        + len(config.isolation_similarities)
        + len(config.dominant_background_similarities)
        + len(config.disagreement_reliabilities)
    )
    e2_injected_per_branch = len(config.injection_ratios) * (
        definition_arms + len(config.model_affected_fractions)
    )
    condition_artifacts = branches * (e1_per_branch + e2_injected_per_branch)
    comparison_artifacts = branches * (1 + e2_injected_per_branch)
    return {
        "profile": config.profile,
        "datasets": list(config.dataset_ids),
        "policies": [policy.value for policy in config.policies],
        "outer_repetitions": config.n_repetitions,
        "repetition_numbers": list(config.repetition_numbers),
        "dataset_policy_repetition_branches": branches,
        "E1_condition_artifacts": branches * e1_per_branch,
        "E2_injected_condition_artifacts": branches * e2_injected_per_branch,
        "total_unique_E1_E2_condition_artifacts": condition_artifacts,
        "E3_comparison_artifacts": comparison_artifacts,
        "E3_framework_model_evaluations": branches
        * (1 + e2_injected_per_branch)
        * 5
        * 3,
        "bayesian_tuning_during_each_repetition": config.profile == "exploratory",
        "development_fraction": config.development_fraction,
        "executes_experiment": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        experiments = tuple(item.strip().upper() for item in args.experiments.split(",") if item.strip())
        condition_aliases = {
            "S1": "benign_near_duplicate",
            "S2": "contradictory_near_duplicate",
            "S3": "isolated_instance",
            "S4": "dominant_neighbour_pair",
            "S5": "metric_disagreement_instance",
            "S6": "model_family_disagreement",
        }
        conditions = None
        if args.conditions:
            conditions = tuple(
                condition_aliases.get(item.strip().upper(), item.strip())
                for item in args.conditions.split(",")
                if item.strip()
            )
        outcome = run_study(_config(args), results_root=args.results, cache_root=args.cache, experiments=experiments, conditions=conditions)
    elif args.command == "compare":
        outcome = {"comparison_artifacts_written": run_comparisons_from_saved_results(args.results, overwrite=args.overwrite)}
    elif args.command == "report":
        outcome = generate_report(args.results, args.output, tuning_root=args.model_config)
    elif args.command == "all":
        study = run_study(_config(args), results_root=args.results, cache_root=args.cache, experiments=("E1","E2"))
        comparisons = run_comparisons_from_saved_results(args.results)
        outcome = {"study": study, "comparison_artifacts_written": comparisons, "report": generate_report(args.results, args.output, tuning_root=args.model_config)}
    elif args.command == "status":
        root = Path(args.results)
        tuning_root = Path(args.model_config)
        condition_paths = list(root.glob("**/manifest.json"))
        comparison_paths = list(root.glob("**/comparisons.csv"))
        recent_cutoff = time.time() - 60 * 60
        recent_conditions = sum(
            path.stat().st_mtime >= recent_cutoff for path in condition_paths
        )
        recent_comparisons = sum(
            path.stat().st_mtime >= recent_cutoff for path in comparison_paths
        )
        outcome = {
            "tuning_model_checkpoints": len(
                [
                    path
                    for path in tuning_root.glob("**/*.json")
                    if path.name != "tuning.json"
                ]
            ),
            "complete_tuning_bundles": len(
                list(tuning_root.glob("**/tuning.json"))
            ),
            "condition_artifacts": len(condition_paths),
            "comparison_artifacts": len(comparison_paths),
            "rate_window_minutes": 60,
            "condition_artifacts_per_hour_recent": recent_conditions,
            "comparison_artifacts_per_hour_recent": recent_comparisons,
        }
        if args.repetitions is not None:
            workload = _workload(_config(args))
            expected_conditions = workload[
                "total_unique_E1_E2_condition_artifacts"
            ]
            expected_comparisons = workload["E3_comparison_artifacts"]
            remaining_conditions = max(
                0, expected_conditions - len(condition_paths)
            )
            condition_percent = 100 * len(condition_paths) / expected_conditions
            comparison_percent = 100 * len(comparison_paths) / expected_comparisons
            outcome.update(
                {
                    "condition_artifacts": (
                        f"{len(condition_paths)}/{expected_conditions} "
                        f"({condition_percent:.2f}%)"
                    ),
                    "comparison_artifacts": (
                        f"{len(comparison_paths)}/{expected_comparisons} "
                        f"({comparison_percent:.2f}%)"
                    ),
                    "estimated_hours_remaining_at_recent_condition_rate": (
                        round(remaining_conditions / recent_conditions, 1)
                        if recent_conditions
                        else None
                    ),
                }
            )
    elif args.command == "migrate":
        def show_progress(done, total, state, path):
            print(
                f"[{done}/{total}] {state}: {path.parent}",
                file=sys.stderr,
                flush=True,
            )

        outcome = migrate_legacy_results(
            args.source_results,
            args.results,
            n_jobs=args.n_jobs,
            progress=show_progress,
        )
    elif args.command == "plan":
        outcome = _workload(_config(args))
    else:
        dataset_ids = tuple(
            item.strip().upper() for item in args.datasets.split(",") if item.strip()
        )
        outcome = tune_frozen_model_bundles(
            dataset_ids=dataset_ids,
            policies=(
                ProtectedAttributePolicy.INCLUDE_PRIMARY_PROTECTED,
                ProtectedAttributePolicy.EXCLUDE_PROTECTED,
            ),
            cache_root=args.cache,
            output_root=args.output,
            root_seed=args.root_seed,
            development_fraction=args.development_fraction,
            budget=args.budget,
            n_jobs=args.n_jobs,
            folds=args.folds,
        )
    print(json.dumps(outcome, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
