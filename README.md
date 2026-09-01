# IF-CRED experimental implementation

This repository contains the reproducible implementation of the IF-CRED E1–E3
study described in `protocols/experiment_catalogue.md`. It supports one-time
development-set tuning, repeated clean and controlled-injection experiments,
inline prior-framework comparisons, resumable execution, and generation of the
tables and figures used for analysis.

Generated datasets, tuning runs, experiment outputs, figures, and manuscript
files are intentionally kept outside the versioned code surface.

## Installation

From this directory:

```bash
python3 -m pip install -e '.[test]'
```

## Reproducing the study

The repeated study should use hyperparameters selected once on a dedicated 20%
development population. Those development rows are excluded from every later
E1–E3 split.

First tune all five model families for each of the six dataset-policy branches.
Downloaded UCI data are cached locally under `data/original/` and are not
committed:

```bash
ifcred tune \
  --budget pilot \
  --development-fraction 0.20 \
  --output configs/tuned_models \
  --n-jobs 5
```

This creates six final tuning bundles and 30 model-level specifications under
`configs/tuned_models/`. Each completed model search is checkpointed
separately, so rerunning the command continues with missing model families.

`--n-jobs` is the one concurrency setting for tuning, calibration, model
fitting, and neighbour construction. IF-CRED uses joblib's threading backend
for this work, avoiding macOS/Python 3.13 `loky` resource-tracker failures and
nested process pools. You do not need to edit individual `n_jobs` values.

Inspect the planned workload without loading any data or running anything:

```bash
ifcred plan \
  --profile frozen \
  --model-config configs/tuned_models
```

For a quick installation check, the smoke profile exercises every code path
with D8, two repetitions, fixed engineering hyperparameters, and a small
condition grid:

```bash
ifcred all --profile smoke --results results/smoke --output artifacts/smoke
```

Run E1 only (clean CDFM and audit-design sensitivities):

```bash
ifcred run --profile frozen --model-config configs/tuned_models --experiments E1 --results results/frozen
```

Run E2 only (all controlled injections, severities, and seeds):

```bash
ifcred run --profile frozen --model-config configs/tuned_models --experiments E2 --results results/frozen
```

E3 is calculated inline while each condition's fitted models are already in
memory. This avoids saving and reloading large estimator/matrix packages. The
standalone command remains only for results produced by the former format:

```bash
ifcred compare --results results/legacy
```

Run the whole pipeline with compact outputs (recommended and fastest):

```bash
ifcred all --profile frozen --model-config configs/tuned_models --results results/frozen --output artifacts/frozen
```

Regenerate all tables and figures from existing outputs only—no dataset load,
model fit, graph construction, or experiment rerun occurs:

```bash
ifcred report --model-config configs/tuned_models --results results/frozen --output artifacts/frozen
```

Inspect resumable progress:

```bash
ifcred status --model-config configs/tuned_models --results results/frozen
```

Existing condition directories are immutable and skipped on resumed E1/E2
runs. Each applicable condition atomically includes its E3 `comparisons.csv`.
Each figure folder contains both PNG/PDF outputs and the exact `source_data.csv`
used to draw it.
E2 model-level fairness figures plot every declared model at `ρ=0` (the matched
clean baseline) and `ρ=0.05, 0.10, 0.15, 0.20, 0.25, 0.30`, with repeated-seed
95% uncertainty bands. They are generated separately by injection mechanism,
mechanism severity, and protected-feature policy.

Legacy full-model checkpoints can be reused once without modifying their
source directory. Migration computes their outstanding E3 rows and writes the
compact equivalents to a fresh result root:

```bash
ifcred migrate \
  --source-results results/frozen_v2 \
  --results results/frozen_v3 \
  --n-jobs 5
```

The command is resumable at condition boundaries. A subsequent compact run
recognizes and skips every migrated condition.

## Useful bounded runs

You can restrict datasets or repetitions while profiling compute:

```bash
ifcred tune --datasets D8 --budget pilot --output configs/pilot_d8
ifcred run --profile frozen --model-config configs/pilot_d8 --experiments E1 --datasets D8 --repetitions 2 --results results/pilot_d8
```

## CPU shards and Colab

The sklearn model family is CPU-based; selecting a Colab GPU or TPU does not
accelerate it. Long runs can instead be divided into non-overlapping repetition
ranges. For example, six five-seed shards cover the frozen seeds `0..29`:

```bash
ifcred run \
  --profile frozen \
  --model-config configs/tuned_models \
  --experiments E1,E2 \
  --repetition-start 0 \
  --repetitions 5 \
  --results results/frozen \
  --n-jobs 2
```

Subsequent shards start at `5`, `10`, `15`, `20`, and `25`. They may write to
the same shared result root because repetition paths do not overlap. E3 is
computed inline, so there is no separate comparison command for compact runs.
After all shards finish, run `ifcred report` once. Install
`requirements-experiment.txt` on every machine so numerical-library versions
remain matched; worker counts may differ without changing the scientific
configuration identifier.

The three prior-framework adapters follow chronological labels: VF1 is John et
al. (2020), VF2 is Maity et al. (2021), and VF3 is Kitamura et al. (2024,
IFT-V). They are documented tabular approximations ported from the inspected
sibling implementation.
They retain their native decisions/rates instead of forcing them onto the
IF-CRED scale. Their settings should be reviewed and frozen before a
confirmatory journal submission.

The legacy `--profile exploratory` remains available for methodological
sensitivity work, but it repeats Bayesian tuning inside all 180 outer branches
and is not recommended for the main run.

## Repository layout

- `src/ifcred/`: implementation and command-line interface
- `tests/`: automated regression and workflow tests
- `protocols/`: experiment design and reproducibility decisions
- `requirements-experiment.txt`: exact package versions used for the completed
  runs
- `scripts/`: small utilities used to produce study schematics

Runtime products belong in `results/`, `artifacts/`, and
`configs/tuned_models/`. These paths are ignored by Git so the repository stays
focused on code and protocol documentation. To reproduce the exact software
environment used for the completed run, install the frozen requirements before
installing the local package:

```bash
python3 -m pip install -r requirements-experiment.txt
python3 -m pip install -e . --no-deps
```
