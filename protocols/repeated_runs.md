# Repeated-seed experiment protocol

## Purpose

A single random state can produce an unusually easy split, favourable model
initialization, or atypical synthetic anchors. The experiment therefore uses a
declared repetition plan rather than selecting or emphasizing one seed.

The recommended exploratory plan contains 30 repetitions derived from one
recorded root seed. Thirty is provisional until the full-data compute pilot and
precision analysis; it is not a guarantee of evidential adequacy by itself.

## Named independent streams

Each repetition deterministically derives separate unsigned 32-bit seeds for:

- train/test splitting;
- Bayesian hyperparameter proposals;
- model fitting and probability-calibration folds;
- synthetic injection;
- stochastic prior-framework comparison procedures;
- bootstrap or other uncertainty calculations.

Seeds are derived from SHA-256 names rather than consumed from a shared random
generator. Adding a new condition or changing execution order therefore does
not change existing streams.

## Paired condition design

Within one repetition:

- clean and injected conditions use the same real-data split;
- hyperparameters come from the dataset-policy tuning bundle fitted on the
  disjoint development population and remain fixed across all repetitions and
  injection conditions;
- model seeds are shared across paired conditions;
- severity levels within one injection type share an injection seed, allowing
  the same anchor/random direction to be varied by severity;
- similarity bandwidths are fitted once from the clean training matrix and
  frozen across the paired evaluation conditions.

This permits paired condition-minus-baseline effects. Conditions from different
repetitions must not be paired.

## Reporting requirements

Store every repetition-level score and diagnostic. Planned summaries should
include mean, median, standard deviation, interquartile range, confidence
intervals, and paired changes from the clean baseline. Individual test rows
must not be treated as independent repetitions of a dataset-level experiment.

All prespecified repetitions must be reported. Runs may be excluded only under
a predefined technical-failure rule, with failures and retries retained in the
provenance record.

This repetition policy applies to E1, E2, and E3. E3 reuses each repetition's
matched E1/E2 data and derives framework-specific comparator streams; it does
not compare IF-CRED and prior frameworks on unrelated random splits.

The reporting layer computes repetition-level descriptive summaries,
percentile bootstrap confidence intervals, and paired injected-minus-clean
deltas. Individual rows are retained for diagnostics but are not treated as
independent experimental repetitions.
