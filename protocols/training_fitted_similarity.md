# Training-fitted similarity scaling

## Separation of fitting and evaluation

For each dataset and protected-feature policy, the clean training matrix is used
to estimate one positive bandwidth for every declared distance definition. The
fitting graph uses the declared `K`, primary metric, retained-pair construction,
and bandwidth estimator. No outcomes, protected-group labels outside the chosen
feature matrix, or model predictions enter this operation.

The resulting bandwidth mapping is frozen. Evaluation graphs—including graphs
augmented with synthetic test instances—select their own test-to-test neighbours
but convert distances to similarities using those unchanged training-fitted
values.

Consequently, `C`, `D`, `F`, `M`, and `V` remain test-population quantities.
Training contributes only the distance-to-similarity scale.

## Safeguards and provenance

The calibration object:

- requires a non-fixed fitting policy;
- checks that every declared metric receives one finite positive bandwidth;
- records training row and feature counts;
- records a SHA-256 fingerprint of the exact training matrix;
- refuses evaluation matrices with a different feature width;
- converts the fitting specification into an explicit fixed-bandwidth audit
  specification;
- serializes both the fitting and application policies.

Bandwidth fitting is repeated for each outer data-split repetition because the
training population changes. Within that repetition, the fitted values are
held fixed across the clean baseline and every paired injection condition.

## Remaining scientific choice

The implementation supports the median positive retained-pair distance and the
median across individuals of their furthest retained-neighbour distance. The
choice between them, along with `K` and the metric set, remains subject to the
exploratory pilot and protocol freeze.
