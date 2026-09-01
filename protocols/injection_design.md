# Exploratory real-anchored injection design

Status: implemented for smoke testing; parameter values are not frozen for a
confirmatory run.

## Shared experimental rule

Synthetic candidates are derived from real anchors and assigned to the same
train or test partition as their anchor. Models are trained on augmented
training data and produce their own predictions. No injection overwrites model
probabilities.

The injection ratio `rho` is the number of selected synthetic rows divided by
the number of original rows. Candidate priority is fixed for a random seed, so
higher ratios contain lower-ratio candidates. Multi-row injection groups are
selected atomically, which may make the realised ratio differ slightly from
the requested ratio.

Feature matrices supplied to the generators are expected to be transformed by
preprocessing fitted on original training data. Each injection experiment uses
one shared matrix for both prediction and similarity. Generators perturb only
the declared legitimate continuous columns. Protected fields and categorical
encodings are copied unless a future protocol explicitly defines a
protected-counterfactual condition.

## Implemented conditions

### Benign near duplicate

For anchor `i`, the perturbation radius is
`radius_fraction * kth_neighbour_distance_i`. The label is retained. This is
the geometry-matched negative control for contradictory near duplicates.

Expected primary response: no material fairness reduction beyond sampling and
model-fitting variation.

### Contradictory near duplicate

Uses exactly the benign condition's features, anchors, radii, directions, and
priority order, but flips the binary label. This targets `F` through learned
model behaviour.

A flipped label is stored as a known training contradiction, not automatically
as a known unfair prediction. Pair-level unfairness claims require observed
model-output differences under a preregistered criterion.

### Localized subgroup contradiction (available generator, excluded from the main study)

The library can apply the contradictory near-duplicate mechanism only to anchors selected by
a protocol-supplied Boolean subgroup mask. The generator enforces a minimum
eligible subgroup size. Both global injection prevalence and subgroup-relative
prevalence must be reported.

This generator is not scheduled by E1–E3 because the agreed main study concerns
individual-fairness validation and does not add a group-fairness experiment.

### Isolated instance

Moves same-label candidates in the declared continuous feature subspace until
their maximum Gaussian similarity to original background rows is at or below
`maximum_background_similarity`. The Gaussian bandwidth is the median original
`k`th-neighbour distance.

Expected primary response: lower coverage strength `C` around injected rows.

### Dominant-neighbour pair

Creates atomic two-row groups outside normal support. Within-pair similarity is
at least `minimum_pair_similarity`; similarity to original background rows is
at most `maximum_background_similarity`. Labels and non-perturbed columns are
copied from the anchor.

Expected primary response: reduced ESS balance within `C`, caused by one strong
comparison dominating weak alternatives.

### Metric-disagreement instance

Samples local perturbations at declared fractions of each anchor's local
`k`th-neighbour distance. A protocol-supplied similarity evaluator calculates
all declared similarities. The selected candidate must remain sufficiently
similar under the primary metric while reaching the requested cross-metric
pair-reliability level.

Expected primary response: lower `D`. Confirmatory runs must require the target
reliability to be achieved; exploratory runs may retain and flag the closest
candidate when it is not.

The frozen confirmatory bank uses strict reliability ceilings of `0.90`
(moderate) and `0.75` (strong), a radius-fraction search over `0.02`, `0.05`,
`0.10`, `0.20`, `0.40`, `0.80`, `1.20`, `2.0`, `3.0`, and `4.0`, and eight
directions per radius. Infeasible anchors are excluded rather than replaced by
above-threshold candidates. The retained bank must support the largest declared
injection ratio, and candidates enter in ascending achieved reliability so the
nested dose sequence begins with the strongest disagreements.

### Model-family disagreement

Creates a reproducible nested allocation of model systems. Unaffected systems
train on benign near-duplicate augmentation; affected systems train on the
geometry-identical contradictory augmentation. Every system is evaluated on a
common benign-labelled test matrix.

Expected primary response: increasing dispersion of model-level selected
fairness scores and therefore lower `M` when exposure creates heterogeneous
model behaviour. Exposure does not guarantee an effect, so achieved prediction
and fairness dispersion must be retained.

## Parameters requiring exploratory calibration

- neighbourhood size `k`;
- near-duplicate `radius_fraction`;
- injection ratios `rho`;
- isolated and dominant-pair similarity thresholds;
- metric-disagreement candidate radii and target reliability;
- subgroup eligibility and minimum support;
- affected-model fraction and allocation seed;
- criterion for calling an injected contradiction an observed unfair pair.

Calibration choices must be made without confirmatory outcomes and written to
the frozen protocol before the final run.
