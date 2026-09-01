# End-to-end experiment orchestration

## Repetition lifecycle

For one dataset, protected-feature policy, and outer repetition, the orchestrator
performs the following clean-training-only work once:

1. derive the named split seed and create a stratified train/test split;
2. fit preprocessing on the training rows and transform both partitions;
3. apply the same runtime feature projection to prediction and similarity;
4. load the five dataset-policy specifications selected previously on the
   disjoint development population; the legacy nested-tuning profile remains
   available only for sensitivity work;
5. fit and freeze every similarity bandwidth from the clean training matrix.

It then evaluates paired conditions. The clean baseline and every injection
condition reuse the split, selected hyperparameters, similarity scales, and
model seed. Training injections cause model retraining and recalibration but not
retuning. Test injections enter the common evaluation population and graph.

For each condition, the orchestrator:

1. assembles real and selected synthetic rows without crossing anchor
   partitions;
2. trains/calibrates all systems and obtains native plus calibrated test
   probabilities;
3. constructs one test graph using frozen training scales;
4. computes `C` and its balance/strength diagnostics;
5. computes `D` and pair reliabilities;
6. invokes the selected `F` binding on the same graph and probabilities;
7. derives `M`, `V`, `F_min`, and `V_worst`;
8. retains predictive utility separately;
9. retains row, pair, model, injection, and seed provenance.

The model-family-disagreement condition is also supported. It supplies
model-specific training partitions while enforcing one common evaluation
population and graph.

## Result storage

Condition outputs are written into immutable paths keyed by dataset, feature
policy, repetition, configuration hash, condition, variant, and requested ratio. A pre-existing result is
never overwritten. Each result contains:

- a compact JSON manifest containing C, D, F, M, predictive diagnostics,
  configuration, seed provenance, counts, and hashes of large trace vectors;
- an E3 comparison CSV for each primary clean or injected case;
- a SHA-256 checksum for the comparison table.

Fitted estimators, repeated training matrices, and full neighbourhood tensors
are released after each condition rather than persisted. E3 runs across the
five models in parallel using the declared worker count.

Writing uses a temporary sibling directory followed by an atomic rename, so an
interrupted write cannot appear as a complete result.

The dedicated tuning stage is separately resumable at model-family boundaries.
Its protocol is documented in `protocols/frozen_hyperparameter_selection.md`.

## Scope boundary

The orchestrator executes clean baselines and prepared synthetic conditions.
Prediction controls, statistical aggregation, artifact-only plotting, and the
VF1–VF3 comparison adapters are implemented as separate layers because their
units and native outputs differ from an ordinary five-model condition. E3 runs
on the live matched case before its models are released. Reporting reads only
compact outputs and never imports the study runner.
