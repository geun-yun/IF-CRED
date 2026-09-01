# Training-only K-fold hyperparameter tuning

## Implemented sequence

The recommended main workflow is now the dedicated-development frozen strategy
in `protocols/frozen_hyperparameter_selection.md`. The sequence below remains
available as a nested-tuning sensitivity profile and as the underlying search
implementation.

Hyperparameter tuning is a stage before probability calibration and final model
fitting:

1. Start with the clean training partition under one declared protected-feature
   policy.
2. Construct shuffled, stratified `K` folds using a recorded seed.
3. Evaluate every candidate in the declared finite grid on the same folds.
4. Select the candidate with the highest mean validation score. Ties follow the
   deterministic candidate order recorded in the manifest.
5. Return one immutable `ModelSpec` for each of the five model families.
6. Freeze those selected specifications across the relevant synthetic injection
   conditions.
7. For each condition, retrain and calibrate the models using the condition's
   training data without repeating hyperparameter selection.

The current default scoring capability is AUROC. The scoring rule is explicit
and serializable rather than embedded in a model implementation.

The declared `n_jobs` value controls parallel fold evaluation through joblib's
`threading` backend. Candidate estimators remain single-worker within those
folds, preventing nested worker pools.

## Leakage boundary

The tuning function accepts only `X_train` and `y_train`. It has no evaluation
or test arguments. Probability calibration occurs later through separate
stratified training folds. Test outcomes are used only after final prediction
for utility reporting and IF-CRED evaluation.

This design uses the held-out test set as the final evaluation rather than as a
source of model-selection feedback. Cross-validation scores are selection
evidence, not final predictive-performance estimates.

## Recorded evidence

The tuning manifest retains:

- the splitter, fold count, shuffle policy, scoring rule, and selection rule;
- the base seed and deterministic per-model seed;
- the complete parameter grid;
- every candidate's full hyperparameters;
- every fold-level validation score;
- candidate mean, population standard deviation, and rank;
- the selected immutable model specification.

## Decisions not yet frozen

The following remain exploratory protocol choices:

- the value of `K` for tuning;
- the AUROC search grids for each model;
- whether class weighting is allowed in the grids;
- whether tuning is separate for protected-included and protected-excluded
  policies or a shared specification is used for the paired comparison;
- whether grids are dataset-specific;
- whether a one-standard-error or complexity-aware selection rule should be a
  sensitivity analysis;
- the computational budget and any early-stopping rules.

The engineering smoke tests use deliberately tiny grids and subsets. Their
selected values are not scientific results and must not be copied into the
confirmatory configuration.

The primary sequential search implementation and its proposed exploratory
budgets are documented in `protocols/bayesian_optimization.md`. Grid search is
retained as the deterministic fallback and sensitivity analysis.
