# Bayesian cross-validated model selection

## Primary tuning strategy

The implemented primary strategy is seeded sequential model-based optimization
with Optuna's TPE sampler. A trial proposes one model configuration and receives
the mean stratified training-fold AUROC as its objective. Bayesian optimization
therefore selects configurations; it does not replace K-fold validation.

Every proposed configuration is evaluated on all folds. Trial pruning is not
used, so candidates are compared using equal validation evidence. Model fitting
inside a trial is single-worker; outer fold evaluation may use the declared
worker count. The optimizer itself runs sequentially to preserve deterministic
proposal order.

Fold evaluation uses joblib's in-process `threading` backend. This avoids
platform-sensitive `loky` process, semaphore, and memory-map cleanup while
retaining parallel fold evaluation. Base estimators remain single-worker inside
each fold, so there is only one active parallel layer.

Grid search remains implemented as a transparent fallback and sensitivity
analysis.

## Recommended exploratory pilot spaces

The named `recommended_bayesian_search_spaces()` configuration declares:

| Model | Trial budget | Search dimensions |
|---|---:|---|
| Logistic regression | 30 | log-scaled `C` from `1e-4` to `1e4`; L1/L2 penalty; optional balanced class weights |
| MLP | 60 | four one/two-layer architectures; ReLU/tanh; log-scaled regularization and learning rate; batch size 32/64/128 |
| Gaussian Naive Bayes | 20 | log-scaled variance smoothing from `1e-12` to `1e-6` |
| Random Forest | 50 | trees, depth, split/leaf support, feature sampling, and class weighting |
| Decision tree | 40 | criterion, depth, split/leaf support, feature sampling, and pruning alpha |

These full budgets reflect the relative dimensionality of the spaces. They
total 200 trials per dataset/policy combination. With five-fold CV that is
1,000 candidate fits. Across three datasets and two protected-feature policies,
one-time full tuning therefore implies 6,000 CV fits. The recommended pilot
budget uses 95 trials, or 2,850 five-fold candidate fits across all six branches.

This is a computational planning figure, not a claim that equal trial counts or
these exact ranges are optimal.

## Reproducibility record

The manifest includes:

- Optuna and sampler identity/version;
- sampler startup-trial count and seed lineage;
- fold construction and scoring rule;
- every distribution and categorical label-to-value mapping;
- each model's fixed trial budget;
- every resolved trial configuration;
- every fold score, mean score, and population standard deviation;
- the selected immutable `ModelSpec`.

## Freeze boundary

The spaces above are sensible starting choices for a full-data computational
pilot, not yet confirmatory defaults. The pilot must measure runtime,
convergence, failed configurations, boundary-seeking selections, and score
stability across seeds. We then either revise the ranges or freeze them before
the confirmatory experiment. Test-set performance must not inform that revision.
