# Dedicated-development hyperparameter selection

## Purpose

The main repeated experiment does not repeat Bayesian optimization inside all
30 outer seeds. Instead, each dataset is partitioned once into a dedicated 20%
development population and an 80% repeated-experiment population. The
partition is stratified by outcome × primary protected attribute when support
permits and is derived from the recorded root seed.

Development rows are used only for model selection. They are excluded from all
later E1, E2, and E3 train/test splits. This prevents a row that influenced
hyperparameter selection from later being treated as held-out experimental
evidence.

## Six tuning branches and 30 specifications

Tuning is performed for:

```text
D6, D7, D8 × protected included, protected excluded
```

Each of these six branches independently searches Logistic Regression, MLP,
Gaussian Naive Bayes, Random Forest, and Decision Tree settings. The output is therefore 30 model
specifications, not six shared parameter vectors. Only hyperparameters are
frozen; every experimental repetition still fits fresh model weights.

## Pilot budget

The default pilot budgets are 15, 25, 10, 25, and 20 trials respectively.
Every trial uses five-fold development-only cross-validation. The original
30/60/20/50/40 budgets remain available through `--budget full` if convergence
evidence justifies them.

Each completed model-family search is written atomically before the next model
starts. A resumed tuning command reuses compatible model checkpoints.

## Freeze boundary

The pilot should be inspected for convergence, boundary-seeking selections,
and failed configurations. A confirmatory run must use an immutable tuning
directory, root seed, development fraction, and source dataset snapshot. The
runner validates these fields before accepting frozen specifications.
