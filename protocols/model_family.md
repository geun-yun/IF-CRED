# Model-family implementation boundary

## Declared primary family

The primary model-stability component `M` will be evaluated across exactly five
model families selected for this project:

- logistic regression;
- multilayer perceptron (MLP);
- Gaussian Naive Bayes;
- Random Forest;
- a single decision tree.

This family spans linear, neural, probabilistic-generative, ensemble-tree, and
single-tree inductive biases. `M` remains conditional on this declared family; it does not
establish stability across every possible decision system.

## Shared-data contract

Within one run, every model receives the exact same training matrix and the
exact same evaluation matrix. The runtime protected-attribute policy is applied
before the runner, so prediction and similarity use the same feature view.

The runner fits models and probability calibrators only from training rows.
Evaluation outcomes are accessed only after predictions have been produced, to
calculate predictive diagnostics. The software includes a test that changes
evaluation outcomes and verifies that neither native nor calibrated predictions
change.

For a training-injection condition, the runner is called again on the augmented
training data. For a test-injection condition, trained systems predict the
augmented evaluation matrix. The experiment orchestrator implements this
routing while preserving anchor partition membership.

## Probability outputs

The primary weighted-prediction-consistency `F` receives probabilities from a
common cross-validated training-only calibrator. Sigmoid and isotonic
calibration are supported; the method and number of folds are recorded.

Each model's native probability is also retained for sensitivity analysis.
Gaussian Naive Bayes and Random Forest both expose native positive-class
probabilities. As with the other families, the primary analysis uses outputs
from the same external training-only calibration interface. Native outputs are
retained only for the declared probability sensitivity analysis.

Predictive utility is kept outside IF-CRED and reported separately for native
and calibrated outputs:

- accuracy at threshold 0.5;
- AUROC;
- Brier score;
- expected calibration error (ECE).

## Reproducibility

The runner controls model and calibration-fold seeds centrally. A run manifest
contains the base seed, per-model derived seed, matrix dimensions, calibration
configuration, ECE bin count, and the fully resolved estimator parameters.
Cross-validated calibration and estimator internals use the centrally declared
joblib `threading` backend and worker count; the backend is also recorded in the
experiment manifest.

## Deliberately not frozen

The following are not encoded as confirmatory scientific defaults:

- model hyperparameters;
- hyperparameter search spaces and selection rule;
- sigmoid versus isotonic calibration;
- calibration fold count;
- class weighting or resampling;
- the probability threshold used for secondary hard-label analyses;
- whether model hyperparameters are shared or dataset-specific.

These choices require exploratory evaluation and must then be frozen before the
confirmatory experiment. Once selected on clean training data, hyperparameters
must remain fixed across synthetic injection conditions so that condition-wise
retuning does not confound the intervention response.

The implemented training-only selection process and its remaining scientific
decisions are specified in `protocols/hyperparameter_tuning.md`.
