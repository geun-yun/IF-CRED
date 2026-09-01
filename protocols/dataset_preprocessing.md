# Exploratory dataset and preprocessing protocol

Status: implemented and smoke-tested; normative feature roles and final split
settings are not frozen for confirmation.

## Fixed dataset identities

| ID | UCI record | Outcome | Primary protected field |
|---|---:|---|---|
| D6 Adult Census Income | 2 | `income` mapped to income above USD 50,000 | `sex` |
| D7 Default of Credit Card Clients | 350 | `Y`, default payment | `X2` |
| D8 Cleveland Heart Disease | 45 | `num > 0`, disease presence | `sex` |

Acquisition uses `ucimlrepo.fetch_ucirepo` with the registered UCI ID. The
original artifact retains feature and target values as returned. An optional
cache writes content-addressed, immutable directories containing those source
values and a manifest. The manifest records the UCI record, DOI, retrieval
time, licence, dimensions, source null/token counts, source target counts, and
a SHA-256 checksum.

No default subsampling is performed.

## Preprocessing normalization

- Adult strips surrounding whitespace, converts both target-file spellings
  (`>50K` and `>50K.`) to one positive class, and converts literal `?` tokens
  to missing values.
- Credit retains UCI names `X1`–`X23`, encodes `Y` as binary, treats `X2`,
  `X3`, and `X4` as categorical, and excludes no rows.
- Cleveland converts `num > 0` to disease presence and retains missing `ca`
  and `thal` values for train-fitted imputation.
- Categorical values are normalized to stable string tokens so numerical UCI
  codes are not accidentally treated as continuous measurements.

## Split policy

For the recommended frozen-profile study, a fixed 20% development population
is first separated from each original dataset. It is used only for one-time
hyperparameter selection and never appears in E1–E3. Repeated train/test splits
are drawn from the remaining 80%. The requested and realized development
fractions, original row indices, seed, stratification strategy, and source
snapshot checksum are recorded.

The implemented primary splitter stratifies by outcome × recorded primary
protected value when every cell has sufficient support for both partitions.
It falls back to outcome-only stratification otherwise. The resolved strategy
is recorded with the split.

The split is made before fitting any transformation. Split indices refer only
to original rows; synthetic rows are subsequently assigned to their anchor's
partition by the injection layer.

## Preprocessing

One technical transformation is fitted on original training rows:

- numeric fields: median imputation followed by standard scaling;
- categorical fields: explicit `__MISSING__` imputation followed by dense
  one-hot encoding;
- unseen test categories: ignored by the fitted encoder;
- binary categorical fields: one encoded column rather than two redundant
  columns;
- protected exclusion: an optional runtime projection from the one stored
  preprocessed dataset.

For each UCI source there are exactly two dataset artifacts: the original UCI
dataset as received and one train-fitted preprocessed dataset. The preprocessed
dataset contains the shared features plus the primary protected attribute and
retains a source-feature-to-transformed-column map.

## Symmetric experimental matrices

The default runtime view includes the primary recorded protected field. A run
may request its exclusion; this removes the protected transformed column from
the same preprocessed dataset and does not create a second processed copy.
Within either option, prediction models and neighbourhood construction receive
the same matrix. IF-CRED is calculated separately when both options are run;
their model-level values are not mixed within one `M` score. Adult race remains
protected context and is not part of the preprocessed experiment matrix.

## Exploratory shared feature roles

### D6 Adult

Used identically for prediction and similarity: age, workclass, education-num,
occupation, capital gain/loss, and hours per week.

Excluded: sex and race; survey weight `fnlwgt`; redundant categorical
education; marital status, relationship, and native country for normative and
proxy-risk reasons.

### D7 Credit Default

Used identically for prediction and similarity: credit limit, age, repayment
history, bill amounts, and payment amounts (`X1`, `X5`–`X23`).

Excluded: sex (`X2`), education (`X3`), and marital status (`X4`).

### D8 Cleveland

Used identically for prediction and similarity: all clinical features except
sex. Continuous perturbations are restricted to age, resting blood pressure,
cholesterol, maximum heart rate, and oldpeak.

## Decisions still requiring review

- whether Adult workclass and occupation are morally legitimate comparison
  features rather than consequences of structural inequality;
- whether Adult capital gain/loss should define similarity;
- whether Credit repayment history and credit limit encode prior discriminatory
  decisions and require sensitivity exclusions;
- clinical justification and scaling for Cleveland categorical variables;
- final test fraction and repeated/nested split design;
- final reporting hierarchy for the protected-excluded and protected-included
  experiments;
- whether subgroup-stratified splitting is adequate for Cleveland's small
  intersectional cells.

These choices must be reviewed before the confirmatory configuration is frozen.
