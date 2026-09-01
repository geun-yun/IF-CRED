# Planned IF-CRED experiments

The study contains three main experiments. All concern individual fairness.
Protected-attribute inclusion is an audit-design choice and context variable;
the study does not introduce a separate group-fairness experiment.

## E1 — CDFM inspection without synthetic injection

Run IF-CRED on the unmodified test populations of D6 Adult, D7 Credit Default,
and D8 Cleveland.

The repeated experimental dimensions are:

```text
dataset × protected-feature policy × outer seed
        × prespecified audit-design setting
```

Audit-design settings cover:

- primary and sensitivity values of neighbourhood `K`;
- the primary and defensible alternative distance-metric sets;
- the bandwidth estimator;
- calibrated versus native probability sensitivity;
- primary protected attribute included versus excluded.

One configuration must be designated primary before confirmatory evaluation;
the alternatives are sensitivity analyses rather than a search for favourable
scores.

Primary outputs are `C`, `D`, model-level and aggregate `F`, `M`, `V`, utility,
and individual-level diagnostic distributions. No demographic-parity or other
group-fairness criterion is included.

## E2 — CDFM inspection with synthetic injection

Run the same pipeline after real-anchored controlled injections. Within an
outer repetition, every condition shares the E1 split, clean-training-selected
hyperparameters, similarity scales, and paired random streams.

The component-targeted arms are:

- `C`: isolated instances and dominant-neighbour pairs;
- `D`: metric-disagreement instances and a matched stable-similarity control;
- selected `F`: geometry-matched benign and contradictory near duplicates;
- `M`: nested fractions of models receiving contradictory versus benign
  training exposure.

Localized contradictions may remain as an individual-level concentration
stress test, but they will not be interpreted as a group-fairness experiment or
used to add a group criterion to `V`.

Each mechanism uses multiple nested injection ratios and, where meaningful,
mechanism-specific severity levels. The analysis measures:

```text
ρ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
```

`ρ` is a controlled experimental factor, not a model hyperparameter and not an
input to Bayesian optimization. Model-F dose-response figures add the matched
clean condition at `ρ=0` for visual reference.

- whether the intended component responds in the expected direction;
- component cross-effects rather than assuming perfect isolation;
- monotonic or graded dose response;
- local detection around injected instances;
- paired changes relative to the clean E1 baseline;
- predictive utility separately from IF-CRED.

Constant, random, and smooth-but-uninformative prediction controls are retained
inside E2 as interpretation safeguards. They demonstrate that individual-
fairness consistency and predictive usefulness are distinct.

## E3 — Prior-framework comparison

Compare IF-CRED with:

- VF1: search-based individual-fairness verification (John et al., 2020);
- VF2: loss-ratio statistical individual-fairness auditing (Maity et al., 2021);
- VF3 (IFT-V): validity-filtered discriminatory-instance testing.

The comparators should run on applicable E1 clean settings and E2 controlled
conditions. The injected conditions provide known intended stressors for
comparing detection behaviour; clean observational data provide realistic
applicability and failure-case comparisons.

Native comparator outputs must remain on their own scales. E3 compares:

- detection decisions and violating-instance rates;
- response to known synthetic stressors and severities;
- agreement and complementary failure cases;
- cases where IF-CRED identifies weak evidence through `C`, `D`, or `M` even
  when a prior framework reports only its native fairness result;
- runtime, failure modes, and not-applicable cases.

E3 is the main incremental-validity experiment supporting the claim that
IF-CRED captures validation weaknesses not represented by existing approaches.

## Shared repeated design

The current provisional default is 30 outer seeds for **every experiment**.
Before those repetitions, each dataset contributes a disjoint 20% development
population for one-time model selection. The resulting five specifications per
dataset-policy branch are frozen; fresh model weights are still fitted within
every repetition. Development rows never enter E1–E3.
E1 and E2 reuse a paired repetition structure. E3 evaluates the corresponding
clean and injected case while its fitted models and matrices remain in memory,
rather than generating unrelated splits or serializing models for a later
stage. If a prior framework is stochastic, it receives its own named comparator
seed derived inside that repetition.

Consequently, all reported E1, E2, and E3 results must include repetition-level
distributions and uncertainty. No main result may be based on one preferred
random state. Final `K` values, metric definitions, severity grids, uncertainty
procedures, and hypotheses must be frozen after the exploratory computational
pilot.
