# Audit graph implementation boundary

## Purpose

The audit graph turns the single preprocessed experiment matrix into the local
comparison evidence used by IF-CRED. In a run, model prediction and neighbour
construction receive exactly the same runtime matrix. That matrix includes the
primary protected attribute by default; the paired exclusion experiment removes
the same protected column from both uses.

## Implemented contract

For a declared primary distance, the implementation constructs a directed,
fixed-`K` nearest-neighbour graph with no self-neighbours. It retains row IDs so
that each comparison can be traced back to its real or synthetic source row.

Only the primary metric performs neighbour search. Every alternative plausible
distance is evaluated on those same retained pairs. This is necessary for `D`:
it measures whether the similarity assigned to a fixed comparison is stable
across definitions, rather than comparing unrelated neighbourhoods.

Each retained distance is converted to a similarity in `[0, 1]`. The primary
similarities are the graph weights used by `C` and the selected plug-in `F`.
The complete metric-by-individual-by-neighbour similarity tensor is used by
`D`.

The implementation does not allocate a full all-pairs distance matrix.
Nearest-neighbour search is delegated to a configured search backend, and
alternative retained-pair distances are evaluated in bounded batches.

## Reproducibility record

The graph specification serializes:

- `K`;
- the primary metric;
- every declared metric, implementation name, and parameter mapping;
- the bandwidth policy and any fixed bandwidths;
- the distance-to-similarity weighting rule;
- neighbour-search and retained-pair batching settings.

Neighbour ordering is made deterministic when returned distances tie by using
the source row position as the secondary key.

## Deliberately not frozen

The software currently supports an exploratory Gaussian conversion and three
bandwidth policies: a fixed declared value, the median positive retained-pair
distance, or the median across individuals of their furthest retained primary
neighbour distance. These are capabilities, not a scientific endorsement.

The following remain protocol decisions and must be justified through the
exploratory study before any confirmatory run:

- the value or grid for `K`;
- which distance is primary;
- which alternative distance definitions form the set assessed by `D`;
- feature weighting or learned metric parameters;
- the bandwidth policy and its calibration data;
- whether one graph specification is defensible across all three domains.

No default confirmatory graph is encoded in the library.

The train/evaluation separation used to estimate and freeze the similarity
scales is documented in `protocols/training_fitted_similarity.md`.
