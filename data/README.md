# Data acquisition

The repository does not commit downloaded UCI records. Create immutable,
content-addressed normalized snapshots with:

```bash
python -m ifcred.data --cache-root data/snapshots
```

Install the project first (for example, `python -m pip install -e '.[test]'`)
or run with `PYTHONPATH=src` during development.

Each snapshot contains the original UCI feature and target values as received,
plus a manifest with source identifiers, retrieval time, source null/token
counts, target-value counts, and SHA-256 checksum. Cleaning, target conversion,
imputation, scaling, and encoding occur only in the preprocessing layer.
