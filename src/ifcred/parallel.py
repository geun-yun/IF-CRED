"""Central joblib policy for reproducible in-process parallelism."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from joblib import parallel_config


PARALLEL_BACKEND = "threading"


@contextmanager
def sklearn_parallelism(n_jobs: int) -> Iterator[None]:
    """Run joblib-backed sklearn work without spawning loky processes.

    Thread workers avoid loky's temporary memmap folders and semaphore
    resource tracker. This is particularly important on macOS with Python
    3.13, while keeping ``n_jobs`` as the single public concurrency knob.
    """

    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    with parallel_config(backend=PARALLEL_BACKEND, n_jobs=n_jobs):
        yield
