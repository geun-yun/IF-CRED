import pytest
from joblib.parallel import get_active_backend

from ifcred.parallel import PARALLEL_BACKEND, sklearn_parallelism


def test_parallel_policy_uses_threads_for_multiple_workers():
    with sklearn_parallelism(2):
        backend, _ = get_active_backend()
        assert backend.__class__.__name__ == "ThreadingBackend"
        assert PARALLEL_BACKEND == "threading"


def test_parallel_policy_rejects_zero_workers():
    with pytest.raises(ValueError, match="non-zero"):
        with sklearn_parallelism(0):
            pass
