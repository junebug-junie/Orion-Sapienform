"""SHED_BACKOFF is process-wide state in the pool client: a test that ends in a shed refusal must
not leave later tests' leases refused locally."""
import pytest

from orion.gpu_pool.client import SHED_BACKOFF


@pytest.fixture(autouse=True)
def _reset_shed_backoff():
    SHED_BACKOFF.reset()
    yield
    SHED_BACKOFF.reset()
