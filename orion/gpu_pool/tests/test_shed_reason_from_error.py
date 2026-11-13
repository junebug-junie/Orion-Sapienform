"""Reading a deliberate GPU-pool shed back out of a refusal string (2026-10-20)."""
from __future__ import annotations

import pytest

from orion.gpu_pool.shed import is_gpu_pool_refusal, shed_reason_from_error


@pytest.mark.parametrize(
    "error, expected",
    [
        ("gpu_pool_unavailable:shed:cabinet_hot", "cabinet_hot"),
        ("gpu_pool_unavailable:shed:orion_self_shed: pool refused new background grant", "orion_self_shed"),
        ("gpu_pool_unavailable:deadline", None),
        ("gpu_pool_unavailable:pool_bus_unavailable", None),
        ("gpu_pool_unavailable", None),
        ("gpu_pool_unavailable:shed:", None),
        ("timeout", None),
        ("shed:cabinet_hot", None),  # not a pool refusal string
        (None, None),
        (123, None),
    ],
)
def test_shed_reason_from_error(error, expected):
    assert shed_reason_from_error(error) == expected


@pytest.mark.parametrize(
    "error, expected",
    [
        ("gpu_pool_unavailable", True),
        ("gpu_pool_unavailable:deadline", True),
        ("gpu_pool_unavailable:shed:cabinet_hot", True),
        ("gateway_capacity_rejected:capacity_wait_budget_exhausted", False),
        ("timeout", False),
        ("gpu_pool_unavailable_extra", False),
        (None, False),
    ],
)
def test_is_gpu_pool_refusal(error, expected):
    assert is_gpu_pool_refusal(error) is expected
