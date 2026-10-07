"""A generate with no durable-run hold takes a GPU pool `diffusion` lease (stage 5.4). `evals/`
is a sibling of `tests/`, so tests/conftest.py's fixture never applies here: same fixture, a pool
that grants at once, patched on ``orion.gpu_pool.client`` (visual_chain calls it through the
module, so every re-imported copy of app.visual_chain sees it). Without it the honesty eval would
wait on a pool that is not there and record resource_deferred instead of the scenario it tests.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _gpu_pool_grants(monkeypatch):
    from orion.gpu_pool import client

    @contextlib.asynccontextmanager
    async def grant(bus, **kw):
        yield SimpleNamespace(lease_id="eval-lease", release_outcome=None, release_detail=None)

    monkeypatch.setattr(client, "gpu_lease", grant)
