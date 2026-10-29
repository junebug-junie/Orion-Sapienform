"""Ensure orion-thought ``app`` resolves to this service during tests."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_THOUGHT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _THOUGHT_ROOT.parents[1]


def _ensure_thought_paths() -> None:
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_THOUGHT_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_THOUGHT_ROOT))


def pytest_configure() -> None:
    # Runs before collection, so module-top ``import orion...``/``import app...``
    # in test files resolve. The autouse fixture below only fires at test
    # execution, which is too late for collection-time imports.
    _ensure_thought_paths()


@pytest.fixture(autouse=True)
def _thought_service_isolation() -> None:
    _ensure_thought_paths()
    yield


class GrantingPool:
    """Stands in for ``orion.gpu_pool.client.gpu_lease``: grants at once and records each call.
    Patched on the client module itself (visual_chain calls it through ``gpu_pool_client``), so
    one patch covers every copy of ``app.visual_chain`` the per-test isolation re-imports."""

    def __init__(self):
        self.calls: list[dict] = []
        self.held = 0

    @contextlib.asynccontextmanager
    async def __call__(self, bus, **kw):
        self.calls.append({"bus": bus, **kw})
        self.held += 1
        try:
            yield SimpleNamespace(lease_id="test-lease", release_outcome=None, release_detail=None)
        finally:
            self.held -= 1


@pytest.fixture(autouse=True)
def gpu_pool(monkeypatch):
    """A generate with no durable-run hold takes a pool `diffusion` lease (GPU pool stage 5.4).
    Tests have no pool, so by default it grants at once; a test that exercises refusal patches
    ``orion.gpu_pool.client.gpu_lease`` itself."""
    from orion.gpu_pool import client

    pool = GrantingPool()
    monkeypatch.setattr(client, "gpu_lease", pool)
    return pool


@pytest.fixture(autouse=True)
def _no_live_database(monkeypatch):
    """Self-readings (app/self_readings_cue.py) read Postgres on every stance/reverie call and
    are on by default. Point any unset DSN at a refused local port so a test never waits on DNS
    for the compose hostname; the cue renders nothing (fetch_failed), as it would live."""
    import os

    if not os.environ.get("POSTGRES_URI"):
        monkeypatch.setenv("POSTGRES_URI", "postgresql://x:x@127.0.0.1:1/nope")


@pytest.fixture(autouse=True)
def _no_live_physiology_bid(monkeypatch):
    """Unit tests must never POST a shadow bid to the live body economy (Patch C3b): the client
    returns None without a call. Tests of the shadow seam stub app.physiology_shadow.abid."""
    monkeypatch.setenv("ORION_PHYSIOLOGY_BID_CLIENT_ENABLED", "false")
