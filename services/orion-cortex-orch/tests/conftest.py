"""Pytest path setup so `app` and `orion` import from repo root without manual PYTHONPATH."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_orch_root = Path(__file__).resolve().parents[1]
_repo_root = Path(__file__).resolve().parents[3]
for _p in (_repo_root, _orch_root):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


@pytest.fixture(autouse=True)
def _isolate_substrate_control_surface(tmp_path, monkeypatch):
    """Never let this suite write Orion's live routing threshold.

    `decision_router.py` reads `routing.chat_reflective_lane_threshold` from a
    shared control surface that orion-hub's mutation applier writes, and
    `test_auto_router.py` sets it through the module-global store to exercise the
    gate. That global resolves its database from the ambient environment, so with
    a `DATABASE_URL` in scope -- which is the case inside the container, since
    `Dockerfile` COPYs `tests/` into the image -- the test writes production and
    never restores it. `decision_router.py` would then demote every depth>=2
    decision below the leftover threshold, indefinitely.

    This is not hypothetical for this surface: production held
    `value=0.5, actor="scheduler_seed"` -- a pytest fixture string -- with 4,925
    updates on the row, from the equivalent leak in the hub suite.

    Autouse and unconditional: the point is that the next test to touch this
    surface cannot reintroduce the leak by forgetting to opt in.
    """
    from orion.substrate import mutation_control_surface

    for key in (
        "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL",
        "SUBSTRATE_POLICY_POSTGRES_URL",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    previous = mutation_control_surface._CONTROL_SURFACE_STORE
    mutation_control_surface._CONTROL_SURFACE_STORE = (
        mutation_control_surface.RuntimeControlSurfaceStore(
            sql_db_path=str(tmp_path / "control-surface-isolated.sqlite3")
        )
    )
    try:
        yield
    finally:
        mutation_control_surface._CONTROL_SURFACE_STORE = previous


@pytest.fixture(autouse=True)
def compactor_durable_sim(request, monkeypatch):
    """Compactor passes hand their LLM digest calls to a ``compactor.digest`` durable run
    (orion-durable-runs). In this suite that run is simulated in-line (tests/compactor_durable_sim.py)
    so orch tests keep covering fetch -> digest -> finalize end to end. Tests of the real submission
    path opt out with ``@pytest.mark.real_durable_submit``."""
    if request.node.get_closest_marker("real_durable_submit"):
        yield None
        return
    import types

    from app import workflow_runtime as wr

    from compactor_durable_sim import InlineDurableRun

    namespaces = [vars(wr)]
    for obj in list(vars(request.module).values()):
        ns = getattr(obj, "__globals__", None)
        if ns is None and isinstance(obj, types.ModuleType):
            ns = vars(obj)
        if (isinstance(ns, dict) and ns.get("__name__") == "app.workflow_runtime"
                and not any(ns is seen for seen in namespaces)):
            namespaces.append(ns)
    yield InlineDurableRun(namespaces).install(monkeypatch)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_durable_submit: use the real compactor durable-run submission")
