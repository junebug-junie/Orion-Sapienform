"""Hub's "I am done with this run's seat" call must reach the release, not the generic control route.

Live 2026-11-15 (#3255): every Door-A hold in 30 days ended ``outreach_timeout`` after sitting idle
for DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC (1,800 s); none ended ``outreach_done``. Hub did call
``POST /runs/{id}/release-outreach-lease`` the moment it refused to compose, and got
``400 {"detail":"action must be pause, resume or cancel"}``: the generic
``/runs/{run_id}/{action}`` route was declared first and FastAPI matches in declaration order.
"""
from __future__ import annotations

import pytest


class _Admission:
    def __init__(self):
        self.released, self.controlled = [], []

    async def release_outreach(self, run_id):
        self.released.append(run_id)
        return {"released": True, "run_id": run_id}

    async def control(self, run_id, action):
        self.controlled.append((run_id, action))
        return {"run_id": run_id, "action": action}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from fastapi.testclient import TestClient

    from app import main

    admission = _Admission()
    monkeypatch.setattr(main, "_admission", lambda: admission)
    return admission, TestClient(main.app)  # no `with`: the lifespan (bus, pools) never starts


def test_hub_release_call_releases_the_seat_at_once(client):
    admission, c = client
    # The exact path Hub builds (curiosity_investigation._outreach_lease_release_url).
    response = c.post("/runs/run-one/release-outreach-lease")
    assert response.status_code == 200, response.text
    assert response.json() == {"released": True, "run_id": "run-one"}
    assert admission.released == ["run-one"]
    assert admission.controlled == []


def test_release_route_maps_runtime_errors(client, monkeypatch):
    admission, c = client

    async def missing(run_id):
        raise KeyError(run_id)

    async def not_door_a(run_id):
        raise ValueError("no door-a outreach lease hold on this run")

    monkeypatch.setattr(admission, "release_outreach", missing)
    assert c.post("/runs/nope/release-outreach-lease").status_code == 404
    monkeypatch.setattr(admission, "release_outreach", not_door_a)
    assert c.post("/runs/live/release-outreach-lease").status_code == 409


def test_generic_control_route_is_unchanged(client):
    admission, c = client
    assert c.post("/runs/run-one/cancel").status_code == 200
    assert admission.controlled == [("run-one", "cancel")]
    assert c.post("/runs/run-one/bogus").status_code == 400
    assert admission.released == []


def _bare_runtime(monkeypatch, *, pool_confirms: bool):
    """release_outreach on a runtime with no database: a completed run with one kept hold."""
    import asyncio

    from app import admission_runtime as ar

    lease = {"lease_id": "hold-1", "generation": 1, "role": "agent", "holder": "durable-runs:run-one"}
    rt = object.__new__(ar.AdmissionRuntime)
    rt.outreach = {"run-one": {"lease": lease, "since": None, "beat": 0.0}}
    rt._wake = asyncio.Event()

    class _Store:
        async def get_run(self, run_id):
            return {"terminal": "completed", "request": {}}

        async def history(self, run_id):
            return []

    rt.store = _Store()
    ended = []

    async def end_hold(state, lease_id, lease, reason):
        ended.append((state["run_id"], lease_id, reason))
        return pool_confirms

    monkeypatch.setattr(rt, "_end_hold", end_hold)
    return rt, ended


def test_release_says_released_only_when_the_pool_confirmed(monkeypatch):
    import asyncio

    rt, ended = _bare_runtime(monkeypatch, pool_confirms=True)
    assert asyncio.run(rt.release_outreach("run-one")) == {"released": True, "run_id": "run-one"}
    assert ended == [("run-one", "hold-1", "outreach_done")]

    # The pool call failed: the release is queued for retry, and Hub is not told it is done.
    rt, _ = _bare_runtime(monkeypatch, pool_confirms=False)
    assert asyncio.run(rt.release_outreach("run-one")) == {
        "released": False, "run_id": "run-one", "reason": "pending_retry"}
