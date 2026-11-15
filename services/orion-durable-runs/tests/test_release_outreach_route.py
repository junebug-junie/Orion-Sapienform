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
