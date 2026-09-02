"""Hub board route tests.

Deliberately does not hit Postgres: the point here is that the route contract
holds and degrades honestly, not that the database is up. Live-data correctness
is covered by orion/sentience_striving_program/tests and by the CI gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import scripts.sentience_program_routes as routes


@pytest.fixture()
def client(monkeypatch):
    # No engine -> conn is None -> SQL claims must report ERROR, never HOLDS.
    monkeypatch.setattr(routes, "_engine", lambda: None)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_state_endpoint_returns_every_instrument(client):
    body = client.get("/api/sentience-program").json()
    assert body["instruments"], "no instruments returned"
    assert set(body["outcomes"]) == {"O1", "O2", "O3", "O4"}
    assert body["consumers_resolved"] is False


def test_sql_claims_error_without_a_database_rather_than_passing(client):
    """A board that renders green when it cannot see the data is worse than none."""
    body = client.get("/api/sentience-program").json()
    sql_claims = [
        c
        for i in body["instruments"]
        for c in i["claims"]
        if c["observed"] is None and c["status"] != "MANUAL"
    ]
    assert sql_claims, "expected at least one claim needing a database"
    assert all(c["status"] == "ERROR" for c in sql_claims)


def test_every_instrument_carries_outcome_and_unlock(client):
    body = client.get("/api/sentience-program").json()
    for inst in body["instruments"]:
        assert inst["outcome"] in body["outcomes"]
        assert inst["unlock"].strip(), f"{inst['id']} has an empty unlock narrative"
        assert inst["module"]


def test_page_renders_and_references_its_asset(client, monkeypatch):
    """The page route substitutes the asset version and serves the real template.

    `scripts.main` is stubbed rather than imported: importing it constructs Hub's
    full Settings at module scope, which needs a populated `.env` that is
    gitignored and therefore absent from any worktree or CI checkout. Stubbing
    keeps this a test of THIS route's logic instead of a test of whether the
    environment happens to be provisioned.
    """
    import sys
    import types

    stub = types.ModuleType("scripts.main")
    stub.TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
    stub.build_hub_ui_asset_version = lambda: "test-asset-v"
    monkeypatch.setitem(sys.modules, "scripts.main", stub)

    html = client.get("/sentience-program").text
    assert "Sentience Striving Program" in html
    assert "/static/js/sentience-program.js" in html
    # The placeholder must be substituted, or browsers cache a stale bundle
    # across deploys.
    assert "{{HUB_UI_ASSET_VERSION}}" not in html
    assert "test-asset-v" in html


def test_page_404s_when_the_template_is_missing(client, monkeypatch):
    """A missing template must 404, not render a blank page that reads as 'no data'."""
    import sys
    import types

    stub = types.ModuleType("scripts.main")
    stub.TEMPLATES_DIR = Path("/nonexistent-templates-dir")
    stub.build_hub_ui_asset_version = lambda: "v"
    monkeypatch.setitem(sys.modules, "scripts.main", stub)

    resp = client.get("/sentience-program")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "sentience_program_template_missing"


def test_board_still_renders_when_the_database_is_unreachable(monkeypatch):
    """An outage must degrade to manifest-only, never to a 500 or a blank board.

    Regression guard for a flaw found by a live run: the route originally raised
    500 on any connection failure, which would have removed the operator's only
    view of the program at exactly the moment something was wrong.
    """

    class _DeadEngine:
        def raw_connection(self):
            raise RuntimeError("could not translate host name 'orion-sql-db'")

    monkeypatch.setattr(routes, "_engine", lambda: _DeadEngine())
    app = FastAPI()
    app.include_router(routes.router)
    resp = TestClient(app).get("/api/sentience-program")

    assert resp.status_code == 200
    body = resp.json()
    assert body["instruments"], "board went blank on a database outage"
    assert "could not translate host name" in body["db_error"]
    # Code-presence and retention facts do not need the database and must survive.
    assert all(i["module"] for i in body["instruments"])
    assert any(i["retention_hours"] for i in body["instruments"])
