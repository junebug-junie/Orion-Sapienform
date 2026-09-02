"""Hub board route tests.

Deliberately does not hit Postgres: the point here is that the route contract
holds and degrades honestly, not that the database is up. Live-data correctness
is covered by orion/sentience_striving_program/tests and by the CI gate.
"""

from __future__ import annotations

import os
from pathlib import Path

# Importing `scripts.api_routes` (below, in the router-binding test) constructs
# Hub's full Settings at module scope, which requires these five channel keys
# from a `.env` that is gitignored and therefore absent from any worktree or CI
# checkout. Same module-level setdefault convention as
# tests/test_causal_geometry_api.py:13-17. setdefault, not assignment, so a real
# environment still wins; these are not keys conftest pops.
os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

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


def test_engine_is_autocommit_and_read_only(monkeypatch):
    """Guard the two engine settings that carry this path's safety.

    Without AUTOCOMMIT, psycopg2 opens an implicit transaction and the first
    failing statement poisons every later one on the same connection with
    "current transaction is aborted" -- and `build_state` reuses ONE connection
    across every instrument's storage read and every SQL claim, so a single
    genuinely drifted claim rendered the whole board ERROR while naming the wrong
    cause. Without `default_transaction_read_only` this path silently drops the
    guarantee `orion.db_readonly.open_readonly_connection` gives the CLI.

    This is a construction assertion, not a runtime one: it proves the settings
    are still requested, not that Postgres honoured them. Both were confirmed
    live against the real database on 2026-09-02 (the cascade reproduced, then
    stopped; a write failed with "cannot execute CREATE TABLE in a read-only
    transaction").
    """
    seen = {}

    def _fake_create_engine(uri, **kwargs):
        seen["uri"] = uri
        seen.update(kwargs)
        return object()

    import sqlalchemy

    monkeypatch.setattr(routes, "_ENGINE", None)
    monkeypatch.setattr(sqlalchemy, "create_engine", _fake_create_engine)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://u@h/db")

    assert routes._engine() is not None
    assert seen["isolation_level"] == "AUTOCOMMIT"
    assert "default_transaction_read_only=on" in seen["connect_args"]["options"]
    monkeypatch.setattr(routes, "_ENGINE", None)


def test_routes_are_registered_exactly_once_in_the_assembled_hub_router():
    """Both paths must be bound, once each, to THIS module's handlers.

    A duplicate path does not raise in FastAPI -- it silently serves whichever
    router registered first. A grep for the literal path proves no collision
    exists today; this proves the registration in `api_routes.py` actually took
    effect and still points here, which grep cannot.
    """
    from scripts import api_routes

    bound: dict[str, list[str]] = {}
    for route in api_routes.router.routes:
        path = getattr(route, "path", None)
        if path in ("/sentience-program", "/api/sentience-program"):
            bound.setdefault(path, []).append(
                getattr(route.endpoint, "__module__", "?")
            )

    for path in ("/sentience-program", "/api/sentience-program"):
        registrations = bound.get(path, [])
        assert len(registrations) == 1, (
            f"{path} registered {len(registrations)} times; FastAPI would serve "
            "the first silently"
        )
        assert "sentience_program_routes" in registrations[0]


def test_handlers_are_sync_so_they_cannot_block_the_event_loop():
    """Blocking work (Postgres, a subprocess, a 4,300-file walk) must not be async.

    An `async def` handler doing that work freezes every other Hub route and
    websocket for its duration. Pinned as a test because the regression is
    invisible in review -- the code reads fine either way.
    """
    import inspect

    for fn in (routes.sentience_program_state, routes.sentience_program_page):
        assert not inspect.iscoroutinefunction(fn), (
            f"{fn.__name__} is async but performs blocking work"
        )
