"""Hub Personality tab routes on durable-runs (rev 4 move 4): input checks, 503/404/409, no database."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from fastapi.testclient import TestClient

    from app import main

    return main, TestClient(main.app)  # no `with`: the lifespan (bus, pools) never starts


class Drv:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    async def juniper_verdict(self, disposition_id, verdict):
        self.calls.append((disposition_id, verdict))
        if self.exc is not None:
            raise self.exc
        return self.result


def test_routes_503_when_personality_is_disabled(client, monkeypatch):
    main, c = client
    monkeypatch.setattr(main, "personality", None)
    for path in ("/personality/choices", "/personality/tendencies", "/personality/arms"):
        assert c.get(path).status_code == 503, path
    assert c.post("/personality/tendencies/disp-1/verdict", json={"verdict": "no"}).status_code == 503


def test_verdict_rejects_anything_but_yes_or_no_before_touching_the_driver(client, monkeypatch):
    main, c = client
    drv = Drv(result={})
    monkeypatch.setattr(main, "personality", drv)
    monkeypatch.setattr(main, "_checkpointer_cm", object())
    for body in ({"verdict": "maybe"}, {}, {"verdict": "no", "extra": 1}):
        assert c.post("/personality/tendencies/disp-1/verdict", json=body).status_code == 422
    assert drv.calls == []


def test_verdict_404_409_and_ok(client, monkeypatch):
    from app.personality_store import TendencyRetired

    main, c = client
    monkeypatch.setattr(main, "_checkpointer_cm", object())
    monkeypatch.setattr(main, "personality", Drv(result=None))
    assert c.post("/personality/tendencies/disp-x/verdict", json={"verdict": "yes"}).status_code == 404
    monkeypatch.setattr(main, "personality", Drv(exc=TendencyRetired("disp-x")))
    assert c.post("/personality/tendencies/disp-x/verdict", json={"verdict": "yes"}).status_code == 409
    ok = {"verdict_id": "jv-1", "disposition_id": "disp-x", "verdict": "no", "status": "retired", "retired": True,
          "at": "2030-01-01T00:00:00+00:00"}
    drv = Drv(result=ok)
    monkeypatch.setattr(main, "personality", drv)
    resp = c.post("/personality/tendencies/disp-x/verdict", json={"verdict": "no"})
    assert resp.status_code == 200 and resp.json() == ok and drv.calls == [("disp-x", "no")]


def test_verdict_503_names_the_migration_when_the_table_is_missing(client, monkeypatch):
    from psycopg.errors import UndefinedTable

    main, c = client
    monkeypatch.setattr(main, "_checkpointer_cm", object())
    monkeypatch.setattr(main, "personality", Drv(exc=UndefinedTable("personality_juniper_verdict")))
    resp = c.post("/personality/tendencies/disp-x/verdict", json={"verdict": "yes"})
    assert resp.status_code == 503 and "manual_migration_personality_juniper_verdict_v1.sql" in resp.json()["detail"]


def test_choice_view_orders_passed_over_by_market_rank_then_unranked():
    from datetime import datetime, timezone

    from app.personality_store import choice_view

    at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    row = {"choice_id": "c1", "lane": "curiosity", "occurred_at": at,
           "chosen": '{"key": "k", "text": "walkway misnaming", "market_rank": 4}',
           "alternatives": [{"key": "a", "text": "u1", "market_rank": None}, {"key": "b", "text": "r2", "market_rank": 2},
                            {"key": "c", "text": "r1", "market_rank": 1}, "junk"],
           "chosen_market_rank": 4, "declined_all": False, "care_promoted": False, "picks_in_decision": 2}
    v = choice_view(row)
    assert [a["text"] for a in v["passed_over"]] == ["r1", "r2", "u1"]
    assert v["chosen"] == {"key": "k", "text": "walkway misnaming", "market_rank": 4}
    assert v["occurred_at"] == at.isoformat() and "tendency" not in v
    dec = choice_view({**row, "chosen": None, "declined_all": True, "disposition_id": None,
                       "tendency_label": None, "tendency_status": None})
    assert dec["chosen"] is None and dec["declined_all"] is True and dec["tendency"] is None
