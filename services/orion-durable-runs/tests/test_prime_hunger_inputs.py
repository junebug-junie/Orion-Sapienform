"""GET /temporal-self/hunger (integration design Part B, Prime hunger) and the now's ``include`` pins."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

T = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self):
        self.calls: list[dict] = []
        self.shown: dict[str, dict] = {}

    async def hunger_inputs(self, **kw):
        self.calls.append(kw)
        surprises = [
            {"event_id": "prediction_surprise:a", "source_kind": "prediction_surprise", "occurred_at": T,
             "label": "cabinet forecast crossed", "verdict": None, "payload_json": {}, "surprise_nats": 4.2,
             "p90": 0.0123, "n": 138},
            {"event_id": "mystery:b", "source_kind": "not_a_registered_kind", "occurred_at": T, "label": None,
             "verdict": None, "payload_json": {}, "surprise_nats": 9.0, "p90": 0.1, "n": 50},
        ]
        msgs = [{"id": f"c{i}", "at": T - timedelta(hours=24 - i), "prompt": f"  hello\n{i}  "} for i in range(3)]
        return surprises, msgs, "chat_turn:c2"

    async def last_shown(self, organ):
        return None

    async def candidates(self, *, since, as_of, own_kinds, per_kind):
        return [{"event_id": "reading_turn:r", "source_kind": "reading_turn", "occurred_at": T, "label": "read",
                 "verdict": None, "related_refs": [], "payload": {}, "salience": 0.5, "surprise_nats": None}], 1

    async def by_id(self, event_ids, *, own_kinds):
        rows = {"chat_turn:c2": {"event_id": "chat_turn:c2", "source_kind": "chat_turn", "occurred_at": T,
                                 "label": None, "verdict": None, "related_refs": [], "payload": {"speaker": "juniper"},
                                 "salience": None, "surprise_nats": None},
                "prime_thought:own": {"event_id": "prime_thought:own", "source_kind": "prime_thought",
                                      "occurred_at": T, "label": None, "verdict": None, "related_refs": [],
                                      "payload": {}, "salience": None, "surprise_nats": None}}
        return [rows[i] for i in event_ids if i in rows and rows[i]["source_kind"] not in own_kinds]

    async def insert_shown(self, **row):
        self.shown[row["shown_id"]] = row


@pytest.fixture
def routes(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from fastapi.testclient import TestClient

    import app.temporal_self_store as store_mod
    from app import main

    monkeypatch.setattr(main, "_checkpointer_cm", object())
    db = FakeStore()
    monkeypatch.setattr(store_mod, "ExperienceStore", lambda _pool: db)
    return main, TestClient(main.app), db


def test_hunger_inputs_are_a_registered_contract_and_read_only(routes):
    from orion.schemas.registry import resolve
    from orion.schemas.temporal_self import TemporalHungerInputsV1

    main, client, db = routes
    r = client.get("/temporal-self/hunger", params={"organ": "prime"})
    assert r.status_code == 200
    body = TemporalHungerInputsV1.model_validate(r.json())
    assert resolve("TemporalHungerInputsV1") is TemporalHungerInputsV1
    # An unregistered kind is never passed on; the organ comes from the registry.
    assert [(s.event_id, s.organ, s.source_p90, s.source_n) for s in body.surprises] == [
        ("prediction_surprise:a", "prediction", 0.0123, 138)]
    assert body.juniper.last_event_id == "chat_turn:c2" and body.juniper.last_chat_id == "c2"
    assert body.juniper.last_prompt == "hello 2" and len(body.juniper.message_times) == 3
    # The reader's own kinds are excluded, the windows are 24 h (surprise) and 30 days (Juniper), and nothing was shown.
    kw = db.calls[0]
    assert "prime_thought" in kw["own_kinds"]
    assert kw["as_of"] - kw["lo"] == timedelta(hours=24) and kw["as_of"] - kw["juniper_lo"] == timedelta(days=30)
    assert db.shown == {}


def test_hunger_refuses_an_unknown_organ_and_obeys_the_now_flag(routes, monkeypatch):
    main, client, _ = routes
    assert client.get("/temporal-self/hunger", params={"organ": "nobody"}).status_code == 422
    monkeypatch.setattr(main._settings, "temporal_self_now_enabled", False)
    assert client.get("/temporal-self/hunger", params={"organ": "prime"}).status_code == 404


def test_now_pins_included_rows_first_and_stores_them_as_shown(routes):
    main, client, db = routes
    r = client.get("/temporal-self/now", params={"organ": "prime", "since": "2026-11-16T00:00:00+00:00",
                                                 "include": "chat_turn:c2,prime_thought:own,unknown:z"}).json()
    ids = [i["event_id"] for i in r["items"]]
    assert ids == ["chat_turn:c2", "reading_turn:r"]  # pinned first; own organ and unknown ids dropped
    assert db.shown[r["shown_id"]]["event_ids"] == ids  # so a cite of the pinned row is valid
