"""Orion's now on durable-runs (integration design 2026-11-15, Part A patch 1), no database.

* ``SalienceBook``: salience at insert from the kind's own 30-day history, loaded once, extended only
  after a commit, never re-scoring a stored row.
* The lifecycle, end to end through the real chronicle, the real routes and an in-memory store with the
  same contract as ``ExperienceStore``: an organ emits -> its experience is stored with salience -> it
  appears in another organ's now -> that reader cites it -> the cite is checked and stored -> the
  integration metric moves from 0. The same path on real Postgres is test_experience_now_postgres.py.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(TESTS)]

from app.temporal_self_chronicle import Chronicler, SalienceBook  # noqa: E402
from app.temporal_self_store import WindowWrite, event_row  # noqa: E402
from orion.integration.organs import load_organs  # noqa: E402
from orion.temporal_self import sources as S  # noqa: E402
from temporal_self_world import FakeReader, FakeStore, World, utc  # noqa: E402
from test_temporal_self_chronicle import STEP, cfg  # noqa: E402

T0 = utc(2026, 11, 15, 15, 0)


def gpu(i, at, waited):
    return {"event_id": f"g{i}", "generated_at": at, "event": "granted", "priority": "background",
            "holder": "orion-mind", "work_class": "x", "waited_ms": waited}


def reading(i, at, surprise=0.001, note="stage1_completed"):
    return {"act_id": f"act_{i:024x}", "object_ref": f"https://example.org/{i}", "correlation_id": f"seed-{i}",
            "opened_at": at - timedelta(minutes=2), "closed_at": at, "outcome_value": 1, "outcome_note": note,
            "surprise_nats": surprise, "seed_title": f"Article {i}"}


def prime(i, at, note="woke: quiet | no move"):
    return {"act_id": f"act_{0xfeed0000 + i:024x}", "object_ref": f"prime_thought:{i:016x}", "correlation_id": None,
            "opened_at": at - timedelta(seconds=14), "closed_at": at, "outcome_value": 1, "outcome_note": note,
            "surprise_nats": 0.0004}


# --------------------------------------------------------------------------- SalienceBook


def test_salience_book_loads_history_once_scores_in_time_order_and_skips_stored_rows():
    loads = []
    stored = [{"event_id": f"gpu_wait:gh{i}", "occurred_at": T0 - timedelta(hours=i + 1), "verdict": None,
               "payload": {"waited_ms": 600.0 + i}, "surprise_nats": None} for i in range(25)]

    async def load(kind, since):
        loads.append((kind, since))
        return stored if kind == "gpu_wait" else []

    book = SalienceBook({"gpu_wait": "waited_ms"}, load)
    small = S.gpu_wait(gpu(1, T0, 550.0))
    big = S.gpu_wait(gpu(2, T0 + timedelta(seconds=5), 10_000.0))
    chat = S.chat_turn({"id": "c1", "session_id": "s", "created_at": T0, "has_prompt": True})
    again = S.gpu_wait({**gpu("h3", T0 - timedelta(hours=4), 603.0)})  # already stored: never re-scored
    scores, pending = asyncio.run(book.score([big, small, chat, again], T0))
    assert scores[small.event_id] == 0.0  # smaller than every past wait: the rest point
    # big sees the 25 stored waits and `small` (earlier in the same window): all strictly smaller.
    assert scores[big.event_id] == 1.0
    assert chat.event_id not in scores and again.event_id not in scores
    assert [k for k, _ in loads] == ["gpu_wait"]
    book.commit(pending, T0)
    asyncio.run(book.score([S.gpu_wait(gpu(3, T0 + timedelta(minutes=1), 900.0))], T0))
    assert len(loads) == 1  # history is extended in memory, not re-read


def test_salience_is_none_until_the_kind_has_twenty_past_events():
    async def load(kind, since):
        return []

    book = SalienceBook({"gpu_wait": "waited_ms"}, load)
    events = [S.gpu_wait(gpu(i, T0 + timedelta(minutes=i), 500.0 + i)) for i in range(21)]
    scores, _ = asyncio.run(book.score(events, T0))
    assert [scores[e.event_id] for e in events[:20]] == [None] * 20
    assert scores[events[20].event_id] == 1.0


def test_event_row_writes_salience_and_the_sources_surprisal():
    e = S.reading_turn(reading(1, T0, surprise=0.25))
    r = event_row(e, salience=0.4)
    assert r["salience"] == 0.4 and r["surprise_nats"] == 0.25
    assert event_row(S.chat_turn({"id": "c", "created_at": T0}))["surprise_nats"] is None


# --------------------------------------------------------------------------- the lifecycle


class MemDB(FakeStore):
    """FakeStore plus what the now needs: salience and ingest time per stored event, and the two tables."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self.world = world
        self.meta: dict[str, dict] = {}
        self.shown_rows: dict[str, dict] = {}
        self.use_rows: dict[str, dict] = {}

    async def commit_window(self, w: WindowWrite) -> None:
        await super().commit_window(w)
        for e in [*w.events, *w.late_events]:
            if e.event_id not in self.meta:
                self.meta[e.event_id] = dict(event_row(e, salience=w.salience.get(e.event_id)),
                                             ingested_at=self.world.now)

    async def magnitude_history(self, kind, since):
        return [{"event_id": m["event_id"], "occurred_at": m["occurred_at"], "verdict": m["verdict"],
                 "payload": __import__("json").loads(m["payload"]), "surprise_nats": m["surprise_nats"]}
                for m in self.meta.values() if m["source_kind"] == kind and m["occurred_at"] >= since]

    # ExperienceStore's contract
    async def last_shown(self, organ):
        ats = [r["shown_at"] for r in self.shown_rows.values() if r["reader_organ"] == organ]
        return max(ats) if ats else None

    async def candidates(self, *, since, as_of, own_kinds, per_kind):
        import json

        rows = [m for m in self.meta.values() if since < m["ingested_at"] <= as_of and m["source_kind"] not in own_kinds]
        out, by_kind = [], {}
        for m in rows:
            by_kind.setdefault(m["source_kind"], []).append(m)
        for kind, ms in by_kind.items():
            ms.sort(key=lambda m: (m["salience"] is None, -(m["salience"] or 0), -m["occurred_at"].timestamp(), m["event_id"]))
            for m in ms[:per_kind]:
                out.append({"event_id": m["event_id"], "source_kind": kind, "occurred_at": m["occurred_at"],
                            "label": m["label"], "verdict": m["verdict"], "related_refs": m["related_refs"],
                            "payload": json.loads(m["payload"]), "salience": m["salience"],
                            "surprise_nats": m["surprise_nats"]})
        return out, len(rows)

    async def insert_shown(self, **row):
        self.shown_rows[row["shown_id"]] = row

    async def shown(self, shown_id):
        return self.shown_rows.get(shown_id)

    async def insert_uses(self, rows):
        for r in rows:
            self.use_rows.setdefault(r["use_id"], r)

    async def rate_inputs(self, lo, hi):
        kinds: dict[str, int] = {}
        for m in self.meta.values():
            if lo <= m["occurred_at"] < hi:
                kinds[m["source_kind"]] = kinds.get(m["source_kind"], 0) + 1
        uses = sorted({(u["event_id"], self.meta[u["event_id"]]["source_kind"], u["citing_organ"])
                       for u in self.use_rows.values()
                       if lo <= self.meta[u["event_id"]]["occurred_at"] < hi
                       and u["cited_at"] < self.meta[u["event_id"]]["occurred_at"] + timedelta(hours=24)})
        made: dict[str, int] = {}
        for u in self.use_rows.values():
            if lo <= u["cited_at"] < hi:
                made[u["citing_organ"]] = made.get(u["citing_organ"], 0) + 1
        return kinds, uses, made


@pytest.fixture
def routes(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main, "_checkpointer_cm", object())
    return main, TestClient(main.app)  # no `with`: the lifespan (bus, pools) never starts


def test_lifecycle_emit_now_cite_store_metric(routes, monkeypatch):
    from datetime import datetime as real_datetime

    import app.temporal_self_store as store_mod

    main, client = routes
    w = World(T0)
    db = MemDB(w)
    monkeypatch.setattr(store_mod, "ExperienceStore", lambda _pool: db)
    # 25 past readings with small surprisal, then one that surprised Orion; Prime thinks meanwhile.
    for i in range(25):
        w.add("reading_turn", reading(i, T0 + timedelta(minutes=i), surprise=0.001 + i * 1e-5))
    w.add("reading_turn", reading(99, T0 + timedelta(minutes=30), surprise=3.0, note="no_read_evidence"))
    w.add("prime_thought", prime(1, T0 + timedelta(minutes=31)))
    rules = dict(load_organs().magnitude_rule)
    ch = Chronicler(store=db, reader=FakeReader(w), cfg=cfg(), now=lambda: w.now,
                    salience=SalienceBook(rules, db.magnitude_history))
    while w.now < T0 + timedelta(minutes=45):
        w.now += STEP
        assert asyncio.run(ch.step())["error"] is None

    # 1. the organ emitted: experiences stored with salience
    surprising = f"reading_turn:act_{99:024x}"
    assert db.meta[surprising]["salience"] == 1.0 and db.meta[surprising]["surprise_nats"] == 3.0
    assert sum(1 for m in db.meta.values() if m["source_kind"] == "reading_turn") == 26

    # 2. it appears in a different organ's now; never in its own organ's now
    class Clock:
        @staticmethod
        def now(tz=None):
            return w.now.astimezone(tz) if tz else w.now

    monkeypatch.setattr("datetime.datetime", type("DT", (real_datetime,), {"now": Clock.now}))
    since = (T0 - timedelta(hours=1)).isoformat()
    now = client.get("/temporal-self/now", params={"organ": "prime", "since": since, "run_ref": "prime_thought:x"}).json()
    ids = [i["event_id"] for i in now["items"]]
    assert ids[0] == surprising and now["items"][0]["salience"] == 1.0 and now["items"][0]["organ"] == "reading"
    assert all(i["organ"] != "prime" for i in now["items"]) and len([i for i in now["items"] if i["organ"] == "reading"]) <= 2
    own = client.get("/temporal-self/now", params={"organ": "reading", "since": since}).json()
    assert {i["organ"] for i in own["items"]} == {"prime"}
    assert db.shown_rows[now["shown_id"]]["event_ids"] == ids

    # 0. before any cite, the metric reads 0 (the rest point), and no organ is closed
    before = client.get("/temporal-self/integration", params={"hours": 24}).json()
    assert before["rate"] == 0.0 and before["closed_organs"] == []

    # 3. the reader cites it (plus one id it was never shown) -> 4. checked and stored
    res = client.post("/temporal-self/now/use", json={"shown_id": now["shown_id"], "citing_ref": "prime_thought:x",
                                                      "cites": [f"experience:{surprising}", "dream_cycle:never-shown"]}).json()
    assert [u["event_id"] for u in res["uses"]] == [surprising] and res["rejected_not_shown"] == ["dream_cycle:never-shown"]
    assert [u["citing_organ"] for u in db.use_rows.values()] == ["prime"]
    # re-sending the same cite stores nothing new
    client.post("/temporal-self/now/use", json={"shown_id": now["shown_id"], "citing_ref": "prime_thought:x",
                                                 "cites": [surprising]})
    assert len(db.use_rows) == 1

    # 5. the metric moved
    w.now += timedelta(seconds=1)
    after = client.get("/temporal-self/integration", params={"hours": 24}).json()
    assert after["used_by_other"] == 1 and after["rate"] == pytest.approx(1 / after["experiences"])
    assert after["closed_organs"] == ["reading"]
    assert {o["organ"]: o["cites_made"] for o in after["organs"]}["prime"] == 1


def test_routes_refuse_unknown_organs_unknown_shown_rows_and_obey_the_flag(routes, monkeypatch):
    import app.temporal_self_store as store_mod

    main, client = routes
    db = MemDB(World(T0))
    monkeypatch.setattr(store_mod, "ExperienceStore", lambda _pool: db)
    assert client.get("/temporal-self/now", params={"organ": "nobody"}).status_code == 422
    assert client.get("/temporal-self/now", params={"organ": "prime", "since": "2026-11-15T00:00:00"}).status_code == 422
    r = client.post("/temporal-self/now/use", json={"shown_id": "nope", "citing_ref": "x", "cites": ["a:b"]})
    assert r.status_code == 404 and db.use_rows == {}
    assert client.post("/temporal-self/now/use", json={"shown_id": "s", "cites": []}).status_code == 422
    monkeypatch.setattr(main._settings, "temporal_self_now_enabled", False)
    assert client.get("/temporal-self/now", params={"organ": "prime"}).status_code == 404
    assert client.post("/temporal-self/now/use", json={"shown_id": "s", "citing_ref": "x"}).status_code == 404


def test_flag_ships_on_in_settings_env_example_and_compose():
    text = (SERVICE_ROOT / ".env_example").read_text()
    assert "TEMPORAL_SELF_NOW_ENABLED=true" in text
    assert "TEMPORAL_SELF_NOW_ENABLED=${TEMPORAL_SELF_NOW_ENABLED:-true}" in (SERVICE_ROOT / "docker-compose.yml").read_text()
    from app.settings import Settings

    assert Settings.model_fields["temporal_self_now_enabled"].default is True


def test_dockerfile_copies_the_organ_registry():
    assert "COPY config/integration/organs.yaml /app/config/integration/organs.yaml" in (SERVICE_ROOT / "Dockerfile").read_text()
