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
        ats = [r["shown_at"] for r in self.shown_rows.values()
               if r["reader_organ"] == organ and r.get("advances_cursor", True)]
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

    async def held_candidates(self, *, keys, lo, as_of, own_kinds, limit=50):
        import json

        ks = set(keys)
        rows = [m for m in self.meta.values() if lo < m["ingested_at"] <= as_of and m["source_kind"] not in own_kinds
                and ({m["event_id"], m["source_ref"], m["correlation_id"], m["subject_ref"]} | set(m["related_refs"])) & ks]
        rows.sort(key=lambda m: (-m["occurred_at"].timestamp(), m["event_id"]))
        return [{"event_id": m["event_id"], "source_kind": m["source_kind"], "occurred_at": m["occurred_at"],
                 "label": m["label"], "verdict": m["verdict"], "related_refs": m["related_refs"],
                 "payload": json.loads(m["payload"]), "salience": m["salience"], "surprise_nats": m["surprise_nats"],
                 "source_ref": m["source_ref"], "correlation_id": m["correlation_id"], "subject_ref": m["subject_ref"]}
                for m in rows[:limit]]

    async def arc_subjects(self, arc_ids):
        return {a: s for a, s in getattr(self, "arcs", {}).items() if a in set(arc_ids)}

    async def insert_shown(self, **row):
        self.shown_rows[row["shown_id"]] = row

    async def record_cites(self, shown_id, decide):
        shown = self.shown_rows.get(shown_id)
        if shown is None:
            return None
        result = decide(shown, [u["event_id"] for u in self.use_rows.values() if u["shown_id"] == shown_id])
        for u in result.uses:
            self.use_rows.setdefault(u.use_id, u.model_dump())
        return result

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
    # 30 readings with small, FALLING surprisal (so the newest is never the biggest) and one that surprised
    # Orion in the middle; Prime thinks after. The rule is set here (the mechanism under test); the live
    # registry gives reading_turn no magnitude (its ledger surprisal drifts with age).
    for i in range(30):
        w.add("reading_turn", reading(i, T0 + timedelta(minutes=i), surprise=0.01 - i * 1e-4))
    w.add("reading_turn", reading(99, T0 + timedelta(minutes=22, seconds=30), surprise=3.0, note="no_read_evidence"))
    w.add("prime_thought", prime(1, T0 + timedelta(minutes=31)))
    rules = {**load_organs().magnitude_rule, "reading_turn": "surprise_nats"}
    ch = Chronicler(store=db, reader=FakeReader(w), cfg=cfg(), now=lambda: w.now,
                    salience=SalienceBook(rules, db.magnitude_history))
    while w.now < T0 + timedelta(minutes=45):
        w.now += STEP
        assert asyncio.run(ch.step())["error"] is None

    # 1. the organ emitted: experiences stored with salience
    surprising = f"reading_turn:act_{99:024x}"
    assert db.meta[surprising]["salience"] == 1.0 and db.meta[surprising]["surprise_nats"] == 3.0
    assert sum(1 for m in db.meta.values() if m["source_kind"] == "reading_turn") == 31
    assert db.meta[f"reading_turn:act_{29:024x}"]["salience"] == 0.0  # newest, smallest: the rest point

    # 2. it appears in a different organ's now; never in its own organ's now
    class Clock:
        @staticmethod
        def now(tz=None):
            return w.now.astimezone(tz) if tz else w.now

    monkeypatch.setattr("datetime.datetime", type("DT", (real_datetime,), {"now": Clock.now}))
    w.now += timedelta(minutes=5)  # past the read's commit margin
    since = (T0 - timedelta(hours=1)).isoformat()
    now = client.get("/temporal-self/now", params={"organ": "prime", "since": since, "run_ref": "prime_thought:x"}).json()
    ids = [i["event_id"] for i in now["items"]]
    assert ids[0] == surprising and now["items"][0]["salience"] == 1.0 and now["items"][0]["organ"] == "reading"
    assert all(i["organ"] != "prime" for i in now["items"]) and len([i for i in now["items"] if i["organ"] == "reading"]) <= 2
    own = client.get("/temporal-self/now", params={"organ": "reading", "since": since}).json()
    assert {i["organ"] for i in own["items"]} == {"prime"}
    assert db.shown_rows[now["shown_id"]]["event_ids"] == ids
    assert db.shown_rows[now["shown_id"]]["advances_cursor"] is False  # an explicit since= is an inspection read

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


def test_cite_cap_is_per_shown_row_across_calls(routes, monkeypatch):
    import app.temporal_self_store as store_mod
    from datetime import datetime, timezone

    main, client = routes
    db = MemDB(World(T0))
    monkeypatch.setattr(store_mod, "ExperienceStore", lambda _pool: db)
    ids = [f"gpu_wait:{i}" for i in range(6)]
    db.shown_rows["s"] = {"shown_id": "s", "reader_organ": "prime", "event_ids": ids,
                          "shown_at": datetime.now(timezone.utc), "advances_cursor": True}
    first = client.post("/temporal-self/now/use", json={"shown_id": "s", "citing_ref": "r", "cites": ids[:2]}).json()
    second = client.post("/temporal-self/now/use", json={"shown_id": "s", "citing_ref": "r", "cites": ids[:1] + ids[2:5]}).json()
    assert len(first["uses"]) == 2 and [u["event_id"] for u in second["uses"]] == [ids[2]]
    assert second["rejected_over_cap"] == ids[3:5] and len(db.use_rows) == 3


def test_a_read_skips_nothing_still_committing_and_inspection_reads_keep_the_cursor(routes, monkeypatch):
    """ingested_at is the commit's START time: a row stamped before a read but committed after it must be
    shown at the next default read (the read cuts NOW_COMMIT_MARGIN_SEC below the clock and the cursor
    starts there)."""
    from datetime import datetime as real_datetime

    import app.temporal_self_store as store_mod

    main, client = routes
    w = World(T0)
    db = MemDB(w)
    monkeypatch.setattr(store_mod, "ExperienceStore", lambda _pool: db)

    class Clock:
        @staticmethod
        def now(tz=None):
            return w.now.astimezone(tz) if tz else w.now

    monkeypatch.setattr("datetime.datetime", type("DT", (real_datetime,), {"now": Clock.now}))

    def land(i, ingested):
        e = S.gpu_wait(gpu(i, T0, 900.0))
        db.meta[e.event_id] = dict(event_row(e), ingested_at=ingested)

    w.now = T0 + timedelta(hours=1)
    first = client.get("/temporal-self/now", params={"organ": "prime"}).json()
    # a window whose commit started 30 s before that read and committed after it
    land(1, w.now - timedelta(seconds=30))
    client.get("/temporal-self/now", params={"organ": "prime", "since": (T0 - timedelta(days=1)).isoformat()})
    w.now += timedelta(minutes=10)
    nxt = client.get("/temporal-self/now", params={"organ": "prime"}).json()
    assert first["items"] == [] and [i["event_id"] for i in nxt["items"]] == ["gpu_wait:g1"]
    assert nxt["since"] == first["as_of"]  # the explicit-since read did not move the cursor


def test_a_salience_history_failure_stores_null_and_never_stalls_the_chronicle():
    w = World(T0)
    db = MemDB(w)
    calls = []

    async def broken(kind, since):
        calls.append(kind)
        raise TimeoutError("statement timeout")

    w.add("gpu_wait", gpu(1, T0 + timedelta(minutes=1), 900.0))
    ch = Chronicler(store=db, reader=FakeReader(w), cfg=cfg(), now=lambda: w.now,
                    salience=SalienceBook({"gpu_wait": "waited_ms"}, broken))
    while w.now < T0 + timedelta(minutes=20):
        w.now += STEP
        assert asyncio.run(ch.step())["error"] is None
    assert db.meta["gpu_wait:g1"]["salience"] is None and len(calls) >= 1


def test_held_refs_bring_a_concerns_reading_from_before_the_window_first_and_mark_it(routes, monkeypatch):
    """#3783: Prime held a concern about one reading (its seed's event_ref). The reading landed before this read's
    window and has no salience; 12 scored gpu waits landed inside it. With ``held=`` the reading comes first and
    names the ref; without it the body is the old shape (no ``held_refs`` key anywhere)."""
    from datetime import datetime as real_datetime

    import app.temporal_self_store as store_mod

    main, client = routes
    w = World(T0)
    db = MemDB(w)
    monkeypatch.setattr(store_mod, "ExperienceStore", lambda _pool: db)

    class Clock:
        @staticmethod
        def now(tz=None):
            return w.now.astimezone(tz) if tz else w.now

    monkeypatch.setattr("datetime.datetime", type("DT", (real_datetime,), {"now": Clock.now}))
    seed_uuid = "50753c3f-d556-5a6f-bfa7-b02b95254905"
    r = S.reading_turn(dict(reading(7, T0), correlation_id=seed_uuid))
    db.meta[r.event_id] = dict(event_row(r), ingested_at=T0)
    for i in range(12):
        g = S.gpu_wait(gpu(i, T0 + timedelta(hours=5), 900.0 + i))
        db.meta[g.event_id] = dict(event_row(g, salience=0.5 + i / 30), ingested_at=T0 + timedelta(hours=5))
    w.now = T0 + timedelta(hours=6)
    since = (T0 + timedelta(hours=4)).isoformat()
    held = f"world_pulse_read_seed:reading:{seed_uuid}"
    got = client.get("/temporal-self/now", params=[("organ", "prime"), ("since", since), ("held", held),
                                                    ("held", "prime_concern:520a5a9f8d084495")]).json()
    assert got["items"][0]["event_id"] == r.event_id and got["items"][0]["held_refs"] == [held]
    assert got["items"][0]["salience"] is None
    assert db.shown_rows[got["shown_id"]]["event_ids"][0] == r.event_id  # so a cite of it is valid
    plain = client.get("/temporal-self/now", params={"organ": "prime", "since": since}).json()
    assert r.event_id not in [i["event_id"] for i in plain["items"]]  # outside the window and not held
    assert all("held_refs" not in i for i in plain["items"])
