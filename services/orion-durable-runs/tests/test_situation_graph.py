"""situation.update (spec 2026-10-07-situation-graph-design.md, step 2, shadow).

Rows here are synthetic; the trip mirrors the live 10-05 incident: "in Chicago until Wednesday"
was a `happened` memory with a place referent in the location role and, before the v5 writer
fix, no end date.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.situation_driver import (  # noqa: E402
    SituationDriver,
    coalesce,
    event_from_chat_turn,
    event_from_run_state,
    thread_for,
)
from app.situation_graph import (  # noqa: E402
    SituationDeps,
    cues_for,
    facts_from_rows,
    lapsed_from,
    rank_primed,
)
from orion.schemas.situation_state import SITUATION_WORKFLOW, SituationStateV1  # noqa: E402

T0 = datetime(2026, 10, 5, 3, 0, tzinfo=timezone.utc)
TTL = timedelta(hours=48)


def trip(**kw):
    row = {"memory_id": "m-trip", "purpose": "happened", "occurred_at": T0, "created_at": T0 + timedelta(hours=1),
           "expires_at": None, "voice": "juniper_said", "confirmation_state": "auto",
           "statement": "Juniper is in Chicago for a team meeting and will be there until Wednesday.",
           "referents": [{"key": "person:juniper", "role": "subject"}, {"key": "place:chicago", "role": "location"}]}
    row.update(kw)
    return row


def follow_up(**kw):
    row = {"memory_id": "m-fu", "purpose": "follow_up", "occurred_at": T0, "created_at": T0,
           "expires_at": T0 + timedelta(days=5), "voice": "orion_thought", "confirmation_state": "auto",
           "statement": "I want to ask Juniper how the Chicago training simulation went.",
           "referents": [{"key": "event:training-simulation-2026-10", "role": "about"}]}
    row.update(kw)
    return row


# --- pure reducers ------------------------------------------------------------------------------


def test_trip_without_end_date_is_only_recent_never_whereabouts():
    """Live check 2026-10-07: "sang karaoke at the Jackalope bar during the Austin trip" (told on
    10-05, happened a week earlier) became whereabouts under a newest-place rule. Without
    Juniper's own end date a place memory is recent, not where she is."""
    f = facts_from_rows([trip()], T0 + timedelta(hours=44), TTL)
    assert f["whereabouts"] is None and f["doing"] == []
    assert [(x["memory_id"], x["until_source"]) for x in f["recent"]] == [("m-trip", "default_ttl")]
    assert facts_from_rows([trip()], T0 + timedelta(hours=49), TTL)["recent"] == []


def test_trip_with_juniper_end_date_holds_until_it_and_no_longer():
    end = T0 + timedelta(days=3, hours=20)
    row = trip(expires_at=end)
    f = facts_from_rows([row], T0 + timedelta(days=3), TTL)
    assert f["whereabouts"]["until_source"] == "juniper_words"
    assert facts_from_rows([row], end + timedelta(minutes=1), TTL)["whereabouts"] is None


def test_follow_up_is_waiting_on_and_long_term_facts_are_not_situation():
    about = trip(memory_id="m-home", purpose="about_juniper", statement="Juniper told me we live in Ogden, Utah.",
                 referents=[{"key": "place:ogden", "role": "location"}])
    f = facts_from_rows([follow_up(), about], T0 + timedelta(hours=2), TTL)
    assert [x["memory_id"] for x in f["waiting_on"]] == ["m-fu"]
    assert f["whereabouts"] is None and f["doing"] == []


def test_newest_place_wins_whereabouts_and_older_place_stays_doing():
    end = T0 + timedelta(days=3)
    hotel = trip(memory_id="m-hotel", occurred_at=T0 + timedelta(hours=1), expires_at=end,
                 statement="Juniper is staying at the Wade, a hotel on the lake in Chicago.",
                 referents=[{"key": "place:the-wade", "role": "location"}])
    f = facts_from_rows([trip(expires_at=end), hotel], T0 + timedelta(hours=3), TTL)
    assert f["whereabouts"]["memory_id"] == "m-hotel"
    assert [d["memory_id"] for d in f["doing"]] == ["m-trip"]
    assert f["doing"][0]["slot"] == "doing"


def test_lapsed_records_what_stopped_being_current():
    before = facts_from_rows([trip()], T0 + timedelta(hours=1), TTL)
    after = facts_from_rows([trip()], T0 + timedelta(hours=50), TTL)
    lapsed = lapsed_from(before, after, [], T0 + timedelta(hours=50))
    assert [x["memory_id"] for x in lapsed] == ["m-trip"]
    assert lapsed_from(after, after, lapsed, T0 + timedelta(hours=51)) == lapsed  # kept, not doubled


def test_cues_come_from_facts_and_names_in_the_turn_never_participants():
    facts = facts_from_rows([trip()], T0 + timedelta(hours=1), TTL)
    cues = cues_for(facts, "just chilling, thinking about Hecate", ["project:hecate", "person:juniper", "place:ogden"])
    assert cues == ["place:chicago", "project:hecate"]


def test_rank_primed_orders_by_decayed_strength_and_names_the_cue():
    now = T0
    rows = [
        {"memory_id": "old", "statement": "Juniper and Vincent sang karaoke in Austin.", "strength": 0.8,
         "half_life_days": 14.0, "last_reinforced_at": now - timedelta(days=28), "referent_keys": ["place:chicago"]},
        {"memory_id": "new", "statement": "Juniper sent me photos of the Chicago skyline.", "strength": 0.8,
         "half_life_days": 14.0, "last_reinforced_at": now - timedelta(days=1), "referent_keys": ["place:chicago"]},
        {"memory_id": "off", "statement": "Unrelated memory with no shared referent.", "strength": 1.0,
         "half_life_days": None, "last_reinforced_at": now, "referent_keys": ["place:ogden"]},
    ]
    out = rank_primed(rows, ["place:chicago"], now)
    assert [p["memory_id"] for p in out] == ["new", "old"]
    assert out[0]["why"] == "place:chicago" and out[0]["score"] > out[1]["score"]


# --- the graph on a real checkpointer -----------------------------------------------------------


END = T0 + timedelta(days=3, hours=20)   # "till Wednesday", grounded by her own words (writer v5)


def stated_trip(**kw):
    return trip(expires_at=END, **kw)


class World:
    def __init__(self, rows):
        self.rows = rows
        self.now = T0 + timedelta(hours=2)
        self.projected: list[SituationStateV1] = []
        self.prime_calls: list[list] = []
        self.prime_rows = [{"memory_id": "m-skyline", "statement": "Juniper sent me photos of the Chicago skyline.",
                            "strength": 0.8, "half_life_days": 14.0, "last_reinforced_at": T0,
                            "referent_keys": ["place:chicago"]}]
        self.prime_delay = 0.0
        self.states = []

    async def load_facts(self, now):
        return {"rows": list(self.rows), "known_keys": ["place:chicago", "project:hecate", "person:juniper"]}

    async def prime(self, cues, exclude, limit):
        self.prime_calls.append(list(cues))
        if self.prime_delay:
            await asyncio.sleep(self.prime_delay)
        return [r for r in self.prime_rows if r["memory_id"] not in exclude]

    async def project(self, model):
        self.projected.append(model)

    async def publish_state(self, row):
        self.states.append(row)

    def deps(self):
        return SituationDeps(load_facts=self.load_facts, prime=self.prime, project=self.project,
                             now=lambda: self.now, default_ttl=TTL, prime_timeout_sec=0.05)


def driver(world, saver=None, retention_days=2):
    return SituationDriver(checkpointer=saver or InMemorySaver(), deps=world.deps(),
                           publish_state=world.publish_state, retention_days=retention_days)


def ev(kind, n, text=""):
    return {"event_id": f"{kind}:{n}", "kind": kind, "correlation_id": f"c-{n}", "text": text}


def test_first_step_projects_revision_one_with_whereabouts_and_primed_memory():
    async def run():
        w = World([stated_trip()])
        d = driver(w)
        out = await d.step(ev("boot", 1))
        (model,) = w.projected
        assert model.revision == 1 and model.thread_id == thread_for(w.now)
        assert model.juniper.whereabouts.memory_id == "m-trip"
        assert model.recall.cues == ["place:chicago"]
        assert [p.memory_id for p in model.recall.primed] == ["m-skyline"]
        assert model.recall.primed_revision == 1
        assert out["workflow"] == SITUATION_WORKFLOW
        (row,) = w.states
        assert (row.workflow, row.status, row.detail["revision"], row.detail["changed"]) == (SITUATION_WORKFLOW, "completed", 1, True)

    asyncio.run(run())


def test_unchanged_situation_does_not_bump_revision_or_reproject_or_reprime():
    async def run():
        w = World([stated_trip()])
        d = driver(w)
        await d.step(ev("boot", 1))
        await d.step(ev("tick", 2))
        assert len(w.projected) == 1 and d.last_revision == 1
        assert len(w.prime_calls) == 1          # cues unchanged and fresh: no second search
        assert len(w.states) == 1               # a quiet step leaves no run-state row

    asyncio.run(run())


def test_same_event_twice_is_skipped():
    async def run():
        w = World([stated_trip()])
        d = driver(w)
        await d.step(ev("chat_turn", 1, "hi"))
        out = await d.step(ev("chat_turn", 1, "hi"))
        assert out["skipped"] is True and len(w.projected) == 1

    asyncio.run(run())


def test_turn_naming_a_new_referent_reprimes_and_bumps_revision():
    async def run():
        w = World([stated_trip()])
        d = driver(w)
        await d.step(ev("boot", 1))
        w.prime_rows.append({"memory_id": "m-hecate", "statement": "Juniper named the new GPU server Hecate.",
                             "strength": 0.9, "half_life_days": 14.0, "last_reinforced_at": T0,
                             "referent_keys": ["project:hecate"]})
        await d.step(ev("chat_turn", 2, "flashing Hecate again tonight"))
        model = w.projected[-1]
        assert model.revision == 2 and model.recall.cues == ["place:chicago", "project:hecate"]
        assert "m-hecate" in [p.memory_id for p in model.recall.primed]
        assert w.prime_calls[-1] == ["place:chicago", "project:hecate"]

    asyncio.run(run())


def test_trip_ending_lapses_and_is_recorded():
    async def run():
        w = World([stated_trip()])
        d = driver(w, retention_days=7)          # the trip spans more days than the default keeps
        await d.step(ev("boot", 1))
        w.now = END + timedelta(hours=1)         # trip over, days later: seeded across gap days
        await d.step(ev("tick", 2))
        model = w.projected[-1]
        assert model.juniper.whereabouts is None
        assert [x.memory_id for x in model.lapsed] == ["m-trip"]
        assert model.revision == 2               # carried across the day boundary by the seed

    asyncio.run(run())


def test_slow_priming_keeps_last_set_and_still_projects_the_facts():
    async def run():
        w = World([stated_trip()])
        d = driver(w)
        await d.step(ev("boot", 1))
        w.prime_delay = 1.0
        w.rows = [stated_trip(), follow_up()]
        out = await d.step(ev("chat_turn", 2, "about Hecate"))
        model = w.projected[-1]
        assert out["prime_error"] == "TimeoutError"
        assert w.states[-1].detail["prime_error"] == "TimeoutError"
        assert [p.memory_id for p in model.recall.primed] == ["m-skyline"]   # kept
        assert model.recall.primed_revision == 1 < model.revision            # visibly behind
        assert [f.memory_id for f in model.juniper.waiting_on] == ["m-fu"]   # facts still moved

    asyncio.run(run())


def test_new_day_seeds_from_yesterday_and_deletes_threads_past_retention():
    async def run():
        saver = InMemorySaver()
        w = World([stated_trip()])
        d = driver(w, saver)
        ancient = thread_for(w.now - timedelta(days=4))
        await d._graph.ainvoke({"event": ev("boot", 0), "thread_id": ancient}, {"configurable": {"thread_id": ancient}})
        await d.step(ev("boot", 1))
        w.now = w.now + timedelta(days=1)
        await d.step(ev("tick", 2))
        today = await d._graph.aget_state({"configurable": {"thread_id": thread_for(w.now)}})
        assert today.values["situation"]["revision"] >= 1
        gone = await d._graph.aget_state({"configurable": {"thread_id": ancient}})
        assert not gone.values

    asyncio.run(run())


# --- driver intake -------------------------------------------------------------------------------


def test_only_finished_episode_distills_are_situation_events():
    assert event_from_run_state({"workflow": "memory.episode_distill", "status": "completed", "run_id": "r1",
                                 "entry_id": "e1", "correlation_id": "c"})["kind"] == "episode_distilled"
    assert event_from_run_state({"workflow": "memory.episode_distill", "status": "running", "run_id": "r1"}) is None
    assert event_from_run_state({"workflow": SITUATION_WORKFLOW, "status": "completed", "run_id": "s"}) is None


def test_chat_turn_event_carries_the_prompt():
    e = event_from_chat_turn({"correlation_id": "c-9", "source": "hub_ws", "prompt": "in Chicago till Wed", "response": "ok"})
    assert e == {"event_id": "chat:c-9", "kind": "chat_turn", "correlation_id": "c-9", "text": "in Chicago till Wed"}
    assert event_from_chat_turn({"nope": 1}) is None


def test_coalesce_names_the_burst_by_its_most_significant_event_and_keeps_the_newest_text():
    merged = coalesce([ev("tick", 1), ev("chat_turn", 2, "first"), ev("episode_distilled", 3), ev("chat_turn", 4, "newest")])
    assert merged["kind"] == "chat_turn" and merged["text"] == "newest" and merged["coalesced"] == 4


def test_resume_sweep_skips_situation_threads_quietly(caplog, monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test@localhost/test")

    async def run():
        import app.settings as settings_mod

        settings_mod._settings = None
        from app.runner import DurableRunner

        saver = InMemorySaver()
        w = World([stated_trip()])
        await driver(w, saver).step(ev("boot", 1))
        runner = DurableRunner(settings_mod.get_settings(), bus=None, checkpointer=saver)
        with caplog.at_level("WARNING"):
            assert await runner.unfinished_threads() == []
        assert "durable_run_resume_unknown_workflow" not in caplog.text

    asyncio.run(run())
