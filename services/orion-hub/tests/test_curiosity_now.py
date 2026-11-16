"""Curiosity reads Orion's now at kickoff and records its cites at the turn's end (integration patch 3).
Fakes only: the durable-runs endpoints are an httpx MockTransport, Postgres is a dict."""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from orion.core.bus.bus_schemas import ServiceRef
from orion.curiosity.journal import INVESTIGATION_TAG
from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.now_section import HEADER, extract_cites, format_now_section, parse_now
from orion.curiosity.self_inquiry import SELF_INQUIRY_TAG
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.worldview import RecentRun, WorldviewSnapshot
from scripts.curiosity_investigation import CuriosityInvestigation
from scripts.curiosity_now import LAST_RUN_SQL, LOAD_SQL, REMEMBER_SQL, CuriosityNow

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
LAST = NOW - timedelta(hours=3)
SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")
VIEW = WorldviewSnapshot(recent_runs=[RecentRun(run_id="r1")])


def _now_body(n=2, shown_id="shown1"):
    items = [
        {"event_id": "visual_run:p1", "organ": "painting", "source_kind": "visual_run",
         "label": "visual run: a lighthouse at dusk", "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
         "salience": 0.93, "surprise_nats": None, "related_refs": []},
        {"event_id": "prediction_surprise:x:7", "organ": "prediction", "source_kind": "prediction_surprise",
         "label": "prediction surprise: cabinet heat", "occurred_at": (NOW - timedelta(hours=2)).isoformat(),
         "salience": None, "surprise_nats": 3.1, "related_refs": []},
    ][:n]
    return {"shown_id": shown_id, "reader_organ": "curiosity", "reader_run_ref": "run1",
            "since": LAST.isoformat(), "as_of": NOW.isoformat(), "items": items, "candidates": n}


# --- the pure section and the cite parser ---------------------------------------------------------


def test_section_lists_each_item_with_its_cite_tag_and_salience():
    lines = format_now_section(parse_now(_now_body()))
    text = "\n".join(lines)
    assert text.startswith(HEADER)
    assert "[experience:visual_run:p1] painting" in text and "(salience 0.93)" in text
    assert "[experience:prediction_surprise:x:7] prediction" in text and "(salience unknown)" in text
    assert "At most 3 count" in text


def test_no_now_or_an_empty_now_renders_no_section():
    assert format_now_section(None) == []
    assert format_now_section(parse_now(_now_body(n=0))) == []
    assert parse_now({"not": "a now"}) is None and parse_now("x") is None


def test_cites_are_distinct_in_order_and_capped_at_three():
    text = ("I followed [experience:a] and [experience:b:1-2]; again [experience:a]. "
            "Then [ experience:c ] and [experience:d]. Not a cite: [arc:z] experience:e")
    assert extract_cites(text) == ["a", "b:1-2", "c"]
    assert extract_cites(text, cap=0) == [] and extract_cites(None) == []


def test_kickoff_and_self_inquiry_prompts_show_the_now_and_omit_it_when_absent():
    now = parse_now(_now_body())
    material = StudyMaterial(generated_at=NOW)
    with_now = build_kickoff_prompt(material, view=VIEW, run_id="r2", now=now)
    assert HEADER in with_now and "[experience:visual_run:p1]" in with_now
    assert build_kickoff_prompt(material, view=VIEW, run_id="r2", now=None) == build_kickoff_prompt(
        material, view=VIEW, run_id="r2")
    assert HEADER in build_kickoff_prompt(material, view=VIEW, run_id="r2", graph_enabled=False, now=now)
    si = build_self_inquiry_prompt(view=VIEW, run_id="r2", now=now)
    assert HEADER in si and HEADER not in build_self_inquiry_prompt(view=VIEW, run_id="r2")


# --- the reader: Postgres fake + MockTransport -----------------------------------------------------


class _Conn:
    def __init__(self, db):
        self.db = db

    async def fetchval(self, sql, *args):
        assert sql == LAST_RUN_SQL
        return self.db.get("last")

    async def execute(self, sql, run_id, shown_id):
        assert sql == REMEMBER_SQL
        if run_id not in self.db["rows"]:
            return "UPDATE 0"
        self.db["rows"][run_id] = shown_id
        return "UPDATE 1"

    async def fetchrow(self, sql, run_id):
        assert sql == LOAD_SQL
        return {"shown_id": self.db["rows"][run_id]} if run_id in self.db["rows"] else None


class _Pool:
    def __init__(self, db):
        self.db = db

    def acquire(self):
        conn = _Conn(self.db)

        class _CM:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _CM()


def _reader(db, handler, enabled=True):
    return CuriosityNow(enabled=enabled, durable_runs_url="http://durable.test/", pool_provider=lambda: _Pool(db),
                        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_kickoff_read_asks_since_the_last_run_and_persists_the_shown_id_on_the_decision_row():
    db, seen = {"last": LAST, "rows": {"run1": None}}, []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=_now_body())

    now = asyncio.run(_reader(db, handler).read_for_kickoff("run1"))
    assert now is not None and now.shown_id == "shown1" and len(now.items) == 2
    (req,) = seen
    assert req.url.path == "/temporal-self/now"
    assert dict(req.url.params) == {"organ": "curiosity", "run_ref": "run1", "since": LAST.isoformat()}
    assert db["rows"]["run1"] == "shown1"


def test_with_no_earlier_run_the_endpoint_default_applies():
    db, seen = {"last": None, "rows": {"run1": None}}, []
    asyncio.run(_reader(db, lambda r: (seen.append(r), httpx.Response(200, json=_now_body()))[1])
                .read_for_kickoff("run1"))
    assert "since" not in dict(seen[0].url.params)


def test_a_failed_or_malformed_now_is_no_section_and_never_raises():
    db = {"last": LAST, "rows": {"run1": None}}

    def boom(req):
        raise httpx.ConnectError("down")

    for handler in (boom, lambda r: httpx.Response(404, json={}), lambda r: httpx.Response(200, json={"x": 1})):
        assert asyncio.run(_reader(db, handler).read_for_kickoff("run1")) is None
    assert db["rows"]["run1"] is None
    off = _reader(db, lambda r: (_ for _ in ()).throw(AssertionError("no call when off")), enabled=False)
    assert asyncio.run(off.read_for_kickoff("run1")) is None


def test_turn_end_posts_at_most_three_cites_with_the_persisted_shown_id_from_another_process():
    db, posted = {"rows": {"run1": "shown1"}}, []

    def handler(req):
        body = json.loads(req.content)
        posted.append((req.url.path, body))
        return httpx.Response(200, json={"shown_id": body["shown_id"], "citing_organ": "curiosity",
                                         "uses": [], "rejected_not_shown": [], "rejected_over_cap": []})

    # A fresh reader: nothing in memory, as when the durable turn runs after a Hub restart.
    text = "[experience:a] [experience:b] [experience:c] [experience:d]"
    assert asyncio.run(_reader(db, handler).record_turn_cites("run1", text)) is not None
    assert posted == [("/temporal-self/now/use",
                       {"shown_id": "shown1", "citing_ref": "curiosity_run:run1", "cites": ["a", "b", "c"]})]


def test_no_cites_or_no_shown_row_posts_nothing_and_a_failed_post_never_raises():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(500, text="boom")

    db = {"rows": {"run1": "shown1"}}
    assert asyncio.run(_reader(db, handler).record_turn_cites("run1", "no cites here")) is None
    assert asyncio.run(_reader({"rows": {}}, handler).record_turn_cites("run9", "[experience:a]")) is None
    assert calls == []
    assert asyncio.run(_reader(db, handler).record_turn_cites("run1", "[experience:a]")) is None
    assert len(calls) == 1

    def boom(req):
        raise httpx.ReadTimeout("slow")

    assert asyncio.run(_reader(db, boom).record_turn_cites("run1", "[experience:a]")) is None


# --- the loop: wiring ------------------------------------------------------------------------------


def _loop(**over) -> CuriosityInvestigation:
    return CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=60.0, daily_cap=3, timeout_sec=60.0,
        session_id="s", pool_provider=lambda: None, source_ref=SOURCE, reader=object(),
        durable_runs_url="http://durable.test", **over)


def test_bare_construction_is_off_and_main_turns_it_on():
    assert _loop()._now.enabled is False
    tree = ast.parse((Path(__file__).resolve().parents[1] / "scripts" / "main.py").read_text())
    kws = {kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call)
           and getattr(node.func, "id", None) == "CuriosityInvestigation" for kw in node.keywords}
    assert "now_enabled" in kws
    from app.settings import Settings

    assert Settings.model_fields["HUB_CURIOSITY_NOW_ENABLED"].default is True


def test_turn_end_schedules_cites_only_for_the_two_curiosity_lines_with_text():
    loop = _loop(now_enabled=True)
    got = []

    async def fake_record(run_id, text):
        got.append((run_id, text))

    loop._now.record_turn_cites = fake_record  # type: ignore[assignment]

    async def go():
        loop._schedule_now_cites("r1", INVESTIGATION_TAG, "[experience:a]")
        loop._schedule_now_cites("r2", SELF_INQUIRY_TAG, "[experience:b]")
        loop._schedule_now_cites("r3", "self_sense_eval", "[experience:c]")
        loop._schedule_now_cites("r4", INVESTIGATION_TAG, "")
        await asyncio.gather(*loop._now_tasks)

    asyncio.run(go())
    assert got == [("r1", "[experience:a]"), ("r2", "[experience:b]")]


def _src(fn) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def _calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) == name or getattr(n.func, "id", None) == name)]


def test_both_kickoffs_hand_the_now_to_their_prompt_and_every_turn_end_records_cites():
    inv = _src(CuriosityInvestigation._investigate)
    assert any(isinstance(a, ast.Call) and getattr(a.func, "attr", None) == "_read_now"
               for g in _calls(inv, "gather") for a in g.args)
    (kick,) = _calls(inv, "build_kickoff_prompt")
    assert "now" in {k.arg for k in kick.keywords}
    si = _src(CuriosityInvestigation._self_inquire)
    (sp,) = _calls(si, "build_self_inquiry_prompt")
    assert "now" in {k.arg for k in sp.keywords} and _calls(si, "_read_now")
    # Every place a curiosity turn ends schedules the register row AND the cites.
    for fn in (CuriosityInvestigation._investigate, CuriosityInvestigation._run_self_turn,
               CuriosityInvestigation._turn_result_for):
        tree = _src(fn)
        assert len(_calls(tree, "_schedule_now_cites")) == len(_calls(tree, "_schedule_finding_register")) >= 1


def test_the_kickoff_now_read_never_raises():
    loop = _loop(now_enabled=True)

    async def boom(run_id):
        raise RuntimeError("bad")

    loop._now.read_for_kickoff = boom  # type: ignore[assignment]
    assert asyncio.run(loop._read_now("r1")) is None
