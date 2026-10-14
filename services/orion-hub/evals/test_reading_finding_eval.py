"""Offline eval of the reading-finding replay: message counts over a planted week.

Planted: read A gets an accepted+projected link and, in the same tick, a belief its run cites ->
one message. Read B gets a link at read time and a cited belief a day later -> two messages
(the live loop sees them in different ticks). Read C's link is accepted but not projected -> held.
Read D changed nothing -> no message, one honest letter line.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

NOW = datetime(2026, 10, 13, 18, 0, tzinfo=timezone.utc)
T = NOW - timedelta(days=3)


def _load():
    path = Path(__file__).with_name("run_reading_finding_replay.py")
    spec = importlib.util.spec_from_file_location("run_reading_finding_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(seed, url, summary="It says things. More.", article=""):
    return {"seed_id": seed, "request_id": None, "url": url, "title": seed, "status": "done", "stage2_status": "done",
            "created_at": T, "handoff_at": T, "stage2_completed_at": T, "landing_at": None,
            "handoff_json": json.dumps({"read_evidence": [{"url": url, "tool_name": "WebFetch", "content_chars": 900}]}),
            "stage2_result_json": json.dumps({"summary": summary, "article_summary": article}),
            "trace_id": None, "stage2_trace_id": None, "why_now": None}


def _link(seed, aid, projected=True):
    return {"seed_id": seed, "stage2_completed_at": T, "latest_state": "provisional",
            "projected_at": T if projected else None,
            "claim": json.dumps({"statement_text": f"{seed} links to something", "quote": "a quote of four words",
                                 "receipt": {"outcome": "accepted_provisional", "assertion_id": aid}})}


READS = [_read("reading:a", "https://example.org/a/page", article="Piece A says X. It also says Y. And Z."),
         _read("reading:b", "https://example.org/b/page"),
         _read("reading:c", "https://example.org/c/page"),
         _read("reading:d", "https://example.org/d/page")]
LINKS = [_link("reading:a", "assertion-a"), _link("reading:b", "assertion-b"), _link("reading:c", "assertion-c", False)]
MOVES = [{"run_id": "run_a", "completed_at": T + timedelta(seconds=30),
          "per_prior": json.dumps([{"kind": "tested", "before": 0.5, "after": 0.7, "prior_id": "p:a"}])},
         {"run_id": "run_b", "completed_at": T + timedelta(days=1),
          "per_prior": json.dumps([{"kind": "formed", "before": None, "after": 0.6, "prior_id": "p:b"}])}]
TEXTS = {"p:a": [{"prior_id": "p:a", "claim": "A claim", "why": "per example.org/a/page", "last_run_id": "run_a"}],
         "p:b": [{"prior_id": "p:b", "claim": "B claim", "formed_from": "read https://example.org/b/page", "run_id": "run_b"}]}


class _Conn:
    async def fetch(self, sql, *args):
        from orion.world_pulse_read import findings as wf
        from orion.world_pulse_read import introspect

        if sql == introspect._WINDOW_SQL:
            return [type("R", (dict,), {"__getitem__": lambda s, k: dict.get(s, k)})(r) for r in READS]
        if sql == wf.READ_LINKS_SQL:
            return LINKS
        if sql == wf.RUN_PRIOR_MOVES_SQL:
            return MOVES
        raise AssertionError(sql[:50])


def test_replay_counts_one_message_per_read_per_tick_and_holds_unprojected_links():
    mod = _load()

    async def lookup(ids):
        return {i: TEXTS.get(i, []) for i in ids}

    out = asyncio.run(mod.replay(_Conn(), days=7, prior_lookup=lookup, now=NOW))
    assert out["reads"] == 4 and out["findings"] == {"link_made": 3, "belief_moved": 2}
    assert out["held_unprojected_links"] == 1
    by_seed = [(m["seed_id"], len(m["findings"])) for m in out["messages"]]
    assert by_seed == [("reading:a", 2), ("reading:b", 1), ("reading:b", 1)]
    a = out["messages"][0]
    assert a["appendix"].endswith("Piece A says X. It also says Y. And Z.")
    assert "Piece A says X." in a["prompt"] and "Belief moved (0.50 -> 0.70): A claim" in a["prompt"]
    d = [r for r in out["letter"] if r["seed_id"] == "reading:d"][0]
    assert d["changed"] == ["no link accepted; no curiosity run cited this read"] and d["article_summary"] is None
