"""Recorded tick sequences for Prime's resident loop (no-behavior-change proof for the loop library).

Run as a script in a fresh interpreter: ``uuid.uuid4`` is replaced by a counter BEFORE anything else is
imported, so every id (thought, act, want, correlation, event) is the same on every run, whichever module
the code that draws it lives in. Prints one JSON document: per scenario, every bus publish, RPC, bid,
release, model call, the saved state and every log line of the ``orion-mind.*`` loggers.

``tests/test_prime_replay.py`` runs this and compares it with ``fixtures/prime_replay_golden.json``, which
was recorded from the loop BEFORE it moved into ``orion/mindloop/`` (2026-11-04). Regenerate only when a
behavior change is intended: ``python tests/prime_replay_scenarios.py > tests/fixtures/prime_replay_golden.json``.
"""

from __future__ import annotations

import uuid

_counter = {"n": 0}


def _fake_uuid4() -> uuid.UUID:
    _counter["n"] += 1
    n = _counter["n"]
    return uuid.UUID(int=(n << 96) | (n << 32) | n, version=4)


uuid.uuid4 = _fake_uuid4  # before any import that binds ``from uuid import uuid4``

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[2])]

import test_prime_loop as tpl  # noqa: E402
import test_prime_regulate_loop as trl  # noqa: E402


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("orion-mind"):
            self.lines.append(f"{record.name} {record.levelname} {record.getMessage()}")


# Wall-clock stamps (envelope created_at, event emitted_at, an act opened/closed with no explicit time) are
# the only values not from the fake clock. Every fake-clock time is on 2026-10-30 (T0 plus a few hours), so a
# stamp on that day is kept and compared: a refactor that drops an explicit ``at=`` still fails the replay.
WALLCLOCK_KEYS = frozenset({"created_at", "emitted_at", "observed_at"})
FAKE_DAY = "2026-10-30"


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: ("<wallclock>" if k in WALLCLOCK_KEYS and isinstance(v, str) and not v.startswith(FAKE_DAY)
                    else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _dump(rt, cap: _Capture) -> dict:
    return _scrub({
        "published": [[c, e.model_dump(mode="json")] for c, e in rt.bus.published],
        "rpcs": [[c, e.model_dump(mode="json")] for c, e in rt.bus.rpcs],
        "bids": [w.model_dump(mode="json") for w in rt.bids],
        "released": list(rt.released),
        "llm_calls": [{k: v for k, v in kw.items()} for kw in getattr(rt.llm, "calls", [])],
        "saved_state": rt.bus.redis.kv.get("orion:prime:state"),
        "hashes": getattr(rt.bus.redis, "h", None),
        "logs": list(cap.lines),
    })


def _ticks(rt, minutes: list[float]) -> None:
    for m in minutes:
        rt.clock_box["now"] = tpl.T0 + timedelta(minutes=m)
        asyncio.run(rt.tick())


def scenarios() -> dict:
    out: dict = {}
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    def run(name, build, minutes):
        cap = _Capture()
        root.addHandler(cap)
        try:
            rt = build()
            _ticks(rt, minutes)
            out[name] = _dump(rt, cap)
        finally:
            root.removeHandler(cap)

    run("funded_move_then_quiet", lambda: tpl.runtime(answer=tpl.MOVE), [0, 1, 400])
    run("held_back", lambda: tpl.runtime(answer=tpl.MOVE, holdback=1.0), [0])
    run("deferred", lambda: tpl.runtime(answer=tpl.MOVE, outcome="deferred"), [0, 1])
    run("no_answer_bid", lambda: tpl.runtime(answer=tpl.MOVE, outcome=None), [0])
    run("unparseable", lambda: tpl.runtime(answer=None), [0, 0.5, 3])
    run("no_seat_then_starved_route",
        lambda: tpl.seq_runtime([tpl.NO_SEAT, tpl.NO_SEAT, tpl.NO_SEAT, tpl.NO_SEAT, (tpl.MOVE, None, {})]),
        [0, 0.25, 0.5, 0.75, 5])
    run("upstream_5xx_retry", lambda: tpl.seq_runtime([tpl.UPSTREAM_5XX, (tpl.MOVE, None, {})]), [0])
    run("two_5xx", lambda: tpl.seq_runtime([tpl.UPSTREAM_5XX, tpl.UPSTREAM_5XX]), [0])
    run("no_reply", lambda: tpl.seq_runtime([(None, "RPC timeout after 700s", {})]), [0])

    turns = tpl.JuniperTurns()

    def juniper_arrives():
        with turns.turn():
            pass

    run("preempted", lambda: tpl.runtime(answer=tpl.MOVE, turns=turns, on_call=juniper_arrives), [0])

    yield_turns = tpl.JuniperTurns()

    def yielding():
        rt = tpl.runtime(answer=tpl.MOVE, turns=yield_turns)
        return rt

    cap = _Capture()
    root.addHandler(cap)
    try:
        rt = yielding()
        with yield_turns.turn():
            asyncio.run(rt.tick())
        out["juniper_yield"] = _dump(rt, cap)
    finally:
        root.removeHandler(cap)

    def crashing_llm():
        rt = tpl.runtime(answer=tpl.MOVE)

        def boom(**kw):
            raise RuntimeError("llm exploded")

        rt.llm.request_json = boom
        return rt

    run("llm_raises", crashing_llm, [0])

    def regulate():
        rt = trl.rt_with(trl.SLOW)
        return rt

    cap = _Capture()
    root.addHandler(cap)
    try:
        rt = regulate()
        asyncio.run(rt.tick())
        rt.llm.answer = {"reading": "watching", "concerns": {}, "move": None}
        _ticks(rt, [10, 125, 130])
        out["regulate_and_settle"] = _dump(rt, cap)
    finally:
        root.removeHandler(cap)

    # Settle across a restart: the second runtime loads the first one's saved state.
    cap = _Capture()
    root.addHandler(cap)
    try:
        rt = tpl.runtime(answer=tpl.MOVE)
        asyncio.run(rt.tick())
        nom = rt.state.open_nominations[0]
        run_row = {"run_id": "r1", "line": "investigate",
                   "started_at_iso": (tpl.T0 + timedelta(hours=1)).isoformat(), "outcome_kind": "finished",
                   "focus": {"focus_id": nom.focus_id, "disposition": "explored"}}
        rt2 = tpl.runtime(answer={"reading": "ok"}, bus=rt.bus, runs=[run_row], now=tpl.T0 + timedelta(hours=3))
        asyncio.run(rt2.load())
        asyncio.run(rt2.tick())
        rt2.bids[:0] = rt.bids
        rt2.released[:0] = rt.released
        out["settle_across_restart"] = _dump(rt2, cap)
    finally:
        root.removeHandler(cap)
    return out


if __name__ == "__main__":
    print(json.dumps(scenarios(), indent=1, sort_keys=True, default=str))
