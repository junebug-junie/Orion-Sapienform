"""Recorded tick sequences for Prime's resident loop (no-behavior-change proof for the loop library).

Run as a script in a fresh interpreter: ``uuid.uuid4`` is replaced by a counter BEFORE anything else is
imported, so every id (thought, act, want, correlation, event) is the same on every run, whichever module
the code that draws it lives in. Prints one JSON document: per scenario, every bus publish, RPC, bid,
release, model call, the saved state and every log line of the ``orion-mind.*`` loggers.

``tests/test_prime_replay.py`` runs this and compares it with ``fixtures/prime_replay_golden.json``, which
was recorded from the loop BEFORE it moved into ``orion/mindloop/`` (2026-11-04), then re-recorded once when Prime
started publishing its concerns view (patch 3: only ``save orion:prime:concerns`` lines were added), and once more
when organs could answer Prime (patch 4: only the two ``organ_*`` scenarios were added). Regenerate only when a
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
        "timeline": list(getattr(rt, "timeline", [])),
    })


def _instrument(rt, timeline=None):
    """One ordered list across channels: every publish, save, bid, release and model call, in the order they
    happened (the per-channel lists above cannot show order between channels, nor how often state is saved)."""
    tl = timeline if timeline is not None else []
    rt.timeline = tl
    bus = rt.bus
    if not getattr(bus, "_timeline_wrapped", False):
        bus._timeline_wrapped = True
        publish, rset = bus.publish, bus.redis.set

        async def pub(channel, envelope):
            p = envelope.payload if isinstance(envelope.payload, dict) else {}
            act = p.get("act") or {}
            tl.append(f"publish {channel} {p.get('event_kind', '-')} {act.get('verb', '-')}")
            return await publish(channel, envelope)

        async def save(key, value):
            tl.append(f"save {key}")
            return await rset(key, value)

        bus.publish, bus.redis.set = pub, save
    bid, release = rt.bid, rt.release

    async def b(want):
        tl.append(f"bid {want.family}")
        return await bid(want)

    async def r(want_id):
        tl.append("release")
        return await release(want_id)

    rt.bid, rt.release = b, r
    call = rt.llm.request_json

    def llm(**kw):
        tl.append(f"llm {kw.get('route')}")
        return call(**kw)

    rt.llm.request_json = llm
    return rt


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
            rt = _instrument(build())
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
        rt = _instrument(yielding())
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
        rt = _instrument(regulate())
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
        rt = _instrument(tpl.runtime(answer=tpl.MOVE))
        asyncio.run(rt.tick())
        nom = rt.state.open_nominations[0]
        run_row = {"run_id": "r1", "line": "investigate",
                   "started_at_iso": (tpl.T0 + timedelta(hours=1)).isoformat(), "outcome_kind": "finished",
                   "focus": {"focus_id": nom.focus_id, "disposition": "explored"}}
        rt2 = tpl.runtime(answer={"reading": "ok"}, bus=rt.bus, runs=[run_row], now=tpl.T0 + timedelta(hours=3))
        _instrument(rt2, rt.timeline)
        asyncio.run(rt2.load())
        asyncio.run(rt2.tick())
        rt2.bids[:0] = rt.bids
        rt2.released[:0] = rt.released
        out["settle_across_restart"] = _dump(rt2, cap)
    finally:
        root.removeHandler(cap)
    # The self-monitor act gate ON (view and ledger faked): the infer gate now lives in the loop library.
    for name, faculty, ledger_rows, minutes in (
        ("gate_infer_refused", "act.prime.infer",
         [{"verb": "infer", "object_ref": "prime_thought:", "test_before": None, "opened_at": tpl.T0 - timedelta(hours=1)}],
         [0, 0.25, 2]),
        ("gate_infer_probe_spent", "act.prime.infer", [], [0, 3]),
        ("gate_predict_refused", "act.prime.predict",
         [{"verb": "predict", "object_ref": "prime_concern:", "test_before": None, "opened_at": tpl.T0 - timedelta(hours=1)}],
         [0, 0.25, 3]),
    ):
        run(name, lambda f=faculty, rows=ledger_rows: _gated(f, rows), minutes)
    # Curiosity replica patch 4 (recorded LAST, so the id counter of every older scenario is unchanged): the organ answers. Answering on, a published replies view: the frame's ORGAN
    # REPLIES section and its prompt paragraph; then a refused question settles with the organ's reason.
    def organ_view(bus, prime_act_id, stance):
        from orion.schemas.curiosity_replica import REPLIES_KEY, CuriosityReplicaRepliesViewV1, CuriosityReplicaReplyV1

        at = tpl.T0 + timedelta(minutes=5)
        bus.redis.kv[REPLIES_KEY] = CuriosityReplicaRepliesViewV1(as_of=at, replies=[CuriosityReplicaReplyV1(
            answer_act_id="act_answer_replay", prime_act_id=prime_act_id, target_kind="focus", target_id="f1",
            stance=stance, reason="three runs died on this kind of question", about="the cabinet question",
            at=at)]).model_dump_json()

    for name, stance in (("organ_challenge_in_frame", "challenge"), ("organ_refusal_settles_question", "refuse")):
        cap = _Capture()
        root.addHandler(cap)
        try:
            rt = _instrument(tpl.runtime(answer=tpl.MOVE))
            rt.s.CURIOSITY_REPLICA_ENABLED = True
            rt.s.CURIOSITY_REPLICA_ANSWERS_ENABLED = True
            asyncio.run(rt.tick())
            organ_view(rt.bus, rt.state.open_nominations[0].act_id, stance)
            rt.llm.answer = {"reading": "the organ answered", "concerns": {}, "move": None}
            rt.state = rt.state.model_copy(update={"pending_changes": ["the organ answered"]})
            _ticks(rt, [6, 10])
            out[name] = _dump(rt, cap)
        finally:
            root.removeHandler(cap)
    return out


def _gated(faculty: str, rows: list):
    """A runtime whose gate is the real build_act_gate with its view read and ledger read replaced, wherever
    build_act_gate lives (app/prime_loop.py before the loop library, orion/mindloop/gate.py after)."""
    from types import SimpleNamespace

    from app import prime_loop
    from orion.self_monitor.view import self_monitor_view

    at = tpl.T0
    raw = json.dumps({"kind": "self_monitor.state.v1", "computed_at": at.isoformat(), "window_days": 7.0,
                      "faculties": [{
                          "faculty": faculty, "gradeability": "graded",
                          "window_start": (at - timedelta(days=7)).isoformat(), "window_end": at.isoformat(),
                          "n_bets": 42, "n_graded": 41, "n_failed_to_execute": 0, "n_expired": 0,
                          "weight_graded": 41.0, "reliability": 0.05, "candidate_status": "unreliable",
                          "status": "unreliable", "status_since": (at - timedelta(days=1)).isoformat()}]})
    g = prime_loop.build_act_gate.__globals__
    g["read_view_sync"] = lambda url, now: self_monitor_view(raw, now=now)
    g["_ledger_rows"] = lambda dsn, actor, verb, secs: [dict(r) for r in rows if r.get("verb") == verb]
    rt = tpl.runtime(answer=tpl.MOVE)
    rt.gate = prime_loop.build_act_gate(SimpleNamespace(**{
        **vars(tpl.SETTINGS), "SELF_MONITOR_ACT_GATE_ENABLED": True, "SELF_MONITOR_PROBE_INTERVAL_SEC": 21600.0,
        "SELF_MONITOR_LEDGER_PG_DSN": "postgresql://x/y", "ORION_BUS_URL": "redis://x:6379/0"}))
    return rt


if __name__ == "__main__":
    print(json.dumps(scenarios(), indent=1, sort_keys=True, default=str))
