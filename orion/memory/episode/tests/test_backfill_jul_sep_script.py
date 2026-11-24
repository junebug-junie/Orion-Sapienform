"""scripts/backfill_episode_memory_jul_sep.py (issue #3537): the July-September backfill takes the
same turns, groups them the same way, and writes rows the same way as the live path."""
from __future__ import annotations

import ast
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "backfill_episode_memory_jul_sep.py"

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("backfill_episode_memory_jul_sep", SCRIPT)
B = importlib.util.module_from_spec(_spec)
import sys  # noqa: E402

sys.modules[_spec.name] = B   # dataclasses look their module up here
_spec.loader.exec_module(B)

T0 = datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc)   # 09:00 Denver


def _sql_writer_initiator():
    """The live rule, lifted from sql-writer's source (its module needs the service's runtime)."""
    src = (ROOT / "services" / "orion-sql-writer" / "app" / "worker.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_memory_turn_initiator")
    ns: dict = {"json": json, "Any": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "worker.py", "exec"), ns)  # noqa: S102
    return ns["_memory_turn_initiator"]


@pytest.mark.parametrize("prompt,meta", [
    ("hi", None), ("", None), ("  ", {"unsolicited": True}), ("", {"unsolicited": True}),
    ("", {"unsolicited": "true"}), ("hi", {"room_claude": {"prompt": "x"}}), ("", '{"unsolicited": true}'),
    ("hi", '{"room_claude": true}'), (None, {}), ("hi", "not json"),
])
def test_initiator_rule_matches_sql_writer(prompt, meta):
    assert B.memory_turn_initiator(prompt, meta) == _sql_writer_initiator()((prompt or "").strip(), meta)  # its caller strips


def _row(cid, at, *, prompt="p", response="r", session="orion_journal", meta=None):
    return {"correlation_id": cid, "created_at": at.replace(tzinfo=None), "prompt": prompt, "response": response,
            "session_id": session, "client_meta": meta}


def test_select_targets_applies_the_live_filters():
    rows = [
        _row("a", T0),
        _row("b", T0 + timedelta(minutes=1), prompt="", meta={"unsolicited": True}),   # Orion outreach
        _row("c", T0 + timedelta(minutes=2), prompt=""),                                 # not memory
        _row("d", T0 + timedelta(minutes=3), response=" "),                              # no response
        _row("e", T0 + timedelta(minutes=4), meta={"external_room": {"platform": "aitown"}}),
        _row("f", T0 + timedelta(minutes=5)),                                             # already tracked
        _row("g", T0 + timedelta(minutes=6), session="reading-receipt-live-smoke-1"),
        _row("h", T0 + timedelta(minutes=7), meta={"room_claude": True}),
    ]
    kept, skipped = B.select_targets(rows, tracked_ids={"f"})
    assert [(r["correlation_id"], r["initiated_by"]) for r in kept] == [("a", "juniper"), ("b", "orion")]
    assert skipped == {"not_a_memory_turn": 2, "no_response": 1, "external_platform": 1,
                       "already_in_a_tracked_episode": 1, "smoke_session": 1}


def _targets(spec):
    """[(cid, minutes from T0, initiator)] -> target rows."""
    return [{**_row(c, T0 + timedelta(minutes=m), prompt="" if who == "orion" else "p",
                    meta={"unsolicited": True} if who == "orion" else None), "initiated_by": who}
            for c, m, who in spec]


def test_grouping_follows_the_live_tracker():
    eps = B.group_episodes(_targets([
        ("j1", 0, "juniper"), ("j2", 1, "juniper"), ("j3", 10, "juniper"),   # one breath
        ("o1", 10 + 240, "orion"),                                           # Orion after 4 h silence: new
        ("j4", 10 + 240 + 60, "juniper"),                                    # her reply joins Orion's
        ("j5", 10 + 240 + 60 + 3 * 24 * 60, "juniper"),                      # 3 days later: stale_thread
    ]), closing_turn={"correlation_id": "live1", "at": (T0 + timedelta(days=10)).isoformat()})
    assert [e["turn_ids"] for e in eps] == [["j1", "j2", "j3"], ["o1", "j4"], ["j5"]]
    assert [e["close_reason"] for e in eps] == ["v2:orion_after_silence", "v2:phase_stale_thread",
                                                "backfill:tracker_started"]
    assert all(e["episode_status"] == "closed" for e in eps)
    assert eps[-1]["closing_turn_id"] == "live1"


def test_orion_only_episode_is_skipped_like_live():
    eps = B.group_episodes(_targets([("j1", 0, "juniper"), ("o1", 300, "orion"), ("o2", 400, "orion")]))
    assert [e["episode_status"] for e in eps] == ["closed", "skipped"]
    assert eps[1]["skip_reason"] == "no_juniper_turn"


def test_episode_ids_are_deterministic_so_a_rerun_writes_nothing_new():
    spec = [("j1", 0, "juniper"), ("j2", 3000, "juniper")]
    a = [e["episode_id"] for e in B.group_episodes(_targets(spec))]
    b = [e["episode_id"] for e in B.group_episodes(_targets(spec))]
    assert a == b and len(set(a)) == 2


def test_import_of_the_live_boundary_leaves_other_app_packages_alone(monkeypatch):
    sentinel = SimpleNamespace(name="someone_elses_app")
    monkeypatch.setitem(__import__("sys").modules, "app", sentinel)
    mod = B._load_live_boundary()
    assert hasattr(mod, "episode_boundary")
    assert __import__("sys").modules["app"] is sentinel


def _mem(keys, purpose="happened", occurred=None):
    return SimpleNamespace(referents=[(k, "subject") for k in keys], purpose=purpose, occurred_at=occurred)


def test_referent_coverage_counts_any_and_non_self():
    cov = B.referent_coverage([_mem(["person:juniper"]), _mem(["person:juniper", "pet:biscuit"]), _mem([])])
    assert (cov["n"], cov["any"], cov["other"]) == (3, 2, 1)


def test_gate_blocks_below_the_live_band():
    ok = [{"week": "2026-08-03", "n": 40, "any_ref": 40, "other_ref": 30}]
    assert B.gate_verdict(ok)["pass"]
    low_any = [{"week": "2026-08-03", "n": 40, "any_ref": 38, "other_ref": 30}]   # 95%
    assert not B.gate_verdict(low_any)["pass"]
    low_other = [{"week": "2026-08-03", "n": 40, "any_ref": 40, "other_ref": 26}]  # 65%
    assert not B.gate_verdict(low_other)["pass"]
    # A tiny week is reported, not judged alone; the pooled total still has to be in band.
    small = ok + [{"week": "2026-08-10", "n": 3, "any_ref": 2, "other_ref": 1}]
    v = B.gate_verdict(small)
    assert v["pass"] and v["small_weeks"] == ["2026-08-10"]
    assert not B.gate_verdict([])["pass"]


def test_occurred_after_episode_flags_impossible_dates_only():
    end = T0
    mems = [_mem([], occurred=T0 - timedelta(days=400)), _mem([], occurred=T0 + timedelta(days=3)),
            _mem([], purpose="follow_up", occurred=T0 + timedelta(days=3)), _mem([], occurred=None)]
    assert B.occurred_after_episode(mems, end) == 1


def test_dry_run_sample_spans_the_range_and_reaches_ten_exchanges():
    eps = [{"episode_status": "closed", "juniper_turn_count": 2, "episode_id": str(i)} for i in range(40)]
    picked = B.dry_run_sample(eps, 10)
    assert sum(e["juniper_turn_count"] for e in picked) >= 10
    assert picked[0]["episode_id"] == "0" and int(picked[-1]["episode_id"]) > 10


def test_persist_writes_as_of_the_episode_with_the_live_referent_policy_and_no_supersede(monkeypatch):
    """created_at must keep episode order: the supersede pass treats a later created_at as the newer
    row, so a July row stamped today could retire a November correction."""
    import orion.memory.episode.store as store

    seen: dict = {}

    async def fake_persist(pool, **kw):
        seen.update(kw)
        return {"memories": 0, "rejections": 0, "questions": 0, "downgrades": 0, "superseded": 0}

    monkeypatch.setattr(store, "persist_episode", fake_persist)
    monkeypatch.setattr(B, "validated_for", lambda *a, **k: SimpleNamespace(memories=[], questions=[]))
    monkeypatch.setattr("orion.memory.episode.validate.coverage", lambda r, t: {"coverage": 1.0})

    class Conn:
        sql = []

        async def execute(self, sql, params=None):
            Conn.sql.append((sql, params))

    class Pool:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def connection(self):
            yield Conn()

    ep = B.group_episodes(_targets([("j1", 0, "juniper"), ("j2", 5, "juniper")]))[0]
    asyncio.run(B.persist(Pool(), ep, [], {"prompt_version": "v6", "model": "m", "answer_text": "{}"}))
    assert seen["now"] == datetime.fromisoformat(ep["last_turn_at"])
    assert seen["run_id"].startswith("backfill3537-")
    assert seen["referent_policy"].grounding_auto_accept and seen["referent_policy"].cooccurrence_auto_accept
    assert "supersede_links" not in seen and "supersede_plan" not in seen
    sql, params = Conn.sql[0]
    assert "memory_episode_shadow" in sql and "closed_event_published_at" in sql and "now()" in sql
    assert params["episode_id"] == ep["episode_id"]


def test_llm_body_matches_the_live_distill_call():
    body = B.llm_body("prompt")
    assert body["max_tokens"] == 8192 and body["temperature"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "model" not in body   # the operator lane names its own route


def test_progress_line_carries_counts_only(tmp_path):
    p = B.Progress(tmp_path / "progress.log", total=4, total_turns=10)
    p.done, p.turns = 1, 3
    p.line("episode_done episode=abc memories=2", anomalies="no_memories")
    line = (tmp_path / "progress.log").read_text()
    for part in ("25.0%", "ETA", "episodes 1/4 turns 3/10", "episodes/h", "errors 0", "no_memories"):
        assert part in line
    assert oct((tmp_path / "progress.log").stat().st_mode & 0o777) == "0o600"


def _scored(spec):
    """[(cid, minutes, score)] Juniper turns carrying a stored classify score."""
    return [{**_row(c, T0 + timedelta(minutes=m)), "spark_meta": {"conversation_boundary_score": sc},
             "initiated_by": "juniper"} for c, m, sc in spec]


def test_resumed_thread_splits_on_the_stored_llm_score_like_live():
    # 30 min later = resumed_thread: live splits only when the stored score is >= 0.92.
    eps = B.group_episodes(_scored([("j1", 0, 0.1), ("j2", 30, 0.95), ("j3", 60, 0.5)]))
    assert [e["turn_ids"] for e in eps] == [["j1"], ["j2", "j3"]]
    assert eps[0]["close_reason"] == "v2:resumed_thread+llm"
    assert B.boundary_score({"conversation_boundary_score": True}) is None
    assert B.boundary_score('{"conversation_boundary_score": 0.5}') == 0.5


def test_session_clock_moves_on_prompts_that_are_not_memory_turns():
    # j1 at 0; a Juniper prompt with no response at 100 min (dropped as a target, still moves the
    # clock); j2 at 110 min is then same_breath-ish (10 min = short_pause), not resumed_thread.
    targets = _scored([("j1", 0, 0.1), ("j2", 110, 0.99)])
    clock = targets + [_row("x", T0 + timedelta(minutes=100), response="")]
    assert len(B.group_episodes(targets)) == 2                       # without it: resumed_thread + 0.99
    assert len(B.group_episodes(targets, clock_rows=clock)) == 1     # with it: short_pause, joins


def test_tracker_reads_skip_the_backfills_own_rows():
    for sql in (B.TRACKED_IDS_SQL, B.CUTOFF_SQL, B.FIRST_TRACKED_SQL):
        assert B.NOT_BACKFILL in sql
    ep = B.group_episodes(_targets([("j1", 0, "juniper")]))[0]
    assert all(t["phase_source"] == B.BACKFILL_PHASE_SOURCE for t in ep["turns"])


def test_answer_cache_is_keyed_by_turn_set():
    a = {"episode_id": "e", "turn_ids": ["1", "2"]}
    assert B.cache_path_for(a) != B.cache_path_for({**a, "turn_ids": ["1"]})


def test_failed_attempt_gives_the_hold_back_before_waiting(monkeypatch):
    import contextlib

    import orion.evals.leased_llm as L

    events = []

    @contextlib.contextmanager
    def lane_cm(*a, **k):
        events.append("acquire")
        yield object()
        events.append("release")

    monkeypatch.setattr(L, "operator_lane_sync", lane_cm)
    monkeypatch.setattr(B.time, "sleep", lambda s: events.append(f"sleep{int(s)}"))
    calls = {"n": 0}

    def handle(lane, ep):
        calls["n"] += 1
        if ep["episode_id"] == "bad":
            raise B.AttemptFailed("boom")

    gave_up = []
    args = SimpleNamespace(bus_url="x", gateway=None, queue_wait_sec=1, max_hold_sec=10_000)
    B._hold_loop(args, [{"episode_id": "bad"}, {"episode_id": "ok"}], handle, lambda e, d: gave_up.append(e["episode_id"]))
    assert gave_up == ["bad"]
    # every backoff sleep happens after a release, never while holding
    for i, e in enumerate(events):
        if e.startswith("sleep"):
            assert events[i - 1] == "release"
    assert events.count("sleep5") == 1 and calls["n"] == B.MAX_ATTEMPTS + 1


def test_supersede_summary_counts_pairs_touching_backfill(tmp_path):
    (tmp_path / "pairs.jsonl").write_text(json.dumps({"old_id": "a", "new_id": "b"}) + "\n"
                                          + json.dumps({"old_id": "c", "new_id": "d"}) + "\n")
    s = B.supersede_summary(tmp_path, backfill_ids={"a"})
    assert (s["pairs"], s["pairs_touching_backfill"]) == (2, 1)
    assert B.supersede_summary(tmp_path / "none") == {"ran": False}


def test_backfill_rows_queue_no_confirmation_asks():
    m_high = SimpleNamespace(confirmation_state="pending_confirmation", stakes="high")
    m_low = SimpleNamespace(confirmation_state="auto", stakes="low")
    res = B.without_asks(SimpleNamespace(memories=[m_high, m_low]))
    assert (m_high.confirmation_state, m_low.confirmation_state) == ("not_asked", "auto")
    kept = SimpleNamespace(confirmation_state="pending_confirmation")
    B.without_asks(SimpleNamespace(memories=[kept]), asks=True)
    assert kept.confirmation_state == "pending_confirmation"
    assert res.memories[0] is m_high


def test_the_asker_selects_only_pending_rows():
    """open_cards (and the do-not-remint guard) pick rows by state = 'pending_confirmation' only, so
    a not_asked row can never become an ask."""
    src = (ROOT / "orion" / "memory" / "episode" / "confirmation.py").read_text(encoding="utf-8")
    start = src.index("async def open_cards(")
    body = src[start: src.index("\nasync def ", start + 10) if "\nasync def " in src[start + 10:] else None]
    assert "confirmation_state = 'pending_confirmation'" in body
    assert "not_asked" not in src


def test_not_asked_is_supersedable_but_never_corrects_and_is_hedged():
    from orion.memory.episode import supersede as S
    from orion.memory.voice_render import VoicedMemory, render_memory

    row = SimpleNamespace(confirmation_state="not_asked", purpose="happened", statement="x")
    assert S.can_be_superseded(row)          # nobody will ever settle it: not held immune
    assert not S.can_correct(row)            # unsettled: never retires anything
    vm = VoicedMemory(voice="juniper_said", channel="chat", statement="s", confirmation_state="not_asked",
                      has_verified_juniper_quote=True)
    assert render_memory(vm).startswith("Unconfirmed")   # recall still hedges it


def test_spread_sample_and_intervals():
    eps = [{"episode_status": "closed", "episode_id": str(i)} for i in range(101)]
    picked = B.spread_sample(eps, 15)
    assert len(picked) == 15 and picked[0]["episode_id"] == "0" and picked[-1]["episode_id"] == "100"
    lo, hi = B.wilson(8, 15)
    assert lo < 8 / 15 < hi
    lo2, hi2 = B.cluster_interval([(3, 10), (5, 5)])
    assert lo2 <= 8 / 15 <= hi2
    assert B.wilson(0, 0) == (None, None)


def test_name_repair_never_turns_backfill_rows_into_asks():
    src = (ROOT / "scripts" / "repair_episode_memory_ungrounded_names.py").read_text(encoding="utf-8")
    assert "m.run_id NOT LIKE 'backfill3537-%'" in src


def test_backfill_floor_is_juniper_decision_and_live_floor_unchanged():
    assert B.BACKFILL_OTHER_MIN == 0.53 and B.BAND_OTHER_MIN == 0.68 and B.BAND_ANY_MIN == 0.97
    week = [{"week": "2026-08-03", "n": 40, "any_ref": 40, "other_ref": 23}]   # 57.5%
    assert not B.gate_verdict(week)["pass"]                                    # live floor
    assert B.gate_verdict(week, other_min=B.BACKFILL_OTHER_MIN)["pass"]       # backfill floor


def test_small_weeks_below_the_floor_are_flagged_not_silently_passed():
    weeks = [{"week": "2026-08-03", "n": 40, "any_ref": 40, "other_ref": 30},
             {"week": "2026-08-10", "n": 4, "any_ref": 4, "other_ref": 1}]   # 25%, small
    v = B.gate_verdict(weeks, other_min=0.53)
    assert v["pass"]   # pooled 31/44 = 70%; the small week alone does not fail
    assert [w["week"] for w in v["small_weeks_below"]] == ["2026-08-10"]
    assert B.week_below(weeks[1], 0.53) and not B.week_below(weeks[0], 0.53)
    big_low = [{"week": "2026-08-03", "n": 40, "any_ref": 40, "other_ref": 20}]   # 50%
    assert not B.gate_verdict(big_low, other_min=0.53)["pass"]


def test_gate_refuses_an_out_of_range_floor(monkeypatch):
    monkeypatch.setattr(B, "_coverage_weeks", lambda pool: None)
    monkeypatch.setattr(B.asyncio, "run", lambda coro: [])
    with pytest.raises(SystemExit):
        B.cmd_gate(SimpleNamespace(dsn="x", other_min=0.0))
    with pytest.raises(SystemExit):
        B.cmd_gate(SimpleNamespace(dsn="x", other_min=1.5))
