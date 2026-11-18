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
