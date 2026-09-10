"""Stage 2 Hub loop: Wallet B gate, handoff claim, round-trip ceiling — never Curiosity."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.world_pulse_read import (
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
    WorldPulseReadStage2ResultV1,
)
from orion.world_pulse_read import wallet_a as wa
from orion.world_pulse_read import wallet_b as wb
from orion.world_pulse_read.queue import enqueue_seeds, mark_seed_done
from scripts.curiosity_investigation import _COOLDOWN_KEY, _DAILY_COUNT_KEY_PREFIX
from scripts.world_pulse_read_stage2 import (
    JOURNAL_WRITE_CHANNEL,
    GenerateOutcome,
    WorldPulseReadStage2Pipeline,
)

SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="test")
NOW = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def incr(self, key):
        self.store[key] = str(int(self.store.get(key, "0")) + 1)
        return int(self.store[key])

    async def expire(self, key, ttl):
        return True


class _FakeBus:
    def __init__(self) -> None:
        self.redis = _FakeRedis()
        self.published: list = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))

    @property
    def journal(self) -> list:
        return [(c, e) for c, e in self.published if c == JOURNAL_WRITE_CHANNEL]


class _FakeConn:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.executed: list[tuple[str, tuple]] = []
        self._created_seq = 0
        self.stage2_claimed_ids: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def _norm(self, sql: str) -> str:
        return " ".join(sql.split())

    def _claim_stage2(self) -> dict | None:
        pending = sorted(
            (
                r
                for r in self.rows.values()
                if r["status"] == "done"
                and r.get("handoff_json") is not None
                and r.get("stage2_status", "pending") == "pending"
            ),
            key=lambda r: (
                r["priority"],
                r.get("handoff_at") or r.get("created_at", 0),
                r["seed_id"],
            ),
        )
        if not pending:
            return None
        row = pending[0]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc)
        self.stage2_claimed_ids.append(row["seed_id"])
        return row

    def _returning(self, row: dict) -> dict:
        return {
            "seed_id": row["seed_id"],
            "kind": row["kind"],
            "run_id": row["run_id"],
            "url": row["url"],
            "title": row["title"],
            "section": row["section"],
            "item_id": row["item_id"],
            "handoff_json": row.get("handoff_json"),
            "trace_id": row.get("trace_id"),
        }

    async def execute(self, sql: str, *args):
        self.executed.append((sql, args))
        sql_n = self._norm(sql)
        if "INSERT INTO world_pulse_read_seed" in sql_n:
            seed_id = args[0]
            if seed_id in self.rows:
                return "INSERT 0 0"
            self._created_seq += 1
            self.rows[seed_id] = {
                "seed_id": args[0],
                "kind": args[1],
                "run_id": args[2],
                "url": args[3],
                "title": args[4],
                "section": args[5],
                "item_id": args[6],
                "priority": args[7],
                "status": "pending",
                "trace_id": None,
                "last_error": None,
                "created_at": self._created_seq,
                "claimed_at": None,
                "completed_at": None,
                "handoff_json": None,
                "handoff_at": None,
                "stage2_status": "pending",
                "stage2_claimed_at": None,
                "stage2_completed_at": None,
                "stage2_error": None,
                "stage2_trace_id": None,
            }
            return "INSERT 0 1"
        if "SET stage2_status = 'pending'" in sql_n and "stage2_status = 'claimed'" in sql_n:
            older = float(args[0]) if args else 0.0
            reason = args[1] if len(args) > 1 else None
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=older)
            n = 0
            for row in self.rows.values():
                claimed_at = row.get("stage2_claimed_at")
                if (
                    row.get("stage2_status") == "claimed"
                    and claimed_at is not None
                    and claimed_at < cutoff
                ):
                    row["stage2_status"] = "pending"
                    row["stage2_claimed_at"] = None
                    row["stage2_error"] = reason
                    n += 1
            return f"UPDATE {n}"
        if "SET stage2_status = 'claimed'" in sql_n:
            row = self._claim_stage2()
            return "UPDATE 1" if row else "UPDATE 0"
        if "SET stage2_status = 'done'" in sql_n:
            seed_id, trace_id = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["stage2_status"] = "done"
                self.rows[seed_id]["stage2_trace_id"] = trace_id
                self.rows[seed_id]["stage2_error"] = None
            return "UPDATE 1"
        if "SET stage2_status = 'failed'" in sql_n:
            seed_id, error = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["stage2_status"] = "failed"
                self.rows[seed_id]["stage2_error"] = error
            return "UPDATE 1"
        if "SET status = 'done'" in sql_n:
            seed_id, trace_id = args[0], args[1]
            handoff_json = args[2] if len(args) > 2 else None
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "done"
                self.rows[seed_id]["trace_id"] = trace_id
                if handoff_json is not None:
                    payload = json.loads(handoff_json) if isinstance(handoff_json, str) else handoff_json
                    self.rows[seed_id]["handoff_json"] = payload
                    self.rows[seed_id]["handoff_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        return "OK"

    async def fetchrow(self, sql: str, *args):
        sql_n = self._norm(sql)
        if "SET stage2_status = 'claimed'" in sql_n:
            self.executed.append((sql, args))
            row = self._claim_stage2()
            if not row:
                return None
            return self._returning(row)
        return None

    async def fetch(self, sql: str, *args):
        return []


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        return self._conn


def _seed() -> WorldPulseReadSeedV1:
    return WorldPulseReadSeedV1(
        seed_id="finding:r1:x",
        kind="finding",
        run_id="r1",
        url="https://ex.com/a",
        title="A",
        section="ai_technology",
    )


def _handoff(seed: WorldPulseReadSeedV1 | None = None) -> WorldPulseReadHandoffV1:
    seed = seed or _seed()
    return WorldPulseReadHandoffV1(
        seed_ref=seed,
        what_i_learned="Learned about packaging.",
        trace_id="tr-pipeline-1",
        created_at=NOW,
    )


def _result(*, urls: list[str] | None = None) -> WorldPulseReadStage2ResultV1:
    return WorldPulseReadStage2ResultV1(
        summary="Priors updated.",
        need_stage1_urls=urls or [],
        trace_id="tr-s2",
        created_at=NOW,
        seed_id="finding:r1:x",
    )


def _count_key_b() -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{wb.WALLET_B_COUNT_KEY_PREFIX}{today}"


def _pipeline(bus: _FakeBus, conn: _FakeConn, **over) -> WorldPulseReadStage2Pipeline:
    kwargs = dict(
        enabled=True,
        tick_interval_sec=60.0,
        min_cooldown_sec=0.0,
        daily_cap=6,
        window_start_hour=0,
        window_end_hour=0,
        timeout_sec=30.0,
        session_id="orion_world_pulse_read_stage2",
        llm_route="agent",
        timezone_name="UTC",
        max_round_trips=5,
        wallet_a_daily_cap=6,
        wallet_a_min_cooldown_sec=0.0,
        wallet_a_window_start_hour=0,
        wallet_a_window_end_hour=0,
        pool_provider=lambda: _FakePool(conn),
        source_ref=SOURCE,
    )
    kwargs.update(over)
    pipe = WorldPulseReadStage2Pipeline(**kwargs)
    pipe._bus = bus
    return pipe


async def _ready_stage2(conn: _FakeConn, seed: WorldPulseReadSeedV1 | None = None) -> None:
    seed = seed or _seed()
    await enqueue_seeds(conn, [seed])
    await mark_seed_done(conn, seed.seed_id, trace_id="tr-pipeline-1", handoff=_handoff(seed))


def test_tick_at_wallet_b_cap_does_not_claim() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    bus.redis.store[_count_key_b()] = "6"
    pipe = _pipeline(bus, conn)

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick()

    assert asyncio.run(_run()) == "daily_cap"
    assert conn.stage2_claimed_ids == []
    assert conn.rows["finding:r1:x"]["stage2_status"] == "pending"


def test_stage2_happy_path_debits_wallet_b_not_curiosity_or_wallet_a() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    bus.redis.store[_COOLDOWN_KEY] = "already"
    bus.redis.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] = "3"
    bus.redis.store[wa.WALLET_A_COOLDOWN_KEY] = "a-already"
    bus.redis.store[wa.WALLET_A_COUNT_KEY_PREFIX + datetime.now(timezone.utc).date().isoformat()] = "2"
    curiosity_before = {
        k: v for k, v in bus.redis.store.items() if k.startswith("orion:curiosity:")
    }
    wallet_a_before = {
        k: v for k, v in bus.redis.store.items() if k.startswith("orion:wp_read:wallet_a:")
    }
    pipe = _pipeline(bus, conn)

    async def _pass(handoff):
        return _result()

    pipe._stage2_pass = _pass  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert conn.rows["finding:r1:x"]["stage2_status"] == "done"
    assert conn.rows["finding:r1:x"]["stage2_trace_id"] == "tr-s2"
    assert bus.redis.store[_count_key_b()] == "1"
    assert wb.WALLET_B_COOLDOWN_KEY in bus.redis.store
    assert len(bus.journal) == 1
    payload = bus.journal[0][1].payload
    assert payload["source_ref"] == "world_pulse_read_stage2:tr-s2"
    assert "Priors updated." in payload["body"]
    curiosity_after = {
        k: v for k, v in bus.redis.store.items() if k.startswith("orion:curiosity:")
    }
    wallet_a_after = {
        k: v for k, v in bus.redis.store.items() if k.startswith("orion:wp_read:wallet_a:")
    }
    assert curiosity_after == curiosity_before
    assert wallet_a_after == wallet_a_before


def test_stage2_reclaim_on_startup_writes_process_restart_reason() -> None:
    """First tick after Hub start (`_startup_reclaim_done` still False)
    reclaims with `older_than_sec=0.0` and must label the row distinctly
    from a periodic stale-timeout reclaim -- confirmed live 2026-09-10 that
    a restart reclaim previously left zero trace."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _run():
        await _ready_stage2(conn)
        row = conn.rows["finding:r1:x"]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert pipe._startup_reclaim_done is False
        await pipe._reclaim_stale_claimed()

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["stage2_status"] == "pending"
    assert conn.rows["finding:r1:x"]["stage2_error"] == "interrupted:process_restart"
    assert pipe._startup_reclaim_done is True


def test_stage2_reclaim_after_startup_writes_stale_timeout_reason() -> None:
    """Once the one-shot startup reclaim has already fired, a later reclaim
    catching a turn stuck past `timeout_sec` must use the distinct
    stale-timeout label, not the startup one."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, timeout_sec=30.0)
    pipe._startup_reclaim_done = True

    async def _run():
        await _ready_stage2(conn)
        row = conn.rows["finding:r1:x"]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=40)
        await pipe._reclaim_stale_claimed()

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["stage2_status"] == "pending"
    assert conn.rows["finding:r1:x"]["stage2_error"] == "interrupted:stale_timeout"


def test_stage2_reclaimed_reason_cleared_by_subsequent_real_success() -> None:
    """A reclaim-then-retry cycle must not leave the `interrupted:*` marker
    behind once the retry genuinely completes."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _pass(handoff):
        return _result()

    pipe._stage2_pass = _pass  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        row = conn.rows["finding:r1:x"]
        row["stage2_status"] = "claimed"
        row["stage2_claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        await pipe._reclaim_stale_claimed()
        assert conn.rows["finding:r1:x"]["stage2_error"] == "interrupted:process_restart"
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert conn.rows["finding:r1:x"]["stage2_status"] == "done"
    assert conn.rows["finding:r1:x"]["stage2_error"] is None


def test_round_trip_ceiling_stops_at_five() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_round_trips=5)
    reentered: list[str] = []

    async def _pass(handoff):
        return _result(urls=[f"https://ex.com/more-{i}" for i in range(10)])

    async def _reenter(url, *, parent_seed):
        reentered.append(url)
        return None

    pipe._stage2_pass = _pass  # type: ignore[method-assign]
    pipe._reenter_stage1 = _reenter  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert reentered == [f"https://ex.com/more-{i}" for i in range(5)]
    assert pipe.last_round_trips == 5
    assert conn.rows["finding:r1:x"]["stage2_status"] == "done"


def test_reentry_stops_when_wallet_a_blocked() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_round_trips=5)
    reentered: list[str] = []

    async def _pass(handoff):
        return _result(urls=["https://ex.com/more-0", "https://ex.com/more-1"])

    async def _reenter(url, *, parent_seed):
        if reentered:
            return "daily_cap"
        reentered.append(url)
        return None

    pipe._stage2_pass = _pass  # type: ignore[method-assign]
    pipe._reenter_stage1 = _reenter  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert reentered == ["https://ex.com/more-0"]
    assert pipe.last_round_trips == 1


def test_default_reentry_enqueues_finding_seed() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _pass(handoff):
        return _result(urls=["https://ex.com/follow"])

    pipe._stage2_pass = _pass  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert pipe.last_round_trips == 1
    follow = [r for r in conn.rows.values() if r["url"] == "https://ex.com/follow"]
    assert len(follow) == 1
    assert follow[0]["kind"] == "finding"
    assert follow[0]["status"] == "pending"


def test_stage2_parse_failure_marks_failed_after_debit() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _boom(handoff):
        raise ValueError("bad json")

    pipe._stage2_pass = _boom  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) == "parse_failed"
    assert conn.rows["finding:r1:x"]["stage2_status"] == "failed"
    assert bus.redis.store[_count_key_b()] == "1"
    assert bus.journal == []


# --- _generate: six distinct failure reasons, not one collapsed "" (#2163 Part 2) ---


def test_generate_returns_bus_unavailable_when_no_bus() -> None:
    conn = _FakeConn()
    pipe = _pipeline(_FakeBus(), conn)
    pipe._bus = None

    outcome = asyncio.run(pipe._generate("prompt", "corr-1"))
    assert outcome.text == ""
    assert outcome.fail_reason == "bus_unavailable"


def test_generate_returns_stage2_turn_timeout_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, timeout_sec=0.01)

    async def _hang(**kwargs):
        await asyncio.sleep(1.0)
        return []

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _hang)

    outcome = asyncio.run(pipe._generate("prompt", "corr-2"))
    assert outcome.text == ""
    assert outcome.fail_reason == "stage2_turn_timeout"
    assert outcome.fail_reason != "empty_generation"


def test_generate_returns_turn_exception_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _boom(**kwargs):
        raise RuntimeError("governor unreachable")

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _boom)

    outcome = asyncio.run(pipe._generate("prompt", "corr-3"))
    assert outcome.text == ""
    assert outcome.fail_reason == "turn_exception:governor unreachable"


def test_generate_pulls_real_reason_off_turn_error_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression this whole patch exists for: a stalled-stream turn_error
    frame must surface `turn_error:fcc_stream_stalled`, not a bare `""` that
    collapses into the same `empty_generation` as five unrelated failures."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _turn_error(**kwargs):
        return [
            {
                "type": "turn_error",
                "error_code": "fcc_stream_stalled",
                "error": (
                    "fcc stream stalled for 180.0s without completing a step "
                    "(turn_timeout=2400.0s, steps_seen=0)"
                ),
            }
        ]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _turn_error)

    outcome = asyncio.run(pipe._generate("prompt", "corr-4"))
    assert outcome.text == ""
    assert outcome.fail_reason == "turn_error:fcc_stream_stalled"


def test_generate_turn_deferred_reason_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """turn_deferred must truncate like every other reason branch -- a review
    finding caught this one path skipping `_FAIL_REASON_DETAIL_MAX_LEN`."""
    from scripts.world_pulse_read_stage2 import _FAIL_REASON_DETAIL_MAX_LEN

    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)
    long_reason = "x" * (_FAIL_REASON_DETAIL_MAX_LEN + 50)

    async def _deferred(**kwargs):
        return [{"type": "turn_deferred", "reason": long_reason}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _deferred)

    outcome = asyncio.run(pipe._generate("prompt", "corr-4b"))
    assert outcome.text == ""
    assert outcome.fail_reason == f"turn_deferred:{'x' * _FAIL_REASON_DETAIL_MAX_LEN}"
    assert len(outcome.fail_reason) <= len("turn_deferred:") + _FAIL_REASON_DETAIL_MAX_LEN


def test_generate_no_final_frame_falls_back_when_nothing_useful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _empty(**kwargs):
        return []

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _empty)

    outcome = asyncio.run(pipe._generate("prompt", "corr-5"))
    assert outcome.text == ""
    assert outcome.fail_reason == "no_final_frame"


def test_generate_blank_final_response_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _blank(**kwargs):
        return [{"type": "final", "llm_response": "   "}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _blank)

    outcome = asyncio.run(pipe._generate("prompt", "corr-6"))
    assert outcome.text == ""
    assert outcome.fail_reason == "blank_final_response"


def test_generate_looks_like_error_text_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _error_text(**kwargs):
        return [{"type": "final", "llm_response": "Error: something broke"}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _error_text)
    monkeypatch.setattr(
        "orion.cognition.cortex_payload_extract.looks_like_error_text", lambda t: True
    )

    outcome = asyncio.run(pipe._generate("prompt", "corr-7"))
    assert outcome.text == ""
    assert outcome.fail_reason == "looks_like_error_text"


def test_generate_success_has_no_fail_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _ok(**kwargs):
        return [{"type": "final", "llm_response": "hello"}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _ok)

    outcome = asyncio.run(pipe._generate("prompt", "corr-8"))
    assert outcome.text == "hello"
    assert outcome.fail_reason is None


def test_stage2_pass_raises_with_specific_reason_not_generic_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a stalled turn must land a specific reason in stage2_error,
    not the same 'empty_generation' every other failure produced before this."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, timeout_sec=0.01)

    async def _hang(**kwargs):
        await asyncio.sleep(1.0)
        return []

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _hang)

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    result = asyncio.run(_run())
    assert result == "parse_failed"
    assert conn.rows["finding:r1:x"]["stage2_error"] == "stage2_turn_timeout"
    assert conn.rows["finding:r1:x"]["stage2_error"] != "empty_generation"
