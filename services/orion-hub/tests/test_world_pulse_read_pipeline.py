"""Stage 1 Hub loop: Wallet A gate, Concept Atlas materialize, journal — never Curiosity."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
)
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.world_pulse_read import wallet_a as wa
from orion.world_pulse_read.queue import enqueue_seeds
from scripts.world_pulse_read_pipeline import (
    JOURNAL_WRITE_CHANNEL,
    WorldPulseReadPipeline,
)
from scripts.curiosity_investigation import _COOLDOWN_KEY, _DAILY_COUNT_KEY_PREFIX

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

    async def decr(self, key):
        self.store[key] = str(int(self.store.get(key, "0")) - 1)
        return int(self.store[key])

    async def delete(self, key):
        self.store.pop(key, None)

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


class _LegacyFakeConn:
    """Interprets the real SQL strings `queue.py` emits."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.executed: list[tuple[str, tuple]] = []
        self.digest_rows: list[dict] = []
        self.article_rows: list[dict] = []
        self._created_seq = 0
        self.claimed_ids: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def _norm(self, sql: str) -> str:
        return " ".join(sql.split())

    def _claim_pending(self) -> dict | None:
        pending = sorted(
            (r for r in self.rows.values() if r["status"] == "pending"),
            key=lambda r: (r["priority"], r.get("created_at", 0), r["seed_id"]),
        )
        if not pending:
            return None
        row = pending[0]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc)
        self.claimed_ids.append(row["seed_id"])
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
            }
            return "INSERT 0 1"
        if (
            "UPDATE world_pulse_read_seed" in sql_n
            and "SET status = 'pending'" in sql_n
            and "status = 'claimed'" in sql_n
        ):
            older = float(args[0]) if args else 0.0
            reason = args[1] if len(args) > 1 else None
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=older)
            n = 0
            for row in self.rows.values():
                claimed_at = row.get("claimed_at")
                if (
                    row["status"] == "claimed"
                    and claimed_at is not None
                    and claimed_at < cutoff
                ):
                    row["status"] = "pending"
                    row["claimed_at"] = None
                    row["last_error"] = reason
                    n += 1
            return f"UPDATE {n}"
        if "UPDATE world_pulse_read_seed" in sql_n and "SET status = 'done'" in sql_n:
            seed_id, trace_id = args[0], args[1]
            handoff_json = args[2] if len(args) > 2 else None
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "done"
                self.rows[seed_id]["trace_id"] = trace_id
                self.rows[seed_id]["last_error"] = None
                if handoff_json is not None:
                    payload = handoff_json
                    if isinstance(payload, str):
                        import json as _json

                        payload = _json.loads(payload)
                    self.rows[seed_id]["handoff_json"] = payload
                    self.rows[seed_id]["handoff_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        if "UPDATE world_pulse_read_seed" in sql_n and "status = 'failed'" in sql_n:
            seed_id, error = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "failed"
                self.rows[seed_id]["last_error"] = error
            return "UPDATE 1"
        if "UPDATE world_pulse_read_seed" in sql_n and "status = 'skipped'" in sql_n:
            seed_id, reason = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["status"] = "skipped"
                self.rows[seed_id]["last_error"] = reason
            return "UPDATE 1"
        return "OK"

    async def fetchrow(self, sql: str, *args):
        sql_n = self._norm(sql)
        if "UPDATE world_pulse_read_seed" in sql_n and "status = 'claimed'" in sql_n:
            self.executed.append((sql, args))
            row = self._claim_pending()
            if not row:
                return None
            return self._returning(row)
        return None

    async def fetch(self, sql: str, *args):
        self.executed.append((sql, args))
        sql_n = self._norm(sql)
        if "FROM world_pulse_digest" in sql_n:
            return list(self.digest_rows)
        if "FROM world_pulse_article" in sql_n:
            return list(self.article_rows)
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


def _fetched(url: str, chars: int = 1200) -> SourceFetchEvidenceV1:
    """What the governor reports for a WebFetch that returned the page."""
    return SourceFetchEvidenceV1(url=url, tool_name="WebFetch", content_chars=chars)


def _handoff(seed: WorldPulseReadSeedV1 | None = None) -> WorldPulseReadHandoffV1:
    seed = seed or _seed()
    return WorldPulseReadHandoffV1(
        seed_ref=seed,
        what_i_learned="Learned about packaging.",
        concept_candidates=[
            WorldPulseReadConceptCandidateV1(
                label="advanced packaging", definition="chip pkg"
            )
        ],
        trace_id="tr-pipeline-1",
        created_at=NOW,
        read_evidence=[_fetched(seed.url)],
    )


def _count_key() -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{wa.WALLET_A_COUNT_KEY_PREFIX}{today}"


def _pipeline(
    bus: _FakeBus,
    conn: _FakeConn,
    store: InMemorySubstrateGraphStore,
    **over,
) -> WorldPulseReadPipeline:
    kwargs = dict(
        enabled=True,
        tick_interval_sec=60.0,
        min_cooldown_sec=0.0,
        daily_cap=6,
        window_start_hour=0,
        window_end_hour=0,
        timeout_sec=30.0,
        session_id="orion_world_pulse_read",
        llm_route="agent",
        timezone_name="UTC",
        pool_provider=lambda: _FakePool(conn),
        source_ref=SOURCE,
        store_provider=lambda: store,
    )
    kwargs.update(over)
    pipe = WorldPulseReadPipeline(**kwargs)
    pipe._bus = bus
    return pipe


async def _seed_queue(conn: _FakeConn, seed: WorldPulseReadSeedV1 | None = None) -> None:
    await enqueue_seeds(conn, [seed or _seed()])


def test_tick_at_daily_cap_does_not_claim_a_seed() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    bus.redis.store[_count_key()] = "6"
    pipe = _pipeline(bus, conn, store)

    async def _run():
        await _seed_queue(conn)
        return await pipe.tick()

    assert asyncio.run(_run()) == "daily_cap"
    assert conn.claimed_ids == []
    assert conn.rows["finding:r1:x"]["status"] == "pending"
    assert bus.redis.store[_count_key()] == "6"
    assert store.snapshot().nodes == {}
    assert bus.journal == []


def test_pipeline_happy_path_writes_concept_and_ignores_curiosity_wallet() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    bus.redis.store[_COOLDOWN_KEY] = "already"
    bus.redis.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] = "3"
    curiosity_before = {
        k: v for k, v in bus.redis.store.items() if k.startswith("orion:curiosity:")
    }
    pipe = _pipeline(bus, conn, store)
    handoff = _handoff()

    async def _fake_read(seed):
        return handoff

    pipe._stage1_read = _fake_read  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    nodes = list(store.snapshot().nodes.values())
    assert len(nodes) == 1
    assert nodes[0].provenance.producer == "world_pulse_read_pipeline"
    assert nodes[0].provenance.source_kind == "world_pulse.read"
    assert "https://ex.com/a" in nodes[0].provenance.evidence_refs
    assert "r1" in nodes[0].provenance.evidence_refs
    assert "tr-pipeline-1" in nodes[0].provenance.evidence_refs
    assert nodes[0].label == "advanced packaging"
    assert len(bus.journal) == 1
    channel, envelope = bus.journal[0]
    assert channel == "orion:journal:write"
    payload = envelope.payload
    assert payload["source_kind"] == "world_pulse"
    assert payload["source_ref"] == "world_pulse_read:tr-pipeline-1"
    assert "Learned about packaging." in payload["body"]
    assert "https://ex.com/a" in payload["body"]
    assert conn.rows["finding:r1:x"]["status"] == "done"
    assert conn.rows["finding:r1:x"]["trace_id"] == "tr-pipeline-1"
    assert conn.rows["finding:r1:x"]["handoff_json"]["what_i_learned"] == "Learned about packaging."
    assert conn.rows["finding:r1:x"]["handoff_json"]["trace_id"] == "tr-pipeline-1"
    curiosity_after = {
        k: v for k, v in bus.redis.store.items() if k.startswith("orion:curiosity:")
    }
    assert curiosity_after == curiosity_before
    assert not any(
        k.startswith("orion:curiosity:") and k not in curiosity_before
        for k in bus.redis.store
    )


def test_force_skips_schedule_gate_and_still_debits_wallet_a() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    bus.redis.store[_count_key()] = "6"
    pipe = _pipeline(bus, conn, store)
    handoff = _handoff()

    async def _fake_read(seed):
        return handoff

    pipe._stage1_read = _fake_read  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        blocked = await pipe.tick()
        forced = await pipe.tick(force=True)
        return blocked, forced

    blocked, forced = asyncio.run(_run())
    assert blocked == "daily_cap"
    assert forced is None
    assert bus.redis.store[_count_key()] == "7"
    assert wa.WALLET_A_COOLDOWN_KEY in bus.redis.store
    assert conn.rows["finding:r1:x"]["status"] == "done"


def test_stage1_parse_failure_marks_seed_failed_after_debit() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)

    async def _boom(seed):
        raise ValueError("unreadable handoff")

    pipe._stage1_read = _boom  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        return await pipe.tick(force=True)

    reason = asyncio.run(_run())
    assert reason in {"empty_generation", "parse_failed", "stage1_failed"}
    assert conn.rows["finding:r1:x"]["status"] == "failed"
    assert bus.redis.store[_count_key()] == "1"
    assert store.snapshot().nodes == {}
    assert bus.journal == []


def test_section_index_url_skipped_without_wallet_debit() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)
    called = {"n": 0}

    async def _fake_read(seed):
        called["n"] += 1
        return _handoff()

    pipe._stage1_read = _fake_read  # type: ignore[method-assign]

    async def _run():
        await enqueue_seeds(
            conn,
            [
                WorldPulseReadSeedV1(
                    seed_id="finding:r1:index",
                    kind="finding",
                    run_id="r1",
                    url="https://www.tomshardware.com/news",
                    title="News index",
                    section="hardware_compute_gpu",
                )
            ],
        )
        return await pipe.tick(force=True)

    reason = asyncio.run(_run())
    assert reason == "skipped_index_url"
    assert conn.rows["finding:r1:index"]["status"] == "skipped"
    assert _count_key() not in bus.redis.store
    assert called["n"] == 0
    assert store.snapshot().nodes == {}


def test_missing_store_fails_seed_when_nodes_exist() -> None:
    """Enablement without Atlas must not mark the seed done."""
    for over in ({"store_provider": None}, {"store_provider": lambda: None}):
        bus = _FakeBus()
        conn = _FakeConn()
        store = InMemorySubstrateGraphStore()
        pipe = _pipeline(bus, conn, store, **over)
        handoff = _handoff()

        async def _fake_read(seed):
            return handoff

        pipe._stage1_read = _fake_read  # type: ignore[method-assign]

        async def _run():
            await _seed_queue(conn)
            return await pipe.tick(force=True)

        reason = asyncio.run(_run())
        assert reason == "post_read_failed"
        assert conn.rows["finding:r1:x"]["status"] == "failed"
        assert conn.rows["finding:r1:x"]["last_error"] == "concept_atlas_store_unavailable"
        assert store.snapshot().nodes == {}
        assert bus.journal == []


def test_tick_reclaims_stale_claimed_seed() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)
    handoff = _handoff()

    async def _fake_read(seed):
        return handoff

    pipe._stage1_read = _fake_read  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=4000)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert conn.rows["finding:r1:x"]["status"] == "done"
    assert conn.claimed_ids == ["finding:r1:x"]
    assert store.snapshot().nodes


def test_reclaim_on_startup_writes_process_restart_reason() -> None:
    """First tick after Hub start (`_startup_reclaim_done` still False)
    reclaims with `older_than_sec=0.0` and must label the row distinctly
    from a periodic stale-timeout reclaim -- confirmed live 2026-09-10 that
    a restart reclaim previously left zero trace."""
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)

    async def _run():
        await _seed_queue(conn)
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert pipe._startup_reclaim_done is False
        await pipe._reclaim_stale_claimed()

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["status"] == "pending"
    assert conn.rows["finding:r1:x"]["last_error"] == "interrupted:process_restart"
    assert pipe._startup_reclaim_done is True


def test_reclaim_after_startup_writes_stale_timeout_reason() -> None:
    """Once the one-shot startup reclaim has already fired, a later reclaim
    catching a turn stuck past `timeout_sec` must use the distinct
    stale-timeout label, not the startup one."""
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, timeout_sec=30.0)
    pipe._startup_reclaim_done = True

    async def _run():
        await _seed_queue(conn)
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=40)
        await pipe._reclaim_stale_claimed()

    asyncio.run(_run())
    assert conn.rows["finding:r1:x"]["status"] == "pending"
    assert conn.rows["finding:r1:x"]["last_error"] == "interrupted:stale_timeout"


def test_reclaimed_seed_reason_cleared_by_subsequent_real_success() -> None:
    """A reclaim-then-retry cycle must not leave the `interrupted:*` marker
    behind once the retry genuinely completes."""
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)
    handoff = _handoff()

    async def _fake_read(seed):
        return handoff

    pipe._stage1_read = _fake_read  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        row = conn.rows["finding:r1:x"]
        row["status"] = "claimed"
        row["claimed_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        await pipe._reclaim_stale_claimed()
        assert conn.rows["finding:r1:x"]["last_error"] == "interrupted:process_restart"
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert conn.rows["finding:r1:x"]["status"] == "done"
    assert conn.rows["finding:r1:x"]["last_error"] is None


def test_generate_passes_stage1_correlation_id_to_unified_turn() -> None:
    from unittest.mock import patch

    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)
    captured: dict = {}

    async def _fake_turn(**kwargs):
        captured.update(kwargs)
        return [{"type": "final", "llm_response": '{"what_i_learned": "ok"}'}]

    async def _run():
        with patch(
            "orion.hub.turn_orchestrator.execute_unified_turn",
            _fake_turn,
        ):
            return await pipe._generate("read this", "tr-stage1-shared")

    outcome = asyncio.run(_run())
    assert outcome.text == '{"what_i_learned": "ok"}'
    assert outcome.fail_reason is None
    assert captured["correlation_id"] == "tr-stage1-shared"


def test_post_read_crash_marks_seed_failed_not_claimed() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)
    handoff = _handoff()

    async def _fake_read(seed):
        return handoff

    pipe._stage1_read = _fake_read  # type: ignore[method-assign]

    class _BoomStore:
        def get_node_id_by_identity(self, *args, **kwargs):
            raise RuntimeError("atlas write failed")

    pipe._store_provider = lambda: _BoomStore()  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        return await pipe.tick(force=True)

    reason = asyncio.run(_run())
    assert reason == "post_read_failed"
    assert conn.rows["finding:r1:x"]["status"] == "failed"
    assert conn.rows["finding:r1:x"]["last_error"] == "atlas write failed"
    assert bus.redis.store[_count_key()] == "1"


# --- _generate: distinct failure reasons, not one collapsed "" (mirrors
# world_pulse_read_stage2.py's #2166 fix, applied to Stage 1) ---


def test_generate_returns_bus_unavailable_when_no_bus() -> None:
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(_FakeBus(), conn, store)
    pipe._bus = None

    outcome = asyncio.run(pipe._generate("prompt", "corr-1"))
    assert outcome.text == ""
    assert outcome.fail_reason == "bus_unavailable"


def test_generate_returns_stage1_turn_timeout_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, timeout_sec=0.01)

    async def _hang(**kwargs):
        await asyncio.sleep(1.0)
        return []

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _hang)

    outcome = asyncio.run(pipe._generate("prompt", "corr-2"))
    assert outcome.text == ""
    assert outcome.fail_reason == "stage1_turn_timeout"
    assert outcome.fail_reason != "empty_generation"


def test_generate_returns_turn_exception_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)

    async def _boom(**kwargs):
        raise RuntimeError("governor unreachable")

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _boom)

    outcome = asyncio.run(pipe._generate("prompt", "corr-3"))
    assert outcome.text == ""
    assert outcome.fail_reason == "turn_exception:governor unreachable"


def test_generate_turn_deferred_reason_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """turn_deferred must truncate like every other reason branch -- mirrors
    Stage 2's dedicated test for this (a review finding there caught one
    path skipping `_FAIL_REASON_DETAIL_MAX_LEN`; this closes the matching
    coverage gap on Stage 1's copy of the same code)."""
    from scripts.world_pulse_read_pipeline import _FAIL_REASON_DETAIL_MAX_LEN

    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)
    long_reason = "x" * (_FAIL_REASON_DETAIL_MAX_LEN + 50)

    async def _deferred(**kwargs):
        return [{"type": "turn_deferred", "reason": long_reason}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _deferred)

    outcome = asyncio.run(pipe._generate("prompt", "corr-3b"))
    assert outcome.text == ""
    assert outcome.fail_reason == f"turn_deferred:{'x' * _FAIL_REASON_DETAIL_MAX_LEN}"
    assert len(outcome.fail_reason) <= len("turn_deferred:") + _FAIL_REASON_DETAIL_MAX_LEN


def test_generate_pulls_real_reason_off_turn_error_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same regression class this whole patch exists for on Stage 2: a
    stalled-stream turn_error frame must surface `turn_error:fcc_stream_stalled`,
    not a bare `""` that collapses into the same `empty_generation` as every
    other unrelated failure."""
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)

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


def test_generate_no_final_frame_falls_back_when_nothing_useful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)

    async def _empty(**kwargs):
        return []

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _empty)

    outcome = asyncio.run(pipe._generate("prompt", "corr-5"))
    assert outcome.text == ""
    assert outcome.fail_reason == "no_final_frame"


def test_generate_blank_final_response_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)

    async def _blank(**kwargs):
        return [{"type": "final", "llm_response": "   "}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _blank)

    outcome = asyncio.run(pipe._generate("prompt", "corr-6"))
    assert outcome.text == ""
    assert outcome.fail_reason == "blank_final_response"


def test_generate_looks_like_error_text_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)

    async def _error_text(**kwargs):
        return [{"type": "final", "llm_response": "Error: something broke"}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _error_text)
    monkeypatch.setattr(
        "orion.cognition.cortex_payload_extract.looks_like_error_text", lambda t: True
    )

    outcome = asyncio.run(pipe._generate("prompt", "corr-7"))
    assert outcome.text == ""
    assert outcome.fail_reason == "looks_like_error_text"


def test_stage1_read_raises_with_specific_reason_not_generic_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a stalled turn must land a specific reason in `last_error`,
    not the same 'empty_generation' every other failure produced before this."""
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, timeout_sec=0.01)

    async def _hang(**kwargs):
        await asyncio.sleep(1.0)
        return []

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _hang)

    async def _run():
        await _seed_queue(conn)
        return await pipe.tick(force=True)

    result = asyncio.run(_run())
    assert result == "parse_failed"
    assert conn.rows["finding:r1:x"]["last_error"] == "stage1_turn_timeout"
    assert conn.rows["finding:r1:x"]["last_error"] != "empty_generation"


# --- Bounded retry: Stage 1 sibling of the Stage 2 retry tests
# (services/orion-hub/tests/test_world_pulse_read_stage2.py). Same predicate
# (orion/world_pulse_read/retry.py), same SQL shape. ---


def test_transient_stage1_failure_is_retried_until_exhausted() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, max_attempts=3)

    async def _boom(seed):
        raise ValueError("turn_deferred:stance_react_failed: exec result missing thought payload")

    pipe._stage1_read = _boom  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        results = []
        for _ in range(3):
            results.append(await pipe.tick(force=True))
        return results

    results = asyncio.run(_run())
    row = conn.rows["finding:r1:x"]
    assert results == ["parse_failed", "parse_failed", "parse_failed"]
    assert row["attempts"] == 3
    assert row["status"] == "failed"


def test_transient_stage1_failure_reclaimable_between_retries() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, max_attempts=3)

    async def _boom(seed):
        raise ValueError("stage1_turn_timeout")

    pipe._stage1_read = _boom  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        await pipe.tick(force=True)
        row = conn.rows["finding:r1:x"]
        return row["status"], row["attempts"], row["claimed_at"]

    status, attempts, claimed_at = asyncio.run(_run())
    assert status == "pending"
    assert attempts == 1
    assert claimed_at is None


def test_non_transient_stage1_failure_is_terminal_on_first_try() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, max_attempts=3)

    async def _boom(seed):
        raise ValueError("empty_learning")

    pipe._stage1_read = _boom  # type: ignore[method-assign]

    async def _run():
        await _seed_queue(conn)
        return await pipe.tick(force=True)

    asyncio.run(_run())
    row = conn.rows["finding:r1:x"]
    assert row["status"] == "failed"
    assert row["attempts"] == 1


from reading_queue_fakes import ReadingQueueFakeMixin


class _FakeConn(ReadingQueueFakeMixin, _LegacyFakeConn):
    pass

pytestmark = pytest.mark.usefixtures("reading_dns")


# --- Wallet A refund: a turn refused before any reading gives its slot back ---
# Live 2026-09-23/24: six stance-phase capacity refusals spent the whole daily
# cap (orion:wp_read:wallet_a:count:<day> = 6) without one read, then every tick
# logged world_pulse_read_blocked reason=daily_cap for the rest of the day.

# Stance reason after GPU pool stage 3 (PR #2328): cortex-exec names the
# gateway's raw.error, stance fails, the turn orchestrator returns turn_deferred.
_POOL_STANCE_REASON = "stance_react_failed: agent=gpu_pool_unavailable:deadline"


def _patch_turn(monkeypatch, frames):
    async def _turn(**kwargs):
        return frames

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _turn)


def _tick(pipe, conn, *, force=True):
    async def _run():
        await _seed_queue(conn)
        return await pipe.tick(force=force)

    return asyncio.run(_run())


def _now():
    return datetime.now(timezone.utc)


def test_real_deferred_frame_refunds_wallet_a_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through _generate -> _reason_from_non_final_frame ->
    _stage1_read's ValueError: pins the label the refund keys on."""
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    bus.redis.store[_count_key()] = "2"
    prior = (_now() - timedelta(hours=10)).isoformat()
    bus.redis.store[wa.WALLET_A_COOLDOWN_KEY] = prior
    pipe = _pipeline(bus, conn, store, min_cooldown_sec=600.0, max_attempts=3)
    _patch_turn(monkeypatch, [{"type": "turn_deferred", "reason": _POOL_STANCE_REASON}])

    _tick(pipe, conn)

    assert bus.redis.store[_count_key()] == "2"
    # last_at goes back to the last debit that counted (dashboard stays honest).
    assert bus.redis.store[wa.WALLET_A_COOLDOWN_KEY] == prior
    # Retry spacing moves to its own key: floor (600s) after the refusal.
    wait = asyncio.run(wa.read_wallet_a_retry_wait(bus.redis, now=_now()))
    assert wait is not None and 590 <= wait <= 600
    # Admission failed before reading, so the seed keeps its attempt budget.
    assert conn.rows["finding:r1:x"]["attempts"] == 0
    assert conn.rows["finding:r1:x"]["status"] == "pending"


def test_real_turn_error_frame_keeps_wallet_a_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    bus.redis.store[_count_key()] = "2"
    pipe = _pipeline(bus, conn, store, min_cooldown_sec=600.0, max_attempts=3)
    _patch_turn(monkeypatch, [{"type": "turn_error", "error_code": "fcc_stream_stalled"}])

    _tick(pipe, conn)

    assert bus.redis.store[_count_key()] == "3"
    assert wa.WALLET_A_RETRY_NOT_BEFORE_KEY not in bus.redis.store


def test_refund_backoff_blocks_the_next_tick_then_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, min_cooldown_sec=600.0, max_attempts=5)
    _patch_turn(monkeypatch, [{"type": "turn_deferred", "reason": _POOL_STANCE_REASON}])

    _tick(pipe, conn)
    # Unforced tick right after: cooldown was refunded, so the backoff is what blocks.
    assert asyncio.run(pipe.tick()) == "refund_backoff"
    # Second consecutive refusal (forced past the backoff) doubles the wait.
    asyncio.run(pipe.tick(force=True))
    wait = asyncio.run(wa.read_wallet_a_retry_wait(bus.redis, now=_now()))
    assert 1190 <= wait <= 1200
    assert bus.redis.store[wa.WALLET_A_REFUND_STREAK_KEY] == "2"


def test_turn_that_reached_reader_resets_refund_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, min_cooldown_sec=600.0, max_attempts=5)
    _patch_turn(monkeypatch, [{"type": "turn_deferred", "reason": _POOL_STANCE_REASON}])
    _tick(pipe, conn)
    assert bus.redis.store[wa.WALLET_A_REFUND_STREAK_KEY] == "1"

    _patch_turn(monkeypatch, [{"type": "turn_error", "error_code": "fcc_stream_stalled"}])
    asyncio.run(pipe.tick(force=True))

    assert wa.WALLET_A_REFUND_STREAK_KEY not in bus.redis.store


def test_other_stance_deferrals_refund_but_reader_failures_do_not() -> None:
    cases = {
        "turn_deferred:stance_react_timeout": "0",
        "turn_deferred:stance_react_failed: agent=gateway_capacity_rejected:capacity_wait_budget_exhausted": "0",
        "turn_deferred:empty_imperative": "0",
        "turn_error:fcc_stream_stalled": "1",
        "stage1_turn_timeout": "1",
        "turn_exception:boom": "1",
        "unreadable handoff": "1",
    }
    for reason, expected in cases.items():
        bus = _FakeBus()
        conn = _FakeConn()
        store = InMemorySubstrateGraphStore()
        pipe = _pipeline(bus, conn, store, min_cooldown_sec=600.0, max_attempts=3)

        async def _boom(seed, _r=reason):
            raise ValueError(_r)

        pipe._stage1_read = _boom  # type: ignore[method-assign]
        _tick(pipe, conn)
        assert bus.redis.store[_count_key()] == expected, reason


def test_forced_tick_overrides_refund_backoff_and_real_turn_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, min_cooldown_sec=600.0, max_attempts=5)
    _patch_turn(monkeypatch, [{"type": "turn_deferred", "reason": _POOL_STANCE_REASON}])
    _tick(pipe, conn)
    assert wa.WALLET_A_RETRY_NOT_BEFORE_KEY in bus.redis.store

    _patch_turn(monkeypatch, [{"type": "turn_error", "error_code": "fcc_stream_stalled"}])
    assert asyncio.run(pipe.tick(force=True)) != "refund_backoff"

    assert wa.WALLET_A_RETRY_NOT_BEFORE_KEY not in bus.redis.store


# --- Read evidence: a Stage 1 turn is `done` only if it actually fetched the
# source. Live 2026-09-25: finding:60d59b10...:9b084fc0f1583da0 was marked done
# with zero tool calls; its handoff began with the text below.

_LIVE_HOLLOW_LEARNING = (
    "Metadata-only extraction; I did not fetch or read the article body this "
    "turn, and no claim about the page's content is grounded."
)


def _final_frame(learning: str, fetches=None, **extra_json):
    import json as _json

    body = {"what_i_learned": learning, "candidate_priors": [], **extra_json}
    frame = {"type": "final", "llm_response": "```json\n" + _json.dumps(body) + "\n```"}
    if fetches is not None:
        frame["harness_source_fetches"] = fetches
    return frame


def test_hollow_read_with_no_fetch_is_failed_not_done(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store, max_attempts=3)
    _patch_turn(monkeypatch, [_final_frame(_LIVE_HOLLOW_LEARNING, fetches=[])])

    assert _tick(pipe, conn) == "no_read_evidence"

    row = conn.rows["finding:r1:x"]
    assert row["status"] == "failed"  # terminal: the reader ran and did not read
    assert row["last_error"] == "no_read_evidence"
    assert row.get("handoff_json") is None
    assert store.snapshot().nodes == {}
    assert bus.journal == []
    # The turn reached the reader, so its Wallet A slot stays spent.
    assert bus.redis.store[_count_key()] == "1"


def test_fetch_of_a_different_site_is_not_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore())
    other = {"url": "https://coverage.example.org/story", "tool_name": "WebFetch", "content_chars": 5000}
    _patch_turn(monkeypatch, [_final_frame("Read a different outlet's story.", fetches=[other])])

    assert _tick(pipe, conn) == "no_read_evidence"
    assert conn.rows["finding:r1:x"]["last_error"] == "no_read_evidence"


def test_near_empty_fetch_result_is_not_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore())
    blocked = {"url": "https://ex.com/a", "tool_name": "WebFetch", "content_chars": 38}
    _patch_turn(monkeypatch, [_final_frame("Tried the page.", fetches=[blocked])])

    assert _tick(pipe, conn) == "no_read_evidence"
    assert conn.rows["finding:r1:x"]["last_error"] == "no_read_evidence:thin_fetch"


def test_fetch_of_the_sites_homepage_is_not_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore())
    home = {"url": "https://ex.com/", "tool_name": "WebFetch", "content_chars": 6000}
    _patch_turn(monkeypatch, [_final_frame("Article 404'd; read the homepage.", fetches=[home])])

    assert _tick(pipe, conn) == "no_read_evidence"
    assert conn.rows["finding:r1:x"]["last_error"] == "no_read_evidence"


def test_stage1_prompt_tells_the_reader_to_fetch() -> None:
    from scripts.world_pulse_read_pipeline import _build_stage1_prompt

    prompt = _build_stage1_prompt(_seed(), "tr-1")
    assert "Fetch the url below with WebFetch" in prompt
    assert "no successful fetch of this url is discarded" in prompt


def test_model_cannot_supply_its_own_read_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore())
    forged = [{"url": "https://ex.com/a", "tool_name": "WebFetch", "content_chars": 9000}]
    _patch_turn(
        monkeypatch,
        [_final_frame("I fetched it, honest.", fetches=[], read_evidence=forged)],
    )

    assert _tick(pipe, conn) == "no_read_evidence"
    assert conn.rows["finding:r1:x"]["status"] == "failed"


def test_unreported_fetches_are_a_retryable_infra_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A governor that predates HarnessRunV1.source_fetches sends no report:
    never treated as a read, but the seed is not burned either."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore(), max_attempts=3)
    _patch_turn(monkeypatch, [_final_frame("Read the page.", fetches=None)])

    assert _tick(pipe, conn) == "no_read_evidence"
    row = conn.rows["finding:r1:x"]
    assert row["last_error"] == "no_read_evidence:harness_unreported"
    assert row["status"] == "pending"


def test_real_fetch_marks_done_and_stores_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    store = InMemorySubstrateGraphStore()
    pipe = _pipeline(bus, conn, store)
    fetches = [
        {"url": "https://ex.com/a", "tool_name": "WebFetch", "content_chars": 2400},
        {"url": "https://elsewhere.example.net/x", "tool_name": "WebFetch", "content_chars": 900},
    ]
    _patch_turn(monkeypatch, [_final_frame("The page says packaging costs rose.", fetches=fetches)])

    assert _tick(pipe, conn) is None

    row = conn.rows["finding:r1:x"]
    assert row["status"] == "done"
    assert row["handoff_json"]["read_evidence"] == [
        {"url": "https://ex.com/a", "tool_name": "WebFetch", "content_chars": 2400}
    ]
    assert len(bus.journal) == 1


def test_injected_reader_without_evidence_is_still_gated() -> None:
    """The tick-level check holds even when _stage1_read is replaced."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore())
    bare = _handoff().model_copy(update={"read_evidence": []})

    async def _fake_read(seed):
        return bare

    pipe._stage1_read = _fake_read  # type: ignore[method-assign]
    assert _tick(pipe, conn) == "no_read_evidence"
    assert conn.rows["finding:r1:x"]["status"] == "failed"


# --- Stale digest items: skipped on the live loop, not by one-off SQL ---


def _stale_sql_calls(conn):
    return [
        args
        for sql, args in conn.executed
        if "kind = 'digest_item'" in sql and "SET status = 'skipped'" in sql
    ]


def test_tick_skips_stale_digest_items_with_configured_age() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    bus.redis.store[_count_key()] = "6"  # blocked: the sweep must still run
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore(), digest_item_max_age_days=5)

    assert asyncio.run(pipe.tick()) == "daily_cap"
    assert _stale_sql_calls(conn) == [(5 * 86400.0, "stale_digest_item")]


def test_zero_max_age_disables_the_stale_sweep() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, InMemorySubstrateGraphStore(), digest_item_max_age_days=0)
    asyncio.run(pipe.tick())
    assert _stale_sql_calls(conn) == []
