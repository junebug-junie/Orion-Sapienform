"""Stage 1 Hub loop: Wallet A gate, Concept Atlas materialize, journal — never Curiosity."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
)
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.world_pulse_read import wallet_a as wa
from orion.world_pulse_read.queue import enqueue_seeds
from scripts.curiosity_investigation import _COOLDOWN_KEY, _DAILY_COUNT_KEY_PREFIX
from scripts.world_pulse_read_pipeline import JOURNAL_WRITE_CHANNEL, WorldPulseReadPipeline

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

    text = asyncio.run(_run())
    assert text == '{"what_i_learned": "ok"}'
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
