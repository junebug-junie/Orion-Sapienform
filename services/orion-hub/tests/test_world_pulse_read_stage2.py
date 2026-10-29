"""Stage 2 Hub loop: Wallet B gate, handoff claim, round-trip ceiling — never Curiosity."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(autouse=True)
def held_model_boundary(monkeypatch):
    """These parsing/wallet tests exercise the relocated model boundary.

    Durable submission, waiting and RPC fencing have separate integration tests.
    """
    from scripts.reading_turn_listener import ReadingTurnListener
    from orion.schemas.reading_turn import ReadingRunBriefV1, ReadingTurnRequestV1
    from scripts.world_pulse_read_stage2 import GenerateOutcome

    async def generate(pipe, prompt, correlation_id, *, seed_id="test-seed", retrieval_query=None):
        listener = ReadingTurnListener(pipe._source_ref, pipe._step_relay_provider)
        listener.bus = pipe._bus
        listener.rpc_bus = pipe._harness_rpc_bus or pipe._bus
        result = await listener._execute(ReadingTurnRequestV1(
            run_id="test-run", correlation_id=correlation_id,
            brief=ReadingRunBriefV1(seed_id=seed_id, stage=2, prompt=prompt,
                session_id=pipe.session_id, timeout_sec=pipe.timeout_sec,
                fcc_model_label=pipe._fcc_model_label, retrieval_query=retrieval_query),
            gpu_lease={"lease_id": "test-hold", "generation": 1,
                       "role": "agent", "holder": "durable-runs:test-run"},
        ))
        return GenerateOutcome(result.text, result.error)

    monkeypatch.setattr(WorldPulseReadStage2Pipeline, "_generate", generate)

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.world_pulse_read import (
    WorldPulseReadHandoffV1,
    WorldPulseReadPriorCandidateV1,
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
    _build_stage2_prompt,
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
        if "SET stage2_status = 'skipped'" in sql_n:
            seed_id, reason = args[0], args[1]
            if seed_id in self.rows:
                self.rows[seed_id]["stage2_status"] = "skipped"
                self.rows[seed_id]["stage2_error"] = reason
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


def _fetched(url: str, chars: int = 1200) -> SourceFetchEvidenceV1:
    """What the governor reports for a WebFetch that returned the page."""
    return SourceFetchEvidenceV1(url=url, tool_name="WebFetch", content_chars=chars)


def _handoff(seed: WorldPulseReadSeedV1 | None = None) -> WorldPulseReadHandoffV1:
    seed = seed or _seed()
    return WorldPulseReadHandoffV1(
        seed_ref=seed,
        what_i_learned="Learned about packaging.",
        trace_id="tr-pipeline-1",
        created_at=NOW,
        read_evidence=[_fetched(seed.url)],
    )


def _linked_by_stage2(pipe, urls: list[str]) -> None:
    """The Stage 2 turn fetched a page that links ``urls`` -- the only way a
    model-named hop is allowed to re-enter Stage 1 (fetch_provenance.py)."""
    pipe._stage2_fetches = [SourceFetchEvidenceV1(
        url="https://ex.com/index-of-links", tool_name="WebFetch", content_chars=900,
        content_text="Related pages:\n" + "\n".join(f"- {u}" for u in urls),
    )]


def _result(*, urls: list[str] | None = None) -> WorldPulseReadStage2ResultV1:
    return WorldPulseReadStage2ResultV1(
        summary="Priors updated.",
        need_stage1_urls=urls or [],
        trace_id="tr-s2",
        created_at=NOW,
        seed_id="finding:r1:x",
    )


def test_stage2_stance_view_hides_the_reader_contract_only() -> None:
    """Live 2026-10-27/28, 6 of 6 Qwen3.6-35B stances on Stage 2 wrote these keys
    (summary in the imperative's slot, or the whole reading answer)."""
    from orion.world_pulse_read.reader_contract import stance_view_of_reading_prompt

    handoff = _handoff()
    prompt = _build_stage2_prompt(handoff, "tr-s2", prior_error="turn_exception:TimeoutError: ")
    view = stance_view_of_reading_prompt(prompt)
    assert view is not None
    for key in ('"summary"', "article_summary", "priors_tested", "need_stage1_urls", '"hops"', "tr-s2"):
        assert key in prompt, key
        assert key not in view, key
    assert "You are continuing a deliberate source reading" in view
    assert handoff.seed_ref.url in view
    assert "RETRY: the previous attempt" in view


def test_prompt_has_no_prior_write_instruction_when_no_candidates() -> None:
    """No bogus instruction to write a prior when there is nothing to write."""
    handoff = _handoff()
    assert handoff.candidate_priors == []
    prompt = _build_stage2_prompt(handoff, "tr-s2")
    assert "MERGE (p:Prior" not in prompt
    assert "WRITE THE CANDIDATE PRIORS" not in prompt
    assert "GRAPH.QUERY $ORION_CURIOSITY_GRAPH_OWN" not in prompt


def test_prompt_preserves_attributed_candidates_without_graph_write_instructions() -> None:
    seed = _seed()
    handoff = WorldPulseReadHandoffV1(
        seed_ref=seed,
        what_i_learned="Learned about packaging.",
        candidate_priors=[
            WorldPulseReadPriorCandidateV1(claim="Packaging costs are rising.", confidence=0.6)
        ],
        trace_id="tr-pipeline-1",
        created_at=NOW,
    )
    prompt = _build_stage2_prompt(handoff, "tr-s2")
    assert "Packaging costs are rising." in prompt
    assert handoff.trace_id in prompt
    assert "tr-s2" in prompt
    assert seed.seed_id in prompt
    assert seed.url in prompt
    assert "attributed candidates, not settled truth" in prompt
    assert "Do not write RDF, execute graph queries, or call Graphiti" in prompt
    for forbidden in ("MERGE (", "GRAPH.QUERY", "redis-cli", "ORION_CURIOSITY_GRAPH_"):
        assert forbidden not in prompt


def test_prompt_includes_all_candidate_priors_not_just_first() -> None:
    seed = _seed()
    handoff = WorldPulseReadHandoffV1(
        seed_ref=seed,
        what_i_learned="Learned about packaging.",
        candidate_priors=[
            WorldPulseReadPriorCandidateV1(claim="Claim one.", confidence=0.4),
            WorldPulseReadPriorCandidateV1(claim="Claim two.", confidence=0.7),
        ],
        trace_id="tr-pipeline-1",
        created_at=NOW,
    )
    prompt = _build_stage2_prompt(handoff, "tr-s2")
    assert "Claim one." in prompt
    assert "Claim two." in prompt


def _count_key_b() -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{wb.WALLET_B_COUNT_KEY_PREFIX}{today}"


def _pipeline(bus: _FakeBus, conn: _FakeConn, **over) -> WorldPulseReadStage2Pipeline:
    kwargs = dict(
        enabled=True,
        tick_interval_sec=60.0,
        window_start_hour=0,
        window_end_hour=0,
        timeout_sec=30.0,
        session_id="orion_world_pulse_read_stage2",
        llm_route="agent",
        timezone_name="UTC",
        max_round_trips=5,
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


def test_tick_when_wallet_b_disabled_does_not_claim() -> None:
    """No daily cap since 2026-09-28: a busy day does not block, the switch does."""
    bus = _FakeBus()
    conn = _FakeConn()
    bus.redis.store[_count_key_b()] = "99"
    pipe = _pipeline(bus, conn, enabled=False)

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick()

    assert asyncio.run(_run()) == "disabled"
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
        urls = [f"https://ex.com/more-{i}" for i in range(10)]
        _linked_by_stage2(pipe, urls)
        return _result(urls=urls)

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


def test_already_read_followup_is_passed_on_without_stopping_the_rest() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_round_trips=5)
    tried: list[str] = []

    async def _pass(handoff):
        urls = ["https://ex.com/read-before", "https://ex.com/new"]
        _linked_by_stage2(pipe, urls)
        return _result(urls=urls)

    async def _reenter(url, *, parent_seed):
        tried.append(url)
        return "already_read" if url.endswith("read-before") else None

    pipe._stage2_pass = _pass  # type: ignore[method-assign]
    pipe._reenter_stage1 = _reenter  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert tried == ["https://ex.com/read-before", "https://ex.com/new"]
    assert pipe.last_round_trips == 1


def test_reentry_stops_when_wallet_a_blocked() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_round_trips=5)
    reentered: list[str] = []

    async def _pass(handoff):
        urls = ["https://ex.com/more-0", "https://ex.com/more-1"]
        _linked_by_stage2(pipe, urls)
        return _result(urls=urls)

    async def _reenter(url, *, parent_seed):
        if reentered:
            return "refund_backoff"
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


def test_default_reentry_enqueues_reading_seed_with_world_pulse_lineage() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _pass(handoff):
        _linked_by_stage2(pipe, ["https://ex.com/follow"])
        return _result(urls=["https://ex.com/follow"])

    pipe._stage2_pass = _pass  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert pipe.last_round_trips == 1
    follow = [r for r in conn.rows.values() if r["url"] == "https://ex.com/follow"]
    assert len(follow) == 1
    assert follow[0]["kind"] == "reading"
    assert follow[0]["request_json"]["requested_by"] == "world_pulse"
    assert follow[0]["request_json"]["parent_request_id"] == str(conn.rows["finding:r1:x"]["request_id"])
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


# --- Schema/prompt agreement: real rejected live payloads must now validate
# (2026-09-19 bug: prompt asked for these fields, schema forbade them; 3 of
# the last 8 live Stage 2 completions were thrown away) ---


def test_as_stage2_result_accepts_live_rejected_payload_2026_09_15() -> None:
    """2026-09-15 00:51 stage2_error: candidate_priors, concept_candidates,
    open_threads, verification_path, hops, stages_completable all rejected."""
    from scripts.world_pulse_read_stage2 import _as_stage2_result

    raw = {
        "summary": "Checked the XOC platform expansion claim.",
        "candidate_priors": [{"claim": "A Gamers Nexus source is credible.", "confidence": 0.5}],
        "concept_candidates": [{"label": "xoc-gpu-platform", "definition": "A GPU platform.", "link_hints": []}],
        "open_threads": ["XOC expansion unresolved."],
        "verification_path": ["Fetch a YouTube transcript."],
        "hops": ["web.fetch https://www.youtube.com/... -> 0 results"],
        "stages_completable": False,
    }
    result = _as_stage2_result(raw, fallback_trace="tr-fallback", seed_id="finding:x")
    assert result.summary == raw["summary"]
    assert result.candidate_priors[0].claim == "A Gamers Nexus source is credible."
    assert result.concept_candidates[0].label == "xoc-gpu-platform"
    assert result.open_threads == ["XOC expansion unresolved."]
    # verification_path and stages_completable are not modeled fields --
    # dropped, not rejected.
    assert not hasattr(result, "verification_path")
    assert not hasattr(result, "stages_completable")


def test_as_stage2_result_accepts_live_rejected_payload_2026_09_14_priors_tested() -> None:
    """2026-09-14 19:06 stage2_error: priors_tested rejected."""
    from scripts.world_pulse_read_stage2 import _as_stage2_result

    raw = {
        "summary": "Re-verified the six-component claim.",
        "priors_tested": [
            {"claim_ref": "six-component claim", "verdict": "supported", "why": "corroboration in source"}
        ],
    }
    result = _as_stage2_result(raw, fallback_trace="tr-fallback", seed_id="finding:x")
    assert result.priors_tested[0].claim_ref == "six-component claim"
    assert result.priors_tested[0].verdict == "supported"
    assert result.priors_tested[0].why == "corroboration in source"


def test_as_stage2_result_accepts_live_rejected_payload_2026_09_14_stance() -> None:
    """2026-09-14 16:55 stage2_error: stage2_stance (an unmodeled dict) rejected."""
    from scripts.world_pulse_read_stage2 import _as_stage2_result

    raw = {
        "summary": "Stance check complete.",
        "stage2_stance": {"verified_this_turn_via_": "web search", "unresolved": ["thing"]},
    }
    result = _as_stage2_result(raw, fallback_trace="tr-fallback", seed_id="finding:x")
    assert result.summary == "Stance check complete."
    assert not hasattr(result, "stage2_stance")


def test_as_stage2_result_logs_and_counts_dropped_unknown_keys(caplog: pytest.LogCaptureFixture) -> None:
    from scripts.world_pulse_read_stage2 import _as_stage2_result

    dropped: list[list[str]] = []
    with caplog.at_level("WARNING", logger="orion-hub.world_pulse_read_stage2"):
        _as_stage2_result(
            {"summary": "ok", "stages_completable": True, "verification_path": ["x"]},
            fallback_trace="tr",
            seed_id="finding:x",
            on_dropped=dropped.append,
        )
    assert dropped == [["stages_completable", "verification_path"]]
    assert any("world_pulse_read_stage2_unknown_keys_dropped" in r.message for r in caplog.records)


def test_prompt_lists_exactly_the_fields_the_schema_accepts() -> None:
    """The regression this whole bug was: prompt promised fields the schema
    forbade. Assert the skeleton names the real fields, not stale ones."""
    handoff = _handoff()
    prompt = _build_stage2_prompt(handoff, "tr-s2")
    for field in ("priors_tested", "candidate_priors", "concept_candidates", "open_threads", "hops", "need_stage1_urls"):
        assert f'"{field}"' in prompt


# --- Bounded retry: a transient failure returns to pending; a non-transient
# one stays terminal on the first try; exhausted attempts stay terminal too
# (2026-09-19 bug: mark_stage2_failed had no path back to pending, so Stage 2
# starved on live turn_deferred/timeout noise) ---


def test_transient_stage2_failure_is_retried_until_exhausted() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_attempts=3)

    async def _boom(handoff):
        raise ValueError("turn_deferred:stance_react_failed: exec result missing thought payload")

    pipe._stage2_pass = _boom  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        results = []
        for _ in range(3):
            results.append(await pipe.tick(force=True))
        return results

    results = asyncio.run(_run())
    row = conn.rows["finding:r1:x"]
    # Attempt 1 and 2 retry (attempts < 3); attempt 3 exhausts and goes terminal.
    assert results == ["parse_failed", "parse_failed", "parse_failed"]
    assert row["stage2_attempts"] == 3
    assert row["stage2_status"] == "failed"
    assert row["stage2_error"].startswith("turn_deferred:")


def test_transient_stage2_failure_reclaimable_between_retries() -> None:
    """A retried seed goes back to `pending`, not `claimed` -- the next tick
    can claim it again exactly like a fresh seed."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_attempts=3)

    async def _boom(handoff):
        raise ValueError("stage2_turn_timeout")

    pipe._stage2_pass = _boom  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        await pipe.tick(force=True)
        row = conn.rows["finding:r1:x"]
        return row["stage2_status"], row["stage2_attempts"], row["stage2_claimed_at"]

    status, attempts, claimed_at = asyncio.run(_run())
    assert status == "pending"
    assert attempts == 1
    assert claimed_at is None


def test_non_transient_stage2_failure_is_terminal_on_first_try() -> None:
    """A schema/parse failure is the seed's fault, not the turn's -- must not
    burn two more wallet slots retrying it."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_attempts=3)

    async def _boom(handoff):
        raise ValueError("1 validation error for WorldPulseReadStage2ResultV1")

    pipe._stage2_pass = _boom  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    asyncio.run(_run())
    row = conn.rows["finding:r1:x"]
    assert row["stage2_status"] == "failed"
    assert row["stage2_attempts"] == 1


def test_max_attempts_one_is_legacy_terminal_on_first_failure() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_attempts=1)

    async def _boom(handoff):
        raise ValueError("turn_deferred:stance_react_failed: x")

    pipe._stage2_pass = _boom  # type: ignore[method-assign]

    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    asyncio.run(_run())
    row = conn.rows["finding:r1:x"]
    assert row["stage2_status"] == "failed"
    assert row["stage2_attempts"] == 1


from reading_queue_fakes import ReadingQueueFakeMixin


class _FakeConn(ReadingQueueFakeMixin, _LegacyFakeConn):
    pass

pytestmark = pytest.mark.usefixtures("reading_dns")


# --- Wallet B refund: a Stage 2 turn refused before any reading gives its slot back ---


def _patch_turn(monkeypatch, frames):
    async def _turn(**kwargs):
        return frames

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _turn)


def _stage2_tick(conn, pipe):
    async def _run():
        await _ready_stage2(conn)
        return await pipe.tick(force=True)

    return asyncio.run(_run())


def test_stage2_real_deferred_frame_refunds_wallet_b(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    bus.redis.store[_count_key_b()] = "4"
    pipe = _pipeline(bus, conn, max_attempts=3)
    _patch_turn(
        monkeypatch,
        [{"type": "turn_deferred", "reason": "stance_react_failed: agent=gpu_pool_unavailable:deadline"}],
    )

    _stage2_tick(conn, pipe)

    assert bus.redis.store[_count_key_b()] == "4"
    assert wb.WALLET_B_COOLDOWN_KEY not in bus.redis.store  # no prior debit -> cleared
    assert wb.WALLET_B_RETRY_NOT_BEFORE_KEY in bus.redis.store
    assert conn.rows["finding:r1:x"]["stage2_attempts"] == 0


def test_stage2_real_turn_error_keeps_wallet_b_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    bus.redis.store[_count_key_b()] = "4"
    pipe = _pipeline(bus, conn, max_attempts=3)
    _patch_turn(monkeypatch, [{"type": "turn_error", "error_code": "fcc_stream_stalled"}])

    _stage2_tick(conn, pipe)

    assert bus.redis.store[_count_key_b()] == "5"
    assert wb.WALLET_B_RETRY_NOT_BEFORE_KEY not in bus.redis.store


def test_stage2_invalid_handoff_is_not_debited() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)

    async def _run():
        await _ready_stage2(conn)
        conn.rows["finding:r1:x"]["handoff_json"] = {"not": "a handoff"}
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) == "handoff_invalid"
    assert _count_key_b() not in bus.redis.store
    assert wb.WALLET_B_COOLDOWN_KEY not in bus.redis.store


def test_stage2_turn_that_reached_reader_resets_refund_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn, max_attempts=5)
    _patch_turn(
        monkeypatch,
        [{"type": "turn_deferred", "reason": "stance_react_failed: agent=gpu_pool_unavailable:deadline"}],
    )
    _stage2_tick(conn, pipe)
    assert bus.redis.store[wb.WALLET_B_REFUND_STREAK_KEY] == "1"

    _patch_turn(monkeypatch, [{"type": "turn_error", "error_code": "fcc_stream_stalled"}])
    asyncio.run(pipe.tick(force=True))

    assert wb.WALLET_B_REFUND_STREAK_KEY not in bus.redis.store
    assert wb.WALLET_B_RETRY_NOT_BEFORE_KEY not in bus.redis.store


# --- Stage 2 never builds on a Stage 1 handoff that has no read evidence ---


def _live_hollow_handoff(seed: WorldPulseReadSeedV1) -> WorldPulseReadHandoffV1:
    """Shape of the live 2026-09-25 networkworld handoff: schema-valid, no
    read_evidence (every row written before this fix looks like this)."""
    return WorldPulseReadHandoffV1(
        seed_ref=seed,
        what_i_learned=(
            "Metadata-only extraction; I did not fetch or read the article body "
            "this turn, and no claim about the page's content is grounded."
        ),
        trace_id="tr-hollow",
        created_at=NOW,
    )


def test_stage2_skips_unread_handoff_before_wallet_b_debit() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)
    called: list = []

    async def _pass(handoff):
        called.append(handoff)
        return _result()

    pipe._stage2_pass = _pass  # type: ignore[method-assign]

    async def _run():
        seed = _seed()
        await enqueue_seeds(conn, [seed])
        await mark_seed_done(
            conn, seed.seed_id, trace_id="tr-hollow", handoff=_live_hollow_handoff(seed)
        )
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) == "no_read_evidence"
    row = conn.rows["finding:r1:x"]
    assert row["stage2_status"] == "skipped"
    assert row["stage2_error"] == "no_read_evidence"
    assert called == []
    assert _count_key_b() not in bus.redis.store
    assert bus.journal == []
    # stage2_started (published at claim) is closed by a terminal event.
    stages = [e.payload.get("stage") for c, e in bus.published if c != JOURNAL_WRITE_CHANNEL]
    assert stages[-1] == "stage2_failed"


def test_reentry_passes_on_an_already_read_url_even_when_wallet_a_is_backing_off() -> None:
    bus = _FakeBus()
    conn = _FakeConn()
    bus.redis.store[wa.WALLET_A_RETRY_NOT_BEFORE_KEY] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat()
    pipe = _pipeline(bus, conn)

    async def _run():
        await _ready_stage2(conn)
        conn.rows["finding:r1:x"]["handoff_json"] = {
            "read_evidence": [{"tool_name": "WebFetch", "url": "https://ex.com/a", "content_chars": 900}]
        }
        read_before = await pipe._reenter_stage1("https://ex.com/a", parent_seed=_seed())
        fresh = await pipe._reenter_stage1("https://ex.com/new", parent_seed=_seed())
        return read_before, fresh

    assert asyncio.run(_run()) == ("already_read", "refund_backoff")


def test_stage2_turn_recalls_about_title_and_stage1_claim(monkeypatch) -> None:
    """Recall retrieval design phase 3: "<source title> — <stage-1 claim>", not the stage-2
    prompt that embeds the whole stage-1 handoff JSON."""
    bus, conn = _FakeBus(), _FakeConn()
    pipe = _pipeline(bus, conn)
    captured: dict = {}

    async def _turn(**kwargs):
        captured.update(kwargs)
        return [{"type": "final", "llm_response": "{}"}]

    monkeypatch.setattr("orion.hub.turn_orchestrator.execute_unified_turn", _turn)
    try:
        asyncio.run(pipe._stage2_pass(_handoff()))
    except Exception:  # noqa: BLE001 -- only the turn's arguments matter here
        pass

    assert captured["retrieval_query"] == "A — Learned about packaging."


# --- created_at is the server clock, never the model's text (live 2026-10-10:
# model-written round future times reached journal rows and wp-read nodes). ---


def test_stage2_model_written_future_created_at_is_overwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = _pipeline(_FakeBus(), _FakeConn())
    future = (datetime.now(timezone.utc) + timedelta(hours=4)).replace(minute=30, second=0, microsecond=0)
    body = {"summary": "Tested the packaging prior.", "created_at": future.isoformat()}
    _patch_turn(monkeypatch, [{"type": "final", "llm_response": "```json\n" + json.dumps(body) + "\n```"}])

    before = datetime.now(timezone.utc)
    result = asyncio.run(pipe._stage2_pass(_handoff()))
    after = datetime.now(timezone.utc)

    assert before <= result.created_at <= after


def test_stage2_prompt_no_longer_asks_the_model_for_created_at() -> None:
    from scripts.world_pulse_read_stage2 import _build_stage2_prompt

    prompt = _build_stage2_prompt(_handoff(), "tr-1")
    assert '"created_at":' not in prompt
    assert "created_at are forced server-side" in prompt


# --- Hop provenance (live 2026-10-13): Stage 2 of a CoreWeave read asked Stage 1
# to read three sibling URLs it had guessed; each was a 404 and each was marked
# `done`. A hop must come from a URL Orion actually fetched or saw linked.

_LIVE_404_NOTICE = (
    "The server returned HTTP 404 Not Found.\n\nThe response body was not retrieved. "
    "If this URL requires authentication, use an authenticated tool (e.g. `gh` for "
    "GitHub, or an MCP-provided fetch tool) instead of WebFetch."
)


def _retained_parent(conn: _FakeConn, seed: WorldPulseReadSeedV1, *texts: str) -> WorldPulseReadHandoffV1:
    import hashlib

    evidence = []
    for text in texts:
        sha = hashlib.sha256(text.encode()).hexdigest()
        conn.__dict__.setdefault("snapshots", {})[sha] = {"content": text, "first_source": seed.url}
        evidence.append(SourceFetchEvidenceV1(url=seed.url, tool_name="WebFetch",
                                              content_chars=len(text), content_sha256=sha))
    return WorldPulseReadHandoffV1(seed_ref=seed, what_i_learned="Learned.", trace_id="tr-pipeline-1",
                                   created_at=NOW, read_evidence=evidence)


def _hop_run(conn, pipe, handoff, urls, *, stage2_fetches=()):
    tried: list[str] = []

    async def _pass(_handoff):
        pipe._stage2_fetches = list(stage2_fetches)
        return _result(urls=urls)

    async def _reenter(url, *, parent_seed):
        tried.append(url)
        return None

    pipe._stage2_pass = _pass  # type: ignore[method-assign]
    pipe._reenter_stage1 = _reenter  # type: ignore[method-assign]

    async def _run():
        seed = handoff.seed_ref
        await enqueue_seeds(conn, [seed])
        await mark_seed_done(conn, seed.seed_id, trace_id="tr-pipeline-1", handoff=handoff)
        return await pipe.tick(force=True)

    return asyncio.run(_run()), tried


def test_invented_hop_url_is_refused_and_logged(caplog) -> None:
    conn = _FakeConn()
    pipe = _pipeline(_FakeBus(), conn)
    page = "CoreWeave offers bare-metal NVIDIA GPUs for AI training. " * 5
    handoff = _retained_parent(conn, _seed(), page)
    guessed = ["https://www.coreweave.com/architecture", "https://www.coreweave.com/infrastructure"]
    with caplog.at_level("INFO"):
        outcome, tried = _hop_run(conn, pipe, handoff, guessed)
    assert outcome is None
    assert tried == []
    assert pipe.last_round_trips == 0
    assert conn.rows["finding:r1:x"]["stage2_status"] == "done"
    refused = [r.getMessage() for r in caplog.records if "reentry_refused" in r.getMessage()]
    assert len(refused) == 2 and all("reason=url_not_in_fetched_content" in m for m in refused)


def test_hop_url_linked_in_parent_retained_text_is_followed() -> None:
    conn = _FakeConn()
    pipe = _pipeline(_FakeBus(), conn)
    page = ("The full benchmark is described at https://www.ex.com/benchmarks/gb200/ "
            "(see methodology). " * 3)
    handoff = _retained_parent(conn, _seed(), page)
    # The model's copy differs only in www / trailing slash; a guessed sibling does not.
    outcome, tried = _hop_run(conn, pipe, handoff,
                              ["https://ex.com/benchmarks/gb200", "https://ex.com/benchmarks/h100"])
    assert outcome is None
    assert tried == ["https://ex.com/benchmarks/gb200"]
    assert pipe.last_round_trips == 1


def test_hop_reached_by_a_stage2_fetch_is_followed_but_not_one_that_404d() -> None:
    conn = _FakeConn()
    pipe = _pipeline(_FakeBus(), conn)
    handoff = _retained_parent(conn, _seed(), "A real page about packaging costs. " * 8)
    fetched = [
        SourceFetchEvidenceV1(url="https://ex.com/real", tool_name="WebFetch", content_chars=900,
                              content_text="Packaging costs rose 12% in Q3. " * 10),
        SourceFetchEvidenceV1(url="https://ex.com/gone", tool_name="WebFetch",
                              content_chars=len(_LIVE_404_NOTICE), content_text=_LIVE_404_NOTICE),
    ]
    outcome, tried = _hop_run(conn, pipe, handoff, ["https://ex.com/gone", "https://ex.com/real"],
                              stage2_fetches=fetched)
    assert outcome is None
    assert tried == ["https://ex.com/real"]


def test_hop_from_a_dead_parent_read_is_never_attempted() -> None:
    """A parent stored `done` on nothing but a 404 notice (pre-fix rows): no Stage 2
    turn, no claims, no hop chain."""
    bus = _FakeBus()
    conn = _FakeConn()
    pipe = _pipeline(bus, conn)
    handoff = _retained_parent(conn, _seed(), _LIVE_404_NOTICE)
    ran: list[str] = []

    async def _pass(_handoff):
        ran.append("pass")
        return _result(urls=["https://ex.com/a/more"])

    pipe._stage2_pass = _pass  # type: ignore[method-assign]

    async def _run():
        await enqueue_seeds(conn, [handoff.seed_ref])
        await mark_seed_done(conn, handoff.seed_ref.seed_id, trace_id="tr-pipeline-1", handoff=handoff)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) == "no_read_evidence:http_error"
    assert ran == []
    row = conn.rows["finding:r1:x"]
    assert row["stage2_status"] == "skipped"
    assert row["stage2_error"] == "no_read_evidence:http_error"
    assert not [r for r in conn.rows.values() if r["url"] == "https://ex.com/a/more"]
    assert bus.redis.store.get(_count_key_b()) in (None, "0")  # no Wallet B debit


def test_stage2_turn_fetches_reach_the_hop_gate_through_the_real_pass() -> None:
    """The production _stage2_pass -> _generate path carries the turn's own fetches
    (ReadingTurnResultV1.source_fetches) to the hop gate; no test-side injection."""
    import json as _json

    conn = _FakeConn()
    pipe = _pipeline(_FakeBus(), conn)
    handoff = _retained_parent(conn, _seed(), "A real page about packaging costs. " * 8)
    body = {"summary": "Priors updated.", "need_stage1_urls": ["https://ex.com/linked", "https://ex.com/guessed"]}
    linked = SourceFetchEvidenceV1(url="https://ex.com/related", tool_name="WebFetch", content_chars=400,
                                   content_text="Further reading: https://ex.com/linked " * 5)

    async def _generate(prompt, correlation_id, *, seed_id, retrieval_query=None):
        return GenerateOutcome("```json\n" + _json.dumps(body) + "\n```", None, None,
                               source_fetches=[linked])

    pipe._generate = _generate  # type: ignore[method-assign]
    tried: list[str] = []

    async def _reenter(url, *, parent_seed):
        tried.append(url)
        return None

    pipe._reenter_stage1 = _reenter  # type: ignore[method-assign]

    async def _run():
        await enqueue_seeds(conn, [handoff.seed_ref])
        await mark_seed_done(conn, handoff.seed_ref.seed_id, trace_id="tr-pipeline-1", handoff=handoff)
        return await pipe.tick(force=True)

    assert asyncio.run(_run()) is None
    assert tried == ["https://ex.com/linked"]
    assert pipe.hops_refused_total == {"url_not_in_fetched_content": 1}


# --- huge-page Stage 2 (seed c31227dc, 2026-10-14) ---


def test_prompt_states_webfetch_limit_once():
    from orion.world_pulse_read.fetch_text import WEBFETCH_LIMIT_NOTE

    prompt = _build_stage2_prompt(_handoff(), "tr-s2")
    assert prompt.count(WEBFETCH_LIMIT_NOTE) == 1
    assert "RETRY:" not in prompt


@pytest.mark.parametrize("error", [
    "turn_exception:TimeoutError: ", "stage2_turn_timeout", "turn_error:fcc_timeout",
    "turn_error:fcc_stream_stalled", "workflow_deadline",
])
def test_retry_after_timeout_tells_model_not_to_fetch(error):
    prompt = _build_stage2_prompt(_handoff(), "tr-s2", prior_error=error)
    assert "RETRY:" in prompt and "Do not call WebFetch or WebSearch" in prompt
    assert error.split(":")[-1].strip() in prompt


@pytest.mark.parametrize("error", [None, "", "interrupted:process_restart", "parse_failed", "blank_final_response",
                                   "reading_binding_unavailable:TimeoutError"])
def test_retry_note_absent_for_non_timeout_failures(error):
    assert "RETRY:" not in _build_stage2_prompt(_handoff(), "tr-s2", prior_error=error)


def test_stage2_pass_builds_prompt_with_the_seeds_prior_error():
    pipe = object.__new__(WorldPulseReadStage2Pipeline)
    pipe._claim_context = None
    pipe._prior_stage2_error = "turn_exception:TimeoutError: "
    pipe._stage2_fetches = []
    seen = {}

    async def generate(prompt, corr, *, seed_id, retrieval_query=None):
        seen["prompt"] = prompt
        return GenerateOutcome("")

    pipe._generate = generate
    with pytest.raises(ValueError):
        asyncio.run(pipe._stage2_pass(_handoff()))
    assert "RETRY:" in seen["prompt"]


def test_prior_error_reads_the_seed_row_and_fails_open():
    pipe = object.__new__(WorldPulseReadStage2Pipeline)

    class Conn:
        async def fetchval(self, sql, seed_id):
            assert "stage2_error" in sql and seed_id == "s1"
            return "turn_exception:TimeoutError: "

    async def with_conn(fn):
        return await fn(Conn())

    pipe._with_conn = with_conn
    assert asyncio.run(pipe._prior_error("s1")) == "turn_exception:TimeoutError: "

    async def broken(fn):
        raise RuntimeError("db down")

    pipe._with_conn = broken
    assert asyncio.run(pipe._prior_error("s1")) is None
