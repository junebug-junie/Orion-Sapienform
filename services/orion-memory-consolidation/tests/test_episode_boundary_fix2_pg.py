"""Boundary Fix 2 regression + shadow episodes, against a real (disposable) Postgres.

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL to a throwaway server's admin DSN,
e.g. postgresql://postgres:test@127.0.0.1:55499/postgres. Each test creates and
drops its own database. Never point this at production.

The fake classifier below reproduces the live defect faithfully: it scores a
turn against ``prior_turns[-1]``, and a turn compared with ITSELF reads "no
boundary" (low score). On 2026-09-28 every closing turn was scored 0.96-1.00 in
the window that it closed and 0.004-0.685 in chat_history_log -- because
sql-writer published each turn twice and the second copy was classified against
a window that already held it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
MIGRATIONS = [
    REPO_ROOT / "services" / "orion-sql-db" / "manual_migration_memory_consolidation_v1.sql",
    REPO_ROOT / "services" / "orion-sql-db" / "manual_migration_memory_episode_v1.sql",
]
ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    spec = importlib.util.spec_from_file_location(name, SERVICE_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load("app/worker.py", "mc_worker_fix2")
from app.episode_shadow import EpisodeShadowStore  # noqa: E402
from app.window_state import WindowStore  # noqa: E402

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1  # noqa: E402


async def _make_db(*, with_episode_migration: bool = True):
    import asyncpg

    name = f"memep_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'CREATE DATABASE "{name}"')
    await admin.close()
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
    for path in MIGRATIONS if with_episode_migration else MIGRATIONS[:1]:
        await pool.execute(path.read_text())
    return name, pool


async def _drop_db(name: str, pool) -> None:
    import asyncpg

    await pool.close()
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    await admin.close()


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel, env):
        self.published.append((channel, env))

    def patches(self, corr: str) -> list[dict]:
        return [
            env.payload["spark_meta"]
            for ch, env in self.published
            if ch.endswith("spark_meta:patch") and env.payload.get("correlation_id") == corr
        ]

    def closed_events(self) -> list[dict]:
        return [env.payload for ch, env in self.published if ch == "orion:memory:episode:closed"]


class _Runner:
    def __init__(self) -> None:
        self.closed: list[dict] = []

    async def consolidate_window(self, window, *, bus):
        self.closed.append(window)


def _classifier(first_pass_scores: dict[str, float]):
    calls: list[tuple[str, str | None]] = []

    async def _fake(bus, *, turn, prior_turns, settings):
        prior = prior_turns[-1]["correlation_id"] if prior_turns else None
        calls.append((turn.correlation_id, prior))
        self_compare = prior == turn.correlation_id
        score = 0.01 if self_compare else first_pass_scores.get(turn.correlation_id, 0.1)
        return {
            "conversation_boundary_score": score,
            "memory_significance_score": 0.5,
            "memory_classify_status": "ok",
            "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
            "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.5, "baseline_mode": "prior_turn"},
        }

    return _fake, calls


T0 = datetime(2026, 9, 28, 12, 26, tzinfo=timezone.utc)  # 06:26 MDT


def _env(corr: str, *, at: datetime, phase: str | None, prompt: str = "p", response: str = "r") -> BaseEnvelope:
    meta = {"conversation_phase": {"phase_change": phase, "delta_user_seconds": None, "crossed_day": False}} if phase else {}
    turn = MemoryTurnPersistedV1(
        correlation_id=corr, prompt=prompt, response=response, spark_meta=meta, created_at=at
    )
    return BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="t", node="t"),
        payload=turn.model_dump(mode="json"),
    )


async def _deliver_twice(env, *, bus, window_store, runner, episode_store):
    """sql-writer's double publish: the same turn arrives twice, back to back."""
    for _ in range(2):
        await worker.handle_memory_turn_persisted(
            env, bus=bus, window_store=window_store, suggest_runner=runner, episode_store=episode_store
        )


async def _all_window_entries(pool) -> list[dict]:
    rows = await pool.fetch("SELECT turn_correlation_ids FROM memory_consolidation_windows ORDER BY created_at")
    out: list[dict] = []
    for r in rows:
        out.extend(json.loads(r["turn_correlation_ids"]))
    return out


@pytest.mark.asyncio
async def test_window_score_equals_chat_log_score_for_every_turn(monkeypatch):
    """Fix 2: the score that decides closing is the score persisted for the turn."""
    corrs = [str(uuid.uuid4()) for _ in range(4)]
    first = {corrs[0]: 0.11, corrs[1]: 0.98, corrs[2]: 0.30, corrs[3]: 0.97}
    fake, calls = _classifier(first)
    monkeypatch.setattr(worker, "classify_turn", fake)
    name, pool = await _make_db()
    try:
        bus, runner = _Bus(), _Runner()
        ws, es = WindowStore(pool), EpisodeShadowStore(pool, worker.settings)
        for i, c in enumerate(corrs):
            await _deliver_twice(
                _env(c, at=T0 + timedelta(minutes=5 * i), phase=None), bus=bus, window_store=ws, runner=runner, episode_store=es
            )
        # Each turn classified exactly once, never against itself.
        assert [c for c, _ in calls] == corrs
        assert all(prior != c for c, prior in calls)
        for c in corrs:
            patches = bus.patches(c)
            assert len(patches) == 1, f"{c}: {len(patches)} spark_meta patches (chat_history_log keeps the last)"
            chat_log_score = patches[-1]["conversation_boundary_score"]
            window_scores = {e["conversation_boundary_score"] for e in await _all_window_entries(pool) if e["correlation_id"] == c}
            assert window_scores == {chat_log_score}, (c, window_scores, chat_log_score)
        # Legacy closing unchanged: unknown phase + >=0.85 closes (turns 1 and 3).
        assert len(runner.closed) == 2
        reasons = await pool.fetch(
            "SELECT close_reason, boundary_score_at_close FROM memory_consolidation_windows WHERE status='closed' ORDER BY closed_at"
        )
        assert [(r["close_reason"], r["boundary_score_at_close"]) for r in reasons] == [
            ("legacy:unknown_phase+llm", 0.98),
            ("legacy:unknown_phase+llm", 0.97),
        ]
    finally:
        await _drop_db(name, pool)


@pytest.mark.asyncio
async def test_shadow_rule3_keeps_a_stamped_morning_as_one_episode(monkeypatch):
    """Austin-shaped: short pauses and a resumed_thread with a low judge score
    stay one episode; the next morning's turn closes it with the lag recorded.
    Live windows are untouched by the shadow decision."""
    corrs = [str(uuid.uuid4()) for _ in range(6)]
    plan = [
        (corrs[0], T0, "next_day", 0.10, "hi", "hello"),
        (corrs[1], T0 + timedelta(minutes=5), "short_pause", 0.20, "which queue?", "the reading queue"),
        (corrs[2], T0 + timedelta(minutes=33), "short_pause", 0.20, "work travel", "ok"),
        (corrs[3], T0 + timedelta(hours=2, minutes=19), "resumed_thread", 0.50, "away from home", "oh"),
        (corrs[4], T0 + timedelta(hours=3, minutes=28), "resumed_thread", 0.30, "Run github compactor.", "Workflow: GitHub Compactor"),
        (corrs[5], T0 + timedelta(hours=21, minutes=35), "next_day", 0.20, "sup", "hey"),
    ]
    fake, _ = _classifier({c: s for c, _, _, s, _, _ in plan})
    monkeypatch.setattr(worker, "classify_turn", fake)
    name, pool = await _make_db()
    try:
        bus, runner = _Bus(), _Runner()
        ws, es = WindowStore(pool), EpisodeShadowStore(pool, worker.settings)
        for c, at, phase, _, p, r in plan:
            await _deliver_twice(_env(c, at=at, phase=phase, prompt=p, response=r), bus=bus, window_store=ws, runner=runner, episode_store=es)
        events = bus.closed_events()
        assert len(events) == 1
        ev = events[0]
        assert ev["turn_ids"] == corrs[:5]
        assert ev["close_reason"] == "v2:phase_next_day"
        assert ev["phase_at_close"] == "next_day"
        assert ev["juniper_turn_count"] == 4 and ev["command_turn_count"] == 1
        assert ev["close_lag_sec"] == pytest.approx((timedelta(hours=21, minutes=35) - timedelta(hours=3, minutes=28)).total_seconds())
        assert ev["episode_status"] == "closed"
        row = await pool.fetchrow("SELECT closed_event_published_at FROM memory_episode_shadow WHERE episode_id=$1", ev["episode_id"])
        assert row["closed_event_published_at"] is not None
        open_rows = await pool.fetch("SELECT turns FROM memory_episode_shadow WHERE status='open'")
        assert len(open_rows) == 1
        assert [t["correlation_id"] for t in json.loads(open_rows[0]["turns"])] == [corrs[5]]
        # Legacy windows: next_day needs >=0.70, never reached -> the live path closed nothing.
        assert runner.closed == []
    finally:
        await _drop_db(name, pool)


@pytest.mark.asyncio
async def test_resumed_thread_with_a_confident_judge_splits(monkeypatch):
    corrs = [str(uuid.uuid4()) for _ in range(2)]
    fake, _ = _classifier({corrs[0]: 0.1, corrs[1]: 0.97})
    monkeypatch.setattr(worker, "classify_turn", fake)
    name, pool = await _make_db()
    try:
        bus, runner = _Bus(), _Runner()
        ws, es = WindowStore(pool), EpisodeShadowStore(pool, worker.settings)
        await _deliver_twice(_env(corrs[0], at=T0, phase="next_day"), bus=bus, window_store=ws, runner=runner, episode_store=es)
        await _deliver_twice(_env(corrs[1], at=T0 + timedelta(hours=1, minutes=46), phase="resumed_thread"), bus=bus, window_store=ws, runner=runner, episode_store=es)
        events = bus.closed_events()
        assert [e["close_reason"] for e in events] == ["v2:resumed_thread+llm"]
        assert events[0]["boundary_score_at_close"] == 0.97
    finally:
        await _drop_db(name, pool)


@pytest.mark.asyncio
async def test_live_path_survives_a_missing_episode_migration(monkeypatch):
    """Deployed before the migration: windows still close, the shadow just logs."""
    corrs = [str(uuid.uuid4()) for _ in range(2)]
    fake, _ = _classifier({corrs[0]: 0.1, corrs[1]: 0.99})
    monkeypatch.setattr(worker, "classify_turn", fake)
    name, pool = await _make_db(with_episode_migration=False)
    try:
        bus, runner = _Bus(), _Runner()
        ws, es = WindowStore(pool), EpisodeShadowStore(pool, worker.settings)
        for i, c in enumerate(corrs):
            await _deliver_twice(_env(c, at=T0 + timedelta(minutes=i), phase=None), bus=bus, window_store=ws, runner=runner, episode_store=es)
        assert len(runner.closed) == 1
        assert bus.closed_events() == []
    finally:
        await _drop_db(name, pool)


@pytest.mark.asyncio
async def test_degraded_retry_rewrites_the_window_score(monkeypatch):
    name, pool = await _make_db()
    try:
        ws = WindowStore(pool)
        corr = str(uuid.uuid4())
        turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="p", response="r", spark_meta={})
        await ws.append_turn(turn, scores={"conversation_boundary_score": None, "memory_classify_ts": "t0"})
        n = await ws.update_turn_scores(
            corr,
            scores={
                "conversation_boundary_score": 0.42,
                "memory_classify_ts": "t1",
                "turn_change_appraisal": {"turn_change_status": "ok"},
            },
        )
        assert n == 1
        entry = await ws.find_windowed_turn(corr)
        assert entry["conversation_boundary_score"] == 0.42
        assert entry["spark_meta"]["turn_change_appraisal"]["turn_change_status"] == "ok"
    finally:
        await _drop_db(name, pool)
