"""The daily old-vs-new memory report, against a disposable Postgres.

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL (admin DSN of a throwaway server). Synthetic rows only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
SQL_DIR = REPO_ROOT / "services" / "orion-sql-db"
ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")

sys.path.insert(0, str(SERVICE_ROOT))
for key in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
    del sys.modules[key]
spec = importlib.util.spec_from_file_location("mc_episode_report", SERVICE_ROOT / "app" / "episode_report.py")
episode_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(episode_report)

LEGACY_TABLES = """
CREATE TABLE memory_crystallizations (crystallization_id uuid PRIMARY KEY, kind text, status text, summary text);
CREATE TABLE memory_crystallization_sources (crystallization_id uuid, source_kind text, source_id text);
"""

# 2026-09-28 06:26-09:54 MDT
T0 = datetime(2026, 9, 28, 12, 26, tzinfo=timezone.utc)


async def _db():
    import asyncpg

    name = f"eprep_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'CREATE DATABASE "{name}"')
    await admin.close()
    pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=2)
    for f in ("manual_migration_memory_consolidation_v1.sql", "manual_migration_memory_episode_v1.sql",
              "manual_migration_episode_memory_v1.sql"):
        await pool.execute((SQL_DIR / f).read_text())
    await pool.execute(LEGACY_TABLES)
    return name, pool


async def _drop(name, pool):
    import asyncpg

    await pool.close()
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    await admin.close()


async def _seed(pool):
    turns = [{"correlation_id": "c-1", "at": T0.isoformat()}, {"correlation_id": "c-2", "at": (T0 + timedelta(hours=3)).isoformat()}]
    await pool.execute(
        """INSERT INTO memory_episode_shadow (episode_id, status, episode_status, turns, started_at, last_turn_at,
           closed_at, close_reason, close_lag_sec, juniper_turn_count, command_turn_count)
           VALUES ('ep-1', 'closed', 'closed', $1::jsonb, $2, $3, $4, 'v2:phase_next_day', 65000, 2, 0)""",
        json.dumps(turns), T0, T0 + timedelta(hours=3), T0 + timedelta(hours=15))
    cid = uuid.uuid4()
    await pool.execute("INSERT INTO memory_crystallizations VALUES ($1, 'semantic', 'active', 'Headed to Austin and will fly back on Wednesday.')", cid)
    await pool.execute("INSERT INTO memory_crystallization_sources VALUES ($1, 'chat_turn', 'c-1')", cid)
    said, unverified, reverie = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for mid, voice, channel, statement in (
        (said, "juniper_said", "chat", "Juniper flew to Austin for her team offsite."),
        (unverified, "juniper_said", "chat", "Juniper prefers the window seat on flights."),
        (reverie, "orion_thought", "reverie", "Juniper is probably nervous about the Austin offsite."),
    ):
        await pool.execute(
            """INSERT INTO episode_memory (memory_id, episode_id, purpose, voice, channel, statement, stakes,
               confirmation_state, strength, half_life_days, last_reinforced_at, occurred_at)
               VALUES ($1, 'ep-1', 'happened', $2, $3, $4, 'low', 'auto', 0.8, 14, now(), $5)""",
            mid, voice, channel, statement, T0)
    # Only the first memory has a verified quote from one of Juniper's own prompts.
    await pool.execute(
        """INSERT INTO episode_memory_evidence (memory_id, source_kind, source_id, quote, quote_sha256, verified)
           VALUES ($1, 'chat_prompt', 'c-1', 'Headed to Austin', 'h1', true),
                  ($2, 'chat_prompt', 'c-1', 'window seat please', 'h2', false)""", said, unverified)
    await pool.execute(
        """INSERT INTO episode_memory_event (event_id, op, actor, episode_id, reason)
           VALUES ($1, 'rejected_invalid', 'memory.episode_distill', 'ep-1', 'no_verified_quote')""", uuid.uuid4())
    await pool.execute(
        """INSERT INTO episode_distill_run (episode_id, run_id, prompt_tokens, completion_tokens, llm_latency_ms,
           hold_wait_ms, memories_kept, memories_rejected, questions_kept, downgrades, coverage)
           VALUES ('ep-1', 'memdistill-ep-1', 900, 120, 4200, 1000, 1, 1, 0, 0, 1.0)""")


@pytest.mark.asyncio
async def test_report_puts_old_rows_next_to_new_memories(tmp_path):
    name, pool = await _db()
    try:
        await _seed(pool)
        settings = SimpleNamespace(MEMORY_EPISODE_REPORT_TZ="America/Denver", MEMORY_EPISODE_REPORT_DIR=str(tmp_path))
        # 2026-09-29 local: the report covers 2026-09-28, when ep-1 closed (21:26 MDT).
        now = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)
        path = await episode_report.write_due_report(pool, settings, now=now)
        assert path is not None and path.name == "2026-09-28.md"
        text = path.read_text()
        assert "1 episodes closed" in text and "Totals: 1 old crystallization rows, 3 new memories." in text
        assert "[semantic, active] Headed to Austin" in text
        # Each memory is shown exactly as Orion would read it (orion.memory.voice_render).
        assert ("[happened, juniper_said/chat] Juniper told me (09-28): "
                "Juniper flew to Austin for her team offsite.") in text
        assert ("[happened, juniper_said/chat] My own note (chat, 09-28), not Juniper's words: "
                "Juniper prefers the window seat on flights.") in text
        assert ("[happened, orion_thought/reverie] Something I was turning over on my own (reverie, 09-28), "
                "not something Juniper and I discussed: Juniper is probably nervous") in text
        assert "rejected_invalid no_verified_quote x1" in text
        assert "900 tokens in, 120 out" in text
        assert (tmp_path / "latest.md").read_text() == text
        assert await episode_report.write_due_report(pool, settings, now=now) is None   # final: written once
    finally:
        await _drop(name, pool)


@pytest.mark.asyncio
async def test_report_is_rewritten_until_every_episode_is_distilled(tmp_path):
    name, pool = await _db()
    try:
        await _seed(pool)
        await pool.execute("DELETE FROM episode_distill_run")             # ep-1 not distilled yet
        settings = SimpleNamespace(MEMORY_EPISODE_REPORT_TZ="America/Denver", MEMORY_EPISODE_REPORT_DIR=str(tmp_path))
        early = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)        # 09:00 MDT the next day
        path = await episode_report.write_due_report(pool, settings, now=early)
        assert "Provisional: 1 episode(s) not distilled yet" in path.read_text()
        assert await episode_report.write_due_report(pool, settings, now=early) is not None   # rewritten
        await pool.execute(
            """INSERT INTO episode_distill_run (episode_id, run_id, memories_kept, memories_rejected,
               questions_kept, downgrades) VALUES ('ep-1', 'memdistill-ep-1', 1, 0, 0, 0)""")
        path = await episode_report.write_due_report(pool, settings, now=early)
        assert "Provisional" not in path.read_text()
        assert await episode_report.write_due_report(pool, settings, now=early) is None       # now final
    finally:
        await _drop(name, pool)
