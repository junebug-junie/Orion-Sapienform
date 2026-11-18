"""The one writer of the shadow episode-memory tables (manual_migration_episode_memory_v1.sql).

Runs on orion-durable-runs' psycopg pool (autocommit, dict rows). One transaction per episode.
Every id is deterministic (memory_id from episode + purpose + statement; event ids from the memory
and op), and every insert is ON CONFLICT DO NOTHING, so a persist replayed after a crash writes
nothing twice. Invalid candidates land only in episode_memory_event as ``rejected_invalid``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from orion.memory.episode.validate import MEMORY_ID_NAMESPACE, ValidationResult

ACTOR = "memory.episode_distill"
DREAM_ACTOR = "dream.day"
logger = logging.getLogger(__name__)

_INSERT_MEMORY = """
INSERT INTO episode_memory (
    memory_id, episode_id, purpose, voice, channel, statement, occurred_at, stakes, stakes_reason,
    confirmation_state, strength, half_life_days, last_reinforced_at, reinforcement_count,
    due_after, expires_at, status, model_route, prompt_version, run_id, created_at, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, %s, 'active', %s, %s, %s, %s, %s)
ON CONFLICT (memory_id) DO NOTHING
"""
_INSERT_EVIDENCE = """
INSERT INTO episode_memory_evidence (memory_id, source_kind, source_id, quote, quote_sha256, verified)
VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
"""
_INSERT_REFERENT = """
INSERT INTO episode_memory_referent (memory_id, referent_key, role) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
"""
_INSERT_EVENT = """
INSERT INTO episode_memory_event (event_id, memory_id, op, actor, episode_id, evidence, reason, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING
"""
_INSERT_QUESTION = """
INSERT INTO memory_tension_shadow (question_id, text, kind, scope, answer_via, source_episode_id, source_refs,
                                   referent_keys, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (question_id) DO NOTHING
"""
_INSERT_RUN = """
INSERT INTO episode_distill_run (episode_id, run_id, model_route, model, prompt_version, prompt_tokens,
    completion_tokens, llm_latency_ms, hold_wait_ms, memories_kept, memories_rejected, questions_kept,
    downgrades, coverage, finished_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (episode_id) DO NOTHING
"""


def _event_id(episode_id: str, key: str) -> str:
    return str(uuid.uuid5(MEMORY_ID_NAMESPACE, f"{episode_id}|event|{key}"))


async def persist_episode(
    pool: Any,
    *,
    episode_id: str,
    run_id: str,
    result: ValidationResult,
    model_route: str,
    model: Optional[str],
    prompt_version: str,
    usage: dict[str, Any],
    llm_latency_ms: Optional[int],
    hold_wait_ms: Optional[int],
    coverage: Optional[float],
    now: Optional[datetime] = None,
    referent_policy: Any = None,
    supersede_links: Optional[list[dict[str, Any]]] = None,
    supersede_plan: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    """``referent_policy`` (orion.memory.referents.resolve.ReferentPolicy) turns on the Stage 2
    referent step; None skips it (MEMORY_REFERENTS_ENABLED=false).

    ``supersede_links`` (orion.memory.episode.supersede.Link dicts, from the distill node's check)
    retire what this episode's memories correct, in the same transaction: the old row superseded,
    the new row's pointer, one event. Each link is re-checked against live state first.

    ``supersede_plan`` (the check's whole plan) is kept as one append-only ``memory_supersede_run``
    row (issue #3660): ids, legs, votes, verdicts, what was applied; never statements."""
    from psycopg.types.json import Jsonb

    now = now or datetime.now(timezone.utc)
    async with pool.connection() as conn:
        async with conn.transaction():
            await _insert_memories(conn, episode_id, run_id, result, model_route, prompt_version, now)
            for r in result.rejections:
                await conn.execute(
                    _INSERT_EVENT,
                    (_event_id(episode_id, f"rejected|{r.kind}|{r.index}"), None, "rejected_invalid", ACTOR,
                     episode_id, Jsonb({"kind": r.kind, "candidate": r.candidate, "run_id": run_id}), r.reason, now),
                )
            for q in result.questions:
                await conn.execute(
                    _INSERT_QUESTION,
                    (q.question_id, q.text, q.kind, q.scope, q.answer_via, episode_id,
                     Jsonb([{"source_kind": e.source_kind, "source_id": e.source_id, "quote": e.quote,
                             "verified": e.verified} for e in q.evidence]),
                     q.referents, now),
                )
            applied: list[dict[str, Any]] = []
            superseded = await _apply_supersede_links(conn, episode_id, result, now, supersede_links, applied)
            if supersede_plan is not None:
                await _record_supersede_run(conn, episode_id, run_id, now, supersede_plan, applied)
            referents = await _persist_referents(conn, episode_id, result, now, referent_policy)
            await conn.execute(
                _INSERT_RUN,
                (episode_id, run_id, model_route, model, prompt_version, usage.get("prompt_tokens"),
                 usage.get("completion_tokens"), llm_latency_ms, hold_wait_ms, len(result.memories),
                 len(result.rejections), len(result.questions), result.downgrades, coverage, now),
            )
    return {
        "memories": len(result.memories),
        "rejections": len(result.rejections),
        "questions": len(result.questions),
        "downgrades": result.downgrades,
        "superseded": superseded,
        **({"referents_resolved": referents["resolved"]} if referents else {}),
    }


async def _insert_memories(conn: Any, episode_id: str, run_id: str, result: ValidationResult, model_route: str,
                           prompt_version: str, now: datetime, *, actor: str = ACTOR) -> None:
    """The memory rows of ``result``: memory, evidence, referents, its created event and its validator events.
    Every insert is ON CONFLICT DO NOTHING (ids are deterministic), so a replay writes nothing twice."""
    from psycopg.types.json import Jsonb

    for m in result.memories:
        await conn.execute(
            _INSERT_MEMORY,
            (
                m.memory_id, episode_id, m.purpose, m.voice, m.channel, m.statement, m.occurred_at,
                m.stakes, m.stakes_reason, m.confirmation_state, m.strength, m.half_life_days, now,
                m.due_after, m.expires_at, model_route, prompt_version, run_id, now, now,
            ),
        )
        for ev in m.evidence:
            await conn.execute(_INSERT_EVIDENCE, (m.memory_id, ev.source_kind, ev.source_id, ev.quote,
                                                  hashlib.sha256(ev.quote.encode("utf-8")).hexdigest(),
                                                  ev.verified))
        for key, role in m.referents:
            await conn.execute(_INSERT_REFERENT, (m.memory_id, key, role))
        await conn.execute(
            _INSERT_EVENT,
            (_event_id(episode_id, f"{m.memory_id}|created"), m.memory_id, "created", actor, episode_id,
             Jsonb({"run_id": run_id, "voice": m.voice, "purpose": m.purpose}), None, now),
        )
        for i, e in enumerate(m.events):
            await conn.execute(
                _INSERT_EVENT,
                (_event_id(episode_id, f"{m.memory_id}|{e.op}|{i}"), m.memory_id, e.op, actor, episode_id,
                 Jsonb(e.detail), e.reason, now),
            )


async def persist_dream_memories(conn: Any, *, episode_id: str, run_id: str, result: ValidationResult,
                                 model_route: str, prompt_version: str, now: datetime) -> int:
    """The day dream's memories (orion/memory/dream_day.py), on the caller's connection INSIDE the caller's
    transaction (the dream_day row is written in the same one). The same rows a distilled episode writes,
    actor ``dream.day``; no episode_distill_run row, no referent resolution, no supersede. Validator
    rejections land as ``rejected_invalid`` events, never as memories. Returns the memories inserted now
    (0 on a replay: every id is deterministic and every insert is ON CONFLICT DO NOTHING)."""
    from psycopg.types.json import Jsonb

    ids = [m.memory_id for m in result.memories]
    before = 0
    if ids:
        cur = await conn.execute("SELECT count(*) AS n FROM episode_memory WHERE memory_id = ANY(%s::uuid[])", (ids,))
        row = await cur.fetchone()
        before = int((row or {}).get("n") or 0)
    await _insert_memories(conn, episode_id, run_id, result, model_route, prompt_version, now, actor=DREAM_ACTOR)
    for r in result.rejections:
        await conn.execute(
            _INSERT_EVENT,
            (_event_id(episode_id, f"rejected|{r.kind}|{r.index}"), None, "rejected_invalid", DREAM_ACTOR,
             episode_id, Jsonb({"kind": r.kind, "candidate": r.candidate, "run_id": run_id}), r.reason, now),
        )
    return len(ids) - before


async def _apply_supersede_links(conn: Any, episode_id: str, result: ValidationResult, now: datetime,
                                 links: Optional[list[dict[str, Any]]],
                                 applied_out: Optional[list[dict[str, Any]]] = None) -> int:
    """Only links whose new row is one of THIS episode's validated memories are applied."""
    if not links:
        return 0
    from orion.memory.episode.supersede import ACTOR_DISTILL, Link, apply_link

    mine = {m.memory_id for m in result.memories}
    applied = 0
    for d in links:
        link = Link.from_dict(d)
        if link.new_id not in mine:
            continue
        written = await apply_link(conn, link, actor=ACTOR_DISTILL, now=now, episode_id=episode_id)
        if written:
            applied += 1
            if applied_out is not None:
                applied_out.append({"old_id": link.old_id, "new_id": link.new_id, "relation": written["relation"],
                                    "invalid_at": written["invalid_at"], "path": written["path"]})
    return applied


async def _record_supersede_run(conn: Any, episode_id: str, run_id: str, now: datetime, plan: dict[str, Any],
                                applied: list[dict[str, Any]]) -> None:
    """One append-only audit row per distill run, in a SAVEPOINT: a fault here (its migration not
    applied yet) is logged and rolled back alone; the memories and any supersede still commit."""
    from orion.memory.episode.supersede import record_run

    try:
        async with conn.transaction():
            await record_run(conn, run_id=f"distill:{episode_id}:{run_id}", source="distill", episode_id=episode_id,
                             plan=plan, applied=applied, finished_at=now)
    except Exception:  # noqa: BLE001 - the audit row must never cost the memories
        logger.exception("supersede_run_record_failed episode_id=%s", episode_id)


async def _persist_referents(conn: Any, episode_id: str, result: ValidationResult, now: datetime,
                             policy: Any) -> Optional[dict[str, int]]:
    """Stage 2 referent step, in a SAVEPOINT: a fault here (e.g. its migration not applied yet)
    is logged and rolled back alone, and the memories above still commit. The checkpoint backfill
    (scripts/backfill_referents_from_episodes.py) can redo it later; every write is idempotent."""
    if policy is None or not result.memories:
        return None
    from orion.memory.referents.store import MemoryReferents, persist_referents

    episode_quotes = [(ev.source_id, ev.quote) for m in result.memories for ev in m.evidence
                      if ev.verified and ev.source_kind == "chat_prompt"]
    inputs = [
        MemoryReferents(
            memory_id=m.memory_id, episode_id=episode_id, created_at=now, referents=list(m.referents),
            aliases_by_key=dict(m.referent_aliases), name_kinds=dict(m.referent_name_kinds),
            prompt_quotes=episode_quotes,
            own_prompt_quotes=[ev.quote for ev in m.evidence if ev.verified and ev.source_kind == "chat_prompt"],
        )
        for m in result.memories if m.referents
    ]
    try:
        async with conn.transaction():
            return await persist_referents(conn, inputs, now=now, policy=policy)
    except Exception:  # noqa: BLE001 - logged loudly; memories must not be lost to a referent fault
        logger.exception("referents_persist_failed episode_id=%s", episode_id)
        return None


async def candidate_referent_keys(pool: Any, *, days: int = 30, limit: int = 60) -> list[str]:
    """Referent keys already in use by recent shadow memories (the prompt's candidate list).
    Fail-open: an unreadable table (migration not applied) gives an empty list."""
    try:
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT r.referent_key, count(*) AS n
                    FROM episode_memory_referent r JOIN episode_memory m USING (memory_id)
                    WHERE m.created_at > now() - make_interval(days => %s)
                    GROUP BY 1 ORDER BY n DESC, 1 LIMIT %s
                    """,
                    (int(days), int(limit)),
                )
            ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [str((r["referent_key"] if isinstance(r, dict) else r[0])) for r in rows]
