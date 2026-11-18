"""The day dream's reads and its one write (app/dream_day_graph.py ``DreamDayDeps``).

* ``gather_material``: the day's dream_cycle rows (+ hypotheses, pair answers), Prime's held concerns (one
  read-only Redis GET of ``orion:prime:state``), and Orion's now for the ``dream`` organ (``GET /temporal-self/now``
  on this same service, so the read is stored as an ``experience_shown`` row like every other reader's).
* ``record_cites``: the run's ``experience:`` cites to ``POST /temporal-self/now/use`` (record_use keeps only
  ids that were shown, at most 3).
* ``read_receipt``: the read receipt ``dig dream-day`` left in the self dir this container mounts.
* ``resolve_refs``: one SELECT per ref kind (orion/memory/dream_day.py ``RESOLVE_SQL``).
* ``persist_dream_day``: the night's memories (orion/memory/episode/store.py ``persist_dream_memories``) and the
  ``dream_day`` row in ONE transaction. First run to persist a day wins; a later run writes nothing and gets the
  stored row back.

This module is the dream organ's ``reads_now`` in config/integration/organs.yaml.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from orion.memory.dream_day import RESOLVE_SQL, day_window, grade, refs_by_kind
from orion.memory.episode.validate import ValidationResult
from orion.prime.state import PrimeStateV1
from orion.schemas.dream_day import (
    DREAM_DAY_PROMPT_VERSION, DREAM_DAY_RECEIPT_DIR, DreamDayBriefV1, DreamDayConcernV1, DreamDayCycleV1,
    DreamDayHypothesisV1, DreamDayMaterialV1, DreamDayNowItemV1, DreamDayReadV1, DreamDayV1,
)
from orion.schemas.temporal_self import TemporalNowV1

logger = logging.getLogger("orion-durable-runs.dream_day")

ORGAN = "dream"
PRIME_STATE_KEY = "orion:prime:state"   # orion/prime/block.py PRIME_STATE_KEY
NOW_ROUTE = "/temporal-self/now"
USE_ROUTE = "/temporal-self/now/use"
# This service's own HTTP port inside its container (Dockerfile CMD): the now is read through the same route
# every other organ uses, so it is stored and cursored the same way.
SELF_URL = "http://127.0.0.1:8121"
HTTP_TIMEOUT_SEC = 10.0
PAIR_ANSWER_CHARS = 300

DREAMS_SQL = """
SELECT cycle_id, status, started_at, ended_at, trigger, hypothesis_count, no_link_count,
       coalesce(cycle_json->'pair_answers', '[]'::jsonb) AS pair_answers
  FROM dream_cycle WHERE started_at >= %s AND started_at < %s ORDER BY started_at
"""
HYPOTHESES_SQL = """
SELECT hypothesis_id, cycle_id, arm, coalesce(claim, '') AS claim, coalesce(why, '') AS why
  FROM dream_hypothesis WHERE cycle_id = ANY(%s) ORDER BY created_at
"""
STORED_SQL = "SELECT run_id, memories_written, memory_ids, produced, note FROM dream_day WHERE day = %s"
COUNT_SQL = "SELECT count(*) AS n FROM episode_memory WHERE memory_id = ANY(%s::uuid[]) AND channel = 'dream'"
INSERT_SQL = """
INSERT INTO dream_day (day, run_id, status, produced, note, memories_written, memory_ids, items_dropped, refs_dropped,
    memories_rejected, plan_rejected, kept, dropped, new, plan, day_text, day_refs, material, read, now_shown_id,
    turn_correlation_id, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (day) DO NOTHING RETURNING run_id
"""


class _AlreadyWritten(Exception):
    """Another run persisted this day first (raised inside the transaction, so this run writes nothing)."""


def _aware(v: Any) -> datetime:
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- gather


async def read_dreams(pool: Any, day: date) -> list[DreamDayCycleV1]:
    lo, hi = day_window(day)
    async with pool.connection() as conn:
        rows = [dict(r) for r in await (await conn.execute(DREAMS_SQL, (lo, hi))).fetchall()]
        ids = [r["cycle_id"] for r in rows]
        hyps = [dict(r) for r in await (await conn.execute(HYPOTHESES_SQL, (ids,))).fetchall()] if ids else []
    out = []
    for r in rows:
        pairs = r.get("pair_answers")
        pairs = json.loads(pairs) if isinstance(pairs, str) else (pairs or [])
        out.append(DreamDayCycleV1(
            cycle_id=r["cycle_id"], status=r["status"], started_at=_aware(r["started_at"]),
            ended_at=_aware(r["ended_at"]) if r.get("ended_at") else None, trigger=r.get("trigger"),
            hypothesis_count=r.get("hypothesis_count"), no_link_count=r.get("no_link_count"),
            pair_answers=[{k: (str(p.get(k) or "")[:PAIR_ANSWER_CHARS] if k == "answer" else p.get(k))
                           for k in ("arm", "ref_a", "ref_b", "outcome", "answer")}
                          for p in pairs if isinstance(p, dict)],
            hypotheses=[DreamDayHypothesisV1(hypothesis_id=h["hypothesis_id"], arm=h.get("arm"), claim=h["claim"],
                                             why=h["why"]) for h in hyps if h["cycle_id"] == r["cycle_id"]]))
    return out


def concerns_from_state(raw: Any, now: datetime) -> tuple[list[DreamDayConcernV1], str]:
    """Prime's held concerns, read-only. (concerns, status): absent when Prime saved nothing."""
    if raw is None:
        return [], "absent"
    try:
        state = PrimeStateV1.model_validate_json(raw)
    except Exception:  # noqa: BLE001 -- never shown as "no concerns"
        return [], "unreadable"
    return [DreamDayConcernV1(
        concern_id=c.concern_id, text=c.text, age_h=max(0.0, (now - c.opened_at).total_seconds() / 3600.0),
        last_evidence_h=max(0.0, (now - c.last_evidence_at).total_seconds() / 3600.0), graded=c.graded)
        for c in state.concerns], "ok"


async def read_now(run_id: str, *, base_url: str = SELF_URL) -> TemporalNowV1:
    """Orion's now for the dream organ: no ``since``, so it moves the dream organ's own cursor."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        resp = await client.get(f"{base_url}{NOW_ROUTE}", params={"organ": ORGAN, "run_ref": f"dream_day:{run_id}"})
        resp.raise_for_status()
        return TemporalNowV1.model_validate(resp.json())


async def record_cites(shown_id: str, cites: list[str], citing_ref: str, *, base_url: str = SELF_URL) -> dict[str, Any]:
    """The run's cites, checked by ``record_use`` against the shown row (the use endpoint)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        resp = await client.post(f"{base_url}{USE_ROUTE}",
                                 json={"shown_id": shown_id, "citing_ref": citing_ref[:300], "cites": cites[:32]})
        resp.raise_for_status()
        return resp.json()


async def gather_material(pool: Any, redis_get: Callable[[str], Awaitable[Any]], brief: DreamDayBriefV1,
                          run_id: str, *, now: datetime,
                          now_reader: Callable[[str], Awaitable[TemporalNowV1]] = read_now) -> DreamDayMaterialV1:
    lo, hi = day_window(brief.day)
    fields: dict[str, Any] = {"day": brief.day, "window_start": lo, "window_end": hi, "gathered_at": now}
    try:
        fields["dreams"] = await read_dreams(pool, brief.day)
    except Exception:  # noqa: BLE001 -- said in the prompt as UNREAD
        logger.warning("dream_day_dreams_unreadable run=%s", run_id, exc_info=True)
        fields["dreams_status"] = "unreadable"
    try:
        raw = await asyncio.wait_for(redis_get(PRIME_STATE_KEY), timeout=2.0)
        fields["concerns"], fields["concerns_status"] = concerns_from_state(raw, now)
    except Exception:  # noqa: BLE001
        logger.warning("dream_day_concerns_unreadable run=%s", run_id, exc_info=True)
        fields["concerns_status"] = "unreadable"
    try:
        now_view = await now_reader(run_id)
        fields["now_shown_id"] = now_view.shown_id
        fields["now_items"] = [DreamDayNowItemV1(event_id=i.event_id, organ=i.organ, label=i.label)
                               for i in now_view.items]
    except Exception:  # noqa: BLE001 -- the run goes on without the now; no cites can be recorded
        logger.warning("dream_day_now_unavailable run=%s", run_id, exc_info=True)
        fields["now_status"] = "unavailable"
    return DreamDayMaterialV1(**fields)


# --------------------------------------------------------------------------- the receipt


def _read_receipt_sync(path: Path) -> Optional[DreamDayReadV1]:
    try:
        return DreamDayReadV1.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 -- a torn or foreign file is "not read", never a crash
        logger.warning("dream_day_receipt_unreadable path=%s", path, exc_info=True)
        return None


async def read_receipt(self_dir: str, day: date) -> Optional[DreamDayReadV1]:
    return await asyncio.to_thread(_read_receipt_sync, Path(self_dir) / DREAM_DAY_RECEIPT_DIR / f"{day.isoformat()}.json")


# --------------------------------------------------------------------------- refs


async def resolve_refs(pool: Any, refs: list[str]) -> dict[str, str]:
    """{canonical ref: text} for every ref that names a real row. Unknown kinds resolve to nothing."""
    out: dict[str, str] = {}
    by_kind = refs_by_kind(refs)
    async with pool.connection() as conn:
        for kind, ids in by_kind.items():
            sql = RESOLVE_SQL.get(kind)
            if sql is None or not ids:
                continue
            params = (ids, ids) if sql.count("%s") == 2 else (ids,)
            for row in await (await conn.execute(sql, params)).fetchall():
                rid = str(dict(row)["id"])
                if rid in ids:
                    out[f"{kind}:{rid}"] = str(dict(row)["text"] or "")
    return out


async def known_referents(pool: Any) -> list[str]:
    from orion.memory.episode.store import candidate_referent_keys

    try:
        return list(await candidate_referent_keys(pool))
    except Exception:  # noqa: BLE001 -- the names check then sees fewer names; never blocks the night
        logger.warning("dream_day_referents_unreadable", exc_info=True)
        return []


# --------------------------------------------------------------------------- persist


async def persist_dream_day(pool: Any, fields: dict[str, Any], result: ValidationResult, model_route: str) -> dict[str, Any]:
    """Memories + the dream_day row, one transaction. ``produced`` counts the memory rows read back inside it."""
    from psycopg.types.json import Jsonb

    from orion.memory.episode.store import persist_dream_memories

    fields = dict(fields)
    over_cap = int(fields.pop("memories_over_cap", 0) or 0)
    day: date = fields["day"]
    ids = [m.memory_id for m in result.memories]
    episode_id = f"dream_day:{day.isoformat()}"
    try:
        async with pool.connection() as conn:
            async with conn.transaction():
                existing = await (await conn.execute(STORED_SQL, (day,))).fetchone()
                if existing is not None:
                    raise _AlreadyWritten()
                await persist_dream_memories(conn, episode_id=episode_id, run_id=fields["run_id"], result=result,
                                             model_route=model_route, prompt_version=DREAM_DAY_PROMPT_VERSION,
                                             now=fields["created_at"])
                n = 0
                if ids:
                    n = int(dict(await (await conn.execute(COUNT_SQL, (ids,))).fetchone())["n"])
                produced, note = grade(n)
                row = DreamDayV1(produced=produced, note=note, memories_written=n, memory_ids=ids,
                                 memories_rejected=len(result.rejections) + over_cap, **fields)
                inserted = await (await conn.execute(INSERT_SQL, (
                    row.day, row.run_id, row.status, row.produced, row.note, row.memories_written, row.memory_ids,
                    row.items_dropped, row.refs_dropped, row.memories_rejected, row.plan_rejected, row.kept,
                    row.dropped, row.new, Jsonb(row.plan), row.day_text, row.day_refs, Jsonb(row.material),
                    Jsonb(row.read) if row.read is not None else None, row.now_shown_id, row.turn_correlation_id,
                    row.created_at))).fetchone()
                if inserted is None:
                    raise _AlreadyWritten()
        return {"outcome": "written", "run_id": row.run_id, "memories_written": n, "memory_ids": ids,
                "produced": produced, "note": note}
    except _AlreadyWritten:
        async with pool.connection() as conn:
            stored = dict(await (await conn.execute(STORED_SQL, (day,))).fetchone())
        ours = stored["run_id"] == fields["run_id"]
        return {"outcome": "written" if ours else "already_written", "run_id": stored["run_id"],
                "memories_written": int(stored["memories_written"]), "memory_ids": list(stored["memory_ids"] or []),
                "produced": bool(stored["produced"]), "note": stored["note"]}
