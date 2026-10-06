"""Recover referent aliases from saved distill runs and resolve already-stored memories.

Stage 1 stored ``(key, role)`` only and dropped the writer's aliases. Those aliases are
still in each run's LangGraph checkpoint (``checkpoint_writes`` channel ``answer_text``,
thread = ``episode_distill_run.run_id``). This re-reads them, matches each distilled
memory to its stored row by the same deterministic id the writer used
(``memory_id_for``), and runs the SAME referent step the live persist runs, with the
memory's stored verified quotes. No LLM call, no re-validation.

Idempotent: deterministic ids, ON CONFLICT DO NOTHING, and descriptor expiry from the
memory's own time, so a second run writes nothing new.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Optional

from orion.memory.episode.distill import parse_distillation
from orion.memory.episode.validate import memory_id_for, normalize_referent_key, normalize_ws

from .resolve import ReferentPolicy
from .store import MemoryReferents, persist_referents

_RUNS = "SELECT episode_id, run_id FROM episode_distill_run ORDER BY finished_at, episode_id"
_ANSWER = """
SELECT type, blob FROM checkpoint_writes WHERE thread_id = %s AND channel = 'answer_text'
ORDER BY checkpoint_id DESC, idx DESC LIMIT 1
"""
_MEMORIES = """
SELECT memory_id::text AS memory_id, created_at FROM episode_memory WHERE episode_id = %s ORDER BY created_at, memory_id
"""
_REFERENTS = """
SELECT memory_id::text AS memory_id, referent_key, role FROM episode_memory_referent
WHERE memory_id = ANY(%s::uuid[]) ORDER BY memory_id, referent_key, role
"""
_EVIDENCE = """
SELECT memory_id::text AS memory_id, source_id, quote FROM episode_memory_evidence
WHERE memory_id = ANY(%s::uuid[]) AND verified AND source_kind = 'chat_prompt' ORDER BY memory_id, source_id, quote
"""


def decode_answer(type_: str, blob: bytes) -> Optional[str]:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    value = JsonPlusSerializer().loads_typed((type_, bytes(blob)))
    return value if isinstance(value, str) else None


def aliases_from_answer(episode_id: str, answer_text: str) -> dict[str, dict[str, Any]]:
    """memory_id -> {"aliases": key -> [(text, alias_kind)], "name_kinds": key -> alias_kind}.

    Answers from prompts before v4 carry bare alias strings and no alias_kind: they parse as
    descriptors, which never decide identity (orion/schemas/memory_episode.py)."""
    out: dict[str, dict[str, Any]] = {}
    for cand in parse_distillation(answer_text).memories:
        memory_id = memory_id_for(episode_id, cand.purpose, normalize_ws(cand.statement))
        entry = out.setdefault(memory_id, {"aliases": {}, "name_kinds": {}})
        for ref in cand.referents:
            key = normalize_referent_key(ref.key)
            if key is None:
                continue
            entry["name_kinds"].setdefault(key, ref.alias_kind)
            kept = entry["aliases"].setdefault(key, [])
            for alias in ref.aliases or []:
                text = normalize_ws(alias.text)
                if text and text not in {t for t, _k in kept}:
                    kept.append((text, alias.alias_kind))
    return out


async def _all(conn: Any, sql: str, params: tuple = ()) -> list[Any]:
    return await (await conn.execute(sql, params)).fetchall()


async def backfill_episode(conn: Any, episode_id: str, run_id: str, *, now: datetime,
                           policy: ReferentPolicy) -> dict[str, int]:
    answers: dict[str, dict[str, Any]] = {}
    rows = await _all(conn, _ANSWER, (run_id,))
    if rows:
        text = decode_answer(rows[0]["type"], rows[0]["blob"])
        if text:
            answers = aliases_from_answer(episode_id, text)
    memories = await _all(conn, _MEMORIES, (episode_id,))
    ids = [m["memory_id"] for m in memories]
    if not ids:
        return {"memories": 0}
    referents: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in await _all(conn, _REFERENTS, (ids,)):
        referents[r["memory_id"]].append((r["referent_key"], r["role"]))
    quotes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in await _all(conn, _EVIDENCE, (ids,)):
        quotes[e["memory_id"]].append((e["source_id"], e["quote"]))
    episode_quotes = [q for m in ids for q in quotes[m]]
    inputs = [
        MemoryReferents(
            memory_id=m["memory_id"], episode_id=episode_id, created_at=m["created_at"],
            referents=referents[m["memory_id"]],
            aliases_by_key=answers.get(m["memory_id"], {}).get("aliases", {}),
            name_kinds=answers.get(m["memory_id"], {}).get("name_kinds", {}),
            prompt_quotes=episode_quotes, own_prompt_quotes=[q for _, q in quotes[m["memory_id"]]],
        )
        for m in memories if referents[m["memory_id"]]
    ]
    counts = await persist_referents(conn, inputs, now=now, policy=policy, proposed_by="checkpoint_backfill")
    return {"memories": len(inputs), "answers_recovered": int(bool(answers)), **counts}


class _DryRunRollback(Exception):
    pass


async def backfill_all(conn: Any, *, now: datetime, policy: ReferentPolicy, dry_run: bool = False,
                       on_progress: Optional[Callable[[dict[str, Any]], None]] = None) -> list[dict[str, Any]]:
    """Every distilled episode, oldest first, each in its own savepoint.

    An episode that fails is rolled back alone, recorded with its error, and the run goes on.
    ``dry_run`` does all the same work inside one outer transaction and rolls it back, so the
    report says exactly what WOULD change. ``on_progress`` gets one dict per episode
    (index, total, processed, errors, elapsed_sec, rate_per_sec, eta_sec, row) as it happens.
    """
    runs = await _all(conn, _RUNS)
    report: list[dict[str, Any]] = []
    started = time.monotonic()

    async def run_all() -> None:
        errors = 0
        for i, run in enumerate(runs, 1):
            try:
                async with conn.transaction():
                    counts = await backfill_episode(conn, run["episode_id"], run["run_id"], now=now, policy=policy)
                row = {"episode_id": run["episode_id"], "ok": True, **counts}
            except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the run
                errors += 1
                row = {"episode_id": run["episode_id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
            report.append(row)
            if on_progress is not None:
                elapsed = time.monotonic() - started
                rate = i / elapsed if elapsed > 0 else 0.0
                on_progress({"index": i, "total": len(runs), "processed": i, "errors": errors,
                             "elapsed_sec": round(elapsed, 2), "rate_per_sec": round(rate, 2),
                             "eta_sec": round((len(runs) - i) / rate, 1) if rate else None, "row": row})

    if not dry_run:
        await run_all()
        return report
    try:
        async with conn.transaction():
            await run_all()
            raise _DryRunRollback()
    except _DryRunRollback:
        pass
    return report
