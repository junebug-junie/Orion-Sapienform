#!/usr/bin/env python3
"""One-shot: mirror any `:LivedAnswer` nodes into self_concept_history.

Use after the durable finish path lagged the graph write (2026-09-19: three
LivedAnswers on orion_worldview, zero self:lived:* history rows).

  docker exec orion-athena-hub python3 /app/services/orion-hub/scripts/backfill_lived_answer_mirrors.py
  # or from a worktree with PYTHONPATH=scripts:.

Dry-run by default. Pass --publish to emit bus writes for sql-writer.
Idempotent on entry_id (self-lived:<run_id>).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

REPO = Path(__file__).resolve().parents[3] if (Path(__file__).resolve().parents[0].name == "scripts") else Path("/repo")
if not (REPO / "orion").is_dir():
    # docker exec copy to /tmp, or odd cwd — prefer mounted repo roots.
    for candidate in (Path("/repo"), Path("/app"), Path(__file__).resolve().parents[3]):
        if (candidate / "orion").is_dir():
            REPO = candidate
            break
sys.path[:0] = [str(REPO), str(REPO / "services" / "orion-hub")]


def _reader():
    from app.settings import get_settings
    from orion.curiosity.worldview import WorldviewReader

    cfg = get_settings()
    return WorldviewReader(
        host=str(cfg.HUB_CURIOSITY_GRAPH_HOST),
        port=int(cfg.HUB_CURIOSITY_GRAPH_PORT),
        graph_name=str(
            getattr(cfg, "HUB_CURIOSITY_GRAPH_OWN_DB", None)
            or getattr(cfg, "HUB_CURIOSITY_GRAPH_NAME", "")
            or "orion_worldview"
        ),
    )


def _list_lived(reader) -> list[dict]:
    rows = reader.query(
        "MATCH (a:LivedAnswer) "
        "RETURN a.run_id AS run_id, a.question_id AS question_id, "
        "a.family AS family, a.text AS text, a.evidence AS evidence, "
        "a.revises AS revises, a.written_at AS written_at "
        "ORDER BY coalesce(a.written_at, 0)"
    )
    return list(rows or [])


async def _already_mirrored(pool, entry_id: str) -> bool:
    if pool is None:
        return False
    async with pool.acquire() as conn:
        row = await conn.fetchval(
            "SELECT 1 FROM self_concept_history WHERE entry_id = $1 LIMIT 1",
            entry_id,
        )
    return bool(row)


async def _next_version(pool, concept_id: str) -> int:
    if pool is None:
        return 1
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM self_concept_history "
            "WHERE concept_id = $1",
            concept_id,
        )
    return int(value or 1)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish SelfConceptHistoryV1 rows onto the write channel",
    )
    args = parser.parse_args(argv)

    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.curiosity.self_inquiry import (
        SELF_DEFINITION_PRODUCER,
        SELF_INQUIRY_TAG,
        build_lived_answer,
        build_lived_answer_history_write,
        lived_concept_id,
        read_lived_answer,
    )

    # Same constants Hub uses (services/orion-hub/scripts/curiosity_investigation.py).
    CHANNEL = "orion:self_concept:history:write"
    KIND = "self_concept.history.write.v1"
    SOURCE = ServiceRef(name="orion-hub", instance="backfill_lived_answer_mirrors")

    reader = _reader()
    rows = _list_lived(reader)
    print(f"lived_answer_nodes={len(rows)}")

    pool = None
    bus = None
    if args.publish:
        import asyncpg

        dsn = os.environ.get("RECALL_PG_DSN") or os.environ.get("POSTGRES_URI")
        if not dsn:
            print("missing RECALL_PG_DSN/POSTGRES_URI", file=sys.stderr)
            return 2
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
        bus_url = os.environ.get("ORION_BUS_URL")
        if not bus_url:
            print("missing ORION_BUS_URL", file=sys.stderr)
            return 2
        bus = OrionBusAsync(bus_url)
        await bus.connect()

    published = 0
    skipped = 0
    refused = 0
    try:
        for row in rows:
            answer = build_lived_answer(row)
            if answer is None:
                # Evidence may be a FalkorDB string; re-read via the canonical helper.
                rid = str(row.get("run_id") or "").strip()
                answer = read_lived_answer(reader, rid) if rid else None
            if answer is None or not answer.is_substantive:
                refused += 1
                print(f"refuse run={row.get('run_id')} question={row.get('question_id')}")
                continue
            entry_id = f"self-lived:{answer.run_id}"
            if await _already_mirrored(pool, entry_id):
                skipped += 1
                print(f"skip_existing {entry_id}")
                continue
            version = await _next_version(pool, lived_concept_id(answer.question_id))
            write = build_lived_answer_history_write(answer, version=version)
            if write is None:
                refused += 1
                print(f"refuse_no_evidence run={answer.run_id}")
                continue
            print(
                f"{'publish' if args.publish else 'would_publish'} "
                f"run={answer.run_id} concept={write.concept_id} "
                f"v={write.version} chars={len(write.content)} "
                f"produced_by={SELF_DEFINITION_PRODUCER}"
            )
            if not args.publish:
                continue
            assert bus is not None
            await bus.publish(
                CHANNEL,
                BaseEnvelope(
                    kind=KIND,
                    source=SOURCE,
                    correlation_id=uuid5(
                        NAMESPACE_URL, f"{SELF_INQUIRY_TAG}:lived-backfill:{answer.run_id}"
                    ),
                    payload=write.model_dump(mode="json"),
                ),
            )
            published += 1
    finally:
        if pool is not None:
            await pool.close()
        if bus is not None:
            await bus.close()

    print(
        f"done publish={published} skipped={skipped} refused={refused} "
        f"mode={'publish' if args.publish else 'dry-run'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
