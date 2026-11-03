#!/usr/bin/env python3
"""One-shot: restore the full text of self answers the old 1200-char cut ended mid-word.

Until 2026-11-03 every durable path mirrored `text[:1200]` of Orion's
`:SelfDefinition` / `:LivedAnswer` into self_concept_history. Live that day:
83 of 141 lived rows and 49 of 68 definition rows sat at exactly 1200 chars.
The graph node still holds the full answer. This rewrites each cut row's
`content` with it.

A row is repaired only when ALL hold (anything else is reported and skipped):
  * produced_by = curiosity_self_inquiry and length(content) = 1200
  * the graph node for that run (and question) is readable
  * the node's text is longer than 1200 and starts with the row's content,
    i.e. the row is provably a cut of that exact answer

The write goes through the normal bus -> sql-writer path. sql-writer merges on
entry_id and replaces the row, so the row's own created_at, version and
evidence_refs are sent back unchanged: "current" (latest created_at per
concept) does not move.

Dry-run by default. --publish sends the writes. Snapshot, progress log,
before/after CSV and report land in --out (default /tmp/self-answer-full-text-backfill).

  PYTHONPATH=.:services/orion-hub python3 services/orion-hub/scripts/backfill_self_answer_full_text.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

CAP = 1200
SELECT_SQL = (
    "SELECT entry_id, created_at, concept_id, version, content, evidence_refs, produced_by "
    "FROM self_concept_history "
    "WHERE produced_by = 'curiosity_self_inquiry' AND length(content) = $1 "
    "ORDER BY created_at"
)
CHANNEL = "orion:self_concept:history:write"
KIND = "self_concept.history.write.v1"


def parse_entry(entry_id: str, concept_id: str) -> tuple[str, str | None] | None:
    """(run_id, question_id or None for a definition), or None if unrecognised."""
    if entry_id.startswith("self-definition:"):
        return entry_id.removeprefix("self-definition:"), None
    if entry_id.startswith("self-lived:") and concept_id.startswith("self:lived:"):
        run_id = entry_id.removeprefix("self-lived:").split(":", 1)[0]
        return run_id, concept_id.removeprefix("self:lived:")
    return None


def graph_text(reader, run_id: str, question_id: str | None) -> str | None:
    from orion.curiosity.self_inquiry import read_lived_answers_for_run, read_self_definition

    if question_id is None:
        definition = read_self_definition(reader, run_id)
        return definition.text if definition is not None else None
    for answer in read_lived_answers_for_run(reader, run_id):
        if answer.question_id == question_id:
            return answer.text
    return None


def decide(row: dict, full: str | None) -> tuple[str, str]:
    """(verdict, reason). verdict is 'repair' or 'skip'."""
    content = str(row["content"] or "")
    if full is None:
        return "skip", "graph_node_not_found"
    if len(full) <= CAP:
        return "skip", "graph_text_not_longer"
    if not full.startswith(content) and not full.strip().startswith(content.strip()):
        return "skip", "graph_text_is_not_a_continuation"
    return "repair", "cut_of_graph_text"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publish", action="store_true", help="send the repaired rows to sql-writer")
    parser.add_argument("--out", default="/tmp/self-answer-full-text-backfill")
    parser.add_argument("--pg-dsn", default=os.environ.get("POSTGRES_URI") or os.environ.get("RECALL_PG_DSN"))
    parser.add_argument("--bus-url", default=os.environ.get("ORION_BUS_URL"))
    parser.add_argument("--graph-host", default=os.environ.get("HUB_CURIOSITY_GRAPH_HOST", "127.0.0.1"))
    parser.add_argument("--graph-port", type=int, default=int(os.environ.get("HUB_CURIOSITY_GRAPH_PORT", "6380")))
    parser.add_argument("--graph-name", default=os.environ.get("HUB_CURIOSITY_GRAPH_OWN", "orion_worldview"))
    args = parser.parse_args(argv)
    if not args.pg_dsn:
        print("missing --pg-dsn / POSTGRES_URI", file=sys.stderr)
        return 2
    if args.publish and not args.bus_url:
        print("missing --bus-url / ORION_BUS_URL", file=sys.stderr)
        return 2

    import asyncpg

    from orion.curiosity.worldview import WorldviewReader
    from orion.schemas.self_concept_history import SelfConceptHistoryV1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = (out / "progress.log").open("a")

    def log(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line)
        progress.write(line + "\n")
        progress.flush()

    conn = await asyncpg.connect(args.pg_dsn)
    try:
        rows = [dict(r) for r in await conn.fetch(SELECT_SQL, CAP)]
    finally:
        await conn.close()

    with (out / "snapshot.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    log(f"snapshot rows={len(rows)} file={out / 'snapshot.jsonl'}")

    reader = WorldviewReader(host=args.graph_host, port=args.graph_port, graph_name=args.graph_name)
    bus = None
    if args.publish:
        from orion.core.bus.async_service import OrionBusAsync

        bus = OrionBusAsync(args.bus_url)
        await bus.connect()

    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    source = ServiceRef(name="orion-hub", instance="backfill_self_answer_full_text")
    decisions: list[dict] = []
    errors = 0
    started = time.monotonic()
    try:
        for i, row in enumerate(rows, 1):
            parsed = parse_entry(str(row["entry_id"]), str(row["concept_id"]))
            full = None
            if parsed is None:
                verdict, reason = "skip", "unrecognised_entry_id"
            else:
                try:
                    full = graph_text(reader, *parsed)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    full = None
                    log(f"graph_read_error entry={row['entry_id']} err={exc}")
                verdict, reason = decide(row, full)
            if verdict == "repair" and bus is not None:
                evidence = row["evidence_refs"]
                if isinstance(evidence, str):
                    evidence = json.loads(evidence)
                write = SelfConceptHistoryV1(
                    entry_id=row["entry_id"],
                    created_at=row["created_at"],
                    concept_id=row["concept_id"],
                    version=row["version"],
                    content=full.strip(),
                    evidence_refs=list(evidence or []),
                    produced_by=row["produced_by"],
                )
                await bus.publish(
                    CHANNEL,
                    BaseEnvelope(
                        kind=KIND,
                        source=source,
                        correlation_id=uuid5(NAMESPACE_URL, f"self-answer-full-text:{row['entry_id']}"),
                        payload=write.model_dump(mode="json"),
                    ),
                )
            decisions.append({
                "entry_id": row["entry_id"],
                "concept_id": row["concept_id"],
                "verdict": verdict,
                "reason": reason,
                "before_chars": len(row["content"] or ""),
                "after_chars": len(full.strip()) if verdict == "repair" and full else len(row["content"] or ""),
                "before_tail": str(row["content"] or "")[-40:],
                "after_tail": full.strip()[-40:] if verdict == "repair" and full else "",
            })
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed else 0.0
            eta = (len(rows) - i) / rate if rate else 0.0
            log(
                f"backfill {100 * i / max(1, len(rows)):.0f}% eta={eta:.0f}s rows={i}/{len(rows)} "
                f"rate={rate:.1f}/s errors={errors} {verdict} {row['entry_id']} {reason}"
            )
    finally:
        if bus is not None:
            await bus.close()

    with (out / "before_after.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(decisions[0].keys()) if decisions else ["entry_id"])
        writer.writeheader()
        writer.writerows(decisions)
    repaired = [d for d in decisions if d["verdict"] == "repair"]
    skipped: dict[str, int] = {}
    for d in decisions:
        if d["verdict"] == "skip":
            skipped[d["reason"]] = skipped.get(d["reason"], 0) + 1
    restored = sum(d["after_chars"] - d["before_chars"] for d in repaired)
    mode = "publish" if args.publish else "dry-run"
    (out / "report.md").write_text(
        "# self answer full-text backfill\n\n"
        f"- mode: {mode}\n- rows at exactly {CAP} chars: {len(rows)}\n"
        f"- {'repaired' if args.publish else 'would repair'}: {len(repaired)}\n"
        f"- characters restored: {restored}\n- skipped: {skipped or 'none'}\n- errors: {errors}\n"
        f"- files: {out / 'snapshot.jsonl'}, {out / 'before_after.csv'}, {out / 'progress.log'}\n"
    )
    log(f"done mode={mode} rows={len(rows)} repair={len(repaired)} restored_chars={restored} "
        f"skipped={skipped} errors={errors}")
    progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
