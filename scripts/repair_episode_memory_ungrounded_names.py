#!/usr/bin/env python3
"""Re-check stored episode memories for names Juniper never said, and queue them for confirmation.

The validator now escalates a statement that names a known person, place, project or service
that Juniper never said in its episode and that none of its own quotes contain
(orion/memory/episode/validate.py ``ungrounded_names``; spec
docs/superpowers/specs/2026-10-07-situation-graph-design.md section 7). Rows stored before that
check are re-run here with the SAME function. Live case: c0dc86c8 "Juniper corrected me that she
lives in Ogden, Utah, not Chicago", whose "Chicago" came from Orion's own reply.

A flagged row is not rewritten or deleted. It moves from settled (low / auto) to asked
(high / "ungrounded_name" / pending_confirmation), so the existing confirmation loop asks
Juniper, and one episode_memory_event records why. Only active, low-stakes, auto rows can move.

Backfill protocol (AGENTS.md section 14), under /tmp/episode-ungrounded-name-repair/:
- dry run by default: computes and reports, writes nothing to the database;
- --apply snapshots every row it will change (snapshot.json) before writing;
- progress.log, report.md and before_after.csv in both modes.

    python scripts/repair_episode_memory_ungrounded_names.py --dsn postgresql://...           # dry run
    python scripts/repair_episode_memory_ungrounded_names.py --dsn postgresql://... --apply   # write

The DSN is required and never defaulted: point it at the database you mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.memory.episode.validate import (  # noqa: E402
    JUNIPER_VOICES,
    MEMORY_ID_NAMESPACE,
    UNGROUNDED_NAME_STAKES_LABEL,
    ungrounded_names,
)

OUT = Path(os.getenv("EPISODE_NAME_REPAIR_OUT", "/tmp/episode-ungrounded-name-repair"))
ACTOR = "repair:ungrounded_name"
SNAPSHOT_ROW_LIMIT = 100_000

CANDIDATES_SQL = """
SELECT m.memory_id::text AS memory_id, m.episode_id, m.statement, m.stakes, m.stakes_reason,
       m.confirmation_state, m.status,
       m.voice,
       COALESCE((SELECT array_agg(e.quote) FROM episode_memory_evidence e
                 WHERE e.memory_id = m.memory_id AND e.verified AND e.source_kind = 'chat_response'), '{}')
         AS reply_quotes,
       COALESCE((SELECT array_agg(h.prompt)
                 FROM memory_episode_shadow s, jsonb_array_elements(s.turns) t
                 JOIN chat_history_log h ON h.correlation_id::text = t->>'correlation_id'
                 WHERE s.episode_id = m.episode_id AND NOT COALESCE((t->>'is_command')::boolean, false)), '{}')
         AS juniper_prompts
FROM episode_memory m
WHERE m.status = 'active' AND m.stakes = 'low' AND m.confirmation_state = 'auto'
ORDER BY m.created_at
"""
KEYS_SQL = "SELECT DISTINCT referent_key FROM episode_memory_referent"
UPDATE_SQL = """
UPDATE episode_memory
SET stakes = 'high', stakes_reason = %s, confirmation_state = 'pending_confirmation', updated_at = %s
WHERE memory_id = %s::uuid AND status = 'active' AND stakes = 'low' AND confirmation_state = 'auto'
"""
EVENT_SQL = """
INSERT INTO episode_memory_event (event_id, memory_id, op, actor, episode_id, evidence, reason, created_at)
VALUES (%s, %s::uuid, 'ungrounded_name', %s, %s, %s::jsonb, 'statement_names_what_juniper_did_not_say', %s)
ON CONFLICT (event_id) DO NOTHING
"""


def find_flags(rows: list[dict], keys: list[str]) -> tuple[list[dict], list[str]]:
    """(flagged rows, memory ids skipped because their episode's prompts did not load). Same rule
    as the validator: Juniper's voice is grounded only by her prompts, Orion's also by their own reply."""
    flags, skipped = [], []
    for r in rows:
        prompts = list(r["juniper_prompts"] or [])
        if not prompts:
            skipped.append(r["memory_id"])  # no episode text: every name would look imported
            continue
        own = [] if r["voice"] in JUNIPER_VOICES else list(r["reply_quotes"] or [])
        names = ungrounded_names(r["statement"], known_keys=keys, grounding_texts=prompts + own)
        if names:
            flags.append({**r, "referents": names})
    return flags, skipped


def _log(line: str) -> None:
    with (OUT / "progress.log").open("a") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} {line}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import psycopg
    from psycopg.rows import dict_row

    OUT.mkdir(parents=True, exist_ok=True)
    mode = "apply" if args.apply else "dry-run"
    _log(f"start mode={mode}")
    with psycopg.connect(args.dsn, row_factory=dict_row) as conn:
        rows = conn.execute(CANDIDATES_SQL).fetchall()
        if len(rows) > SNAPSHOT_ROW_LIMIT:
            _log(f"stop: {len(rows)} candidate rows exceeds {SNAPSHOT_ROW_LIMIT}")
            return 2
        keys = [r["referent_key"] for r in conn.execute(KEYS_SQL).fetchall()]
        flags, skipped = find_flags(rows, keys)
        _log(f"checked rows={len(rows)} known_keys={len(keys)} flagged={len(flags)} "
             f"skipped_no_episode_text={len(skipped)} errors=0")

        with (OUT / "before_after.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["memory_id", "statement", "referents", "before", "after"])
            for f in flags:
                w.writerow([f["memory_id"], f["statement"], " ".join(f["referents"]),
                            f"{f['stakes']}/{f['stakes_reason']}/{f['confirmation_state']}",
                            f"high/{UNGROUNDED_NAME_STAKES_LABEL}/pending_confirmation"])

        written = 0
        if args.apply and flags:
            (OUT / "snapshot.json").write_text(json.dumps(flags, default=str, indent=2))
            now = datetime.now(timezone.utc)
            started = datetime.now(timezone.utc)
            with conn.transaction():
                for i, f in enumerate(flags, 1):
                    cur = conn.execute(UPDATE_SQL, (UNGROUNDED_NAME_STAKES_LABEL, now, f["memory_id"]))
                    if cur.rowcount == 1:
                        event_id = uuid.uuid5(MEMORY_ID_NAMESPACE, f"{ACTOR}|{f['memory_id']}")
                        conn.execute(EVENT_SQL, (event_id, f["memory_id"], ACTOR, f["episode_id"],
                                                 json.dumps({"referents": f["referents"]}), now))
                        written += 1
                    elapsed = max((datetime.now(timezone.utc) - started).total_seconds(), 1e-6)
                    rate = i / elapsed
                    _log(f"repair {100 * i // len(flags)}% rows {i}/{len(flags)} rate={rate:.1f}/s "
                         f"eta={(len(flags) - i) / rate:.1f}s errors=0 memory={f['memory_id'][:8]} "
                         f"updated={cur.rowcount}")

    report = [
        f"# Episode memory ungrounded-name repair ({mode})", "",
        f"- rows checked (active, low, auto): {len(rows)}",
        f"- known referent keys: {len(keys)} (every key in episode_memory_referent; wider than the "
        "validator's per-run candidate list, so this can only flag more, never fewer)",
        f"- skipped, episode text did not load: {len(skipped)}",
        f"- flagged: {len(flags)}",
        f"- written: {written}" if args.apply else "- written: 0 (dry run)",
        "- errors: 0", "",
    ]
    report += [f"- `{f['memory_id']}` names {', '.join(f['referents'])}: {f['statement']}" for f in flags]
    report += ["", f"Files: {OUT}/progress.log, report.md, before_after.csv"
               + (", snapshot.json" if args.apply and flags else "")]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    _log(f"done mode={mode} flagged={len(flags)} written={written}")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
