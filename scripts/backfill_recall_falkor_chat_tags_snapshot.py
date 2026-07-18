#!/usr/bin/env python3
"""Phase 3 of the recall RDF->Falkor cutover: snapshot chat_history_log and
social_room_turns rows to /tmp/<job-name>/snapshot.json before the actual
extraction+write pass (AGENTS.md section 14's backfill protocol).

This is the read-only, host-side stage. Postgres has asyncpg available in
the repo's shared .venv, but spaCy's en_core_web_trf model is only reliably
loaded inside the already-running orion-athena-meta-tags container (the
shared .venv lacks the model download; confirmed live before writing this
script). The extraction+write stage
(scripts/backfill_recall_falkor_chat_tags_extract_and_write.py) runs inside
that container instead, fed by this snapshot.

Row counts as of 2026-07-18 (well under AGENTS.md section 14's 100k-row /
100MB stop-and-ask threshold): 1,708 chat_history_log rows, 33
social_room_turns rows.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any

# Running as `python scripts/backfill_....py` puts scripts/ on sys.path[0],
# which shadows stdlib `platform` via scripts/platform/ and breaks asyncpg
# (same issue documented in scripts/backfill_phi_corpus.py).
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

import asyncpg

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:55432/conjourney"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)}")


async def _snapshot(dsn: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = await asyncpg.connect(dsn)
    try:
        chat_rows = await conn.fetch(
            "SELECT id, correlation_id, prompt, response, session_id, created_at "
            "FROM chat_history_log ORDER BY created_at ASC"
        )
        social_rows = await conn.fetch(
            "SELECT turn_id, correlation_id, prompt, response, text, session_id, created_at "
            "FROM social_room_turns ORDER BY created_at ASC"
        )
    finally:
        await conn.close()

    snapshot = {
        "snapshot_taken_at": datetime.now(timezone.utc).isoformat(),
        "source_dsn_host": dsn.rsplit("@", 1)[-1] if "@" in dsn else "unknown",
        "chat_history_log": [dict(r) for r in chat_rows],
        "social_room_turns": [dict(r) for r in social_rows],
    }
    out_path = out_dir / "snapshot.json"
    out_path.write_text(json.dumps(snapshot, default=_json_default, indent=2))
    return {
        "out_path": str(out_path),
        "chat_history_log_rows": len(chat_rows),
        "social_room_turns_rows": len(social_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot chat_history_log/social_room_turns for Falkor backfill")
    parser.add_argument("--dsn", default=os.environ.get("RECALL_PG_DSN") or DEFAULT_DSN)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/backfill-recall-falkor-chat-tags"))
    args = parser.parse_args()

    summary = asyncio.run(_snapshot(args.dsn, args.out_dir))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
