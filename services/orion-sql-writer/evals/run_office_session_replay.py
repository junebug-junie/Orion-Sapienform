"""Read-only replay of the office session builder on a live database.

Rebuilds every visit to a room camera in a time range from the rows already stored
(presence transitions, camera captions, person records and guesses, scene counts) and prints
each session: span, who, label, timeline. Writes nothing (READ ONLY transaction) and opens no ask.

    PYTHONPATH=<repo>:services/orion-sql-writer python services/orion-sql-writer/evals/run_office_session_replay.py \\
        --uri postgresql://postgres:postgres@localhost:55432/conjourney \\
        --since 2026-11-05T01:00:00Z --until 2026-11-05T02:00:00Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--stream", default="cam0")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from sqlalchemy import create_engine, text

    from app.vision_office_sessions import OfficeSessionsConfig, _transitions, build_spans, compute_session
    from app.vision_person_model import _camera_priors

    since = datetime.fromisoformat(a.since.replace("Z", "+00:00"))
    until = datetime.fromisoformat(a.until.replace("Z", "+00:00"))
    cfg = OfficeSessionsConfig(streams=(a.stream,))
    prior = _camera_priors().get(a.stream) or {}
    eng = create_engine(a.uri)
    out = []
    with eng.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows = [r for r in _transitions(conn, a.stream, since) if r["occurred_at"] < until]
        # "now" = the end of the range, so a visit inside it reads as it would have once settled.
        now = max(until, datetime.now(timezone.utc))
        for span in build_spans(rows):
            row, records, _sightings, _overlap = compute_session(conn, span, now, cfg, prior)
            out.append({
                "session_id": row["sid"], "started_at": span.started_at.isoformat(),
                "ended_at": span.ended_at.isoformat() if span.ended_at else None, "end_reason": span.end_reason,
                "status": row["status"], "who": row["_who"], "label": row["label"],
                "timeline": json.loads(row["timeline"]), "records": len(records),
            })
        conn.rollback()
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    for s in out:
        print(f"session {s['session_id']}  {s['started_at']} -> {s['ended_at']} ({s['end_reason']}, {s['status']})")
        w = s["who"]
        print(f"  who: {w['text']}  (records={w['records']} peak_people={w['peak_people']} unknown={w['unknown']}"
              f" reason={w['unknown_reason']})")
        print(f"  label: {s['label']}")
        for e in s["timeline"]:
            rep = f" (x{e['repeats']})" if e["repeats"] > 1 else ""
            print(f"    {e['at'][11:19]}Z  {e['text']}{rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
