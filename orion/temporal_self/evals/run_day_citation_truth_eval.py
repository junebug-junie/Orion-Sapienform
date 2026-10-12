"""Periodic truth eval for Orion's citations of its own day (Temporal Self patch 4, acceptance check 6).

The citation gate proves a cited arc id exists and was shown. This eval asks the next question:
does the sentence describe that arc correctly? It is a measurement, not a gate (exit 0 unless the
database cannot be read).

Read-only. For each ``thought_decision`` row in the window whose ``day_citations`` label is in
``--labels`` and whose ``cited_valid`` is non-empty, it reads:

* the stance's ``imperative`` and ``tone`` (``thought_decision``),
* the reply Juniper saw: ``chat_history_log.response`` by correlation id, else the agent harness's
  ``harness_turn_trace.run_artifact->>'final_text'``,
* every cited arc's current row (``temporal_self_arc.arc_json``),

and checks each citing sentence with ``orion.temporal_self.citation_truth`` (deterministic only;
no LLM judge -- what it cannot decide is reported as undecidable).

Output: per decision, per sentence ``match`` / ``mismatch`` / ``undecidable`` with reasons, then per
ISO week the counts and ``truth_rate = match / (match + mismatch)``.

Usage:
    make eval-temporal-self-truth                      # last 7 days, text report
    python orion/temporal_self/evals/run_day_citation_truth_eval.py --days 7 --json [--out FILE]
    --dsn postgresql://...   (default: the host-published conjourney DB, localhost:55432)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from orion.temporal_self.citation_truth import ArcFacts, check_decision, summarize  # noqa: E402

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:55432/conjourney"
_ID_RE = re.compile(r"arc:([0-9a-f]{16})", re.I)

DECISIONS_SQL = """
SELECT correlation_id, created_at, imperative, tone, day_citations
FROM thought_decision
WHERE created_at >= %(since)s AND day_citations IS NOT NULL
  AND day_citations->>'label' = ANY(%(labels)s)
  AND jsonb_array_length(COALESCE(day_citations->'cited_valid', '[]'::jsonb)) > 0
ORDER BY created_at
"""
REPLY_SQL = """
SELECT 'chat_history_log' AS src, response AS text, created_at FROM chat_history_log
 WHERE correlation_id = %(c)s AND response IS NOT NULL AND response <> ''
UNION ALL
SELECT 'harness_turn_trace', run_artifact->>'final_text', created_at FROM harness_turn_trace
 WHERE correlation_id = %(c)s AND COALESCE(run_artifact->>'final_text', '') <> ''
ORDER BY 1, 3 DESC
"""
ARCS_SQL = "SELECT arc_json FROM temporal_self_arc WHERE arc_id = ANY(%(ids)s)"


def read_rows(dsn: str, since: datetime, labels: list[str]) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, options="-c default_transaction_read_only=on", row_factory=dict_row) as conn:
        # thought_decision.created_at is a naive UTC timestamp.
        decisions = conn.execute(DECISIONS_SQL, {"since": since.replace(tzinfo=None), "labels": labels}).fetchall()
        replies: dict[str, dict] = {}
        for d in decisions:
            got = conn.execute(REPLY_SQL, {"c": d["correlation_id"]}).fetchall()
            if got:
                replies[d["correlation_id"]] = got[0]
        ids: set[str] = set()
        for d in decisions:
            ids.update(i.lower() for i in _ID_RE.findall(json.dumps(d["day_citations"])))
            for text in (d["imperative"], d["tone"], (replies.get(d["correlation_id"]) or {}).get("text")):
                ids.update(i.lower() for i in _ID_RE.findall(text or ""))
        arcs = {r["arc_json"]["arc_id"]: r["arc_json"]
                for r in conn.execute(ARCS_SQL, {"ids": sorted(ids)}).fetchall()} if ids else {}
    return decisions, replies, arcs


def evaluate(decisions: list[dict], replies: dict[str, dict], arc_rows: dict[str, dict]) -> dict[str, Any]:
    arcs = {k: ArcFacts.from_row(v) for k, v in arc_rows.items()}
    rows = []
    for d in decisions:
        at = d["created_at"] if d["created_at"].tzinfo else d["created_at"].replace(tzinfo=timezone.utc)
        reply = replies.get(d["correlation_id"])
        checks, notes = check_decision(
            at=at, cited_valid=d["day_citations"].get("cited_valid") or [],
            texts={"imperative": d["imperative"], "tone": d["tone"], "reply": (reply or {}).get("text")},
            arcs=arcs,
        )
        rows.append({
            "correlation_id": d["correlation_id"], "created_at": at.isoformat(),
            "label": d["day_citations"].get("label"), "cited_valid": d["day_citations"].get("cited_valid") or [],
            "reply_source": (reply or {}).get("src"), "notes": notes,
            "sentences": [c.as_dict() for c in checks],
        })
    return {"rows": rows, "weeks": summarize(rows)}


def render_text(report: dict[str, Any], since: datetime, labels: list[str]) -> str:
    out = [f"Day-citation truth eval: decisions since {since:%Y-%m-%d %H:%MZ}, labels {labels}, "
           f"{len(report['rows'])} row(s). Deterministic checks only; no LLM judge."]
    for r in report["rows"]:
        out.append(f"\n{r['created_at']} {r['correlation_id']} label={r['label']} cited={len(r['cited_valid'])} "
                   f"reply={r['reply_source'] or 'none stored'}")
        for n in r["notes"]:
            out.append(f"  note: {n}")
        for s in r["sentences"]:
            out.append(f"  [{s['outcome'].upper()}] ({s['source']}, {s['attribution']}) {s['sentence']}")
            for reason in s["reasons"]:
                out.append(f"      - {reason}")
    out.append("\nWeekly summary (truth_rate = match / (match + mismatch)):")
    if not report["weeks"]:
        out.append("  no rows in window")
    for week, w in report["weeks"].items():
        rate = "n/a (nothing decided)" if w["truth_rate"] is None else f"{w['truth_rate']:.3f}"
        out.append(f"  {week}: rows {w['rows']} ({w['rows_with_sentences']} with sentences), sentences {w['sentences']}: "
                   f"match {w['match']}, mismatch {w['mismatch']}, undecidable {w['undecidable']}; truth_rate {rate}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--labels", default="verified", help="comma list of day_citations labels (verified,UNVERIFIED)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the text report")
    ap.add_argument("--out", type=Path, help="also write the JSON report here")
    args = ap.parse_args(argv)
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    decisions, replies, arcs = read_rows(args.dsn, since, labels)
    report = evaluate(decisions, replies, arcs)
    report.update(since=since.isoformat(), labels=labels, llm_judge=False)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str) if args.json else render_text(report, since, labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
