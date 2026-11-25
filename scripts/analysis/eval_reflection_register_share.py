#!/usr/bin/env python3
"""Philosophical share of what each organ writes, one local measure (read-only eval, no gate, no table).

Design: docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md, section 6 and acceptance
check 6. Samples up to ``--per-organ`` items per organ in the window (deterministic: md5 of the row id),
labels them with the finding-register judge on Orion's own local seat (prompt v2, temperature 0, thinking
off; a reply from a model not in PASSED_JUDGE_MODELS is a judge error, never a label), and prints the
philosophical share per organ with a 95% Wilson interval and the judge-error count. Curiosity findings use
the label the live register already stored (the same judge), so they cost no calls.

Rest point: an organ with no labelled items reads None, never 0. A judge error is left out of the share,
never counted as operational.

Read-only: SELECT through ``docker exec orion-athena-sql-db psql``, ``GRAPH.RO_QUERY`` for nothing here
(stored labels), and compute-only judge calls. Every judge call goes through orion/evals/leased_llm.py: ONE
GPU pool operator hold on ``--lane`` (an OPERATOR_LANES name, default ``agent``) for the whole run, each
call posted to orion-llm-gateway under it (#3921). No hold is taken when every organ asked for reads
stored labels. When the pool recalls the hold, the rows judged so far are written as ``*.partial`` files
and the run exits 3. No outside model is called.

    ORION_BUS_URL=redis://<tailnet-ip>:6379/0 python3 scripts/analysis/eval_reflection_register_share.py \\
        [--lane agent] [--days 14 | --start 2026-11-11 --end 2026-11-25] [--per-organ 30] [--organs outreach,day_letter]

Writes ``/tmp/reflection-register-share/<end date>/report.md`` and ``items.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reflection_register_sources import DEFAULT_LANE, ORGANS, Item, judge_one, read_items, sample, wilson  # noqa: E402

from orion.evals.leased_llm import (EXIT_RECALLED, EXIT_UNAVAILABLE, OPERATOR_LANES, LaneRecalled,  # noqa: E402
                                    LaneUnavailable, operator_lane_sync)

HOLDER = "operator:eval-reflection-register-share"

OUT_ROOT = Path("/tmp/reflection-register-share")


def summarize(organ: str, labelled: list[dict]) -> dict:
    """Counts, share and interval for one organ (or sub-group). Pure."""
    c = Counter(x["label"] for x in labelled if x.get("label"))
    errors = sum(1 for x in labelled if not x.get("label"))
    n = sum(c.values())
    k = c.get("philosophical", 0)
    lo, hi = wilson(k, n)
    return {"organ": organ, "sample": len(labelled), "labelled": n, "philosophical": k,
            "operational": c.get("operational", 0), "no_finding": c.get("no_finding", 0), "judge_errors": errors,
            "share": (k / n) if n else None, "lo": lo, "hi": hi}


def fmt_share(s: dict) -> str:
    if s["share"] is None:
        return "None (nothing labelled)"
    return f"{s['share']:.0%} ({s['lo']:.0%}-{s['hi']:.0%})"


def render(rows: list[dict], *, start: str, end: str, lane: str) -> str:
    lines = [f"# Reflection register share, {start} .. {end}", "",
             f"Judge: prompt v2 on {lane or '(stored register labels only)'}; curiosity findings use the live register's "
             "stored label. Share = philosophical / labelled; judge errors are left out.", "",
             "| Organ | Sample | Philosophical | Operational | No finding | Judge errors | Philosophical share (95% interval) |",
             "|---|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| {s['organ']} | {s['sample']} | {s['philosophical']} | {s['operational']} | {s['no_finding']} | "
                     f"{s['judge_errors']} | {fmt_share(s)} |")
    return "\n".join(lines) + "\n"


def label_organs(organs: list[str], *, start: str, end: str, per_organ: int, lane, results: list[dict],
                 summaries: list[dict]) -> None:
    """Label every organ's sample, appending as it goes, so a recall (LaneRecalled, propagated) leaves the
    rows judged so far in ``results``. ``lane`` is None only when every organ reads stored labels."""
    for organ in organs:
        stored = organ.endswith("_finding")
        items: list[Item] = sample(read_items(organ, start=start, end=end, with_text=not stored), per_organ)
        labelled = []
        for it in items:
            if stored:
                rec = {"label": it.stored_label, "error": None if it.stored_label else (it.stored_error or "no_label"),
                       "model": "stored"}
            else:
                try:
                    label, reason, err, model = judge_one(lane, it)
                except LaneRecalled:
                    results.extend(labelled)
                    raise
                rec = {"label": label, "reason": reason, "error": err, "model": model}
            labelled.append({"organ": organ, "sub": it.sub, "ref": it.ref, "at": it.at, **rec})
            print(f"{organ:22s} {it.ref[:48]:48s} {rec['label'] or rec['error']}", file=sys.stderr)
        results.extend(labelled)
        summaries.append(summarize(organ, labelled))
        for sub in sorted({x["sub"] for x in labelled if x["sub"]}):
            summaries.append(summarize(f"{organ} / {sub}", [x for x in labelled if x["sub"] == sub]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", default=DEFAULT_LANE, choices=sorted(OPERATOR_LANES),
                    help="operator lane to judge on (one GPU pool hold, through the gateway)")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--per-organ", type=int, default=30)
    ap.add_argument("--organs", default=",".join(ORGANS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    end = args.end or datetime.now(timezone.utc).isoformat()
    start = args.start or (datetime.fromisoformat(end.replace("Z", "+00:00")) - timedelta(days=args.days)).isoformat()
    organs = [o.strip() for o in args.organs.split(",") if o.strip()]
    for o in organs:
        if o not in ORGANS:
            ap.error(f"unknown organ {o!r}; one of {', '.join(ORGANS)}")
    out_dir = args.out or OUT_ROOT / end[:10]
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    summaries: list[dict] = []
    needs_judge = any(not o.endswith("_finding") for o in organs)
    lane_name = args.lane if needs_judge else ""
    kw = dict(start=start, end=end, per_organ=args.per_organ, results=results, summaries=summaries)
    try:
        if needs_judge:
            with operator_lane_sync(args.lane, holder=HOLDER) as lane:
                label_organs(organs, lane=lane, **kw)
        else:
            label_organs(organs, lane=None, **kw)
    except LaneRecalled as exc:
        # Partial result, then stop (leased_llm: nothing resumes on its own).
        (out_dir / "items.partial.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in results))
        (out_dir / "report.partial.md").write_text(render(summaries, start=start, end=end, lane=lane_name))
        print(f"RECALLED on {args.lane} after {len(results)} rows: {exc}; wrote {out_dir / 'items.partial.jsonl'}",
              file=sys.stderr)
        return EXIT_RECALLED
    except LaneUnavailable as exc:
        print(f"REFUSED on {args.lane}: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    (out_dir / "items.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in results))
    report = render(summaries, start=start, end=end, lane=lane_name)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"wrote {out_dir / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
