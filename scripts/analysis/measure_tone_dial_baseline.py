#!/usr/bin/env python3
"""Read-only: baseline of Orion's reply tone dials, and which dials separate
Juniper's good_tone turns from bad_tone turns.

## Why this exists

P0 of the dynamic personality graph design (PR #2726, section 5 "Tone"). The
current voice is generic assistant prose; before anything tries to move it,
this measures where it sits today on the structural dials in
``orion/personality/tone_dials.py``. The baseline is the thing to move AWAY
from, not a target.

## What it reads

- ``chat_history_log`` over ``--window-days``. Rows with a non-empty
  ``prompt`` and ``response`` are chat replies. Rows with an empty prompt are
  Orion's own outreach; they are reported as a separate group, never mixed.
- ``chat_response_feedback`` joined on ``target_correlation_id =
  chat_history_log.correlation_id``. ``good_tone`` is one side; ``bad_tone``
  (new) plus ``incorrect_tone`` (historical name) is the other. Thumbs
  ``up`` vs ``down`` is reported too, as a coarser secondary split.

## What it writes

Nothing to any DB (read-only session, enforced by
``orion.db_readonly.open_readonly_connection``). Writes ``baseline.json`` and
``summary.txt`` to ``--out-dir``.

## Disclosed scoping

- Feedback n is tiny. Any split with fewer than ``MIN_SEPARATION_N`` rows on
  either side prints "insufficient n" instead of a number.
- Separation is Cliff's delta (good vs bad): +1 means every good reply scores
  higher than every bad one on that dial, -1 the reverse, 0 no separation.

Usage::

    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_tone_dial_baseline.py --window-days 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.personality.tone_dials import DIAL_NAMES, measure_tone_dials  # noqa: E402

MIN_SEPARATION_N = 5
GOOD_CATEGORIES = frozenset({"good_tone"})
BAD_CATEGORIES = frozenset({"bad_tone", "incorrect_tone"})

SQL_HISTORY = """
SELECT correlation_id, prompt, response, created_at
FROM chat_history_log
WHERE created_at >= (now() AT TIME ZONE 'utc') - make_interval(days => %s)
  AND coalesce(btrim(response), '') <> ''
"""

SQL_FEEDBACK = """
SELECT target_correlation_id, feedback_value, categories
FROM chat_response_feedback
WHERE target_correlation_id IS NOT NULL
"""


def _dsn() -> str:
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_DSN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return "postgresql://postgres:postgres@localhost:55432/conjourney"


# ---------------------------------------------------------------- pure stats


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile, q in [0, 1]. None for no values."""
    if not values:
        return None
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 4)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """P(a > b) - P(a < b) over all pairs. None if either side is empty."""
    if not a or not b:
        return None
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return round((gt - lt) / (len(a) * len(b)), 4)


def dial_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Median / p10 / p90 / fraction-zero plus a degeneracy flag."""
    n = len(values)
    if n == 0:
        return {"n": 0, "median": None, "p10": None, "p90": None, "fraction_zero": None, "degenerate": "no_data"}
    frac_zero = round(sum(1 for v in values if v == 0) / n, 4)
    if frac_zero == 1.0:
        degenerate = "always_zero"
    elif len(set(values)) == 1:
        degenerate = "constant"
    elif frac_zero >= 0.95:
        degenerate = "near_always_zero"
    else:
        degenerate = None
    return {
        "n": n,
        "median": percentile(values, 0.5),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "fraction_zero": frac_zero,
        "degenerate": degenerate,
    }


def group_stats(dial_rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    return {name: dial_stats([r[name] for r in dial_rows]) for name in DIAL_NAMES}


def separation(good: Sequence[Dict[str, float]], bad: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """Per-dial good-vs-bad medians and Cliff's delta (or 'insufficient n')."""
    enough = len(good) >= MIN_SEPARATION_N and len(bad) >= MIN_SEPARATION_N
    dials: Dict[str, Any] = {}
    for name in DIAL_NAMES:
        g = [r[name] for r in good]
        b = [r[name] for r in bad]
        dials[name] = {
            "good_median": percentile(g, 0.5),
            "bad_median": percentile(b, 0.5),
            "cliffs_delta": cliffs_delta(g, b) if enough else "insufficient n",
        }
    return {"n_good": len(good), "n_bad": len(bad), "sufficient_n": enough, "dials": dials}


# ------------------------------------------------------------ row assembly


def build_report(
    history: Iterable[Dict[str, Any]],
    feedback: Iterable[Dict[str, Any]],
    *,
    window_days: int,
) -> Dict[str, Any]:
    """history rows: correlation_id, prompt, response. feedback rows:
    target_correlation_id, feedback_value, categories. Pure."""
    chat: List[Dict[str, float]] = []
    outreach: List[Dict[str, float]] = []
    by_corr: Dict[str, Dict[str, float]] = {}
    skipped_unmeasurable = 0
    for row in history:
        dials = measure_tone_dials(row.get("response") or "")
        if dials is None:
            skipped_unmeasurable += 1
            continue
        d = dials.as_dict()
        if (row.get("prompt") or "").strip():
            chat.append(d)
            if row.get("correlation_id"):
                by_corr[row["correlation_id"]] = d
        else:
            outreach.append(d)

    # One verdict per feedback row; a turn rated twice counts twice (disclosed
    # in the output via n). Only chat replies are joined.
    good_tone: List[Dict[str, float]] = []
    bad_tone: List[Dict[str, float]] = []
    thumbs_up: List[Dict[str, float]] = []
    thumbs_down: List[Dict[str, float]] = []
    # Fold feedback per turn first: a turn rated both ways (two rows) is
    # contested and goes into neither side of a split.
    feedback_rows = 0
    per_turn: Dict[str, Dict[str, set]] = {}
    for fb in feedback:
        feedback_rows += 1
        corr = fb.get("target_correlation_id") or ""
        if corr not in by_corr:
            continue
        turn = per_turn.setdefault(corr, {"cats": set(), "values": set()})
        turn["cats"] |= set(fb.get("categories") or [])
        turn["values"].add((fb.get("feedback_value") or "").lower())
    joined = len(per_turn)
    for corr, turn in per_turn.items():
        d = by_corr[corr]
        good, bad = bool(turn["cats"] & GOOD_CATEGORIES), bool(turn["cats"] & BAD_CATEGORIES)
        if good and not bad:
            good_tone.append(d)
        elif bad and not good:
            bad_tone.append(d)
        if turn["values"] == {"up"}:
            thumbs_up.append(d)
        elif turn["values"] == {"down"}:
            thumbs_down.append(d)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "counts": {
            "chat_replies": len(chat),
            "outreach": len(outreach),
            "skipped_unmeasurable": skipped_unmeasurable,
            "feedback_rows": feedback_rows,
            "feedback_joined_in_window": joined,
        },
        "groups": {"chat_replies": group_stats(chat), "outreach": group_stats(outreach)},
        "separation": {
            "good_tone_vs_bad_tone": separation(good_tone, bad_tone),
            "thumbs_up_vs_down": separation(thumbs_up, thumbs_down),
        },
    }


def render_summary(report: Dict[str, Any]) -> str:
    c = report["counts"]
    out = [
        f"Tone dial baseline, last {report['window_days']} days (generated {report['generated_at']})",
        f"rows: chat_replies={c['chat_replies']} outreach={c['outreach']} "
        f"skipped_unmeasurable={c['skipped_unmeasurable']} "
        f"feedback_rows={c['feedback_rows']} feedback_joined_in_window={c['feedback_joined_in_window']}",
    ]
    for group, stats in report["groups"].items():
        out.append("")
        out.append(f"[{group}]  dial: median (p10..p90) frac_zero  flag")
        for name, s in stats.items():
            flag = s["degenerate"] or ""
            out.append(
                f"  {name:38s} {s['median']} ({s['p10']}..{s['p90']}) {s['fraction_zero']}  {flag}".rstrip()
            )
    for split, sep in report["separation"].items():
        out.append("")
        out.append(f"[{split}]  n_good={sep['n_good']} n_bad={sep['n_bad']}")
        if not sep["sufficient_n"]:
            out.append(f"  insufficient n (need >= {MIN_SEPARATION_N} per side); medians shown, no separation statistic")
        for name, s in sep["dials"].items():
            out.append(f"  {name:38s} good={s['good_median']} bad={s['bad_median']} cliffs_delta={s['cliffs_delta']}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------- I/O


def _fetch(conn, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--out-dir", default="/tmp/tone-dial-baseline/")
    args = ap.parse_args(argv)

    from orion.db_readonly import open_readonly_connection

    conn = open_readonly_connection(_dsn(), connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        print("UNKNOWN: could not open a confirmed read-only DB session", file=sys.stderr)
        return 2
    try:
        history = _fetch(conn, SQL_HISTORY, (args.window_days,))
        feedback = _fetch(conn, SQL_FEEDBACK)
    finally:
        conn.close()

    report = build_report(history, feedback, window_days=args.window_days)
    summary = render_summary(report)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"wrote {out / 'baseline.json'} and {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
