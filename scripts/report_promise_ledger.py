#!/usr/bin/env python3
"""The promise ledger in shadow: what opened, what closed and how, and a blinded precision sample. Read-only.

Spec: docs/superpowers/specs/2026-11-11-intention-ledger-design.md (D2 and the 2026-11-16 amendments).
Issue #3531 reads this before any promise act counts toward a track record or reaches stance.

Over the last ``--days`` (default 28) it reports:

* promise acts (``grammar_acts`` actor ``hub.promise``, signal ``promise_kept``): open, kept (by the
  kind of evidence that closed it), expired, void (no Juniper chat turn in the horizon: censored);
* un-closeable: expired acts the judge was never shown at any Juniper turn in their window (more
  open promises than the judge's view, ``PROMISE_JUDGE_VIEW_MAX``). They are not broken promises;
* expired acts whose promising turn got an intention carry reply in the window (Prime took the
  promise on and delivered something; the ledger does not count a carry as a keep in v1);
* keep rate kept / (kept + expired), void and un-closeable left out, with a Beta(1, 1) posterior;
* the self-reinforcement guard: firm promises per JUDGED Juniper turn (judge model pass, not the
  quick gate), against the replay baseline (``REPLAY_BASELINE``); red above 2x. Stored items with no
  ``force``/``done_when`` (written before #3532) are skipped and counted, never guessed;
* ``--sample N``: a blinded sample of stored promise items for hand labels. Each item shows ONLY the
  quote and the reply around it, never the judge's doer/force/act/done_when, so the labeller cannot
  lean on the judge's own reading (amendment 2). Writes a label template with hash keys;
  ``--score FILE`` reads it back and prints precision with a Wilson 95% interval.

    ORION_LIVE_DSN=postgresql://... python3 scripts/report_promise_ledger.py [--days 28] [--json]
    ... --sample 60 --out /tmp/promise-ledger-labels.json      # blinded labelling sheet
    ... --score /tmp/promise-ledger-labels.json                # precision from filled labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.grammar.promise_acts import (  # noqa: E402
    PROMISE_ACTOR,
    PROMISE_SIGNAL,
    kept_evidence_kind,
    parse_promise_ref,
    promise_close_kind,
)
from orion.harness.promise_gate import normalize_for_quote_match  # noqa: E402
from orion.regulation.juniper_turns import JUNIPER_TURN_PREDICATE  # noqa: E402
from orion.schemas.harness_finalize import (  # noqa: E402
    PROMISE_DONE_WHEN_MIN_CHARS,
    PROMISE_FORCE_COMMITS,
    PROMISE_JUDGE_VIEW_MAX,
)

# Replay of the deployed finder (#3532 rules, same model and lane as live), judged turns only:
# dev 10-13..11-12 (179 judged, 15 firm promises), held-out 09-13..10-13 (86 judged, 0) and
# 11-12..11-16 (24 judged, 8). Firm = gated item with offer false. 23 / 289 = 0.080 per judged turn.
REPLAY_BASELINE = (23, 289)
GUARD_RATIO_RED = 2.0
CARRY_SOURCE = "intention_carry"
CONTEXT_CHARS = 280

ACTS_SQL = """
SELECT act_id, test_before, correlation_id, opened_at, horizon_at, closed_at, outcome_source, outcome_note
  FROM grammar_acts
 WHERE actor = %(actor)s AND signal_id = %(signal)s AND opened_at >= %(since)s
 ORDER BY opened_at
"""
JUNIPER_STARTS_SQL = f"""
SELECT created_at FROM chat_history_log WHERE {JUNIPER_TURN_PREDICATE} AND created_at >= %(since)s ORDER BY created_at
"""
JUDGED_SQL = f"""
SELECT t.correlation_id, t.created_at,
       t.run_artifact -> 'reflection' ->> 'reflection_source' AS reflection_source,
       t.run_artifact -> 'reflection' -> 'promises_made' AS made,
       t.run_artifact ->> 'final_text' AS final_text
  FROM harness_turn_trace t
  JOIN chat_history_log c ON c.correlation_id = t.correlation_id
 WHERE t.created_at >= %(since)s AND {JUNIPER_TURN_PREDICATE.replace('prompt', 'c.prompt').replace('client_meta', 'c.client_meta')}
"""
CARRY_SQL = """
SELECT client_meta ->> 'concern_id' AS concern_id, created_at
  FROM chat_history_log
 WHERE client_meta ->> 'source' = %(source)s AND created_at >= %(since)s
"""


def _aware(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, datetime):
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def beta_summary(kept: int, broken: int) -> dict[str, float]:
    """Beta(1, 1) prior: posterior mean and an approximate 90% interval (normal approximation of the
    Beta, clipped; exact enough for a report line, not used as a gate)."""
    a, b = kept + 1, broken + 1
    mean = a / (a + b)
    sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return {"mean": round(mean, 3), "lo90": round(max(0.0, mean - 1.645 * sd), 3),
            "hi90": round(min(1.0, mean + 1.645 * sd), 3), "n": kept + broken}


def unclosable_ids(acts: Sequence[Mapping[str, Any]], juniper_starts: Sequence[datetime],
                   view_max: int = PROMISE_JUDGE_VIEW_MAX) -> set[str]:
    """Expired acts that, at every Juniper turn inside their window, were not among the oldest
    ``view_max`` open promises: the judge never saw them, so they could not be marked kept."""
    out: set[str] = set()
    starts = sorted(t for t in (_aware(s) for s in juniper_starts) if t is not None)
    for act in acts:
        if promise_close_kind(act.get("outcome_source"), act.get("outcome_note"), act.get("closed_at") is not None) != "expired":
            continue
        opened, horizon = _aware(act.get("opened_at")), _aware(act.get("horizon_at"))
        if opened is None or horizon is None:
            continue
        seen = False
        for t in (s for s in starts if opened < s <= horizon):
            open_now = [a for a in acts if _open_at(a, t)]
            shown = {a["act_id"] for a in sorted(open_now, key=lambda a: _aware(a.get("opened_at")))[:view_max]}
            if act["act_id"] in shown:
                seen = True
                break
        if not seen:
            out.add(str(act["act_id"]))
    return out


def _open_at(act: Mapping[str, Any], t: datetime) -> bool:
    opened, horizon, closed = _aware(act.get("opened_at")), _aware(act.get("horizon_at")), _aware(act.get("closed_at"))
    return opened is not None and opened <= t and (horizon is None or horizon > t) and (closed is None or closed > t)


def close_summary(acts: Sequence[Mapping[str, Any]], *, unclosable: set[str],
                  carry_concerns: Mapping[str, list[datetime]] = {}) -> dict[str, Any]:
    from orion.prime.handback import turn_concern_id

    kinds: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    expired_with_carry = 0
    for act in acts:
        kind = promise_close_kind(act.get("outcome_source"), act.get("outcome_note"), act.get("closed_at") is not None)
        if kind == "expired" and str(act["act_id"]) in unclosable:
            kind = "unclosable"
        kinds[kind] += 1
        if kind == "kept":
            evidence[kept_evidence_kind(act.get("outcome_note")) or "unknown"] += 1
        if kind == "expired":
            parsed = parse_promise_ref(str(act.get("test_before") or ""))
            opened, horizon = _aware(act.get("opened_at")), _aware(act.get("horizon_at"))
            if parsed and opened and horizon and any(
                    opened < t <= horizon for t in carry_concerns.get(turn_concern_id(parsed[0]), [])):
                expired_with_carry += 1
    kept, expired = kinds.get("kept", 0), kinds.get("expired", 0)
    return {"acts": len(acts), "open": kinds.get("open", 0), "kept": kept, "expired": expired,
            "void": kinds.get("void", 0), "unclosable": kinds.get("unclosable", 0),
            "kept_by_evidence": dict(evidence), "expired_with_carry_reply": expired_with_carry,
            "keep_rate": beta_summary(kept, expired)}


def _made(raw: Any) -> list[dict]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def _firm(item: Mapping[str, Any]) -> bool:
    return item.get("force") == PROMISE_FORCE_COMMITS and len(str(item.get("done_when") or "").strip()) >= PROMISE_DONE_WHEN_MIN_CHARS


def guard_summary(turns: Sequence[Mapping[str, Any]], baseline: tuple[int, int] = REPLAY_BASELINE) -> dict[str, Any]:
    """Firm (non-offer) promises per JUDGED Juniper turn vs the replay baseline. Items written before
    #3532 (no force/done_when) are counted in ``skipped_pre_force`` and left out of both sides."""
    judged = [t for t in turns if t.get("reflection_source") == "substrate_informed_pass"]
    firm = skipped = offers = 0
    for t in judged:
        for item in _made(t.get("made")):
            if "force" not in item and "done_when" not in item:
                skipped += 1
            elif _firm(item):
                if item.get("offer") is True:
                    offers += 1
                else:
                    firm += 1
    rate = firm / len(judged) if judged else None
    base = baseline[0] / baseline[1]
    ratio = (rate / base) if rate is not None and base > 0 else None
    return {"judged_turns": len(judged), "turns_not_judged": len(turns) - len(judged), "firm_promises": firm,
            "offers": offers, "skipped_pre_force": skipped,
            "rate": None if rate is None else round(rate, 4), "baseline": f"{baseline[0]}/{baseline[1]} replay",
            "baseline_rate": round(base, 4), "ratio": None if ratio is None else round(ratio, 2),
            "red": bool(ratio is not None and ratio > GUARD_RATIO_RED)}


def item_key(correlation_id: str, index: int, quote: str) -> str:
    raw = f"{correlation_id}|{index}|{normalize_for_quote_match(quote)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _context(final_text: str, quote: str) -> str:
    text = " ".join(str(final_text or "").split())
    q = " ".join(str(quote or "").split())
    at = text.find(q)
    if at < 0:
        return text[: 2 * CONTEXT_CHARS]
    lo, hi = max(0, at - CONTEXT_CHARS), min(len(text), at + len(q) + CONTEXT_CHARS)
    return ("..." if lo else "") + text[lo:hi] + ("..." if hi < len(text) else "")


def blinded_sample(turns: Sequence[Mapping[str, Any]], n: int, *, seed: int = 3531) -> list[dict[str, Any]]:
    """Stored firm items from judged Juniper turns, n at random. Each shows only the quote and the reply
    around it: no doer, force, act, done_when or offer flag (the judge's own reading)."""
    pool: list[dict[str, Any]] = []
    for t in turns:
        if t.get("reflection_source") != "substrate_informed_pass":
            continue
        for i, item in enumerate(_made(t.get("made"))):
            if not _firm(item) or not str(item.get("quote") or "").strip():
                continue
            pool.append({"key": item_key(str(t["correlation_id"]), i, item["quote"]),
                         "quote": item["quote"], "reply_context": _context(t.get("final_text") or "", item["quote"]),
                         "label": None, "why": ""})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def score_labels(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    labelled = [x for x in items if x.get("label") in (True, False)]
    k = sum(1 for x in labelled if x["label"] is True)
    lo, hi = wilson(k, len(labelled))
    return {"labelled": len(labelled), "true": k, "precision": round(k / len(labelled), 3) if labelled else None,
            "wilson95": [round(lo, 3), round(hi, 3)]}


def _rows(conn: Any, sql: str, params: Mapping[str, Any]) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def gather(dsn: str, since: datetime) -> dict[str, Any]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as conn:
        acts = _rows(conn, ACTS_SQL, {"actor": PROMISE_ACTOR, "signal": PROMISE_SIGNAL, "since": since})
        starts = [r["created_at"] for r in _rows(conn, JUNIPER_STARTS_SQL, {"since": since})]
        turns = _rows(conn, JUDGED_SQL, {"since": since})
        carry: dict[str, list[datetime]] = {}
        for r in _rows(conn, CARRY_SQL, {"source": CARRY_SOURCE, "since": since}):
            if r.get("concern_id") and _aware(r.get("created_at")):
                carry.setdefault(str(r["concern_id"]), []).append(_aware(r["created_at"]))
    return {"acts": acts, "starts": starts, "turns": turns, "carry": carry}


def report(data: Mapping[str, Any]) -> dict[str, Any]:
    unclosable = unclosable_ids(data["acts"], data["starts"])
    return {"ledger": close_summary(data["acts"], unclosable=unclosable, carry_concerns=data["carry"]),
            "guard": guard_summary(data["turns"])}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", default=os.getenv("ORION_LIVE_DSN") or os.getenv("POSTGRES_URI"))
    ap.add_argument("--days", type=float, default=28.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="write a blinded labelling sheet of N stored items")
    ap.add_argument("--out", default="/tmp/promise-ledger-labels.json")
    ap.add_argument("--score", help="score a filled labelling sheet (no database needed)")
    args = ap.parse_args(argv)
    if args.score:
        print(json.dumps(score_labels(json.loads(Path(args.score).read_text())), indent=2))
        return 0
    if not args.dsn:
        print("set ORION_LIVE_DSN (or POSTGRES_URI, or --dsn); the report only reads", file=sys.stderr)
        return 2
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    data = gather(args.dsn, since)
    out = report(data)
    if args.sample:
        sheet = blinded_sample(data["turns"], args.sample)
        Path(args.out).write_text(json.dumps(sheet, indent=2))
        out["sample"] = {"written": len(sheet), "path": args.out}
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    led, g = out["ledger"], out["guard"]
    print(f"promise acts since {since:%Y-%m-%d %H:%M} UTC: {led['acts']} "
          f"(open {led['open']}, kept {led['kept']}, expired {led['expired']}, void {led['void']}, "
          f"un-closeable {led['unclosable']})")
    print(f"kept by evidence: {led['kept_by_evidence'] or '-'}; expired with a carry reply: {led['expired_with_carry_reply']}")
    kr = led["keep_rate"]
    print(f"keep rate (Beta(1,1)): mean {kr['mean']}, 90% {kr['lo90']}-{kr['hi90']}, n={kr['n']} (void and un-closeable left out)")
    print(f"guard: {g['firm_promises']} firm promises over {g['judged_turns']} judged Juniper turns = {g['rate']} "
          f"vs baseline {g['baseline_rate']} ({g['baseline']}): ratio {g['ratio']}{' RED' if g['red'] else ''}; "
          f"skipped pre-#3532 items {g['skipped_pre_force']}; turns not judged {g['turns_not_judged']}")
    if args.sample:
        print(f"blinded sample: {out['sample']['written']} items -> {out['sample']['path']} (fill label true/false, then --score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
