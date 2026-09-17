#!/usr/bin/env python3
"""Read-only: anti-drives-2.0 acceptance checks on endogenous outreach.

## Why this exists

The stress-test of ``outreach_reason.v1`` (six kinds) concluded: do not mint
kinds before the ledger can answer what Orion actually talked about. This
script measures the live decision log against the checks in
``docs/superpowers/specs/2026-09-16-outreach-content-identity-design.md``.

It does **not** invent motive categories. It prints pass/fail with numbers.

## Checks

1. **Target monoculture** — no ``target_id`` > 70% of sends / 7d
2. **Cap pin** — daily send count must not equal daily_cap every day / 14d
3. **Content-gate falsifiability** — ``tension_without_content`` > 0 / 14d
4. **Identity coverage** — post-patch rows only; pre-patch skipped / **14d**
5. **Repeat content** — same prior_id / curiosity_content_id in two sends / **7d**
   (informational until a novelty gate exists)

## Disclosed scoping

- Read-only session (``default_transaction_read_only = on``).
- Reads ``endogenous_outreach_decisions`` only — what production stored.
- Daily cap default 4 matches ``HUB_ENDOGENOUS_OUTREACH_DAILY_CAP`` example;
  override with ``--daily-cap``.

Usage::

    python3 scripts/analysis/measure_outreach_reason_clusters.py
    POSTGRES_URI=postgresql://... python3 scripts/analysis/measure_outreach_reason_clusters.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _dsn() -> str:
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_DSN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"


def _connect(dsn: str):
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"psycopg2 required: {exc}") from exc
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute("SHOW default_transaction_read_only")
        row = cur.fetchone()
        if not row or str(row[0]).lower() not in ("on", "true"):
            conn.close()
            raise SystemExit("refusing to run: session is not read-only")
    return conn


def _as_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _id_list(grounding: Dict[str, Any], key: str) -> List[str]:
    val = grounding.get(key)
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()]


def check_target_monoculture(
    sends: Sequence[Dict[str, Any]], *, max_share: float = 0.70
) -> Dict[str, Any]:
    targets = [str(r.get("target_id") or "").strip() or "(none)" for r in sends]
    if not targets:
        return {"name": "target_monoculture", "pass": None, "detail": "no sends in window"}
    counts = Counter(targets)
    top, n = counts.most_common(1)[0]
    share = n / len(targets)
    return {
        "name": "target_monoculture",
        "pass": share <= max_share,
        "detail": f"top={top} share={share:.3f} (n={n}/{len(targets)}; bar<={max_share})",
        "counts": dict(counts.most_common(8)),
    }


def check_cap_pin(
    daily_sends: Dict[Any, int], *, daily_cap: int, min_days: int = 7
) -> Dict[str, Any]:
    if len(daily_sends) < min_days:
        return {
            "name": "cap_pin",
            "pass": None,
            "detail": f"only {len(daily_sends)} days with data; need >={min_days}",
        }
    pinned = sum(1 for n in daily_sends.values() if n == daily_cap)
    all_pinned = pinned == len(daily_sends) and daily_cap >= 0
    return {
        "name": "cap_pin",
        "pass": not all_pinned,
        "detail": (
            f"days={len(daily_sends)} pinned_at_cap={pinned} "
            f"daily_cap={daily_cap} (fail if every day == cap)"
        ),
        "daily": {str(k): v for k, v in sorted(daily_sends.items())},
    }


def check_content_gate(tension_without_content: int) -> Dict[str, Any]:
    return {
        "name": "content_gate_falsifiable",
        "pass": tension_without_content > 0,
        "detail": (
            f"tension_without_content={tension_without_content} "
            "(must be >0 over window; 0 means gate never blocked)"
        ),
    }


def check_identity_coverage(sends: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Among post-patch sends with content counts, ID lists must align.

    Pre-patch rows lack ``prior_ids`` / ``curiosity_content_ids`` entirely.
    Those are skipped (not failed) so a mixed 14d window after deploy stays
    honest. All-skipped → UNVERIFIED.
    """
    need_prior = 0
    prior_ok = 0
    need_curiosity = 0
    curiosity_ok = 0
    content_bearing = 0
    scored = 0
    skipped_pre_patch = 0
    for row in sends:
        g = _as_dict(row.get("grounding"))
        if not g:
            continue
        priors_count = int(g.get("priors_count") or 0)
        curiosity_count = int(g.get("curiosity_summaries") or 0)
        if priors_count <= 0 and curiosity_count <= 0:
            continue
        content_bearing += 1
        has_prior_key = "prior_ids" in g
        has_cur_key = "curiosity_content_ids" in g
        if not has_prior_key and not has_cur_key:
            skipped_pre_patch += 1
            continue
        scored += 1
        if priors_count > 0:
            need_prior += 1
            ids = g.get("prior_ids")
            if isinstance(ids, list) and len(ids) == priors_count:
                prior_ok += 1
        if curiosity_count > 0:
            need_curiosity += 1
            ids = g.get("curiosity_content_ids")
            if isinstance(ids, list) and len(ids) == curiosity_count:
                curiosity_ok += 1

    if content_bearing == 0:
        return {
            "name": "identity_coverage",
            "pass": None,
            "detail": "no sends with priors_count/curiosity_summaries > 0",
        }
    if scored == 0:
        return {
            "name": "identity_coverage",
            "pass": None,
            "detail": (
                f"all {skipped_pre_patch} content-bearing sends lack ID keys "
                "(pre-content-identity deploy; UNVERIFIED until new sends land)"
            ),
        }
    ok = (
        (need_prior == 0 or prior_ok == need_prior)
        and (need_curiosity == 0 or curiosity_ok == need_curiosity)
    )
    return {
        "name": "identity_coverage",
        "pass": ok,
        "detail": (
            f"scored={scored} skipped_pre_patch={skipped_pre_patch} "
            f"prior_aligned={prior_ok}/{need_prior} "
            f"curiosity_aligned={curiosity_ok}/{need_curiosity}"
        ),
    }


def check_repeat_content(sends: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Informational: IDs reused across sends in the window."""
    prior_seen: Dict[str, int] = defaultdict(int)
    cur_seen: Dict[str, int] = defaultdict(int)
    for row in sends:
        g = _as_dict(row.get("grounding"))
        for pid in _id_list(g, "prior_ids"):
            prior_seen[pid] += 1
        for sid in _id_list(g, "curiosity_content_ids"):
            cur_seen[sid] += 1
    prior_repeats = {k: v for k, v in prior_seen.items() if v > 1}
    cur_repeats = {k: v for k, v in cur_seen.items() if v > 1}
    return {
        "name": "repeat_content",
        "pass": None,  # informational until novelty gate
        "detail": (
            f"repeated_prior_ids={len(prior_repeats)} "
            f"repeated_curiosity_content_ids={len(cur_repeats)} "
            "(informational; novelty gate not shipped)"
        ),
        "prior_repeats": dict(sorted(prior_repeats.items(), key=lambda kv: -kv[1])[:10]),
        "curiosity_repeats": dict(sorted(cur_repeats.items(), key=lambda kv: -kv[1])[:10]),
    }


def fetch_rows(
    conn, *, days: int
) -> Tuple[List[Dict[str, Any]], Dict[Any, int], List[datetime]]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT decided_at, outreach, reason, target_id, result_json
            FROM endogenous_outreach_decisions
            WHERE decided_at >= %s
            ORDER BY decided_at ASC
            """,
            (since,),
        )
        cols = [d[0] for d in cur.description]
        raw_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    sends: List[Dict[str, Any]] = []
    daily: Dict[Any, int] = defaultdict(int)
    tension_events: List[datetime] = []
    for row in raw_rows:
        reason = str(row.get("reason") or "")
        decided = row.get("decided_at")
        if reason == "tension_without_content" and isinstance(decided, datetime):
            tension_events.append(decided)
        if not row.get("outreach"):
            continue
        day = decided.date() if isinstance(decided, datetime) else None
        if day is not None:
            daily[day] += 1
        grounding = _as_dict(row.get("result_json")).get("grounding")
        sends.append(
            {
                "decided_at": decided,
                "target_id": row.get("target_id"),
                "reason": reason,
                "grounding": grounding if isinstance(grounding, dict) else {},
            }
        )
    return sends, dict(daily), tension_events


def _sends_in_last_days(
    sends: Sequence[Dict[str, Any]], *, days: int, now: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    clock = now or datetime.now(timezone.utc)
    cutoff = clock - timedelta(days=days)
    out: List[Dict[str, Any]] = []
    for row in sends:
        decided = row.get("decided_at")
        if not isinstance(decided, datetime):
            continue
        when = decided if decided.tzinfo else decided.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            out.append(row)
    return out


def run_checks(
    *,
    sends_7d: Sequence[Dict[str, Any]],
    sends_14d: Sequence[Dict[str, Any]],
    daily_sends_14d: Dict[Any, int],
    tension_without_content_14d: int,
    daily_cap: int,
) -> List[Dict[str, Any]]:
    return [
        check_target_monoculture(sends_7d),
        check_cap_pin(daily_sends_14d, daily_cap=daily_cap),
        check_content_gate(tension_without_content_14d),
        check_identity_coverage(sends_14d),
        check_repeat_content(sends_7d),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="outer fetch window (must be >=14 to cover 14d checks; 7d checks slice inside)",
    )
    parser.add_argument("--daily-cap", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    fetch_days = max(int(args.days), 14)
    dsn = _dsn()
    conn = _connect(dsn)
    try:
        sends, _daily, tension_events = fetch_rows(conn, days=fetch_days)
    finally:
        conn.close()

    now = datetime.now(timezone.utc)
    sends_7d = _sends_in_last_days(sends, days=7, now=now)
    sends_14d = _sends_in_last_days(sends, days=14, now=now)
    daily_14d: Dict[Any, int] = defaultdict(int)
    for row in sends_14d:
        decided = row.get("decided_at")
        if isinstance(decided, datetime):
            daily_14d[decided.date()] += 1
    cutoff_14d = now - timedelta(days=14)
    tension_14d = 0
    for when in tension_events:
        stamp = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        if stamp >= cutoff_14d:
            tension_14d += 1

    results = run_checks(
        sends_7d=sends_7d,
        sends_14d=sends_14d,
        daily_sends_14d=dict(daily_14d),
        tension_without_content_14d=tension_14d,
        daily_cap=args.daily_cap,
    )
    payload = {
        "fetch_days": fetch_days,
        "send_count_7d": len(sends_7d),
        "send_count_14d": len(sends_14d),
        "daily_cap": args.daily_cap,
        "checks": results,
    }
    if args.json:
        print(json.dumps(payload, default=str, indent=2))
    else:
        print(
            "outreach reason clusters (read-only) — "
            f"7d/14d checks (fetched {fetch_days}d)"
        )
        print(
            f"sends_7d={len(sends_7d)} sends_14d={len(sends_14d)} "
            f"daily_cap={args.daily_cap}"
        )
        print()
        for check in results:
            status = check["pass"]
            if status is True:
                mark = "PASS"
            elif status is False:
                mark = "FAIL"
            else:
                mark = "UNVERIFIED"
            print(f"[{mark}] {check['name']}: {check['detail']}")
            for key in ("counts", "daily", "prior_repeats", "curiosity_repeats"):
                if check.get(key):
                    print(f"       {key}={check[key]}")
    hard_fails = [c for c in results if c["pass"] is False]
    return 1 if hard_fails else 0


if __name__ == "__main__":
    sys.exit(main())
