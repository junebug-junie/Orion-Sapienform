"""How a plan is graded on its run, and the replica's own track record (design sections 1, 2 and 6).

One row per replica-arm run: Hub's ``curiosity_offer_plans`` row LEFT JOIN its ``curiosity_run_outcomes`` row
(the spend log's score, ``orion/curiosity/value.py diff_snapshots``). The plan act (actor curiosity.replica,
verb infer) closes on the SAME number the three-arm test scores, so the act ledger and the experiment can
never disagree about what a plan did:

* plan not used (any status but ``ok``) -> 0 at once, with the status;
* used, run not scored yet -> wait;
* used, run scored but the score is unknown (``realized_nats`` NULL: no start snapshot) -> not graded here
  (the ledger's horizon expires the act; an unknown is never a 0). This is checked FIRST, so a failed turn
  whose score is unknown is not graded either: "unknown is never a 0" wins over "a failed turn is 0";
* used, run failed (``turn_ok`` false) -> 0, but only once that failure is final: a durable retry overwrites
  the outcome row, so a failure younger than ``failed_final_sec`` waits;
* used, run completed -> 1 when ``realized_nats > 0`` (at least one belief moved), else 0.

Metric quality gate for this grade: docs/superpowers/pr-reports/2026-11-04-curiosity-replica-instance-pr.md.

The track record shows the replica, for its last plans, whether each pick was tested and by how many nats
it moved; pick uptake; and its fallback count. It never shows the arm comparison or any p-value (design
section 2: showing the running score invites chasing noise).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional, Sequence

KIND_TESTED = "tested"
SettleKind = Literal["graded", "pending", "ungradable"]


@dataclass(frozen=True)
class PlanSettlement:
    kind: SettleKind
    produced: bool = False
    note: str = ""


def _json(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def pick_ids(row: Mapping[str, Any]) -> list[str]:
    picks = _json(row.get("picks")) or []
    return [str(p.get("prior_id")) for p in picks if isinstance(p, dict) and p.get("prior_id")]


def tested_nats(row: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """prior_id -> nats, for every prior the run tested (None: tested but unscorable)."""
    out: dict[str, Optional[float]] = {}
    for p in _json(row.get("per_prior")) or []:
        if isinstance(p, dict) and p.get("kind") == KIND_TESTED and p.get("prior_id"):
            n = p.get("nats")
            out[str(p["prior_id"])] = float(n) if isinstance(n, (int, float)) else None
    return out


def _fmt_nats(n: Optional[float]) -> str:
    return "unknown" if n is None else f"{n:.3f}"


def _aware(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def settle_plan(row: Mapping[str, Any], *, now: Optional[datetime] = None,
                failed_final_sec: float = 0.0) -> PlanSettlement:
    status = str(row.get("status") or "")
    if status != "ok":
        reason = row.get("fallback_reason") or status
        return PlanSettlement("graded", False, f"plan not used: {status} ({reason}); the run had the rule order")
    if row.get("turn_ok") is None:
        return PlanSettlement("pending")
    nats = row.get("realized_nats")
    if nats is None:
        return PlanSettlement("ungradable", False,
                              f"the run's score is unknown ({row.get('unknown_reason') or 'no reason recorded'})")
    picks, tested = pick_ids(row), tested_nats(row)
    hit = [p for p in picks if p in tested]
    pick_part = (f"Orion tested {len(hit)} of my {len(picks)} picks"
                 + (" (" + ", ".join(f"{p}: {_fmt_nats(tested[p])} nats" for p in hit) + ")" if hit else ""))
    if not row.get("turn_ok"):
        done = _aware(row.get("completed_at"))
        if now is not None and failed_final_sec > 0 and (done is None or (now - done).total_seconds() < failed_final_sec):
            return PlanSettlement("pending")  # a durable retry may still replace this failure
        return PlanSettlement("graded", False, f"the run failed (turn_ok=false); {pick_part}")
    produced = float(nats) > 0.0
    return PlanSettlement("graded", produced,
                          f"the run moved {float(nats):.3f} nats in all ({'some belief moved' if produced else 'nothing moved'}); "
                          f"{pick_part}")


def record_lines(rows: Sequence[Mapping[str, Any]], *, fallback_counts: Optional[Mapping[str, int]] = None,
                 plans_total: Optional[int] = None) -> list[str]:
    """The TRACK RECORD section, newest first. ``rows``: the last plans (plan row + outcome)."""
    if not rows:
        return ["(no plans yet)"]
    lines: list[str] = []
    picked = tested_total = 0
    for row in rows:
        when = row.get("requested_at")
        stamp = when.strftime("%m-%d %H:%M") if isinstance(when, datetime) else "?"
        head = f"- {stamp} run {row.get('run_id')}:"
        status = str(row.get("status") or "")
        if status != "ok":
            lines.append(f"{head} fallback ({status}); Orion was shown the rule order")
            continue
        picks = pick_ids(row)
        if row.get("turn_ok") is None:
            lines.append(f"{head} plan used ({len(picks)} picks); the run is not scored yet")
            continue
        tested = tested_nats(row)
        hit = [p for p in picks if p in tested]
        picked += len(picks)
        tested_total += len(hit)
        detail = ", ".join(f"{p} moved {_fmt_nats(tested[p])} nats" for p in hit) or "none of them"
        run_part = ("the run failed" if not row.get("turn_ok")
                    else f"the run moved {_fmt_nats(row.get('realized_nats'))} nats in all")
        lines.append(f"{head} plan used; Orion tested {len(hit)} of {len(picks)} picks ({detail}); {run_part}")
    lines.append(f"pick uptake (used plans, scored runs): {tested_total} of {picked} picks tested")
    if fallback_counts is not None:
        total = plans_total if plans_total is not None else sum(fallback_counts.values())
        fb = sum(v for k, v in fallback_counts.items() if k != "ok")
        parts = ", ".join(f"{k} {v}" for k, v in sorted(fallback_counts.items()) if k != "ok")
        lines.append(f"fallbacks, last 30 days: {fb} of {total} plans" + (f" ({parts})" if parts else ""))
    return lines


__all__ = ["PlanSettlement", "pick_ids", "record_lines", "settle_plan", "tested_nats"]
