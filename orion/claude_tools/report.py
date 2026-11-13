"""How Orion's consent answers look over time: the weekly numbers and the rubber-stamp flag. Pure.

Spec section 3, "How Orion's decisions are observed". Input: ``answer``-phase audit rows
(claude_tool_call_log). Per category: asks, yes rate, timeout rate (no answer in time, or the consent
call down), the share of model answers with a non-empty reason, p50/p90 consent latency, and flips (the
same tool answered yes and then no, or no and then yes, within 24 h).

Rubber-stamp: 20 or more model answers in a category, all yes or all no. A consent that cannot vary is
not a decision; then the standing setting is the honest one. Flagged here and shown in the next consent
frame (orion/claude_tools/broker.py ``history_note``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional

RUBBER_STAMP_MIN = 20
FLIP_WINDOW = timedelta(hours=24)


def _pct(values: List[int], q: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return int(ordered[idx])


def _at(row: Mapping[str, Any]) -> Optional[datetime]:
    v = row.get("created_at")
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def consent_summary(answer_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_cat: Dict[str, List[Mapping[str, Any]]] = {}
    for row in answer_rows:
        if row.get("phase") != "answer":
            continue
        by_cat.setdefault(str(row.get("category") or "unknown"), []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for cat, rows in sorted(by_cat.items()):
        rows = sorted(rows, key=lambda r: _at(r) or datetime.min)
        model = [r for r in rows if r.get("consent_path") == "model"]
        yes = [r for r in model if r.get("orion_answer") == "yes"]
        unanswered = [r for r in rows if not r.get("orion_answer")]
        latencies = [int(r["consent_latency_ms"]) for r in rows if isinstance(r.get("consent_latency_ms"), int)]
        flips = 0
        last_by_tool: Dict[str, Mapping[str, Any]] = {}
        for r in model:
            prev = last_by_tool.get(str(r.get("tool")))
            if prev is not None and prev.get("orion_answer") != r.get("orion_answer"):
                a, b = _at(prev), _at(r)
                if a and b and b - a <= FLIP_WINDOW:
                    flips += 1
            last_by_tool[str(r.get("tool"))] = r
        rubber = None
        if len(model) >= RUBBER_STAMP_MIN and len({r.get("orion_answer") for r in model}) == 1:
            rubber = f"{model[0].get('orion_answer')} {len(model)} of {len(model)}"
        out[cat] = {
            "asks": len(rows),
            "model_answers": len(model),
            "yes_rate": round(len(yes) / len(model), 3) if model else None,
            "unanswered_rate": round(len(unanswered) / len(rows), 3) if rows else None,
            "reason_share": (round(sum(1 for r in model if str(r.get("orion_reason") or "").strip()) / len(model), 3)
                             if model else None),
            "latency_ms_p50": _pct(latencies, 0.5),
            "latency_ms_p90": _pct(latencies, 0.9),
            "flips_24h": flips,
            "rubber_stamp": rubber,
        }
    return out


__all__ = ["FLIP_WINDOW", "RUBBER_STAMP_MIN", "consent_summary"]
