"""Settle a ``shed_background_gpu`` dispatch from the GPU pool's ``orion_self_shed`` record.

Sibling of ``visual_settlement.py`` (the ``render_scene`` pattern every world action copies): the
dispatch writes ``shed_pending`` BEFORE the RPC, the pool's terminal state settles it, and the
outcome is emitted once, at settlement. Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-
design.md, "Amendment 2026-09-29" -> "Settle rule".

Terminal pool states: ``expired`` (TTL ended -- the only state that may update the posterior),
``cancelled`` (clear verb or kill switch), ``preempted_by_reflex`` (a cabinet AC incident opened),
``refused:<reason>`` (never started). No terminal state by ``t0 + TTL + 300 s`` -> orphan,
``settlement_timeout``. Its own state name (``shed_pending``, not the render path's ``pending``) so
the render-only parking and replay logic never mistakes one for the other.

Pure: the caller reads the result row and the pool row and persists what comes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

SHED_PENDING = "shed_pending"
SHED_SETTLED = "settled"
SHED_ORPHAN_MARGIN_SEC = 300.0
DEFAULT_SHED_TTL_SEC = 900.0
POOL_TERMINAL = ("expired", "cancelled", "preempted_by_reflex", "refused")


def _ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def is_shed_pending(result_json: Any) -> bool:
    settlement = (result_json or {}).get("settlement") if isinstance(result_json, dict) else None
    return isinstance(settlement, dict) and settlement.get("state") == SHED_PENDING


def terminal_label(pool_row: Mapping[str, Any] | None) -> str | None:
    """``expired`` | ``cancelled`` | ``preempted_by_reflex`` | ``refused:<reason>`` | None (still active)."""
    if not pool_row:
        return None
    state = str(pool_row.get("state") or "")
    if state not in POOL_TERMINAL:
        return None
    return f"refused:{pool_row.get('refusal') or 'unknown'}" if state == "refused" else state


def manipulation_check(pool_row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Did the shed actually happen, and did it withhold anything? Recorded, never a scoring gate."""
    if not pool_row:
        return {"drain": "unknown"}
    started, drained = _ts(pool_row.get("started_at")), _ts(pool_row.get("drained_at"))
    return {
        "shed_id": pool_row.get("shed_id"),
        "started_at": started.isoformat() if started else None,
        "ended_at": (_ts(pool_row.get("ended_at")).isoformat() if _ts(pool_row.get("ended_at")) else None),
        "drained_at": drained.isoformat() if drained else None,
        "drain_sec": round((drained - started).total_seconds(), 1) if started and drained else None,
        "drain": "drained" if drained else "none",
        "grants_withheld": int(pool_row.get("grants_withheld") or 0),
        "delayed_grant_sec": float(pool_row.get("delayed_grant_sec") or 0.0),
        "background_live_at_start": int(pool_row.get("background_live_at_start") or 0),
    }


@dataclass(frozen=True)
class ShedSettlement:
    status: str
    terminal: str
    result_json: dict[str, Any]
    success: bool
    summary: str


def settle_shed_result(
    *,
    result_json: dict[str, Any],
    pool_row: Mapping[str, Any] | None,
    now: datetime,
    orphan_margin_sec: float = SHED_ORPHAN_MARGIN_SEC,
) -> ShedSettlement | None:
    """The settled row, or None while the shed is genuinely still running."""
    settlement = result_json.get("settlement") if isinstance(result_json, dict) else None
    if not isinstance(settlement, dict) or settlement.get("state") != SHED_PENDING:
        return None
    terminal = terminal_label(pool_row)
    if terminal is None:
        decided = _ts(settlement.get("decided_at"))
        ttl = float(settlement.get("ttl_sec") or DEFAULT_SHED_TTL_SEC)
        if decided is None or now < decided + timedelta(seconds=ttl + orphan_margin_sec):
            return None
        terminal = "settlement_timeout"
    check = manipulation_check(pool_row)
    settled_block = {**settlement, "state": SHED_SETTLED, "terminal": terminal, "settled_at": now.isoformat(),
                     "manipulation_check": check}
    expired = terminal == "expired"
    summary = (
        f"orion_self_shed {terminal}: withheld {check.get('grants_withheld', 0)} background grant(s), "
        f"drain={check.get('drain')}"
    )
    return ShedSettlement(
        # `success` = the shed ran its course (the treatment happened). Whether the cabinet cooled is
        # the outcome, scored separately on the sensor -- never folded into this status.
        status="success" if expired else "empty",
        terminal=terminal,
        result_json={**result_json, "settlement": settled_block},
        success=expired,
        summary=summary,
    )


__all__ = ["POOL_TERMINAL", "SHED_PENDING", "SHED_SETTLED", "ShedSettlement", "is_shed_pending",
           "manipulation_check", "settle_shed_result", "terminal_label"]
