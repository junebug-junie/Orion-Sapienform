"""Settle a render_scene dispatch result from its `reverie.visual` durable run.

cortex-exec's RenderSceneVerb returns as soon as the durable run is registered:
`visual_outcome="unknown"`, `settlement.state="pending"`. execution-dispatch
stores that row with NULL `latency_ms` and emits nothing. Later, the run's
terminal row in `substrate_durable_run_state` says what really happened; this
module turns (pending result row, terminal run row) into the settled row.

Rules (docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md,
Settlement): a run that ends without an image is never Orion failing. It settles
`visual_outcome="unknown"` with a reason, `success=False`, and `latency_ms` NULL
-- no motor seconds charged, no cost sample. Only a produced image carries the
run's own measured GPU seconds (`detail.visual_elapsed_sec`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, get_args

from orion.schemas.reverie_visual import VisualRunOutcome

RENDER_SCENE_VERB = "skills.imagination.render_scene.v1"
VISUAL_OUTCOMES: frozenset[str] = frozenset(get_args(VisualRunOutcome))
TERMINAL_RUN_STATUSES: tuple[str, ...] = ("completed", "failed", "cancelled", "abandoned")
# How long past the run's own deadline a pending row waits for a terminal state
# before it is settled as an orphan. Covers durable-runs finishing its last retry
# and the state event crossing sql-writer.
SETTLEMENT_ORPHAN_MARGIN_SEC = 1800.0
# Only for a pending row that lost its deadline_at: the visual baseline interval.
DEFAULT_RETRY_WINDOW_SEC = 5400.0
SETTLEMENT_PENDING = "pending"
SETTLEMENT_SETTLED = "settled"
# The kickoff receipt was not confirmed (e.g. RPC timeout), but orch may still
# have admitted the run. Settled only if a terminal run row shows up -- never by
# orphan timeout, since the run may never have existed.
SETTLEMENT_NOT_SUBMITTED = "not_submitted"
SETTLEABLE_STATES: tuple[str, ...] = (SETTLEMENT_PENDING, SETTLEMENT_NOT_SUBMITTED)


def settlement_of(result: Any) -> dict[str, Any] | None:
    """The settlement block on a render_scene verb result (or stored result_json)."""
    if not isinstance(result, dict):
        return None
    settlement = result.get("settlement")
    return settlement if isinstance(settlement, dict) else None


def is_pending(result: Any) -> bool:
    settlement = settlement_of(result)
    return bool(settlement) and settlement.get("state") == SETTLEMENT_PENDING


def is_unsettled(result: Any) -> bool:
    """Pending, or an unconfirmed kickoff: no outcome may be emitted for it
    except by settlement."""
    settlement = settlement_of(result)
    return bool(settlement) and settlement.get("state") in SETTLEABLE_STATES


def verb_settlement(structured: Any) -> dict[str, Any] | None:
    """The settlement block of a parsed render_scene verb result, whether the
    verb's dict sits at the top level or under `result`."""
    if not isinstance(structured, dict):
        return None
    inner = structured.get("result") if isinstance(structured.get("result"), dict) else structured
    return settlement_of(inner)


def normalize_visual_outcome(value: Any) -> str:
    return value if isinstance(value, str) and value in VISUAL_OUTCOMES else "unknown"


@dataclass(frozen=True)
class VisualSettlement:
    status: str
    result_json: dict[str, Any]
    latency_ms: float | None
    visual_outcome: str
    success: bool
    summary: str


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _elapsed_ms(detail: dict[str, Any]) -> float | None:
    raw = detail.get("visual_elapsed_sec")
    # durable-runs writes 0.0 when it recorded nothing; a produced image is never free.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return None
    return float(raw) * 1000.0


def _settled_structured_result(
    structured: Any, settled_block: dict[str, Any], visual_outcome: str
) -> dict[str, Any] | None:
    """The verb's own result copy, so it no longer claims the submit-time pending state."""
    if not isinstance(structured, dict):
        return None
    inner = structured.get("result") if isinstance(structured.get("result"), dict) else structured
    if settlement_of(inner) is None:
        return None
    updated_inner = {**inner, "settlement": settled_block, "outcome": visual_outcome}
    if inner is structured:
        return updated_inner
    return {**structured, "result": updated_inner}


def settle_visual_result(
    *,
    result_json: dict[str, Any],
    run_status: str | None,
    run_detail: Any,
    created_at: datetime | None,
    now: datetime,
    dispatch_kind: str | None = None,
    target_id: str | None = None,
    orphan_margin_sec: float = SETTLEMENT_ORPHAN_MARGIN_SEC,
) -> VisualSettlement | None:
    """The settled row, or None while the run is still genuinely in flight."""
    settlement = settlement_of(result_json)
    if settlement is None or settlement.get("state") not in SETTLEABLE_STATES:
        return None
    if settlement.get("state") == SETTLEMENT_NOT_SUBMITTED and (
        run_status not in TERMINAL_RUN_STATUSES or not settlement.get("durable_run_id")
    ):
        return None
    detail = run_detail if isinstance(run_detail, dict) else {}
    where = f"{dispatch_kind or 'express'} on {target_id or 'unknown target'}"

    if run_status in TERMINAL_RUN_STATUSES:
        reason: str | None
        if run_status == "completed":
            visual_outcome = normalize_visual_outcome(detail.get("outcome"))
            reason = None if visual_outcome == "produced" else str(
                detail.get("reason") or detail.get("terminal_reason") or f"run_outcome:{visual_outcome}"
            )
        else:
            visual_outcome = "unknown"
            reason = str(detail.get("error") or detail.get("last_error") or f"run_{run_status}")
        durable_status = run_status
    else:
        deadline = _parse_ts(settlement.get("deadline_at"))
        if deadline is None and created_at is not None:
            deadline = _parse_ts(created_at) + timedelta(seconds=DEFAULT_RETRY_WINDOW_SEC)
        if deadline is None or now < deadline + timedelta(seconds=orphan_margin_sec):
            return None
        visual_outcome = "unknown"
        reason = "settlement_timeout"
        durable_status = None

    produced = visual_outcome == "produced"
    latency_ms = _elapsed_ms(detail) if produced else None
    settled_block = {
        **settlement,
        "state": SETTLEMENT_SETTLED,
        "settled_at": now.isoformat(),
        "durable_status": durable_status,
    }
    if durable_status is not None:
        # When the run actually ended (queue + hold + GPU), for consumers that
        # must wait for the image itself -- latency_ms is GPU seconds only.
        finished = _parse_ts(detail.get("finished_at"))
        settled_block["finished_at"] = (finished or now).isoformat()
    if settlement.get("state") == SETTLEMENT_NOT_SUBMITTED:
        settled_block["settled_from"] = SETTLEMENT_NOT_SUBMITTED
    if reason is not None:
        settled_block["reason"] = reason
    else:
        settled_block.pop("reason", None)
    if produced and latency_ms is None:
        settled_block["latency_missing"] = True

    receipt = detail.get("execution_receipt") if isinstance(detail.get("execution_receipt"), dict) else None
    settled_json = {
        **result_json,
        "visual_outcome": visual_outcome,
        "settlement": settled_block,
        "latency_ms": latency_ms,
    }
    nested = _settled_structured_result(result_json.get("structured_result"), settled_block, visual_outcome)
    if nested is not None:
        settled_json["structured_result"] = nested
    if receipt is not None:
        settled_json["execution_receipt"] = receipt
    if detail.get("chain_id"):
        settled_json["chain_id"] = detail.get("chain_id")

    if produced:
        summary = f"render on {where} produced an image"
        if latency_ms is not None:
            summary += f" ({latency_ms / 1000.0:.1f}s of GPU work)"
        status = "success"
    elif run_status == "completed" and visual_outcome in {"deferred_thermal", "deferred_busy", "deferred_resource", "already_satisfied"}:
        # The run answered for real (e.g. baseline already satisfied elsewhere):
        # the same verb-level success the direct path records for a refusal.
        summary = f"render on {where} settled {visual_outcome} without an image"
        status = "success"
    else:
        summary = f"render on {where} settled without an image ({reason}); not counted as a failure"
        status = "empty"
    return VisualSettlement(
        status=status,
        result_json=settled_json,
        latency_ms=latency_ms,
        visual_outcome=visual_outcome,
        success=produced,
        summary=summary,
    )
