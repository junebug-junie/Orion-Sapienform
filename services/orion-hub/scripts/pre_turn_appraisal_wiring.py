from __future__ import annotations

import logging
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.cortex.contracts import CortexChatRequest
from orion.schemas.pre_turn_appraisal import (
    PreTurnAppraisalOptionsV1,
    PreTurnAppraisalRequestV1,
    TurnAppraisalBundleV1,
)
from orion.substrate.appraisal.turn_window import build_turn_window
from orion.substrate.appraisal.view_model import pressure_label

logger = logging.getLogger("hub.pre_turn_appraisal_wiring")


def apply_pre_turn_appraisal_bundle(
    req: CortexChatRequest,
    bundle: TurnAppraisalBundleV1 | None,
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled or bundle is None:
        return None
    if bundle.metadata_attachments:
        meta = dict(req.metadata or {})
        meta.update(bundle.metadata_attachments)
        req.metadata = meta
    rp = bundle.paradigms.get("repair_pressure")
    if rp is None:
        return None
    level = float(rp.level)
    confidence = float(rp.confidence)
    before_attached = bool(bundle.metadata_attachments)
    behavior = None
    if before_attached:
        contract = bundle.metadata_attachments.get("repair_pressure_contract") or rp.contract_delta
        behavior = str((contract or {}).get("mode") or "")
    return {
        "turn_id": bundle.correlation_id,
        "appraisal_kind": "repair_pressure",
        "level": level,
        "level_label": pressure_label(level),
        "confidence": confidence,
        "behavior_applied": behavior,
        "evidence_count": len(rp.evidence),
        "changed_behavior": behavior,
        "chip_label": f"{behavior or 'no behavior change'} · {pressure_label(level)} repair pressure · {len(rp.evidence)} evidence drivers",
    }


async def run_pre_turn_appraisal_wiring(
    req: CortexChatRequest,
    *,
    bus: OrionBusAsync | None,
    correlation_id: str,
    session_id: str,
    continuity_messages: list[dict[str, Any] | Any],
    user_prompt: str,
    paradigms: str,
    timeout_ms: int,
) -> tuple[dict[str, Any] | None, TurnAppraisalBundleV1 | None]:
    """Hub-side v2 wiring: build window, RPC appraise, attach bundle metadata."""
    if bus is None:
        logger.warning("pre_turn_appraisal_skipped_no_bus corr=%s", correlation_id)
        return None, None

    from scripts.pre_turn_appraisal_client import PreTurnAppraisalClient

    turn_window = build_turn_window(
        continuity_messages or [{"role": "user", "content": user_prompt}]
    )
    bundle = await PreTurnAppraisalClient(bus).appraise(
        PreTurnAppraisalRequestV1(
            correlation_id=correlation_id,
            session_id=session_id,
            turn_window=turn_window,
            paradigms_requested=[p.strip() for p in paradigms.split(",") if p.strip()],
            contract_before={"mode": "default"},
            options=PreTurnAppraisalOptionsV1(timeout_ms=timeout_ms),
        )
    )
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=True)
    if summary is not None:
        meta = dict(req.metadata or {})
        meta["substrate_effect_summary"] = summary
        req.metadata = meta
    return summary, bundle


def repair_pressure_grammar_scalars(
    *,
    pre_turn_bundle: TurnAppraisalBundleV1 | None,
    substrate_summary: dict[str, Any] | None,
) -> tuple[float, float]:
    if pre_turn_bundle is not None:
        scalars = (pre_turn_bundle.grammar_scalars or {}).get("repair_pressure") or {}
        return float(scalars.get("level", 0.0)), float(scalars.get("confidence", 0.0))
    return float((substrate_summary or {}).get("level", 0.0)), float(
        (substrate_summary or {}).get("confidence", 0.0)
    )
