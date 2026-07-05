from __future__ import annotations

import logging
from typing import Any

from orion.hub.association import build_hub_association_bundle
from orion.hub.turn_request import build_orion_turn_request
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.schemas.thought import StanceReactRequestV1
from scripts.thought_client import ThoughtClient

logger = logging.getLogger("hub.unified_turn_stub")


async def run_orion_unified_turn_stub(
    *,
    bus: Any,
    correlation_id: str,
    session_id: str | None,
    user_message: str,
    repair_bundle: TurnAppraisalBundleV1 | None = None,
) -> dict[str, Any]:
    """Partial unified-turn wire: build turn request + optional thought RPC until Task 18."""
    build_orion_turn_request(
        correlation_id=correlation_id,
        session_id=session_id,
        user_message=user_message,
        repair_bundle=repair_bundle,
    )
    thought_received = False
    if bus is not None:
        association = build_hub_association_bundle(
            correlation_id=correlation_id,
            repair_bundle=repair_bundle,
        )
        stance_req = StanceReactRequestV1(
            correlation_id=correlation_id,
            session_id=session_id,
            user_message=user_message,
            association=association,
            repair_bundle=repair_bundle,
            stance_inputs={"user_message": user_message},
        )
        try:
            thought = await ThoughtClient(bus).react(stance_req, correlation_id=correlation_id)
            thought_received = thought is not None
        except Exception:
            logger.debug("unified turn stub thought RPC failed", exc_info=True)
    return {
        "type": "turn_deferred",
        "reason": "unified turn wiring pending",
        "correlation_id": correlation_id,
        "thought_received": thought_received,
    }
