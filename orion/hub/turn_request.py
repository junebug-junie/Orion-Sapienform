from __future__ import annotations

from typing import Any

from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1


def build_orion_turn_request(
    *,
    correlation_id: str,
    session_id: str | None,
    user_message: str,
    repair_bundle: TurnAppraisalBundleV1 | None,
) -> dict[str, Any]:
    """Thin Orion-mode turn dict — not the Brain chat request builder."""
    req: dict[str, Any] = {
        "mode": "orion",
        "correlation_id": correlation_id,
        "session_id": session_id,
        "user_message": user_message,
    }
    if repair_bundle is not None:
        req["repair_bundle"] = repair_bundle.model_dump(mode="json")
    return req
