from __future__ import annotations

from typing import Any

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.substrate.appraisal.view_model import pressure_label


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
