"""Wire substrate repair contract into Hub → Cortex chat metadata."""

from __future__ import annotations

from typing import Any

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY

from .substrate_effect_cache import SubstrateEffectSnapshot


def _contract_changed(snapshot: SubstrateEffectSnapshot) -> bool:
    before = str((snapshot.contract_before or {}).get("mode") or "")
    after = str((snapshot.contract_after or {}).get("mode") or "")
    return before != after


def attach_repair_pressure_contract(
    req: CortexChatRequest,
    snapshot: SubstrateEffectSnapshot | None,
    *,
    enabled: bool,
) -> None:
    """Mutate req.metadata in place when repair pressure changed behavior."""
    if not enabled or snapshot is None or not _contract_changed(snapshot):
        return
    meta: dict[str, Any] = dict(req.metadata or {})
    meta[REPAIR_PRESSURE_CONTRACT_METADATA_KEY] = dict(snapshot.contract_after)
    req.metadata = meta
