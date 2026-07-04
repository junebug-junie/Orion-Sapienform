"""Orchestrate the repair_pressure appraisal pipeline for one chat turn.

Failure must never propagate to the chat handler. On any exception we log
and return (None, None). The chat response then ships without a chip.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from typing import Any

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal import (
    apply_repair_pressure_contract,
    appraise_repair_pressure,
    extract_repair_evidence,
    repair_appraisal_to_signal,
    select_recent_chat_molecules,
)
from orion.substrate.appraisal.models import RepairPressureAppraisalV1
from orion.substrate.appraisal.view_model import pressure_label
from orion.substrate.molecules import SubstrateMoleculeV1

from .substrate_effect_cache import (
    SubstrateEffectCache,
    SubstrateEffectSnapshot,
    substrate_effect_cache,
)

logger = logging.getLogger("orion-hub.substrate_effect_pipeline")

# Per-source rolling buffer of observation molecules. Keyed by `source_id`
# (we use session_id or correlation_id). Bounded so memory stays flat.
_RECENT_OBSERVATIONS: OrderedDict[str, deque[SubstrateMoleculeV1]] = OrderedDict()
_RECENT_MAX = 32
_RECENT_SOURCES_MAX = 256


def _push_observation(source_id: str, mol: SubstrateMoleculeV1) -> list[SubstrateMoleculeV1]:
    buf = _RECENT_OBSERVATIONS.setdefault(source_id, deque(maxlen=_RECENT_MAX))
    buf.append(mol)
    _RECENT_OBSERVATIONS.move_to_end(source_id)
    while len(_RECENT_OBSERVATIONS) > _RECENT_SOURCES_MAX:
        _RECENT_OBSERVATIONS.popitem(last=False)
    return list(buf)


def _summary_dict(
    *,
    turn_id: str,
    appraisal: RepairPressureAppraisalV1 | None,
    contract_before: dict[str, Any],
    contract_after: dict[str, Any],
    evidence_count: int,
) -> dict[str, Any]:
    level = float(appraisal.dimensions.get("level", 0.0)) if appraisal else 0.0
    confidence = float(appraisal.confidence) if appraisal else 0.0
    level_lbl = pressure_label(level)
    before_mode = str(contract_before.get("mode") or "")
    after_mode = str(contract_after.get("mode") or "")
    changed = before_mode != after_mode
    behavior_applied = after_mode if changed else None
    chip_label = (
        f"{behavior_applied or 'no behavior change'} · "
        f"{level_lbl} repair pressure · "
        f"{evidence_count} evidence driver{'s' if evidence_count != 1 else ''}"
    )
    return {
        "turn_id": turn_id,
        "appraisal_kind": "repair_pressure" if appraisal else "none",
        "level": level,
        "level_label": level_lbl,
        "confidence": confidence,
        "behavior_applied": behavior_applied,
        "evidence_count": evidence_count,
        "changed_behavior": changed,
        "chip_label": chip_label,
    }


def run_substrate_effect_pipeline(
    *,
    turn_id: str,
    message_id: str | None,
    user_text: str,
    source_id: str,
    contract_before: dict[str, Any],
    cache: SubstrateEffectCache | None = None,
) -> tuple[dict[str, Any] | None, SubstrateEffectSnapshot | None]:
    """Run the appraiser end-to-end. Stash a snapshot in `cache`. Return summary."""

    from scripts.settings import settings as hub_settings

    if getattr(hub_settings, "ENABLE_PRE_TURN_APPRAISAL", False):
        logger.debug("substrate_effect_pipeline_skipped_v2_enabled turn_id=%s", turn_id)
        return None, None

    store_in = cache if cache is not None else substrate_effect_cache
    try:
        if not (user_text or "").strip():
            logger.debug("substrate_effect_pipeline_skipped_empty_text turn_id=%s", turn_id)
            return None, None

        mol = emit_observation(surface_text=user_text, source_id=source_id)
        window = select_recent_chat_molecules(_push_observation(source_id, mol), source_id=source_id)
        evidence = extract_repair_evidence(window)
        appraisal = appraise_repair_pressure(window, window_id=f"win-{turn_id}")
        signal = repair_appraisal_to_signal(appraisal)
        contract_after = apply_repair_pressure_contract(contract_before, signal)

        snapshot = SubstrateEffectSnapshot(
            turn_id=turn_id,
            message_id=message_id,
            user_text=user_text,
            appraisal=appraisal,
            signal=signal,
            evidence=list(evidence),
            contract_before=dict(contract_before),
            contract_after=dict(contract_after),
            causal_molecule_ids=list(appraisal.causal_molecule_ids),
        )
        store_in.store(snapshot)

        summary = _summary_dict(
            turn_id=turn_id,
            appraisal=appraisal,
            contract_before=contract_before,
            contract_after=contract_after,
            evidence_count=len(evidence),
        )
        return summary, snapshot
    except Exception:  # noqa: BLE001
        logger.warning("substrate_effect_pipeline_failed turn_id=%s", turn_id, exc_info=True)
        return None, None
