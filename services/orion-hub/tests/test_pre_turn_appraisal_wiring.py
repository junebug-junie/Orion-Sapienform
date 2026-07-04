from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1, TurnAppraisalParadigmSliceV1
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY

_wiring_path = Path(__file__).resolve().parents[1] / "scripts" / "pre_turn_appraisal_wiring.py"
_spec = importlib.util.spec_from_file_location("hub_pre_turn_appraisal_wiring", _wiring_path)
assert _spec and _spec.loader
_wiring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wiring)
apply_pre_turn_appraisal_bundle = _wiring.apply_pre_turn_appraisal_bundle


def test_apply_bundle_attaches_metadata_when_mode_changes() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    bundle = TurnAppraisalBundleV1(
        correlation_id="c1",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.82,
                confidence=0.71,
                contract_delta={"mode": "repair_concrete", "rules": ["include file/module boundaries"]},
            )
        },
        metadata_attachments={
            "repair_pressure_contract": {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
        },
        grammar_scalars={"repair_pressure": {"level": 0.82, "confidence": 0.71}},
    )
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=True)
    assert req.metadata[REPAIR_PRESSURE_CONTRACT_METADATA_KEY]["mode"] == "repair_concrete"
    assert summary is not None
    assert summary["level"] == 0.82
    assert summary["changed_behavior"] == "repair_concrete"
    assert req.metadata.get("substrate_effect_summary") is None


def test_substrate_effect_summary_attached_to_request_metadata() -> None:
    """Mirrors run_pre_turn_appraisal_wiring post-apply metadata attach (lines 86-89)."""
    req = CortexChatRequest(prompt="paste test", mode="brain")
    bundle = TurnAppraisalBundleV1(
        correlation_id="00000000-0000-4000-8000-000000000005",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.698,
                confidence=0.65,
                contract_delta={"mode": "concrete_bias", "rules": ["be more specific"]},
            )
        },
        metadata_attachments={
            "repair_pressure_contract": {"mode": "concrete_bias", "rules": ["be more specific"]}
        },
    )
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=True)
    assert summary is not None
    meta = dict(req.metadata or {})
    meta["substrate_effect_summary"] = summary
    req.metadata = meta
    assert req.metadata["substrate_effect_summary"]["level_label"] == "MEDIUM"
    assert req.metadata[REPAIR_PRESSURE_CONTRACT_METADATA_KEY]["mode"] == "concrete_bias"


def test_apply_bundle_skips_when_disabled() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    bundle = TurnAppraisalBundleV1(correlation_id="c1")
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=False)
    assert summary is None
    assert req.metadata is None


@pytest.mark.asyncio
async def test_run_pre_turn_wiring_skips_when_bus_missing() -> None:
    _wiring_path = Path(__file__).resolve().parents[1] / "scripts" / "pre_turn_appraisal_wiring.py"
    _spec = importlib.util.spec_from_file_location("hub_pre_turn_appraisal_wiring_async", _wiring_path)
    assert _spec and _spec.loader
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    req = CortexChatRequest(prompt="hi", mode="brain")
    summary, bundle = await _mod.run_pre_turn_appraisal_wiring(
        req,
        bus=None,
        correlation_id="00000000-0000-4000-8000-000000000004",
        session_id="sess",
        continuity_messages=[{"role": "user", "content": "hello"}],
        user_prompt="hello",
        paradigms="repair_pressure",
        timeout_ms=800,
    )
    assert summary is None
    assert bundle is None
