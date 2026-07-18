from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1, TurnAppraisalParadigmSliceV1
from orion.schemas.repair_evidence import RepairEvidenceV1
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


def test_apply_bundle_includes_typed_evidence_breakdown() -> None:
    """evidence_count alone was forwarded before; now the full typed
    evidence_kind/score/confidence breakdown is forwarded too (Task 4)."""
    req = CortexChatRequest(prompt="hi", mode="brain")
    bundle = TurnAppraisalBundleV1(
        correlation_id="c1",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.8,
                confidence=0.85,
                evidence=[
                    RepairEvidenceV1(
                        evidence_id="e1",
                        source_molecule_id="m1",
                        evidence_kind="trust_rupture",
                        detector="d1",
                        score=0.7,
                        confidence=0.9,
                    )
                ],
            )
        },
    )
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=True)
    assert summary is not None
    assert summary["evidence_count"] == 1
    assert summary["evidence"] == [
        {"evidence_kind": "trust_rupture", "score": 0.7, "confidence": 0.9}
    ]


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


_REQUIRED_HUB_ENV = {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}


@pytest.mark.asyncio
async def test_publish_repair_pressure_appraisal_publishes_to_new_channel(monkeypatch) -> None:
    """Option A (new channel), per the redesign doc: repair_pressure evidence is
    published as its own envelope on orion:repair_pressure:appraisal so
    orion-equilibrium-service's repair_pressure_metacog_gate can consume it."""
    import sys

    for key, value in _REQUIRED_HUB_ENV.items():
        monkeypatch.setenv(key, value)
    for mod_name in ("app.settings", "scripts.settings"):
        sys.modules.pop(mod_name, None)

    _wiring_path = Path(__file__).resolve().parents[1] / "scripts" / "pre_turn_appraisal_wiring.py"
    _spec = importlib.util.spec_from_file_location("hub_pre_turn_appraisal_wiring_publish", _wiring_path)
    assert _spec and _spec.loader
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    mock_bus = AsyncMock()
    summary = {
        "level": 0.8,
        "level_label": "HIGH",
        "confidence": 0.85,
        "evidence": [{"evidence_kind": "trust_rupture", "score": 0.7, "confidence": 0.9}],
        "behavior_applied": "repair_concrete",
    }

    await _mod._publish_repair_pressure_appraisal(
        mock_bus,
        correlation_id="00000000-0000-4000-8000-000000000009",
        summary=summary,
    )

    mock_bus.publish.assert_called_once()
    channel_arg, env = mock_bus.publish.call_args[0]
    assert channel_arg == "orion:repair_pressure:appraisal"
    assert env.kind == "repair_pressure.appraisal.v1"
    assert env.payload["level"] == 0.8
    assert env.payload["evidence"] == summary["evidence"]
    assert env.payload["behavior_applied"] == "repair_concrete"


@pytest.mark.asyncio
async def test_publish_repair_pressure_appraisal_noop_when_bus_none() -> None:
    _wiring_path = Path(__file__).resolve().parents[1] / "scripts" / "pre_turn_appraisal_wiring.py"
    _spec = importlib.util.spec_from_file_location("hub_pre_turn_appraisal_wiring_publish_noop", _wiring_path)
    assert _spec and _spec.loader
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    # Should not raise even though bus is None.
    await _mod._publish_repair_pressure_appraisal(None, correlation_id="00000000-0000-4000-8000-000000000009", summary={"level": 0.5})
