from __future__ import annotations

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from scripts.repair_pressure_wiring import attach_repair_pressure_contract
from scripts.substrate_effect_cache import SubstrateEffectSnapshot
from scripts.substrate_effect_pipeline import run_substrate_effect_pipeline


def test_metadata_key_is_stable_string() -> None:
    assert REPAIR_PRESSURE_CONTRACT_METADATA_KEY == "repair_pressure_contract"


def _snapshot(*, before_mode: str, after_mode: str, rules: list[str] | None = None) -> SubstrateEffectSnapshot:
    return SubstrateEffectSnapshot(
        turn_id="t1",
        message_id=None,
        user_text="x",
        appraisal=None,
        signal=None,
        contract_before={"mode": before_mode},
        contract_after={"mode": after_mode, "rules": rules or ["be more specific"]},
    )


def test_attach_skips_when_snapshot_none() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    attach_repair_pressure_contract(req, None, enabled=True)
    assert req.metadata is None or REPAIR_PRESSURE_CONTRACT_METADATA_KEY not in (req.metadata or {})


def test_attach_skips_when_disabled() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    attach_repair_pressure_contract(req, _snapshot(before_mode="default", after_mode="repair_concrete"), enabled=False)
    assert req.metadata is None or REPAIR_PRESSURE_CONTRACT_METADATA_KEY not in (req.metadata or {})


def test_attach_skips_when_mode_unchanged() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    attach_repair_pressure_contract(req, _snapshot(before_mode="default", after_mode="default"), enabled=True)
    assert req.metadata is None or REPAIR_PRESSURE_CONTRACT_METADATA_KEY not in (req.metadata or {})


def test_attach_writes_contract_when_mode_changed() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain", metadata={"source": "hub_http"})
    snap = _snapshot(before_mode="default", after_mode="repair_concrete", rules=["include tests/acceptance checks"])
    attach_repair_pressure_contract(req, snap, enabled=True)
    assert isinstance(req.metadata, dict)
    payload = req.metadata[REPAIR_PRESSURE_CONTRACT_METADATA_KEY]
    assert payload["mode"] == "repair_concrete"
    assert "include tests/acceptance checks" in payload["rules"]


def test_ws_path_can_attach_after_pipeline_before_cortex():
    """WS builds chat_req before pipeline; attach must run after pipeline."""
    high = (
        "you gave me garbage directions — stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    _, snap = run_substrate_effect_pipeline(
        turn_id="ws-turn",
        message_id=None,
        user_text=high,
        source_id="sess-ws",
        contract_before={"mode": "default"},
    )
    req = CortexChatRequest(prompt=high, mode="brain")
    attach_repair_pressure_contract(req, snap, enabled=True)
    assert REPAIR_PRESSURE_CONTRACT_METADATA_KEY in (req.metadata or {})
