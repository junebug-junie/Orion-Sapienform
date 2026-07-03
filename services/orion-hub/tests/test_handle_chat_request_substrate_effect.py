from __future__ import annotations

import asyncio

import pytest

from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from scripts import api_routes
from scripts.substrate_effect_cache import substrate_effect_cache


class _FakeCortex:
    last_req = None

    async def chat(self, req, correlation_id=None):
        self.last_req = req
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_general",
            status="success",
            final_text="ok",
            correlation_id=correlation_id or "corr-substrate",
        )
        return CortexChatResult(cortex_result=result, final_text="ok")


@pytest.fixture(autouse=True)
def _reset_cache():
    substrate_effect_cache.clear()
    yield


def _make_payload(text: str) -> dict:
    return {"mode": "brain", "messages": [{"role": "user", "content": text}]}


def test_handle_chat_request_attaches_repair_summary_for_high_pressure():
    cortex = _FakeCortex()
    payload = _make_payload(
        "you gave me garbage directions — stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    result = asyncio.run(api_routes.handle_chat_request(cortex, payload, "session-x", no_write=True))
    summary = result.get("substrate_effect_summary")
    assert summary is not None, "chat result must carry substrate_effect_summary"
    assert summary["appraisal_kind"] == "repair_pressure"
    assert isinstance(summary["chip_label"], str) and summary["chip_label"]
    assert summary["evidence_count"] >= 1


def test_handle_chat_request_summary_marks_no_change_for_benign():
    cortex = _FakeCortex()
    payload = _make_payload("what's the weather like in Paris?")
    result = asyncio.run(api_routes.handle_chat_request(cortex, payload, "session-y", no_write=True))
    summary = result.get("substrate_effect_summary")
    assert summary is not None
    assert summary["changed_behavior"] is False
    assert summary["behavior_applied"] is None
    assert cortex.last_req is not None
    assert REPAIR_PRESSURE_CONTRACT_METADATA_KEY not in (cortex.last_req.metadata or {})


def test_handle_chat_request_attaches_repair_contract_metadata_on_high_pressure():
    cortex = _FakeCortex()
    payload = _make_payload(
        "you gave me garbage directions — stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    asyncio.run(api_routes.handle_chat_request(cortex, payload, "session-x", no_write=True))
    assert cortex.last_req is not None
    md = cortex.last_req.metadata or {}
    contract = md.get(REPAIR_PRESSURE_CONTRACT_METADATA_KEY)
    assert isinstance(contract, dict)
    assert contract.get("mode") in {"repair_concrete", "concrete_bias"}
    assert contract.get("rules")


def test_handle_chat_request_omits_repair_contract_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(api_routes.settings, "ENABLE_REPAIR_PRESSURE_SPEECH_WIRING", False)
    cortex = _FakeCortex()
    payload = _make_payload(
        "you gave me garbage directions — stop, build me a design spec for claude, "
        "arsonist pov only, nuts and bolts"
    )
    asyncio.run(api_routes.handle_chat_request(cortex, payload, "session-flag-off", no_write=True))
    assert cortex.last_req is not None
    assert REPAIR_PRESSURE_CONTRACT_METADATA_KEY not in (cortex.last_req.metadata or {})
