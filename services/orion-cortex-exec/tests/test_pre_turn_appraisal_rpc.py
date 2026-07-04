from __future__ import annotations

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from app.pre_turn_appraisal import handle_pre_turn_appraisal_request


@pytest.mark.asyncio
async def test_handler_returns_bundle_with_correlation_id(monkeypatch) -> None:
    from orion.schemas.pre_turn_appraisal import TurnAppraisalParadigmSliceV1

    class _FakeParadigm:
        name = "repair_pressure"

        async def run(self, _req):
            return TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.80,
                confidence=0.70,
                dimensions={"level": 0.80},
                contract_delta={"mode": "repair_concrete", "rules": ["be specific"]},
            )

    def _fake_factory(_ctx):
        return _FakeParadigm()

    import app.pre_turn_appraisal as handler_module

    monkeypatch.setitem(handler_module.PARADIGM_REGISTRY, "repair_pressure", _fake_factory)
    monkeypatch.setattr("app.pre_turn_appraisal._BUS", object())

    payload = PreTurnAppraisalRequestV1(
        correlation_id="00000000-0000-4000-8000-000000000001",
        session_id="sess",
        turn_window=[TurnWindowMessageV1(role="user", content="nuts and bolts")],
    ).model_dump(mode="json")

    env = BaseEnvelope(
        kind="pre_turn_appraisal.request.v1",
        source=ServiceRef(name="orion-hub", version="test"),
        correlation_id="00000000-0000-4000-8000-000000000001",
        payload=payload,
    )
    reply = await handle_pre_turn_appraisal_request(env)
    bundle = reply.payload
    assert bundle["correlation_id"] == "00000000-0000-4000-8000-000000000001"
    assert "repair_pressure" in bundle["paradigms"]


@pytest.mark.asyncio
async def test_handler_marks_unknown_paradigm_failed(monkeypatch) -> None:
    monkeypatch.setattr("app.pre_turn_appraisal._BUS", object())

    payload = PreTurnAppraisalRequestV1(
        correlation_id="00000000-0000-4000-8000-000000000002",
        session_id="sess",
        turn_window=[TurnWindowMessageV1(role="user", content="nuts and bolts")],
        paradigms_requested=["not_a_real_paradigm"],
    ).model_dump(mode="json")

    env = BaseEnvelope(
        kind="pre_turn_appraisal.request.v1",
        source=ServiceRef(name="orion-hub", version="test"),
        correlation_id="00000000-0000-4000-8000-000000000002",
        payload=payload,
    )
    reply = await handle_pre_turn_appraisal_request(env)
    bundle = reply.payload
    assert bundle["failed_paradigms"] == ["not_a_real_paradigm"]
    assert bundle["paradigms"] == {}
