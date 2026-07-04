from __future__ import annotations

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from app.pre_turn_appraisal import handle_pre_turn_appraisal_request, normalize_llm_gateway_probe_payload


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


def test_normalize_llm_gateway_probe_payload_maps_content_and_meta_uncertainty() -> None:
    normalized = normalize_llm_gateway_probe_payload(
        {
            "content": "specificity_demand: YES\n",
            "meta": {
                "llm_uncertainty": {
                    "available": True,
                    "content": [
                        {
                            "token": "YES",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "YES", "logprob": -0.1},
                                {"token": "NO", "logprob": -2.0},
                            ],
                        }
                    ],
                }
            },
            "raw": {"probs": []},
        }
    )
    assert normalized["text"] == "specificity_demand: YES\n"
    assert normalized["llm_uncertainty"]["available"] is True


@pytest.mark.asyncio
async def test_handler_parses_gateway_shaped_llm_probe(monkeypatch) -> None:
    """End-to-end: gateway ChatResultPayload shape → non-zero repair level when thread warrants it."""
    captured: list[str] = []

    async def _fake_llm_probe(_bus, *, prompt, route, timeout_sec):
        captured.append(prompt)
        return normalize_llm_gateway_probe_payload(
            {
                "content": "\n".join(
                    f"{kind}: YES"
                    for kind in (
                        "specificity_demand",
                        "trust_rupture",
                        "coherence_gap",
                        "repetition_failure",
                        "operational_block",
                        "explicit_repair_command",
                        "assistant_accountability_demand",
                    )
                ),
                "meta": {
                    "llm_uncertainty": {
                        "available": True,
                        "content": [
                            {
                                "token": "YES",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "YES", "logprob": -0.1},
                                    {"token": "NO", "logprob": -2.5},
                                ],
                            }
                        ]
                        * 7,
                    }
                },
                "raw": {},
            }
        )

    import app.pre_turn_appraisal as handler_module

    monkeypatch.setattr(handler_module, "_llm_probe_call", _fake_llm_probe)
    monkeypatch.setattr(handler_module, "_BUS", object())

    payload = PreTurnAppraisalRequestV1(
        correlation_id="00000000-0000-4000-8000-000000000004",
        session_id="sess",
        turn_window=[
            TurnWindowMessageV1(role="user", content="you gave garbage directions — again"),
            TurnWindowMessageV1(role="assistant", content="Here is another high-level overview."),
            TurnWindowMessageV1(
                role="user",
                content="Stop hand waving. Build me a spec with file boundaries and tests.",
            ),
        ],
        paradigms_requested=["repair_pressure"],
    ).model_dump(mode="json")

    env = BaseEnvelope(
        kind="pre_turn_appraisal.request.v1",
        source=ServiceRef(name="orion-hub", version="test"),
        correlation_id="00000000-0000-4000-8000-000000000004",
        payload=payload,
    )
    reply = await handle_pre_turn_appraisal_request(env)
    bundle = reply.payload
    rp = bundle["paradigms"]["repair_pressure"]
    assert rp["level"] >= 0.45, rp
    assert bundle["metadata_attachments"].get("repair_pressure_contract", {}).get("mode") in {
        "repair_concrete",
        "concrete_bias",
    }
    assert captured, "probe should have run"
