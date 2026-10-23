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
    assert bundle["failure_reasons"] == {"not_a_real_paradigm": "unknown_paradigm"}


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

    async def _fake_llm_probe(_bus, *, prompt, route, timeout_sec, turn_correlation_id=None):
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


def _repair_request_env(corr: str) -> BaseEnvelope:
    payload = PreTurnAppraisalRequestV1(
        correlation_id=corr,
        session_id="sess",
        turn_window=[TurnWindowMessageV1(role="user", content="you did it again")],
        paradigms_requested=["repair_pressure"],
    ).model_dump(mode="json")
    return BaseEnvelope(
        kind="pre_turn_appraisal.request.v1",
        source=ServiceRef(name="orion-hub", version="test"),
        correlation_id=corr,
        payload=payload,
    )


def test_probe_route_defaults_to_quick_turn() -> None:
    """quick_turn = fast class at interactive pool priority: the heat reflex never sheds it."""
    from app.settings import Settings
    from orion.gpu_pool.config import load_pool_config

    field = Settings.model_fields["repair_pressure_probe_route"]
    assert field.default == "quick_turn"
    assert load_pool_config().routes["quick_turn"].priority == "interactive"


@pytest.mark.asyncio
async def test_shed_probe_is_a_failed_paradigm_with_its_reason_not_a_zero_level(monkeypatch) -> None:
    """Regression (2026-10-20..21): a heat-shed probe came back as level=0.0 and Hub read `ok`."""
    calls: list[dict] = []

    async def _shed_probe(_bus, *, prompt, route, timeout_sec, turn_correlation_id=None):
        calls.append({"route": route, "turn_correlation_id": turn_correlation_id})
        return normalize_llm_gateway_probe_payload(
            {"text": "", "content": "", "raw": {"error": "gpu_pool_unavailable",
                                                "details": {"reason": "shed:cabinet_hot"}}}
        )

    import app.pre_turn_appraisal as handler_module

    monkeypatch.setattr(handler_module, "_llm_probe_call", _shed_probe)
    monkeypatch.setattr(handler_module, "_BUS", object())
    corr = "00000000-0000-4000-8000-0000000000a1"
    bundle = (await handle_pre_turn_appraisal_request(_repair_request_env(corr))).payload
    assert bundle["failed_paradigms"] == ["repair_pressure"]
    assert bundle["failure_reasons"] == {"repair_pressure": "shed:cabinet_hot"}
    assert "repair_pressure" not in bundle["paradigms"]          # no slice, so no 0.0 level
    assert "repair_pressure" not in bundle["grammar_scalars"]
    assert calls == [{"route": handler_module.settings.repair_pressure_probe_route,
                      "turn_correlation_id": corr}]


@pytest.mark.asyncio
async def test_probe_envelope_carries_the_turn_correlation_id() -> None:
    """The gateway leases with turn_correlation_id=env.correlation_id, so the probe's pool events
    join to the turn only if its envelope carries the turn's id. The reply channel stays unique."""
    import app.pre_turn_appraisal as handler_module

    sent: list = []

    class _Codec:
        def decode(self, _data):
            class _D:
                ok = False
                envelope = None
            return _D()

    class _Bus:
        codec = _Codec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            sent.append((env, reply_channel))
            return {"data": b""}

    corr = "00000000-0000-4000-8000-0000000000b2"
    out = await handler_module._llm_probe_call(
        _Bus(), prompt="p", route="quick_turn", timeout_sec=1.0, turn_correlation_id=corr
    )
    (env, reply_channel), = sent
    assert str(env.correlation_id) == corr
    assert corr not in reply_channel
    assert env.payload["route"] == "quick_turn"
    # an undecodable reply is a stated failure, not an empty "calm" answer
    assert out["raw"]["error"] == "llm_reply_undecodable"
    # a non-UUID turn id falls back to a fresh one rather than breaking the envelope
    assert handler_module._envelope_correlation_id("not-a-uuid") != "not-a-uuid"
