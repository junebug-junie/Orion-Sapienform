from __future__ import annotations

from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.substrate_telemetry import SubstrateTierOutcomesPayloadV1


def test_codec_roundtrip_tier_outcomes_envelope() -> None:
    corr = uuid4()
    payload = SubstrateTierOutcomesPayloadV1(
        generated_at="2026-05-14T12:00:00+00:00",
        cold_anchors=["a1"],
        tier_outcomes={"a1": ["operator_static_protected:2"]},
        degraded_producers=["p1"],
    )
    env = BaseEnvelope(
        kind="substrate.tier_outcomes.v1",
        source=ServiceRef(name="orion-cortex-exec", node="n1"),
        correlation_id=corr,
        payload=payload.model_dump(mode="json"),
    )
    raw = OrionCodec().encode(env)
    dec = OrionCodec().decode(raw)
    assert dec.ok
    assert dec.envelope is not None
    assert dec.envelope.kind == "substrate.tier_outcomes.v1"
    body = SubstrateTierOutcomesPayloadV1.model_validate(dec.envelope.payload)
    assert body.cold_anchors == ["a1"]
