from datetime import datetime, timezone

from orion.schemas.registry import resolve
from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1


def _sample() -> InnerStateFeaturesV1:
    return InnerStateFeaturesV1(
        generated_at=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
        self_state_id="self.state:tick_abc:policy.v1",
        features=[
            InnerFeatureV1(name="coherence", raw_value=1.0, scaled_value=0.0,
                           source="self_state.dimensions.coherence"),
        ],
        infra=[
            InnerFeatureV1(name="contract_pressure", raw_value=1.0, scaled_value=0.0,
                           source="self_state.dominant_field_channels.contract_pressure"),
        ],
        headline=0.7,
        headline_source="cold_start_aggregate",
        phi_health="ok",
    )


def test_registry_resolves_inner_state_features() -> None:
    model = resolve("InnerStateFeaturesV1")
    assert model is InnerStateFeaturesV1


def test_round_trip_every_feature_has_raw_scaled_source() -> None:
    payload = _sample()
    data = payload.model_dump()
    restored = InnerStateFeaturesV1.model_validate(data)
    assert restored.headline == 0.7
    for f in restored.features + restored.infra:
        assert f.name and f.source is not None
        assert isinstance(f.raw_value, float)
        assert isinstance(f.scaled_value, float)
