"""Shape checks for the phi_reward SQL write path (no Postgres required)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.telemetry.phi_encoder import PhiIntrinsicRewardV1  # noqa: E402

from app.models.phi_reward import PhiRewardSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, settings  # noqa: E402
from app.worker import MODEL_MAP, _normalize_phi_reward_payload  # noqa: E402


def _make_reward(**overrides) -> PhiIntrinsicRewardV1:
    defaults = dict(
        generated_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
        encoder_version="phi-enc-v1",
        features_version="inner-v2",
        phi=0.42,
        delta_phi=0.01,
        recon_error=0.08,
        delta_recon_error=-0.002,
        latent={"z0": 0.1},
        attribution_top=[],
    )
    defaults.update(overrides)
    return PhiIntrinsicRewardV1(**defaults)


def test_default_route_map_points_phi_reward_at_phi_reward_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("self.phi_reward.v1") == "PhiRewardSQL"


def test_model_map_registers_phi_reward_sql_with_schema() -> None:
    assert MODEL_MAP["PhiRewardSQL"] == (PhiRewardSQL, PhiIntrinsicRewardV1)


def test_channel_is_subscribed() -> None:
    assert "orion:self:phi_reward" in settings.effective_subscribe_channels


def test_normalize_phi_reward_payload_maps_to_real_columns() -> None:
    reward = _make_reward()
    row_data = _normalize_phi_reward_payload(
        reward.model_dump(mode="json"),
        correlation_id="corr-phi-1",
    )
    mapper = inspect(PhiRewardSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    missing = [field for field in row_data if field not in valid_keys]
    assert not missing, f"normalized phi_reward fields missing from PhiRewardSQL: {missing}"
    assert row_data["correlation_id"] == "corr-phi-1"
    assert row_data["generated_at"] == reward.generated_at
    assert row_data["payload"]["phi"] == 0.42


def test_phi_reward_row_constructs_without_raising() -> None:
    row_data = _normalize_phi_reward_payload(
        _make_reward(phi=0.9).model_dump(mode="json"),
        correlation_id="corr-phi-2",
    )
    row = PhiRewardSQL(**row_data)
    assert row.correlation_id == "corr-phi-2"
    assert row.payload["phi"] == 0.9


def test_merge_redelivery_upserts_one_row() -> None:
    engine = create_engine("sqlite://")
    PhiRewardSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _merge(corr_id: str, phi: float) -> None:
        row_data = _normalize_phi_reward_payload(
            _make_reward(phi=phi).model_dump(mode="json"),
            correlation_id=corr_id,
        )
        sess = Session()
        try:
            sess.merge(PhiRewardSQL(**row_data))
            sess.commit()
        finally:
            sess.close()

    _merge("corr-phi-3", 0.1)
    _merge("corr-phi-3", 0.2)

    sess = Session()
    try:
        rows = sess.query(PhiRewardSQL).all()
        assert len(rows) == 1
        assert rows[0].payload["phi"] == 0.2
    finally:
        sess.close()
