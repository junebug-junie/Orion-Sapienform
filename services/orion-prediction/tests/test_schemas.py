from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml
from pydantic import ValidationError

from orion.schemas.prediction import PredictionForecastV1
from orion.schemas.registry import resolve
from tests.conftest import REPO


def test_registry_resolves_the_forecast_payload():
    assert resolve("PredictionForecastV1") is PredictionForecastV1


def test_channels_catalog_declares_the_forecast_channel_and_its_consumer():
    doc = yaml.safe_load((REPO / "orion" / "bus" / "channels.yaml").read_text())
    rows = {c["name"]: c for c in doc["channels"]}
    row = rows["orion:prediction:forecast"]
    assert row["schema_id"] == "PredictionForecastV1" and row["message_kind"] == "prediction.forecast.v1"
    assert row["consumer_services"] == ["orion-hardware-watch"]
    assert "orion:prediction:alert" not in rows


def test_forecast_payload_forbids_unknown_fields_and_bounds_probability():
    now = datetime.now(timezone.utc)
    base = {"series_id": "a.b", "model": "naive@5", "served_baseline": True, "run_id": "r", "issued_at": now,
            "origin_ts": now, "grain_min": 5, "steps": []}
    PredictionForecastV1.model_validate(base)
    with pytest.raises(ValidationError):
        PredictionForecastV1.model_validate({**base, "extra": 1})
    with pytest.raises(ValidationError):
        PredictionForecastV1.model_validate({**base, "exceedance": [
            {"name": "x", "threshold": 1, "window_min": 5, "probability": 1.5, "n_paths": 10}]})


def test_surprise_payload_is_registered_and_cataloged():
    from orion.schemas.prediction import PredictionSurpriseV1

    assert resolve("PredictionSurpriseV1") is PredictionSurpriseV1
    doc = yaml.safe_load((REPO / "orion" / "bus" / "channels.yaml").read_text())
    row = {c["name"]: c for c in doc["channels"]}["orion:prediction:surprise"]
    assert row["schema_id"] == "PredictionSurpriseV1" and row["message_kind"] == "prediction.surprise.v1"
    assert row["producer_services"] == ["orion-prediction"]


def test_surprise_payload_missing_outcome_carries_no_numbers():
    from orion.schemas.prediction import PredictionSurpriseV1

    now = datetime.now(timezone.utc)
    ev = PredictionSurpriseV1(series_id="a.b", horizon_min=15, issued_at=now, target_ts=now, scored_at=now,
                              outcome="missing", point=22.0, calm_surprisal=0.95)
    assert ev.p_value is None and ev.surprisal is None and ev.excess is None
    with pytest.raises(ValidationError):
        PredictionSurpriseV1.model_validate({**ev.model_dump(), "extra": 1})
