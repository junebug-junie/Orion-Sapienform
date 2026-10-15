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
