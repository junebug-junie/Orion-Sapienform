"""Tests for the passthrough validator."""
import pytest
from datetime import datetime, timezone

from orion.signals.models import OrganClass
from app.passthrough import PassthroughValidator


@pytest.fixture
def validator():
    return PassthroughValidator()


def valid_signal_payload() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "signal_id": "abc123",
        "organ_id": "biometrics",
        "organ_class": "exogenous",
        "signal_kind": "biometrics_state",
        "dimensions": {"level": 0.7, "confidence": 0.9},
        "causal_parents": [],
        "observed_at": now,
        "emitted_at": now,
    }


def test_valid_signal_passes_through(validator):
    payload = valid_signal_payload()
    signal = validator.validate(payload)
    assert signal is not None
    assert signal.organ_id == "biometrics"


def test_unknown_organ_id_rejected(validator):
    payload = valid_signal_payload()
    payload["organ_id"] = "not_a_real_organ"
    signal = validator.validate(payload)
    assert signal is None


def test_missing_required_field_rejected(validator):
    payload = valid_signal_payload()
    del payload["signal_kind"]
    signal = validator.validate(payload)
    assert signal is None


def test_invalid_schema_rejected(validator):
    signal = validator.validate({"garbage": True})
    assert signal is None
