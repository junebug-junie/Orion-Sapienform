"""The mood-arc telemetry-anomaly metacog trigger ships OFF (PE suite rebuild
step 5, Juniper 2026-11-11) until the encoder is retrained. The scorer and its
score table stay live; only the trigger is quiet. Pins code default, compose
default and .env_example together so one cannot silently flip back alone."""
from pathlib import Path

from app.settings import Settings

SVC = Path(__file__).resolve().parents[1]
KEY = "EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE"


def test_code_default_is_off(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    assert Settings().metacog_telemetry_anomaly_trigger_enable is False


def test_compose_and_env_example_default_off():
    assert f"{KEY}:-false}}" in (SVC / "docker-compose.yml").read_text()
    lines = [l.strip() for l in (SVC / ".env_example").read_text().splitlines()]
    assert f"{KEY}=false" in lines
