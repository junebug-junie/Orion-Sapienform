"""POWER_GUARD_UPS_BACKEND selects NIS vs SNMP without touching Athena defaults."""

from __future__ import annotations

import pytest

from app.main import build_ups_client
from app.settings import Settings
from app.ups_nis_client import NISUPSClient
from app.ups_snmp_client import SNMPUPSClient


@pytest.fixture
def clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "POWER_GUARD_UPS_BACKEND",
        "POWER_GUARD_UPS_HOST",
        "POWER_GUARD_SNMP_PORT",
        "POWER_GUARD_SNMP_COMMUNITY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_backend_is_nis(clear_backend_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POWER_GUARD_UPS_BACKEND", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.POWER_GUARD_UPS_BACKEND == "nis"
    client = build_ups_client(settings)
    assert isinstance(client, NISUPSClient)


def test_snmp_backend_builds_snmp_client(clear_backend_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POWER_GUARD_UPS_BACKEND", "snmp")
    monkeypatch.setenv("POWER_GUARD_UPS_HOST", "192.168.1.41")
    monkeypatch.setenv("POWER_GUARD_SNMP_COMMUNITY", "public-2")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.POWER_GUARD_UPS_BACKEND == "snmp"
    client = build_ups_client(settings)
    assert isinstance(client, SNMPUPSClient)
    assert client.host == "192.168.1.41"
    assert client.community == "public-2"


def test_unknown_backend_rejected(clear_backend_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POWER_GUARD_UPS_BACKEND", "usb")
    with pytest.raises(Exception):
        Settings(_env_file=None)  # type: ignore[call-arg]
