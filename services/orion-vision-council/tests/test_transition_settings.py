import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _clear_transition_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "COUNCIL_TRANSITION_GATE_ENABLED",
        "COUNCIL_TRANSITION_REFRESH_SEC",
        "COUNCIL_EVIDENCE_SKIP_ENABLED",
        "COUNCIL_EVIDENCE_SKIP_MAX_SEC",
    ):
        monkeypatch.delenv(key, raising=False)


def test_settings_reads_legacy_skip_enabled_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COUNCIL_EVIDENCE_SKIP_ENABLED", "false")
    from app.settings import Settings

    assert Settings().COUNCIL_TRANSITION_GATE_ENABLED is False


def test_settings_reads_legacy_skip_max_sec_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COUNCIL_EVIDENCE_SKIP_MAX_SEC", "45")
    from app.settings import Settings

    assert Settings().COUNCIL_TRANSITION_REFRESH_SEC == 45.0
