"""The metacog "hop 0" arena producer is gone, with no fallback (2026-10-16).

It produced 36 proposal candidates in its whole life and none after 2026-10-08
while its flag stayed on. Kill means kill: no module, no setting, no env key,
no worker branch. This guards against a partial resurrection (e.g. the env key
coming back in compose while the code is gone, or the reverse).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]


def test_module_is_gone():
    assert importlib.util.find_spec("orion.metacog.proposal") is None


def test_no_flag_or_branch_left_in_the_service():
    for rel in ("app/settings.py", "app/worker.py", "app/store.py", ".env_example", "docker-compose.yml"):
        text = (SERVICE / rel).read_text()
        assert "METACOG_HOP_PROPOSE" not in text.upper(), rel
        assert "orion.metacog.proposal" not in text, rel
        assert "load_repair_pressure_readings" not in text, rel
