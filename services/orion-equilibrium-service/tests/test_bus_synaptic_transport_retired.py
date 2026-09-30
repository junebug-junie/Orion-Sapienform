"""Regression gate: the bus_synaptic transport metacog source stays retired.

Retired 2026-09-30 after its metric quality gate failed on live data (see
docs/superpowers/pr-reports/2026-09-30-retire-bus-synaptic-transport-trigger-pr.md).
Kill means kill: no builder, no poll loop, no settings, no env keys -- not a
disabled flag that a later patch can flip back on without re-running the gate.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def test_gate_module_has_no_bus_synaptic_builder() -> None:
    from app import transport_metacog_gate

    assert not hasattr(transport_metacog_gate, "build_transport_metacog_trigger_from_bus_synaptic")


def test_service_has_no_bus_synaptic_poll_loop() -> None:
    from app.service import EquilibriumService

    assert not hasattr(EquilibriumService, "_bus_synaptic_poll_loop")
    assert not hasattr(EquilibriumService, "_get_bus_synaptic_falkor_client")
    assert "bus_synaptic" not in inspect.getsource(EquilibriumService)


def test_settings_have_no_bus_synaptic_or_falkordb_fields() -> None:
    from app.settings import Settings

    fields = set(Settings.model_fields)
    assert not {f for f in fields if "bus_synaptic" in f or f.startswith("falkordb_")}


def test_env_example_and_compose_carry_no_retired_keys() -> None:
    for name in (".env_example", "docker-compose.yml"):
        text = (SERVICE_ROOT / name).read_text()
        # Key assignments only (compose "- KEY=..." or env "KEY=..."); the
        # retirement comment in .env_example names the keys on purpose.
        assigned = re.findall(
            r"^\s*(?:-\s*)?((?:EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_|FALKORDB_)\w*)\s*=",
            text,
            flags=re.MULTILINE,
        )
        assert assigned == [], (name, assigned)
