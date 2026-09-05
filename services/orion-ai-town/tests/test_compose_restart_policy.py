"""Gate: AI Town containers must come back after a host reboot.

2026-09-03 Circe power-key shutdown stopped Docker; AI Town stayed exited
because compose had no restart policy (default `no`). Sibling Circe stacks
(biometrics, diffusion-host) already use `unless-stopped` and recovered.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = SERVICE_ROOT / "docker-compose.yml"
# Named so a rename still fails loudly if someone drops a core service.
REQUIRED_SERVICES = frozenset({"frontend", "backend", "dashboard"})


def test_all_ai_town_services_restart_unless_stopped():
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose["services"]
    missing = sorted(REQUIRED_SERVICES - set(services))
    assert not missing, f"compose missing services: {missing}"

    # Every defined service — including any future fourth — must opt in.
    bad = {
        name: svc.get("restart")
        for name, svc in services.items()
        if svc.get("restart") != "unless-stopped"
    }
    assert not bad, (
        "every AI Town service must set restart: unless-stopped so a Circe "
        f"reboot does not leave town offline; got {bad}"
    )
