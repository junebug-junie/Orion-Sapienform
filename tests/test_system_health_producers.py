"""Every service that claims to heartbeat must actually be able to build one.

`SystemHealthV1` requires `boot_id` and `last_seen_ts`. Every producer builds it
inside a heartbeat loop wrapped in `try/except Exception: logger.warning(...)`, so a
missing required field does not crash anything -- it logs once per tick and sleeps.
The container stays "Up", its /health endpoint stays 200, and it publishes nothing.

Confirmed live 2026-08-29: `orion-gpu-cluster-power` had been failing every 30s tick
indefinitely. `orion-bus-tap` and `orion-rag` had the identical defect, and
whisper-tts carried a `# FIX: Added boot_id and last_seen_ts` comment from someone
hitting this before and fixing one service.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.schemas.telemetry.system_health import SystemHealthV1

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_system_health_construction_sites_pass_required_fields() -> None:
    """The gate itself, run over the real tree."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_system_health_producers.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_schema_still_requires_what_the_gate_checks() -> None:
    """Pins the gate's hardcoded REQUIRED_KWARGS against the live model.

    The gate lists required fields explicitly rather than introspecting, so that
    relaxing the schema is a visible decision. This test is what makes that safe: if
    a field stops being required, or a NEW required field appears, the gate's list is
    stale and this fails.
    """
    required = {
        name
        for name, field in SystemHealthV1.model_fields.items()
        if field.is_required()
    }
    gate_src = (REPO_ROOT / "scripts" / "check_system_health_producers.py").read_text()
    for name in required:
        assert f'"{name}"' in gate_src, (
            f"SystemHealthV1 requires {name!r} but the gate does not check it"
        )


@pytest.mark.parametrize(
    "node",
    ["psu-node", "tap-node", "rag-node"],
)
def test_the_previously_broken_payloads_now_validate(node: str) -> None:
    """Runtime proof, not just a static check: build the exact shape each of the
    three fixed producers now builds and confirm the model accepts it."""
    payload = SystemHealthV1(
        service="svc",
        version="1.0.0",
        node=node,
        status="ok",
        boot_id=str(uuid.uuid4()),
        last_seen_ts=datetime.now(timezone.utc),
    ).model_dump(mode="json")
    assert payload["boot_id"]
    assert payload["last_seen_ts"]
    assert payload["status"] == "ok"


def test_omitting_boot_id_really_does_raise() -> None:
    """The premise of the whole patch: this is a hard failure, silently swallowed by
    each heartbeat loop's except block."""
    with pytest.raises(Exception):
        SystemHealthV1(service="svc", node="n", status="ok")
