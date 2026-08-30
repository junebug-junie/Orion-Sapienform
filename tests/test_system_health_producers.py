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

import importlib.util
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pydantic import ValidationError

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


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "orion_system_health_gate_under_test",
        REPO_ROOT / "scripts" / "check_system_health_producers.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_required_list_matches_the_live_schema_exactly() -> None:
    """Pins the gate's hardcoded REQUIRED_KWARGS against the live model.

    The gate lists required fields explicitly rather than introspecting, so that
    relaxing the schema stays a visible decision. This is what makes that safe.
    Compares the imported set both directions -- an earlier version grepped the
    gate's SOURCE TEXT for each field name, which a mere docstring mention would
    have satisfied while REQUIRED_KWARGS went stale, and which could never catch a
    name left in the list after it stopped being required.
    """
    gate = _load_gate_module()
    schema_required = {
        name
        for name, field in SystemHealthV1.model_fields.items()
        if field.is_required()
    }
    assert set(gate.REQUIRED_KWARGS) == schema_required, (
        f"gate checks {sorted(gate.REQUIRED_KWARGS)} but the model requires "
        f"{sorted(schema_required)}"
    )


def test_gate_also_requires_the_interval_that_is_not_schema_required() -> None:
    """heartbeat_interval_sec has a schema default, so it is not in REQUIRED_KWARGS
    -- but omitting it is a live false alarm, not a style issue: equilibrium computes
    grace = interval * 3.0 and every loop in this repo sleeps 30s, so the 10.0
    default yields zero margin. The gate must enforce it separately."""
    gate = _load_gate_module()
    assert gate.INTERVAL_KWARG == "heartbeat_interval_sec"
    assert gate.INTERVAL_KWARG not in gate.REQUIRED_KWARGS
    field = SystemHealthV1.model_fields[gate.INTERVAL_KWARG]
    assert not field.is_required(), "if this became required, fold it into REQUIRED_KWARGS"


def test_gate_fails_when_it_inspects_nothing() -> None:
    """A gate that reaches no code must fail, not print OK. This repo has shipped an
    inert gate before."""
    gate = _load_gate_module()
    assert gate.MIN_EXPECTED_SITES > 0


def test_declared_interval_matches_each_producer_real_sleep() -> None:
    """The failure review caught: publishing works but the service reads as `down`.

    A payload can pass the AST gate and still be wrong if the interval it declares
    disagrees with the loop's actual period. Reads both numbers out of each producer
    rather than hand-writing a payload, so it fails if either side drifts.
    Restricted to loops whose period is a literal; the ones threading a settings
    value are correct by construction.
    """
    import re

    mismatches = []
    for path in (REPO_ROOT / "services").rglob("*.py"):
        if "tests" in path.parts or "SystemHealthV1(" not in path.read_text(
            encoding="utf-8", errors="ignore"
        ):
            continue
        text = path.read_text(encoding="utf-8")
        declared = re.search(r"heartbeat_interval_sec=([0-9.]+),", text)
        slept = re.search(r"asyncio\.sleep\(([0-9.]+)\)\s*$", text, re.M)
        if not declared or not slept:
            continue
        if float(declared.group(1)) != float(slept.group(1)):
            mismatches.append(
                f"{path.relative_to(REPO_ROOT)}: declares "
                f"{declared.group(1)}s but sleeps {slept.group(1)}s"
            )
    assert not mismatches, "\n".join(mismatches)


def test_a_realistic_payload_validates_and_round_trips() -> None:
    """Runtime proof the fixed shape is actually accepted by the model."""
    payload = SystemHealthV1(
        service="svc",
        version="1.0.0",
        node="psu-node",
        status="ok",
        boot_id=str(uuid.uuid4()),
        last_seen_ts=datetime.now(timezone.utc),
        heartbeat_interval_sec=30.0,
    ).model_dump(mode="json")
    assert payload["boot_id"]
    assert payload["last_seen_ts"]
    assert payload["heartbeat_interval_sec"] == 30.0
    # equilibrium re-validates every heartbeat off the bus; the dumped form must
    # survive that round trip.
    assert SystemHealthV1.model_validate(payload).status == "ok"


def test_omitting_boot_id_really_does_raise() -> None:
    """The premise of the whole patch: this is a hard failure, silently swallowed by
    each heartbeat loop's `except Exception` block.

    Omits ONLY boot_id and catches ValidationError specifically -- a bare
    `pytest.raises(Exception)` around a call missing two fields would pass on an
    unrelated TypeError or import failure and prove nothing.
    """
    with pytest.raises(ValidationError):
        SystemHealthV1(
            service="svc",
            node="n",
            status="ok",
            last_seen_ts=datetime.now(timezone.utc),
        )
