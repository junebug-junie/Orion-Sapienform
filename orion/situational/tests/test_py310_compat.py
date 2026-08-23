"""Regression guard for the 2026-08-23 production outage: `orion/situational/
context.py` shipped `from datetime import UTC` (Python 3.11+ only,
`datetime.UTC` is a 3.11 alias for `timezone.utc`) while the real
orion-athena-hub container runs Python 3.10.12. Every "orion" mode chat
turn crashed with an unhandled ImportError; the shared dev venv (Python
3.12) never caught it because `datetime.UTC` imports fine there.

This session's own venv is 3.12 too (same root cause as the loguru/Hub
incident earlier the same day: local environment newer/more permissive
than the actual container runtime), so this can't literally run under
3.10 to prove the fix -- the real proof is the live verification recorded
in this PR's report (`docker exec orion-athena-hub python3 -c
"from orion.hub.turn_orchestrator import run_unified_turn"`). What CAN be
tested here, cheaply and deterministically, is that the exact reported
bug pattern doesn't reappear: a static source check for `UTC` imported
from `datetime` directly, the same technique CLAUDE.md's "no keyword
cathedral" sibling static-wiring tests already use elsewhere in this repo
for structural regressions that don't need a live process.
"""
from __future__ import annotations

import re
from pathlib import Path

_SITUATIONAL_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = Path(__file__).resolve().parents[3] / "scripts"

# `datetime.UTC` (bare attribute access, e.g. `dt.UTC`) is Python 3.11+;
# `from datetime import UTC` is the same alias imported by name. Either
# form breaks on this container's actual Python 3.10 runtime.
_BAD_IMPORT = re.compile(r"from\s+datetime\s+import\s+[^\n]*\bUTC\b")


def _assert_no_py311_only_utc_import(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    match = _BAD_IMPORT.search(source)
    assert match is None, (
        f"{path} imports datetime.UTC (Python 3.11+ only) -- the real "
        "orion-athena-hub container runs Python 3.10 and this import "
        "crashes every orion-mode chat turn on startup. Use "
        "`from datetime import timezone` and `timezone.utc` instead "
        "(a true drop-in: CPython defines datetime.UTC = timezone.utc)."
    )


def test_context_module_does_not_import_py311_only_datetime_utc() -> None:
    _assert_no_py311_only_utc_import(_SITUATIONAL_ROOT / "context.py")


def test_situation_grounding_smoke_script_does_not_import_py311_only_datetime_utc() -> None:
    # scripts/smoke_situation_grounding.py -- the designated smoke test for
    # this exact feature had the identical bug, so it could not even have
    # caught this against a real Python 3.10 target.
    _assert_no_py311_only_utc_import(_SCRIPTS_ROOT / "smoke_situation_grounding.py")


def test_context_module_actually_builds_a_situation_brief() -> None:
    """Real execution, not just a static check -- proves the datetime
    handling in context.py works correctly under whatever Python this
    test suite runs on, exercising every `datetime.now(timezone.utc)`
    call site added by this fix."""
    import asyncio
    from types import SimpleNamespace

    from orion.situational.context import build_situation_for_ctx

    settings = SimpleNamespace(
        orion_situation_enabled=True,
        orion_situation_ttl_seconds=300,
        orion_situation_prompt_max_chars=1200,
        orion_situation_timezone="America/Denver",
        orion_situation_location_label="Unknown",
        orion_situation_locality=None,
        orion_situation_region=None,
        orion_situation_country=None,
        orion_situation_location_precision="city",
        orion_situation_weather_enabled=False,
        orion_situation_weather_provider="stub",
        orion_situation_weather_lat=None,
        orion_situation_weather_lon=None,
        orion_situation_weather_ttl_seconds=600,
        orion_situation_umbrella_precip_prob_threshold=40,
        orion_situation_jacket_temp_f_threshold=55,
        orion_situation_high_wind_mph_threshold=25,
        orion_situation_hot_car_temp_f_threshold=80,
        orion_situation_agenda_enabled=False,
        orion_situation_lab_context_enabled=False,
        orion_situation_lab_provider="stub",
        orion_situation_perception_enabled=False,
        orion_situation_perception_max_age_seconds=900,
        orion_situation_perception_stream_id="cam0",
        orion_situation_runtime_enabled=False,
        orion_situation_runtime_route="chat",
        orion_situation_runtime_ttl_seconds=120,
        orion_situation_runtime_probe_timeout_sec=2.0,
        cortex_exec_llm_gateway_url="http://llm-gateway:8210",
        orion_presence_default_requestor="Juniper",
        orion_presence_persist_allowed=False,
    )
    brief, fragment = asyncio.run(build_situation_for_ctx({"session_id": "test-py310-compat"}, settings))
    assert brief["generated_at"]
    assert fragment["compact_text"]
