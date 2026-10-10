"""Regression gate: the "flow" metacog trigger stays retired.

Retired 2026-10-10 (docs/superpowers/pr-reports/2026-10-10-metacog-flow-trigger-
calibration-pr.md). The gate was calibrated 2026-07-30 to fire on 3.2% of
20-tick windows of `prediction_error_confidence`; by October 38% of real windows
qualified and publishes were spaced by the 1800s cooldown, not by the data. The
field is `1 - mean` prediction error over five domains, three of which read
exactly 0 whenever nothing is happening, so a sustained high plateau is what
*idleness* looks like -- no threshold on it separates "flow" from "nothing
happened". Kill means kill: no gate module, no detector, no settings, no env
keys, no cooldown lane -- not a disabled flag a later patch can flip back on.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
FIXTURE = SERVICE_ROOT / "tests" / "fixtures" / "prediction_error_confidence_live_2026-10-03.json"

# The retired gate's shipped calibration, kept here ONLY as the oracle that
# documents the bug -- it is not a live definition anywhere in the code.
_OLD_FLOOR = 0.90
_OLD_MAX_STDEV = 0.02
_OLD_MIN_TICKS = 20


def _live_series() -> list[float]:
    data = json.loads(FIXTURE.read_text())
    values = data["values"]
    assert len(values) == 240
    return values


def test_live_series_reproduces_the_drift_under_the_old_calibration() -> None:
    """The bug, on real data: a 2h slice of live ticks (2026-10-03) qualifies
    ~40% of 20-tick windows under the old floor/stdev rule, versus the 3.2%
    it was calibrated to. Most windows qualified, so the 1800s cooldown, not
    the data, set the firing rate."""
    values = _live_series()
    windows = [values[i - _OLD_MIN_TICKS + 1 : i + 1] for i in range(_OLD_MIN_TICKS - 1, len(values))]
    qualifying = [
        w for w in windows if min(w) >= _OLD_FLOOR and statistics.stdev(w) <= _OLD_MAX_STDEV
    ]
    fraction = len(qualifying) / len(windows)
    assert fraction > 0.30, fraction  # vs the intended 0.032
    # And the field sits near its ceiling: the median tick is ~0.98.
    assert statistics.median(values) > 0.97


@pytest.mark.asyncio
async def test_poll_loop_publishes_no_flow_trigger_on_the_live_series(monkeypatch) -> None:
    """Drive the real generative poll loop over every trailing 20-tick window of
    the live slice. Pre-retirement this published trigger_kind="flow" on the
    first qualifying window; now nothing in the loop can produce that kind."""
    from app.service import EquilibriumService, settings
    from orion.substrate.metacog_trigger_signals import ConfidenceSample

    monkeypatch.setattr(settings, "metacog_enable", True)
    monkeypatch.setattr(settings, "metacog_insight_trigger_enable", True)
    monkeypatch.setattr(settings, "metacog_generative_poll_interval_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 0.0)
    monkeypatch.setattr(settings, "metacog_insight_cooldown_sec", 0.0)

    values = _live_series()
    end = datetime.now(timezone.utc)
    samples = [
        ConfidenceSample(generated_at=end - (len(values) - 1 - i) * timedelta(seconds=30), value=v)
        for i, v in enumerate(values)
    ]
    windows = [samples[i - 19 : i + 1] for i in range(19, len(samples))]
    # Re-stamp each window so it is fresh at poll time (the freshness guard is
    # not what this test is about).
    shift = [end - w[-1].generated_at for w in windows]
    windows = [
        [ConfidenceSample(generated_at=s.generated_at + d, value=s.value) for s in w]
        for w, d in zip(windows, shift)
    ]

    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock()
    it = iter(windows)

    class _Reader:
        def fetch_recent_samples(self, *, limit: int):
            try:
                return next(it)
            except StopIteration:
                svc._stop.set()
                return []

    svc._attention_self_model_reader = _Reader()
    await asyncio.wait_for(svc._generative_metacog_poll_loop(), timeout=30)

    kinds = [call.args[1].payload["trigger_kind"] for call in svc.bus.publish.call_args_list]
    assert "flow" not in kinds, kinds


def test_flow_gate_module_and_detector_are_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.flow_metacog_gate")
    from orion.substrate import metacog_trigger_signals

    assert not hasattr(metacog_trigger_signals, "detect_flow_regime")
    assert not hasattr(metacog_trigger_signals, "FlowRegime")


def test_service_has_no_flow_gate_or_cooldown_lane() -> None:
    from app.service import EquilibriumService

    assert not hasattr(EquilibriumService, "_evaluate_flow_gate")
    assert "flow" not in EquilibriumService._PER_KIND_COOLDOWN_SETTINGS_ATTR


def test_settings_have_no_flow_fields() -> None:
    from app.settings import Settings

    assert not {f for f in Settings.model_fields if f.startswith("metacog_flow")}


def test_env_example_and_compose_carry_no_retired_keys() -> None:
    for name in (".env_example", "docker-compose.yml"):
        text = (SERVICE_ROOT / name).read_text()
        # Key assignments only; the retirement comment names the prefix on purpose.
        assigned = re.findall(
            r"^\s*(?:-\s*)?(EQUILIBRIUM_METACOG_FLOW_\w*)\s*=", text, flags=re.MULTILINE
        )
        assert assigned == [], (name, assigned)


def test_no_service_constructs_a_flow_trigger() -> None:
    """No producer anywhere under services/*/app may emit trigger_kind="flow"."""
    pattern = re.compile(r"""trigger_kind\s*=\s*["']flow["']""")
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "services").glob("*/app/**/*.py")
        if pattern.search(p.read_text(errors="ignore"))
    ]
    assert offenders == [], offenders
