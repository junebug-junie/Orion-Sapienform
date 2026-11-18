"""_vision_organ_tick: router windows in, and on silence a clock-driven alarm.

The lifecycle that matters: a router that stops reporting must push
capability:vision toward alarm on a clock, at a bounded rate, and the first real
window afterwards must bring it back -- not hold the last calm reading forever.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(SUBSTRATE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import app.worker as worker_module
from app.worker import REDUCER_SPEC_BY_KEY, REDUCER_SPECS, BiometricsSubstrateWorker
from orion.schemas.vision_organ_projection import VisionOrganProjectionV1
from orion.substrate.vision_organ_loop.constants import (
    VISION_ORGAN_GRAMMAR_CURSOR_NAME,
    VISION_ORGAN_SOURCE_SERVICE,
)


class _Store:
    def __init__(self, projection: VisionOrganProjectionV1 | None) -> None:
        self.projection = projection
        self.receipts: list = []
        self.events: list = []

    def fetch_vision_organ_grammar_events(self, *, limit: int):
        out, self.events = self.events, []
        return out

    def load_vision_organ_projection(self, _pid):
        return self.projection

    def save_vision_organ_projection(self, projection):
        self.projection = projection

    def save_receipt(self, receipt):
        self.receipts.append(receipt)


def _worker(store: _Store, *, started: datetime) -> BiometricsSubstrateWorker:
    w = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    w._settings = MagicMock()
    w._settings.vision_organ_silence_sec = 180.0
    w._settings.vision_organ_grammar_batch_limit = 200
    w._store = store
    w._process_started_at = started
    w._vision_organ_last_silence_write = None
    w._vision_organ_last_silence_check = None
    return w


class _Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def now(self, tz=None):
        return self.t


def test_spec_is_registered_with_its_own_cursor() -> None:
    spec = REDUCER_SPEC_BY_KEY["vision_organ"]
    assert spec.reducer_key == "vision_organ"
    assert spec.cursor_name == VISION_ORGAN_GRAMMAR_CURSOR_NAME
    assert spec.source_service == VISION_ORGAN_SOURCE_SERVICE
    assert len({s.cursor_name for s in REDUCER_SPECS}) == len(REDUCER_SPECS)


def test_silent_router_alarms_on_a_clock_at_a_bounded_rate(monkeypatch) -> None:
    t0 = datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc)
    proj = VisionOrganProjectionV1(
        projection_id="active_vision_organ_projection",
        generated_at=t0,
        last_window_id="w",
        last_window_end=t0,
        vision_frame_staleness=0.0,
    )
    store = _Store(proj)
    w = _worker(store, started=t0 - timedelta(hours=1))
    clock = _Clock(t0 + timedelta(seconds=120))
    monkeypatch.setattr(worker_module, "datetime", _patched_datetime(clock))

    assert w._vision_organ_tick() is None
    assert store.receipts == []  # 120 s: inside the 180 s silence bound

    clock.t = t0 + timedelta(seconds=200)
    w._vision_organ_tick()
    assert len(store.receipts) == 1
    hints = store.receipts[0].state_deltas[0].after["pressure_hints"]
    assert hints == {"vision_frame_staleness": 1.0}
    assert store.projection.status == "silent"

    clock.t = t0 + timedelta(seconds=230)  # within silence_sec / 3 of the last write
    w._vision_organ_tick()
    assert len(store.receipts) == 1

    clock.t = t0 + timedelta(seconds=270)
    w._vision_organ_tick()
    assert len(store.receipts) == 2


def test_never_heard_from_after_boot_alarms_once_past_the_bound(monkeypatch) -> None:
    t0 = datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc)
    store = _Store(None)
    w = _worker(store, started=t0)
    clock = _Clock(t0 + timedelta(seconds=60))
    monkeypatch.setattr(worker_module, "datetime", _patched_datetime(clock))
    w._vision_organ_tick()
    assert store.receipts == []  # a normal restart must not cry outage
    clock.t = t0 + timedelta(seconds=181)
    w._vision_organ_tick()
    assert store.receipts[0].state_deltas[0].after["pressure_hints"]["vision_frame_staleness"] == 1.0


def _patched_datetime(clock: _Clock):
    real = datetime

    class _DT(real):  # type: ignore[misc]
        @classmethod
        def now(cls, tz=None):
            return clock.t

    return _DT


def test_silence_writes_stay_under_the_digester_expiry_for_long_silence_settings(monkeypatch) -> None:
    """VISION_ORGAN_SILENCE_SEC=1800 must still rewrite 1.0 at least every 60 s, or the
    digester's 300 s expiry turns "can't see" into "unmeasured"."""
    t0 = datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc)
    store = _Store(None)
    w = _worker(store, started=t0)
    w._settings.vision_organ_silence_sec = 1800.0
    clock = _Clock(t0 + timedelta(seconds=1801))
    monkeypatch.setattr(worker_module, "datetime", _patched_datetime(clock))
    w._vision_organ_tick()
    clock.t += timedelta(seconds=61)
    w._vision_organ_tick()
    assert len(store.receipts) == 2
