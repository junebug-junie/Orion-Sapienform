"""The recovery notification must survive the container being recreated.

Headline test: `test_the_2026_08_29_incident_three_alerts_zero_recoveries`
replays the exact live shape -- three `vision_blind` alerts on 2026-08-29
(20:25, 21:00, 22:13), each reporting failing "for 3m" with ~88 samples, and
zero `vision_recovered` records ever. It asserts the OLD behaviour reproduces
the incident and the NEW behaviour does not, so this file fails if the fix is
reverted rather than merely describing it.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest

SERVICE_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

from app.liveness import VisionLivenessWatcher  # noqa: E402
from app.liveness_state import (  # noqa: E402
    MAX_STATE_AGE_SEC,
    LivenessStateStore,
    PersistedLivenessState,
)

WINDOW, MIN_SAMPLES, SUSTAIN = 300.0, 10, 180.0


def _watcher(store=None, **kw):
    kw.setdefault("window_sec", WINDOW)
    kw.setdefault("min_samples", MIN_SAMPLES)
    kw.setdefault("sustain_sec", SUSTAIN)
    kw.setdefault("cooldown_sec", 3600.0)
    return VisionLivenessWatcher(state_store=store, **kw)


def _drive_to_alert(w, *, t0=1000.0):
    """Fail continuously until the watcher arms. Returns (decision, now)."""
    now = t0
    for _ in range(400):
        d = w.record(ok=False, error_code="gpu_hard_floor", now=now)
        if d.alert:
            w.note_alert_delivered(True)
            return d, now
        now += 5.0
    raise AssertionError("watcher never armed")


# --------------------------------------------------------------------------
# the incident
# --------------------------------------------------------------------------

def test_the_2026_08_29_incident_three_alerts_zero_recoveries(tmp_path) -> None:
    """Without persistence: restart mid-incident, re-alert, never recover.

    That is the live record -- 8 vision_blind since 08-21, 0 vision_recovered
    ever. The three alerts that day each said "for 3m" (== sustain_sec) with
    ~88 samples, which is a *fresh* sustain clock every time: a live process
    cannot re-alert, because `if self._alerting: return` blocks it.
    """
    # --- old behaviour: no store, state dies with the process
    w1 = _watcher()
    d1, _ = _drive_to_alert(w1)
    assert d1.alert is True

    w2 = _watcher()          # container recreated mid-incident
    recovered = [w2.record(ok=True, now=2000.0 + i).recovered for i in range(MIN_SAMPLES + 2)]
    assert not any(recovered), "reproduces the incident: recovery is lost across a restart"

    # --- new behaviour: same sequence, with a store
    store = LivenessStateStore(str(tmp_path / "state.json"))
    w3 = _watcher(store=store)
    d3, _ = _drive_to_alert(w3)
    assert d3.alert is True
    assert store.load().alerting is True, "a delivered alert must be durable"

    w4 = _watcher(store=store)   # same restart
    assert w4._alerting is True, "restored as still-alerting"
    decisions = [w4.record(ok=True, now=2000.0 + i) for i in range(MIN_SAMPLES + 2)]
    assert any(d.recovered for d in decisions), "the recovery that was being lost now fires"
    assert store.load().alerting is False, "and the clear is persisted too"


def test_recovery_fires_only_once_after_a_restart() -> None:
    """A recovery is a transition, not a level -- it must not repeat per task."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        store = LivenessStateStore(os.path.join(d, "s.json"))
        w = _watcher(store=store)
        _drive_to_alert(w)
        w2 = _watcher(store=store)
        fired = [w2.record(ok=True, now=3000.0 + i).recovered for i in range(30)]
        assert sum(1 for f in fired if f) == 1


# --------------------------------------------------------------------------
# clock translation -- the subtle part
# --------------------------------------------------------------------------

def test_monotonic_values_are_not_persisted_raw(tmp_path) -> None:
    """`time.monotonic()` is process-relative; on this host it is uptime-based
    and wildly different from wall clock. Persisting it raw would restore a
    cooldown deadline from another process's epoch."""
    store = LivenessStateStore(str(tmp_path / "s.json"))
    w = _watcher(store=store)
    _drive_to_alert(w)
    raw = json.loads(pathlib.Path(store.path).read_text())
    now_wall = time.time()
    for key in ("failing_since_wall", "last_alert_at_wall", "saved_at_wall"):
        val = raw.get(key)
        if val is None:
            continue
        assert abs(val - now_wall) < MAX_STATE_AGE_SEC, (
            f"{key}={val} is not a wall-clock timestamp (now={now_wall})"
        )


def test_a_stale_state_file_is_ignored(tmp_path) -> None:
    """Coming back after days down must not resurrect a forgotten incident."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "version": 1, "alerting": True,
        "failing_since_wall": time.time() - MAX_STATE_AGE_SEC - 10_000,
        "last_alert_at_wall": time.time() - MAX_STATE_AGE_SEC - 10_000,
        "saved_at_wall": time.time() - MAX_STATE_AGE_SEC - 10_000,
    }))
    assert LivenessStateStore(str(path)).load().alerting is False


def test_a_future_dated_state_file_is_ignored(tmp_path) -> None:
    """A clock step forward would otherwise park a cooldown permanently ahead."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "version": 1, "alerting": True, "saved_at_wall": time.time() + 86_400,
        "failing_since_wall": time.time(), "last_alert_at_wall": time.time(),
    }))
    assert LivenessStateStore(str(path)).load().alerting is False


# --------------------------------------------------------------------------
# nothing in the alerting path may stop the service from seeing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["", "{", "null", "[]", '{"version": 99, "alerting": true}'])
def test_corrupt_state_degrades_to_clean_and_never_raises(tmp_path, content) -> None:
    path = tmp_path / "s.json"
    path.write_text(content)
    assert LivenessStateStore(str(path)).load().alerting is False


def test_missing_file_is_not_an_error(tmp_path) -> None:
    assert LivenessStateStore(str(tmp_path / "nope.json")).load().alerting is False


def test_an_unwritable_path_does_not_raise_or_break_the_watcher(tmp_path) -> None:
    store = LivenessStateStore("/proc/definitely/not/writable/s.json")
    assert store.save(PersistedLivenessState(alerting=True)) is False
    w = _watcher(store=store)
    d, _ = _drive_to_alert(w)
    assert d.alert is True, "alerting still works with an unwritable state path"


def test_a_store_that_raises_is_swallowed() -> None:
    class Exploding:
        def load(self): raise RuntimeError("boom")
        def save(self, *_a, **_k): raise RuntimeError("boom")

    w = _watcher(store=Exploding())
    d, _ = _drive_to_alert(w)
    assert d.alert is True, "a broken store must never stop the service from seeing"


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path) -> None:
    store = LivenessStateStore(str(tmp_path / "s.json"))
    assert store.save(PersistedLivenessState(alerting=True)) is True
    names = [p.name for p in tmp_path.iterdir()]
    assert names == ["s.json"], f"temp file left behind: {names}"


# --------------------------------------------------------------------------
# rollback
# --------------------------------------------------------------------------

def test_an_undelivered_alert_is_not_persisted_as_armed(tmp_path) -> None:
    """If the POST failed, the watcher rolls back so it retries. A stale armed
    file would resurrect an alert this process decided never happened."""
    store = LivenessStateStore(str(tmp_path / "s.json"))
    w = _watcher(store=store)
    now = 1000.0
    for _ in range(400):
        d = w.record(ok=False, error_code="gpu_hard_floor", now=now)
        if d.alert:
            w.note_alert_delivered(False)
            break
        now += 5.0
    else:
        raise AssertionError("never armed")
    assert store.load().alerting is False
    assert _watcher(store=store)._alerting is False


def test_persistence_is_off_when_no_path_is_configured() -> None:
    """Empty VISION_LIVENESS_STATE_PATH keeps the prior in-memory behaviour."""
    w = _watcher(store=None)
    d, _ = _drive_to_alert(w)
    assert d.alert is True
    assert _watcher(store=None)._alerting is False


def test_the_clear_threshold_branch_also_persists(tmp_path) -> None:
    """There are TWO recovery branches and the tests above only reach one.

    A restored watcher has an empty sample deque, so the first success hits the
    below-min_samples path (`count > 0 and rate == 0.0`) and returns before the
    clear-threshold branch is ever evaluated. Mutation-testing caught this:
    deleting `_persist_state()` from the clear branch changed nothing.

    Reaching it needs `count >= min_samples` AND `rate <= clear_fail_rate` while
    still alerting -- so one failure first (which keeps rate != 0 and blocks the
    other branch), then successes until the sample floor is crossed:
    1 failure + 9 successes = 10 samples at rate 0.1, under the 0.2 clear rate.
    """
    store = LivenessStateStore(str(tmp_path / "s.json"))
    w = _watcher(store=store)
    _drive_to_alert(w)
    assert store.load().alerting is True

    w2 = _watcher(store=store)
    assert w2._alerting is True

    now = 5000.0
    d = w2.record(ok=False, error_code="gpu_hard_floor", now=now)
    assert d.recovered is False, "a failure must not read as recovery"
    recovered = False
    for i in range(1, MIN_SAMPLES):          # 9 successes -> 10 samples, rate 0.1
        d = w2.record(ok=True, now=now + i)
        recovered = recovered or d.recovered
    assert recovered is True, "the clear-threshold branch should have fired"
    assert store.load().alerting is False, "and it must persist the clear"
