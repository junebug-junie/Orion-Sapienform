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
    STATE_VERSION,
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


# --------------------------------------------------------------------------
# the restore path, driven by a REAL clock
#
# Review finding (MEDIUM): every test above injects `now=`, but `_restore_state`
# reads `time.monotonic()`/`time.time()` directly and takes no `now`. So the
# restored `_failing_since` / `_last_alert_at` were ~630,000 (real host
# monotonic) while records were fed at now=2000.0, and three mutations of the
# restore path survived the whole suite. These tests drive ONE fake clock
# through both modules so the restore path is actually exercised.
# --------------------------------------------------------------------------

class _Clock:
    def __init__(self, t=1_000_000.0): self.t = float(t)
    def monotonic(self): return self.t
    def time(self): return self.t


@pytest.fixture()
def clock(monkeypatch):
    import app.liveness as liveness_mod
    import app.liveness_state as state_mod
    c = _Clock()
    monkeypatch.setattr(liveness_mod, "time", c, raising=True)
    monkeypatch.setattr(state_mod, "time", c, raising=True)
    return c


def _arm(store, clock, **kw):
    w = _watcher(store=store, **kw)
    for _ in range(400):
        d = w.record(ok=False, error_code="gpu_hard_floor")
        if d.alert:
            w.note_alert_delivered(True)
            return w
        clock.t += 5.0
    raise AssertionError("never armed")


def test_a_restored_arm_does_not_declare_recovery_on_one_sample(tmp_path, clock) -> None:
    """Review finding (HIGH), reproduced then fixed.

    Container recreated mid-incident, GPU still broken, one task happens to
    succeed. A restored watcher has an EMPTY deque, so the thin-sample clear
    branch fired `recovered=True` off that single sample and posted "Vision is
    working again ... 100% of recent tasks succeeding".
    """
    store = LivenessStateStore(str(tmp_path / "s.json"))
    _arm(store, clock)
    clock.t += 10.0
    w2 = _watcher(store=store)
    assert w2._alerting is True and w2._restored_alerting is True
    d = w2.record(ok=True)
    assert d.recovered is False, "one lucky success is not a recovery"


def test_a_restart_does_not_lengthen_the_window_where_orion_looks_fine(tmp_path, clock) -> None:
    """Same scenario, measured as: how long does the watcher believe all is well
    while the GPU is still broken?

    Measuring "time until it alerts AGAIN" was the wrong question and this test
    originally asked it -- with persistence the watcher never stops alerting, so
    there is no second alert to wait for and the measurement is `None`. That is
    the desired outcome, not a failure. What matters to an operator is the
    stretch where nothing is flagged: pre-fix that was 3590s (restored cooldown
    holding off the re-alert), pre-patch 225s, and it must never get worse.
    """
    def not_alerting_seconds(store):
        clock.t = 1_000_000.0
        _arm(store, clock)
        clock.t += 10.0
        w2 = _watcher(store=store)     # the restart; unarmed without a store
        w2.record(ok=True)             # the one lucky success
        silent = 0.0
        for _ in range(3000):
            clock.t += 5.0
            w2.record(ok=False, error_code="gpu_hard_floor")
            if not w2._alerting:
                silent += 5.0
            elif silent:
                break
        return silent

    with_state = not_alerting_seconds(LivenessStateStore(str(tmp_path / "s.json")))
    without_state = not_alerting_seconds(None)
    assert without_state > 0.0, "without persistence there is a real silent window"
    assert with_state == 0.0, (
        f"a restored arm must stay armed while the outage continues; "
        f"looked fine for {with_state}s"
    )


def test_a_restored_arm_still_recovers_once_the_sample_floor_is_met(tmp_path, clock) -> None:
    """The gate must delay the clear, not prevent it -- otherwise the fix for
    the regression would reintroduce the bug this whole patch exists for."""
    store = LivenessStateStore(str(tmp_path / "s.json"))
    _arm(store, clock)
    clock.t += 10.0
    w2 = _watcher(store=store)
    recovered = False
    for _ in range(MIN_SAMPLES + 2):
        clock.t += 1.0
        recovered = recovered or w2.record(ok=True).recovered
    assert recovered is True
    assert store.load().alerting is False


def test_the_restored_cooldown_does_not_suppress_a_genuinely_new_outage(tmp_path, clock) -> None:
    store = LivenessStateStore(str(tmp_path / "s.json"))
    _arm(store, clock)
    clock.t += 10.0
    w2 = _watcher(store=store)
    assert w2._last_alert_at is None, "restoring the cooldown deadline is what caused the HIGH"
    for _ in range(MIN_SAMPLES + 2):
        clock.t += 1.0
        w2.record(ok=True)
    fired = False
    for _ in range(400):
        clock.t += 5.0
        if w2.record(ok=False, error_code="gpu_hard_floor").alert:
            fired = True
            break
    assert fired, "a new outage after a restored-then-cleared one must still alert"


# --------------------------------------------------------------------------
# the recovery notification needs delivery confirmation too
# --------------------------------------------------------------------------

def test_an_undelivered_recovery_rolls_back_and_retries(tmp_path, clock) -> None:
    """Review finding (MEDIUM): the arm path got a delivery rollback in PR #1805;
    the clear path never did. A recovery whose POST failed was consumed, silently
    persisted as closed, and never retried -- a SECOND live route to the observed
    "zero vision_recovered, ever", which this patch would have made durable."""
    store = LivenessStateStore(str(tmp_path / "s.json"))
    _arm(store, clock)
    clock.t += 10.0
    w2 = _watcher(store=store)
    for _ in range(MIN_SAMPLES + 2):
        clock.t += 1.0
        d = w2.record(ok=True)
        if d.recovered:
            break
    assert d.recovered is True
    w2.note_alert_delivered(False)               # notify was down
    assert w2._alerting is True, "an unannounced clear must roll back"
    assert store.load().alerting is True, "and must not persist as closed"

    clock.t += 1.0
    assert w2.record(ok=True).recovered is True, "and must retry on the next sample"


def test_a_delivered_recovery_stays_closed(tmp_path, clock) -> None:
    store = LivenessStateStore(str(tmp_path / "s.json"))
    _arm(store, clock)
    clock.t += 10.0
    w2 = _watcher(store=store)
    for _ in range(MIN_SAMPLES + 2):
        clock.t += 1.0
        if w2.record(ok=True).recovered:
            break
    w2.note_alert_delivered(True)
    assert w2._alerting is False
    assert store.load().alerting is False
    clock.t += 1.0
    assert w2.record(ok=True).recovered is False, "a recovery is a transition, not a level"


# --------------------------------------------------------------------------
# store contract
# --------------------------------------------------------------------------

def test_load_never_raises_on_a_non_numeric_version(tmp_path) -> None:
    """The version check sat outside the try, so this raised TypeError straight
    out of a method whose docstring says "Never raises"."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": [1], "alerting": True, "saved_at_wall": time.time()}))
    assert LivenessStateStore(str(path)).load().alerting is False


def test_a_future_version_is_rejected_by_the_VERSION_guard(tmp_path) -> None:
    """The parametrized corrupt-file case for version 99 carried no
    `saved_at_wall`, so it was rejected by the STALENESS guard and the version
    guard itself was never exercised -- deleting it survived the whole suite."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "version": STATE_VERSION + 98, "alerting": True,
        "saved_at_wall": time.time(), "failing_since_wall": time.time(),
    }))
    assert LivenessStateStore(str(path)).load().alerting is False


def test_save_is_atomic(tmp_path, monkeypatch) -> None:
    """Asserts the rename, not just that one file exists afterwards -- a
    non-atomic copyfile+unlink satisfied the previous test."""
    import app.liveness_state as state_mod
    seen = {}
    real = state_mod.os.replace

    def spy(src, dst):
        seen["replaced"] = (src, dst)
        return real(src, dst)

    monkeypatch.setattr(state_mod.os, "replace", spy)
    store = LivenessStateStore(str(tmp_path / "s.json"))
    assert store.save(PersistedLivenessState(alerting=True)) is True
    assert seen.get("replaced"), "save must publish via an atomic os.replace"


def test_a_failed_write_leaves_no_temp_file_behind(tmp_path, monkeypatch) -> None:
    """Reaches the cleanup path, which no previous test could: the unwritable
    fixture failed at makedirs, before mkstemp ever ran."""
    import app.liveness_state as state_mod
    monkeypatch.setattr(state_mod.os, "replace",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    store = LivenessStateStore(str(tmp_path / "s.json"))
    assert store.save(PersistedLivenessState(alerting=True)) is False
    assert list(tmp_path.iterdir()) == [], f"temp file left behind: {list(tmp_path.iterdir())}"


def test_snapshot_exposes_whether_persistence_is_actually_working(tmp_path, clock) -> None:
    """A read-only or full volume made the entire fix inert with nothing saying so."""
    store = LivenessStateStore(str(tmp_path / "s.json"))
    w = _arm(store, clock)
    snap = w.snapshot()
    assert snap["state_path"] == store.path
    assert snap["state_write_ok"] is True
    broken = _watcher(store=LivenessStateStore("/proc/nope/s.json"))
    broken._persist_state()
    assert broken.snapshot()["state_write_ok"] is False


def test_a_future_failing_since_cannot_stall_the_sustain_clock(tmp_path, clock) -> None:
    """The file-level future guard rejects a future `saved_at_wall`, but a file
    with a sane `saved_at_wall` and a future `failing_since_wall` slips past it
    -- a partial corruption, or a clock step between two writes.

    Without the `max(0.0, ...)` clamp that restores as a monotonic value in the
    FUTURE, so `failing_for = ts - _failing_since` goes negative, stays below
    `sustain_sec` forever, and the watcher can never alert again.
    """
    path = tmp_path / "s.json"
    now = clock.time()
    path.write_text(json.dumps({
        "version": STATE_VERSION,
        "alerting": True,
        "failing_since_wall": now + 86_400,     # a day ahead of the file itself
        "last_alert_at_wall": now,
        "saved_at_wall": now,
    }))
    w = _watcher(store=LivenessStateStore(str(path)))
    assert w._failing_since is not None
    assert w._failing_since <= clock.monotonic(), (
        "a future failing_since must clamp to now, or the sustain clock never elapses"
    )
    # and the watcher must still be able to clear, which needs a sane clock
    recovered = False
    for _ in range(MIN_SAMPLES + 2):
        clock.t += 1.0
        recovered = recovered or w.record(ok=True).recovered
    assert recovered is True
