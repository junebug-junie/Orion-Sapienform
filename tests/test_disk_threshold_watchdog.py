from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import disk_threshold_watchdog as watchdog  # noqa: E402


def _now(offset_minutes: float = 0.0) -> datetime:
    return datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)


# ---------------------------------------------------------------------------
# measure_path
# ---------------------------------------------------------------------------


def test_measure_path_returns_percent_used_for_real_path(tmp_path):
    pct, err = watchdog.measure_path(str(tmp_path))
    assert err is None
    assert pct is not None
    assert 0.0 <= pct <= 100.0


def test_measure_path_returns_error_for_missing_path():
    pct, err = watchdog.measure_path("/nonexistent/path/that/should/not/exist")
    assert pct is None
    assert err is not None


def test_measure_path_never_raises_on_missing_path():
    # Documented contract: measure_path never raises, only reports (None, error).
    watchdog.measure_path("/definitely/not/a/real/mount/anywhere")


# ---------------------------------------------------------------------------
# evaluate_path -- pure state-transition logic
# ---------------------------------------------------------------------------


def test_evaluate_path_ok_when_under_threshold():
    state, status, is_new = watchdog.evaluate_path({}, 50.0, None, 90.0, _now())
    assert status == "ok"
    assert is_new is False
    assert state["last_percent_used"] == 50.0
    assert state["first_detected_at"] is None


def test_evaluate_path_breach_is_new_event_first_time():
    state, status, is_new = watchdog.evaluate_path({}, 95.0, None, 90.0, _now())
    assert status == "breached"
    assert is_new is True
    assert state["first_detected_at"] == _now().isoformat()


def test_evaluate_path_repeated_breach_after_successful_notify_does_not_refire():
    state1, _, _ = watchdog.evaluate_path({}, 95.0, None, 90.0, _now())
    state1["notified"] = True  # simulates run() persisting a confirmed notify
    state2, status, should_notify = watchdog.evaluate_path(state1, 96.0, None, 90.0, _now(1))
    assert status == "breached"
    assert should_notify is False
    # first_detected_at carries forward from the original breach, not reset.
    assert state2["first_detected_at"] == _now().isoformat()


def test_evaluate_path_repeated_breach_retries_if_previous_notify_never_confirmed():
    # notified stays False (the default) -- simulates a failed/unconfirmed
    # publish attempt on the prior tick. Regression test for the code-review
    # finding: a failed notify must not be silently debounced into permanent
    # silence.
    state1, _, _ = watchdog.evaluate_path({}, 95.0, None, 90.0, _now())
    assert state1["notified"] is False
    state2, status, should_notify = watchdog.evaluate_path(state1, 96.0, None, 90.0, _now(1))
    assert status == "breached"
    assert should_notify is True


def test_evaluate_path_recovery_clears_breach_state_silently():
    state1, _, _ = watchdog.evaluate_path({}, 95.0, None, 90.0, _now())
    state1["notified"] = True
    state2, status, should_notify = watchdog.evaluate_path(state1, 50.0, None, 90.0, _now(1))
    assert status == "ok"
    assert should_notify is False
    assert state2["first_detected_at"] is None
    assert state2["notified"] is False


def test_evaluate_path_breach_after_recovery_fires_again():
    state1, _, _ = watchdog.evaluate_path({}, 95.0, None, 90.0, _now())
    state1["notified"] = True
    state2, _, _ = watchdog.evaluate_path(state1, 50.0, None, 90.0, _now(1))
    state3, status, should_notify = watchdog.evaluate_path(state2, 95.0, None, 90.0, _now(2))
    assert status == "breached"
    assert should_notify is True


def test_evaluate_path_exactly_at_threshold_counts_as_breached():
    _, status, _ = watchdog.evaluate_path({}, 90.0, None, 90.0, _now())
    assert status == "breached"


def test_evaluate_path_error_is_new_event():
    state, status, should_notify = watchdog.evaluate_path({}, None, "permission denied", 90.0, _now())
    assert status == "error"
    assert should_notify is True
    assert state["last_error"] == "permission denied"
    assert state["last_percent_used"] is None


def test_evaluate_path_repeated_error_after_successful_notify_does_not_refire():
    state1, _, _ = watchdog.evaluate_path({}, None, "permission denied", 90.0, _now())
    state1["notified"] = True
    state2, status, should_notify = watchdog.evaluate_path(state1, None, "permission denied", 90.0, _now(1))
    assert status == "error"
    assert should_notify is False


def test_evaluate_path_repeated_error_retries_if_previous_notify_never_confirmed():
    state1, _, _ = watchdog.evaluate_path({}, None, "permission denied", 90.0, _now())
    assert state1["notified"] is False
    state2, status, should_notify = watchdog.evaluate_path(state1, None, "permission denied", 90.0, _now(1))
    assert status == "error"
    assert should_notify is True


def test_evaluate_path_error_then_ok_then_error_fires_again():
    state1, _, _ = watchdog.evaluate_path({}, None, "boom", 90.0, _now())
    state1["notified"] = True
    state2, _, _ = watchdog.evaluate_path(state1, 10.0, None, 90.0, _now(1))
    state3, status, should_notify = watchdog.evaluate_path(state2, None, "boom again", 90.0, _now(2))
    assert status == "error"
    assert should_notify is True


def test_evaluate_path_breach_directly_to_error_is_new_event():
    # status changes from "breached" to "error" -- still a distinct new event
    # (different failure mode), not swallowed by "already flagged."
    state1, _, _ = watchdog.evaluate_path({}, 95.0, None, 90.0, _now())
    state1["notified"] = True
    state2, status, should_notify = watchdog.evaluate_path(state1, None, "mount vanished", 90.0, _now(1))
    assert status == "error"
    assert should_notify is True


# ---------------------------------------------------------------------------
# run() -- orchestration, notify wiring, debounce across ticks
# ---------------------------------------------------------------------------


def _fake_notify(ok: bool = True) -> MagicMock:
    """Mirrors the real NotifyClient.attention_request() contract: it never
    raises for a network failure, it returns NotificationAccepted(ok=...)."""
    notify = MagicMock()
    notify.attention_request = MagicMock(return_value=MagicMock(ok=ok, detail=None))
    return notify


def test_run_fires_notify_once_on_new_breach_and_writes_state(tmp_path):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = _fake_notify()

    # threshold_pct=0 guarantees a breach against any real filesystem usage.
    state, any_bad = watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now())

    assert any_bad is True
    assert notify.attention_request.call_count == 1
    call_kwargs = notify.attention_request.call_args.kwargs
    assert call_kwargs["severity"] == "warning"
    assert call_kwargs["context"]["path"] == str(small_dir)
    assert state_file.exists()
    persisted = json.loads(state_file.read_text())
    assert persisted["paths"][str(small_dir)]["last_status"] == "breached"
    assert persisted["paths"][str(small_dir)]["notified"] is True


def test_run_does_not_refire_notify_on_second_tick_still_breached(tmp_path):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = _fake_notify()

    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now())
    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now(15))

    assert notify.attention_request.call_count == 1


def test_run_retries_notify_next_tick_when_notify_returns_ok_false(tmp_path):
    # Regression test for the code-review finding: the REAL NotifyClient
    # never raises on network failure -- it returns ok=False. A prior
    # version of this script treated the call as "handled" the moment it
    # was attempted, regardless of the result, which meant a breach that
    # first occurred while orion-notify was down/unreachable would never
    # be retried and no Pending Attention card would ever land.
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = _fake_notify(ok=False)

    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now())
    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now(15))
    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now(30))

    assert notify.attention_request.call_count == 3
    persisted = json.loads(state_file.read_text())
    assert persisted["paths"][str(small_dir)]["notified"] is False


def test_run_stops_retrying_once_a_notify_attempt_succeeds(tmp_path):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = MagicMock()
    notify.attention_request = MagicMock(
        side_effect=[
            MagicMock(ok=False, detail="connection refused"),
            MagicMock(ok=True, detail=None),
        ]
    )

    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now())      # fails
    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now(15))    # succeeds
    watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now(30))    # should not retry

    assert notify.attention_request.call_count == 2
    persisted = json.loads(state_file.read_text())
    assert persisted["paths"][str(small_dir)]["notified"] is True


def test_run_does_not_fire_notify_when_under_threshold(tmp_path):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = _fake_notify()

    state, any_bad = watchdog.run([str(small_dir)], 100.0, state_file, notify, now=_now())

    assert any_bad is False
    assert notify.attention_request.call_count == 0


def test_run_fires_error_severity_for_missing_path(tmp_path):
    state_file = tmp_path / "state.json"
    notify = _fake_notify()

    state, any_bad = watchdog.run(
        ["/nonexistent/definitely/not/real"], 90.0, state_file, notify, now=_now()
    )

    assert any_bad is True
    assert notify.attention_request.call_count == 1
    assert notify.attention_request.call_args.kwargs["severity"] == "error"


def test_run_checks_multiple_paths_independently(tmp_path):
    breached_dir = tmp_path / "full"
    breached_dir.mkdir()
    ok_dir = tmp_path / "empty"
    ok_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = _fake_notify()

    # Force one path to read as breached via a monkeypatched measure_path
    # keyed by path, since we can't actually fill a real tmp filesystem.
    original = watchdog.measure_path

    def fake_measure(path):
        if path == str(breached_dir):
            return 99.0, None
        return original(path)

    watchdog.measure_path = fake_measure
    try:
        state, any_bad = watchdog.run(
            [str(breached_dir), str(ok_dir)], 90.0, state_file, notify, now=_now()
        )
    finally:
        watchdog.measure_path = original

    assert any_bad is True
    assert notify.attention_request.call_count == 1
    persisted_paths = state["paths"]
    assert persisted_paths[str(breached_dir)]["last_status"] == "breached"
    assert persisted_paths[str(ok_dir)]["last_status"] == "ok"


def test_run_notify_failure_does_not_crash_watchdog(tmp_path):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = _fake_notify()
    notify.attention_request.side_effect = RuntimeError("orion-notify unreachable")

    # Must not raise even though the notify call inside failed.
    state, any_bad = watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now())
    assert any_bad is True
    assert state_file.exists()


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------


def test_state_lock_blocks_concurrent_acquisition(tmp_path):
    state_file = tmp_path / "state.json"
    with watchdog._StateLock(state_file):
        with pytest.raises(watchdog.WatchdogLockedError):
            with watchdog._StateLock(state_file):
                pass


def test_state_lock_is_released_after_context_exit(tmp_path):
    state_file = tmp_path / "state.json"
    with watchdog._StateLock(state_file):
        pass
    # Should not raise -- the lock was released.
    with watchdog._StateLock(state_file):
        pass


def test_run_skips_cleanly_when_lock_already_held(tmp_path):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    notify = _fake_notify()

    with watchdog._StateLock(state_file):
        with pytest.raises(watchdog.WatchdogLockedError):
            watchdog.run([str(small_dir)], 0.0, state_file, notify, now=_now())
    # No notify call happened because run() never got past the lock.
    assert notify.attention_request.call_count == 0


# ---------------------------------------------------------------------------
# load_state resilience
# ---------------------------------------------------------------------------


def test_load_state_returns_empty_for_missing_file(tmp_path):
    state = watchdog.load_state(tmp_path / "does_not_exist.json")
    assert state == {"paths": {}}


def test_load_state_recovers_from_corrupt_json(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{not valid json")
    state = watchdog.load_state(state_file)
    assert state == {"paths": {}}


def test_load_state_recovers_from_wrong_shape(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"unexpected": "shape"}))
    state = watchdog.load_state(state_file)
    assert state == {"paths": {}}


# ---------------------------------------------------------------------------
# main() CLI / exit codes
# ---------------------------------------------------------------------------


def test_main_exits_zero_when_all_paths_ok(tmp_path, monkeypatch):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(watchdog, "NotifyClient", lambda **kw: _fake_notify())

    rc = watchdog.main(
        ["--paths", str(small_dir), "--threshold-pct", "100", "--state-file", str(state_file)]
    )
    assert rc == 0


def test_main_exits_one_when_a_path_is_breached(tmp_path, monkeypatch):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(watchdog, "NotifyClient", lambda **kw: _fake_notify())

    rc = watchdog.main(
        ["--paths", str(small_dir), "--threshold-pct", "0", "--state-file", str(state_file)]
    )
    assert rc == 1


def test_main_exits_two_on_state_write_permission_error(tmp_path, monkeypatch):
    small_dir = tmp_path / "mount"
    small_dir.mkdir()
    unwritable_state_file = tmp_path / "no_such_parent" / "state.json"
    monkeypatch.setattr(watchdog, "NotifyClient", lambda **kw: _fake_notify())

    def boom(*a, **kw):
        raise OSError("permission denied")

    monkeypatch.setattr(watchdog, "_atomic_write_json", boom)

    rc = watchdog.main(
        ["--paths", str(small_dir), "--threshold-pct", "100", "--state-file", str(unwritable_state_file)]
    )
    assert rc == 2


def test_main_exits_three_on_unexpected_exception(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(watchdog, "NotifyClient", lambda **kw: _fake_notify())

    def boom(*a, **kw):
        raise ValueError("bug")

    monkeypatch.setattr(watchdog, "run", boom)

    rc = watchdog.main(["--paths", "/tmp", "--state-file", str(state_file)])
    assert rc == 3


def test_default_paths_include_docker_scripts_telemetry():
    assert watchdog.DEFAULT_PATHS == ("/mnt/docker", "/mnt/scripts", "/mnt/telemetry")


def test_default_state_file_lives_under_telemetry_root():
    path = watchdog.default_state_file("/mnt/telemetry", "orion-athena")
    assert str(path) == "/mnt/telemetry/orion-athena/disk-watchdog/state.json"
