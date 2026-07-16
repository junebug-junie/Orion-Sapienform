from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bus_core_health_watchdog as watchdog  # noqa: E402


def _now(offset_minutes: float = 0.0) -> datetime:
    return datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)


# ---------------------------------------------------------------------------
# container_name_for_project / default paths
# ---------------------------------------------------------------------------


def test_container_name_matches_compose_pattern():
    # services/orion-bus/docker-compose.yml: container_name: orion-${PROJECT}-bus-core
    assert watchdog.container_name_for_project("orion-athena") == "orion-orion-athena-bus-core"


def test_default_state_and_marker_paths_do_not_collide_with_aof_data_mount():
    state_file = watchdog.default_state_file("/mnt/telemetry", "orion-athena")
    marker = watchdog.default_alert_marker("/mnt/telemetry", "orion-athena")
    # services/orion-bus/docker-compose.yml mounts ${TELEMETRY_ROOT}/${PROJECT}/bus/data
    # -- neither watchdog path may live inside that directory.
    data_dir = Path("/mnt/telemetry/orion-athena/bus/data")
    assert data_dir not in state_file.parents
    assert data_dir not in marker.parents
    assert str(state_file).startswith("/mnt/telemetry/orion-athena/bus/")
    assert str(marker).startswith("/mnt/telemetry/orion-athena/bus/")


# ---------------------------------------------------------------------------
# evaluate() -- pure threshold/alert logic, no docker/filesystem/clock
# ---------------------------------------------------------------------------


def test_healthy_observation_resets_streak_and_stamps_last_healthy():
    state = watchdog._empty_state()
    state["consecutive_unhealthy_count"] = 2
    new_state, detected, reasons = watchdog.evaluate(state, "healthy", restart_count=0, now=_now())
    assert new_state["consecutive_unhealthy_count"] == 0
    assert new_state["last_known_healthy_at"] == _now().isoformat()
    assert detected is False
    assert reasons == []


def test_unhealthy_streak_below_threshold_does_not_alert():
    state = watchdog._empty_state()
    for i in range(2):
        state, detected, reasons = watchdog.evaluate(
            state, "unhealthy", restart_count=0, now=_now(i), unhealthy_streak_threshold=3
        )
    assert state["consecutive_unhealthy_count"] == 2
    assert detected is False


def test_unhealthy_streak_crossing_threshold_alerts():
    state = watchdog._empty_state()
    for i in range(3):
        state, detected, reasons = watchdog.evaluate(
            state, "unhealthy", restart_count=0, now=_now(i), unhealthy_streak_threshold=3
        )
    assert state["consecutive_unhealthy_count"] == 3
    assert detected is True
    assert any("consecutive unhealthy" in r for r in reasons)


def test_starting_status_is_neutral_does_not_increment_or_reset():
    state = watchdog._empty_state()
    state["consecutive_unhealthy_count"] = 2
    new_state, detected, _ = watchdog.evaluate(state, "starting", restart_count=0, now=_now())
    assert new_state["consecutive_unhealthy_count"] == 2  # unchanged
    assert detected is False


def test_none_health_status_is_neutral():
    state = watchdog._empty_state()
    state["consecutive_unhealthy_count"] = 1
    new_state, detected, _ = watchdog.evaluate(state, "none", restart_count=0, now=_now())
    assert new_state["consecutive_unhealthy_count"] == 1
    assert detected is False


def test_missing_container_counts_as_unhealthy():
    state = watchdog._empty_state()
    for i in range(3):
        state, detected, reasons = watchdog.evaluate(
            state, "missing", restart_count=0, now=_now(i), unhealthy_streak_threshold=3
        )
    assert detected is True


def test_docker_unreachable_counts_as_unhealthy():
    state = watchdog._empty_state()
    for i in range(3):
        state, detected, reasons = watchdog.evaluate(
            state, "docker_unreachable", restart_count=0, now=_now(i), unhealthy_streak_threshold=3
        )
    assert detected is True


def test_restart_count_within_window_crosses_threshold():
    # Simulates a rapid crash loop that never settles into "unhealthy" between
    # polls: health flaps back to "starting" each time but RestartCount climbs.
    state = watchdog._empty_state()
    restart_counts = [0, 1, 2, 3]
    for i, rc in enumerate(restart_counts):
        state, detected, reasons = watchdog.evaluate(
            state,
            "starting",
            restart_count=rc,
            now=_now(i),
            restart_count_threshold=3,
            restart_window_minutes=10,
        )
    assert detected is True
    assert any("restart" in r for r in reasons)
    assert state["restarts_in_window"] == 3


def test_restart_count_samples_prune_outside_window():
    state = watchdog._empty_state()
    # First restart sample far outside the window...
    state, _, _ = watchdog.evaluate(state, "healthy", restart_count=1, now=_now(0), restart_window_minutes=10)
    # ...then a burst of restarts well after the window has rolled past the first sample.
    state, detected, reasons = watchdog.evaluate(
        state, "healthy", restart_count=2, now=_now(30), restart_count_threshold=1, restart_window_minutes=10
    )
    # Only the current sample is in-window (30min gap > 10min window), so the
    # delta against the oldest in-window sample is 0, not 1.
    assert state["restarts_in_window"] == 0
    assert detected is False


def test_restart_count_threshold_not_crossed_stays_healthy():
    state = watchdog._empty_state()
    for i, rc in enumerate([0, 1]):
        state, detected, reasons = watchdog.evaluate(
            state, "healthy", restart_count=rc, now=_now(i), restart_count_threshold=3, restart_window_minutes=10
        )
    assert detected is False


def test_crash_loop_first_detected_at_is_set_once_and_persists():
    state = watchdog._empty_state()
    for i in range(3):
        state, detected, _ = watchdog.evaluate(
            state, "unhealthy", restart_count=0, now=_now(i), unhealthy_streak_threshold=3
        )
    first_detected = state["crash_loop_first_detected_at"]
    assert first_detected is not None
    # A subsequent still-unhealthy check must not reset first_detected_at.
    state, detected, _ = watchdog.evaluate(
        state, "unhealthy", restart_count=0, now=_now(3), unhealthy_streak_threshold=3
    )
    assert state["crash_loop_first_detected_at"] == first_detected
    assert state["crash_loop_active"] is True


def test_recovery_clears_crash_loop_active_but_keeps_first_detected_history():
    state = watchdog._empty_state()
    for i in range(3):
        state, _, _ = watchdog.evaluate(state, "unhealthy", restart_count=0, now=_now(i), unhealthy_streak_threshold=3)
    assert state["crash_loop_active"] is True
    state, detected, reasons = watchdog.evaluate(state, "healthy", restart_count=0, now=_now(3))
    assert detected is False
    assert state["crash_loop_active"] is False


def test_unknown_health_status_treated_conservatively_as_unhealthy():
    state = watchdog._empty_state()
    for i in range(3):
        state, detected, _ = watchdog.evaluate(
            state, "some-new-docker-status", restart_count=0, now=_now(i), unhealthy_streak_threshold=3
        )
    assert detected is True


# ---------------------------------------------------------------------------
# inspect_container() -- subprocess boundary, mocked
# ---------------------------------------------------------------------------


def _fake_completed(returncode=0, stdout="", stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_inspect_container_parses_healthy_state():
    payload = json.dumps([{"State": {"Status": "running", "Health": {"Status": "healthy"}}, "RestartCount": 2}])
    with patch("subprocess.run", return_value=_fake_completed(0, payload)):
        result = watchdog.inspect_container("orion-orion-athena-bus-core")
    assert result == {"health_status": "healthy", "restart_count": 2, "container_status": "running"}


def test_inspect_container_no_health_block_reports_none():
    payload = json.dumps([{"State": {"Status": "running"}, "RestartCount": 0}])
    with patch("subprocess.run", return_value=_fake_completed(0, payload)):
        result = watchdog.inspect_container("some-container")
    assert result["health_status"] == "none"


def test_inspect_container_missing_container_lowercase_error():
    # Real docker CLI observed wording (this host): "error: no such object: <name>"
    with patch("subprocess.run", return_value=_fake_completed(1, "", "error: no such object: nope")):
        result = watchdog.inspect_container("nope")
    assert result["health_status"] == "missing"


def test_inspect_container_docker_daemon_unreachable():
    with patch("subprocess.run", return_value=_fake_completed(1, "", "Cannot connect to the Docker daemon")):
        result = watchdog.inspect_container("some-container")
    assert result["health_status"] == "docker_unreachable"


def test_inspect_container_binary_not_found_raises():
    with patch("subprocess.run", side_effect=FileNotFoundError("no docker")):
        with pytest.raises(watchdog.DockerUnavailableError):
            watchdog.inspect_container("some-container", docker_bin="/nonexistent/docker")


def test_inspect_container_timeout_is_docker_unreachable_not_a_hard_failure():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=15)):
        result = watchdog.inspect_container("some-container")
    assert result["health_status"] == "docker_unreachable"


# ---------------------------------------------------------------------------
# state persistence + alert marker, end to end via run()
# ---------------------------------------------------------------------------


def test_run_writes_state_file_and_no_marker_when_healthy(tmp_path):
    state_file = tmp_path / "state.json"
    marker = tmp_path / "ALERT.txt"
    payload = json.dumps([{"State": {"Status": "running", "Health": {"Status": "healthy"}}, "RestartCount": 0}])
    with patch("subprocess.run", return_value=_fake_completed(0, payload)):
        state, detected, reasons = watchdog.run(
            container="fake-container", state_file=state_file, alert_marker=marker, now=_now()
        )
    assert detected is False
    assert state_file.exists()
    assert not marker.exists()
    persisted = json.loads(state_file.read_text())
    assert persisted["last_health_status"] == "healthy"


def test_run_writes_alert_marker_on_crash_loop(tmp_path):
    state_file = tmp_path / "state.json"
    marker = tmp_path / "ALERT.txt"
    payload = json.dumps([{"State": {"Status": "restarting", "Health": {"Status": "unhealthy"}}, "RestartCount": 5}])
    with patch("subprocess.run", return_value=_fake_completed(0, payload)):
        for i in range(3):
            state, detected, reasons = watchdog.run(
                container="fake-container",
                state_file=state_file,
                alert_marker=marker,
                unhealthy_streak_threshold=3,
                now=_now(i),
            )
    assert detected is True
    assert marker.exists()
    content = marker.read_text()
    assert "CRASH-LOOP ALERT" in content
    assert "fake-container" in content
    assert "consecutive unhealthy" in content


def test_run_marker_is_not_deleted_on_recovery_but_gets_resolved_footer(tmp_path):
    state_file = tmp_path / "state.json"
    marker = tmp_path / "ALERT.txt"
    unhealthy_payload = json.dumps([{"State": {"Status": "restarting", "Health": {"Status": "unhealthy"}}, "RestartCount": 5}])
    healthy_payload = json.dumps([{"State": {"Status": "running", "Health": {"Status": "healthy"}}, "RestartCount": 5}])
    with patch("subprocess.run", return_value=_fake_completed(0, unhealthy_payload)):
        for i in range(3):
            watchdog.run(
                container="fake-container",
                state_file=state_file,
                alert_marker=marker,
                unhealthy_streak_threshold=3,
                now=_now(i),
            )
    assert marker.exists()
    original_content = marker.read_text()

    with patch("subprocess.run", return_value=_fake_completed(0, healthy_payload)):
        state, detected, _ = watchdog.run(
            container="fake-container", state_file=state_file, alert_marker=marker, now=_now(3)
        )
    assert detected is False
    assert marker.exists()  # never auto-deleted
    new_content = marker.read_text()
    assert new_content.startswith(original_content)
    assert "RESOLVED" in new_content


def test_run_state_file_is_valid_json_shape_against_real_container_semantics(tmp_path):
    # Regression guard for the exact shape asserted by the live-container smoke
    # test in this PR: consecutive_unhealthy_count, last_known_healthy_at,
    # restart_samples, crash_loop_active must all be present.
    state_file = tmp_path / "state.json"
    marker = tmp_path / "ALERT.txt"
    payload = json.dumps([{"State": {"Status": "running", "Health": {"Status": "healthy"}}, "RestartCount": 0}])
    with patch("subprocess.run", return_value=_fake_completed(0, payload)):
        watchdog.run(container="fake-container", state_file=state_file, alert_marker=marker, now=_now())
    persisted = json.loads(state_file.read_text())
    for key in (
        "consecutive_unhealthy_count",
        "last_known_healthy_at",
        "restart_samples",
        "crash_loop_active",
        "crash_loop_first_detected_at",
        "last_check_at",
        "last_health_status",
        "last_restart_count",
        "restarts_in_window",
    ):
        assert key in persisted


def test_load_state_recovers_from_corrupt_file(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{not valid json")
    state = watchdog.load_state(state_file)
    assert state == watchdog._empty_state()


def test_atomic_write_survives_no_partial_file_on_success(tmp_path):
    state_file = tmp_path / "nested" / "state.json"
    watchdog._atomic_write_json(state_file, {"a": 1})
    assert state_file.exists()
    assert json.loads(state_file.read_text()) == {"a": 1}
    # no leftover temp files
    leftovers = list(state_file.parent.glob(".tmp-*"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------


def test_main_exits_zero_when_healthy(tmp_path):
    payload = json.dumps([{"State": {"Status": "running", "Health": {"Status": "healthy"}}, "RestartCount": 0}])
    with patch("subprocess.run", return_value=_fake_completed(0, payload)):
        exit_code = watchdog.main([
            "--container", "fake-container",
            "--state-file", str(tmp_path / "state.json"),
            "--alert-marker", str(tmp_path / "ALERT.txt"),
        ])
    assert exit_code == 0


def test_main_exits_one_when_crash_loop_detected(tmp_path):
    payload = json.dumps([{"State": {"Status": "restarting", "Health": {"Status": "unhealthy"}}, "RestartCount": 9}])
    state_file = tmp_path / "state.json"
    marker = tmp_path / "ALERT.txt"
    with patch("subprocess.run", return_value=_fake_completed(0, payload)):
        for _ in range(3):
            exit_code = watchdog.main([
                "--container", "fake-container",
                "--state-file", str(state_file),
                "--alert-marker", str(marker),
                "--unhealthy-streak-threshold", "3",
            ])
    assert exit_code == 1
    assert marker.exists()


def test_main_exits_two_when_docker_binary_missing(tmp_path):
    with patch("subprocess.run", side_effect=FileNotFoundError("no docker")):
        exit_code = watchdog.main([
            "--docker-bin", "/nonexistent/docker",
            "--state-file", str(tmp_path / "state.json"),
            "--alert-marker", str(tmp_path / "ALERT.txt"),
        ])
    assert exit_code == 2


def test_main_uses_project_default_container_name(tmp_path):
    payload = json.dumps([{"State": {"Status": "running", "Health": {"Status": "healthy"}}, "RestartCount": 0}])
    captured_args = {}

    def _fake_run(cmd, **kwargs):
        captured_args["cmd"] = cmd
        return _fake_completed(0, payload)

    with patch("subprocess.run", side_effect=_fake_run):
        watchdog.main([
            "--project", "orion-athena",
            "--state-file", str(tmp_path / "state.json"),
            "--alert-marker", str(tmp_path / "ALERT.txt"),
        ])
    assert "orion-orion-athena-bus-core" in captured_args["cmd"]
