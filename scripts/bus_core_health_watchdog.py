#!/usr/bin/env python3
"""Host-level crash-loop watchdog for `bus-core` (the Redis container behind
`services/orion-bus/docker-compose.yml`).

Context: `bus-core` periodically crash-loops from AOF corruption (auto-repair
for that is a separate, parallel piece of work -- this script does not touch
`bus-core`'s own container definition or entrypoint). The real gap this
closes: detecting "bus is stuck in a crash loop" currently depends on either
a human noticing cascading failures across many *other* services, or on
signals that themselves route through the bus or Postgres -- both of which
can be down *at the same time* (confirmed, not hypothetical). This script is
a detection floor that survives both being down simultaneously:

- No Redis connection, ever. Health comes from `docker inspect` only.
- No Postgres connection, ever. State is a local JSON file, not a DB row.
- No bus publish. Alerting is a marker file on local disk, not an event.

Same category as `scripts/check_concept_relation_digest_liveness.py`
(standalone script, run on demand or via cron/systemd timer, not a live
service loop) but for container liveness instead of a Postgres backlog.

Two independent crash-loop signatures, either one fires an alert:

1. **Consecutive-unhealthy streak**: `--unhealthy-streak-threshold`
   (default 3) consecutive watchdog runs where `docker inspect`'s
   `.State.Health.Status` was not `healthy` (`unhealthy`, the container was
   missing, or the Docker daemon itself was unreachable -- all three count
   against the streak; `starting`/`none` are neutral and leave the streak
   unchanged, since a normal restart briefly reports `starting` and that
   alone should not false-positive).
2. **Restart-count-in-window**: `--restart-count-threshold` (default 3)
   container restarts (per `docker inspect`'s `.RestartCount`, a monotonic
   cumulative counter) within `--restart-window-minutes` (default 10). This
   catches a crash loop that never settles into `unhealthy` between polls
   (e.g. the container dies and restarts faster than the poll interval).

State (consecutive-unhealthy count, last-known-healthy-at, a pruned rolling
window of `(timestamp, restart_count)` samples) is persisted to
`--state-file` (default `${TELEMETRY_ROOT}/${PROJECT}/bus/watchdog/
health_watchdog_state.json` -- a sibling of, not inside, the AOF data mount
at `${TELEMETRY_ROOT}/${PROJECT}/bus/data` that a parallel AOF-repair task
may also be touching).

On a crash-loop signature: this repo has no notify-send/osascript/desktop-
notification script anywhere (checked before writing this one) and the one
HTTP alerting path that exists (`orion-notify`'s `/attention/request`, used
by `orion-mesh-guardian` and the Fuseki recover job) is itself infrastructure
that can plausibly be down in the same incident this script exists to catch.
So the alert is a plain, loud, impossible-to-miss marker file:
`--alert-marker` (default `${TELEMETRY_ROOT}/${PROJECT}/bus/
ORION_BUS_CRASH_LOOP_ALERT.txt`, a sibling of `bus/data/` so it shows up the
moment anyone lists the bus telemetry directory). The marker is never
auto-deleted -- if the condition later clears, the file is updated in place
with a RESOLVED footer rather than removed, so a human still sees that an
incident happened. A human removes the file once they've looked.

Usage:
    python scripts/bus_core_health_watchdog.py
    python scripts/bus_core_health_watchdog.py --project orion-athena --json
    python scripts/bus_core_health_watchdog.py --container orion-orion-athena-bus-core \\
        --unhealthy-streak-threshold 3 --restart-count-threshold 3 --restart-window-minutes 10

Exit codes: 0 = healthy, no crash-loop signature met.
            1 = crash-loop signature met -- alert marker written/updated.
            2 = could not run the check at all (docker binary missing).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/bus_core_health_watchdog.py` puts scripts/ on
# sys.path[0], which can shadow stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_activation_saturation.py
# / scripts/check_concept_relation_digest_liveness.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

DEFAULT_UNHEALTHY_STREAK_THRESHOLD = 3
DEFAULT_RESTART_COUNT_THRESHOLD = 3
DEFAULT_RESTART_WINDOW_MINUTES = 10.0

_UNHEALTHY_LIKE_STATUSES = {"unhealthy", "missing", "docker_unreachable"}
_NEUTRAL_STATUSES = {"starting", "none"}


def container_name_for_project(project: str) -> str:
    """Matches services/orion-bus/docker-compose.yml's
    `container_name: orion-${PROJECT}-bus-core`."""
    return f"orion-{project}-bus-core"


def default_state_file(telemetry_root: str, project: str) -> Path:
    return Path(telemetry_root) / project / "bus" / "watchdog" / "health_watchdog_state.json"


def default_alert_marker(telemetry_root: str, project: str) -> Path:
    return Path(telemetry_root) / project / "bus" / "ORION_BUS_CRASH_LOOP_ALERT.txt"


class DockerUnavailableError(RuntimeError):
    """The `docker` binary itself could not be run -- a hard tooling failure,
    distinct from the container being missing or unhealthy (both of which are
    themselves alarming signals, not script errors)."""


def inspect_container(container: str, docker_bin: str = "docker") -> dict[str, Any]:
    """Runs `docker inspect <container>` and returns a normalized dict:

        {"health_status": "healthy"|"unhealthy"|"starting"|"none"|"missing"|"docker_unreachable",
         "restart_count": int,
         "container_status": str}  # docker's .State.Status, informational only

    Never raises for "the container isn't in the state we want" -- only for
    genuine tooling failure (docker binary not found).
    """
    try:
        proc = subprocess.run(
            [docker_bin, "inspect", container],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise DockerUnavailableError(f"'{docker_bin}' binary not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        # A hung docker daemon is functionally the same signal as "can't reach
        # it" for our purposes, not a hard tooling failure -- keep going.
        return {"health_status": "docker_unreachable", "restart_count": 0, "container_status": f"timeout: {exc}"}

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        # Observed real docker CLI wording is lowercase ("error: no such
        # object: <name>"), matched case-insensitively so older/newer docker
        # versions using "No such container"/"No such object" still match.
        if "no such object" in stderr.lower() or "no such container" in stderr.lower():
            return {"health_status": "missing", "restart_count": 0, "container_status": stderr or "missing"}
        return {"health_status": "docker_unreachable", "restart_count": 0, "container_status": stderr or "docker inspect failed"}

    try:
        parsed = json.loads(proc.stdout)
        entry = parsed[0]
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        return {"health_status": "docker_unreachable", "restart_count": 0, "container_status": f"unparseable docker inspect output: {exc}"}

    state = entry.get("State", {}) or {}
    restart_count = int(entry.get("RestartCount", 0) or 0)
    container_status = state.get("Status", "unknown")
    health = state.get("Health")
    if not health:
        health_status = "none"
    else:
        health_status = health.get("Status", "none")

    return {
        "health_status": health_status,
        "restart_count": restart_count,
        "container_status": container_status,
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        with path.open("r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("state file did not contain a JSON object")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(
            f"bus_core_health_watchdog: WARNING -- state file {path} unreadable/corrupt "
            f"({exc}), starting from a fresh state.",
            file=sys.stderr,
        )
        return _empty_state()


def _empty_state() -> dict[str, Any]:
    return {
        "consecutive_unhealthy_count": 0,
        "last_known_healthy_at": None,
        "restart_samples": [],
        "crash_loop_active": False,
        "crash_loop_first_detected_at": None,
    }


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def prune_restart_samples(
    samples: list[dict[str, Any]], now: datetime, window_minutes: float
) -> list[dict[str, Any]]:
    cutoff = now.timestamp() - (window_minutes * 60.0)
    kept = []
    for sample in samples:
        try:
            at = _parse_iso(sample["at"])
        except (KeyError, ValueError):
            continue
        if at.timestamp() >= cutoff:
            kept.append(sample)
    return kept


def evaluate(
    state: dict[str, Any],
    health_status: str,
    restart_count: int,
    now: datetime,
    unhealthy_streak_threshold: int = DEFAULT_UNHEALTHY_STREAK_THRESHOLD,
    restart_count_threshold: int = DEFAULT_RESTART_COUNT_THRESHOLD,
    restart_window_minutes: float = DEFAULT_RESTART_WINDOW_MINUTES,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Pure function: given the current persisted state plus one fresh
    (health_status, restart_count) observation, returns
    (updated_state, crash_loop_detected, reasons).

    Deliberately takes no docker/filesystem/clock dependency directly so it
    can be unit-tested against a fake sequence of health-check results
    without touching a real container or the real clock.
    """
    new_state = dict(state)
    new_state.setdefault("restart_samples", [])
    now_iso = now.isoformat()

    if health_status == "healthy":
        new_state["consecutive_unhealthy_count"] = 0
        new_state["last_known_healthy_at"] = now_iso
    elif health_status in _UNHEALTHY_LIKE_STATUSES:
        new_state["consecutive_unhealthy_count"] = int(state.get("consecutive_unhealthy_count", 0)) + 1
    elif health_status in _NEUTRAL_STATUSES:
        pass  # leave streak unchanged -- ambiguous signal (e.g. just restarted)
    else:
        # Unknown health_status string -- treat conservatively like "unhealthy"
        # rather than silently ignoring it (no empty-shell success states).
        new_state["consecutive_unhealthy_count"] = int(state.get("consecutive_unhealthy_count", 0)) + 1

    samples = prune_restart_samples(list(state.get("restart_samples", [])), now, restart_window_minutes)
    samples.append({"at": now_iso, "restart_count": restart_count})
    new_state["restart_samples"] = samples

    restarts_in_window = 0
    if samples:
        oldest_in_window = min(s["restart_count"] for s in samples)
        restarts_in_window = max(0, restart_count - oldest_in_window)

    reasons: list[str] = []
    streak = new_state["consecutive_unhealthy_count"]
    if streak >= unhealthy_streak_threshold:
        reasons.append(
            f"{streak} consecutive unhealthy check(s) (threshold {unhealthy_streak_threshold})"
        )
    if restarts_in_window >= restart_count_threshold:
        reasons.append(
            f"{restarts_in_window} restart(s) within {restart_window_minutes:.1f} minute window "
            f"(threshold {restart_count_threshold})"
        )

    crash_loop_detected = bool(reasons)
    new_state["restarts_in_window"] = restarts_in_window
    new_state["last_health_status"] = health_status
    new_state["last_restart_count"] = restart_count
    new_state["last_check_at"] = now_iso

    if crash_loop_detected and not state.get("crash_loop_active", False):
        new_state["crash_loop_first_detected_at"] = now_iso
    new_state["crash_loop_active"] = crash_loop_detected

    return new_state, crash_loop_detected, reasons


def write_alert_marker(path: Path, container: str, state: dict[str, Any], reasons: list[str], now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ORION BUS-CORE CRASH-LOOP ALERT",
        "================================",
        f"container:            {container}",
        f"detected_first_at:    {state.get('crash_loop_first_detected_at')}",
        f"last_check_at:        {now.isoformat()}",
        f"last_known_healthy_at:{state.get('last_known_healthy_at')}",
        f"consecutive_unhealthy_count: {state.get('consecutive_unhealthy_count')}",
        f"restarts_in_window:   {state.get('restarts_in_window')}",
        "",
        "Reason(s):",
    ]
    lines += [f"  - {r}" for r in reasons]
    lines += [
        "",
        "This means bus-core (Redis) is very likely stuck in a crash loop.",
        "Everything routed through the bus, and anything that depends on the",
        "bus to report its own failures, may be silently degraded right now.",
        "",
        "Check:",
        f"  docker logs --tail=200 {container}",
        f"  docker inspect {container} --format '{{{{json .State}}}}'",
        "  services/orion-bus/docker-compose.yml (AOF corruption is the known",
        "  recurring cause -- a separate task is adding auto-repair for that).",
        "",
        "This file is written by scripts/bus_core_health_watchdog.py and is",
        "NOT auto-deleted. Once you've looked, delete it by hand:",
        f"  rm {path}",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_resolved_footer(path: Path, now: datetime) -> None:
    if not path.exists():
        return
    with path.open("a") as fh:
        fh.write(f"\n[RESOLVED as of {now.isoformat()} -- watchdog no longer sees a crash-loop signature.\n")
        fh.write(" This file is left in place so the incident isn't silently lost. Delete by hand once reviewed.]\n")


def run(
    container: str,
    state_file: Path,
    alert_marker: Path,
    docker_bin: str = "docker",
    unhealthy_streak_threshold: int = DEFAULT_UNHEALTHY_STREAK_THRESHOLD,
    restart_count_threshold: int = DEFAULT_RESTART_COUNT_THRESHOLD,
    restart_window_minutes: float = DEFAULT_RESTART_WINDOW_MINUTES,
    now: datetime | None = None,
) -> tuple[dict[str, Any], bool, list[str]]:
    now = now or datetime.now(timezone.utc)
    observation = inspect_container(container, docker_bin=docker_bin)
    state = load_state(state_file)
    was_active = bool(state.get("crash_loop_active", False))

    new_state, crash_loop_detected, reasons = evaluate(
        state,
        observation["health_status"],
        observation["restart_count"],
        now,
        unhealthy_streak_threshold=unhealthy_streak_threshold,
        restart_count_threshold=restart_count_threshold,
        restart_window_minutes=restart_window_minutes,
    )
    new_state["container_status"] = observation["container_status"]
    _atomic_write_json(state_file, new_state)

    if crash_loop_detected:
        write_alert_marker(alert_marker, container, new_state, reasons, now)
    elif was_active and not crash_loop_detected:
        append_resolved_footer(alert_marker, now)

    return new_state, crash_loop_detected, reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--project",
        default=os.getenv("PROJECT", "orion-athena"),
        help="Compose project name, matches $PROJECT (see .env). Used to derive the default "
        "container name and telemetry paths. Default: $PROJECT or 'orion-athena'.",
    )
    parser.add_argument(
        "--telemetry-root",
        default=os.getenv("TELEMETRY_ROOT", "/mnt/telemetry"),
        help="Root of the telemetry tree, matches $TELEMETRY_ROOT. Default: $TELEMETRY_ROOT or /mnt/telemetry.",
    )
    parser.add_argument(
        "--container",
        default=None,
        help="Exact container name to inspect. Default: orion-<project>-bus-core "
        "(matches services/orion-bus/docker-compose.yml's container_name).",
    )
    parser.add_argument("--state-file", default=None, help="Path to the watchdog's own state JSON file.")
    parser.add_argument("--alert-marker", default=None, help="Path to the crash-loop alert marker file.")
    parser.add_argument("--docker-bin", default="docker", help="docker binary to invoke. Default: docker.")
    parser.add_argument(
        "--unhealthy-streak-threshold",
        type=int,
        default=DEFAULT_UNHEALTHY_STREAK_THRESHOLD,
        help=f"Consecutive unhealthy checks before alerting. Default: {DEFAULT_UNHEALTHY_STREAK_THRESHOLD}.",
    )
    parser.add_argument(
        "--restart-count-threshold",
        type=int,
        default=DEFAULT_RESTART_COUNT_THRESHOLD,
        help=f"Restarts within the window before alerting. Default: {DEFAULT_RESTART_COUNT_THRESHOLD}.",
    )
    parser.add_argument(
        "--restart-window-minutes",
        type=float,
        default=DEFAULT_RESTART_WINDOW_MINUTES,
        help=f"Window (minutes) the restart-count threshold is measured over. Default: {DEFAULT_RESTART_WINDOW_MINUTES}.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    container = args.container or container_name_for_project(args.project)
    state_file = Path(args.state_file) if args.state_file else default_state_file(args.telemetry_root, args.project)
    alert_marker = Path(args.alert_marker) if args.alert_marker else default_alert_marker(args.telemetry_root, args.project)

    try:
        state, crash_loop_detected, reasons = run(
            container=container,
            state_file=state_file,
            alert_marker=alert_marker,
            docker_bin=args.docker_bin,
            unhealthy_streak_threshold=args.unhealthy_streak_threshold,
            restart_count_threshold=args.restart_count_threshold,
            restart_window_minutes=args.restart_window_minutes,
        )
    except DockerUnavailableError as exc:
        print(f"bus_core_health_watchdog: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "container": container,
            "state_file": str(state_file),
            "alert_marker": str(alert_marker),
            "crash_loop_detected": crash_loop_detected,
            "reasons": reasons,
            **{k: v for k, v in state.items() if k != "restart_samples"},
        }))
    else:
        print(
            f"bus_core_health_watchdog: {container} health={state.get('last_health_status')} "
            f"consecutive_unhealthy={state.get('consecutive_unhealthy_count')} "
            f"restarts_in_window={state.get('restarts_in_window')}"
        )
        if crash_loop_detected:
            print(f"CRASH LOOP DETECTED -- see {alert_marker}", file=sys.stderr)
            for r in reasons:
                print(f"  - {r}", file=sys.stderr)
        else:
            print("bus_core_health_watchdog: OK -- no crash-loop signature.")

    return 1 if crash_loop_detected else 0


if __name__ == "__main__":
    raise SystemExit(main())
