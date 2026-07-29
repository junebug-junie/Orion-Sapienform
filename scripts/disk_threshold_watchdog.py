#!/usr/bin/env python3
"""Host-level disk-usage threshold watchdog for the mount points that back
this repo's Docker images, source checkout, databases, graph store, and
warm/lukewarm bulk storage tiers.

Context: nothing in this repo checks disk usage on `/`, `/mnt/docker`,
`/mnt/scripts/`, `/mnt/telemetry`, `/mnt/postgres`, `/mnt/graphdb`,
`/mnt/storage-warm`, or `/mnt/storage-lukewarm` and surfaces a breach
anywhere an operator would actually see it. Each mount is a distinct
physical filesystem on this host (confirmed via `df -h`: root ->
/dev/mapper/ubuntu--vg-ubuntu--lv, `/dev/sda` -> /mnt/docker, `/dev/sde1`
-> /mnt/scripts, `/dev/sdf1` -> /mnt/telemetry, `/dev/sdg1` ->
/mnt/postgres, `/dev/sdg2` -> /mnt/graphdb, `/dev/sdc` ->
/mnt/storage-warm, `/dev/sdh` -> /mnt/storage-lukewarm), so usage on one
says nothing about the others -- all eight are checked independently.

Same category as `scripts/bus_core_health_watchdog.py` (standalone,
host-level, cron-run, not a live service loop; pure `evaluate_path()` for
the threshold/debounce logic; flock-guarded atomic state). The one
deliberate difference: that script avoided `orion-notify`'s
`/attention/request` because a bus-core crash loop can take Postgres/the
bus down in the same incident notify might depend on. A slowly filling
disk is not that kind of bootstrapping failure -- `orion-notify` reusing
the same alerting path already proven live by `orion-mesh-guardian` and
the Fuseki recover job is the right call here, and it's what actually
lands this as a Hub Pending Attention card (the point of this script).

Per monitored path, on each run:

- `shutil.disk_usage(path)` gives (total, used, free); percent_used =
  used / total * 100.
- If percent_used >= `--threshold-pct` (default 90.0) and the path was NOT
  already flagged as breached on the previous run, this is a new breach:
  one `NotifyClient.attention_request()` call fires (severity "warning").
  While a path stays breached across runs AND the notify call already
  succeeded, no repeat notification fires (debounced via persisted state)
  -- Hub Pending Attention cards are ack'd by a human, not auto-resolved
  by this script, so repeat spam while waiting for an ack would be
  actively unhelpful. Debounce state is only persisted as "notified" once
  `NotifyClient.attention_request()` actually confirms success
  (`NotificationAccepted.ok`) -- a failed attempt (orion-notify down,
  unreachable, or erroring; note the real client never raises for this,
  it returns `ok=False`) retries on every subsequent tick until it
  succeeds, rather than being silently swallowed the moment `last_status`
  flips to "breached". See evaluate_path()'s docstring for the full
  rationale (found by code review, live-reproduced before the fix).
- If `shutil.disk_usage(path)` itself raises (path missing, permission
  denied, mount vanished) and this is a NEW error for that path, that is
  also worth attention (a vanished mount can BE the disk problem) --
  fires a separate "error" attention_request, same debounce/retry rule.
- Recovery (breached -> under threshold) clears the debounce state
  silently, no notification -- once a card is ack'd, there's nothing to
  auto-resolve; a human already saw it.

State (per-path: last status, first-breach timestamp, last percent_used,
whether the current status was successfully notified) persists to
`--state-file` (default `${TELEMETRY_ROOT}/${PROJECT}/disk-watchdog/
state.json`).

Concurrency: same non-blocking `flock` pattern as
`bus_core_health_watchdog.py` guards the read-evaluate-write cycle against
overlapping cron invocations. A run that can't acquire the lock skips
cleanly (exit 0).

Usage:
    python scripts/disk_threshold_watchdog.py
    python scripts/disk_threshold_watchdog.py --threshold-pct 85 --json
    python scripts/disk_threshold_watchdog.py --paths /mnt/docker,/mnt/scripts,/mnt/telemetry

Exit codes: 0 = all monitored paths under threshold (also returned if this
                run was skipped because another run already holds the lock).
            1 = at least one monitored path is at/over threshold, or could
                not be statted, at check time (regardless of whether a new
                notification fired this tick -- mirrors
                bus_core_health_watchdog.py's exit-code contract so a
                monitoring wrapper keying off exit code behaves the same
                way across both scripts).
            2 = could not complete the check for a reason outside any
                single monitored path (a filesystem error writing state --
                e.g. the telemetry tree not yet created/chowned on a fresh
                host). Deliberately distinct from exit 1 so a permissions
                failure can never be misread as "disk full."
            3 = the watchdog itself broke on an unexpected/unhandled
                exception (a bug in this script). Deliberately distinct
                from both exit 1 and exit 2, same convention as
                bus_core_health_watchdog.py.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/disk_threshold_watchdog.py` puts scripts/ on
# sys.path[0], which can shadow stdlib modules (same issue documented in
# scripts/bus_core_health_watchdog.py / scripts/check_inner_state_registry.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

from orion.notify.client import NotifyClient  # noqa: E402

DEFAULT_PATHS = (
    "/",
    "/mnt/docker",
    "/mnt/scripts",
    "/mnt/telemetry",
    "/mnt/postgres",
    "/mnt/graphdb",
    "/mnt/storage-warm",
    "/mnt/storage-lukewarm",
)
DEFAULT_THRESHOLD_PCT = 90.0


def default_state_file(telemetry_root: str, project: str) -> Path:
    return Path(telemetry_root) / project / "disk-watchdog" / "state.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """mkstemp in the same directory + os.replace -- atomic on the same
    filesystem, no torn writes if this run is interrupted mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"paths": {}}
    try:
        with path.open("r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not isinstance(data.get("paths"), dict):
            raise ValueError("state file did not contain the expected {'paths': {...}} shape")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(
            f"disk_threshold_watchdog: WARNING -- state file {path} unreadable/corrupt "
            f"({exc}), starting from a fresh state.",
            file=sys.stderr,
        )
        return {"paths": {}}


def measure_path(path: str) -> tuple[float | None, str | None]:
    """Returns (percent_used, error). Exactly one is non-None.

    Never raises -- a missing mount or permission error is itself a signal
    this script reports on, not a tooling failure that should crash the run.
    """
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return None, str(exc)
    if usage.total <= 0:
        return None, f"disk_usage reported non-positive total ({usage.total}) for {path}"
    return (usage.used / usage.total) * 100.0, None


def evaluate_path(
    state_for_path: dict[str, Any],
    percent_used: float | None,
    error: str | None,
    threshold_pct: float,
    now: datetime,
) -> tuple[dict[str, Any], str, bool]:
    """Pure function: given one path's persisted state plus one fresh
    (percent_used, error) observation, returns
    (updated_state_for_path, status, should_notify).

    status is one of "ok" / "breached" / "error". should_notify is True on a
    transition INTO "breached" or INTO "error", AND on every subsequent tick
    where the status is still bad but `state_for_path["notified"]` is not
    True -- i.e. the caller (run()) never confirmed a prior notify attempt
    actually succeeded. This is deliberate: `run()` only persists
    `notified=True` after `NotifyClient.attention_request()` reports
    success. A notify call that raised, or came back with `ok=False` (the
    real `NotifyClient` never raises on network failure -- it always
    returns `NotificationAccepted(ok=False, ...)`), must not be silently
    treated as "handled." Without this, a breach that first occurs while
    orion-notify is down/unreachable would be debounced into permanent
    silence the moment `last_status` flips to "breached", even though no
    Pending Attention card ever actually landed. Caught live during code
    review: a mocked notify failure showed the state file committing
    last_status="breached" on tick 1 with zero retry attempted on tick 2.
    """
    now_iso = now.isoformat()
    prev_status = state_for_path.get("last_status", "ok")
    prev_notified = bool(state_for_path.get("notified", False))
    new_state = dict(state_for_path)

    if error is not None:
        status = "error"
        new_state["last_error"] = error
        new_state["last_percent_used"] = None
    else:
        assert percent_used is not None
        status = "breached" if percent_used >= threshold_pct else "ok"
        new_state["last_error"] = None
        new_state["last_percent_used"] = percent_used

    is_status_transition = status in ("breached", "error") and status != prev_status
    should_notify = status in ("breached", "error") and (is_status_transition or not prev_notified)

    if is_status_transition:
        new_state["first_detected_at"] = now_iso
        new_state["notified"] = False
    elif status == "ok":
        new_state["first_detected_at"] = None
        new_state["notified"] = False
    else:
        # Status unchanged and still bad -- carry the previous notified flag
        # forward; run() overwrites it with the real outcome of this tick's
        # notify attempt if should_notify is True.
        new_state["notified"] = prev_notified

    new_state["last_status"] = status
    new_state["last_check_at"] = now_iso
    return new_state, status, should_notify


class WatchdogLockedError(RuntimeError):
    """Another watchdog run already holds the state-file lock -- this run
    skips cleanly rather than racing it for the write."""


class _StateLock:
    """Non-blocking flock on `<state_file>.lock`, same pattern as
    bus_core_health_watchdog.py's _StateLock -- guards against overlapping
    cron invocations corrupting the read-evaluate-write cycle."""

    def __init__(self, state_file: Path) -> None:
        self._lock_path = state_file.with_suffix(state_file.suffix + ".lock")
        self._fh = None

    def __enter__(self) -> "_StateLock":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._lock_path, "w")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise WatchdogLockedError(
                f"lock {self._lock_path} already held -- another watchdog run is in progress, skipping"
            ) from exc
        self._fh = fh
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()


def _publish_attention(
    notify: NotifyClient,
    *,
    path: str,
    status: str,
    percent_used: float | None,
    error: str | None,
    threshold_pct: float,
) -> bool:
    """Returns True only if orion-notify actually confirmed the attention
    request (`NotificationAccepted.ok`). The real `NotifyClient.
    attention_request()` does not raise on network failure -- it catches
    everything internally and returns `NotificationAccepted(ok=False, ...)`
    -- so both that case AND a genuine exception (e.g. a pydantic
    construction error) must return False here. run() uses this return
    value, not "did the call happen," to decide whether the debounce state
    may be marked notified -- see evaluate_path()'s docstring.
    """
    if status == "breached":
        message = (
            f"Disk usage on {path} is at {percent_used:.1f}% "
            f"(threshold {threshold_pct:.1f}%)."
        )
        reason = f"[Orion disk watchdog] {path} over threshold"
    else:
        message = f"Could not check disk usage on {path}: {error}"
        reason = f"[Orion disk watchdog] {path} check failed"

    context = {
        "source_service": "disk-threshold-watchdog",
        "event_kind": "orion.disk.threshold.attention.v1",
        "path": path,
        "status": status,
        "percent_used": percent_used,
        "threshold_pct": threshold_pct,
        "error": error,
        "reason": reason,
    }
    try:
        result = notify.attention_request(
            message=message,
            severity="warning" if status == "breached" else "error",
            require_ack=True,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 -- notify failure must not crash the watchdog
        print(
            f"disk_threshold_watchdog: WARNING -- failed to publish attention request for "
            f"{path}: {exc}. Will retry next tick.",
            file=sys.stderr,
        )
        return False

    ok = bool(getattr(result, "ok", False))
    if not ok:
        detail = getattr(result, "detail", None)
        print(
            f"disk_threshold_watchdog: WARNING -- orion-notify did not confirm the attention "
            f"request for {path} (ok=False, detail={detail!r}). Will retry next tick.",
            file=sys.stderr,
        )
    return ok


def run(
    paths: list[str],
    threshold_pct: float,
    state_file: Path,
    notify: NotifyClient,
    now: datetime | None = None,
) -> tuple[dict[str, Any], bool]:
    now = now or datetime.now(timezone.utc)

    with _StateLock(state_file):
        state = load_state(state_file)
        any_bad = False
        for path in paths:
            percent_used, error = measure_path(path)
            path_state = state["paths"].get(path, {})
            new_path_state, status, should_notify = evaluate_path(
                path_state, percent_used, error, threshold_pct, now
            )
            if status != "ok":
                any_bad = True
            if should_notify:
                new_path_state["notified"] = _publish_attention(
                    notify,
                    path=path,
                    status=status,
                    percent_used=percent_used,
                    error=error,
                    threshold_pct=threshold_pct,
                )
            state["paths"][path] = new_path_state
        _atomic_write_json(state_file, state)

    return state, any_bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--paths",
        default=os.getenv("DISK_WATCHDOG_PATHS", ",".join(DEFAULT_PATHS)),
        help=f"Comma-separated mount paths to check. Default: {','.join(DEFAULT_PATHS)}.",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=float(os.getenv("DISK_WATCHDOG_THRESHOLD_PCT", DEFAULT_THRESHOLD_PCT)),
        help=f"Percent-used threshold that triggers a breach. Default: {DEFAULT_THRESHOLD_PCT}.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("PROJECT", "orion-athena"),
        help="Compose project name, matches $PROJECT (see .env). Used to derive the default "
        "state-file path. Default: $PROJECT or 'orion-athena'.",
    )
    parser.add_argument(
        "--telemetry-root",
        default=os.getenv("TELEMETRY_ROOT", "/mnt/telemetry"),
        help="Root of the telemetry tree, matches $TELEMETRY_ROOT. Default: $TELEMETRY_ROOT or /mnt/telemetry.",
    )
    parser.add_argument("--state-file", default=None, help="Path to the watchdog's own state JSON file.")
    parser.add_argument(
        "--notify-base-url",
        default=os.getenv("NOTIFY_BASE_URL", "http://localhost:7140"),
        help="orion-notify base URL, reachable from the host (not the Docker-internal "
        "hostname). Default: $NOTIFY_BASE_URL or http://localhost:7140.",
    )
    parser.add_argument(
        "--notify-api-token",
        default=os.getenv("NOTIFY_API_TOKEN"),
        help="orion-notify API token, if configured. Default: $NOTIFY_API_TOKEN.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    state_file = Path(args.state_file) if args.state_file else default_state_file(args.telemetry_root, args.project)
    notify = NotifyClient(base_url=args.notify_base_url, api_token=args.notify_api_token, timeout=10)

    try:
        state, any_bad = run(paths, args.threshold_pct, state_file, notify)
    except WatchdogLockedError as exc:
        print(f"disk_threshold_watchdog: SKIPPED -- {exc}", file=sys.stderr)
        return 0
    except OSError as exc:
        print(
            f"disk_threshold_watchdog: FAILED -- could not write state file: {exc}. "
            "Check directory ownership/permissions.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 -- deliberate catch-all, see docstring's Exit codes
        print(
            f"disk_threshold_watchdog: UNEXPECTED ERROR -- {type(exc).__name__}: {exc}. "
            "This is a bug in the watchdog itself, not a real disk signal.",
            file=sys.stderr,
        )
        return 3

    if args.json:
        print(json.dumps({"state_file": str(state_file), "threshold_pct": args.threshold_pct, **state}))
    else:
        for path in paths:
            p_state = state["paths"].get(path, {})
            status = p_state.get("last_status", "unknown")
            pct = p_state.get("last_percent_used")
            pct_str = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "n/a"
            print(f"disk_threshold_watchdog: {path} status={status} used={pct_str}")
        print("disk_threshold_watchdog: OK -- all paths under threshold." if not any_bad else "disk_threshold_watchdog: ATTENTION -- see above.", file=sys.stderr if any_bad else sys.stdout)

    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
