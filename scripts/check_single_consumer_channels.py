#!/usr/bin/env python3
"""Deterministic gate: single-consumer channels must have exactly one subscriber.

The Orion bus is Redis pub/sub, so every subscriber executes every message.
Any channel with execute-once semantics (RPC request / work dispatch) therefore
must have exactly one live subscriber. Two subscribers means duplicated
execution -- the exact failure mode behind the duplicate-LLM-execution bug
fixed in PR #994, where two containers both subscribed to orion:verb:request.

This gate reads every channel marked `single_consumer: true` in
orion/bus/channels.yaml, asks the live bus for subscriber counts via
`redis-cli PUBSUB NUMSUB`, and fails if any such channel has more than one
subscriber. Zero subscribers is a warning by default (consumer down is a
liveness problem, not a duplication problem) unless --strict-zero is passed.

Infra problems (redis-cli missing, connection refused, no bus URL) exit 2 --
an unreachable bus is not evidence of a violation.

Usage:
    ORION_BUS_URL=redis://<tailscale-ip>:6379/0 python scripts/check_single_consumer_channels.py
    python scripts/check_single_consumer_channels.py --bus-url redis://<host>:6379/0
    python scripts/check_single_consumer_channels.py --strict-zero
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_single_consumer_channels.py` puts scripts/
# on sys.path[0], which shadows stdlib `platform` via scripts/platform/ (same
# issue documented in scripts/check_inner_state_registry.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHANNELS_FILE = REPO_ROOT / "orion" / "bus" / "channels.yaml"

_GLOB_CHARS = ("*", "?", "[")


def load_single_consumer_channels(channels_file: Path = DEFAULT_CHANNELS_FILE) -> list[str]:
    """All catalog channel names marked single_consumer: true, glob names skipped.

    Glob-shaped names (e.g. reply patterns like `orion:llm:reply*`) cannot be a
    single concrete pub/sub channel, so a subscriber count is meaningless.
    """
    entries: list[dict] | None = None
    if channels_file == DEFAULT_CHANNELS_FILE:
        # Prefer the shared catalog loader so this gate reads the catalog the
        # same way scripts/platform/audit_channels.py does.
        try:
            from scripts.platform._common import load_channels_catalog

            entries = list(load_channels_catalog(REPO_ROOT).values())
        except Exception:
            entries = None  # fall through to plain yaml
    if entries is None:
        import yaml

        data = yaml.safe_load(channels_file.read_text(encoding="utf-8")) or {}
        entries = [e for e in (data.get("channels") or []) if isinstance(e, dict)]

    names: list[str] = []
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        if entry.get("single_consumer") is not True:
            continue
        if any(ch in name for ch in _GLOB_CHARS):
            continue
        names.append(name)
    return names


def parse_numsub_output(stdout: str) -> dict[str, int]:
    """Parse `redis-cli PUBSUB NUMSUB ch1 ch2 ...` output.

    redis-cli prints alternating lines: channel name, subscriber count.
    Parses defensively -- non-integer count lines are skipped rather than
    crashing, so one weird line cannot take down the whole gate run.
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    counts: dict[str, int] = {}
    for i in range(0, len(lines) - 1, 2):
        channel, raw_count = lines[i], lines[i + 1]
        # Some redis-cli versions prefix list output with "N) ".
        if ") " in channel:
            channel = channel.split(") ", 1)[1].strip()
        if ") " in raw_count:
            raw_count = raw_count.split(") ", 1)[1].strip()
        channel = channel.strip('"')
        # "(integer) 2" form appears in non-raw interactive output.
        if raw_count.startswith("(integer)"):
            raw_count = raw_count[len("(integer)"):].strip()
        try:
            counts[channel] = int(raw_count)
        except ValueError:
            continue
    return counts


def evaluate_counts(
    counts: dict[str, int], *, strict_zero: bool = False
) -> tuple[list[str], list[str]]:
    """Pure decision core: (violations, warnings) from channel -> subscriber count.

    count > 1  -> violation (duplicate execution: the PR #994 class of bug)
    count == 0 -> warning (consumer down = liveness issue, not duplication),
                  promoted to violation under strict_zero.
    """
    violations: list[str] = []
    warnings: list[str] = []
    for channel in sorted(counts):
        count = counts[channel]
        if count > 1:
            violations.append(
                f"{channel} has {count} subscribers, expected exactly 1 "
                f"(duplicate execution risk)"
            )
        elif count == 0:
            msg = f"{channel} has 0 subscribers, expected exactly 1 (consumer down?)"
            if strict_zero:
                violations.append(msg)
            else:
                warnings.append(msg)
    return violations, warnings


def fetch_live_counts(bus_url: str, channels: list[str]) -> dict[str, int]:
    """Query live subscriber counts via redis-cli. Raises RuntimeError on infra failure."""
    cmd = ["redis-cli", "-u", bus_url, "PUBSUB", "NUMSUB", *channels]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        raise RuntimeError("redis-cli not found on PATH; install redis-tools")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"redis-cli timed out after 15s connecting to {bus_url}")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"redis-cli failed (exit {proc.returncode}): {detail}")
    # redis-cli sometimes exits 0 but prints the error to stdout/stderr.
    combined = (proc.stdout + proc.stderr).lower()
    if "could not connect" in combined or "connection refused" in combined:
        raise RuntimeError(f"redis-cli could not connect: {(proc.stderr or proc.stdout).strip()}")
    return parse_numsub_output(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bus-url",
        default=os.environ.get("ORION_BUS_URL") or None,
        help="Redis bus URL (default: ORION_BUS_URL env var)",
    )
    parser.add_argument(
        "--strict-zero",
        action="store_true",
        help="treat 0 subscribers as a violation instead of a warning",
    )
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=DEFAULT_CHANNELS_FILE,
        help="override for testing against a fixture channels.yaml",
    )
    args = parser.parse_args(argv)

    if not args.bus_url:
        print(
            "single_consumer gate: no bus URL. "
            "set ORION_BUS_URL=redis://<tailscale-ip>:6379/0 or pass --bus-url",
            file=sys.stderr,
        )
        return 2

    channels = load_single_consumer_channels(args.channels_file)
    if not channels:
        print(
            "single_consumer gate: no channels marked single_consumer: true in "
            f"{args.channels_file} -- catalog annotation missing?",
            file=sys.stderr,
        )
        return 2

    try:
        counts = fetch_live_counts(args.bus_url, channels)
    except RuntimeError as exc:
        print(f"single_consumer gate: infra error (not a violation): {exc}", file=sys.stderr)
        return 2

    # Channels redis-cli did not echo back are reported as 0 rather than dropped.
    for ch in channels:
        counts.setdefault(ch, 0)

    violations, warnings = evaluate_counts(counts, strict_zero=args.strict_zero)
    for ch in sorted(counts):
        count = counts[ch]
        if count > 1 or (count == 0 and args.strict_zero):
            status = "FAIL"
        elif count == 0:
            status = "WARN"
        else:
            status = "OK"
        print(f"{status} {count} {ch}")

    if violations:
        print(
            f"single_consumer gate FAILED: {len(violations)} violation(s), "
            f"{len(warnings)} warning(s) across {len(counts)} channel(s)",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print(
        f"single_consumer gate OK: {len(counts)} channel(s) checked, "
        f"{len(warnings)} warning(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
