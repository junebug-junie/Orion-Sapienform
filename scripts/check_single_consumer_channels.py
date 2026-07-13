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
    # Plain yaml only: the shared load_channels_catalog helper falls back to a
    # regex parse when PyYAML is absent, which silently drops single_consumer
    # flags -- for a gate, a missing parser must be a loud infra error, not an
    # empty result.
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read channels.yaml") from exc

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


_REDIS_ERROR_PREFIXES = ("ERR", "NOAUTH", "WRONGPASS", "NOPERM", "LOADING", "MOVED", "CLUSTERDOWN")


def parse_numsub_output(stdout: str) -> dict[str, int]:
    """Parse `redis-cli PUBSUB NUMSUB ch1 ch2 ...` piped (non-TTY) output.

    redis-cli prints alternating lines: channel name, subscriber count.
    Unparseable pairs are skipped here; the caller decides whether missing
    channels are an infra error (they are -- NUMSUB always echoes every
    requested channel, so absence means the reply was not a NUMSUB reply).
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    counts: dict[str, int] = {}
    for i in range(0, len(lines) - 1, 2):
        channel, raw_count = lines[i].strip('"'), lines[i + 1]
        try:
            counts[channel] = int(raw_count)
        except ValueError:
            continue
    return counts


def evaluate_counts(
    counts: dict[str, int], *, strict_zero: bool = False
) -> tuple[list[str], list[str], dict[str, str]]:
    """Pure decision core: (violations, warnings, status_by_channel).

    count > 1  -> FAIL (duplicate execution: the PR #994 class of bug)
    count == 0 -> WARN (consumer down = liveness issue, not duplication),
                  promoted to FAIL under strict_zero.
    The printed per-channel status and the exit code both derive from this
    one function so they cannot drift apart.
    """
    violations: list[str] = []
    warnings: list[str] = []
    statuses: dict[str, str] = {}
    for channel in sorted(counts):
        count = counts[channel]
        if count > 1:
            violations.append(
                f"{channel} has {count} subscribers, expected exactly 1 "
                f"(duplicate execution risk)"
            )
            statuses[channel] = "FAIL"
        elif count == 0:
            msg = f"{channel} has 0 subscribers, expected exactly 1 (consumer down?)"
            if strict_zero:
                violations.append(msg)
                statuses[channel] = "FAIL"
            else:
                warnings.append(msg)
                statuses[channel] = "WARN"
        else:
            statuses[channel] = "OK"
    return violations, warnings, statuses


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
    # redis-cli exits 0 on Redis error replies (NOAUTH, ERR, LOADING, ...)
    # with the error text on stdout -- a reply that is not a NUMSUB reply.
    first_line = next((ln.strip() for ln in proc.stdout.splitlines() if ln.strip()), "")
    if any(first_line.startswith(p) for p in _REDIS_ERROR_PREFIXES):
        raise RuntimeError(f"redis error reply: {first_line}")
    combined = (proc.stdout + proc.stderr).lower()
    if "could not connect" in combined or "connection refused" in combined:
        raise RuntimeError(f"redis-cli could not connect: {(proc.stderr or proc.stdout).strip()}")

    counts = parse_numsub_output(proc.stdout)
    # NUMSUB echoes every requested channel (0 if no subscribers). A channel
    # missing from the parsed reply therefore means the output was not a
    # NUMSUB reply (auth error, banner noise, format drift) -- infra error,
    # never "0 subscribers". This is what keeps the gate fail-closed.
    missing = [ch for ch in channels if ch not in counts]
    if missing:
        raise RuntimeError(
            f"{len(missing)} requested channel(s) absent from NUMSUB reply "
            f"(unparseable redis-cli output?): {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else "")
        )
    return counts


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

    try:
        channels = load_single_consumer_channels(args.channels_file)
    except RuntimeError as exc:
        print(f"single_consumer gate: infra error (not a violation): {exc}", file=sys.stderr)
        return 2
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

    violations, warnings, statuses = evaluate_counts(counts, strict_zero=args.strict_zero)
    for ch in sorted(counts):
        print(f"{statuses[ch]} {counts[ch]} {ch}")

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
