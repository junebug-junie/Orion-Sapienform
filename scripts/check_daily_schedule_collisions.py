#!/usr/bin/env python3
"""Report-only collision detector for `services/orion-actions`'s daily-cadence jobs.

Context: `.env_example` declares three independent daily hour/minute pairs --
ACTIONS_DAILY_PULSE_HOUR_LOCAL/MINUTE_LOCAL, ACTIONS_WORLD_PULSE_HOUR_LOCAL/MINUTE_LOCAL,
and ACTIONS_DAILY_METACOG_HOUR_LOCAL/MINUTE_LOCAL -- but there is a fourth daily job,
Daily Journal, that has no env var pair of its own. `services/orion-actions/app/main.py`
(~line 2125-2131) computes `journal_should_run` by calling `should_run_daily(...,
hour_local=settings.actions_daily_pulse_hour_local, minute_local=settings.
actions_daily_pulse_minute_local, ...)` -- i.e. Daily Journal literally reuses Daily
Pulse's schedule. That is not a coincidence and not a bug in the scheduler; it means
Daily Pulse and Daily Journal always land in the same local minute, by config. Two
independently-LLM-generated daily artifacts landing in the same notification slot reads
as duplication even though the pipelines are genuinely different. This script hardcodes
that known reuse (there is no env var to read it from) and computes pairwise
time-of-day distance across all four named cadences, flagging any pair within
--threshold-minutes of each other.

This is intentionally NOT a hard gate today. Same-slot might be an accepted trade-off
once the volume bugs elsewhere are fixed (see the design spec this script was requested
from); default behavior is report-only (always exits 0) so it can be run for visibility
without blocking anything. Pass --fail-on-collision to make it a real gate once/if
Juniper decides same-slot collisions should fail CI.

Uses the same KEY_LINE regex as scripts/sync_local_env_from_example.py to parse
.env_example (see scripts/check_service_env_compose_parity.py for the same import
convention) rather than writing a second ad hoc parser.

Usage:
    python scripts/check_daily_schedule_collisions.py
    python scripts/check_daily_schedule_collisions.py --threshold-minutes 45
    python scripts/check_daily_schedule_collisions.py --json
    python scripts/check_daily_schedule_collisions.py --fail-on-collision

Exit codes: 0 = report-only default -- always exits 0 unless --fail-on-collision is
                passed and a collision was found.
            1 = --fail-on-collision was passed and at least one pair is within
                --threshold-minutes of another.
            2 = could not run the check (missing .env_example, missing required keys).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_daily_schedule_collisions.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_activation_saturation.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
# Re-added (not at position 0) so the sibling-script import below resolves, matching
# scripts/check_service_env_compose_parity.py's convention.
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

_REPO_ROOT = Path(__file__).resolve().parents[1]

from sync_local_env_from_example import KEY_LINE as _ENV_EXAMPLE_KEY_RE  # noqa: E402

# Env var pairs for the three cadences that have their own config. Daily Journal is
# deliberately absent here -- it has no env var pair, see module docstring.
_CADENCE_ENV_KEYS: dict[str, tuple[str, str]] = {
    "Daily Pulse": ("ACTIONS_DAILY_PULSE_HOUR_LOCAL", "ACTIONS_DAILY_PULSE_MINUTE_LOCAL"),
    "World Pulse": ("ACTIONS_WORLD_PULSE_HOUR_LOCAL", "ACTIONS_WORLD_PULSE_MINUTE_LOCAL"),
    "Daily Metacog": ("ACTIONS_DAILY_METACOG_HOUR_LOCAL", "ACTIONS_DAILY_METACOG_MINUTE_LOCAL"),
}

# Daily Journal reuses Daily Pulse's hour/minute verbatim -- see
# services/orion-actions/app/main.py's journal_should_run call (~line 2125-2131), which
# passes hour_local=settings.actions_daily_pulse_hour_local and
# minute_local=settings.actions_daily_pulse_minute_local. There is no
# ACTIONS_DAILY_JOURNAL_HOUR_LOCAL / MINUTE_LOCAL env var; this is hardcoded knowledge
# of that reuse, not a separate config surface.
_JOURNAL_REUSES_CADENCE = "Daily Pulse"


def _read_env_example_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_EXAMPLE_KEY_RE.match(stripped)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _load_cadences(env_example_path: Path) -> dict[str, int]:
    """Returns {cadence_name: minute_of_day} for all four named cadences (including the
    synthetic Daily Journal entry), read from .env_example. Raises KeyError (caught by
    main, exit 2) if a required key is missing."""
    values = _read_env_example_values(env_example_path)

    cadence_minutes: dict[str, int] = {}
    for name, (hour_key, minute_key) in _CADENCE_ENV_KEYS.items():
        if hour_key not in values or minute_key not in values:
            raise KeyError(f"{hour_key}/{minute_key} not found in {env_example_path}")
        hour = int(values[hour_key])
        minute = int(values[minute_key])
        cadence_minutes[name] = hour * 60 + minute

    # Daily Journal is not read from its own keys -- it shares Daily Pulse's minute of
    # day by construction (see module docstring / _JOURNAL_REUSES_CADENCE above).
    cadence_minutes["Daily Journal"] = cadence_minutes[_JOURNAL_REUSES_CADENCE]

    return cadence_minutes


def _time_of_day_distance_minutes(a: int, b: int, minutes_per_day: int = 24 * 60) -> int:
    """Minimal distance in minutes between two minute-of-day values, generic over
    same-day wraparound (not that it matters for these daytime-ish hours today)."""
    diff = abs(a - b) % minutes_per_day
    return min(diff, minutes_per_day - diff)


def _find_collisions(cadence_minutes: dict[str, int], threshold_minutes: int) -> list[dict[str, object]]:
    names = sorted(cadence_minutes)
    collisions: list[dict[str, object]] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            distance = _time_of_day_distance_minutes(cadence_minutes[name_a], cadence_minutes[name_b])
            if distance <= threshold_minutes:
                collisions.append({
                    "a": name_a,
                    "b": name_b,
                    "distance_minutes": distance,
                })
    return collisions


def _format_time_of_day(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--env-example",
        type=Path,
        default=_REPO_ROOT / "services" / "orion-actions" / ".env_example",
        help="path to orion-actions' .env_example (default: services/orion-actions/.env_example)",
    )
    parser.add_argument(
        "--threshold-minutes",
        type=int,
        default=30,
        help="flag any pair of cadences within this many minutes of each other (default: 30).",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    parser.add_argument(
        "--fail-on-collision",
        action="store_true",
        help=(
            "exit 1 if any collision is found. Without this flag the script always "
            "exits 0 -- it is report-only by design, see module docstring."
        ),
    )
    args = parser.parse_args(argv)

    if not args.env_example.is_file():
        print(f"check_daily_schedule_collisions: missing {args.env_example}", file=sys.stderr)
        return 2

    try:
        cadence_minutes = _load_cadences(args.env_example)
    except KeyError as exc:
        print(f"check_daily_schedule_collisions: {exc}", file=sys.stderr)
        return 2

    collisions = _find_collisions(cadence_minutes, args.threshold_minutes)

    if args.json:
        print(json.dumps({
            "env_example": str(args.env_example),
            "threshold_minutes": args.threshold_minutes,
            "cadences": {name: _format_time_of_day(m) for name, m in sorted(cadence_minutes.items())},
            "collisions": collisions,
        }))
    else:
        print(
            f"check_daily_schedule_collisions: {len(cadence_minutes)} daily cadence(s) "
            f"read from {args.env_example} (threshold {args.threshold_minutes}m):"
        )
        for name, minute_of_day in sorted(cadence_minutes.items()):
            note = " (reuses Daily Pulse's schedule -- no dedicated env var)" if name == "Daily Journal" else ""
            print(f"    {name:<14} {_format_time_of_day(minute_of_day)}{note}")

        if collisions:
            print(f"COLLISION: {len(collisions)} pair(s) within {args.threshold_minutes} minutes of each other:")
            for c in collisions:
                print(f"    {c['a']} <-> {c['b']}: {c['distance_minutes']}m apart")
            if not args.fail_on_collision:
                print(
                    "check_daily_schedule_collisions: report-only by default -- not failing "
                    "the build. Pass --fail-on-collision to gate on this."
                )
        else:
            print("check_daily_schedule_collisions: OK -- no collisions within threshold.")

    if collisions and args.fail_on_collision:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
