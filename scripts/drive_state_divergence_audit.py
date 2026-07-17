#!/usr/bin/env python3
"""Point-in-time divergence snapshot between the two independently-computed
drive-pressure signals that share the 6-key `DRIVE_KEYS` taxonomy
(coherence, continuity, capability, relational, predictive, autonomy):

1. `drive_state.v1` -- computed by `DriveEngine`
   (orion/spark/concept_induction/drives.py), run inside
   orion/spark/concept_induction/bus_worker.py. Only the *current* value is
   persisted: a local JSON file via
   `LocalProfileStore.load_drive_state("orion")`
   (orion/spark/concept_induction/store.py), path from
   `ConceptSettings.store_path` (env `CONCEPT_STORE_PATH`, default
   `DEFAULT_CONCEPT_STORE_PATH` in orion/spark/concept_induction/settings.py).
2. `autonomy_state_v2` -- historically computed by `orion.autonomy.reducer`,
   which was retired 2026-07-16 and deleted 2026-07-17 (see
   orion/self_state/inner_state_registry.py); this script now reads a
   frozen/historical persisted row rather than a live-updating signal. Only
   the *current* value is persisted: a single-row-per-subject UPSERT in
   Postgres via `load_autonomy_state_v2("orion")`
   (orion/autonomy/state_store.py), DSN from env `ORION_AUTONOMY_STATE_DB_URL`.
   That loader fails open -- it never raises, and returns None on any error,
   missing DSN, or missing row.

`orion/self_state/inner_state_registry.py` marks both signals `DUPLICATE` of
each other and explicitly defers the merge-or-keep-separate decision to a
later phase -- that decision is NOT made here. This script only measures and
reports how far apart the two CURRENT values are; it never merges, blends,
or picks a winner between them.

Neither backing store keeps history -- the JSON file is overwritten in place
and the Postgres row is UPSERTed -- so this is not a time-window trend
report. It is a snapshot comparison, safe to re-run any time.

Usage:
    python scripts/drive_state_divergence_audit.py
    python scripts/drive_state_divergence_audit.py --subject orion --json

If neither --store-path nor $CONCEPT_STORE_PATH is given, this falls back to a
host-local dev default that is almost never the live Docker container's real
store, and prints a loud warning saying so (2026-07-13 incident: a stale file
sitting at that fallback path was misread as a 24h+-stale live drive_state.v1
-- the live container was fine and ticking every few seconds). Point this at
the real store instead -- find it with:
    docker inspect <concept-induction-container> \\
        --format '{{json .Config.Env}}' | grep CONCEPT_STORE_PATH
then cross-reference services/orion-spark-concept-induction/docker-compose.yml's
volumes: section for the host-side bind mount, e.g.:
    CONCEPT_STORE_PATH=/mnt/graphdb/orion/concepts/concept-induction-state.json \\
    ORION_AUTONOMY_STATE_DB_URL=postgresql://user:pass@host:port/db \\
        python scripts/drive_state_divergence_audit.py

This is a report-only diagnostic, not a pass/fail gate (mirrors
scripts/check_activation_saturation.py's report-only-by-default posture --
there is no known-good baseline for divergence between these two signals).
It always exits 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/drive_state_divergence_audit.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_single_consumer_channels.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orion.autonomy.models import AutonomyStateV2  # noqa: E402
from orion.autonomy.state_store import load_autonomy_state_v2  # noqa: E402
from orion.spark.concept_induction.drives import DRIVE_KEYS  # noqa: E402
from orion.spark.concept_induction.settings import DEFAULT_CONCEPT_STORE_PATH  # noqa: E402
from orion.spark.concept_induction.store import LocalProfileStore  # noqa: E402


def load_drive_state_v1(store_path: str, subject: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load the current drive_state.v1 raw dict for `subject`.

    Returns (state_or_None, error_or_None). `LocalProfileStore.load_drive_state`
    already never raises and returns `{}` for a cold/missing store or unknown
    subject (see store.py) -- this wraps it defensively anyway (e.g. an
    unwritable parent dir in `LocalProfileStore.__init__`'s mkdir) so a bad
    store path is reported, not a crash. A `None` return means "no data
    available", not "zero divergence" -- callers must not treat it as pressures
    of 0.0.
    """
    try:
        store = LocalProfileStore(store_path)
        raw = store.load_drive_state(subject)
    except Exception as exc:  # pragma: no cover - defensive, store.py itself never raises
        return None, f"LocalProfileStore.load_drive_state raised: {exc}"
    if not isinstance(raw, dict) or "pressures" not in raw:
        return None, None
    return raw, None


def load_autonomy_state(subject: str) -> Tuple[Optional[AutonomyStateV2], Optional[str]]:
    """Load the current autonomy_state_v2 for `subject`.

    `load_autonomy_state_v2` is documented to fail open (never raises, returns
    None on any error/missing DSN/missing row) -- this wraps it anyway per the
    same defensive posture as `load_drive_state_v1`.
    """
    try:
        state = load_autonomy_state_v2(subject)
    except Exception as exc:  # pragma: no cover - defensive, state_store.py itself never raises
        return None, f"load_autonomy_state_v2 raised: {exc}"
    return state, None


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort numeric coercion for values read from the unvalidated
    LocalProfileStore JSON file (no pydantic model backs it, unlike
    AutonomyStateV2.drive_pressures). A hand-edited or corrupted store could
    hold a non-numeric pressure value -- treat that as "not available" for
    this key rather than raising and crashing the whole report."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_drives(
    drive_raw: Optional[Dict[str, Any]],
    autonomy_state: Optional[AutonomyStateV2],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Per-drive pressure divergence + activation-flag agreement.

    `drive_state.v1` carries `activations: Dict[str, bool]` (one flag per
    drive). `AutonomyStateV2` (via its `AutonomyStateV1` base) has no such
    per-drive bool map -- it has `active_drives: list[str]`, so the
    activation-equivalent comparison here is membership in that list, not a
    dict lookup. Never fabricates a value for a side with no data: any field
    that can't be computed is `None`, not 0.0/False.
    """
    pressures_a = (drive_raw or {}).get("pressures") or {}
    activations_a = (drive_raw or {}).get("activations") or {}
    pressures_b = autonomy_state.drive_pressures if autonomy_state is not None else {}
    active_b = set(autonomy_state.active_drives) if autonomy_state is not None else set()

    per_drive: Dict[str, Dict[str, Any]] = {}
    diffs: Dict[str, float] = {}
    activation_agreements = 0
    activation_disagreements = 0

    for key in DRIVE_KEYS:
        # pb is safe to treat as already-numeric: AutonomyStateV2.drive_pressures
        # is pydantic-typed dict[str, float], so a malformed row would have
        # failed validation upstream in load_autonomy_state_v2 (which fails
        # open to None) rather than reaching us. pa has no such guarantee.
        pa = _coerce_float(pressures_a.get(key)) if drive_raw is not None else None
        pb = pressures_b.get(key) if autonomy_state is not None else None
        abs_diff = abs(pa - pb) if (pa is not None and pb is not None) else None

        aa = activations_a.get(key) if drive_raw is not None else None
        ab = (key in active_b) if autonomy_state is not None else None
        agree = (bool(aa) == bool(ab)) if (aa is not None and ab is not None) else None
        if agree is True:
            activation_agreements += 1
        elif agree is False:
            activation_disagreements += 1

        per_drive[key] = {
            "pressure_drive_state_v1": pa,
            "pressure_autonomy_state_v2": pb,
            "abs_diff": abs_diff,
            "activation_drive_state_v1": aa,
            "active_autonomy_state_v2": ab,
            "activation_agree": agree,
        }
        if abs_diff is not None:
            diffs[key] = abs_diff

    max_diff_drive = max(diffs, key=diffs.get) if diffs else None
    summary = {
        "drives_compared_pressure": len(diffs),
        "drives_total": len(DRIVE_KEYS),
        "mean_abs_diff": (sum(diffs.values()) / len(diffs)) if diffs else None,
        "max_abs_diff": diffs[max_diff_drive] if max_diff_drive else None,
        "max_abs_diff_drive": max_diff_drive,
        "activation_drives_compared": activation_agreements + activation_disagreements,
        "activation_agreements": activation_agreements,
        "activation_disagreements": activation_disagreements,
    }
    return per_drive, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subject",
        default="orion",
        help="subject key both signals are keyed on (default: orion, the repo-wide convention).",
    )
    parser.add_argument(
        "--store-path",
        default=None,
        help=(
            "path to the concept-induction LocalProfileStore JSON file. "
            "Defaults to $CONCEPT_STORE_PATH, falling back to "
            f"{DEFAULT_CONCEPT_STORE_PATH!r} (a host-local dev default that is "
            "almost never the live Docker container's real store -- see the "
            "printed warning when this fallback is used)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    # Track *why* store_path has its value, distinct from the value itself, so
    # we can warn when nobody actually pointed this at a real store (2026-07-13
    # incident: an operator ran this with neither --store-path nor
    # $CONCEPT_STORE_PATH set, silently landed on DEFAULT_CONCEPT_STORE_PATH, and
    # that path happened to hold an unrelated stale dev-leftover file -- read as
    # "drive_state.v1 is 24h+ stale" when the live container's actual store,
    # a bind-mounted host path configured in
    # services/orion-spark-concept-induction/docker-compose.yml /
    # .env's CONCEPT_STORE_PATH, was updating every few seconds).
    env_store_path = os.getenv("CONCEPT_STORE_PATH")
    store_path_warning: Optional[str] = None
    if args.store_path is not None:
        store_path_source = "cli:--store-path"
    elif env_store_path:
        store_path_source = "env:CONCEPT_STORE_PATH"
        args.store_path = env_store_path
    else:
        store_path_source = "default_fallback"
        args.store_path = DEFAULT_CONCEPT_STORE_PATH
        store_path_warning = (
            f"neither --store-path nor $CONCEPT_STORE_PATH was set -- falling back to "
            f"{DEFAULT_CONCEPT_STORE_PATH!r}, a host-local dev default. This is almost "
            "certainly NOT the path the live Docker container writes to. The live "
            "container's real CONCEPT_STORE_PATH is set in "
            "services/orion-spark-concept-induction/docker-compose.yml + .env, typically "
            "a bind-mounted host path under $ORION_DATA_ROOT (e.g. "
            "/mnt/graphdb/orion/concepts/concept-induction-state.json). Verify with: "
            "docker inspect <container> --format '{{json .Config.Env}}' | grep CONCEPT_STORE_PATH "
            "-- then cross-reference the compose file's volumes: section for the host-side "
            "mount. A stale file left over from an old local run can sit at the fallback "
            "path and look like real staleness."
        )

    drive_raw, drive_error = load_drive_state_v1(args.store_path, args.subject)
    autonomy_state, autonomy_error = load_autonomy_state(args.subject)

    drive_available = drive_raw is not None
    autonomy_available = autonomy_state is not None

    per_drive, summary = compare_drives(drive_raw, autonomy_state)

    if args.json:
        payload = {
            "subject": args.subject,
            "drive_state_v1": {
                "available": drive_available,
                "store_path": args.store_path,
                "store_path_source": store_path_source,
                "store_path_warning": store_path_warning,
                "error": drive_error,
                "pressures": (drive_raw or {}).get("pressures") if drive_available else None,
                "activations": (drive_raw or {}).get("activations") if drive_available else None,
                "updated_at": (drive_raw or {}).get("updated_at") if drive_available else None,
            },
            "autonomy_state_v2": {
                "available": autonomy_available,
                "dsn_configured": bool(os.getenv("ORION_AUTONOMY_STATE_DB_URL", "").strip()),
                "error": autonomy_error,
                "drive_pressures": autonomy_state.drive_pressures if autonomy_available else None,
                "active_drives": autonomy_state.active_drives if autonomy_available else None,
                "generated_at": (
                    autonomy_state.generated_at.isoformat()
                    if autonomy_available and autonomy_state.generated_at
                    else None
                ),
            },
            "per_drive": per_drive,
            "summary": summary,
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"drive_state_divergence_audit: subject={args.subject!r}")
    print()
    if store_path_warning:
        print(f"WARNING: {store_path_warning}")
        print()
    if drive_available:
        updated_at = (drive_raw or {}).get("updated_at")
        print(f"drive_state.v1: available (store_path={args.store_path}, updated_at={updated_at})")
    elif drive_error:
        print(f"drive_state.v1: UNAVAILABLE -- {drive_error}")
    else:
        print(
            f"drive_state.v1: UNAVAILABLE -- no data at store_path={args.store_path} "
            f"(cold store, wrong path, or subject {args.subject!r} never ticked)"
        )

    if autonomy_available:
        print(f"autonomy_state_v2: available (generated_at={autonomy_state.generated_at})")
    elif autonomy_error:
        print(f"autonomy_state_v2: UNAVAILABLE -- {autonomy_error}")
    elif not os.getenv("ORION_AUTONOMY_STATE_DB_URL", "").strip():
        print("autonomy_state_v2: UNAVAILABLE -- $ORION_AUTONOMY_STATE_DB_URL is unset")
    else:
        print(
            "autonomy_state_v2: UNAVAILABLE -- load_autonomy_state_v2 returned None "
            f"(no persisted row yet for subject={args.subject!r}, or a DB error was logged and swallowed)"
        )
    print()

    if not drive_available and not autonomy_available:
        print("Cannot compute divergence: both signals are unavailable.")
        return 0
    if not drive_available or not autonomy_available:
        missing = "drive_state.v1" if not drive_available else "autonomy_state_v2"
        print(f"Cannot compute divergence: {missing} has no data. Per-drive comparison below is all n/a.")
        print()

    header = f"{'drive':<12} {'drive_state.v1':>16} {'autonomy_v2':>14} {'abs_diff':>10} {'active_a':>9} {'active_b':>9} {'agree':>7}"
    print(header)
    print("-" * len(header))
    for key in DRIVE_KEYS:
        entry = per_drive[key]

        def _fmt(v: Any, precision: int = 3) -> str:
            if v is None:
                return "n/a"
            if isinstance(v, float):
                return f"{v:.{precision}f}"
            return str(v)

        print(
            f"{key:<12} {_fmt(entry['pressure_drive_state_v1']):>16} "
            f"{_fmt(entry['pressure_autonomy_state_v2']):>14} {_fmt(entry['abs_diff']):>10} "
            f"{_fmt(entry['activation_drive_state_v1']):>9} {_fmt(entry['active_autonomy_state_v2']):>9} "
            f"{_fmt(entry['activation_agree']):>7}"
        )

    print()
    if summary["drives_compared_pressure"]:
        print(
            f"summary: {summary['drives_compared_pressure']}/{summary['drives_total']} drives compared, "
            f"mean_abs_diff={summary['mean_abs_diff']:.3f}, "
            f"max_abs_diff={summary['max_abs_diff']:.3f} (drive={summary['max_abs_diff_drive']})"
        )
    else:
        print("summary: no drive had pressures available on both sides -- 0 drives compared.")
    if summary["activation_drives_compared"]:
        print(
            f"activation flags: {summary['activation_agreements']}/{summary['activation_drives_compared']} agree, "
            f"{summary['activation_disagreements']}/{summary['activation_drives_compared']} disagree"
        )
    else:
        print("activation flags: no drive had an activation flag available on both sides -- 0 compared.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
