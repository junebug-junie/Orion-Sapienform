#!/usr/bin/env python3
"""Schema-drift gate for orion-substrate-runtime's persisted singleton projection rows.

Context (2026-07-24 incident): PR #1331 (chore/rename-transport-pressure-stream-
backlog, merged same day) renamed `TransportBusStateV1` fields `bus_health` ->
`stream_backlog_health` and `transport_pressure` -> `stream_backlog_pressure`
(orion/schemas/transport_projection.py). The rename itself was correct and reviewed.

`services/orion-substrate-runtime` maintains exactly ONE persisted singleton row for
its transport-bus projection: table `substrate_transport_bus_projection`,
`projection_id = 'active_transport_bus_projection'` -- a materialized cache rebuilt
from live `world_pulse` Redis Stream data on every reducer tick, NOT source-of-truth
data (see app/worker.py's `_transport_tick()` and app/store.py's
`load_transport_bus_projection()`). That loader does
`TransportBusProjectionV1.model_validate(payload)` against a model with
`extra="forbid"`. The persisted row still had the OLD field names baked into its
stored JSON from before the rename. When `orion-athena-substrate-runtime` restarted
onto the new code, EVERY tick's `load_projection()` call threw
`pydantic_core.ValidationError: ... Extra inputs are not permitted
[type=extra_forbidden]` -- and since the reducer cannot complete a tick without
first successfully loading its previous state, this was a hard crash-loop: the
service ran for ~10 hours writing nothing, undetected until someone happened to
check `age_sec` on the live data by hand. Fixed live by deleting the one stale row
(safe, since it self-heals from live stream data) -- but nothing in the repo would
have caught this class of bug BEFORE a deploy crash-loops in production for hours.

This script is that gate: for every known persisted singleton/materialized
projection row in this codebase, load the CURRENT live row (if reachable) against
the CURRENT schema and fail loudly with a clear diagnostic if it does not validate.

Scope (grepped 2026-07-24 across services/*/app/store.py and orion/substrate/*):
`services/orion-substrate-runtime/app/store.py` is the ONLY place in this repo that
persists a projection as a fixed `projection_id` singleton row, UPSERTed in place on
every tick (`ON CONFLICT (projection_id) DO UPDATE ...`) -- exactly the shape that
produced the incident above. Seven such rows exist there today, all loaded via
`<Model>.model_validate(payload)` against a model with `extra="forbid"` (confirmed
directly against each model class in orion/schemas/*.py, not assumed) -- see the
PROJECTIONS tuple below for the full list of (table, projection_id, model).

Other services with a `store.py` that persists projections (orion-attention-runtime,
orion-consolidation-runtime, orion-execution-dispatch-runtime, orion-feedback-
runtime, orion-field-digester, orion-policy-runtime, orion-proposal-runtime) all use
a genuinely different shape: an append-only log table, loaded via
`ORDER BY generated_at DESC LIMIT 1` ("latest row"), not a fixed-projection_id UPSERT
target. That shape could in principle hit the same extra="forbid" + stale-field
crash on its own most-recent row, but a fresh write there immediately supersedes any
stale one -- no row is ever silently "stuck" forever the way a singleton UPSERT row
is. Genuinely different failure mode; out of scope for this patch, not built here.

`orion/substrate/felt_state_reader.py`'s `SubstrateFeltStateReader` reads five of
the same seven tables (including `substrate_transport_bus_projection` itself) via
raw SQL with no `model_validate()` call at all -- it hydrates a plain dict and is
fail-open by design (catches and swallows every exception), so it was never at
crash-loop risk from this incident and needs no gate of its own.

Usage:
    POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/check_substrate_projection_schema_drift.py
    python scripts/check_substrate_projection_schema_drift.py --postgres-uri postgresql://...
    python scripts/check_substrate_projection_schema_drift.py --json

Exit codes: 0 = every reachable row (if any) validates against the current schema,
                OR Postgres is unreachable (see "DB unavailability" note below).
            1 = at least one persisted row fails schema validation -- FAIL, this is
                the exact incident class this gate exists to catch.
            2 = the check could not run for an unrelated reason: a per-table query
                error once connected (e.g. a typo'd SQL identifier), a stored row
                whose payload could not be read/parsed (a genuinely different, and
                arguably worse, incident than the one this gate was built for --
                see `payload_error` below), a model that failed to import, or any
                other unexpected error raised after a successful connect. NOT the
                same as "no live Postgres available at all".

Note on DB unavailability, and why this diverges from
scripts/check_activation_saturation.py / scripts/check_concept_relation_digest_liveness.py
(which exit 2 when POSTGRES_URI is unset or the connection fails): those two are
standing, run-by-hand gates that are meaningless without a live Postgres, so a hard
failure is the right call there. This gate is meant to run unattended as part of a
pre-deploy/pre-PR check in environments -- a laptop, CI -- that may have no live
Postgres at all; treating that as a failure would be noise, not signal. So: no
POSTGRES_URI configured, or the `asyncpg.connect()` call itself fails for ANY reason
(refused, DNS, timeout, wrong password, TLS -- deliberately not narrowed to a
specific exception type; see `_ConnectFailure`), prints `SKIPPED: ...` and exits 0.

Everything that goes wrong AFTER a successful connect is a different story and is
never folded into that same skip path (a real bug found in review: an earlier
version of this script caught connection failures with a `try/except` wrapped
around the *entire* per-table check loop, which meant a corrupted/unparseable
stored row -- arguably a more alarming version of the exact incident this gate
exists to catch -- silently printed "SKIPPED: could not connect to Postgres" and
exited 0). Now: a per-table query error, a payload that fails to parse
(`payload_error`), a model import failure, or a missing table are all handled
per-row inside `_check_projection` and reported individually (a missing table is
the one non-error case among these: treated as an unmigrated/fresh deploy). Any
*other* exception that still escapes all of that is caught once more in `main()`,
separately from the connect-failure path, and reported as a real error (exit 2) --
never silently downgraded to "no live DB".

This is a detection gate only. It does not delete stale rows or otherwise remediate
-- the fix (row reset, or a `validation_alias` backward-compat shim at rename time)
stays a human/agent judgment call, same as how the live incident above was handled.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_substrate_projection_schema_drift.py` puts
# scripts/ on sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_activation_saturation.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class ProjectionSpec:
    table: str
    projection_id: str
    model_module: str
    model_name: str
    payload_column: str = "projection_json"

    @property
    def label(self) -> str:
        return f"{self.table}[{self.projection_id}]"


# The full, deliberately short list -- see the module docstring's "Scope" section
# for why this is the complete set of fixed-projection_id singleton rows in the
# repo today, all living in services/orion-substrate-runtime/app/store.py.
PROJECTIONS: tuple[ProjectionSpec, ...] = (
    ProjectionSpec(
        table="substrate_transport_bus_projection",
        projection_id="active_transport_bus_projection",
        model_module="orion.schemas.transport_projection",
        model_name="TransportBusProjectionV1",
    ),
    ProjectionSpec(
        table="substrate_node_biometrics_projection",
        projection_id="node_biometrics_projection",
        model_module="orion.schemas.biometrics_projection",
        model_name="NodeBiometricsProjectionV1",
    ),
    ProjectionSpec(
        table="substrate_active_node_pressure_projection",
        projection_id="active_node_pressure_projection",
        model_module="orion.schemas.biometrics_projection",
        model_name="ActiveNodePressureProjectionV1",
    ),
    ProjectionSpec(
        table="substrate_execution_trajectory_projection",
        projection_id="active_execution_trajectory",
        model_module="orion.schemas.execution_projection",
        model_name="ExecutionTrajectoryProjectionV1",
    ),
    ProjectionSpec(
        table="substrate_chat_session_projection",
        projection_id="active_chat_session",
        model_module="orion.schemas.chat_projection",
        model_name="ChatSessionProjectionV1",
    ),
    ProjectionSpec(
        table="substrate_route_arbitration_projection",
        projection_id="active_route_arbitration",
        model_module="orion.schemas.route_projection",
        model_name="RouteArbitrationProjectionV1",
    ),
    ProjectionSpec(
        table="substrate_attention_broadcast_projection",
        projection_id="substrate.attention.broadcast.v1",
        model_module="orion.schemas.attention_frame",
        model_name="AttentionBroadcastProjectionV1",
    ),
)


@dataclass
class ProjectionResult:
    spec: ProjectionSpec
    # ok | no_row | table_missing | validation_failed | query_error |
    # payload_error | model_import_error
    status: str
    detail: str = ""


# Statuses that represent a real, reportable problem with a specific row/table --
# as opposed to "connection to Postgres itself never succeeded" (see
# _ConnectFailure below), which is handled entirely separately so it can never be
# confused with one of these.
_ERROR_STATUSES = ("query_error", "payload_error", "model_import_error")


class _ConnectFailure(Exception):
    """Raised only when `asyncpg.connect()` itself fails. Deliberately kept
    separate from any exception raised while checking an individual row -- main()
    treats this class as "no live DB reachable" (skip, exit 0) and everything else
    as a real bug in the check (exit 2), so a corrupt row or an unrelated crash can
    never get silently mislabeled as "could not connect to Postgres"."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _load_model(spec: ProjectionSpec) -> type:
    module = importlib.import_module(spec.model_module)
    return getattr(module, spec.model_name)


async def _check_projection(conn: Any, spec: ProjectionSpec) -> ProjectionResult:
    import asyncpg

    query = f"""
        SELECT {spec.payload_column} FROM {spec.table}
        WHERE projection_id = $1
    """
    try:
        row = await conn.fetchrow(query, spec.projection_id)
    except asyncpg.exceptions.UndefinedTableError as exc:
        return ProjectionResult(spec, "table_missing", str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        return ProjectionResult(spec, "query_error", str(exc))

    if row is None:
        return ProjectionResult(spec, "no_row", "no persisted row (fresh deploy, or self-healed after a reset)")

    try:
        payload = row[spec.payload_column]
        if isinstance(payload, str):
            payload = json.loads(payload)
    except Exception as exc:  # noqa: BLE001 - malformed/truncated stored JSON
        return ProjectionResult(spec, "payload_error", str(exc))

    try:
        model = _load_model(spec)
    except Exception as exc:  # noqa: BLE001
        return ProjectionResult(spec, "model_import_error", str(exc))

    try:
        model.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, printed verbatim
        return ProjectionResult(spec, "validation_failed", str(exc))

    return ProjectionResult(spec, "ok", "validates against current schema")


async def _run_all(postgres_uri: str) -> list[ProjectionResult]:
    """Connects, then checks every known projection. A failure to connect raises
    `_ConnectFailure` (mapped by main() to a skip); any other exception here is a
    real bug in this check and is left to propagate as-is, so main() can tell the
    two apart -- see `_ConnectFailure`'s docstring."""
    import asyncpg

    try:
        conn = await asyncpg.connect(postgres_uri, timeout=5)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure to
        # establish the connection at all (refused, DNS, timeout, auth, TLS, ...)
        # is "no live DB reachable" for this gate's purposes.
        raise _ConnectFailure(exc) from exc
    try:
        return [await _check_projection(conn, spec) for spec in PROJECTIONS]
    finally:
        await conn.close()


def _print_result(result: ProjectionResult) -> None:
    label = result.spec.label
    if result.status == "ok":
        print(f"  OK      {label} -- {result.detail}")
    elif result.status == "no_row":
        print(f"  OK      {label} -- {result.detail}")
    elif result.status == "table_missing":
        print(f"  NOTE    {label} -- table not present yet ({result.detail})")
    elif result.status == "validation_failed":
        print(f"  FAIL    {label} -- persisted row does not validate against {result.spec.model_name}:")
        for line in result.detail.splitlines():
            print(f"            {line}")
        print(
            "            Fix: either reset this row (safe if it self-heals from "
            "live data on the next tick -- confirm before deleting) or add a "
            "validation_alias/migration for the old field name(s) before shipping "
            "the rename that caused this."
        )
    elif result.status == "query_error":
        print(f"  ERROR   {label} -- query failed: {result.detail}")
    elif result.status == "payload_error":
        print(f"  ERROR   {label} -- stored {result.spec.payload_column} could not be read/parsed: {result.detail}")
    elif result.status == "model_import_error":
        print(f"  ERROR   {label} -- could not import {result.spec.model_module}.{result.spec.model_name}: {result.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env or services/orion-substrate-runtime/.env).",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        if args.json:
            print(json.dumps({"status": "skipped", "reason": "no_postgres_uri", "checked": 0}))
        else:
            print(
                "check_substrate_projection_schema_drift: SKIPPED -- no --postgres-uri given "
                "and $POSTGRES_URI is unset. This is expected in environments with no live "
                "Postgres (a laptop without services running, some CI runners); check "
                "services/orion-substrate-runtime/.env for POSTGRES_URI if a live check was "
                "expected here."
            )
        return 0

    try:
        results = asyncio.run(_run_all(args.postgres_uri))
    except _ConnectFailure as exc:
        # Only a failure inside asyncpg.connect() itself lands here -- refused,
        # DNS, timeout, auth, TLS, whatever. Anything that goes wrong *after* a
        # successful connect (a corrupt row, a bug in this script) falls through
        # to the `except Exception` below instead, so it can never be mislabeled
        # as "no live DB" -- see _ConnectFailure's docstring.
        if args.json:
            print(json.dumps({
                "status": "skipped",
                "reason": "connection_failed",
                "detail": str(exc.cause),
                "checked": 0,
            }))
        else:
            print(
                f"check_substrate_projection_schema_drift: SKIPPED -- could not connect to "
                f"Postgres ({exc.cause}). Treating this as a no-live-DB environment, not a "
                f"failure. If Postgres should have been reachable, this needs a human look."
            )
        return 0
    except Exception as exc:  # noqa: BLE001 - a real bug in this check, not a DB-
        # reachability problem (e.g. an unhandled shape in a fetched row). Must
        # NOT be silently reported as "skipped" -- that would hide the fact that
        # something is actually broken.
        if args.json:
            print(json.dumps({
                "status": "error",
                "reason": "unexpected_error",
                "detail": str(exc),
                "checked": 0,
            }))
        else:
            print(
                f"check_substrate_projection_schema_drift: ERROR -- unexpected failure while "
                f"running the check (not a DB-reachability issue): {exc!r}",
                file=sys.stderr,
            )
        return 2

    failures = [r for r in results if r.status == "validation_failed"]
    errors = [r for r in results if r.status in _ERROR_STATUSES]

    if args.json:
        print(json.dumps({
            "status": "ran",
            "checked": len(results),
            "results": [
                {
                    "table": r.spec.table,
                    "projection_id": r.spec.projection_id,
                    "model": f"{r.spec.model_module}.{r.spec.model_name}",
                    "status": r.status,
                    "detail": r.detail,
                }
                for r in results
            ],
            "failures": len(failures),
            "errors": len(errors),
        }))
    else:
        print(
            f"check_substrate_projection_schema_drift: checked {len(results)} persisted "
            f"singleton projection row(s):"
        )
        for result in results:
            _print_result(result)
        if failures:
            print(
                f"check_substrate_projection_schema_drift FAILED: {len(failures)} of "
                f"{len(results)} persisted row(s) do not validate against the current schema. "
                f"This is the exact failure class that crash-looped orion-substrate-runtime "
                f"for ~10 hours on 2026-07-24 (PR #1331) -- do not deploy a field rename "
                f"without either resetting the stale row or shimming backward compatibility.",
                file=sys.stderr,
            )
        elif errors:
            print(
                f"check_substrate_projection_schema_drift: {len(errors)} row(s) could not be "
                f"checked due to an unrelated error (see ERROR lines above).",
                file=sys.stderr,
            )
        else:
            print("check_substrate_projection_schema_drift: OK -- no schema drift detected.")

    if failures:
        return 1
    if errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
