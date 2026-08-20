#!/usr/bin/env python3
"""Live-fire round-trip drill for the `top_down_override` branch (2026-08-20
metric-quality-gate correction, part 2 of 2 -- see `measure_ast_hot_reducer.py`
for part 1, the `bottom_up_salience` correctness cross-check).

**What this answers, and why it exists.** `measure_ast_hot_reducer.py`'s
170h/98,785-tick replay found ZERO real `voluntary_override` events in
production history -- confirmed structurally rare/upstream-gated (see the
program README's item-2/item-4 entries), not a defect in this reducer. That
leaves one real, disclosed gap this program's own PR report named but never
closed: "whether the replay script's own JSON deserialization of a real
`voluntary_override` blob from `projection_json` round-trips cleanly
(datetime formatting, key casing)" was never actually checked, because no
real blob has ever existed to check it against. The existing unit test
(`TestVoluntaryOverridePresent`, `orion/substrate/tests/
test_attention_self_model.py`) only proves the PURE FUNCTION is correct
against a hand-built Python object -- it never touches Postgres, so it
cannot catch a JSONB round-trip bug (e.g. a datetime that serializes one way
and deserializes another, or a field that silently drops during
`model_dump(mode="json")` / `model_validate`).

**What this script does.** Builds one real `VoluntaryOverrideV1`-bearing
`AttentionBroadcastProjectionV1` (the exact production schema), writes it
into the REAL `substrate_attention_broadcast_log` table using the same
INSERT this repo's own production writer uses
(`services/orion-substrate-runtime/app/store.py::
save_attention_broadcast_history` -- reproduced here rather than imported,
since that service's hyphenated directory name isn't an importable Python
package from repo root), reads it back through a fresh connection (the same
`open_readonly_connection` this program's own replay script uses), runs it
through the REAL production `reduce_attention_self_model()`, and asserts the
override is narrated correctly end-to-end. Then deletes the row it inserted,
verified, in a `finally` block that runs even on failure.

**Safety.** Requires `--yes` to actually touch the database; without it,
prints what it would do and exits 3. The inserted row uses a fixed,
unmistakably-synthetic `log_id` (`SENTINEL_LOG_ID` below, not the
hash-derived format real ticks use) so it can never collide with a real row
and is trivially identifiable if ever inspected mid-run. Total live-DB
exposure window is the few hundred milliseconds between INSERT and DELETE in
this script's own single run -- not a standing fixture, not a backfill (this
program's `/tmp/<job-name>/` backfill protocol does not apply: one row,
written and deleted synchronously, not a batch write job).

Run:
    POSTGRES_URI=postgresql://... python scripts/analysis/verify_voluntary_override_pipeline.py --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
SENTINEL_LOG_ID = "synthetic-probe-voluntary-override-verify-v1"


def _build_probe_projection(ts: datetime):
    from orion.schemas.attention_frame import (
        AttentionBroadcastProjectionV1,
        AttentionFrameV1,
        CuriosityCandidateActionV1,
        OpenLoopV1,
        VoluntaryOverrideV1,
    )

    override = VoluntaryOverrideV1(
        goal_artifact_id="probe-goal-001",
        goal_drive_origin=None,
        chosen_loop_id="probe-chosen-loop",
        beat_loop_id="probe-beat-loop",
        chosen_bottom_up=0.42,
        beat_bottom_up=0.55,
        applied_bias=0.30,
        effort_spent=0.25,
    )
    winner_action = CuriosityCandidateActionV1(
        action_type="reflect", open_loop_id="probe-chosen-loop", score=0.72,
        rationale="synthetic probe winner",
    )
    frame = AttentionFrameV1(
        generated_at=ts,
        open_loops=[
            OpenLoopV1(id="probe-chosen-loop", description="synthetic probe chosen loop", salience=0.42),
            OpenLoopV1(id="probe-beat-loop", description="synthetic probe beat loop", salience=0.55),
        ],
        candidate_actions=[winner_action],
        selected_action=winner_action,
        voluntary_override=override,
    )
    return AttentionBroadcastProjectionV1(
        generated_at=ts,
        frame=frame,
        selected_action_type="reflect",
        selected_open_loop_id="probe-chosen-loop",
        selected_description="synthetic probe: verify voluntary_override JSON round-trip",
        attended_node_ids=["node:probe"],
        dwell_ticks=1,
        coalition_stability_score=0.77,
    )


def _write_probe_row(conn, projection) -> None:
    """Same INSERT statement `save_attention_broadcast_history()` uses
    (services/orion-substrate-runtime/app/store.py), with a fixed sentinel
    log_id instead of the production hash-derived one."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO substrate_attention_broadcast_log (
                log_id, generated_at, projection_json, created_at
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (log_id) DO UPDATE SET
                generated_at = EXCLUDED.generated_at,
                projection_json = EXCLUDED.projection_json,
                created_at = EXCLUDED.created_at
            """,
            (
                SENTINEL_LOG_ID,
                projection.generated_at,
                json.dumps(projection.model_dump(mode="json")),
                datetime.now(timezone.utc),
            ),
        )
    conn.commit()


def _fetch_probe_row(conn) -> tuple[datetime, dict] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT projection_json, generated_at FROM substrate_attention_broadcast_log WHERE log_id = %s",
            (SENTINEL_LOG_ID,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    payload, ts = row
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ts, payload


def _delete_probe_row(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM substrate_attention_broadcast_log WHERE log_id = %s", (SENTINEL_LOG_ID,))
        deleted = cur.rowcount
    conn.commit()
    return deleted


def run(dsn: str) -> int:
    import psycopg2

    from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
    from orion.substrate.attention_self_model import reduce_attention_self_model

    ts = datetime.now(timezone.utc)
    projection = _build_probe_projection(ts)

    conn = psycopg2.connect(dsn)
    try:
        # Clean up any leftover row from a prior crashed run before starting,
        # so a failure never accumulates residue across runs.
        _delete_probe_row(conn)

        print(f"writing synthetic probe row (log_id={SENTINEL_LOG_ID!r}) ...")
        _write_probe_row(conn, projection)

        print("reading it back (fresh SELECT, real JSONB round-trip) ...")
        fetched = _fetch_probe_row(conn)
        if fetched is None:
            print("FAIL: row not found immediately after insert+commit")
            return 1
        fetched_ts, payload = fetched

        print("deserializing into AttentionBroadcastProjectionV1 (real schema) ...")
        broadcast = AttentionBroadcastProjectionV1.model_validate(payload)

        print("running the REAL production reduce_attention_self_model() ...")
        model = reduce_attention_self_model(broadcast, field_frame=None, now=fetched_ts)

        failures = []
        if model.attention_reason != "top_down_override":
            failures.append(f"attention_reason={model.attention_reason!r}, expected 'top_down_override'")
        if model.voluntary_override is None:
            failures.append("voluntary_override is None after round-trip")
        elif model.voluntary_override.chosen_loop_id != "probe-chosen-loop":
            failures.append(
                f"voluntary_override.chosen_loop_id={model.voluntary_override.chosen_loop_id!r}, "
                "expected 'probe-chosen-loop'"
            )
        if "probe-chosen-loop" not in model.reason_narrative or "probe-beat-loop" not in model.reason_narrative:
            failures.append(f"reason_narrative missing expected loop ids: {model.reason_narrative!r}")

        if failures:
            print("FAIL: real Postgres round-trip did NOT narrate correctly:")
            for f in failures:
                print(f"  - {f}")
            return 1

        print("PASS: real INSERT -> real SELECT -> real JSON deserialize -> real reducer")
        print(f"  attention_reason = {model.attention_reason}")
        print(f"  reason_narrative = {model.reason_narrative}")
        return 0
    finally:
        # Code review, 2026-08-20: if `conn` itself broke mid-write (network
        # blip during _write_probe_row's own commit), reusing that same
        # broken connection for cleanup would fail too, silently widening
        # the exposure window this script's docstring claims is "a few
        # hundred milliseconds." Retry cleanup on a FRESH connection before
        # giving up -- the pre-run cleanup at the top of this function is
        # still the real backstop for the case where even this retry fails,
        # but this closes the common case within the same run instead of
        # deferring it to the next invocation.
        try:
            deleted = _delete_probe_row(conn)
        except Exception as exc:
            print(f"cleanup on original connection failed ({exc!r}); retrying on a fresh connection ...")
            try:
                conn.close()
            except Exception:
                pass
            import psycopg2 as _psycopg2

            retry_conn = _psycopg2.connect(dsn)
            try:
                deleted = _delete_probe_row(retry_conn)
            finally:
                retry_conn.close()
        else:
            conn.close()
        print(f"cleanup: deleted {deleted} synthetic probe row(s) (log_id={SENTINEL_LOG_ID!r})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true",
        help="actually write/read/delete one synthetic row in the real substrate_attention_broadcast_log table",
    )
    args = parser.parse_args(argv)
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    if not args.yes:
        print(
            "Dry run (default). This would INSERT one synthetic, unmistakably-tagged "
            f"row (log_id={SENTINEL_LOG_ID!r}) into the REAL substrate_attention_broadcast_log "
            "table, read it back, run it through the real production reducer, then DELETE it "
            "in a finally block -- proving the top_down_override branch survives real Postgres "
            "JSONB persistence end-to-end, not just the pure-function unit test. "
            "Pass --yes to actually run it."
        )
        return 3

    return run(dsn)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
