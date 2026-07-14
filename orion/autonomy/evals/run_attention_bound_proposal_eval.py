"""Kill-criterion eval: attention-bound proposal target diversity (P5).

`inspect_attended_target` (config/proposals/proposal_policy.v1.yaml) resolves
its target_id/target_kind from self_state.dominant_attention_target_details[0]
instead of a hardcoded literal. The original brainstorm's own falsifiable
design says this binding is only worth keeping if it actually tracks a moving
target: over a real 7-day window, distinct(target_id) across this template's
candidates should be >= 3. If it stays pinned to one or two targets forever,
the binding isn't doing anything the literal templates don't already do.

This script queries substrate_proposal_frames (real Postgres, the same table
orion-proposal-runtime writes via ProposalRuntimeStore.save_proposal_frame)
for candidates whose proposal_id starts with "proposal:inspect_attended_target:"
within the window, and reports PASS/FAIL/insufficient-data.

Not run automatically in this patch and not wired into any test suite or CI --
needs real live traffic to accumulate over days. Ships now so it is runnable
once that traffic exists.

Run: python orion/autonomy/evals/run_attention_bound_proposal_eval.py
Env: POSTGRES_URI (falls back to the standard local dev DSN below).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("orion.autonomy.evals.run_attention_bound_proposal_eval")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

TEMPLATE_KEY = "inspect_attended_target"
PROPOSAL_ID_PREFIX = f"proposal:{TEMPLATE_KEY}:"
MIN_DISTINCT_TARGETS = 3
WINDOW_DAYS = 7


def open_readonly_connection(dsn: str):
    """Open a psycopg2 connection and force a read-only session.

    Returns None on any connection failure or if psycopg2 is unavailable --
    this eval must degrade gracefully, never crash the whole run.
    """
    try:
        import psycopg2  # lazy import so the module imports cleanly without the driver
    except Exception:
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None

    try:
        conn = psycopg2.connect(dsn)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None

    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            cur.execute("SHOW default_transaction_read_only;")
            value = cur.fetchone()
        if not value or str(value[0]).lower() != "on":
            logger.error("failed to force read-only session; refusing connection")
            conn.close()
            return None
    except Exception:
        logger.error("failed to set read-only session", exc_info=True)
        try:
            conn.close()
        except Exception:
            pass
        return None

    return conn


def fetch_attended_target_ids(conn, window_start: datetime) -> list[str]:
    """Return the target_id of every inspect_attended_target candidate whose
    parent proposal_frame was generated at/after window_start.

    Candidates live inside proposal_frame_json.candidates (a JSON array); this
    template's proposal_id is stable per self-state
    ("proposal:inspect_attended_target:{self_state_id}"), so we filter in
    Python after pulling the JSON rather than assuming a JSONB index exists.
    """
    target_ids: list[str] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT proposal_frame_json
            FROM substrate_proposal_frames
            WHERE generated_at >= %s
            ORDER BY generated_at DESC
            """,
            (window_start,),
        )
        rows = cur.fetchall()
    for (payload,) in rows:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        for candidate in payload.get("candidates", []) or []:
            proposal_id = candidate.get("proposal_id", "")
            if isinstance(proposal_id, str) and proposal_id.startswith(PROPOSAL_ID_PREFIX):
                target_id = candidate.get("target_id")
                if target_id:
                    target_ids.append(target_id)
    return target_ids


def run() -> int:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=WINDOW_DAYS)
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    print(f"\n=== Attention-bound proposal target-diversity eval ({TEMPLATE_KEY}) ===")
    print(f"window: {window_start.isoformat()} .. {now.isoformat()}")

    conn = open_readonly_connection(dsn)
    if conn is None:
        print("RESULT: insufficient data (postgres unavailable or not read-only)")
        return 0

    try:
        target_ids = fetch_attended_target_ids(conn, window_start)
    finally:
        conn.close()

    if not target_ids:
        print(f"candidates observed: 0 (no {TEMPLATE_KEY} candidates in window)")
        print("RESULT: insufficient data")
        return 0

    distinct_targets = sorted(set(target_ids))
    passed = len(distinct_targets) >= MIN_DISTINCT_TARGETS

    print(f"candidates observed: {len(target_ids)}")
    print(f"distinct target_id count: {len(distinct_targets)} (need >= {MIN_DISTINCT_TARGETS})")
    print(f"distinct targets: {distinct_targets}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(run())
