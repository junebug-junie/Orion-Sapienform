"""Eval: substrate_proposal_frames retention never deletes a row a stage still owes work on.

Lives in evals/, not tests/, because it needs the REAL database and the real pipeline
state -- and that is the entire point. The unit tests in
`tests/test_grammar_retention_periodic.py` drive the floor with fabricated connections and
assert on its return value. They can prove the floor computes what it intends to compute.
They cannot prove the intent matches what the live tables actually contain, and the two
most serious bugs on this patch were exactly that kind of mismatch:

  * the floor probed the pending row's OWN created_at, when the row that must survive is its
    PARENT proposal -- always older, therefore always deleted first;
  * the min-aggregate silently planned as a full-table scan, cheapest under backlog and most
    expensive when healthy, i.e. invisible in the place you would look.

Neither was reachable from a mock. Both are checked here against live data.

Read-only. This eval writes nothing and deletes nothing.

Run: pytest services/orion-sql-writer/evals/test_substrate_retention_integrity_eval.py -q
"""
from __future__ import annotations

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DSN = dict(host="localhost", port=55432, user="postgres",
           password="postgres", dbname="conjourney")

# Kept in step with services/orion-sql-writer/app/settings.py's default. If the env is
# raised, this eval only becomes more conservative, never less.
RETENTION_DAYS = 10

# The floor probe cost that HIGH-1 was about. Unfenced it measured 490 ms / 102,343 buffers.
# Fenced it measured 0.18 / 5.4 / 1.6 ms. 100 ms is far above the fenced numbers and far
# below the unfenced one, so it catches a regression without being flaky on a busy host.
MAX_PROBE_MS = 100.0

# services/orion-sql-writer/.env PORT. Only used to tell "draining" from "stopped".
SQL_WRITER_PORT = 8220


def _connect():
    try:
        return psycopg2.connect(connect_timeout=5, **DSN)
    except psycopg2.OperationalError as exc:  # pragma: no cover - infra gate
        pytest.skip(f"conjourney Postgres unavailable: {exc}")


@pytest.fixture(scope="module")
def conn():
    c = _connect()
    c.set_session(readonly=True, autocommit=True)
    yield c
    c.close()


def _retention_state():
    """This table's live retention block from the sql-writer, or None if unreachable."""
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(
            f"http://localhost:{SQL_WRITER_PORT}/grammar/truth", timeout=5
        ) as resp:
            payload = json.load(resp)
    except Exception:
        return None
    return (payload.get("other_table_retention") or {}).get("substrate_proposal_frames")


def _one(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or {})
        row = cur.fetchone()
    return row


@pytest.mark.parametrize(
    "stage,sql",
    [
        (
            "policy->dispatch",
            """
            SELECT count(*)
              FROM substrate_policy_decision_frames d
              LEFT JOIN substrate_proposal_frames p
                     ON p.frame_id = d.source_proposal_frame_id
             WHERE d.dispatch_pending
               AND d.source_proposal_frame_id IS NOT NULL
               AND p.frame_id IS NULL
            """,
        ),
        (
            "dispatch->feedback",
            """
            SELECT count(*)
              FROM substrate_execution_dispatch_frames d
              LEFT JOIN substrate_proposal_frames p
                     ON p.frame_id = d.source_proposal_frame_id
             WHERE d.feedback_pending
               AND d.source_proposal_frame_id IS NOT NULL
               AND p.frame_id IS NULL
            """,
        ),
    ],
)
def test_no_pending_stage_has_lost_its_parent_proposal(conn, stage, sql):
    """The property the whole floor exists to hold, checked against live rows.

    A stage with its pending marker still set is owed work on a proposal. If that proposal
    has been deleted, retention took a row it was supposed to protect -- silently, since no
    FK exists to complain (see _verify_delete_safe's caveat in grammar_truth.py).

    This is the check that would have caught the parent-before-child bug. Under the buggy
    version the floor sat at the CHILD's timestamp, so every parent older than that by the
    measured mean of 123.2s was eligible for deletion.
    """
    orphaned = _one(conn, sql)[0]
    assert orphaned == 0, (
        f"{orphaned} rows at stage {stage} are still pending but their parent proposal is "
        f"gone -- retention deleted a row the chain floor was supposed to protect"
    )


def test_a_backlog_is_always_explained_by_either_the_floor_or_active_pruning(conn):
    """Retention is converging, or something explains why not. Never neither.

    Rows older than the window are NOT automatically a failure -- an initial backlog takes
    hours to drain at 1000 rows x 3 batches per 60s cycle, and a genuinely-behind pipeline
    stage legitimately pins the floor for as long as it is behind. Both are healthy.

    What is NOT healthy is a backlog with neither explanation, which is what a silently
    stopped retention loop looks like: the row count just sits there. An earlier draft of
    this eval asserted on the floor alone and failed on a perfectly healthy mid-drain
    database -- a backlog is the normal state right after deploy, so the assertion has to
    distinguish the three cases rather than treat any backlog as broken.
    """
    backlog = _one(
        conn,
        "SELECT count(*) FROM substrate_proposal_frames "
        "WHERE created_at < now() - %(days)s * interval '1 day'",
        {"days": RETENTION_DAYS},
    )[0]
    if backlog == 0:
        return  # fully converged, nothing to explain

    cutoff = _one(conn, "SELECT now() - %(days)s * interval '1 day'",
                  {"days": RETENTION_DAYS})[0]
    floor = _one(
        conn,
        """
        SELECT least(
          (SELECT MIN(s.created_at) FROM (
             SELECT created_at FROM substrate_proposal_frames WHERE policy_pending OFFSET 0
           ) s),
          (SELECT MIN(p.created_at) FROM (
             SELECT source_proposal_frame_id FROM substrate_policy_decision_frames
              WHERE dispatch_pending OFFSET 0
           ) d JOIN substrate_proposal_frames p ON p.frame_id = d.source_proposal_frame_id),
          (SELECT MIN(p.created_at) FROM (
             SELECT source_proposal_frame_id FROM substrate_execution_dispatch_frames
              WHERE feedback_pending OFFSET 0
           ) d JOIN substrate_proposal_frames p ON p.frame_id = d.source_proposal_frame_id)
        )
        """,
    )[0]
    if floor is not None and floor <= cutoff:
        return  # explanation 1: a stage is genuinely behind and owns those rows

    # explanation 2: retention is actively draining. Ask the service, not the table -- the
    # table alone cannot distinguish "draining" from "stopped".
    state = _retention_state()
    if state is None:
        pytest.skip(
            f"{backlog} rows past the window and no stage is behind, but the sql-writer "
            f"/grammar/truth endpoint is unreachable, so 'draining' and 'stopped' cannot "
            f"be told apart from the table alone. Not asserting on an unanswerable question."
        )

    assert state.get("enabled") and state.get("rows_pruned_last_run", 0) > 0, (
        f"{backlog} rows are older than the {RETENTION_DAYS}-day window, no pipeline stage "
        f"is behind (floor={floor}, cutoff={cutoff}), and the last retention run pruned "
        f"{state.get('rows_pruned_last_run')} rows "
        f"(enabled={state.get('enabled')}, failure_reason={state.get('failure_reason')}). "
        f"Retention has stopped converging with nothing to explain it."
    )


@pytest.mark.parametrize(
    "stage,sql",
    [
        ("proposal->policy",
         "SELECT MIN(s.created_at) FROM (SELECT created_at FROM substrate_proposal_frames "
         "WHERE policy_pending OFFSET 0) s"),
        ("policy->dispatch",
         "SELECT MIN(p.created_at) FROM (SELECT source_proposal_frame_id FROM "
         "substrate_policy_decision_frames WHERE dispatch_pending OFFSET 0) d JOIN "
         "substrate_proposal_frames p ON p.frame_id = d.source_proposal_frame_id"),
        ("dispatch->feedback",
         "SELECT MIN(p.created_at) FROM (SELECT source_proposal_frame_id FROM "
         "substrate_execution_dispatch_frames WHERE feedback_pending OFFSET 0) d JOIN "
         "substrate_proposal_frames p ON p.frame_id = d.source_proposal_frame_id"),
    ],
)
def test_each_floor_probe_stays_cheap_on_live_data(conn, stage, sql):
    """The probe cost regression that a unit test structurally cannot catch.

    Whether `MIN(created_at) WHERE marker` uses the partial index or degenerates into a
    full-table scan is a PLANNER decision, made from live statistics. It flipped once
    already: `idx_substrate_policy_decision_frames_dispatch_pending` had bloated to 9312 kB,
    which tipped the planner into scanning 474,708 rows and discarding all of them, 490 ms
    at a time, every 60 seconds. No amount of mocked SQL-string assertion sees that.
    """
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)
        plan = cur.fetchone()[0][0]

    ms = plan["Execution Time"]
    node = plan["Plan"]

    def walk(n):
        yield n
        for child in n.get("Plans", []):
            yield from walk(child)

    scans = [n["Node Type"] for n in walk(node) if "Scan" in n["Node Type"]]
    assert ms < MAX_PROBE_MS, (
        f"{stage} floor probe took {ms:.1f} ms (limit {MAX_PROBE_MS}); plan used {scans}. "
        f"The OFFSET 0 fence may have been removed, or the partial index has bloated far "
        f"enough to flip the planner back to a full scan."
    )
    assert not any(s == "Seq Scan" or s.startswith("Parallel Seq") for s in scans), (
        f"{stage} floor probe fell back to a sequential scan: {scans}"
    )
