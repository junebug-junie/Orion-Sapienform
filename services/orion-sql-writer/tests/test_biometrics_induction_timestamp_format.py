"""`orion_biometrics_induction.timestamp` must stay lexically sortable.

`orion/substrate/metacog_trend_signals.py::latest_biometrics_induction_by_node`
orders on this varchar column directly (`ORDER BY b.timestamp DESC LIMIT 1`),
because `varchar::timestamptz` is not IMMUTABLE and therefore cannot be
indexed. That is only correct while text order matches chronological order,
which holds because every value is rendered `YYYY-MM-DD HH:MM:SS[.ffffff]+00`.

That rendering is NOT a property of the column. `app/worker.py:1981` calls
`obj.model_dump()` without `mode="json"`, so the field stays a `datetime`,
psycopg2 binds it as a timestamptz literal, and Postgres assignment-casts it
into the varchar column using the WRITER SESSION's DateStyle and TimeZone. Set
`PGTZ=America/Denver` on the sql-writer container and new rows render `-06`,
at which point ordering silently diverges: `LIMIT 1` starts returning an older
row, the max-age filter drops it, and the Hub's Biometrics card reads "no
reading" with no error raised anywhere.

CLAUDE.md: "the right fix for a forgotten invariant is not a louder comment,
it is a failing gate." This is that gate.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import pytest

# The exact shape the ordering argument depends on. Fractional seconds are
# right-trimmed by Postgres, hence the optional group.
CANONICAL = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d{1,6})?\+00$")

_DSN = os.getenv(
    "ORION_TEST_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"
)


def _engine():
    try:
        from sqlalchemy import create_engine
    except ImportError:
        return None
    try:
        engine = create_engine(_DSN, connect_args={"connect_timeout": 2})
        with engine.connect():
            pass
        return engine
    except Exception:
        return None


ENGINE = _engine()


def test_canonical_regex_rejects_a_non_utc_offset() -> None:
    """The gate itself must be able to fail (no live DB needed)."""
    assert CANONICAL.match("2026-09-03 05:32:41.437679+00")
    assert CANONICAL.match("2026-09-03 05:32:41+00")
    # The exact regression this file exists to catch.
    assert not CANONICAL.match("2026-09-02 23:32:41.437679-06")
    # ISO 'T' separator, i.e. someone switched to model_dump(mode="json").
    assert not CANONICAL.match("2026-09-03T05:32:41.437679+00:00")


@pytest.mark.skipif(ENGINE is None, reason="local Postgres not reachable")
def test_stored_timestamps_are_all_canonical() -> None:
    from sqlalchemy import text

    with ENGINE.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT timestamp FROM orion_biometrics_induction "
                "ORDER BY timestamp DESC LIMIT 2000"
            )
        ).all()
    if not rows:
        pytest.skip("no orion_biometrics_induction rows")
    bad = [r[0] for r in rows if not CANONICAL.match(r[0])]
    assert not bad, f"{len(bad)} non-canonical timestamp(s), e.g. {bad[:3]}"


@pytest.mark.skipif(ENGINE is None, reason="local Postgres not reachable")
def test_text_order_matches_chronological_order() -> None:
    """The invariant itself, asserted against the data rather than argued.

    Compared over the whole table, not a sample: a single disagreement
    anywhere is enough to make `LIMIT 1` wrong for that node.
    """
    from sqlalchemy import text

    with ENGINE.connect() as conn:
        disagreements = conn.execute(
            text(
                """
                SELECT count(*) FROM (
                  SELECT row_number() OVER (PARTITION BY node ORDER BY timestamp DESC, id) rt,
                         row_number() OVER (PARTITION BY node ORDER BY timestamp::timestamptz DESC, id) rc
                  FROM orion_biometrics_induction
                ) s WHERE rt <> rc
                """
            )
        ).scalar()
    assert disagreements == 0, (
        f"{disagreements} rows where text ordering and chronological ordering disagree -- "
        "ORDER BY on the bare varchar column is no longer safe"
    )


@pytest.mark.skipif(ENGINE is None, reason="local Postgres not reachable")
def test_a_non_utc_write_would_actually_break_ordering() -> None:
    """Prove the hazard is real, not hypothetical -- without writing a row.

    Renders 'now' the way a writer session with a non-UTC TimeZone would, and
    shows that value sorts BELOW the current newest row despite being newer.
    That is exactly the silent failure: LIMIT 1 would keep returning the older
    row forever.
    """
    from sqlalchemy import text

    with ENGINE.connect() as conn:
        newest = conn.execute(
            text("SELECT max(timestamp) FROM orion_biometrics_induction")
        ).scalar()
        if newest is None:
            pytest.skip("no orion_biometrics_induction rows")
        # SET TIME ZONE is what PGTZ on the writer container actually does.
        # (An earlier version of this test used
        # `(now() AT TIME ZONE 'America/Denver')::timestamptz::text`, which
        # re-interprets the value in the SESSION zone and therefore still
        # renders `+00` -- it demonstrated nothing. Kept as a note because the
        # wrong form looks equally plausible.)
        conn.execute(text("SET TIME ZONE 'America/Denver'"))
        as_denver = conn.execute(text("SELECT now()::text")).scalar()

    assert CANONICAL.match(newest), newest
    assert not CANONICAL.match(as_denver), (
        "expected a non-UTC render to be non-canonical; the gate above would not catch it"
    )
    # Newer in wall-clock terms, yet it does not sort above the existing max.
    assert as_denver < newest, (
        "a non-UTC-rendered timestamp sorted ABOVE the newest row, so this "
        "particular demonstration no longer demonstrates the hazard -- "
        "re-derive it before trusting the ordering invariant"
    )


def test_writer_does_not_serialise_the_timestamp_as_iso_json() -> None:
    """`model_dump(mode="json")` would emit `2026-09-03T05:32:41+00:00`.

    That renders with a 'T' separator and a `+00:00` offset, which sorts
    differently from the existing `YYYY-MM-DD ...+00` rows -- mixing both
    formats in one column breaks ordering across the boundary.
    """
    from pathlib import Path

    worker = Path(__file__).resolve().parents[1] / "app" / "worker.py"
    src = worker.read_text()

    # Scoped to the SQL persistence path only. worker.py legitimately uses
    # model_dump(mode="json") ~10 times elsewhere for BUS payloads, where ISO
    # strings are correct -- a file-wide assertion here is a false positive,
    # which is how the first version of this test failed.
    marker = 'data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()'
    assert marker in src, (
        "the SQL persistence path in worker.py no longer looks the way this "
        "gate expects -- re-derive how the timestamp reaches the varchar column "
        "before assuming the ordering invariant still holds"
    )
    # And the row actually goes to Postgres from there.
    idx = src.index(marker)
    assert "_write_row" in src[idx : idx + 800], src[idx : idx + 200]
