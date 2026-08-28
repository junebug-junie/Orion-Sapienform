"""Eval: the fleet power aggregate is actually landing in the live database.

Lives in evals/, not tests/, for the reason the sibling eval states: unit tests drive
fabricated inputs and can prove intent, not that the intent matches what the live tables
contain. This one exists because of a specific near-miss.

The first version of this feature shipped with a correct route map, a correct model,
working retention, a created table and ten green unit tests -- and never subscribed to
the channel, because `SQL_WRITER_SUBSCRIBE_CHANNELS` REPLACES the Python default
wholesale instead of merging the way `route_map` does. Every gate passed. The table
would have stayed empty forever. A live eval asking "are rows landing?" is the only
check in the family that would have caught it, which is exactly the argument the
substrate retention eval already makes for its own existence.

Expected to FAIL until orion-sql-writer is redeployed with the new subscription -- that
failure is the eval doing its job, not a defect.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("psycopg2")
import psycopg2  # noqa: E402


def _dsn() -> str:
    dsn = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("no POSTGRES_URI/DATABASE_URL in env")
    return dsn


@pytest.fixture(scope="module")
def conn():
    c = psycopg2.connect(_dsn())
    try:
        yield c
    finally:
        c.close()


def _one(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()


def test_the_table_exists(conn) -> None:
    (exists,) = _one(conn, "SELECT to_regclass('public.orion_biometrics_cluster') IS NOT NULL")
    assert exists, "table missing -- writer has not booted with this model yet"


def test_rows_are_landing(conn) -> None:
    """The check the unit tests structurally cannot make."""
    (n,) = _one(
        conn,
        "SELECT count(*) FROM orion_biometrics_cluster "
        "WHERE observed_at > now() - interval '30 minutes'",
    )
    assert n > 0, (
        "no cluster rows in the last 30 minutes. The channel publishes roughly every "
        "30s, so zero means the subscription is not live -- check "
        "SQL_WRITER_SUBSCRIBE_CHANNELS contains orion:biometrics:cluster"
    )


def test_the_fleet_total_is_not_degenerate(conn) -> None:
    """A row that lands with no watts is a schema-valid payload with no cognitive
    content -- the empty-shell failure, in table form."""
    row = _one(
        conn,
        "SELECT count(*), count(pdu_watts), count(chassis_watts), "
        "       round(max(chassis_watts)::numeric, 0) "
        "FROM orion_biometrics_cluster WHERE observed_at > now() - interval '2 hours'",
    )
    total, with_pdu, with_chassis, peak = row
    if not total:
        pytest.skip("no rows yet -- test_rows_are_landing reports the real failure")
    assert with_chassis > 0, "every row landed without chassis_watts"
    assert peak and peak > 0, "fleet chassis_watts never exceeded zero"


def test_circe_is_present_via_the_proxy_and_says_so(conn) -> None:
    """circe has no LAN path to the PDU and no BMC, so its watts reach the fleet only
    as a proxied reading. The provenance must survive persistence -- otherwise a later
    reader concludes its dead NIC came back."""
    row = _one(
        conn,
        "SELECT count(*) FROM orion_biometrics_cluster "
        "WHERE observed_at > now() - interval '2 hours' "
        "  AND measurements_proxied ? 'circe'",
    )
    if row and row[0] == 0:
        (total,) = _one(
            conn,
            "SELECT count(*) FROM orion_biometrics_cluster "
            "WHERE observed_at > now() - interval '2 hours'",
        )
        if not total:
            pytest.skip("no rows yet")
    assert row[0] > 0, "no row credits circe as proxied -- fleet total is missing a machine"
