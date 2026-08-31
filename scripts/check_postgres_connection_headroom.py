#!/usr/bin/env python3
"""Report how close Postgres is to refusing connections, and fail loudly near the wall.

Why this exists
---------------
On 2026-08-31 the deployment was sitting at 97 of 100 connections and actively
refusing new ones (`FATAL: sorry, too many clients already`). Nothing was watching,
so it was found by accident while investigating something else. `max_connections`
was never declared anywhere in the repo -- we were running on the postgres:15
default of 100 and had quietly grown 25 distinct clients into it.

Two details that make the raw number misleading, and that this script handles:

1. Services cannot use all of `max_connections`. `superuser_reserved_connections`
   (default 3) is held back for superusers, so the ceiling a normal service hits is
   `max_connections - superuser_reserved_connections`. At 97/100 in use, the
   service-visible pool was already fully gone.
2. A check that dies when the database is full is worse than no check, because it
   goes quiet exactly when it should be shouting. If connecting fails *because of
   saturation*, that is the finding: report it and exit non-zero. Only a genuinely
   unrelated connection error is an error.

This measures capacity, not health. A high number is not automatically a leak --
when this was written, only 8 of 94 idle connections had been idle longer than two
hours; the rest were live pool connections cycling normally. Use --verbose to see
that split before concluding anything about a leak.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

# Postgres reports saturation with this SQLSTATE. Matching the code rather than the
# English message keeps this working under a non-English server locale.
SQLSTATE_TOO_MANY_CONNECTIONS = "53300"

DEFAULT_MIN_FREE_PCT = 15.0


@dataclass(frozen=True)
class Headroom:
    """A capacity reading.

    `used` counts CLIENT backends only -- superuser sessions included, but not the
    background processes (checkpointer, walwriter, autovacuum launcher, ...) that
    show up in pg_stat_activity without consuming a max_connections slot.
    """

    max_connections: int
    reserved_for_superuser: int
    used: int

    @property
    def service_ceiling(self) -> int:
        """What a non-superuser client can actually reach before it is refused."""
        return max(0, self.max_connections - self.reserved_for_superuser)

    @property
    def free_for_services(self) -> int:
        """Slots left for services. Negative is impossible; zero means refusing."""
        return max(0, self.service_ceiling - self.used)

    @property
    def free_pct(self) -> float:
        if self.service_ceiling <= 0:
            return 0.0
        return 100.0 * self.free_for_services / self.service_ceiling

    def summary(self) -> str:
        return (
            f"{self.used}/{self.service_ceiling} used "
            f"({self.free_for_services} free, {self.free_pct:.0f}%) "
            f"[max_connections={self.max_connections}, "
            f"reserved={self.reserved_for_superuser}]"
        )


def is_saturation_error(exc: BaseException) -> bool:
    """True when a connection failed because Postgres is out of slots.

    Checked by SQLSTATE first (locale-independent). The message fallback exists
    because psycopg2 raises a bare OperationalError with no pgcode when the refusal
    happens during the startup handshake -- which is precisely the case we care
    about most.
    """
    pgcode = getattr(exc, "pgcode", None)
    if pgcode == SQLSTATE_TOO_MANY_CONNECTIONS:
        return True
    return bool(re.search(r"too many clients", str(exc), re.IGNORECASE))


def resolve_dsn(explicit: Optional[str] = None) -> str:
    """Same resolution order the rest of scripts/ uses (see agent_board_lib.py)."""
    dsn = explicit or os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit(
            "no DSN: pass --dsn, or set $POSTGRES_URI / $DATABASE_URL "
            "(e.g. from services/orion-hub/.env)"
        )
    return dsn


def read_headroom(conn) -> Headroom:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_setting('max_connections')::int, "
            "current_setting('superuser_reserved_connections')::int, "
            # Only client backends consume max_connections slots. The checkpointer,
            # walwriter, background writer, autovacuum launcher and logical
            # replication launcher all appear in pg_stat_activity but are budgeted
            # separately -- counting them made this script report "102/97 used" on a
            # server whose max_connections is 100.
            "(SELECT count(*) FROM pg_stat_activity "
            " WHERE backend_type = 'client backend')"
        )
        row = cur.fetchone()
    return Headroom(
        max_connections=int(row[0]),
        reserved_for_superuser=int(row[1]),
        used=int(row[2]),
    )


def read_idle_split(conn, stale_hours: int = 2) -> tuple[int, int]:
    """(idle_total, idle_longer_than_stale_hours). Distinguishes churn from a leak."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FILTER (WHERE state = 'idle'), "
            "count(*) FILTER (WHERE state = 'idle' "
            "                 AND state_change < now() - make_interval(hours => %s)) "
            "FROM pg_stat_activity WHERE backend_type = 'client backend'",
            (stale_hours,),
        )
        row = cur.fetchone()
    return int(row[0]), int(row[1])


def top_clients(conn, limit: int = 10) -> list[tuple[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(host(client_addr), '<local>'), count(*) "
            "FROM pg_stat_activity WHERE backend_type = 'client backend' "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT %s",
            (limit,),
        )
        return [(str(a), int(n)) for a, n in cur.fetchall()]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default $POSTGRES_URI)")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero when free headroom is below --min-free-pct",
    )
    ap.add_argument("--min-free-pct", type=float, default=DEFAULT_MIN_FREE_PCT)
    ap.add_argument("--stale-hours", type=int, default=2)
    ap.add_argument("--verbose", action="store_true", help="idle split and top clients")
    args = ap.parse_args(argv)

    dsn = resolve_dsn(args.dsn)

    try:
        import psycopg2
    except ImportError:
        print("psycopg2 not installed; cannot check connection headroom", file=sys.stderr)
        return 2

    try:
        conn = psycopg2.connect(dsn, connect_timeout=10)
    except Exception as exc:  # noqa: BLE001 - we re-raise anything unrelated below
        if is_saturation_error(exc):
            # The check could not get in because the thing it checks for is happening.
            # This is the alarm, not a failure of the alarm.
            print(
                "SATURATED: Postgres refused this connection -- no slots left "
                f"({exc.__class__.__name__}: {str(exc).strip()})",
                file=sys.stderr,
            )
            return 1
        raise

    try:
        headroom = read_headroom(conn)
        print(f"postgres connections: {headroom.summary()}")
        if args.verbose:
            idle_total, idle_stale = read_idle_split(conn, args.stale_hours)
            print(
                f"  idle: {idle_total} total, {idle_stale} idle >"
                f"{args.stale_hours}h (a leak looks like a large second number)"
            )
            for addr, count in top_clients(conn):
                print(f"  {addr:<20} {count}")
    finally:
        conn.close()

    if args.gate and headroom.free_pct < args.min_free_pct:
        print(
            f"FAIL: only {headroom.free_pct:.0f}% of service connection slots free "
            f"(threshold {args.min_free_pct:.0f}%). {headroom.summary()}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
