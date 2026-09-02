#!/usr/bin/env python3
"""Report how close Postgres is to refusing connections, and fail loudly near the wall.

NOT IN CI, AND WHY
------------------
This needs a live Postgres. CI has none, so it follows the pattern in
`scripts/check_merge_domination.py`: run it from cron on the host. See the
`make postgres-headroom` target and the crontab line in the PR report.

Why this exists
---------------
On 2026-08-31 the deployment was refusing connections and nothing was watching, so
it was found by accident while investigating something else. The server log held
**217** `FATAL: sorry, too many clients already` over five days.

What the naive reading gets wrong
---------------------------------
1. `pg_stat_activity` lists background processes -- checkpointer, walwriter,
   background writer, autovacuum launcher, logical replication launcher -- that hold
   no connection slot. Counting them made an earlier version of this script report
   "102/97 used" against a server whose max_connections is 100.

2. `superuser_reserved_connections` does NOT reduce the ceiling when the clients are
   themselves superusers. On this deployment every one of the client backends
   connects as `postgres`, a superuser, so the wall is max_connections (100), not
   max_connections - reserved (97). The log proves it: 217 refusals were all
   "sorry, too many clients already" and ZERO were "remaining connection slots are
   reserved for non-replication superuser connections", which is the message a
   non-superuser gets when it hits the lower ceiling.

3. That is not a happy accident, it is a hazard, and this script names it. The whole
   point of the reserve is to guarantee an operator can still get in during an
   incident. Services connecting as a superuser spend that reserve like any other
   slot, so when the database fills there is no emergency door -- which is exactly
   why `psql` was refused three times while diagnosing the original incident.

This measures capacity, not health. A high number is not automatically a leak: when
this was written only 8 of 93 idle connections had been idle over two hours; the rest
were live pool connections cycling normally. Use --verbose before concluding.

Exit codes
----------
0  fine
1  alarm: saturated, or below --min-free-pct (only when --gate is passed)
2  cannot check: no psycopg2, or a connection error unrelated to saturation

Same exit-code convention as scripts/check_sql_migrations_applied.py, so an infra
failure can never be mistaken for a pass.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

EXIT_OK = 0
EXIT_ALARM = 1
EXIT_CANNOT_CHECK = 2

# Postgres reports both forms of connection refusal with this SQLSTATE. Matching the
# code rather than English text keeps this working under a non-English server locale.
SQLSTATE_TOO_MANY_CONNECTIONS = "53300"

# Message fallbacks for the case psycopg2 raises a bare OperationalError with no
# pgcode -- which is what happens when the refusal lands during the startup
# handshake, the case this gate exists for. BOTH forms must be here: the first is
# what a superuser sees, the second is what a non-superuser sees when it hits
# max_connections - superuser_reserved_connections. Only the first has ever appeared
# in this deployment's log, and it will stay that way exactly as long as every
# service connects as `postgres`.
_SATURATION_PATTERNS = (
    r"too many clients",
    r"remaining connection slots are reserved",
)

DEFAULT_MIN_FREE_PCT = 15.0


@dataclass(frozen=True)
class Headroom:
    """A capacity reading.

    `used` and `superuser_used` count CLIENT backends only -- not the background
    processes that appear in pg_stat_activity without consuming a slot.
    """

    max_connections: int
    reserved_for_superuser: int
    used: int
    superuser_used: int = 0

    @property
    def nonsuperuser_ceiling(self) -> int:
        """Where an ordinary (non-superuser) role is refused."""
        return max(0, self.max_connections - self.reserved_for_superuser)

    @property
    def free(self) -> int:
        """Slots before the postmaster refuses everyone, superusers included."""
        return max(0, self.max_connections - self.used)

    @property
    def free_pct(self) -> float:
        if self.max_connections <= 0:
            return 0.0
        return 100.0 * self.free / self.max_connections

    @property
    def reserve_is_decorative(self) -> bool:
        """True when the superuser reserve guarantees an operator nothing.

        The reserve only holds slots back from *ordinary* roles. If every client is
        a superuser it is spent like any other slot, and the operator is locked out
        at exactly the moment they need to get in.
        """
        return self.used > 0 and self.superuser_used == self.used

    def summary(self) -> str:
        return (
            f"{self.used}/{self.max_connections} used "
            f"({self.free} free, {self.free_pct:.0f}%) "
            f"[reserved={self.reserved_for_superuser}, "
            f"superuser clients={self.superuser_used}]"
        )


def is_saturation_error(exc: BaseException) -> bool:
    """True when a connection failed because Postgres had no slot for it."""
    if getattr(exc, "pgcode", None) == SQLSTATE_TOO_MANY_CONNECTIONS:
        return True
    text = str(exc)
    return any(re.search(p, text, re.IGNORECASE) for p in _SATURATION_PATTERNS)


def connection_params(explicit: Optional[str] = None):
    """A DSN string, or psycopg2 kwargs built from the host-side defaults.

    Matches `scripts/check_sql_migrations_applied.py` exactly (ORION_PG_* with a
    localhost:55432 default) so this works from the host with no setup. That
    matters: the root .env's POSTGRES_URI names a docker-internal hostname which
    does not resolve from the host, so defaulting to it would make the tool look
    broken to the operator most likely to run it.
    """
    dsn = explicit or os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    return dict(
        host=os.environ.get("ORION_PG_HOST", "localhost"),
        port=int(os.environ.get("ORION_PG_PORT", "55432")),
        user=os.environ.get("ORION_PG_USER", "postgres"),
        password=os.environ.get(
            "ORION_PG_PASSWORD", os.environ.get("PGPASSWORD", "postgres")
        ),
        dbname=os.environ.get("ORION_PG_DB", "conjourney"),
    )


CLIENT_BACKENDS = "backend_type = 'client backend'"


def read_headroom(conn) -> Headroom:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_setting('max_connections')::int, "
            "current_setting('superuser_reserved_connections')::int, "
            f"(SELECT count(*) FROM pg_stat_activity WHERE {CLIENT_BACKENDS}), "
            "(SELECT count(*) FROM pg_stat_activity a "
            " JOIN pg_roles r ON r.rolname = a.usename "
            f" WHERE a.{CLIENT_BACKENDS} AND r.rolsuper)"
        )
        row = cur.fetchone()
    return Headroom(
        max_connections=int(row[0]),
        reserved_for_superuser=int(row[1]),
        used=int(row[2]),
        superuser_used=int(row[3]),
    )


def read_idle_split(conn, stale_hours: int = 2) -> tuple[int, int]:
    """(idle_total, idle_longer_than_stale_hours). Distinguishes churn from a leak."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FILTER (WHERE state = 'idle'), "
            "count(*) FILTER (WHERE state = 'idle' "
            "                 AND state_change < now() - make_interval(hours => %s)) "
            f"FROM pg_stat_activity WHERE {CLIENT_BACKENDS}",
            (stale_hours,),
        )
        row = cur.fetchone()
    return int(row[0]), int(row[1])


def top_clients(conn, limit: int = 10) -> list[tuple[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(host(client_addr), '<local>'), count(*) "
            f"FROM pg_stat_activity WHERE {CLIENT_BACKENDS} "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT %s",
            (limit,),
        )
        return [(str(a), int(n)) for a, n in cur.fetchall()]


DEFAULT_NOTIFY_BASE_URL = os.getenv("NOTIFY_BASE_URL", "http://localhost:7140")


def default_state_file() -> str:
    """Same telemetry-tree convention as scripts/disk_threshold_watchdog.py."""
    root = os.getenv("TELEMETRY_ROOT", "/mnt/telemetry")
    project = os.getenv("PROJECT", "orion-athena")
    return os.path.join(root, project, "postgres-headroom", "state.json")


class _StateLock:
    """Non-blocking flock on `<state_file>.lock`.

    Same pattern as disk_threshold_watchdog.py: overlapping cron ticks must not
    interleave the read-evaluate-write cycle, and a run that cannot get the lock
    skips rather than waits -- the next tick is only minutes away.
    """

    def __init__(self, state_file: str) -> None:
        self._path = f"{state_file}.lock"
        self._fh = None

    def __enter__(self) -> bool:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._fh = open(self._path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            return False
        return True

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def _load_state(state_file: str) -> dict[str, Any]:
    try:
        with open(state_file, encoding="utf-8") as fh:
            loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(state_file: str, state: dict[str, Any]) -> None:
    """Atomic replace so a crash mid-write cannot leave unparseable state."""
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(state_file) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, state_file)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


SEVERITY_RANK = {"warning": 1, "critical": 2}


def notify_alarm(args, *, reason: str, message: str, severity: str) -> None:
    """Fire at most one Pending Attention card per alarm EPISODE, per severity.

    Two rules, both of which have a way to go wrong that is worse than no
    debounce at all:

    1. State records `notified=True` ONLY after the client confirms `ok`. A card
       that failed to land (orion-notify down or erroring -- NotifyClient returns
       ok=False, it does not raise) is retried on the next tick instead of being
       debounced into permanent silence by the mere fact that we noticed once.

    2. The episode is keyed on SEVERITY RANK, not on `reason`. `saturated` and
       `headroom_low` are two readings of one incident, and at the wall which one
       a given tick gets is close to a coin flip -- whether this tick's own
       connection attempt happens to win a slot. Keying on `reason` made an
       alternating run fire a card EVERY tick, indefinitely, during exactly the
       incident this exists for (driven in review: 8 ticks, 8 cards). Ranking
       instead means the escalation warning -> critical still fires once, and the
       flap back down is silent.

    A human acks these cards; nothing here auto-resolves them, so re-firing while
    one is already open would be noise, not signal.
    """
    if not args.notify:
        return
    try:
        _notify_alarm_locked(args, reason=reason, message=message, severity=severity)
    except Exception as exc:  # noqa: BLE001 - escalation must never mask the alarm
        # An unwritable telemetry root used to surface as an unhandled traceback,
        # whose Python exit status 1 is indistinguishable from EXIT_ALARM.
        print(
            f"  escalation failed ({exc.__class__.__name__}: {exc}); "
            "the alarm itself is still reported",
            file=sys.stderr,
        )


def _notify_alarm_locked(args, *, reason: str, message: str, severity: str) -> None:
    with _StateLock(args.state_file) as acquired:
        if not acquired:
            return
        state = _load_state(args.state_file)
        rank = SEVERITY_RANK.get(severity, 1)
        prev_rank = int(state.get("episode_rank") or 0)
        confirmed = state.get("notified") is True
        if prev_rank and confirmed and rank <= prev_rank:
            return
        try:
            from orion.notify.client import NotifyClient
        except ImportError as exc:
            print(f"  notify unavailable ({exc}); alarm not escalated", file=sys.stderr)
            return
        client = NotifyClient(
            base_url=args.notify_base_url, api_token=args.notify_api_token, timeout=10
        )
        accepted = client.attention_request(
            message=message,
            severity=severity,
            require_ack=True,
            context={
                "source_service": "check_postgres_connection_headroom",
                "reason": reason,
            },
        )
        ok = bool(getattr(accepted, "ok", False))
        state.update(
            {
                "reason": reason,
                "episode_rank": rank,
                "notified": ok,
                "last_alarm_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save_state(args.state_file, state)
        print(
            f"  attention card {'sent' if ok else 'FAILED to send (will retry next tick)'}",
            file=sys.stderr,
        )


def clear_alarm(args) -> None:
    """Alarm cleared: forget the episode so the next one notifies again.

    Silent by design -- an ack'd card has already been seen by a human, and there
    is nothing for this script to auto-resolve.
    """
    if not args.notify:
        return
    try:
        with _StateLock(args.state_file) as acquired:
            if not acquired:
                return
            state = _load_state(args.state_file)
            if state.get("episode_rank") or state.get("reason") or state.get("notified"):
                # Keep last_alarm_at: when the previous episode ended is worth
                # more to whoever reads this file than a tidy dict.
                state.update({"reason": None, "episode_rank": 0, "notified": False})
                _save_state(args.state_file, state)
    except Exception as exc:  # noqa: BLE001 - see notify_alarm
        print(
            f"  could not clear escalation state ({exc.__class__.__name__}: {exc})",
            file=sys.stderr,
        )


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
    ap.add_argument(
        "--notify",
        action="store_true",
        help="raise a Hub Pending Attention card on alarm (debounced; one per episode)",
    )
    ap.add_argument("--notify-base-url", default=DEFAULT_NOTIFY_BASE_URL)
    ap.add_argument("--notify-api-token", default=os.getenv("NOTIFY_API_TOKEN"))
    ap.add_argument("--state-file", default=None, help="debounce state (see default_state_file)")
    args = ap.parse_args(argv)
    if args.state_file is None:
        args.state_file = default_state_file()

    params = connection_params(args.dsn)

    try:
        import psycopg2
    except ImportError:
        print(
            "psycopg2 not installed; cannot check connection headroom. Run this with "
            "the repo venv, e.g. .venv/bin/python scripts/check_postgres_connection_headroom.py",
            file=sys.stderr,
        )
        return EXIT_CANNOT_CHECK

    try:
        if isinstance(params, str):
            conn = psycopg2.connect(params, connect_timeout=10)
        else:
            conn = psycopg2.connect(connect_timeout=10, **params)
        # Read-only: this script must never be the thing that writes to production.
        conn.set_session(readonly=True, autocommit=True)
    except Exception as exc:  # noqa: BLE001 - classified immediately below
        if is_saturation_error(exc):
            # The check could not get in because the thing it checks for is
            # happening. This is the alarm, not a failure of the alarm.
            print(
                "SATURATED: Postgres refused this connection -- no slots left "
                f"({exc.__class__.__name__}: {str(exc).strip()})",
                file=sys.stderr,
            )
            notify_alarm(
                args,
                reason="saturated",
                message=(
                    "Postgres refused a new connection: every slot is in use. "
                    "Operators cannot get in either -- all client backends are "
                    "superusers, so the reserved slots hold nothing back."
                ),
                severity="critical",
            )
            return EXIT_ALARM
        print(f"cannot check connection headroom: {exc}", file=sys.stderr)
        return EXIT_CANNOT_CHECK

    try:
        headroom = read_headroom(conn)
        print(f"postgres connections: {headroom.summary()}")
        if headroom.reserve_is_decorative:
            print(
                f"  WARNING: all {headroom.used} client backends are superusers, so "
                f"the {headroom.reserved_for_superuser} reserved slots hold nothing "
                "back -- there is no emergency door for an operator when this fills."
            )
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

    # Evaluated independently of --gate. When these were fused, a --notify run
    # WITHOUT --gate fell through to clear_alarm() on a still-alarming reading,
    # wiping the episode and letting the next gated tick fire a second card for
    # the same incident. --gate now decides the exit code and nothing else.
    alarm = headroom.free_pct < args.min_free_pct
    if alarm:
        if args.gate:
            print(
                f"FAIL: only {headroom.free_pct:.0f}% of connection slots free "
                f"(threshold {args.min_free_pct:.0f}%). {headroom.summary()}",
                file=sys.stderr,
            )
        notify_alarm(
            args,
            reason="headroom_low",
            message=(
                f"Postgres connection headroom low: {headroom.summary()}. "
                f"Below the {args.min_free_pct:.0f}% free threshold."
                + (
                    " Every client backend is a superuser, so the reserved slots "
                    "will not hold a door open for an operator."
                    if headroom.reserve_is_decorative
                    else ""
                )
            ),
            severity="warning",
        )
        if args.gate:
            return EXIT_ALARM
        return EXIT_OK
    clear_alarm(args)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
