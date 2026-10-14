"""Skip boot-time DDL when the schema file is already applied.

Each idempotent schema file (memory_cards.sql, memory_crystallizations.sql) is
re-run on every boot. "Idempotent" is not "lock-free": ``ALTER TABLE ... ADD
COLUMN IF NOT EXISTS`` takes ACCESS EXCLUSIVE and ``CREATE INDEX IF NOT
EXISTS`` takes SHARE on the table *before* finding there is nothing to do. The
nightly ``orion-backup-databases`` pg_dump (03:45 UTC, ~55 min) holds ACCESS
SHARE on every table for its whole transaction, so a hub restart in that
window hit ``lock_timeout`` on a no-op ALTER -- 3 of the 4 failures 10-12..14
fell inside it. The failure also skipped the crystallizations file behind it.

The fix records sha256(file) in ``orion_schema_fingerprints`` after a
successful apply and skips the file entirely while the fingerprint matches.
The check reads only that small table, never the schema's own tables, so it
does not queue behind a backup. Editing the .sql file changes its hash and the
next boot applies it. Trade-off: an object dropped by hand under an unchanged
file is not re-created until the file changes or ``force=True`` is passed.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

FINGERPRINT_TABLE = "orion_schema_fingerprints"

_CREATE_FINGERPRINT_TABLE = (
    f"CREATE TABLE IF NOT EXISTS {FINGERPRINT_TABLE} ("
    " name text PRIMARY KEY,"
    " sha256 text NOT NULL,"
    " applied_at timestamptz NOT NULL DEFAULT now())"
)
_UPSERT_FINGERPRINT = (
    f"INSERT INTO {FINGERPRINT_TABLE} (name, sha256, applied_at) VALUES (%s, %s, now()) "
    "ON CONFLICT (name) DO UPDATE SET sha256 = EXCLUDED.sha256, applied_at = EXCLUDED.applied_at"
)


def sql_fingerprint(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _stored_fingerprint(cur: Any, name: str) -> str | None:
    cur.execute("SELECT to_regclass(%s)", (FINGERPRINT_TABLE,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    cur.execute(f"SELECT sha256 FROM {FINGERPRINT_TABLE} WHERE name = %s", (name,))
    row = cur.fetchone()
    return str(row[0]) if row and row[0] else None


def apply_schema_file_once(
    dsn: str,
    *,
    name: str,
    sql: str,
    lock_timeout_ms: int,
    statement_timeout_ms: int,
    connect: Callable[..., Any],
    force: bool = False,
) -> bool:
    """Apply ``sql`` unless its fingerprint is already recorded.

    Returns True when the DDL ran, False when it was skipped as current.
    Raises whatever the DDL raises (e.g. ``LockNotAvailable``); the
    fingerprint is written only after the whole file succeeded, so a failed
    or partial apply is retried on the next call.
    """
    fingerprint = sql_fingerprint(sql)
    options = f"-c lock_timeout={lock_timeout_ms} -c statement_timeout={statement_timeout_ms}"
    with connect(dsn, options=options) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            if not force and _stored_fingerprint(cur, name) == fingerprint:
                logger.info("schema_apply_skipped name=%s reason=fingerprint_current sha256=%s", name, fingerprint[:12])
                return False
            cur.execute(sql)
            cur.execute(_CREATE_FINGERPRINT_TABLE)
            cur.execute(_UPSERT_FINGERPRINT, (name, fingerprint))
    logger.info("schema_apply_ran name=%s sha256=%s", name, fingerprint[:12])
    return True
