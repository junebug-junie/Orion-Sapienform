from __future__ import annotations

import hashlib
from typing import Optional

import psycopg2


def _lock_key(model_version: str) -> int:
    digest = hashlib.sha256(model_version.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def acquire_lock(dsn: str, model_version: str) -> Optional[psycopg2.extensions.connection]:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    key = _lock_key(model_version)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s::bigint);", (key,))
        locked = cur.fetchone()[0]
    if not locked:
        conn.close()
        return None
    return conn


def release_lock(conn: psycopg2.extensions.connection, model_version: str) -> None:
    key = _lock_key(model_version)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s::bigint);", (key,))
    finally:
        conn.close()
