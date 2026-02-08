from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException, Query

from app.settings import settings
from app.storage.pg import pg_conn


router = APIRouter()

_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _allowed_schemas() -> List[str]:
    return [s.strip() for s in settings.topic_foundry_introspect_schemas.split(",") if s.strip()]


def _get_cached(key: Tuple[str, str, str]) -> Dict[str, Any] | None:
    cached = _CACHE.get(key)
    if not cached:
        return None
    if time.time() - cached["ts"] > settings.topic_foundry_introspect_cache_secs:
        _CACHE.pop(key, None)
        return None
    return cached["payload"]


def _set_cached(key: Tuple[str, str, str], payload: Dict[str, Any]) -> None:
    _CACHE[key] = {"ts": time.time(), "payload": payload}


@router.get("/introspect/schemas")
def list_schemas() -> Dict[str, Any]:
    allowed = _allowed_schemas()
    if not allowed:
        return {"schemas": []}
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = ANY(%s) ORDER BY schema_name",
                (allowed,),
            )
            rows = cur.fetchall()
    schemas = [row[0] for row in rows]
    return {"schemas": schemas}


@router.get("/introspect/tables")
def list_tables(schema: str = Query("public")) -> Dict[str, Any]:
    allowed = _allowed_schemas()
    if schema not in allowed:
        raise HTTPException(status_code=400, detail="Schema not allowlisted")
    cache_key = ("tables", schema, "")
    cached = _get_cached(cache_key)
    if cached:
        return cached
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
                LIMIT %s
                """,
                (schema, settings.topic_foundry_introspect_max_tables),
            )
            rows = cur.fetchall()
    tables = [{"table_name": row[0], "table_type": row[1]} for row in rows]
    payload = {"schema": schema, "tables": tables}
    _set_cached(cache_key, payload)
    return payload


@router.get("/introspect/columns")
def list_columns(schema: str = Query("public"), table: str = Query(...)) -> Dict[str, Any]:
    allowed = _allowed_schemas()
    if schema not in allowed:
        raise HTTPException(status_code=400, detail="Schema not allowlisted")
    if not _IDENTIFIER_RE.match(table):
        raise HTTPException(status_code=400, detail="Invalid table name")
    cache_key = ("cols", schema, table)
    cached = _get_cached(cache_key)
    if cached:
        return cached
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type, udt_name, is_nullable,
                       character_maximum_length, numeric_precision, numeric_scale, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                LIMIT %s
                """,
                (schema, table, settings.topic_foundry_introspect_max_columns),
            )
            rows = cur.fetchall()
    columns = [
        {
            "column_name": row[0],
            "data_type": row[1],
            "udt_name": row[2],
            "is_nullable": row[3] == "YES",
            "character_maximum_length": row[4],
            "numeric_precision": row[5],
            "numeric_scale": row[6],
            "ordinal_position": row[7],
        }
        for row in rows
    ]
    payload = {"schema": schema, "table": table, "columns": columns}
    _set_cached(cache_key, payload)
    return payload


@router.get("/introspect/table_fingerprint")
def table_fingerprint(schema: str = Query("public"), table: str = Query(...)) -> Dict[str, Any]:
    allowed = _allowed_schemas()
    if schema not in allowed:
        raise HTTPException(status_code=400, detail="Schema not allowlisted")
    if not _IDENTIFIER_RE.match(table):
        raise HTTPException(status_code=400, detail="Invalid table name")
    cache_key = ("fingerprint", schema, table)
    cached = _get_cached(cache_key)
    if cached:
        return cached
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            exists = cur.fetchone()[0] > 0
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (schema, table),
            )
            columns = [
                {"column_name": row[0], "data_type": row[1], "is_nullable": row[2] == "YES"}
                for row in cur.fetchall() or []
            ]
    payload = {"schema": schema, "table": table, "exists": exists, "columns": columns}
    _set_cached(cache_key, payload)
    return payload
