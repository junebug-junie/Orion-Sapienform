from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from app.models import DatasetCreateRequest, DatasetSpec
from app.storage.pg import pg_conn


logger = logging.getLogger("topic-foundry.data-access")

class InvalidSourceTableError(ValueError):
    pass


def split_qualified_table(name: str) -> Tuple[Optional[str], str]:
    cleaned = name.strip()
    if not cleaned:
        raise InvalidSourceTableError("source_table is empty")
    if any(ch in cleaned for ch in ('"', "'", ";")) or any(ch.isspace() for ch in cleaned):
        raise InvalidSourceTableError("source_table contains invalid characters")
    if cleaned.startswith(".") or cleaned.endswith("."):
        raise InvalidSourceTableError("source_table has invalid format")
    if cleaned.count(".") > 1:
        raise InvalidSourceTableError("source_table has too many qualifiers")
    if "." in cleaned:
        schema, table = cleaned.split(".", 1)
        if not schema or not table:
            raise InvalidSourceTableError("source_table has invalid format")
        return schema, table
    return None, cleaned


def _build_table_identifier(name: str) -> sql.Composable:
    schema, table = split_qualified_table(name)
    if schema:
        return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    return sql.Identifier(table)


def validate_dataset_source_table(dataset: DatasetCreateRequest | DatasetSpec) -> None:
    dataset_id = getattr(dataset, "dataset_id", None)
    try:
        table_ident = _build_table_identifier(dataset.source_table)
    except InvalidSourceTableError:
        logger.exception(
            "Invalid source_table dataset_id=%s source_table=%s",
            dataset_id,
            dataset.source_table,
        )
        raise
    query = sql.SQL("SELECT 1 FROM {table} LIMIT 1").format(table=table_ident)
    with pg_conn() as conn:
        with conn.cursor() as cur:
            try:
                query_str = query.as_string(conn)
                cur.execute(query)
            except Exception:
                logger.exception(
                    "Failed to validate source_table dataset_id=%s source_table=%s query=%s",
                    dataset_id,
                    dataset.source_table,
                    query_str if "query_str" in locals() else "(unavailable)",
                )
                raise


def fetch_dataset_rows(
    *,
    dataset: DatasetCreateRequest | DatasetSpec,
    start_at: Optional[datetime],
    end_at: Optional[datetime],
    limit: int,
) -> List[Dict[str, Any]]:
    fields: List[str] = [dataset.id_column, dataset.time_column] + list(dataset.text_columns)
    select_fields = sql.SQL(", ").join([sql.Identifier(field) for field in fields])
    dataset_id = getattr(dataset, "dataset_id", None)
    try:
        table_ident = _build_table_identifier(dataset.source_table)
    except InvalidSourceTableError:
        logger.exception(
            "Invalid source_table dataset_id=%s source_table=%s",
            dataset_id,
            dataset.source_table,
        )
        raise
    query = sql.SQL("SELECT {fields} FROM {table}").format(fields=select_fields, table=table_ident)

    where_clauses: List[sql.SQL] = []
    params: Dict[str, Any] = {}
    if start_at is not None:
        where_clauses.append(sql.SQL("{col} >= %(start_at)s").format(col=sql.Identifier(dataset.time_column)))
        params["start_at"] = start_at
    if end_at is not None:
        where_clauses.append(sql.SQL("{col} <= %(end_at)s").format(col=sql.Identifier(dataset.time_column)))
        params["end_at"] = end_at
    if dataset.where_sql:
        where_clauses.append(sql.SQL(dataset.where_sql))
        if dataset.where_params:
            params.update(dataset.where_params)

    if where_clauses:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)

    query = query + sql.SQL(" ORDER BY {time_col} ASC, {id_col} ASC").format(
        time_col=sql.Identifier(dataset.time_column),
        id_col=sql.Identifier(dataset.id_column),
    )
    query = query + sql.SQL(" LIMIT {limit}").format(limit=sql.Literal(int(limit)))

    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                query_str = query.as_string(conn)
                cur.execute(query, params)
                rows = cur.fetchall() or []
            except Exception:
                logger.exception(
                    "Failed to fetch dataset rows dataset_id=%s source_table=%s query=%s",
                    dataset_id,
                    dataset.source_table,
                    query_str if "query_str" in locals() else "(unavailable)",
                )
                raise
    return rows
