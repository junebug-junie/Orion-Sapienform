from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from app.models import DatasetCreateRequest, DatasetSpec
from app.storage.pg import pg_conn


logger = logging.getLogger("topic-foundry.data-access")


def fetch_dataset_rows(
    *,
    dataset: DatasetCreateRequest | DatasetSpec,
    start_at: Optional[datetime],
    end_at: Optional[datetime],
    limit: int,
) -> List[Dict[str, Any]]:
    fields: List[str] = [dataset.id_column, dataset.time_column] + list(dataset.text_columns)
    select_fields = sql.SQL(", ").join([sql.Identifier(field) for field in fields])
    query = sql.SQL("SELECT {fields} FROM {table}").format(
        fields=select_fields,
        table=sql.Identifier(dataset.source_table),
    )

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
            cur.execute(query, params)
            rows = cur.fetchall() or []
    return rows
