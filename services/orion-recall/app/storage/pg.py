# app/storage/pg.py
from __future__ import annotations

import psycopg2
from app.settings import settings


def get_pg_connection():
    """
    Simple helper in case we want a shared way to connect later.
    Currently sql_adapter uses its own _connect(), but this keeps
    the file from being an empty stub.
    """
    return psycopg2.connect(settings.RECALL_PG_DSN)
