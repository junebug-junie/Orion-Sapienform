from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2

from app.settings import settings


logger = logging.getLogger("orion-exo-exploration.pg")


@contextmanager
def pg_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    conn = psycopg2.connect(settings.exo_exploration_pg_dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
