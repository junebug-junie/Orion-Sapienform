"""Load latest dream row from Postgres for wake readout (canonical path)."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional

from sqlalchemy import create_engine, text

from app.settings import settings

logger = logging.getLogger("dream-app.sql_readout")


def fetch_dream_row_from_sql(
    on_date: Optional[date] = None,
    *,
    use_latest_if_missing: bool = True,
) -> Optional[Dict[str, Any]]:
    uri = (settings.POSTGRES_URI or "").strip()
    if not uri:
        return None
    try:
        eng = create_engine(uri, pool_pre_ping=True)
        with eng.connect() as conn:
            if on_date is not None:
                row = conn.execute(
                    text(
                        "SELECT dream_date, tldr, themes, symbols, narrative "
                        "FROM dreams WHERE dream_date = CAST(:d AS DATE) "
                        "ORDER BY id DESC LIMIT 1"
                    ),
                    {"d": on_date.isoformat()},
                ).mappings().first()
                if row:
                    return dict(row)
                if not use_latest_if_missing:
                    return None
            row2 = conn.execute(
                text(
                    "SELECT dream_date, tldr, themes, symbols, narrative "
                    "FROM dreams ORDER BY created_at DESC NULLS LAST, id DESC LIMIT 1"
                )
            ).mappings().first()
            return dict(row2) if row2 else None
    except Exception as exc:
        logger.debug("SQL dream readout unavailable: %s", exc)
        return None
