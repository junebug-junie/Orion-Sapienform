"""Read substrate_turn_referent rows for reverie semantic lift."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import create_engine, text

from orion.schemas.reverie import ConcernCardV1

logger = logging.getLogger("orion.reverie.referent_loader")

_HARNESS_CLOSURE_RE = re.compile(r"^harness_closure:(?P<corr>.+)$")


def parse_harness_closure_ref(ref: str) -> str | None:
    m = _HARNESS_CLOSURE_RE.match((ref or "").strip())
    return m.group("corr") if m else None


@dataclass(frozen=True)
class TurnReferentRow:
    correlation_id: str
    coalition_ref: str
    user_message_excerpt: str
    stance_imperative: str
    created_at: datetime

    def to_concern_card(self, *, now: datetime | None = None) -> ConcernCardV1 | None:
        return ConcernCardV1.from_harness_turn(
            coalition_ref=self.coalition_ref,
            user_message_excerpt=self.user_message_excerpt,
            stance_imperative=self.stance_imperative,
            created_at=self.created_at,
            now=now,
        )


class ReferentLoader(Protocol):
    def load_by_correlation_id(self, correlation_id: str) -> TurnReferentRow | None: ...
    def load_by_coalition_ref(self, coalition_ref: str) -> TurnReferentRow | None: ...


def _database_url() -> str:
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def default_referent_loader(*, max_age_hours: float = 24.0) -> ReferentLoader:
    return SqlReferentLoader(max_age_hours=max_age_hours)


class SqlReferentLoader:
    def __init__(self, *, max_age_hours: float = 24.0) -> None:
        self.max_age_hours = max_age_hours
        self._engine = create_engine(_database_url(), pool_pre_ping=True)

    def _row_from_record(self, rec: dict) -> TurnReferentRow | None:
        created = rec.get("created_at")
        if not isinstance(created, datetime):
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_h = (now - created).total_seconds() / 3600.0
        if age_h > self.max_age_hours:
            return None
        return TurnReferentRow(
            correlation_id=str(rec["correlation_id"]),
            coalition_ref=str(rec["coalition_ref"]),
            user_message_excerpt=str(rec.get("user_message_excerpt") or ""),
            stance_imperative=str(rec.get("stance_imperative") or ""),
            created_at=created,
        )

    def load_by_correlation_id(self, correlation_id: str) -> TurnReferentRow | None:
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT correlation_id, coalition_ref, user_message_excerpt,
                               stance_imperative, created_at
                        FROM substrate_turn_referent
                        WHERE correlation_id = :cid
                        LIMIT 1
                        """
                    ),
                    {"cid": correlation_id},
                ).mappings().first()
            if not row:
                return None
            return self._row_from_record(dict(row))
        except Exception as exc:
            logger.warning("referent_load_failed cid=%s err=%s", correlation_id, exc)
            return None

    def load_by_coalition_ref(self, coalition_ref: str) -> TurnReferentRow | None:
        corr = parse_harness_closure_ref(coalition_ref)
        if corr:
            return self.load_by_correlation_id(corr)
        return None
