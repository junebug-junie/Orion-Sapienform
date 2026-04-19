from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Select, and_, desc, or_, select
from sqlalchemy.orm import Session

from app.models import JournalEntryIndexSQL

_FACET_FIELDS = (
    "active_identity_facets",
    "active_growth_axes",
    "active_relationship_facets",
    "social_posture",
    "reflective_themes",
    "active_tensions",
    "dream_motifs",
    "response_hazards",
)


def _tokens(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    return [token for token in text.strip().split() if token][:8]


def build_journal_entry_index_select(
    *,
    mode: str | None = None,
    source_kind: str | None = None,
    trigger_kind: str | None = None,
    author: str | None = None,
    source_ref: str | None = None,
    correlation_id: str | None = None,
    created_at_gte: datetime | None = None,
    created_at_lte: datetime | None = None,
    facets: dict[str, list[str]] | None = None,
    text_query: str | None = None,
    limit: int = 50,
) -> Select:
    stmt = select(JournalEntryIndexSQL)
    clauses = []

    if mode:
        clauses.append(JournalEntryIndexSQL.mode == mode)
    if source_kind:
        clauses.append(JournalEntryIndexSQL.source_kind == source_kind)
    if trigger_kind:
        clauses.append(JournalEntryIndexSQL.trigger_kind == trigger_kind)
    if author:
        clauses.append(JournalEntryIndexSQL.author == author)
    if source_ref:
        clauses.append(JournalEntryIndexSQL.source_ref == source_ref)
    if correlation_id:
        clauses.append(JournalEntryIndexSQL.correlation_id == correlation_id)
    if created_at_gte is not None:
        clauses.append(JournalEntryIndexSQL.created_at >= created_at_gte)
    if created_at_lte is not None:
        clauses.append(JournalEntryIndexSQL.created_at <= created_at_lte)

    facets = facets or {}
    for field in _FACET_FIELDS:
        values = [str(v).strip() for v in (facets.get(field) or []) if str(v).strip()]
        for value in values:
            clauses.append(getattr(JournalEntryIndexSQL, field).contains([value]))

    token_clauses = []
    for token in _tokens(text_query):
        needle = f"%{token}%"
        token_clauses.append(
            or_(
                JournalEntryIndexSQL.title.ilike(needle),
                JournalEntryIndexSQL.body.ilike(needle),
                JournalEntryIndexSQL.trigger_summary.ilike(needle),
                JournalEntryIndexSQL.stance_summary.ilike(needle),
            )
        )

    if clauses:
        stmt = stmt.where(and_(*clauses))
    if token_clauses:
        stmt = stmt.where(and_(*token_clauses))

    return stmt.order_by(desc(JournalEntryIndexSQL.created_at)).limit(limit)


def query_journal_entry_index(session: Session, **kwargs: Any) -> list[dict[str, Any]]:
    stmt = build_journal_entry_index_select(**kwargs)
    rows = session.execute(stmt).scalars().all()
    return [
        {
            "entry_id": row.entry_id,
            "created_at": row.created_at,
            "author": row.author,
            "mode": row.mode,
            "source_kind": row.source_kind,
            "source_ref": row.source_ref,
            "correlation_id": row.correlation_id,
            "trigger_kind": row.trigger_kind,
            "trigger_summary": row.trigger_summary,
            "stance_summary": row.stance_summary,
        }
        for row in rows
    ]
