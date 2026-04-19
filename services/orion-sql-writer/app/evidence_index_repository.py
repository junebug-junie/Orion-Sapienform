from __future__ import annotations

from typing import Any

from sqlalchemy import Select, and_, desc, or_, select
from sqlalchemy.orm import Session

from app.models import EvidenceUnitSQL
from orion.schemas.evidence_index import EvidenceQueryResultItemV1, EvidenceQueryV1


def _tokens(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    return [token for token in text.strip().split() if token][:8]


def build_evidence_query_select(query: EvidenceQueryV1) -> Select:
    stmt = select(EvidenceUnitSQL)
    clauses = []

    if query.unit_kinds:
        clauses.append(EvidenceUnitSQL.unit_kind.in_(query.unit_kinds))
    if query.search_level == "document":
        clauses.append(EvidenceUnitSQL.unit_kind == "document")
    elif query.search_level == "section":
        clauses.append(EvidenceUnitSQL.unit_kind == "document_section")
    elif query.search_level == "leaf":
        clauses.append(EvidenceUnitSQL.unit_kind == "document_leaf")
    if query.source_family:
        clauses.append(EvidenceUnitSQL.source_family.in_(query.source_family))
    if query.source_kind:
        clauses.append(EvidenceUnitSQL.source_kind.in_(query.source_kind))
    if query.source_ref:
        clauses.append(EvidenceUnitSQL.source_ref == query.source_ref)
    if query.correlation_id:
        clauses.append(EvidenceUnitSQL.correlation_id == query.correlation_id)
    if query.created_after is not None:
        clauses.append(EvidenceUnitSQL.created_at >= query.created_after)
    if query.created_before is not None:
        clauses.append(EvidenceUnitSQL.created_at <= query.created_before)

    for facet in [f.strip() for f in query.required_facets if f.strip()]:
        clauses.append(EvidenceUnitSQL.facets.contains([facet]))

    text_tokens = _tokens(query.text_query)
    if text_tokens:
        text_clauses = []
        for token in text_tokens:
            needle = f"%{token}%"
            text_clauses.append(
                or_(
                    EvidenceUnitSQL.title.ilike(needle),
                    EvidenceUnitSQL.summary.ilike(needle),
                    EvidenceUnitSQL.body.ilike(needle),
                )
            )
        clauses.append(and_(*text_clauses))

    if clauses:
        stmt = stmt.where(and_(*clauses))

    return stmt.order_by(desc(EvidenceUnitSQL.created_at)).limit(query.limit).offset(query.offset)


def query_evidence_units(session: Session, query: EvidenceQueryV1) -> list[EvidenceQueryResultItemV1]:
    rows = session.execute(build_evidence_query_select(query)).scalars().all()
    out: list[EvidenceQueryResultItemV1] = []
    tokens = _tokens(query.text_query)
    applied_filters = []
    if query.unit_kinds:
        applied_filters.append("unit_kinds")
    if query.search_level != "auto":
        applied_filters.append(f"search_level:{query.search_level}")
    if query.source_family:
        applied_filters.append("source_family")
    if query.source_kind:
        applied_filters.append("source_kind")
    if query.source_ref:
        applied_filters.append("source_ref")
    if query.correlation_id:
        applied_filters.append("correlation_id")
    if query.required_facets:
        applied_filters.append("required_facets")
    if query.created_after is not None or query.created_before is not None:
        applied_filters.append("created_range")

    def _matched_fields(row: EvidenceUnitSQL) -> list[str]:
        if not tokens:
            return []
        matched: list[str] = []
        text_fields = {
            "title": (row.title or "").lower(),
            "summary": (row.summary or "").lower(),
            "body": (row.body or "").lower(),
        }
        for field, value in text_fields.items():
            if any(token.lower() in value for token in tokens):
                matched.append(field)
        return matched

    def _compact(row: EvidenceUnitSQL) -> dict[str, Any]:
        return {
            "unit_id": row.unit_id,
            "unit_kind": row.unit_kind,
            "title": row.title,
            "summary": row.summary,
            "created_at": row.created_at,
        }

    for row in rows:
        matched_facets = [facet for facet in query.required_facets if facet in (row.facets or [])]
        parent_context = None
        if query.include_parent_context and row.parent_unit_id:
            parent = session.get(EvidenceUnitSQL, row.parent_unit_id)
            if parent is not None:
                parent_context = _compact(parent)
        child_context: list[dict[str, Any]] = []
        if query.include_child_context:
            children = session.execute(
                select(EvidenceUnitSQL)
                .where(EvidenceUnitSQL.parent_unit_id == row.unit_id)
                .order_by(desc(EvidenceUnitSQL.created_at))
                .limit(10)
            ).scalars().all()
            child_context = [_compact(child) for child in children]
        out.append(
            EvidenceQueryResultItemV1(
                unit_id=row.unit_id,
                unit_kind=row.unit_kind,
                source_kind=row.source_kind,
                source_ref=row.source_ref,
                title=row.title,
                summary=row.summary,
                created_at=row.created_at,
                matched_facets=matched_facets,
                matched_fields=_matched_fields(row),
                applied_filters=list(applied_filters),
                provenance={
                    "source_family": row.source_family,
                    "source_kind": row.source_kind,
                    "source_ref": row.source_ref,
                    "correlation_id": row.correlation_id,
                },
                parent_context=parent_context,
                child_context=child_context,
            )
        )
    return out


def get_evidence_unit(session: Session, unit_id: str) -> EvidenceUnitSQL | None:
    return session.get(EvidenceUnitSQL, unit_id)


def expand_evidence_context(session: Session, unit_id: str, *, include_siblings: bool = True) -> dict[str, Any]:
    current = session.get(EvidenceUnitSQL, unit_id)
    if current is None:
        return {"current": None, "parent": None, "children": [], "siblings": []}

    parent = session.get(EvidenceUnitSQL, current.parent_unit_id) if current.parent_unit_id else None
    children = session.execute(
        select(EvidenceUnitSQL)
        .where(EvidenceUnitSQL.parent_unit_id == current.unit_id)
        .order_by(desc(EvidenceUnitSQL.created_at))
    ).scalars().all()

    siblings: list[EvidenceUnitSQL] = []
    if include_siblings and current.parent_unit_id:
        siblings = session.execute(
            select(EvidenceUnitSQL)
            .where(EvidenceUnitSQL.parent_unit_id == current.parent_unit_id)
            .where(EvidenceUnitSQL.unit_id != current.unit_id)
            .order_by(desc(EvidenceUnitSQL.created_at))
        ).scalars().all()

    return {"current": current, "parent": parent, "children": children, "siblings": siblings}
