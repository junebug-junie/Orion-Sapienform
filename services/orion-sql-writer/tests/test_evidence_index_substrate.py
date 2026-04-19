from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy.dialects import postgresql

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.evidence_index.ingest import build_evidence_units
from orion.schemas.evidence_index import EvidenceQueryV1, EvidenceUnitV1
from app.evidence_index_repository import build_evidence_query_select, expand_evidence_context, query_evidence_units

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_evidence_index_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_evidence_unit_contract_validates() -> None:
    model = EvidenceUnitV1(
        unit_id="u1",
        unit_kind="journal_entry",
        source_family="journal",
        source_kind="manual",
        source_ref="ref-1",
        created_at=datetime(2026, 4, 14, tzinfo=timezone.utc),
    )
    assert model.unit_id == "u1"


def test_journal_adapter_produces_expected_unit() -> None:
    payload = {
        "entry_id": str(uuid4()),
        "created_at": "2026-04-14T00:00:00+00:00",
        "author": "orion",
        "mode": "manual",
        "title": "Daily note",
        "body": "Journal body text",
        "source_kind": "manual",
        "source_ref": "jrnl-1",
        "correlation_id": "corr-j1",
    }
    units = build_evidence_units("journal.entry.write.v1", {"payload": payload})
    assert len(units) == 1
    assert units[0].source_family == "journal"
    assert "mode:manual" in units[0].facets


def test_collapse_adapter_produces_expected_unit() -> None:
    units = build_evidence_units(
        "collapse.mirror",
        {
            "id": "cm-1",
            "correlation_id": "corr-c1",
            "observer": "orion",
            "source_service": "orion-cortex-exec",
            "summary": "short collapse summary",
            "what_changed_summary": "state shifted",
            "type": "flow",
            "timestamp": "2026-04-14T01:00:00+00:00",
        },
    )
    assert len(units) == 1
    assert units[0].unit_kind == "collapse_mirror_entry"
    assert units[0].source_kind == "flow"


def test_filtered_retrieval_and_lexical_sql_shape() -> None:
    query = EvidenceQueryV1(
        text_query="shifted state",
        unit_kinds=["collapse_mirror_entry"],
        source_family=["collapse_mirror"],
        source_kind=["flow"],
        source_ref="orion-cortex-exec",
        correlation_id="corr-c1",
        required_facets=["type:flow"],
        limit=10,
        offset=5,
    )
    stmt = build_evidence_query_select(query)
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "FROM evidence_units" in sql
    assert "evidence_units.unit_kind IN" in sql
    assert "ILIKE" in sql
    assert "ORDER BY evidence_units.created_at DESC" in sql


def test_context_expansion_parent_children_siblings() -> None:
    current = SimpleNamespace(unit_id="u2", parent_unit_id="u1")
    parent = SimpleNamespace(unit_id="u1", parent_unit_id=None)
    child = SimpleNamespace(unit_id="u3", parent_unit_id="u2")
    sibling = SimpleNamespace(unit_id="u4", parent_unit_id="u1")

    class _Scalars:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def scalars(self):
            return _Scalars(self._rows)

    class FakeSession:
        def get(self, _model, key):
            if key == "u2":
                return current
            if key == "u1":
                return parent
            return None

        def execute(self, stmt):
            sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
            if "parent_unit_id = 'u2'" in sql:
                return _Result([child])
            if "parent_unit_id = 'u1'" in sql and "unit_id !=" in sql:
                return _Result([sibling])
            return _Result([])

    context = expand_evidence_context(FakeSession(), "u2", include_siblings=True)
    assert context["current"].unit_id == "u2"
    assert context["parent"].unit_id == "u1"
    assert [row.unit_id for row in context["children"]] == ["u3"]
    assert [row.unit_id for row in context["siblings"]] == ["u4"]


def test_markdown_adapter_emits_document_section_and_leaf_hierarchy() -> None:
    markdown_payload = {
        "doc_id": "doc-1",
        "title": "Evidence Spec",
        "body": "# Evidence Spec\n## Goals\nWe need a hierarchy.\n\n```python\nprint('ok')\n```\n\n## Data\n| key | value |\n| --- | --- |\n| a | b |",
        "source_ref": "spec://evidence",
        "source_kind": "markdown_spec",
        "correlation_id": "corr-doc-1",
        "created_at": "2026-04-18T00:00:00+00:00",
    }
    units = build_evidence_units("document.markdown.spec.v1", markdown_payload)

    doc_units = [u for u in units if u.unit_kind == "document"]
    section_units = [u for u in units if u.unit_kind == "document_section"]
    leaf_units = [u for u in units if u.unit_kind == "document_leaf"]

    assert len(doc_units) == 1
    assert len(section_units) >= 2
    assert len(leaf_units) >= 2
    assert all(section.parent_unit_id == doc_units[0].unit_id for section in section_units)
    assert all(leaf.parent_unit_id and leaf.parent_unit_id.startswith("doc-1::section::") for leaf in leaf_units)
    assert any(leaf.sibling_next_id is not None or leaf.sibling_prev_id is not None for leaf in leaf_units)


def test_markdown_adapter_only_path_writes_evidence_units(monkeypatch) -> None:
    writes: list[tuple[str, dict]] = []

    def _fake_write_row(sql_model_cls, data):
        writes.append((sql_model_cls.__tablename__, data))
        return True

    monkeypatch.setattr(worker, "_write_row", _fake_write_row)

    env = worker.BaseEnvelope(
        kind="document.markdown.spec.v1",
        source=worker.ServiceRef(name="doc-test", version="0.0.1", node="local"),
        payload={
            "doc_id": "doc-2",
            "title": "Doc Two",
            "body": "# Doc Two\n## Section A\nalpha paragraph",
            "source_ref": "spec://doc-two",
            "source_kind": "markdown_spec",
            "created_at": "2026-04-18T00:00:00+00:00",
        },
    )

    import asyncio

    asyncio.run(worker.handle_envelope(env, bus=None))
    assert "evidence_units" in [table for table, _ in writes]


def test_markdown_section_search_sql_shape_uses_title_summary_body() -> None:
    query = EvidenceQueryV1(
        text_query="hierarchy",
        unit_kinds=["document_section"],
        source_family=["markdown_spec"],
        limit=25,
    )
    stmt = build_evidence_query_select(query)
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "evidence_units.unit_kind IN" in sql
    assert "evidence_units.title ILIKE" in sql
    assert "evidence_units.summary ILIKE" in sql
    assert "evidence_units.body ILIKE" in sql


def test_section_only_and_leaf_only_search_level_filters() -> None:
    section_sql = str(
        build_evidence_query_select(
            EvidenceQueryV1(search_level="section", text_query="architecture")
        ).compile(dialect=postgresql.dialect())
    )
    leaf_sql = str(
        build_evidence_query_select(
            EvidenceQueryV1(search_level="leaf", text_query="code")
        ).compile(dialect=postgresql.dialect())
    )
    assert "evidence_units.unit_kind = %(unit_kind_1)s" in section_sql
    assert "evidence_units.unit_kind = %(unit_kind_1)s" in leaf_sql


def test_auto_search_with_parent_and_child_expansion_and_explanations() -> None:
    doc = SimpleNamespace(
        unit_id="doc-9",
        unit_kind="document",
        source_family="markdown_spec",
        source_kind="markdown_spec",
        source_ref="spec://9",
        correlation_id="corr-9",
        title="Doc Title",
        summary="Parent summary",
        body="Parent body",
        facets=["artifact:document"],
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        parent_unit_id=None,
    )
    section = SimpleNamespace(
        unit_id="doc-9::section::1",
        unit_kind="document_section",
        source_family="markdown_spec",
        source_kind="markdown_spec",
        source_ref="spec://9",
        correlation_id="corr-9",
        title="Goals",
        summary="Hierarchy summary",
        body="Hierarchy details body",
        facets=["artifact:section", "heading_level:2"],
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        parent_unit_id="doc-9",
    )
    leaf = SimpleNamespace(
        unit_id="doc-9::section::1::leaf::1",
        unit_kind="document_leaf",
        source_family="markdown_spec",
        source_kind="markdown_spec",
        source_ref="spec://9",
        correlation_id="corr-9",
        title="Goals [paragraph]",
        summary="Hierarchy leaf",
        body="Hierarchy leaf body",
        facets=["artifact:leaf", "block_type:paragraph"],
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        parent_unit_id="doc-9::section::1",
    )

    class _Scalars:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def scalars(self):
            return _Scalars(self._rows)

    class FakeSession:
        def execute(self, stmt):
            sql = str(stmt.compile(dialect=postgresql.dialect()))
            if "parent_unit_id = %(parent_unit_id_1)s" in sql:
                return _Result([leaf])
            if "FROM evidence_units" in sql:
                return _Result([section])
            return _Result([])

        def get(self, _model, key):
            if key == "doc-9":
                return doc
            return None

    results = query_evidence_units(
        FakeSession(),
        EvidenceQueryV1(
            text_query="Hierarchy",
            source_family=["markdown_spec"],
            search_level="auto",
            include_parent_context=True,
            include_child_context=True,
            required_facets=["artifact:section"],
        ),
    )
    assert len(results) == 1
    result = results[0]
    assert "summary" in result.matched_fields or "body" in result.matched_fields
    assert "source_family" in result.applied_filters
    assert result.parent_context and result.parent_context["unit_id"] == "doc-9"
    assert result.child_context and result.child_context[0]["unit_id"] == "doc-9::section::1::leaf::1"


def test_notify_adapter_ingestion_and_retrieval_shape() -> None:
    units = build_evidence_units(
        "notify.notification.request.v1",
        {
            "notification_id": "4f29e050-2d32-4da5-9bc1-4ed85cc80f8a",
            "source_service": "orion-notify",
            "event_kind": "ops.alert",
            "severity": "warning",
            "title": "Queue backed up",
            "body_text": "Queue depth exceeded threshold",
            "recipient_group": "juniper_primary",
            "created_at": "2026-04-19T01:00:00+00:00",
            "correlation_id": "corr-notify-1",
        },
    )
    assert len(units) == 1
    assert units[0].unit_kind == "notify_event"
    assert units[0].source_family == "notify_output"

    query = EvidenceQueryV1(search_level="auto", source_family=["notify_output"], text_query="Queue")
    sql = str(build_evidence_query_select(query).compile(dialect=postgresql.dialect()))
    assert "evidence_units.source_family IN" in sql
    assert "ILIKE" in sql


def test_parsed_document_adapter_hierarchical_emission_and_page_metadata() -> None:
    payload = {
        "doc_id": "parsed-1",
        "title": "Parsed Ops Runbook",
        "source_ref": "s3://docs/runbook.pdf#parsed",
        "source_kind": "pdf_parsed",
        "correlation_id": "corr-parsed-1",
        "created_at": "2026-04-19T02:00:00+00:00",
        "source_provenance": {"parser": "layout-v1", "source_uri": "s3://docs/runbook.pdf"},
        "sections": [
            {
                "section_id": "s1",
                "title": "Checklist",
                "summary": "Start checklist",
                "body": "Run health checks.",
                "page_start": 2,
                "page_end": 3,
                "heading_path": ["Runbook", "Checklist"],
                "blocks": [
                    {
                        "block_id": "b1",
                        "block_type": "paragraph",
                        "body": "Verify services are up.",
                        "page_start": 2,
                        "page_end": 2,
                        "heading_path": ["Runbook", "Checklist"],
                        "source_provenance": {"bbox": [0, 0, 100, 100]},
                    },
                    {
                        "block_id": "b2",
                        "block_type": "table",
                        "body": "| check | status |",
                        "page_start": 3,
                        "page_end": 3,
                        "heading_path": ["Runbook", "Checklist"],
                    },
                ],
            }
        ],
    }
    units = build_evidence_units("document.parsed.v1", payload)
    doc = [u for u in units if u.unit_kind == "document"][0]
    section = [u for u in units if u.unit_kind == "document_section"][0]
    leaves = [u for u in units if u.unit_kind == "document_leaf"]

    assert section.parent_unit_id == doc.unit_id
    assert len(leaves) == 2
    assert all(leaf.parent_unit_id == section.unit_id for leaf in leaves)
    assert doc.metadata["page_span"]["page_start"] == 2
    assert leaves[0].metadata["page_start"] == 2
    assert leaves[1].metadata["page_start"] == 3


def test_parsed_leaf_hit_with_parent_expansion() -> None:
    parent_section = SimpleNamespace(
        unit_id="parsed-1::section::s1",
        unit_kind="document_section",
        source_family="parsed_document",
        source_kind="pdf_parsed",
        source_ref="s3://docs/runbook.pdf#parsed",
        correlation_id="corr-parsed-1",
        title="Checklist",
        summary="Section summary",
        body="Section body",
        facets=["artifact:section", "block_type:table"],
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        parent_unit_id="parsed-1",
    )
    leaf = SimpleNamespace(
        unit_id="parsed-1::section::s1::leaf::b2",
        unit_kind="document_leaf",
        source_family="parsed_document",
        source_kind="pdf_parsed",
        source_ref="s3://docs/runbook.pdf#parsed",
        correlation_id="corr-parsed-1",
        title="Checklist [table]",
        summary="table summary",
        body="| check | status |",
        facets=["artifact:leaf", "block_type:table"],
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        parent_unit_id="parsed-1::section::s1",
    )

    class _Scalars:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def scalars(self):
            return _Scalars(self._rows)

    class FakeSession:
        def execute(self, stmt):
            sql = str(stmt.compile(dialect=postgresql.dialect()))
            if "FROM evidence_units" in sql:
                return _Result([leaf])
            return _Result([])

        def get(self, _model, key):
            if key == "parsed-1::section::s1":
                return parent_section
            return None

    results = query_evidence_units(
        FakeSession(),
        EvidenceQueryV1(
            search_level="leaf",
            text_query="status",
            source_family=["parsed_document"],
            include_parent_context=True,
        ),
    )
    assert len(results) == 1
    assert results[0].parent_context and results[0].parent_context["unit_id"] == "parsed-1::section::s1"
    assert "body" in results[0].matched_fields


def test_parsed_section_search_with_block_type_filtering_sql_shape() -> None:
    query = EvidenceQueryV1(
        search_level="section",
        source_family=["parsed_document"],
        required_facets=["block_type:table"],
        text_query="Checklist",
    )
    sql = str(build_evidence_query_select(query).compile(dialect=postgresql.dialect()))
    assert "evidence_units.unit_kind = %(unit_kind_1)s" in sql
    assert "evidence_units.source_family IN" in sql
    assert "evidence_units.facets" in sql
    assert "ILIKE" in sql
