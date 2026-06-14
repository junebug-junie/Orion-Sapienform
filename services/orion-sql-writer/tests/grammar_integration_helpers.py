"""Shared Postgres fixtures for grammar ledger integration tests."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAMMAR_MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_grammar_atlas.sql"
SUBSTRATE_CURSOR_MIGRATION = (
    REPO_ROOT / "services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql"
)


def postgres_uri() -> str:
    return os.environ.get(
        "POSTGRES_URI",
        "postgresql://postgres:postgres@localhost:5432/conjourney",
    )


def _is_executable_sql(stmt: str) -> bool:
    for line in stmt.splitlines():
        line = line.strip()
        if line and not line.startswith("--"):
            return True
    return False


def apply_sql_file(engine: Engine, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    statements: list[str] = []
    for chunk in sql.split(";"):
        stmt = chunk.strip()
        if stmt and _is_executable_sql(stmt):
            statements.append(stmt)
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def ensure_grammar_schema(engine: Engine) -> None:
    apply_sql_file(engine, GRAMMAR_MIGRATION)
    apply_sql_file(engine, SUBSTRATE_CURSOR_MIGRATION)


def grammar_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def bus_transport_trace_batch(*, trace_suffix: str, event_count: int = 13) -> list[GrammarEventV1]:
    """Minimal orion-bus transport trace matching typical 13-event observer rollups."""
    if event_count < 2:
        raise ValueError("event_count must be at least 2")
    now = datetime.now(timezone.utc)
    trace_id = f"bus.transport:athena:ci{trace_suffix}"
    prov = GrammarProvenanceV1(
        source_service="orion-bus",
        source_component="bus_transport_grammar_emit",
        source_event_id=f"ci:{trace_suffix}",
    )
    events: list[GrammarEventV1] = [
        GrammarEventV1(
            event_id=f"gev_{trace_suffix}_000",
            event_kind="trace_started",
            trace_id=trace_id,
            emitted_at=now,
            observed_at=now,
            layer="transport",
            dimensions=["bus", "transport"],
            provenance=prov,
        )
    ]
    for idx in range(1, event_count - 1):
        atom = GrammarAtomV1(
            atom_id=f"atom_{trace_suffix}_{idx:03d}",
            trace_id=trace_id,
            atom_type="observation",
            semantic_role="transport",
            layer="transport",
            summary=f"ci transport observation {idx}",
        )
        events.append(
            GrammarEventV1(
                event_id=f"gev_{trace_suffix}_{idx:03d}",
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=now,
                observed_at=now,
                layer="transport",
                dimensions=["bus", "transport"],
                atom=atom,
                provenance=prov,
            )
        )
    events.append(
        GrammarEventV1(
            event_id=f"gev_{trace_suffix}_{event_count - 1:03d}",
            event_kind="trace_ended",
            trace_id=trace_id,
            emitted_at=now,
            observed_at=now,
            layer="transport",
            dimensions=["bus", "transport"],
            provenance=prov,
        )
    )
    return events


def delete_trace(engine: Engine, trace_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM grammar_atoms WHERE trace_id = :trace_id"), {"trace_id": trace_id})
        conn.execute(text("DELETE FROM grammar_edges WHERE trace_id = :trace_id"), {"trace_id": trace_id})
        conn.execute(text("DELETE FROM grammar_events WHERE trace_id = :trace_id"), {"trace_id": trace_id})
        conn.execute(text("DELETE FROM grammar_traces WHERE trace_id = :trace_id"), {"trace_id": trace_id})


def load_apply_grammar_trace_batch():
    """Import ledger batch helper with sql-writer models on sys.path."""
    writer_root = REPO_ROOT / "services/orion-sql-writer"
    paths = [str(writer_root), str(REPO_ROOT)]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    from orion.grammar.ledger import apply_grammar_trace_batch

    return apply_grammar_trace_batch


def _clear_app_namespace() -> None:
    """Drop cached app.* modules so a service-specific app package can load."""
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]


def _register_service_app_package(service_root: Path) -> None:
    import types

    service_root_str = str(service_root)
    if service_root_str not in sys.path:
        sys.path.insert(0, service_root_str)
    app_dir = service_root / "app"
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(app_dir)]  # type: ignore[attr-defined]
    app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg


def load_biometrics_substrate_store_class():
    """Import substrate store without colliding with sql-writer's app package."""
    substrate_root = REPO_ROOT / "services/orion-substrate-runtime"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _clear_app_namespace()
    _register_service_app_package(substrate_root)
    from app.store import BiometricsSubstrateStore

    return BiometricsSubstrateStore


def assert_grammar_event_indexes_valid(engine: Engine) -> None:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT c.relname AS index_name, i.indisvalid AS is_valid
                FROM pg_index i
                JOIN pg_class c ON c.oid = i.indexrelid
                JOIN pg_class t ON t.oid = i.indrelid
                WHERE t.relname = 'grammar_events'
                ORDER BY c.relname
                """
            )
        ).mappings().all()
    assert rows, "expected grammar_events indexes"
    invalid = [row["index_name"] for row in rows if not row["is_valid"]]
    assert not invalid, f"invalid grammar_events indexes: {invalid}"
