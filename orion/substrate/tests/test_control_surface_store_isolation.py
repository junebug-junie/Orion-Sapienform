"""An explicit sqlite path must not be overridden by an ambient Postgres URL.

Found live 2026-08-30: `substrate_runtime_control_surface` in production Postgres
holds `routing.chat_reflective_lane_threshold = 0.5` written by actor
"scheduler_seed" -- a string that appears nowhere in the repo except a pytest
fixture -- with 4,925 updates recorded. Test runs asked for isolation by passing
`sql_db_path=<tmp>`, and `__post_init__` resolved `DATABASE_URL` out of the
ambient environment anyway, so they wrote Orion's live routing threshold.
"""

from __future__ import annotations

from orion.substrate.mutation_control_surface import RuntimeControlSurfaceStore


def test_explicit_sqlite_path_is_not_overridden_by_ambient_postgres(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://should-never-be-used@127.0.0.1:1/nope")
    store = RuntimeControlSurfaceStore(sql_db_path=str(tmp_path / "isolated.sqlite3"))
    assert store.source_kind() == "sqlite"
    assert store.postgres_url is None


def test_explicit_postgres_url_still_wins_when_both_are_passed(tmp_path, monkeypatch) -> None:
    """The guard is scoped to an ambient URL; an explicitly-passed one is a
    deliberate choice and must be honoured."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = RuntimeControlSurfaceStore(
        postgres_url="postgresql://postgres:postgres@127.0.0.1:1/unreachable",
        sql_db_path=str(tmp_path / "fallback.sqlite3"),
    )
    # Unreachable, so it falls through to sqlite -- but it TRIED postgres first,
    # which is the behaviour being preserved.
    assert store.postgres_url is not None
    assert store.last_error() is not None


def test_ambient_postgres_is_still_used_when_no_sqlite_path_is_requested(monkeypatch) -> None:
    """Production has no explicit sql_db_path, so the env must still resolve --
    otherwise this fix would silently disconnect every real service."""
    monkeypatch.delenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", raising=False)
    monkeypatch.delenv("SUBSTRATE_POLICY_POSTGRES_URL", raising=False)
    monkeypatch.delenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", raising=False)
    monkeypatch.delenv("SUBSTRATE_MUTATION_SQL_DB_PATH", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/unreachable")
    store = RuntimeControlSurfaceStore()
    assert store.postgres_url == "postgresql://postgres:postgres@127.0.0.1:1/unreachable"
