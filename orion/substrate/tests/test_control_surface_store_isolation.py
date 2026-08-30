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
    # `postgres_url is None` is the load-bearing assertion. Asserting only
    # source_kind == "sqlite" would pass under a full revert of this fix too,
    # because the unreachable URL falls through to sqlite anyway -- the store
    # would have TRIED production and merely failed to reach it.
    assert store.postgres_url is None
    assert store.source_kind() == "sqlite"


def test_explicit_postgres_url_is_kept_when_both_are_passed(tmp_path, monkeypatch) -> None:
    """The guard keys off sql_db_path, so an explicitly-passed postgres_url must
    survive it rather than being dropped along with the ambient ones."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    explicit = "postgresql://postgres:postgres@127.0.0.1:1/unreachable"
    store = RuntimeControlSurfaceStore(
        postgres_url=explicit,
        sql_db_path=str(tmp_path / "fallback.sqlite3"),
    )
    # Deterministic and load-bearing: the URL is still there. Asserting on
    # last_error() instead would pass for the wrong reason on a machine without
    # SQLAlchemy, where the error is ImportError rather than a failed connection.
    assert store.postgres_url == explicit


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


class TestHotPathEngine:
    """`decision_router.py` calls get_chat_reflective_lane_threshold() on EVERY
    routing decision, so the Postgres path is on the chat hot path."""

    def test_engine_is_built_once_and_reused(self) -> None:
        """A per-call create_engine() builds a fresh pool and a fresh TCP connect
        plus auth handshake per routing decision."""
        store = RuntimeControlSurfaceStore(postgres_url="postgresql://u:p@127.0.0.1:1/db")
        first = store._engine()
        assert store._engine() is first

    def test_connect_timeout_is_passed_to_create_engine(self, monkeypatch) -> None:
        """SQLAlchemy has no default connect timeout: against a host that is
        unreachable-but-not-refusing, the routing path would block on the OS TCP
        timeout, which can be minutes.

        Asserts on what this module passes, not on SQLAlchemy internals --
        connect_args are merged by the pool at connect time and are not readable
        back off the Engine.
        """
        import sqlalchemy

        captured: dict = {}

        def _fake_create_engine(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(sqlalchemy, "create_engine", _fake_create_engine)
        store = RuntimeControlSurfaceStore(postgres_url="postgresql://u:p@127.0.0.1:1/db")
        store._engine()

        assert captured["connect_args"]["connect_timeout"] == 5
        assert captured["pool_pre_ping"] is True

    def test_timeout_is_clamped_not_trusted(self, monkeypatch) -> None:
        from orion.substrate import mutation_control_surface as mcs

        monkeypatch.setenv("SUBSTRATE_CONTROL_SURFACE_CONNECT_TIMEOUT_SEC", "0")
        assert mcs._connect_timeout_sec() == 1
        monkeypatch.setenv("SUBSTRATE_CONTROL_SURFACE_CONNECT_TIMEOUT_SEC", "9999")
        assert mcs._connect_timeout_sec() == 30
        monkeypatch.setenv("SUBSTRATE_CONTROL_SURFACE_CONNECT_TIMEOUT_SEC", "not-a-number")
        assert mcs._connect_timeout_sec() == 5
        monkeypatch.delenv("SUBSTRATE_CONTROL_SURFACE_CONNECT_TIMEOUT_SEC", raising=False)
        assert mcs._connect_timeout_sec() == 5
