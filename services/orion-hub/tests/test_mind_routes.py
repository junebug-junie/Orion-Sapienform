"""Hub Mind run read APIs (session-gated; mock PG pool)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

_HUB = Path(__file__).resolve().parents[1]
_REPO = _HUB.parents[1]
for p in (str(_REPO), str(_HUB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import mind_routes


def _mock_pool(conn: AsyncMock) -> tuple[MagicMock, SimpleNamespace]:
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(memory_pg_pool=pool)))
    return pool, req


def test_get_mind_run_returns_row() -> None:
    row = {
        "mind_run_id": "mid-1",
        "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "session_id": "sess",
        "trigger": "user_turn",
        "ok": True,
        "error_code": None,
        "snapshot_hash": "abc",
        "router_profile_id": "default",
        "result_jsonb": {},
        "request_summary_jsonb": {},
        "redaction_profile_id": None,
        "created_at_utc": None,
    }

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess"):
        out = asyncio.run(mind_routes.get_mind_run("mid-1", req, None, None))
        assert out["mind_run_id"] == "mid-1"
        assert out["session_visibility"] == "session_match"
        fetch_args = conn.fetchrow.await_args.args
        assert "mind_run_id = $1" in fetch_args[0]
        assert fetch_args[1] == "mid-1"
        assert fetch_args[2] == "sess"


def test_get_mind_run_not_found_when_other_session() -> None:
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-a"):
        try:
            asyncio.run(mind_routes.get_mind_run("mid-other-session", req, None, None))
            assert False, "Expected HTTPException"
        except Exception as exc:  # fastapi.HTTPException
            assert getattr(exc, "status_code", None) == 404


def test_get_mind_run_allows_context_session_id() -> None:
    row = {
        "mind_run_id": "mid-ctx",
        "correlation_id": "corr-ctx",
        "session_id": "sess-message",
        "trigger": "user_turn",
        "ok": True,
        "error_code": None,
        "snapshot_hash": "abc",
        "router_profile_id": "default",
        "result_jsonb": {},
        "request_summary_jsonb": {},
        "redaction_profile_id": None,
        "created_at_utc": None,
    }
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-browser"):
        out = asyncio.run(
            mind_routes.get_mind_run("mid-ctx", req, context_session_id="sess-message", x_orion_session_id=None)
        )
    assert out["mind_run_id"] == "mid-ctx"
    assert out["session_visibility"] == "session_other"


def test_list_recent_mind_runs_returns_summary_payload() -> None:
    recent_rows = [
        {
            "mind_run_id": "mid-2",
            "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "created_at_utc": None,
            "ok": True,
            "trigger": "user_turn",
            "error_code": None,
            "router_profile_id": "default",
        }
    ]
    summary_row = {"total_runs": 3, "ok_count": 2, "failed_count": 1}
    top_error_rows = [{"error_code": "timeout", "run_count": 1}]
    top_router_rows = [{"router_profile_id": "default", "run_count": 3}]
    bucket_rows = [{"bucket_utc": None, "run_count": 3}]

    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=[recent_rows, top_error_rows, top_router_rows, bucket_rows])
    conn.fetchrow = AsyncMock(return_value=summary_row)
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-123"):
        out = asyncio.run(
            mind_routes.list_recent_mind_runs(
                req,
                hours=24,
                limit=100,
                ok=None,
                trigger=None,
                error_code=None,
                router_profile_id=None,
                x_orion_session_id=None,
            )
        )

    assert out["next_cursor"] is None
    assert len(out["items"]) == 1
    assert out["aggregates"]["total_runs"] == 3
    assert out["aggregates"]["ok_count"] == 2
    assert out["aggregates"]["failed_count"] == 1
    assert out["aggregates"]["top_error_codes"][0]["error_code"] == "timeout"
    assert out["aggregates"]["top_router_profile_ids"][0]["router_profile_id"] == "default"
    assert out["aggregates"]["time_buckets"][0]["run_count"] == 3


def test_list_recent_mind_runs_scopes_by_session_id() -> None:
    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=[[], [], [], []])
    conn.fetchrow = AsyncMock(return_value={"total_runs": 0, "ok_count": 0, "failed_count": 0})
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-abc"):
        out = asyncio.run(
            mind_routes.list_recent_mind_runs(
                req,
                hours=12,
                limit=50,
                ok=True,
                trigger="user_turn",
                error_code=None,
                router_profile_id=None,
                x_orion_session_id=None,
            )
        )

    assert conn.fetch.await_count == 4
    first_fetch_args = conn.fetch.await_args_list[0].args
    assert "WHERE session_id = $1" in first_fetch_args[0]
    assert first_fetch_args[1] == "sess-abc"
    assert first_fetch_args[2] == 12
    assert first_fetch_args[3] is True
    assert out.get("diagnostics", {}).get("hint")


def test_list_mind_runs_returns_session_matching_rows() -> None:
    rows = [
        {
            "mind_run_id": "mid-1",
            "correlation_id": "corr-1",
            "session_id": "sess-x",
            "trigger": "user_turn",
            "ok": True,
            "error_code": None,
            "snapshot_hash": "abc",
            "router_profile_id": "default",
            "result_jsonb": {},
            "request_summary_jsonb": {},
            "redaction_profile_id": None,
            "created_at_utc": None,
        }
    ]
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-x"):
        out = asyncio.run(mind_routes.list_mind_runs(req, correlation_id="corr-1", limit=10, x_orion_session_id=None))
    assert len(out) == 1
    assert out[0]["session_visibility"] == "session_match"
    fetch_args = conn.fetch.await_args.args
    assert "correlation_id = $1" in fetch_args[0]
    assert "session_id = $2" in fetch_args[0]
    assert fetch_args[1] == "corr-1"
    assert fetch_args[2] == "sess-x"


def test_list_mind_runs_returns_null_session_fallback_rows() -> None:
    rows = [
        {
            "mind_run_id": "mid-null",
            "correlation_id": "corr-1",
            "session_id": None,
            "trigger": "user_turn",
            "ok": True,
            "error_code": None,
            "snapshot_hash": "abc",
            "router_profile_id": "default",
            "result_jsonb": {},
            "request_summary_jsonb": {},
            "redaction_profile_id": None,
            "created_at_utc": None,
        }
    ]
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-x"):
        out = asyncio.run(mind_routes.list_mind_runs(req, correlation_id="corr-1", limit=10, x_orion_session_id=None))
    assert len(out) == 1
    assert out[0]["session_visibility"] == "session_null_fallback"


def test_list_mind_runs_allows_context_session_without_current_match() -> None:
    rows = [
        {
            "mind_run_id": "mid-msg",
            "correlation_id": "corr-1",
            "session_id": "sess-message",
            "trigger": "user_turn",
            "ok": True,
            "error_code": None,
            "snapshot_hash": "abc",
            "router_profile_id": "default",
            "result_jsonb": {},
            "request_summary_jsonb": {},
            "redaction_profile_id": None,
            "created_at_utc": None,
        }
    ]
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-browser"):
        out = asyncio.run(
            mind_routes.list_mind_runs(
                req,
                correlation_id="corr-1",
                limit=10,
                context_session_id="sess-message",
                x_orion_session_id=None,
            )
        )
    assert len(out) == 1
    fetch_args = conn.fetch.await_args.args
    assert fetch_args[4] == "sess-message"


def test_list_mind_runs_excludes_other_non_null_session_without_context() -> None:
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    _, req = _mock_pool(conn)

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-browser"):
        out = asyncio.run(mind_routes.list_mind_runs(req, correlation_id="corr-1", limit=10, x_orion_session_id=None))
    assert out == []
    fetch_sql = conn.fetch.await_args.args[0]
    assert "session_id IS NULL" in fetch_sql
    assert "session_id = $4" not in fetch_sql or "$4::text IS NOT NULL" in fetch_sql


def test_list_mind_runs_returns_503_when_postgres_unreachable() -> None:
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=ConnectionRefusedError(111, "Connection refused"))
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(memory_pg_pool=pool)))

    with patch.object(mind_routes, "_need_session", new_callable=AsyncMock, return_value="sess-x"):
        try:
            asyncio.run(
                mind_routes.list_mind_runs(
                    req,
                    correlation_id="corr-1",
                    limit=10,
                    x_orion_session_id=None,
                )
            )
            assert False, "Expected HTTPException"
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 503
            assert getattr(exc, "detail", None) == "mind_store_unavailable"
