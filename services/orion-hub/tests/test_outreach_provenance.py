from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from scripts.outreach_provenance import (
    fetch_latest_outreach_provenance,
    format_outreach_provenance_block,
    merge_situation_with_outreach_provenance,
    select_active_outreach_provenance,
    _valid_outreach_capsule,
)


def _capsule(**overrides):
    base = {
        "schema": "outreach_provenance.v1",
        "decision_id": "dec-1",
        "correlation_id": "corr-1",
        "generated_at": "2026-09-15T19:27:05+00:00",
        "lanes": {"priors_count": 3, "tension": False},
        "prompt_text": "You are Orion. Juniper has not asked you anything — FULL PROMPT",
        "summary_line": "Outreach from open priors (3)",
    }
    base.update(overrides)
    return base


def test_format_outreach_provenance_block_includes_prompt_and_anti_confabulation() -> None:
    block = format_outreach_provenance_block(_capsule())
    assert "FULL PROMPT" in block
    # Header uses "unsolicited endogenous outreach" (contiguous "unsolicited outreach"
    # is not present); assert the real anti-confabulation phrases.
    assert "unsolicited endogenous outreach" in block.lower()
    assert "collapse-mirror" in block.lower()
    assert "do not invent" in block.lower()


def test_format_returns_empty_on_bad_capsule() -> None:
    assert format_outreach_provenance_block({}) == ""
    assert format_outreach_provenance_block({"prompt_text": ""}) == ""
    assert format_outreach_provenance_block(None) == ""


def test_format_returns_empty_on_wrong_schema() -> None:
    assert (
        format_outreach_provenance_block(
            _capsule(schema="outreach_provenance.v0")
        )
        == ""
    )
    assert format_outreach_provenance_block(_capsule(schema="other.v1")) == ""
    # Missing schema key
    bad = _capsule()
    del bad["schema"]
    assert format_outreach_provenance_block(bad) == ""


def test_format_returns_empty_on_non_string_prompt_text() -> None:
    assert format_outreach_provenance_block(_capsule(prompt_text=None)) == ""
    assert format_outreach_provenance_block(_capsule(prompt_text=12345)) == ""
    assert format_outreach_provenance_block(_capsule(prompt_text=["list"])) == ""
    assert format_outreach_provenance_block(_capsule(prompt_text="   ")) == ""


def test_valid_outreach_capsule_rejects_malformed() -> None:
    assert _valid_outreach_capsule(_capsule()) is not None
    assert _valid_outreach_capsule(_capsule(schema="nope")) is None
    assert _valid_outreach_capsule(_capsule(prompt_text=42)) is None
    assert _valid_outreach_capsule("not-a-dict") is None


def test_merge_situation_appends_block() -> None:
    merged = merge_situation_with_outreach_provenance("local afternoon", "OUTREACH BLOCK")
    assert merged.startswith("local afternoon")
    assert "OUTREACH BLOCK" in merged


def test_merge_situation_block_only() -> None:
    assert merge_situation_with_outreach_provenance(None, "OUTREACH BLOCK") == "OUTREACH BLOCK"


def test_merge_situation_none_when_both_empty() -> None:
    assert merge_situation_with_outreach_provenance(None, None) is None
    assert merge_situation_with_outreach_provenance("  ", "") is None


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_merges_block(monkeypatch) -> None:
    from orion.hub import turn_orchestrator as orch

    monkeypatch.setattr(
        orch,
        "fetch_latest_outreach_provenance",
        lambda sid, **kw: _capsule(),
    )
    merged = await orch._situation_with_outreach_provenance("local afternoon", "sess-1")
    assert merged is not None
    assert merged.startswith("local afternoon")
    assert "FULL PROMPT" in merged
    assert "unsolicited endogenous outreach" in merged.lower()


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_skips_bad_schema(monkeypatch) -> None:
    from orion.hub import turn_orchestrator as orch

    monkeypatch.setattr(
        orch,
        "fetch_latest_outreach_provenance",
        lambda sid, **kw: _capsule(schema="wrong.v1"),
    )
    merged = await orch._situation_with_outreach_provenance("local afternoon", "sess-1")
    assert merged == "local afternoon"


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_uses_to_thread(monkeypatch) -> None:
    """Fetch must leave the event loop via asyncio.to_thread."""
    from orion.hub import turn_orchestrator as orch

    calls: list[object] = []

    def _fake_fetch(sid, **kw):
        calls.append(("fetch", sid))
        return _capsule()

    async def _fake_to_thread(fn, /, *args, **kwargs):
        calls.append(("to_thread", fn, args))
        return fn(*args, **kwargs)

    monkeypatch.setattr(orch, "fetch_latest_outreach_provenance", _fake_fetch)
    monkeypatch.setattr(orch.asyncio, "to_thread", _fake_to_thread)

    merged = await orch._situation_with_outreach_provenance("sit", "sess-9")
    assert any(c[0] == "to_thread" for c in calls)
    assert any(c[0] == "fetch" and c[1] == "sess-9" for c in calls)
    assert merged is not None and "FULL PROMPT" in merged


@pytest.mark.asyncio
async def test_situation_with_outreach_provenance_no_session_passthrough() -> None:
    from orion.hub import turn_orchestrator as orch

    assert (
        await orch._situation_with_outreach_provenance("local afternoon", None)
        == "local afternoon"
    )
    assert await orch._situation_with_outreach_provenance(None, None) is None


def _ts(minute: int) -> datetime:
    return datetime(2026, 9, 15, 12, minute, tzinfo=timezone.utc)


def _outreach_row(*, minute: int = 0, capsule=None, response: str = "hey"):
    return {
        "created_at": _ts(minute),
        "client_meta": {
            "unsolicited": "true",
            "outreach_provenance": capsule if capsule is not None else _capsule(),
        },
        "response": response,
    }


def _solicited_row(*, minute: int, response: str = "normal reply"):
    return {
        "created_at": _ts(minute),
        "client_meta": {"unsolicited": "false"},
        "response": response,
    }


def test_select_active_returns_capsule_when_no_later_solicited() -> None:
    """(a) Latest unsolicited+provenance with no later solicited response."""
    rows = [
        {"created_at": _ts(0), "client_meta": {}, "response": "earlier chat"},
        _outreach_row(minute=1),
    ]
    got = select_active_outreach_provenance(rows)
    assert got is not None
    assert got["prompt_text"].startswith("You are Orion")
    assert got["correlation_id"] == "corr-1"


def test_select_active_cleared_by_later_non_unsolicited_response() -> None:
    """(b) Later non-unsolicited assistant response clears injection."""
    rows = [
        _outreach_row(minute=1),
        _solicited_row(minute=2, response="thanks for asking"),
    ]
    assert select_active_outreach_provenance(rows) is None


def test_select_active_later_empty_response_does_not_clear() -> None:
    rows = [
        _outreach_row(minute=1),
        _solicited_row(minute=2, response=""),
        {
            "created_at": _ts(3),
            "client_meta": {},
            "response": None,
        },
    ]
    got = select_active_outreach_provenance(rows)
    assert got is not None
    assert got["schema"] == "outreach_provenance.v1"


def test_select_active_later_unsolicited_does_not_clear_prior() -> None:
    """A second outreach does not clear; newest outreach capsule wins."""
    first = _capsule(correlation_id="corr-old", prompt_text="OLD PROMPT text")
    second = _capsule(correlation_id="corr-new", prompt_text="NEW PROMPT text")
    rows = [
        _outreach_row(minute=1, capsule=first),
        _outreach_row(minute=2, capsule=second),
    ]
    got = select_active_outreach_provenance(rows)
    assert got is not None
    assert got["correlation_id"] == "corr-new"
    assert "NEW PROMPT" in got["prompt_text"]


def test_select_active_malformed_client_meta_returns_none() -> None:
    """(c) Empty / malformed provenance → None (even when unsolicited)."""
    assert select_active_outreach_provenance([]) is None
    assert (
        select_active_outreach_provenance(
            [
                {
                    "created_at": _ts(0),
                    "client_meta": {"unsolicited": "true"},
                    "response": "x",
                }
            ]
        )
        is None
    )
    assert (
        select_active_outreach_provenance(
            [
                {
                    "created_at": _ts(0),
                    "client_meta": {
                        "unsolicited": "true",
                        "outreach_provenance": "not-a-dict",
                    },
                    "response": "x",
                }
            ]
        )
        is None
    )
    assert (
        select_active_outreach_provenance(
            [
                {
                    "created_at": _ts(0),
                    "client_meta": {
                        "unsolicited": "true",
                        "outreach_provenance": _capsule(schema="wrong.v1"),
                    },
                    "response": "x",
                }
            ]
        )
        is None
    )
    assert (
        select_active_outreach_provenance(
            [{"created_at": _ts(0), "client_meta": "bad", "response": "x"}]
        )
        is None
    )


def _install_fake_engine(monkeypatch, rows: list[dict]):
    """Hub-style create_engine monkeypatch returning controlled mappings rows."""
    import sqlalchemy

    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    class _Mappings:
        def all(self):
            return rows

    class _Result:
        def mappings(self):
            return _Mappings()

    conn.execute.return_value = _Result()

    def _fake_create_engine(uri, **kwargs):
        return fake_engine

    monkeypatch.setenv("POSTGRES_URI", "postgresql://test/db")
    # fetch imports create_engine inside the function body.
    monkeypatch.setattr(sqlalchemy, "create_engine", _fake_create_engine)
    return fake_engine


def test_fetch_latest_returns_capsule_via_mocked_engine(monkeypatch) -> None:
    """(a) through thin SQL reader + select_active."""
    rows = [_outreach_row(minute=1)]
    engine = _install_fake_engine(monkeypatch, rows)
    got = fetch_latest_outreach_provenance("sess-a")
    assert got is not None
    assert got["correlation_id"] == "corr-1"
    engine.dispose.assert_called_once()


def test_fetch_latest_cleared_via_mocked_engine(monkeypatch) -> None:
    """(b) through thin SQL reader when later solicited response exists."""
    rows = [
        _outreach_row(minute=1),
        _solicited_row(minute=2),
    ]
    _install_fake_engine(monkeypatch, rows)
    assert fetch_latest_outreach_provenance("sess-b") is None


def test_fetch_latest_malformed_meta_via_mocked_engine(monkeypatch) -> None:
    """(c) through thin SQL reader with malformed outreach_provenance."""
    rows = [
        {
            "created_at": _ts(0),
            "client_meta": {
                "unsolicited": "true",
                "outreach_provenance": {"schema": "nope"},
            },
            "response": "x",
        }
    ]
    _install_fake_engine(monkeypatch, rows)
    assert fetch_latest_outreach_provenance("sess-c") is None


def test_fetch_latest_missing_env_or_session_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    assert fetch_latest_outreach_provenance("sess") is None
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test/db")
    assert fetch_latest_outreach_provenance(None) is None
    assert fetch_latest_outreach_provenance("  ") is None


# ---------------------------------------------------------------------------
# Reply stamp (2026-09-22): the same clearing rule, without the capsule
# requirement, gives Juniper's next inbound turn `client_meta.in_reply_to`.
# ---------------------------------------------------------------------------

# NOTE: the async tests below call `mod.reply_stamp_for_session` off a module
# imported INSIDE the test. `tests/conftest.py::_ensure_hub_paths` evicts every
# `scripts.*` entry from sys.modules, so a function bound at this file's import
# time can read globals from a different module object than the one a later
# `import scripts.outreach_provenance as mod` + monkeypatch touches.
from scripts.outreach_provenance import (  # noqa: E402
    fetch_reply_target,
    reply_stamp_from_target,
    select_active_unsolicited_row,
    select_reply_target,
    _meta_unsolicited,
)


def _plain_outreach_row(
    *, minute: int, row_id: str, source: str | None = "curiosity_outreach", corr: str | None = None
):
    """A curiosity reach-out: unsolicited, NO provenance capsule."""
    meta: dict = {"unsolicited": True}
    if source:
        meta["source"] = source
    return {
        "id": row_id,
        "correlation_id": corr if corr is not None else row_id,
        "created_at": _ts(minute),
        "client_meta": meta,
        "response": "hey",
    }


def test_meta_unsolicited_matches_the_live_jsonb_boolean() -> None:
    """Live rows store `unsolicited` as jsonb `true`; SQLAlchemy returns
    Python `True`. `str(True) == "True"`, so the old `== "true"` check never
    matched a real row (checked live 2026-09-22)."""
    assert _meta_unsolicited({"unsolicited": True})
    assert _meta_unsolicited({"unsolicited": "true"})
    assert _meta_unsolicited({"unsolicited": "True"})
    assert not _meta_unsolicited({"unsolicited": False})
    assert not _meta_unsolicited({"unsolicited": "false"})
    assert not _meta_unsolicited({})


def test_provenance_selection_still_requires_the_capsule() -> None:
    """Widening for the reply stamp must not widen provenance injection: a
    capsule-less curiosity row is a reply target but not an injection."""
    rows = [_plain_outreach_row(minute=1, row_id="row-a")]
    assert select_active_outreach_provenance(rows) is None
    assert select_active_unsolicited_row(rows, require_capsule=True) is None
    assert select_active_unsolicited_row(rows, require_capsule=False) is rows[0]


def test_reply_target_after_an_unsolicited_row() -> None:
    rows = [
        {"id": "u0", "created_at": _ts(0), "client_meta": {}, "response": "earlier"},
        _plain_outreach_row(minute=1, row_id="row-a"),
    ]
    assert select_reply_target(rows) == {"correlation_id": "row-a", "source": "curiosity_outreach"}
    assert reply_stamp_from_target(select_reply_target(rows)) == {
        "in_reply_to": "row-a",
        "in_reply_to_source": "curiosity_outreach",
    }


def test_reply_target_cleared_by_a_later_solicited_reply() -> None:
    rows = [
        _plain_outreach_row(minute=1, row_id="row-a"),
        {"id": "s1", "created_at": _ts(2), "client_meta": {"unsolicited": "false"},
         "response": "normal reply"},
    ]
    assert select_reply_target(rows) is None
    assert reply_stamp_from_target(None) == {}


def test_reply_target_newest_unsolicited_wins_and_source_is_optional() -> None:
    rows = [
        _plain_outreach_row(minute=1, row_id="row-old"),
        _plain_outreach_row(minute=2, row_id="row-new", source=None),
    ]
    assert select_reply_target(rows) == {"correlation_id": "row-new", "source": None}
    assert reply_stamp_from_target(select_reply_target(rows)) == {"in_reply_to": "row-new"}


def test_reply_target_without_an_id_is_not_a_stamp() -> None:
    rows = [{"created_at": _ts(1), "client_meta": {"unsolicited": True}, "response": "x"}]
    assert select_reply_target(rows) is None


def test_reply_target_prefers_the_correlation_id_column_over_id() -> None:
    """The run story joins on `correlation_id` (the run-derived uuid5), not
    on `id`; sql-writer happens to set both equal, but that is its invariant."""
    rows = [_plain_outreach_row(minute=1, row_id="pk-row", corr="corr-uuid5")]
    assert select_reply_target(rows)["correlation_id"] == "corr-uuid5"
    rows = [{"id": "only-id", "created_at": _ts(1), "client_meta": {"unsolicited": True}, "response": "x"}]
    assert select_reply_target(rows)["correlation_id"] == "only-id"


def test_fetch_reply_target_window_is_enforced_by_the_query(monkeypatch) -> None:
    """The 12h window is a SQL predicate: a 13h-old row is never loaded, so
    the selector sees no rows. Pinned by asserting the bound parameters."""
    engine = _install_fake_engine(monkeypatch, [])
    assert fetch_reply_target("sid-1", max_age_hours=12.0) is None
    conn = engine.connect.return_value.__enter__.return_value
    sql, params = conn.execute.call_args.args
    assert params["sid"] == "sid-1"
    assert params["max_age_secs"] == 12.0 * 3600.0
    assert params["lim"] == 500
    assert "LIMIT :lim" in str(sql)
    assert "correlation_id" in str(sql)
    assert "make_interval(secs => :max_age_secs)" in str(sql)


def test_fetch_reply_target_via_mocked_engine(monkeypatch) -> None:
    rows = [_plain_outreach_row(minute=1, row_id="row-a")]
    _install_fake_engine(monkeypatch, rows)
    assert fetch_reply_target("sid-1") == {"correlation_id": "row-a", "source": "curiosity_outreach"}


def test_fetch_reply_target_db_failure_is_none(monkeypatch) -> None:
    import sqlalchemy

    def _boom(uri, **kwargs):
        raise RuntimeError("no db")

    monkeypatch.setenv("POSTGRES_URI", "postgresql://test/db")
    monkeypatch.setattr(sqlalchemy, "create_engine", _boom)
    assert fetch_reply_target("sid-1") is None


@pytest.mark.asyncio
async def test_reply_stamp_for_session_sets_stamp(monkeypatch) -> None:
    import scripts.outreach_provenance as mod

    monkeypatch.setattr(
        mod, "fetch_reply_target",
        lambda sid, *, max_age_hours=12.0: {"correlation_id": "row-a", "source": "curiosity_outreach"},
    )
    assert await mod.reply_stamp_for_session("sid-1") == {
        "in_reply_to": "row-a",
        "in_reply_to_source": "curiosity_outreach",
    }


@pytest.mark.asyncio
async def test_reply_stamp_for_session_db_failure_is_empty_and_does_not_raise(monkeypatch) -> None:
    import scripts.outreach_provenance as mod

    def _boom(sid, *, max_age_hours=12.0):
        raise RuntimeError("db down")

    monkeypatch.setattr(mod, "fetch_reply_target", _boom)
    assert await mod.reply_stamp_for_session("sid-1") == {}


@pytest.mark.asyncio
async def test_reply_stamp_for_session_times_out_to_empty(monkeypatch) -> None:
    import time as _time

    import scripts.outreach_provenance as mod

    def _slow(sid, *, max_age_hours=12.0):
        _time.sleep(0.3)
        return {"correlation_id": "late"}

    monkeypatch.setattr(mod, "fetch_reply_target", _slow)
    assert await mod.reply_stamp_for_session("sid-1", timeout_sec=0.05) == {}


@pytest.mark.asyncio
async def test_reply_stamp_for_session_no_session_is_empty(monkeypatch) -> None:
    import scripts.outreach_provenance as mod

    def _never(*a, **k):
        raise AssertionError("must not query without a session id")

    monkeypatch.setattr(mod, "fetch_reply_target", _never)
    assert await mod.reply_stamp_for_session(None) == {}
    assert await mod.reply_stamp_for_session("  ") == {}
