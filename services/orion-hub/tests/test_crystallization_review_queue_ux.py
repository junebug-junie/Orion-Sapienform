"""Bulk decide + per-turn evidence removal, and the UI wiring for both.

Backs the 2026-08-14 review-queue UX work. The queue had no multi-select, the
only click target was an "Open" label at the far right of each row, the action
buttons sat at the bottom of a scrolling sub-pane, and a decision left the
detail pane open on the item it had just decided -- so the next click hit a
proposal whose status was no longer "proposed" and errored.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# scripts.main constructs the full hub Settings at import time; the cache-bust
# test below imports it. Same convention as test_organ_signals_graph_hub_tab.py.
for _key, _value in (
    ("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript"),
    ("CHANNEL_VOICE_LLM", "orion:voice:llm"),
    ("CHANNEL_VOICE_TTS", "orion:voice:tts"),
    ("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"),
    ("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage"),
):
    os.environ.setdefault(_key, _value)

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

UI_JS = HUB_ROOT / "static" / "js" / "memory-crystallization-ui.js"


def _crys(*, crystallization_id: str, status: str = "proposed", evidence_ids=("t1", "t2")):
    from orion.memory.crystallization.schemas import (
        CrystallizationEvidenceRefV1,
        CrystallizationGovernanceV1,
        MemoryCrystallizationV1,
    )

    now = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)
    return MemoryCrystallizationV1(
        crystallization_id=crystallization_id,
        kind="stance",
        subject="test subject",
        summary="test summary",
        status=status,
        evidence=[
            CrystallizationEvidenceRefV1(
                source_kind="chat_turn", source_id=sid, excerpt="p\nr", strength=0.75
            )
            for sid in evidence_ids
        ],
        governance=CrystallizationGovernanceV1(proposed_by="test"),
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def ctx(monkeypatch):
    """TestClient plus the mocks the routes reach through, so each test can set
    up its own row states without rebuilding the app."""
    from scripts.crystallization_routes import router

    app = FastAPI()
    app.include_router(router)
    app.state.memory_pg_pool = MagicMock()

    async def _need_session(_sid):
        return "sess-1"

    async def _bus():
        return None

    monkeypatch.setattr("scripts.crystallization_routes._need_session", _need_session)
    monkeypatch.setattr("scripts.crystallization_routes._bus", _bus)
    # The lifecycle-emit call reads SERVICE_NAME/NODE_NAME off the real hub
    # Settings, which needs a full env to construct. Stub it -- these tests are
    # about decision routing, not config loading.
    monkeypatch.setattr(
        "scripts.crystallization_routes._settings",
        lambda: MagicMock(SERVICE_NAME="orion-hub", NODE_NAME="test"),
    )
    monkeypatch.setattr(
        "scripts.crystallization_routes.emit_crystallization_lifecycle", AsyncMock()
    )
    monkeypatch.setattr("scripts.crystallization_routes.update_crystallization", AsyncMock())
    history = AsyncMock()
    monkeypatch.setattr("scripts.crystallization_routes.insert_history", history)

    rows: dict[str, object] = {}

    async def _get(_pool, cid):
        return rows.get(cid)

    monkeypatch.setattr("scripts.crystallization_routes.get_crystallization", _get)
    return {"client": TestClient(app), "rows": rows, "history": history, "app": app}


# --------------------------------------------------------------------------
# bulk decide
# --------------------------------------------------------------------------


def test_bulk_reject_decides_every_id(ctx):
    ctx["rows"].update({f"c{i}": _crys(crystallization_id=f"c{i}") for i in range(3)})
    res = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk",
        json={"ids": ["c0", "c1", "c2"], "action": "reject"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["succeeded"] == 3
    assert body["failed"] == 0
    assert {r["crystallization_id"] for r in body["results"]} == {"c0", "c1", "c2"}


def test_bulk_reject_reports_partial_failure_without_sinking_the_batch(ctx):
    """One bad id must not cost the caller the other decisions."""
    ctx["rows"]["good"] = _crys(crystallization_id="good")
    ctx["rows"]["already"] = _crys(crystallization_id="already", status="active")
    res = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk",
        json={"ids": ["good", "already", "missing"], "action": "reject"},
    )
    body = res.json()
    assert body["succeeded"] == 1
    assert body["failed"] == 2
    by_id = {r["crystallization_id"]: r for r in body["results"]}
    assert by_id["good"]["ok"] is True
    assert by_id["already"]["error"] == "already_active"
    assert by_id["missing"]["error"] == "not_found"


def test_bulk_dedupes_repeated_ids(ctx):
    ctx["rows"]["c0"] = _crys(crystallization_id="c0")
    res = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk",
        json={"ids": ["c0", "c0", "c0"], "action": "reject"},
    )
    body = res.json()
    assert body["requested"] == 1
    assert body["succeeded"] == 1


def test_bulk_rejects_unknown_action(ctx):
    res = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk",
        json={"ids": ["c0"], "action": "delete"},
    )
    assert res.status_code == 400


def test_bulk_requires_ids(ctx):
    res = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk", json={"ids": [], "action": "reject"}
    )
    assert res.status_code == 400


def test_bulk_caps_batch_size(ctx):
    from scripts.crystallization_routes import BULK_DECIDE_MAX

    res = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk",
        json={"ids": [str(i) for i in range(BULK_DECIDE_MAX + 1)], "action": "reject"},
    )
    assert res.status_code == 400
    assert str(BULK_DECIDE_MAX) in str(res.json()["detail"])


def test_approve_has_a_much_lower_cap_than_reject(ctx):
    """Each approve also runs a card/chroma projection and a second write, so a
    reject-sized batch of them would be minutes of serialized I/O in one HTTP
    request -- the client disconnects mid-way over a half-applied batch."""
    from scripts.crystallization_routes import BULK_APPROVE_MAX, BULK_DECIDE_MAX

    assert BULK_APPROVE_MAX < BULK_DECIDE_MAX
    ids = [str(i) for i in range(BULK_APPROVE_MAX + 1)]
    approve = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk", json={"ids": ids, "action": "approve"}
    )
    assert approve.status_code == 400
    assert str(BULK_APPROVE_MAX) in str(approve.json()["detail"])
    # the same batch size is fine for reject
    reject = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk", json={"ids": ids, "action": "reject"}
    )
    assert reject.status_code == 200


def test_bulk_path_is_not_captured_as_a_crystallization_id(ctx):
    """"bulk" must reach the bulk handler, not be read as an id by a sibling
    route. A 404 body of proposal_not_found would mean it was."""
    ctx["rows"]["c0"] = _crys(crystallization_id="c0")
    res = ctx["client"].post(
        "/api/memory/crystallizations/proposals/bulk",
        json={"ids": ["c0"], "action": "reject"},
    )
    assert res.status_code == 200
    assert "results" in res.json()


# --------------------------------------------------------------------------
# per-turn evidence removal
# --------------------------------------------------------------------------


def test_drop_turn_removes_only_that_source(ctx):
    ctx["rows"]["c0"] = _crys(crystallization_id="c0", evidence_ids=("t1", "t2", "t3"))
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="DELETE 1")
    ctx["app"].state.memory_pg_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    ctx["app"].state.memory_pg_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    res = ctx["client"].request(
        "DELETE", "/api/memory/crystallizations/c0/evidence/t2"
    )
    assert res.status_code == 200
    sql, cid, source_id = conn.execute.await_args.args
    assert "DELETE FROM memory_crystallization_sources" in sql
    assert (cid, source_id) == ("c0", "t2")


def test_drop_turn_writes_history(ctx):
    ctx["rows"]["c0"] = _crys(crystallization_id="c0", evidence_ids=("t1", "t2"))
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="DELETE 1")
    ctx["app"].state.memory_pg_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    ctx["app"].state.memory_pg_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    ctx["client"].request("DELETE", "/api/memory/crystallizations/c0/evidence/t2")

    kwargs = ctx["history"].await_args.kwargs
    assert kwargs["op"] == "evidence_removed"
    assert kwargs["after"]["removed_source_id"] == "t2"
    assert kwargs["after"]["evidence_count"] == 1


def test_drop_unknown_turn_is_404(ctx):
    ctx["rows"]["c0"] = _crys(crystallization_id="c0", evidence_ids=("t1", "t2"))
    res = ctx["client"].request("DELETE", "/api/memory/crystallizations/c0/evidence/nope")
    assert res.status_code == 404


def test_cannot_drop_the_last_turn(ctx):
    """A proposal with zero evidence is not reviewable, and validate_proposal
    would reject it anyway -- refuse rather than create that state."""
    ctx["rows"]["c0"] = _crys(crystallization_id="c0", evidence_ids=("t1",))
    res = ctx["client"].request("DELETE", "/api/memory/crystallizations/c0/evidence/t1")
    assert res.status_code == 409


def test_cannot_drop_from_an_active_crystallization(ctx):
    """Active rows are already projected into cards/chroma/graphiti; removing a
    cited source underneath those projections would desync them."""
    ctx["rows"]["c0"] = _crys(crystallization_id="c0", status="active")
    res = ctx["client"].request("DELETE", "/api/memory/crystallizations/c0/evidence/t1")
    assert res.status_code == 409


# --------------------------------------------------------------------------
# UI wiring (text smoke -- this page has no browser harness; same convention as
# test_memory_crystallization_ui.py)
# --------------------------------------------------------------------------


def test_drop_turn_normalizes_a_crys_prefixed_id(ctx):
    """new_crystallization_id() mints `crys_<hex32>`. Every repository helper
    rewrites that to dashed UUID form before binding ::uuid; the inline DELETE
    here originally did not, so such an id cleared the 404/409 guards and then
    died on the cast with a misleading 503."""
    cid = "crys_" + "ab" * 16
    ctx["rows"][cid] = _crys(crystallization_id=cid, evidence_ids=("t1", "t2"))
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="DELETE 1")
    ctx["app"].state.memory_pg_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    ctx["app"].state.memory_pg_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    res = ctx["client"].request("DELETE", f"/api/memory/crystallizations/{cid}/evidence/t2")
    assert res.status_code == 200
    _sql, bound_id, _src = conn.execute.await_args.args
    assert bound_id == "abababab-abab-abab-abab-abababababab"


def test_ui_surfaces_server_error_detail_not_just_the_status():
    """A bare "HTTP 400" hid the bulk endpoint's own too_many_ids_max_N."""
    ui = UI_JS.read_text(encoding="utf-8")
    assert "body.detail" in ui


def test_ui_chunks_bulk_requests_under_the_server_caps():
    from scripts.crystallization_routes import BULK_APPROVE_MAX, BULK_DECIDE_MAX

    ui = UI_JS.read_text(encoding="utf-8")
    approve_chunk = int(re.search(r"const APPROVE_CHUNK = (\d+)", ui).group(1))
    reject_chunk = int(re.search(r"const REJECT_CHUNK = (\d+)", ui).group(1))
    assert approve_chunk <= BULK_APPROVE_MAX
    assert reject_chunk <= BULK_DECIDE_MAX


def test_ui_toggling_a_checkbox_does_not_refetch_the_queue():
    """Every tick used to fire an un-awaited loadInbox(): three quick ticks
    launched three overlapping loads whose innerHTML="" and appends interleaved.
    Selection is local state and must be reflected locally."""
    ui = UI_JS.read_text(encoding="utf-8")
    toggle = ui[ui.index("(row, isChecked) => {") : ui.index("selected.has(item.crystallization_id),")]
    assert "refreshSelectionUi()" in toggle
    assert "loadInbox" not in toggle


def test_ui_select_all_excludes_undecidable_rows():
    ui = UI_JS.read_text(encoding="utf-8")
    assert "function isDecidable" in ui
    assert "const decidable = items.filter(isDecidable)" in ui


def test_ui_handles_open_detail_rejection():
    ui = UI_JS.read_text(encoding="utf-8")
    open_handler = ui[ui.index("(row) => {") : ui.index("(row, isChecked) => {")]
    assert ".catch(" in open_handler
    assert "closeDetail" in open_handler


def test_ui_has_multi_select_and_bulk_actions():
    ui = UI_JS.read_text(encoding="utf-8")
    assert 'type = "checkbox"' in ui
    assert "/api/memory/crystallizations/proposals/bulk" in ui
    assert "Approve selected" in ui
    assert "Reject selected" in ui
    assert "select all" in ui


def test_ui_opens_on_whole_row_not_just_the_open_button():
    ui = UI_JS.read_text(encoding="utf-8")
    assert 'row.addEventListener("click"' in ui
    # the checkbox must not double as an open trigger
    assert "stopPropagation" in ui


def test_ui_closes_detail_after_a_decision():
    ui = UI_JS.read_text(encoding="utf-8")
    assert "function closeDetail" in ui
    assert "detailEl.dataset.crystallizationId" in ui


def test_ui_exposes_per_turn_drop():
    ui = UI_JS.read_text(encoding="utf-8")
    assert "data-drop-turn" in ui
    assert "/evidence/" in ui


def test_ui_actions_are_pinned_to_the_top_of_the_detail_pane():
    ui = UI_JS.read_text(encoding="utf-8")
    actions_at = ui.index('data-act="approve"')
    summary_at = ui.index("Projection refs:")
    assert actions_at < summary_at, "approve/reject must render above the detail body"
    assert "sticky top-0" in ui


def test_ui_asset_cache_bust_covers_this_module():
    """Until 2026-08-14 the cache-bust token read a hardcoded four-file list that
    did not include memory-crystallization-ui.js, so every edit in this PR would
    have shipped behind a stale browser cache.

    Asserted behaviorally against the real file rather than by grepping main.py
    for the glob: a grep passes for any implementation that merely *mentions*
    the right shape, including one that globs a directory this file is not in.
    """
    import os

    from scripts.main import _ui_asset_mtime_token

    before = _ui_asset_mtime_token()
    original = UI_JS.stat().st_mtime
    try:
        os.utime(UI_JS, (original + 10_000, original + 10_000))
        assert _ui_asset_mtime_token() != before, (
            "editing memory-crystallization-ui.js must move the ?v= token"
        )
    finally:
        os.utime(UI_JS, (original, original))
    assert _ui_asset_mtime_token() == before
