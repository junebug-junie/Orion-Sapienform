"""`_read_self_panel_payload()` in curiosity_routes.py -- the glue between the
Hub's Self panel and `orion.curiosity.self_panel.read_self_panel`.

Isolates `from . import main as hub_main` with a fake module injected into
`sys.modules` (and the `scripts` package attribute, covering both import
resolution paths) rather than importing the real, heavy `scripts.main`
module -- this test cares only about the pool hand-off and the "never 500s"
fallback, not Hub's startup sequence.
"""

from __future__ import annotations

import asyncio
import sys
import types

import scripts.curiosity_routes as cr
from orion.curiosity.self_panel import SelfDefinitionVersion, SelfPanelView


def _install_fake_main(monkeypatch, *, pool=None, app_missing=False):
    if app_missing:
        fake_main = types.SimpleNamespace()
    else:
        fake_main = types.SimpleNamespace(
            app=types.SimpleNamespace(state=types.SimpleNamespace(memory_pg_pool=pool))
        )
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    import scripts as scripts_pkg

    monkeypatch.setattr(scripts_pkg, "main", fake_main, raising=False)
    return fake_main


def test_the_found_pool_is_handed_to_read_self_panel(monkeypatch) -> None:
    sentinel_pool = object()
    _install_fake_main(monkeypatch, pool=sentinel_pool)

    captured: dict = {}

    async def fake_read(pool):
        captured["pool"] = pool
        return SelfPanelView(history=[SelfDefinitionVersion(version=1, created_at=None, content="I am", evidence_refs=["x"])])

    monkeypatch.setattr(cr, "read_self_panel", fake_read)

    payload = asyncio.run(cr._read_self_panel_payload())
    assert captured["pool"] is sentinel_pool
    assert payload["available"] is True
    assert payload["current"]["content"] == "I am"


def test_a_missing_app_state_falls_back_to_no_pool_not_a_crash(monkeypatch) -> None:
    """`hub_main.app` may not exist yet (startup race) or the whole import may
    fail; either way this must return a payload, never raise -- the same
    'a dashboard never 500s' contract every other reader in this file keeps."""
    _install_fake_main(monkeypatch, app_missing=True)

    async def fake_read(pool):
        assert pool is None
        return SelfPanelView(unavailable_reason="no_pool")

    monkeypatch.setattr(cr, "read_self_panel", fake_read)

    payload = asyncio.run(cr._read_self_panel_payload())
    assert payload == {"available": False, "reason": "no_pool"}


def test_atlas_api_includes_the_self_key(monkeypatch) -> None:
    """The self panel rides on the SAME `/api/atlas` read as everything else
    (one endpoint, so panels cannot disagree -- curiosity_atlas_api's own
    docstring), not a second network round trip."""

    async def fake_self_payload():
        return {"available": True, "current": {"version": 1, "content": "I am", "created_at": None, "evidence_refs": [], "produced_by": "curiosity_self_inquiry"}, "history": [], "journal_entries": [], "latest_eval_run_id": None, "latest_eval": []}

    monkeypatch.setattr(cr, "_build_reader", lambda: None)
    monkeypatch.setattr(cr, "_read_self_panel_payload", fake_self_payload)

    async def fake_schedule():
        return {}

    monkeypatch.setattr(cr, "_read_schedule", fake_schedule)
    monkeypatch.setattr(cr, "_wrote_on", lambda *a, **k: None)

    response = asyncio.run(cr.curiosity_atlas_api())
    import json

    body = json.loads(response.body)
    assert body["self"]["available"] is True
    assert body["self"]["current"]["content"] == "I am"
