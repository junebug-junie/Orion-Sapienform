"""The catalog must show every route that exists, and say why two share a worker.

Until 2026-08-19 four independent lists hardcoded ("chat", "quick", "agent", "metacog"). Orion's
own journalling had been running on `quick_background` since PR #1708, and that lane was absent
from `GET /routes`, from the Hub, and from the routes smoke -- which passed green throughout.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app import route_catalog
from app.llm_backend import RouteTarget
from orion.llm.routes import ACCEPTED_LLM_ROUTES, LLM_ROUTE_DISPLAY_ORDER


def _targets():
    return {
        "chat": RouteTarget(url="http://circe:8011", served_by="circe-worker-1", backend="llamacpp"),
        "quick": RouteTarget(url="http://atlas:8013", served_by="atlas-worker-fast-1", backend="llamacpp"),
        "quick_background": RouteTarget(
            url="http://atlas:8013", served_by="atlas-worker-fast-1", backend="llamacpp",
            priority="background", reserved_free_slots=2,
        ),
        "metacog": RouteTarget(url="http://atlas:8012", served_by="atlas-worker-2", backend="llamacpp"),
        "agent": RouteTarget(url="http://circe:8014", served_by="circe-worker-agent-1", backend="llamacpp"),
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    route_catalog._cache.clear()
    route_catalog._last_refresh_mono = 0.0
    yield
    route_catalog._cache.clear()
    route_catalog._last_refresh_mono = 0.0


def test_catalog_covers_every_accepted_route():
    """Behavioural, not a re-export check: the catalog IS the display order, and the display
    order is pinned to the accepted set by an import-time assertion in orion/llm/routes.py."""
    assert set(route_catalog.CATALOG_ROUTE_IDS) == set(ACCEPTED_LLM_ROUTES)
    assert "quick_background" in route_catalog.CATALOG_ROUTE_IDS


def test_display_order_cannot_silently_omit_a_route():
    """The gate itself. Adding a route to ACCEPTED without placing it in the display order must
    fail loudly at import rather than quietly dropping it from every operator surface."""
    assert len(LLM_ROUTE_DISPLAY_ORDER) == len(ACCEPTED_LLM_ROUTES)
    assert len(set(LLM_ROUTE_DISPLAY_ORDER)) == len(LLM_ROUTE_DISPLAY_ORDER)


@pytest.mark.asyncio
async def test_background_route_declares_its_priority(monkeypatch):
    monkeypatch.setattr(route_catalog, "get_route_targets", _targets)
    monkeypatch.setattr(route_catalog, "_probe_backend",
                        AsyncMock(return_value=("up", 12, "model.gguf", False)))
    payload = await route_catalog.get_routes_payload()
    by_id = {r["id"]: r for r in payload["routes"]}
    assert by_id["quick_background"]["priority"] == "background"
    assert by_id["quick_background"]["reserved_free_slots"] == 2
    # ...and an ordinary lane does not claim to be one.
    assert by_id["quick"]["priority"] is None


@pytest.mark.asyncio
async def test_routes_sharing_a_worker_are_probed_once(monkeypatch):
    """`quick` and `quick_background` are one llama.cpp process under two admission policies.
    Probing per route would send a duplicate health+model+vision round trip to
    atlas-worker-fast-1 every 15s for no new information."""
    monkeypatch.setattr(route_catalog, "get_route_targets", _targets)
    probe = AsyncMock(return_value=("up", 12, "model.gguf", False))
    monkeypatch.setattr(route_catalog, "_probe_backend", probe)
    await route_catalog.refresh_route_health_cache(force=True)
    probed_urls = sorted(call.args[0].url for call in probe.call_args_list)
    assert probed_urls == ["http://atlas:8012", "http://atlas:8013",
                           "http://circe:8011", "http://circe:8014"]
    assert len(probed_urls) == 4, "5 routes, 4 distinct workers"


@pytest.mark.asyncio
async def test_both_routes_still_report_their_own_identity(monkeypatch):
    """Deduping the probe must not collapse the two rows into one."""
    monkeypatch.setattr(route_catalog, "get_route_targets", _targets)
    monkeypatch.setattr(route_catalog, "_probe_backend",
                        AsyncMock(return_value=("up", 12, "model.gguf", False)))
    payload = await route_catalog.get_routes_payload()
    ids = [r["id"] for r in payload["routes"]]
    assert ids == list(LLM_ROUTE_DISPLAY_ORDER)
    by_id = {r["id"]: r for r in payload["routes"]}
    assert by_id["quick"]["status"] == by_id["quick_background"]["status"] == "up"


def test_cold_cache_still_reports_priority(monkeypatch):
    """The Hub filters its picker on `priority`. Reporting None before the first health refresh
    would make a yielding lane look pickable for the first 15s of gateway uptime -- which is
    exactly when someone is most likely looking at a freshly restarted service."""
    monkeypatch.setattr(route_catalog, "get_route_targets", _targets)
    payload = route_catalog.build_routes_response()   # cache is empty
    by_id = {r["id"]: r for r in payload["routes"]}
    assert by_id["quick_background"]["status"] == "unknown"
    assert by_id["quick_background"]["priority"] == "background"
    assert by_id["quick_background"]["reserved_free_slots"] == 2


def test_unconfigured_routes_still_declare_what_they_are(monkeypatch):
    """Nothing configured at all. Every row is `not_configured` -- but a background lane says
    so regardless, because that is a property of the route, not of the route table."""
    monkeypatch.setattr(route_catalog, "get_route_targets", dict)
    payload = route_catalog.build_routes_response()
    by_id = {r["id"]: r for r in payload["routes"]}
    assert all(r["status"] == "not_configured" for r in payload["routes"])
    assert by_id["quick_background"]["priority"] == "background"
    assert by_id["quick"]["priority"] is None
    assert by_id["chat"]["priority"] is None


class TestReviewFixes:
    """Each of these is a hole a reviewer found in the first cut of this patch."""

    @pytest.mark.asyncio
    async def test_a_trailing_slash_does_not_defeat_the_url_dedup(self, monkeypatch):
        """The probe helpers all rstrip('/'), so keying the dedup on the raw string would treat
        one worker as two and send six calls where three would do."""
        targets = _targets()
        targets["quick_background"] = RouteTarget(
            url="http://atlas:8013/", served_by="atlas-worker-fast-1", backend="llamacpp",
            priority="background", reserved_free_slots=2,
        )
        monkeypatch.setattr(route_catalog, "get_route_targets", lambda: targets)
        probe = AsyncMock(return_value=("up", 12, "model.gguf", False))
        monkeypatch.setattr(route_catalog, "_probe_backend", probe)
        await route_catalog.refresh_route_health_cache(force=True)
        assert probe.call_count == 4, "5 routes, 4 distinct workers, trailing slash notwithstanding"

    def test_an_unconfigured_background_route_still_declares_itself(self, monkeypatch):
        """A route absent from LLM_GATEWAY_ROUTE_TABLE_JSON has no configured priority. Saying
        `None` there is what lets a consumer offer a yielding lane to a human."""
        targets = _targets()
        del targets["quick_background"]
        monkeypatch.setattr(route_catalog, "get_route_targets", lambda: targets)
        payload = route_catalog.build_routes_response()
        bg = next(r for r in payload["routes"] if r["id"] == "quick_background")
        assert bg["status"] == "not_configured"
        assert bg["priority"] == "background"

    @pytest.mark.asyncio
    async def test_a_configured_route_with_an_empty_url_is_down_not_absent(self, monkeypatch):
        """Before the URL dedup this was probed, failed, and reported `down` with its identity.
        Collapsing it into `not_configured` makes a misconfigured URL indistinguishable from a
        route that was never in the table."""
        targets = _targets()
        targets["chat"] = RouteTarget(url="", served_by="circe-worker-1", backend="llamacpp")
        monkeypatch.setattr(route_catalog, "get_route_targets", lambda: targets)
        monkeypatch.setattr(route_catalog, "_probe_backend",
                            AsyncMock(return_value=("up", 12, "model.gguf", False)))
        payload = await route_catalog.get_routes_payload()
        chat = next(r for r in payload["routes"] if r["id"] == "chat")
        assert chat["status"] == "down"
        assert chat["served_by"] == "circe-worker-1", "identity survives; it is misconfigured, not unknown"

    def test_probe_one_is_gone(self):
        """It had no callers after the dedup. Leaving it invites reintroducing per-route
        probing, which is exactly what the dedup removed."""
        assert not hasattr(route_catalog, "_probe_one")
