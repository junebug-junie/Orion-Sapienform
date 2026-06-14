"""Hub read-only proposal review client and routes."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def _reload_hub_modules(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, api_url: str = "http://proposal-review.test") -> None:
    monkeypatch.setenv("HUB_PROPOSAL_REVIEW_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("HUB_PROPOSAL_REVIEW_API_URL", api_url)
    for mod in (
        "app.settings",
        "scripts.settings",
        "scripts.proposal_review_client",
        "scripts.proposal_review_routes",
        "scripts.main",
    ):
        sys.modules.pop(mod, None)
    app_settings = importlib.import_module("app.settings")
    app_settings.get_settings.cache_clear()
    importlib.import_module("scripts.settings")
    importlib.import_module("scripts.proposal_review_client")
    importlib.import_module("scripts.proposal_review_routes")


@pytest.fixture
def hub_client(monkeypatch: pytest.MonkeyPatch):
    _reload_hub_modules(monkeypatch, enabled=False)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        yield client


def test_hub_proposal_review_disabled_returns_empty_state(hub_client: TestClient) -> None:
    response = hub_client.get("/api/proposal-review/pending")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["available"] is False
    assert body["state"] == "disabled"
    assert body["proposals"] == []
    assert body["count"] == 0


def test_hub_proposal_review_lists_pending_review_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    async def fake_list(*, status: str | None = "pending_review") -> dict:
        assert status == "pending_review"
        return {
            "proposals": [
                {
                    "proposal_id": "prop_demo_1",
                    "proposal_type": "memory_correction",
                    "title": "Demo memory correction",
                    "risk": "medium",
                    "status": "pending_review",
                    "attention_required": True,
                }
            ],
            "count": 1,
        }

    monkeypatch.setattr(routes.client, "list_proposals", fake_list)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        response = client.get("/api/proposal-review/pending")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["available"] is True
    assert body["count"] == 1
    assert body["proposals"][0]["status"] == "pending_review"


def test_hub_proposal_review_detail_fetches_record(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    expected = {
        "proposal_id": "prop_demo_1",
        "status": "pending_review",
        "attention_reason": "identity memory correction",
        "envelope": {"title": "Demo", "summary": "Fix belief", "proposal_type": "memory_correction"},
        "execution_eligibility": {"eligible": False, "reason": "pending human review"},
    }

    async def fake_get(proposal_id: str) -> dict:
        assert proposal_id == "prop_demo_1"
        return expected

    monkeypatch.setattr(routes.client, "get_proposal", fake_get)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        response = client.get("/api/proposal-review/proposals/prop_demo_1")
    assert response.status_code == 200
    assert response.json()["proposal_id"] == "prop_demo_1"
    assert response.json()["attention_reason"] == "identity memory correction"


def test_hub_proposal_review_unavailable_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_client as client_mod

    async def fake_list(*, status: str | None = "pending_review") -> dict:
        raise client_mod.ProposalReviewUnavailable("down")

    monkeypatch.setattr(client_mod, "list_proposals", fake_list)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        response = client.get("/api/proposal-review/pending")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["state"] == "unavailable"
    assert body["message"] == "Proposal review API unavailable."
    assert body["proposals"] == []


def test_hub_proposal_review_does_not_call_post_review_or_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_client as client_mod

    assert not hasattr(client_mod, "post_triage")
    assert not hasattr(client_mod, "post_review")
    assert not hasattr(client_mod, "triage_proposal")
    assert not hasattr(client_mod, "review_proposal")

    captured_methods: list[str] = []

    class FakeResponse:
        status = 200

        async def text(self) -> str:
            return '{"proposals":[],"count":0}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class FakeSession:
        def get(self, url: str, params: dict | None = None):
            captured_methods.append("GET")
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(client_mod.aiohttp, "ClientSession", lambda **kwargs: FakeSession())

    asyncio.run(client_mod.list_proposals(status="pending_review"))
    assert captured_methods == ["GET"]

    with pytest.raises(client_mod.ProposalReviewClientError):
        asyncio.run(client_mod._get_json("/proposals/prop_1/triage"))  # noqa: SLF001

    with pytest.raises(client_mod.ProposalReviewClientError):
        asyncio.run(client_mod._get_json("/proposals/prop_1/review"))  # noqa: SLF001

    routes_source = (HUB_ROOT / "scripts" / "proposal_review_routes.py").read_text(encoding="utf-8")
    assert "@router.post" not in routes_source
    assert "POST" not in routes_source
