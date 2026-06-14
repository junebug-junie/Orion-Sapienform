"""Hub read-only proposal review client and routes."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

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
    for mod in list(sys.modules):
        if mod == "app" or mod.startswith("app."):
            sys.modules.pop(mod, None)
    for mod in (
        "scripts.settings",
        "scripts.proposal_review_client",
        "scripts.proposal_review_routes",
        "scripts.main",
    ):
        sys.modules.pop(mod, None)
    context_exec_dir = str(REPO_ROOT / "services" / "orion-context-exec")
    while context_exec_dir in sys.path:
        sys.path.remove(context_exec_dir)
    hub_root = str(HUB_ROOT)
    if hub_root in sys.path:
        sys.path.remove(hub_root)
    sys.path.insert(0, hub_root)
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


def test_hub_proposal_review_client_get_allowlist_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_client as client_mod

    allowed = (
        "/health",
        "/proposals",
        "/proposals/prop_1",
        "/proposals/prop_1/eligibility",
    )
    for path in allowed:
        client_mod._assert_get_path(path)  # noqa: SLF001

    forbidden = (
        "/proposals/prop_1/triage",
        "/proposals/prop_1/review",
        "/triage",
        "/review",
    )
    for path in forbidden:
        with pytest.raises(client_mod.ProposalReviewClientError):
            client_mod._assert_get_path(path)  # noqa: SLF001


def test_hub_proposal_review_client_rejects_non_get_http_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_client as client_mod

    class _NoopSession:
        def get(self, *args: object, **kwargs: object) -> object:
            return None

        async def __aenter__(self) -> _NoopSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(client_mod.aiohttp, "ClientSession", lambda **kwargs: _NoopSession())
    session = client_mod._GetOnlyClientSession()  # noqa: SLF001
    for method in ("post", "put", "patch", "delete"):
        with pytest.raises(client_mod.ProposalReviewClientError, match="forbidden HTTP method"):
            asyncio.run(getattr(session, method)())


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


def test_hub_pending_decisions_shows_denver_memory_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    denver_detail = {
        "proposal_id": "prop_denver_vertical",
        "status": "pending_review",
        "attention_required": True,
        "attention_reason": "memory correction involving identity",
        "risk": "medium",
        "envelope": {
            "title": "Memory correction: Denver identity claim",
            "proposal_type": "memory_correction_proposal",
            "summary": "Mark Denver location belief uncertain",
            "mutation_allowed": False,
            "requires_human_approval": True,
            "risk": "medium",
        },
        "inner_artifact_summary": {
            "artifact_type": "MemoryCorrectionProposalV1",
            "current_belief": "User is from Denver",
            "correction_type": "mark_uncertain",
            "rationale": "Unsupported Denver identity claim",
            "confidence": 0.5,
            "risk": "medium",
            "supporting_evidence": ["user mentioned Colorado once"],
            "contradicting_evidence": [],
            "missing_evidence": ["verified Denver residency evidence"],
            "mutation_allowed": False,
            "requires_human_approval": True,
        },
        "execution_eligibility": {"eligible": False, "reason": "pending human review"},
    }

    async def fake_list(*, status: str | None = "pending_review") -> dict:
        assert status == "pending_review"
        return {
            "proposals": [
                {
                    "proposal_id": "prop_denver_vertical",
                    "proposal_type": "memory_correction_proposal",
                    "title": "Memory correction: Denver identity claim",
                    "risk": "medium",
                    "status": "pending_review",
                    "attention_required": True,
                }
            ],
            "count": 1,
        }

    async def fake_get(proposal_id: str) -> dict:
        assert proposal_id == "prop_denver_vertical"
        return denver_detail

    monkeypatch.setattr(routes.client, "list_proposals", fake_list)
    monkeypatch.setattr(routes.client, "get_proposal", fake_get)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        pending = client.get("/api/proposal-review/pending")
        detail = client.get("/api/proposal-review/proposals/prop_denver_vertical")

    assert pending.status_code == 200
    body = pending.json()
    assert body["count"] == 1
    row = body["proposals"][0]
    assert row["proposal_type"] == "memory_correction_proposal"
    assert row["status"] == "pending_review"
    assert "Denver" in row["title"] or row["proposal_id"] == "prop_denver_vertical"

    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["status"] == "pending_review"
    assert "identity" in detail_body["attention_reason"].lower()
    inner = detail_body["inner_artifact_summary"]
    assert "denver" in inner["current_belief"].lower()
    assert inner["correction_type"] == "mark_uncertain"
    assert inner["mutation_allowed"] is False
    assert inner["requires_human_approval"] is True
    assert inner.get("rationale")
    assert inner.get("risk") in {"low", "medium", "high", "unknown"}
    assert inner.get("confidence") is not None
    assert inner.get("supporting_evidence") or inner.get("missing_evidence")

    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    ui = (HUB_ROOT / "static" / "js" / "proposal-review-ui.js").read_text(encoding="utf-8")
    assert "Pending Decisions" in template
    assert 'id="proposalReviewPanel"' in template
    for token in (
        "Current belief:",
        "Proposed correction:",
        "Rationale:",
        "Evidence:",
        "Risk:",
        "Confidence:",
        "mutation_allowed=",
        "requires_human_approval=",
    ):
        assert token in ui
    ui_lower = ui.lower()
    assert 'method="post"' not in ui_lower
    assert "fetch(" in ui  # read-only GET via apiFetch
    for forbidden in ("/review", "/triage", "request-changes", "request changes"):
        assert forbidden not in ui_lower
    for forbidden_button in ('id="approve', 'id="reject', ">approve<", ">reject<", "approve proposal", "reject proposal"):
        assert forbidden_button not in ui_lower
    for token in ('type="submit"',):
        assert token not in ui_lower
    routes_source = (HUB_ROOT / "scripts" / "proposal_review_routes.py").read_text(encoding="utf-8")
    assert "@router.post" not in routes_source
