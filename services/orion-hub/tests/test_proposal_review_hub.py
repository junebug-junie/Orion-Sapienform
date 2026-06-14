"""Hub proposal review client, routes, and review actions."""

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


@pytest.fixture(autouse=True)
def _cleanup_hub_path_pollution() -> None:
    """Prevent hub reload from shadowing orion-context-exec `app` in later tests."""
    yield
    hub_root = str(HUB_ROOT)
    while hub_root in sys.path:
        sys.path.remove(hub_root)
    for mod in list(sys.modules):
        if mod == "app" or mod.startswith("app."):
            sys.modules.pop(mod, None)


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


def test_hub_proposal_review_client_post_allowlist_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_client as client_mod

    client_mod._assert_post_path("/proposals/prop_1/review")  # noqa: SLF001

    for path in (
        "/proposals/prop_1/triage",
        "/proposals/prop_1/execute",
        "/triage",
        "/review",
    ):
        with pytest.raises(client_mod.ProposalReviewClientError):
            client_mod._assert_post_path(path)  # noqa: SLF001


def test_hub_proposal_review_client_rejects_non_get_http_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_client as client_mod

    class _NoopSession:
        def get(self, *args: object, **kwargs: object) -> object:
            return None

        def post(self, *args: object, **kwargs: object) -> object:
            return None

        async def __aenter__(self) -> _NoopSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(client_mod.aiohttp, "ClientSession", lambda **kwargs: _NoopSession())
    session = client_mod._RestrictedClientSession()  # noqa: SLF001
    base = "http://proposal-review.test"
    for method in ("put", "patch", "delete"):
        with pytest.raises(client_mod.ProposalReviewClientError, match="forbidden HTTP method"):
            asyncio.run(getattr(session, method)())
    with pytest.raises(client_mod.ProposalReviewClientError, match="forbidden proposal review path"):
        session.post(f"{base}/proposals/prop_1/triage", json={})


def test_hub_review_does_not_call_triage_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_client as client_mod

    assert hasattr(client_mod, "post_review")
    assert not hasattr(client_mod, "post_triage")
    assert not hasattr(client_mod, "triage_proposal")

    captured: list[tuple[str, str]] = []

    class FakeResponse:
        status = 200

        async def text(self) -> str:
            return '{"status":"rejected","execution_eligibility":{"eligible":false,"execution_requested":false}}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class FakeSession:
        def get(self, url: str, params: dict | None = None):
            captured.append(("GET", url))
            return FakeResponse()

        def post(self, url: str, **kwargs: object):
            captured.append(("POST", url))
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(client_mod.aiohttp, "ClientSession", lambda **kwargs: FakeSession())

    asyncio.run(
        client_mod.post_review(
            "prop_1",
            {"decision": "reject", "rationale": "no", "reviewer_type": "human", "reviewer_id": "hub"},
        )
    )
    assert captured == [("POST", "http://proposal-review.test/proposals/prop_1/review")]

    with pytest.raises(client_mod.ProposalReviewClientError):
        asyncio.run(client_mod._post_json("/proposals/prop_1/triage", body={"action": "store_only"}))  # noqa: SLF001

    with pytest.raises(client_mod.ProposalReviewClientError):
        asyncio.run(client_mod._post_json("/proposals/prop_1/execute", body={}))  # noqa: SLF001

    routes_source = (HUB_ROOT / "scripts" / "proposal_review_routes.py").read_text(encoding="utf-8")
    assert routes_source.count("@router.post") == 1
    assert "/proposals/{proposal_id}/review" in routes_source
    assert "/triage" not in routes_source


def test_hub_review_does_not_call_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    async def fake_post(proposal_id: str, body: dict) -> dict:
        assert proposal_id == "prop_demo_1"
        assert body["decision"] == "approve"
        return {
            "proposal_id": proposal_id,
            "status": "approved",
            "execution_eligibility": {
                "eligible": True,
                "execution_requested": False,
            },
        }

    monkeypatch.setattr(routes.client, "post_review", fake_post)
    import scripts.main as hub_main

    client_source = (HUB_ROOT / "scripts" / "proposal_review_client.py").read_text(encoding="utf-8")
    assert "execute" not in client_source.lower().split("_ALLOWED")[0]
    assert "/execute" not in client_source

    with TestClient(hub_main.app) as client:
        response = client.post(
            "/api/proposal-review/proposals/prop_demo_1/review",
            json={
                "decision": "approve",
                "rationale": "bounded and reversible",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "approved"
    assert payload["execution_eligibility"]["eligible"] is True
    assert payload["execution_eligibility"]["execution_requested"] is False
    assert "execution_receipt" not in payload
    assert "receipt" not in payload
    assert "changed_targets" not in payload


def test_hub_review_reject_posts_review_decision_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    captured: dict[str, object] = {}

    async def fake_post(proposal_id: str, body: dict) -> dict:
        captured["proposal_id"] = proposal_id
        captured["body"] = body
        return {
            "proposal_id": proposal_id,
            "status": "rejected",
            "execution_eligibility": {"eligible": False, "execution_requested": False},
        }

    monkeypatch.setattr(routes.client, "post_review", fake_post)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        response = client.post(
            "/api/proposal-review/proposals/prop_demo_1/review",
            json={"decision": "reject", "rationale": "unsupported evidence"},
        )

    assert response.status_code == 200
    assert captured["proposal_id"] == "prop_demo_1"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["decision"] == "reject"
    assert body["rationale"] == "unsupported evidence"
    assert response.json()["status"] == "rejected"
    assert response.json()["execution_eligibility"]["eligible"] is False


def test_hub_review_request_changes_posts_review_decision_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    captured: dict[str, object] = {}

    async def fake_post(proposal_id: str, body: dict) -> dict:
        captured["body"] = body
        return {
            "proposal_id": proposal_id,
            "status": "request_changes",
            "execution_eligibility": {"eligible": False, "execution_requested": False},
        }

    monkeypatch.setattr(routes.client, "post_review", fake_post)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        response = client.post(
            "/api/proposal-review/proposals/prop_demo_1/review",
            json={"decision": "request_changes", "rationale": "needs more evidence"},
        )

    assert response.status_code == 200
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["decision"] == "request_changes"
    assert response.json()["status"] == "request_changes"
    assert response.json()["execution_eligibility"]["eligible"] is False


def test_hub_review_approve_creates_eligibility_not_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    async def fake_post(proposal_id: str, body: dict) -> dict:
        assert body["decision"] == "approve"
        return {
            "proposal_id": proposal_id,
            "status": "approved",
            "review_decision": {"decision": "approve", "rationale": body["rationale"]},
            "execution_eligibility": {
                "eligible": True,
                "execution_requested": False,
                "reason": "approved by review decision",
            },
        }

    monkeypatch.setattr(routes.client, "post_review", fake_post)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        response = client.post(
            "/api/proposal-review/proposals/prop_demo_1/review",
            json={
                "decision": "approve",
                "rationale": "bounded and reversible",
                "constraints": {"note": "scope limited"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "approved"
    assert payload["execution_eligibility"]["eligible"] is True
    assert payload["execution_eligibility"]["execution_requested"] is False


def test_hub_review_requires_rationale(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_hub_modules(monkeypatch, enabled=True)
    import scripts.proposal_review_routes as routes

    called = {"value": False}

    async def fake_post(proposal_id: str, body: dict) -> dict:
        called["value"] = True
        return {}

    monkeypatch.setattr(routes.client, "post_review", fake_post)
    import scripts.main as hub_main

    with TestClient(hub_main.app) as client:
        empty = client.post(
            "/api/proposal-review/proposals/prop_demo_1/review",
            json={"decision": "reject", "rationale": "   "},
        )
        missing = client.post(
            "/api/proposal-review/proposals/prop_demo_1/review",
            json={"decision": "reject"},
        )

    assert empty.status_code == 422
    assert missing.status_code == 422
    assert called["value"] is False

    ui = (HUB_ROOT / "static" / "js" / "proposal-review-ui.js").read_text(encoding="utf-8")
    assert "Rationale is required" in ui


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
    for label in ("Approve", "Reject", "Request changes"):
        assert label in ui
    assert "/api/proposal-review/proposals/" in ui
    assert "/review" in ui
    assert "/triage" not in ui.lower()
    assert "execute" not in ui.lower()
    assert "fetch(" in ui
    ui_lower = ui.lower()
    for token in ('type="submit"',):
        assert token not in ui_lower
    routes_source = (HUB_ROOT / "scripts" / "proposal_review_routes.py").read_text(encoding="utf-8")
    assert "/proposals/{proposal_id}/review" in routes_source
