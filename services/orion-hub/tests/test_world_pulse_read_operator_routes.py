"""Reading tab routes: guard, error mapping, durable cancel, page wiring (no DB)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from orion.schemas.reading import ReadingRequestedV1, ReadingToolBindingV1
from orion.world_pulse_read.operator import OperatorActionError, operator_request

HUB_ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"X-Requested-With": "orion-hub", "Content-Type": "application/json"}


class _Ctx:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


class _Pool:
    def acquire(self):
        return _Ctx()


@pytest.fixture
def routes(monkeypatch):
    from scripts import world_pulse_read_routes as routes

    monkeypatch.setattr(routes, "_pool", lambda: _Pool())
    monkeypatch.setattr(routes, "_settings", lambda: SimpleNamespace(
        HUB_READING_DURABLE_URL="http://durable.test", HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS=5.0,
    ))
    monkeypatch.setattr(routes, "_source_ref", lambda: None)
    monkeypatch.delenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", raising=False)
    return routes


@pytest.fixture
def client(routes):
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_operator_context_is_juniper_only_and_not_model_claimable():
    assert operator_request(url="https://example.org/x").requested_by == "juniper"
    with pytest.raises(ValidationError):
        ReadingRequestedV1(url="https://example.org/x", requested_by="orion", invocation_context="operator")
    with pytest.raises(ValidationError):
        ReadingToolBindingV1(invocation_context="operator", parent_run_id="r", parent_trace_id="t")


@pytest.mark.parametrize("headers", [
    {"Content-Type": "application/json"},
    {"X-Requested-With": "orion-hub", "Content-Type": "text/plain"},
])
def test_controls_require_hub_page_headers(client, routes, monkeypatch, headers):
    called = []

    async def cancel(conn, seed_id):
        called.append(seed_id)
        return {"action": "skipped", "stage": 1}

    monkeypatch.setattr(routes.reading_operator, "cancel_read", cancel)
    resp = client.post("/world-pulse-read/api/reads/reading:x/cancel", headers=headers, content="{}")
    assert resp.status_code == 403
    assert resp.json()["detail"] == "reading_control_requires_hub_page"
    assert called == []


def test_configured_operator_token_is_required_via_header_or_cookie(client, routes, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "sekrit")

    ages = []

    async def retry(conn, seed_id, *, stage, digest_item_max_age_sec):
        ages.append(digest_item_max_age_sec)
        return {"action": "requeued", "stage": stage}

    monkeypatch.setattr(routes.reading_operator, "retry_read", retry)
    url = "/world-pulse-read/api/reads/reading:x/retry"
    assert client.post(url, headers=HEADERS, json={"stage": 1}).json()["detail"] == "operator_guard_rejected"
    wrong = {**HEADERS, "X-Orion-Operator-Token": "nope"}
    assert client.post(url, headers=wrong, json={"stage": 1}).status_code == 403
    ok = {**HEADERS, "X-Orion-Operator-Token": "sekrit"}
    assert client.post(url, headers=ok, json={"stage": 2}).json() == {"action": "requeued", "stage": 2}
    client.cookies.set("orion_operator_token", "sekrit")
    assert client.post(url, headers=HEADERS, json={"stage": 1}).status_code == 200
    # Retry must refuse exactly what the Stage 1 stale sweep would re-skip.
    assert ages == [5 * 86400.0, 5 * 86400.0]


def test_retry_rejects_bad_stage_before_touching_queue(client):
    resp = client.post("/world-pulse-read/api/reads/reading:x/retry", headers=HEADERS, json={"stage": 3})
    assert resp.status_code == 422


def test_refusals_map_to_operator_codes(client, routes, monkeypatch):
    async def cancel(conn, seed_id):
        raise OperatorActionError("not_active")

    async def detail(conn, seed_id):
        raise OperatorActionError("not_found", 404)

    async def listing(conn, **kwargs):
        raise OperatorActionError("unknown_phase", 400)

    monkeypatch.setattr(routes.reading_operator, "cancel_read", cancel)
    monkeypatch.setattr(routes.reading_operator, "read_detail", detail)
    monkeypatch.setattr(routes.reading_operator, "list_reads", listing)
    resp = client.post("/world-pulse-read/api/reads/reading:x/cancel", headers=HEADERS, content="{}")
    assert (resp.status_code, resp.json()["detail"]) == (409, "not_active")
    assert client.get("/world-pulse-read/api/reads/missing").status_code == 404
    assert client.get("/world-pulse-read/api/reads?phase=bogus").status_code == 400


def test_reads_are_503_without_database(client, routes, monkeypatch):
    monkeypatch.setattr(routes, "_pool", lambda: None)
    assert client.get("/world-pulse-read/api/reads").json()["detail"] == "reading_db_unavailable"


def test_list_and_detail_pass_through_filters(client, routes, monkeypatch):
    seen = {}

    async def listing(conn, **kwargs):
        seen.update(kwargs)
        return {"items": [], "total": 0, "phase": kwargs["phase"]}

    async def detail(conn, seed_id):
        return {"seed_id": seed_id}

    monkeypatch.setattr(routes.reading_operator, "list_reads", listing)
    monkeypatch.setattr(routes.reading_operator, "read_detail", detail)
    resp = client.get("/world-pulse-read/api/reads?phase=failed&kind=reading&include_stale=true&limit=5&offset=10")
    assert resp.json()["phase"] == "failed"
    assert seen == {"phase": "failed", "kind": "reading", "include_stale": True, "limit": 5, "offset": 10}
    seed = "finding:a84a6755:169b39cc"
    assert client.get(f"/world-pulse-read/api/reads/{seed}").json() == {"seed_id": seed}


def _durable(monkeypatch, routes, handler):
    real = httpx.AsyncClient
    calls = []

    def factory(*args, **kwargs):
        def record(request):
            calls.append(str(request.url))
            return handler(request)
        return real(*args, transport=httpx.MockTransport(record), **kwargs)

    monkeypatch.setattr(routes.httpx, "AsyncClient", factory)
    return calls


@pytest.mark.parametrize("run_status,finished", [
    ("cancelled", False), ("running", False), ("completed", True), ("failed", True),
])
def test_cancel_bound_run_goes_through_durable_runner(client, routes, monkeypatch, run_status, finished):
    async def cancel(conn, seed_id):
        return {"action": "cancel_durable_run", "stage": 1, "run_id": "reading-abc"}

    monkeypatch.setattr(routes.reading_operator, "cancel_read", cancel)
    calls = _durable(monkeypatch, routes, lambda r: httpx.Response(200, json={"status": run_status}))
    resp = client.post("/world-pulse-read/api/reads/reading:x/cancel", headers=HEADERS, content="{}")
    assert resp.status_code == 200
    assert resp.json()["durable_status"] == run_status
    assert resp.json()["run_already_finished"] is finished
    assert calls == ["http://durable.test/runs/reading-abc/cancel"]


@pytest.mark.parametrize("status,code,detail", [
    (404, 409, "durable_run_not_found_retry_shortly"),
    (500, 502, "durable_cancel_failed:500"),
])
def test_cancel_durable_failures_are_explicit(client, routes, monkeypatch, status, code, detail):
    async def cancel(conn, seed_id):
        return {"action": "cancel_durable_run", "stage": 2, "run_id": "reading-abc"}

    monkeypatch.setattr(routes.reading_operator, "cancel_read", cancel)
    _durable(monkeypatch, routes, lambda r: httpx.Response(status, json={}))
    resp = client.post("/world-pulse-read/api/reads/reading:x/cancel", headers=HEADERS, content="{}")
    assert (resp.status_code, resp.json()["detail"]) == (code, detail)


@pytest.mark.parametrize("url,detail", [
    ("not a url", "invalid_source_url"),
    ("http://127.0.0.1/admin", "source must be a public HTTP(S) URL"),
])
def test_submit_rejections_surface_url_policy_codes(client, routes, monkeypatch, url, detail):
    from orion.world_pulse_read.urls import normalize_source_url

    async def submit(conn, *, url, why_now, title, bus, source):
        normalize_source_url(url)
        operator_request(url=url, why_now=why_now, title=title)
        return {}

    monkeypatch.setattr(routes.reading_operator, "submit_read", submit)
    resp = client.post("/world-pulse-read/api/reads", headers=HEADERS, json={"url": url})
    assert (resp.status_code, resp.json()["detail"]) == (400, detail)


def test_submit_returns_ingress_receipt(client, routes, monkeypatch):
    seen = {}

    async def submit(conn, **kwargs):
        seen.update(kwargs)
        return {"seed_id": "reading:1", "status": "queued"}

    monkeypatch.setattr(routes.reading_operator, "submit_read", submit)
    resp = client.post("/world-pulse-read/api/reads", headers=HEADERS,
                       json={"url": " https://example.org/p ", "why_now": " because "})
    assert resp.json() == {"seed_id": "reading:1", "status": "queued"}
    assert (seen["url"], seen["why_now"], seen["title"]) == ("https://example.org/p", "because", "")


def test_reading_page_and_tab_are_wired():
    from scripts import world_pulse_read_routes as routes

    main_src = (HUB_ROOT / "scripts" / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(reading_page_router)" in main_src
    template = (HUB_ROOT / "templates" / "reading.html").read_text(encoding="utf-8")
    assert "/static/js/reading.js?v={{HUB_UI_ASSET_VERSION}}" in template
    assert (HUB_ROOT / "static" / "js" / "reading.js").is_file()
    index = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="readingTabButton"' in index and 'data-hash-target="#reading"' in index
    assert 'data-panel="reading"' in index and 'src="/reading"' in index
    assert "/static/js/reading_tab.js?v={{HUB_UI_ASSET_VERSION}}" in index
    assert (HUB_ROOT / "static" / "js" / "reading_tab.js").is_file()
    assert any(getattr(r, "path", "") == "/reading" for r in routes.page_router.routes)
