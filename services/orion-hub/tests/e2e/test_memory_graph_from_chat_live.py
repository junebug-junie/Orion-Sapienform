"""
Live Playwright verification: Memory graph from chat (bridge modal).

Requires a running Hub. Set HUB_E2E_BASE_URL (default http://127.0.0.1:8080).
Artifacts: services/orion-hub/tests/e2e/artifacts/memory-graph-from-chat/<case>/

Run:
  cd .worktrees/fix-memory-graph-strict-suggest-draft-v1
  PYTHONPATH=. ../../venv/bin/python -m pytest \
    services/orion-hub/tests/e2e/test_memory_graph_from_chat_live.py -v --tb=short
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from memory_graph_from_chat_cases import LIVE_CASES, STALE_UI_PHRASES

E2E_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = E2E_DIR / "artifacts" / "memory-graph-from-chat"
HUB_BASE = os.environ.get("HUB_E2E_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
SUGGEST_TIMEOUT_MS = int(os.environ.get("HUB_E2E_SUGGEST_TIMEOUT_MS", "210000"))


def _artifact_dir(case_id: str) -> Path:
    d = ARTIFACT_ROOT / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _hub_reachable() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"{HUB_BASE}/", timeout=5) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _is_strict_suggest_draft(obj: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("ontology_version") != "orionmem-2026-05":
        return False
    for key in ("utterance_ids", "entities", "situations", "edges", "dispositions"):
        if key not in obj or not isinstance(obj[key], list):
            return False
    return True


def _is_evidence_only(obj: dict[str, Any]) -> bool:
    keys = set(obj.keys())
    allowed = {"utterance_ids", "utterance_text_by_id", "ontology_version"}
    has_ids = "utterance_ids" in keys
    missing = [k for k in ("entities", "situations", "edges", "dispositions") if k not in keys]
    return has_ids and bool(missing)


def _entity_labels(obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for ent in obj.get("entities") or []:
        if isinstance(ent, dict) and ent.get("label"):
            out.append(str(ent["label"]).lower())
    return out


def _looks_like_prose_outside_json(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    try:
        json.loads(s)
        return False
    except json.JSONDecodeError:
        pass
    if s.startswith("{") and s.endswith("}"):
        return False
    return len(s) > 80 and not s.lstrip().startswith("{")


@pytest.fixture(scope="module")
def playwright_browser():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=os.environ.get("HUB_E2E_HEADED") != "1")
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def hub_available():
    if not _hub_reachable():
        pytest.skip(f"Hub not reachable at {HUB_BASE} — start stack then re-run")
    return HUB_BASE


@pytest.mark.parametrize("case", LIVE_CASES, ids=[c["id"] for c in LIVE_CASES])
def test_memory_graph_from_chat_live(case: dict[str, Any], playwright_browser, hub_available: str):
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    base = hub_available
    case_id = str(case["id"])
    art = _artifact_dir(case_id)

    page = playwright_browser.new_page(viewport={"width": 1400, "height": 1000})
    console_lines: list[str] = []
    page_errors: list[str] = []
    network_events: list[dict[str, Any]] = []
    suggest_request: dict[str, Any] | None = None
    suggest_response: dict[str, Any] | None = None
    chat_requests: list[str] = []

    page.on("console", lambda msg: console_lines.append(f"{msg.type}: {msg.text}"))
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    def on_request(req):
        url = req.url
        if "/api/chat" in url and req.method == "POST":
            chat_requests.append(url)
        if "/api/memory/graph/suggest" in url and req.method == "POST":
            nonlocal suggest_request
            try:
                suggest_request = {"url": url, "method": req.method, "body": req.post_data_json}
            except Exception:
                suggest_request = {"url": url, "method": req.method, "body_raw": req.post_data}

    def on_response(resp):
        url = resp.url
        entry: dict[str, Any] = {
            "url": url,
            "status": resp.status,
            "method": resp.request.method,
        }
        if "/api/memory/graph/suggest" in url:
            nonlocal suggest_response
            try:
                suggest_response = {"status": resp.status, "body": resp.json()}
            except Exception:
                try:
                    suggest_response = {"status": resp.status, "body_text": resp.text()}
                except Exception as exc:
                    suggest_response = {"status": resp.status, "error": str(exc)}
        if "/api/memory/graph/suggest" in url or "/api/chat" in url:
            network_events.append(entry)

    page.on("request", on_request)
    page.on("response", on_response)

    url = f"{base}/?hub_e2e=1"
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_function("() => window.__ORION_HUB_E2E__ != null", timeout=30000)

    page.evaluate(
        """(payload) => {
          window.__ORION_HUB_E2E__.seedMemoryGraphTurns(payload.turns);
          window.__ORION_HUB_E2E__.openMemoryGraphBridgeForAssistantTurn(payload.assistantTurnId);
        }""",
        {"turns": case["turns"], "assistantTurnId": case["assistant_turn_id"]},
    )

    page.wait_for_selector("#memoryGraphBridgeModal:not(.hidden)", timeout=15000)
    page.screenshot(path=str(art / "screenshot_before_suggest.png"), full_page=True)

    modal_text_before = page.locator("#memoryGraphBridgeModal").inner_text()
    _write(art / "modal_text.txt", modal_text_before)

    checkboxes = page.locator('#memoryGraphBridgeTurnList input[type="checkbox"]')
    count = checkboxes.count()
    for i in range(count):
        checkboxes.nth(i).check()

    page.click("#memoryGraphBridgeSuggest")

    try:
        page.wait_for_function(
            """() => {
              const ta = document.getElementById('memoryGraphBridgeDraft');
              return ta && ta.value && ta.value.trim().length > 2;
            }""",
            timeout=SUGGEST_TIMEOUT_MS,
        )
    except PlaywrightTimeout:
        page.screenshot(path=str(art / "screenshot_after_suggest.png"), full_page=True)
        _write(art / "browser_console.log", "\n".join(console_lines))
        _write(art / "page_errors.log", "\n".join(page_errors))
        _write(art / "network_summary.json", json.dumps(network_events, indent=2))
        _write(art / "full_page_html.html", page.content())
        pytest.fail(f"Suggest did not populate draft within {SUGGEST_TIMEOUT_MS}ms — see {art}")

    page.wait_for_timeout(500)
    page.screenshot(path=str(art / "screenshot_after_suggest.png"), full_page=True)

    draft_raw = page.locator("#memoryGraphBridgeDraft").input_value()
    _write(art / "draft_json.txt", draft_raw)

    parsed: dict[str, Any] | None = None
    parse_err = ""
    try:
        parsed = json.loads(draft_raw)
        _write(art / "draft_json.parsed.json", json.dumps(parsed, indent=2))
    except json.JSONDecodeError as exc:
        parse_err = str(exc)

    if suggest_request is not None:
        _write(art / "suggest_request.json", json.dumps(suggest_request, indent=2))
    if suggest_response is not None:
        _write(art / "suggest_response.json", json.dumps(suggest_response, indent=2))

    _write(art / "browser_console.log", "\n".join(console_lines))
    _write(art / "page_errors.log", "\n".join(page_errors))
    _write(art / "network_summary.json", json.dumps(network_events, indent=2))

    status_text = page.locator("#memoryGraphBridgeStatus").inner_text()
    diag_text = page.evaluate(
        """() => {
          const el = document.getElementById('memoryGraphBridgeDiagnostics');
          return el ? (el.textContent || '').trim() : '';
        }"""
    )
    _write(art / "diagnostics_text.txt", f"status:\n{status_text}\n\ndiagnostics:\n{diag_text}")

    modal_after = page.locator("#memoryGraphBridgeModal").inner_text()
    _write(art / "modal_text.txt", modal_after)

    failures: list[str] = []

    if not any("/api/memory/graph/suggest" in e.get("url", "") for e in network_events):
        failures.append("A: /api/memory/graph/suggest was not called")
    if chat_requests:
        failures.append("A: /api/chat was called during suggest flow")

    if parse_err or parsed is None:
        failures.append(f"Draft JSON did not parse: {parse_err}")
    elif not _is_strict_suggest_draft(parsed):
        failures.append("Draft JSON is not strict SuggestDraftV1 shape")
    elif _is_evidence_only(parsed):
        failures.append("Draft JSON is evidence-only envelope")
    elif _looks_like_prose_outside_json(draft_raw):
        failures.append("Draft textarea contains prose outside JSON")

    combined_ui = f"{status_text}\n{modal_after}\n{diag_text}"
    for phrase in STALE_UI_PHRASES:
        if phrase.lower() in combined_ui.lower():
            failures.append(f"F: stale UI copy contains {phrase!r}")

    if parsed and case.get("expect_nonempty_graph"):
        ents = parsed.get("entities") or []
        sits = parsed.get("situations") or []
        if not ents and not sits:
            failures.append("Expected nonempty graph (entities/situations) but got empty arrays")

    if parsed and case.get("expect_user_entity"):
        labels = _entity_labels(parsed)
        if not any("user" in lb or "juniper" in lb for lb in labels):
            failures.append("Expected User/Juniper entity in draft")

    if parsed and case.get("expect_orion_entity"):
        labels = _entity_labels(parsed)
        if not any("orion" in lb for lb in labels):
            failures.append("Expected Orion entity in draft")

    summary = {
        "case": case_id,
        "hub_base": base,
        "suggest_called": any("/api/memory/graph/suggest" in e.get("url", "") for e in network_events),
        "chat_called": bool(chat_requests),
        "entity_count": len(parsed.get("entities", [])) if parsed else 0,
        "situation_count": len(parsed.get("situations", [])) if parsed else 0,
        "edge_count": len(parsed.get("edges", [])) if parsed else 0,
        "has_user_entity": any("user" in lb or "juniper" in lb for lb in _entity_labels(parsed or {})),
        "has_orion_entity": any("orion" in lb for lb in _entity_labels(parsed or {})),
        "status": status_text,
        "failures": failures,
    }
    _write(art / "run_summary.json", json.dumps(summary, indent=2))

    if failures:
        _write(art / "full_page_html.html", page.content())
        pytest.fail(
            f"Case {case_id} failed ({len(failures)} issue(s)):\n"
            + "\n".join(f"  - {f}" for f in failures)
            + f"\nArtifacts: {art}"
        )

    page.close()
