from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

INDEX_HTML = HUB_ROOT / "templates" / "index.html"
APP_JS = HUB_ROOT / "static" / "js" / "app.js"
MIND_PROVENANCE_JS = HUB_ROOT / "static" / "js" / "mind_provenance.js"


def _ancestor_ids(html: str, target_id: str) -> list[str]:
    """Open-element path (tag#id) above the element with ``target_id``."""
    from html.parser import HTMLParser

    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    class _P(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.found: list[str] | None = None

        def handle_starttag(self, tag, attrs):
            el_id = dict(attrs).get("id") or ""
            if el_id == target_id and self.found is None:
                self.found = list(self.stack)
            if tag not in void:
                self.stack.append(f"{tag}#{el_id}")

        def handle_endtag(self, tag):
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i].split("#", 1)[0] == tag:
                    del self.stack[i:]
                    break

    parser = _P()
    parser.feed(html)
    assert parser.found is not None, f"#{target_id} not found"
    return parser.found


def test_mind_runs_modal_is_sibling_of_schedule_modal_not_nested() -> None:
    """mindRunsModal must not live under #scheduleModal.hidden or it never paints.

    Checked structurally: the old check matched the exact closing-tag text
    right before mindRunsModal, which broke as soon as another modal (GPU
    leases) was placed between the two -- without any nesting change.
    """
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="scheduleModalHistory"' in index_html
    mind_path = _ancestor_ids(index_html, "mindRunsModal")
    schedule_path = _ancestor_ids(index_html, "scheduleModal")
    assert not any(p.endswith("#scheduleModal") for p in mind_path), mind_path
    assert mind_path == schedule_path, (mind_path, schedule_path)


def test_index_has_mind_tab_and_panel() -> None:
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="mindTabButton"' in index_html
    assert 'href="#mind"' in index_html
    assert 'data-hash-target="#mind"' in index_html
    assert ">Mind<" in index_html
    assert '<section id="mind" data-panel="mind"' in index_html
    assert 'id="mindHoursInput"' in index_html
    assert 'id="mindRefreshButton"' in index_html
    assert 'id="mindFilterOk"' in index_html
    assert 'id="mindDefaultOnSendToggle"' in index_html
    assert 'id="mindRunsTableBody"' in index_html
    assert 'id="mindRunsModal"' in index_html
    assert 'id="mindRunsModalList"' in index_html
    assert 'id="mindRunsModalDetails"' in index_html


def test_app_js_wires_mind_hash_and_recent_api() -> None:
    app_js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("mindTabButton")' in app_js
    assert 'getElementById("mind")' in app_js
    assert 'setActiveTab("mind")' in app_js
    assert 'history.replaceState(null, "", "#mind")' in app_js
    assert 'window.location.hash === "#mind"' in app_js or 'h === "#mind"' in app_js
    assert "/api/mind/runs/recent" in app_js
    assert "/api/mind/runs?" in app_js
    assert "/api/mind/runs/" in app_js
    assert "refreshMindRuns" in app_js
    assert "openMindRunsModal" in app_js
    assert "mindRunsEmptyDiagnostics" in app_js
    assert "formatMindRunsApiError" in app_js
    assert "context_session_id" in app_js
    assert "meta.sessionId = orionSessionId" in app_js
    assert "payload.context.metadata.mind_enabled = true" in app_js
    assert "OrionMindProvenance.renderMindProvenanceSections" in app_js


def test_app_js_lane_api_uses_global_this_not_node_global() -> None:
    """Bare `global` is undefined in browsers and aborts DOMContentLoaded before tab wiring."""
    app_js = APP_JS.read_text(encoding="utf-8")
    assert "global.OrionHubGroundedSmallLane" not in app_js
    assert "globalThis" in app_js and "OrionHubGroundedSmallLane" in app_js


def test_mind_provenance_js_exports_normalizers_and_renderers() -> None:
    mind_js = MIND_PROVENANCE_JS.read_text(encoding="utf-8")
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    assert "mind_provenance.js" in index_html
    for symbol in (
        "normalizeMindRunProvenance",
        "normalizeMindDerailments",
        "normalizeMindPhaseRows",
        "normalizeMindDecision",
        "renderMindProvenance",
        "renderMindDerailmentCallouts",
        "renderMindPhaseTable",
        "Where it went off rails",
        "Cognition path",
        "Phase provenance",
        "MIND_PROVENANCE_FIXTURES",
    ):
        assert symbol in mind_js, symbol
