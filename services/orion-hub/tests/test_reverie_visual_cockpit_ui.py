"""Static-content contract for the Reverie visual cockpit (template + JS).

Modeled on test_cabinet_sensors_panel.py's style: assert the rendered
template and the JS file actually declare the pieces the cockpit needs
(diagram container, pager container, pagination fetch using offset, pipeline
stage data) -- a template that renders and a server that starts are not
proof the changed interaction is actually wired (AGENTS.md §9).
"""
from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
REVERIE_JS = (HUB_ROOT / "static" / "js" / "reverie-tab.js").read_text(encoding="utf-8")


def test_template_declares_diagram_and_pager_containers() -> None:
    assert 'id="reverieVisualDiagram"' in INDEX_HTML
    assert 'id="reverieVisualPager"' in INDEX_HTML
    assert 'id="reverieVisualGrid"' in INDEX_HTML
    assert "/static/js/reverie-tab.js?v={{HUB_UI_ASSET_VERSION}}" in INDEX_HTML


def test_js_renders_pipeline_diagram_with_real_code_pointers() -> None:
    assert "PIPELINE_STAGES" in REVERIE_JS
    assert "EGRESS_NODE" in REVERIE_JS
    # Every stage must point at a real, inspectable file:function -- not a
    # decorative label (CLAUDE.md §0A "inspectable evidence" requirement).
    assert "services/orion-thought/app/visual_chain.py" in REVERIE_JS
    assert "services/orion-thought/app/store.py" in REVERIE_JS
    assert "renderPipelineDiagram" in REVERIE_JS


def test_js_uses_real_offset_based_pagination() -> None:
    assert "visualOffset" in REVERIE_JS
    assert "offset=${visualOffset}" in REVERIE_JS
    assert "reveriePagerNext" in REVERIE_JS
    assert "reveriePagerPrev" in REVERIE_JS
    # Refresh must reset paging back to the first page, not silently continue
    # from wherever the operator had paged to.
    assert "visualOffset = 0" in REVERIE_JS


def test_js_shows_the_realized_prompt_and_error_state() -> None:
    assert "chain.prompt" in REVERIE_JS
    assert "chain.error" in REVERIE_JS
    assert "egress" in REVERIE_JS
