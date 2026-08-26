"""Static-content contract for the Reverie visual cockpit (template + JS).

Modeled on test_cabinet_sensors_panel.py's style: assert the rendered
template and the JS file actually declare the pieces the cockpit needs
(diagram container, pager container, cursor-based pagination, pipeline
stage data) -- a template that renders and a server that starts are not
proof the changed interaction is actually wired (AGENTS.md §9).
"""
from __future__ import annotations

import re
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
REVERIE_JS = (HUB_ROOT / "static" / "js" / "reverie-tab.js").read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    """Slice out one top-level `function name() { ... }` body by brace
    matching, so an assertion can be scoped to what that function actually
    does -- not just "this string appears somewhere in the file", which a
    prior review finding caught passing on an unrelated line (module-scope
    variable init) while the real reset code it claimed to cover was gone."""
    m = re.search(rf"function {re.escape(name)}\s*\([^)]*\)\s*{{", REVERIE_JS)
    assert m, f"function {name}() not found in reverie-tab.js"
    start = m.end() - 1  # position of the opening brace
    depth = 0
    for i in range(start, len(REVERIE_JS)):
        if REVERIE_JS[i] == "{":
            depth += 1
        elif REVERIE_JS[i] == "}":
            depth -= 1
            if depth == 0:
                return REVERIE_JS[start : i + 1]
    raise AssertionError(f"unbalanced braces scanning function {name}()")


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
    # Mesh-context seeding (2026-08-26) must point at the real reader, not
    # just describe it in prose -- same inspectable-evidence bar.
    assert "load_recent_reverie_interpretation" in REVERIE_JS


def test_js_uses_cursor_based_pagination_not_offset() -> None:
    """OFFSET pagination is a deliberately rejected approach here (review
    finding: no stable meaning against a table with live ~600s inserts) --
    pin the cursor-based replacement, not the id/class names that could be
    satisfied by either implementation."""
    assert "cursorStack" in REVERIE_JS
    assert "pageIndex" in REVERIE_JS
    assert "has_more" in REVERIE_JS
    assert "before=" in REVERIE_JS
    assert "reveriePagerNext" in REVERIE_JS
    assert "reveriePagerPrev" in REVERIE_JS
    # The rejected OFFSET approach must be gone from the fetch URL/state --
    # not from comments, which legitimately explain why it was rejected.
    assert "visualOffset" not in REVERIE_JS
    assert "offset=" not in REVERIE_JS


def test_refresh_actually_resets_pagination_state() -> None:
    """Scoped to refresh()'s own body (see _function_body) so this can only
    pass if the reset code is really there -- not satisfied by an unrelated
    line elsewhere in the file, the exact gap a prior review finding caught."""
    body = _function_body("refresh")
    assert "cursorStack = [null]" in body
    assert "pageIndex = 0" in body


def test_load_visual_guards_against_out_of_order_responses() -> None:
    """A double-click or Refresh-mid-flight race (review finding) must not
    let a stale response overwrite newer data -- pin the sequence-token
    guard inside loadVisual() specifically."""
    body = _function_body("loadVisual")
    assert "visualRequestSeq" in body
    assert "seq !== visualRequestSeq" in body


def test_empty_state_shown_on_any_page_not_just_the_first() -> None:
    """Review finding: the old check only showed a 'no chains' message when
    offset === 0, so a later page returning zero rows rendered a silently
    blank grid. Pin that the empty-state branch no longer gates on being
    page 0."""
    body = _function_body("loadVisual")
    assert "if (!data.chains.length)" in body
    assert "pageIndex === 0" in body  # still used to pick *which* message, not whether to show one


def test_js_shows_the_realized_prompt_and_error_state() -> None:
    assert "chain.prompt" in REVERIE_JS
    assert "chain.error" in REVERIE_JS
    assert "egress" in REVERIE_JS


def test_js_shows_the_real_mesh_context_that_influenced_the_prompt() -> None:
    """2026-08-26: the visual chain now weaves a real mesh signal (the
    parallel text-reverie chain's own interpretation) into its prompt to
    break the pure self-referential loop that fell into multi-hour attractor
    basins. The cockpit must show it in `renderVisualChain`'s own body, not
    just as a decorative string elsewhere in the file."""
    body = _function_body("renderVisualChain")
    assert "chain.mesh_context" in body
    # Real explicit disclosure when a run had no mesh signal available --
    # never silently renders nothing with no explanation.
    assert "no mesh context available" in body
    # Review finding: the fallback message must be driven by the server's own
    # ground-truth flags (visual_chain.py::_prompt_source_flags), never
    # guessed client-side -- guessing produced a false "fell back to
    # continuity-only" claim on runs that used neither prior nor mesh context.
    assert "chain.used_mesh" in body
    assert "chain.used_prior" in body
