"""Field Attention operator tab: wiring contract + static-content assertions.

This tab visualizes FieldAttentionFrameV1 (orion/attention/field_attention/,
PR #1484/#1488) -- a different subsystem from the Attention Organ tab
(attention-organ.js / attention_organ_routes.py), which visualizes
orion-heartbeat's AST/HOT dissipation ensemble. Modeled directly on
test_attention_organ_page.py: this tab is the *fifth* panel registered in
index.html that also needs five separate app.js registration points
(element binding, missing-panel fallback, visibility toggle, button styling,
hash routing) plus a script tag plus a nav anchor -- missing any one produces
a tab that silently falls back to Hub on a specific interaction (deep link,
switching away and back, a fresh reload) rather than failing loudly.

The backend route (GET /api/substrate/attention/latest) is not touched by
this patch -- it is already covered by test_substrate_attention_debug_api.py,
which this file does not duplicate.
"""

from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
FIELD_ATTENTION_JS = (HUB_ROOT / "static" / "js" / "field-attention.js").read_text(encoding="utf-8")
ATTENTION_ORGAN_JS = (HUB_ROOT / "static" / "js" / "attention-organ.js").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Template wiring
# --------------------------------------------------------------------------


def test_template_declares_nav_button_panel_and_script_tag() -> None:
    assert 'data-hash-target="#field-attention"' in INDEX_HTML
    assert 'id="fieldAttentionTabButton"' in INDEX_HTML
    assert 'id="field-attention" data-panel="field-attention"' in INDEX_HTML
    assert "/static/js/field-attention.js?v={{HUB_UI_ASSET_VERSION}}" in INDEX_HTML
    for mount_id in (
        "fieldAttentionStatus",
        "fieldAttentionHeader",
        "fieldAttentionDominantTargets",
        "fieldAttentionSuppressedTargets",
        "fieldAttentionBottomStrip",
        "fieldAttentionRefreshBtn",
    ):
        assert f'id="{mount_id}"' in INDEX_HTML, mount_id


def test_template_panel_does_not_reuse_attention_organ_naming() -> None:
    """These are two distinct subsystems -- a collision here would mean one
    tab's DOM lookups silently bind to the other's elements."""
    field_attention_section_start = INDEX_HTML.index('id="field-attention" data-panel="field-attention"')
    field_attention_section_end = INDEX_HTML.index("</section>", field_attention_section_start)
    section_html = INDEX_HTML[field_attention_section_start:field_attention_section_end]
    assert "attentionOrgan" not in section_html
    assert 'id="attention-organ"' not in section_html


def test_template_notes_the_overall_salience_normalization_design() -> None:
    """Design requirement: don't headline overall_salience as a big gauge,
    and the on-page explanation must cite the real normalization design
    (normalize_across_targets() / build_attention_frame()), not an invented
    rationale."""
    assert "normalize_across_targets" in INDEX_HTML
    assert "build_attention_frame" in INDEX_HTML
    assert "not a" in INDEX_HTML and "gauge" in INDEX_HTML


def test_template_notes_confidence_is_binary_in_practice() -> None:
    assert "grounded" in INDEX_HTML.lower()
    assert "cold-start" in INDEX_HTML.lower()


# --------------------------------------------------------------------------
# app.js registration points
# --------------------------------------------------------------------------


def test_app_js_registers_the_panel_in_every_place_setactivetab_needs() -> None:
    """All five registration points, not just the visible one -- see this
    module's docstring for why each omission fails silently and differently."""
    assert 'document.getElementById("fieldAttentionTabButton")' in APP_JS  # element binding (button)
    assert 'document.getElementById("field-attention")' in APP_JS  # element binding (panel)
    assert 'tabKey === "field-attention" && !fieldAttentionPanel' in APP_JS  # fallback
    assert 'const isFieldAttention = effectiveTab === "field-attention";' in APP_JS  # flag
    assert 'fieldAttentionPanel.classList.toggle("hidden", !isFieldAttention);' in APP_JS  # toggle
    assert "styleTabButton(fieldAttentionTabButton, isFieldAttention);" in APP_JS  # styling
    assert 'h === "#field-attention" && fieldAttentionPanel' in APP_JS  # hash routing
    assert 'history.replaceState(null, "", "#field-attention");' in APP_JS  # click handler


def test_app_js_resets_an_unresolvable_field_attention_hash_to_hub() -> None:
    """Mirrors attention-organ's own fallback: an unresolvable #field-attention
    hash (panel/button missing) must not leave the URL pointing at a tab that
    never actually rendered."""
    assert '|| h === "#field-attention"' in APP_JS


def test_app_js_drives_the_panels_poll_lifecycle_on_tab_switch() -> None:
    """Polling must stop when the tab is hidden -- both activate() and
    deactivate() have to actually be called from setActiveTab, not just
    defined in the module."""
    assert "window.OrionFieldAttention.activate()" in APP_JS
    assert "window.OrionFieldAttention.deactivate()" in APP_JS


def test_app_js_does_not_confuse_field_attention_with_attention_organ() -> None:
    """These are separate globals/ids; a copy-paste error here would make one
    tab silently drive the other's lifecycle."""
    assert "window.OrionFieldAttention" not in ATTENTION_ORGAN_JS
    assert "window.OrionAttentionOrgan" not in FIELD_ATTENTION_JS


# --------------------------------------------------------------------------
# field-attention.js content
# --------------------------------------------------------------------------


def test_field_attention_js_is_standalone_and_reads_only_its_own_api() -> None:
    assert "window.OrionFieldAttention" in FIELD_ATTENTION_JS
    assert "activate:" in FIELD_ATTENTION_JS and "deactivate:" in FIELD_ATTENTION_JS
    assert '"/api/substrate/attention/latest"' in FIELD_ATTENTION_JS
    # Must not depend on app.js's globals -- same standalone contract as
    # attention-organ.js and the Causal Geometry bundle.
    assert "window.OrionHub" not in FIELD_ATTENTION_JS
    # Read-only surface: no writes to any Orion endpoint from this tab.
    assert '"POST"' not in FIELD_ATTENTION_JS and "method: 'POST'" not in FIELD_ATTENTION_JS


def test_field_attention_js_renders_structure_not_a_raw_json_dump() -> None:
    assert "JSON.stringify" not in FIELD_ATTENTION_JS
    for renderer in (
        "renderHeader",
        "renderTargetsTable",
        "renderBottomStrip",
        "renderFrame",
    ):
        assert f"function {renderer}(" in FIELD_ATTENTION_JS, renderer


def test_field_attention_js_candidate_badge_matches_real_reason_prefixes() -> None:
    """Candidate A/B are derived from reasons[0]'s real text, produced by
    orion/attention/field_attention/selectors.py's select_node_targets()
    (Candidate A) and _novelty_targets() (Candidate B) -- these exact prefix
    strings must stay byte-for-byte in sync with the producer or every
    target would silently fall back to the "no candidate" badge."""
    assert "precision-weighted prediction-error salience" in FIELD_ATTENTION_JS
    assert "Candidate B novelty-only salience" in FIELD_ATTENTION_JS
    assert "function candidateBadgeFor(" in FIELD_ATTENTION_JS


def test_field_attention_js_confidence_rendered_as_binary_badge_not_gradient() -> None:
    """Design requirement: confidence_score is binary in practice (verified
    live: 10,747 recent target-observations, only ever 0.0 or 1.0) -- must be
    a two-state badge, not a progress bar or sparkline."""
    assert "function confidenceBadge(" in FIELD_ATTENTION_JS
    assert '"grounded"' in FIELD_ATTENTION_JS
    assert '"cold-start"' in FIELD_ATTENTION_JS


def test_field_attention_js_dominant_and_suppressed_targets_both_rendered() -> None:
    """Suppressed targets must stay visible (muted), not hidden -- this is
    where an operator would have caught the pre-#1488 confidence-bucket bug
    live."""
    assert "dominant_targets" in FIELD_ATTENTION_JS
    assert "suppressed_targets" in FIELD_ATTENTION_JS
    assert "muted" in FIELD_ATTENTION_JS


def test_field_attention_js_renders_recent_perturbations_and_warnings() -> None:
    assert "recent_perturbations" in FIELD_ATTENTION_JS
    assert "warnings" in FIELD_ATTENTION_JS


def test_field_attention_js_does_not_mislabel_system_targets_as_cold_start() -> None:
    """Review finding (2026-07-30): select_system_targets() in
    orion/attention/field_attention/selectors.py hardcodes
    confidence_score=0.0 for the 'field:recent_perturbations' system target
    as a schema-required placeholder, not a real "not grounded yet" reading
    -- no confidence concept applies to that aggregate signal at all.
    Rendering it through the ordinary binary grounded/cold-start badge would
    show a permanent, never-resolving "cold-start" that reads as "still
    warming up." confidenceBadge() must special-case target_kind === "system"
    into a distinct "n/a" badge instead."""
    assert "function confidenceBadge(value, targetKind)" in FIELD_ATTENTION_JS
    assert 'targetKind === "system"' in FIELD_ATTENTION_JS
    assert '"n/a"' in FIELD_ATTENTION_JS
    assert "confidenceBadge(target.confidence_score, target.target_kind)" in FIELD_ATTENTION_JS


def test_field_attention_js_guards_polling_on_real_visibility() -> None:
    """Review finding from attention-organ.js's own history (M1): app.js's
    setActiveTab is not the only thing that can hide this panel --
    self_observability.js hides every section[data-panel] directly and
    preventDefault()s its click, so deactivate() never fires on that path.
    The timer must check the DOM, not trust the router."""
    assert 'els.panel.classList.contains("hidden")' in FIELD_ATTENTION_JS
    guard_region = FIELD_ATTENTION_JS[
        FIELD_ATTENTION_JS.index("function startTimer") : FIELD_ATTENTION_JS.index("function activate")
    ]
    assert 'contains("hidden")' in guard_region
    assert "deactivate();" in guard_region


def test_field_attention_js_handles_404_as_no_frame_yet_not_a_crash() -> None:
    """The backend 404s when no frame has been persisted yet
    (substrate_attention_routes.py's attention_latest()) -- that is a real,
    expected state (fresh deploy, substrate cold-start), not an error to
    surface as a red "failed to load" the way a genuine fetch failure would."""
    assert "resp.status === 404" in FIELD_ATTENTION_JS


def test_field_attention_js_does_not_grow_unbounded_client_state() -> None:
    """Unlike attention-organ.js's liveTrace, this tab renders only the
    latest single frame on every poll -- there is no rolling client-side
    array to bound in the first place, so nothing here should be
    accumulating history across polls."""
    assert "state.lastFrame = frame;" in FIELD_ATTENTION_JS
    assert ".push(" not in FIELD_ATTENTION_JS
