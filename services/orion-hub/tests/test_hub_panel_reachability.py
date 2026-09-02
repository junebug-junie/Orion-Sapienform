"""Every Hub panel must be reachable from the Hub UI.

Why this exists: the Sentience Program board shipped in PR #2026 as a working
page at `/sentience-program`, with a route, a template, an asset and passing
tests -- and no link from anywhere in Hub. It was completely unreachable through
the UI, and that was only noticed when Juniper asked which tab to click.

Nothing caught it because every existing panel test asserts its OWN wiring by
string match, so a panel with no test simply has nothing checking it. This is
the generic form: it enumerates `data-panel` sections straight out of the
rendered template and requires each to be reachable, so a future panel is
covered without anyone remembering to add a test for it.
"""

from __future__ import annotations

import re
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
INDEX_HTML = (HUB / "templates" / "index.html").read_text()
APP_JS = (HUB / "static" / "js" / "app.js").read_text()

# Panels deliberately not reachable via a top-level tab + app.js setActiveTab.
# Each needs a real, checked reason -- this is not a place to silence failures.
PANELS_WITHOUT_APP_JS_TAB = {
    # Relocated 2026-09-02 from a top-level tab into the Biometrics modal's
    # Cabinet subview; see the comment above `id="cabinet"` in index.html and
    # biometrics-view.js. Intentionally has no nav entry of its own.
    "cabinet",
    # Has a nav entry, but routes itself: self_observability.js owns this
    # panel's show/hide and hash handling instead of app.js's setActiveTab.
    "self-observability",
}


def _panels() -> set[str]:
    return set(re.findall(r'data-panel="([a-z0-9-]+)"', INDEX_HTML))


def _nav_targets() -> set[str]:
    return set(re.findall(r'data-hash-target="#([a-z0-9-]+)"', INDEX_HTML))


def test_every_panel_has_a_nav_entry():
    """A panel with no nav entry cannot be clicked to."""
    missing = {
        p for p in _panels() - PANELS_WITHOUT_APP_JS_TAB if p not in _nav_targets()
    }
    assert not missing, (
        f"panels with no nav entry (unreachable by clicking): {sorted(missing)}"
    )


def test_every_panel_can_be_activated():
    """A nav entry with no setActiveTab call does nothing when clicked."""
    missing = {
        p
        for p in _panels() - PANELS_WITHOUT_APP_JS_TAB
        if f'setActiveTab("{p}")' not in APP_JS
    }
    assert not missing, (
        f"panels with no setActiveTab call in app.js: {sorted(missing)}"
    )


def test_exception_list_is_not_stale():
    """Every named exception must still be a real panel.

    Without this, a panel deleted or renamed leaves a dead entry that would
    silently excuse a genuinely unreachable panel of the same name later.
    """
    stale = PANELS_WITHOUT_APP_JS_TAB - _panels()
    assert not stale, f"exception list names panels that no longer exist: {sorted(stale)}"


# ---------------------------------------------------------------------------
# The panel this test file was written for
# ---------------------------------------------------------------------------


def test_sentience_program_panel_is_fully_wired():
    """All the pieces, not just the section -- each was a separate way to fail."""
    assert 'id="sentience-program" data-panel="sentience-program"' in INDEX_HTML
    assert 'id="sentienceProgramTabButton"' in INDEX_HTML
    assert 'data-hash-target="#sentience-program"' in INDEX_HTML
    assert 'src="/sentience-program"' in INDEX_HTML
    assert 'href="/sentience-program"' in INDEX_HTML  # "Open standalone"

    for fragment in (
        'document.getElementById("sentienceProgramTabButton")',
        'document.getElementById("sentience-program")',
        'const isSentienceProgram = effectiveTab === "sentience-program"',
        'sentienceProgramPanel.classList.toggle("hidden", !isSentienceProgram)',
        "styleTabButton(sentienceProgramTabButton, isSentienceProgram)",
        'setActiveTab("sentience-program")',
        '|| h === "#sentience-program"',
    ):
        assert fragment in APP_JS, f"app.js missing wiring: {fragment}"


def test_sentience_program_board_exposes_the_panel_contract():
    """The Hub shell drives panels through activate/refresh; the page must offer it."""
    board_js = (HUB / "static" / "js" / "sentience-program.js").read_text()
    assert "window.OrionSentienceProgram" in board_js
    assert "refresh:" in board_js and "activate:" in board_js
