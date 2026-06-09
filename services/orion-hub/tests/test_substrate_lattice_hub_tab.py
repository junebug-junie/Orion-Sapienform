from __future__ import annotations

import os
import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

INDEX_HTML = HUB_ROOT / "templates" / "index.html"
APP_JS = HUB_ROOT / "static" / "js" / "app.js"
LATTICE_STATIC = HUB_ROOT / "static" / "substrate-lattice.html"
LATTICE_JS = HUB_ROOT / "static" / "js" / "substrate-lattice.js"


def test_index_has_substrate_lattice_tab_nav_button() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="substrateLatticeTabButton"' in html
    assert 'href="#substrate-lattice"' in html
    assert 'data-hash-target="#substrate-lattice"' in html
    assert ">Substrate Lattice<" in html


def test_index_has_substrate_lattice_section_and_frame() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert '<section id="substrate-lattice" data-panel="substrate-lattice"' in html
    assert 'id="substrateLatticeFrame"' in html
    assert 'src="/static/substrate-lattice.html?v={{HUB_UI_ASSET_VERSION}}"' in html


def test_app_js_wires_substrate_lattice_hash_and_tab() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("substrateLatticeTabButton")' in js
    assert 'getElementById("substrate-lattice")' in js
    assert 'getElementById("substrateLatticeFrame")' in js
    assert 'setActiveTab("substrate-lattice")' in js
    assert "#substrate-lattice" in js


def test_lattice_static_page_has_root_ids() -> None:
    html = LATTICE_STATIC.read_text(encoding="utf-8")
    for needle in [
        'id="substrateLatticeRoot"',
        'id="producerLaneRail"',
        'id="transportProofChain"',
        'id="gateOverlay"',
        'id="latticeInspector"',
    ]:
        assert needle in html, f"Missing: {needle}"


def test_lattice_js_exists_and_has_fetch_calls() -> None:
    js = LATTICE_JS.read_text(encoding="utf-8")
    assert "/api/substrate-lattice/transport/latest" in js
    assert "/api/substrate-lattice/lanes" in js
    assert "/api/substrate-lattice/transport/gates" in js
