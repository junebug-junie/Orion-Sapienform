"""Behavioral tests for the Cockpit Soft HUD modal (cockpit-hud.js).

Runs the real JS via node (same pattern as test_turn_trace_panel_ui.py)
rather than only asserting the source text contains expected strings.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = REPO_ROOT / "services" / "orion-hub"
COCKPIT_HUD_JS_PATH = HUB_ROOT / "static" / "js" / "cockpit-hud.js"
COCKPIT_HUD_CSS_PATH = HUB_ROOT / "static" / "css" / "cockpit-hud.css"
INDEX_HTML_PATH = HUB_ROOT / "templates" / "index.html"


def _node_call(method: str, payload: Dict[str, Any] | None = None) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for cockpit HUD behavioral tests")
    if not COCKPIT_HUD_JS_PATH.is_file():
        raise AssertionError(f"missing {COCKPIT_HUD_JS_PATH}")
    arg = "undefined" if payload is None else json.dumps(payload)
    script = f"""
const fs = require('fs');
global.window = {{}};
eval(fs.readFileSync({json.dumps(str(COCKPIT_HUD_JS_PATH))}, 'utf8'));
const api = window.OrionCockpitHud;
if (!api || typeof api[{json.dumps(method)}] !== 'function') {{
  throw new Error('OrionCockpitHud.{method} missing');
}}
const html = api[{json.dumps(method)}]({arg});
console.log(typeof html === 'string' ? html : JSON.stringify(html));
"""
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"node {method} failed")
    return proc.stdout


def test_build_shell_includes_soft_hud_hooks():
    html = _node_call("buildShell", {"correlationId": "c1"})
    assert "cockpit-hud" in html
    assert "cockpit-inspector" in html
    assert "cockpit-scrubber" in html


def test_render_orders_hops_and_marks_gap():
    html = _node_call(
        "renderFixture",
        {
            "hops": [
                {"seq": 0, "stage": "ingress", "status": "gap", "visor_line": "gap"},
                {"seq": 1, "stage": "stance_decision", "status": "ok", "visor_line": "proceed", "raw": {"x": 1}},
            ],
            "selectedSeq": 1,
        },
    )
    assert "stance_decision" in html
    assert "cockpit-hop-gap" in html or 'status="gap"' in html or 'data-status="gap"' in html


def test_inspector_shows_prompt_section_for_motor_boot():
    html = _node_call(
        "renderFixture",
        {
            "hops": [
                {
                    "seq": 4,
                    "stage": "motor_boot",
                    "status": "ok",
                    "visor_line": "motor_boot · 12 chars",
                    "summary": {"prompt_char_len": 12},
                    "raw": {"prompt": "HELLO PREFIX", "prompt_char_len": 12},
                },
            ],
            "selectedSeq": 4,
        },
    )
    assert 'data-cockpit-section="prompt"' in html or "Prompt/Prefix" in html
    assert "HELLO PREFIX" in html


def test_render_fixture_shows_selected_raw_and_visor():
    html = _node_call(
        "renderFixture",
        {
            "hops": [
                {"seq": 0, "stage": "ingress", "status": "gap", "visor_line": "gap"},
                {"seq": 1, "stage": "stance_decision", "status": "ok", "visor_line": "proceed", "raw": {"x": 1}},
            ],
            "selectedSeq": 1,
        },
    )
    assert "proceed" in html
    assert '"x": 1' in html or '"x":1' in html


def test_ingest_hop_reorders_by_seq() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for cockpit HUD behavioral tests")
    if not COCKPIT_HUD_JS_PATH.is_file():
        raise AssertionError(f"missing {COCKPIT_HUD_JS_PATH}")
    script = f"""
const fs = require('fs');
global.window = {{}};
eval(fs.readFileSync({json.dumps(str(COCKPIT_HUD_JS_PATH))}, 'utf8'));
const api = window.OrionCockpitHud;
api.ingestHop({{seq: 2, stage: 'finalize', status: 'ok', visor_line: 'voice', raw: {{done: true}}}});
api.ingestHop({{seq: 0, stage: 'ingress', status: 'gap', visor_line: 'gap'}});
const html = api.render();
console.log(html);
"""
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "node ingestHop failed")
    html = proc.stdout
    assert html.find("ingress") < html.find("finalize")
    assert "cockpit-hop-gap" in html or 'data-status="gap"' in html


def test_api_exports_open_close_ingest_and_render() -> None:
    src = COCKPIT_HUD_JS_PATH.read_text(encoding="utf-8")
    for name in (
        "function open",
        "function close",
        "function ingestHop",
        "function markComplete",
        "function buildShell",
        "function render",
        "function renderFixture",
        "OrionCockpitHud",
    ):
        assert name in src
    assert "/api/chat/turn/" in src
    assert "/cockpit" in src


def test_index_links_css_js_and_modal_root() -> None:
    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    assert 'id="cockpitHudRoot"' in html
    assert "/static/css/cockpit-hud.css" in html
    assert "/static/js/cockpit-hud.js?v={{HUB_UI_ASSET_VERSION}}" in html
    assert "/static/js/app.js?v={{HUB_UI_ASSET_VERSION}}" in html
    hud_pos = html.index("/static/js/cockpit-hud.js?v={{HUB_UI_ASSET_VERSION}}")
    app_pos = html.index("/static/js/app.js?v={{HUB_UI_ASSET_VERSION}}")
    assert hud_pos < app_pos


def test_css_is_soft_hud_not_neon() -> None:
    css = COCKPIT_HUD_CSS_PATH.read_text(encoding="utf-8")
    assert "backdrop-filter" in css
    assert "vignette" in css
    lowered = css.lower()
    assert "#00ff00" not in lowered
    assert "#39ff14" not in lowered
    assert "neon" not in lowered


def test_app_js_wires_cockpit_beside_turn_trace() -> None:
    text = (REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "appendTurnTracePanel" in text
    assert "OrionCockpitHud" in text
    assert "cockpit_hop" in text
    start = text.index("function appendCockpitButton")
    end = text.index("function renderThoughtProcessSection", start)
    block = text[start:end]
    assert "thoughtProcessApi.resolveCorrelationId" in block
    assert "mindCorrelationFromMeta" in block


def test_open_ignores_stale_fetch_after_correlation_changes() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for cockpit HUD behavioral tests")
    if not COCKPIT_HUD_JS_PATH.is_file():
        raise AssertionError(f"missing {COCKPIT_HUD_JS_PATH}")
    script = f"""
const fs = require('fs');
global.window = {{}};
eval(fs.readFileSync({json.dumps(str(COCKPIT_HUD_JS_PATH))}, 'utf8'));
const api = window.OrionCockpitHud;

(async () => {{
  let resolveA;
  const pendingA = new Promise((resolve) => {{ resolveA = resolve; }});
  global.fetch = async function (url) {{
    if (String(url).includes('corr-A')) {{
      const payload = await pendingA;
      return {{ ok: true, json: async () => payload }};
    }}
    return {{
      ok: true,
      json: async () => ({{
        hops: [{{seq: 0, stage: 'ingress-B', status: 'ok', visor_line: 'from-B'}}],
        complete: true,
      }}),
    }};
  }};
  const pA = api.open({{ correlationId: 'corr-A', apiBaseUrl: '' }});
  const pB = api.open({{ correlationId: 'corr-B', apiBaseUrl: '' }});
  resolveA({{
    hops: [{{seq: 0, stage: 'ingress-A', status: 'ok', visor_line: 'from-A'}}],
    complete: true,
  }});
  await Promise.all([pA, pB]);
  const html = api.render();
  if (html.includes('from-A') || html.includes('ingress-A')) {{
    throw new Error('stale A hops ingested');
  }}
  if (!html.includes('corr-B')) {{
    throw new Error('expected corr-B in render');
  }}
  if (!html.includes('from-B') && !html.includes('ingress-B')) {{
    throw new Error('expected B hops in render');
  }}
  console.log('ok');
}})().catch((err) => {{
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
}});
"""
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "node stale open fetch failed")
    assert "ok" in proc.stdout
