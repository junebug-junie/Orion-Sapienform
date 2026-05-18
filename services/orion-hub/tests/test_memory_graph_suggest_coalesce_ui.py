from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DRAFT_UI_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "memory-graph-draft-ui.js"


def _node_coalesce(payload: Dict[str, Any], *, utterance_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for coalescer behavioral tests")
    options: Dict[str, Any] = {}
    if utterance_ids is not None:
        options["utteranceIds"] = utterance_ids
    script = f"""
const fs = require('fs');
global.window = {{}};
eval(fs.readFileSync({json.dumps(str(DRAFT_UI_PATH))}, 'utf8'));
const fn = window.OrionMemoryGraphDraftUI.coalesceMemoryGraphSuggestEnvelope;
const out = fn({json.dumps(payload)}, {json.dumps(options)});
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "node coalesce failed")
    return json.loads(proc.stdout.strip())


def _parse_draft_text(draft_text: str) -> Dict[str, Any]:
    obj = json.loads(draft_text)
    assert isinstance(obj, dict)
    return obj


def test_coalesce_prefers_draft_object() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }
    out = _node_coalesce({"ok": True, "draft": draft})
    parsed = _parse_draft_text(out["draftText"])
    assert parsed == draft
    assert out.get("error") in (None, "")


def test_coalesce_prefers_appendix_c_json() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }
    out = _node_coalesce({"ok": True, "appendix_c_json": json.dumps(draft)})
    parsed = _parse_draft_text(out["draftText"])
    assert parsed == draft
    assert out.get("error") in (None, "")


def test_coalesce_rejects_prose_text() -> None:
    prose = "There's a kind of sacred tension in those moments..."
    out = _node_coalesce({"ok": True, "text": prose}, utterance_ids=["u1"])
    parsed = _parse_draft_text(out["draftText"])
    assert parsed["ontology_version"] == "orionmem-2026-05"
    assert parsed["utterance_ids"] == ["u1"]
    assert parsed["entities"] == []
    assert prose not in out["draftText"]
    assert out.get("error") == "invalid_model_output"
    diag = out.get("diagnostics") or {}
    assert diag.get("prose_rejected") is True


def test_coalesce_failed_api_returns_empty_draft() -> None:
    out = _node_coalesce(
        {
            "ok": False,
            "error": "memory_graph_suggest_failed",
            "attempts": [{"phase": "no_json_object"}],
            "validation_errors": ["no_json_object"],
        },
        utterance_ids=["u9"],
    )
    parsed = _parse_draft_text(out["draftText"])
    assert parsed["utterance_ids"] == ["u9"]
    assert out.get("error") == "memory_graph_suggest_failed"
    diag = out.get("diagnostics") or {}
    assert "no_json_object" in (diag.get("validation_errors") or [])


def test_coalesce_source_exports_envelope_helper() -> None:
    src = DRAFT_UI_PATH.read_text(encoding="utf-8")
    assert "function coalesceMemoryGraphSuggestEnvelope" in src
    assert "coalesceMemoryGraphSuggestEnvelope," in src


def test_coalesce_never_assigns_prose_pattern_in_helpers() -> None:
    """Regression: coalescer must not assign data.text directly to draftText without validation."""
    src = DRAFT_UI_PATH.read_text(encoding="utf-8")
    block = src.split("function coalesceMemoryGraphSuggestEnvelope", 1)[1].split("function coalesceChatSuggestDraft", 1)[0]
    assert "draftText = data.text" not in block
    assert re.search(r"draftText\s*=\s*[^;]*data\.text", block) is None
