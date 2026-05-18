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

_EMPTY_DRAFT = {
    "ontology_version": "orionmem-2026-05",
    "utterance_ids": [],
    "entities": [],
    "situations": [],
    "edges": [],
    "dispositions": [],
    "utterance_text_by_id": {},
}


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


def _assert_valid_suggest_draft_shape(parsed: Dict[str, Any]) -> None:
    assert parsed["ontology_version"] == "orionmem-2026-05"
    for key in ("utterance_ids", "entities", "situations", "edges", "dispositions"):
        assert isinstance(parsed[key], list)
    assert isinstance(parsed.get("utterance_text_by_id", {}), dict)


def test_coalesce_prefers_draft_object() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [{"id": "e1", "label": "Test", "entityKind": "person"}],
        "situations": [],
        "edges": [],
        "dispositions": [],
        "utterance_text_by_id": {"u1": "hello"},
    }
    out = _node_coalesce({"ok": True, "draft": draft})
    parsed = _parse_draft_text(out["draftText"])
    assert parsed == draft
    assert out.get("error") in (None, "")


def test_coalesce_prefers_appendix_c_json() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [{"id": "e1", "label": "Test", "entityKind": "person"}],
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
    _assert_valid_suggest_draft_shape(parsed)
    assert parsed["utterance_ids"] == ["u1"]
    assert parsed["entities"] == []
    assert prose not in out["draftText"]
    assert out.get("error") == "invalid_model_output"
    diag = out.get("diagnostics") or {}
    assert diag.get("prose_rejected") is True


def test_coalesce_rejects_evidence_envelope_only() -> None:
    evidence = {
        "utterance_ids": ["u1"],
        "utterance_text_by_id": {"u1": "k, off to shower"},
    }
    out = _node_coalesce({"ok": True, "draft": evidence}, utterance_ids=["u1"])
    parsed = _parse_draft_text(out["draftText"])
    _assert_valid_suggest_draft_shape(parsed)
    assert parsed["utterance_ids"] == ["u1"]
    assert parsed["entities"] == []
    assert parsed.get("utterance_text_by_id", {}).get("u1") == "k, off to shower"
    assert out.get("error") == "evidence_envelope_not_draft"


def test_coalesce_accepts_valid_empty_suggest_draft() -> None:
    draft = dict(_EMPTY_DRAFT)
    draft["utterance_ids"] = ["u1", "u2"]
    out = _node_coalesce({"ok": True, "draft": draft})
    parsed = _parse_draft_text(out["draftText"])
    assert parsed == draft
    assert out.get("error") in (None, "")
    assert out.get("graphEmpty") is True


def test_coalesce_accepts_valid_nonempty_suggest_draft() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [{"id": "e1", "label": "Orion", "entityKind": "agent"}],
        "situations": [],
        "edges": [],
        "dispositions": [],
        "utterance_text_by_id": {},
    }
    out = _node_coalesce({"ok": True, "draft": draft})
    parsed = _parse_draft_text(out["draftText"])
    assert len(parsed["entities"]) == 1
    assert out.get("error") in (None, "")


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
    _assert_valid_suggest_draft_shape(parsed)
    assert out.get("error") == "memory_graph_suggest_failed"
    diag = out.get("diagnostics") or {}
    assert "no_json_object" in (diag.get("validation_errors") or [])


def test_coalesce_source_exports_envelope_helper() -> None:
    src = DRAFT_UI_PATH.read_text(encoding="utf-8")
    assert "function coalesceMemoryGraphSuggestEnvelope" in src
    assert "coalesceMemoryGraphSuggestEnvelope," in src
    assert "function emptySuggestDraft" in src
    assert "function looksLikeEvidenceEnvelopeOnly" in src


def test_coalesce_never_assigns_prose_pattern_in_helpers() -> None:
    """Regression: coalescer must not assign data.text directly to draftText without validation."""
    src = DRAFT_UI_PATH.read_text(encoding="utf-8")
    block = src.split("function coalesceMemoryGraphSuggestEnvelope", 1)[1].split(
        "function extractLlmGatewayDraftFromSteps", 1
    )[0]
    assert "draftText = data.text" not in block
    assert re.search(r"draftText\s*=\s*[^;]*data\.text", block) is None


def test_strict_draft_shape_rejects_utterance_ids_only() -> None:
    src = DRAFT_UI_PATH.read_text(encoding="utf-8")
    assert 'obj.ontology_version !== SUGGEST_DRAFT_ONTOLOGY_VERSION' in src
    assert "Array.isArray(obj.utterance_ids)" in src
    assert "Array.isArray(obj.entities)" in src


def test_status_strings_exclude_no_durable_memory_candidate() -> None:
    src = DRAFT_UI_PATH.read_text(encoding="utf-8")
    assert "No durable memory candidate found" not in src
    assert "no_durable_memory_candidate" not in src
    assert "Loaded validated role-grounded SuggestDraftV1 JSON." in src
    assert "Extractor did not return a valid role-grounded SuggestDraftV1" in src


def test_coalesce_success_nonempty_uses_role_grounded_status() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [{"id": "entity:user", "label": "User", "entityKind": "person", "surfaceForms": ["I"]}],
        "situations": [
            {
                "id": "sit:1",
                "utterance_ids": ["u1"],
                "label": "test",
            }
        ],
        "edges": [],
        "dispositions": [],
    }
    script = f"""
const fs = require('fs');
global.window = {{}};
eval(fs.readFileSync({json.dumps(str(DRAFT_UI_PATH))}, 'utf8'));
const out = window.OrionMemoryGraphDraftUI.coalesceMemoryGraphSuggestEnvelope({json.dumps({"ok": True, "draft": draft})}, {{}});
console.log(window.OrionMemoryGraphDraftUI.formatSuggestCoalesceUserStatus(out));
"""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False, timeout=30)
    assert proc.returncode == 0
    assert "Loaded validated role-grounded SuggestDraftV1 JSON." in proc.stdout
