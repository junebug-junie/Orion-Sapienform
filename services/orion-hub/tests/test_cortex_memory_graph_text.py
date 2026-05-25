from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_imports()

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:ollapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def test_hub_memory_graph_suggest_text_reads_step_content_when_final_text_empty() -> None:
    from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
    from orion.schemas.cortex.schemas import StepExecutionResult

    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [{"id": "e1", "label": "User", "entityKind": "person", "surfaceForms": ["I"]}],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }
    step = StepExecutionResult(
        status="success",
        verb_name="memory_graph_suggest",
        step_name="llm_memory_graph_suggest",
        order=0,
        result={"LLMGatewayService": {"content": json.dumps(draft)}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    cr = CortexClientResult(
        ok=True,
        mode="brain",
        verb="memory_graph_suggest",
        status="success",
        final_text="",
        steps=[step],
    )
    resp = CortexChatResult(cortex_result=cr, final_text="")
    from scripts.cortex_memory_graph_text import hub_memory_graph_suggest_text

    text, diag = hub_memory_graph_suggest_text(resp)
    assert "ontology_version" in text
    assert diag["selected_text_source"] == "LLMGatewayService.content"


def test_hub_memory_graph_suggest_text_reads_raw_openai_choices_when_content_empty() -> None:
    from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
    from orion.schemas.cortex.schemas import StepExecutionResult

    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(draft),
                },
                "finish_reason": "stop",
            }
        ]
    }
    step = StepExecutionResult(
        status="success",
        verb_name="memory_graph_suggest",
        step_name="llm_memory_graph_suggest",
        order=0,
        result={"LLMGatewayService": {"content": "", "raw": raw}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    cr = CortexClientResult(
        ok=True,
        mode="brain",
        verb="memory_graph_suggest",
        status="success",
        final_text="",
        steps=[step],
    )
    resp = CortexChatResult(cortex_result=cr, final_text="")
    from scripts.cortex_memory_graph_text import hub_memory_graph_suggest_text

    text, diag = hub_memory_graph_suggest_text(resp)
    assert "ontology_version" in text
    assert "raw.choices[0].message.content" in str(diag.get("selected_text_source") or "")
