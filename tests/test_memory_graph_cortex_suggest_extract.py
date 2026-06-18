from __future__ import annotations

import json

import pytest

from orion.memory_graph.cortex_suggest_extract import (
    extract_suggest_draft_dict_from_cortex_payload,
    extract_suggest_text_from_cortex_payload,
)


def _minimal_draft() -> dict:
    return {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }


def test_extract_from_step_result_llm_gateway_content() -> None:
    draft = _minimal_draft()
    raw = {
        "ok": True,
        "verb": "memory_graph_suggest",
        "status": "success",
        "final_text": "",
        "steps": [
            {
                "status": "success",
                "verb_name": "memory_graph_suggest",
                "step_name": "llm_memory_graph_suggest",
                "order": 0,
                "result": {"LLMGatewayService": {"content": json.dumps(draft)}},
            }
        ],
    }
    assert extract_suggest_draft_dict_from_cortex_payload(raw) == draft


def test_extract_from_raw_openai_choices_when_content_empty() -> None:
    draft = _minimal_draft()
    raw = {
        "ok": True,
        "final_text": "",
        "steps": [
            {
                "step_name": "llm_memory_graph_suggest",
                "order": 0,
                "result": {
                    "LLMGatewayService": {
                        "content": "",
                        "raw": {
                            "choices": [
                                {
                                    "message": {"role": "assistant", "content": json.dumps(draft)},
                                    "finish_reason": "stop",
                                }
                            ]
                        },
                    }
                },
            }
        ],
    }
    assert extract_suggest_draft_dict_from_cortex_payload(raw) == draft


def test_extract_from_top_level_final_text() -> None:
    draft = _minimal_draft()
    raw = {"final_text": json.dumps(draft), "steps": []}
    assert extract_suggest_draft_dict_from_cortex_payload(raw) == draft


def test_extract_from_explicit_draft_field() -> None:
    draft = _minimal_draft()
    assert extract_suggest_draft_dict_from_cortex_payload({"draft": draft}) == draft


def test_extract_legacy_detail_shape() -> None:
    draft = _minimal_draft()
    raw = {
        "steps": [
            {
                "detail": {"text": json.dumps(draft)},
            }
        ]
    }
    assert extract_suggest_draft_dict_from_cortex_payload(raw) == draft


def test_extract_missing_draft_raises() -> None:
    with pytest.raises(ValueError, match="memory_graph_suggest_draft_not_found|no_json_object|empty_output"):
        extract_suggest_draft_dict_from_cortex_payload({"final_text": "no json here", "steps": []})


def test_extract_text_prefers_json_blob_in_prose() -> None:
    draft = _minimal_draft()
    wrapped = f"Here is the draft:\n{json.dumps(draft)}\nThanks."
    text = extract_suggest_text_from_cortex_payload({"final_text": wrapped, "steps": []})
    assert extract_suggest_draft_dict_from_cortex_payload({"final_text": wrapped, "steps": []}) == draft
    assert json.loads(text) == draft
