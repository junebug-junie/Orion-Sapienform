from __future__ import annotations

from orion.execution_dispatch.result_extraction import (
    extract_final_text,
    parse_structured_observation,
)


def test_extract_final_text_from_result_final_text() -> None:
    payload = {"result": {"final_text": "  hello world  "}}
    assert extract_final_text(payload) == "hello world"


def test_extract_final_text_from_step_result_fallback() -> None:
    payload = {
        "result": {
            "final_text": None,
            "steps": [
                {"result": {"llm": {"text": "step output here"}}},
            ],
        }
    }
    assert extract_final_text(payload) == "step output here"


def test_extract_final_text_missing_returns_empty() -> None:
    assert extract_final_text({}) == ""
    assert extract_final_text({"result": {}}) == ""
    assert extract_final_text({"result": None}) == ""


def test_parse_structured_observation_happy_path() -> None:
    data = parse_structured_observation(
        '{"observation": "steady state", "salient_facts": ["a", "b"], "confidence": 0.75}'
    )
    assert data == {
        "observation": "steady state",
        "salient_facts": ["a", "b"],
        "confidence": 0.75,
    }


def test_parse_structured_observation_empty_text() -> None:
    data = parse_structured_observation("")
    assert data["observation"] == ""
    assert data["salient_facts"] == []
    assert data["confidence"] == 0.0


def test_parse_structured_observation_malformed_json_degrades_empty() -> None:
    data = parse_structured_observation("not json at all {{{")
    assert data["observation"] == ""


def test_parse_structured_observation_non_dict_json_degrades_empty() -> None:
    data = parse_structured_observation("[1, 2, 3]")
    assert data["observation"] == ""


def test_parse_structured_observation_missing_fields_defaults() -> None:
    data = parse_structured_observation('{"unrelated": true}')
    assert data == {"observation": "", "salient_facts": [], "confidence": 0.0}


def test_parse_structured_observation_wrong_types_coerced_safely() -> None:
    data = parse_structured_observation(
        '{"observation": 123, "salient_facts": "not-a-list", "confidence": "n/a"}'
    )
    assert data["observation"] == ""
    assert data["salient_facts"] == []
    assert data["confidence"] == 0.0
