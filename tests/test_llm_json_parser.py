from orion.core.llm_json import parse_json_object


EXPECTED_KEYS = {"thought", "finish", "action", "final_answer"}


def test_parse_json_object_normal_dict() -> None:
    text = '{"thought":"x","finish":true,"action":null,"final_answer":{"content":"ok"}}'
    parsed = parse_json_object(text)
    assert isinstance(parsed, dict)
    assert EXPECTED_KEYS.issubset(parsed.keys())


def test_parse_json_object_markdown_fence() -> None:
    text = '```json\n{"thought":"x","finish":true,"action":null,"final_answer":{"content":"ok"}}\n```'
    parsed = parse_json_object(text)
    assert isinstance(parsed, dict)
    assert EXPECTED_KEYS.issubset(parsed.keys())


def test_parse_json_object_double_encoded_string() -> None:
    text = '"{\\"thought\\":\\"x\\",\\"finish\\":true,\\"action\\":null,\\"final_answer\\":{\\"content\\":\\"ok\\"}}"'
    parsed = parse_json_object(text)
    assert isinstance(parsed, dict)
    assert EXPECTED_KEYS.issubset(parsed.keys())
    assert parsed["final_answer"]["content"] == "ok"


def test_parse_json_object_escaped_object_text() -> None:
    text = '{\\"thought\\":\\"x\\",\\"finish\\":false,\\"action\\":{\\"tool_id\\":\\"triage\\",\\"input\\":{\\"text\\":\\"hi\\"}},\\"final_answer\\":null}'
    parsed = parse_json_object(text)
    assert isinstance(parsed, dict)
    assert EXPECTED_KEYS.issubset(parsed.keys())
    assert parsed["action"]["tool_id"] == "triage"


def test_parse_json_object_single_quoted_wrapper() -> None:
    text = '\'{\"thought\":\"x\",\"finish\":true,\"action\":null,\"final_answer\":{\"content\":\"ok\"}}\''
    parsed = parse_json_object(text)
    assert isinstance(parsed, dict)
    assert EXPECTED_KEYS.issubset(parsed.keys())


def test_parse_json_object_single_quoted_escaped_blob_from_logs() -> None:
    text = '\'{\\n  "thought": "x",\\n  "finish": false,\\n  "action": {\\n    "tool_id": "analyze_text",\\n    "input": {\\n      "text": "hi",\\n      "request": "tag"\\n    }\\n  },\\n  "final_answer": null\\n}\''
    parsed = parse_json_object(text)
    assert isinstance(parsed, dict)
    assert parsed["action"]["tool_id"] == "analyze_text"
    assert parsed["finish"] is False
