from orion.memory_graph.json_extract import extract_first_json_object_text


def test_extract_first_json_object_strips_wrappers() -> None:
    inner = '{"a": 1}'
    wrapped = f'Here you go:\n```json\n{inner}\n```\n'
    assert extract_first_json_object_text(wrapped) == inner
