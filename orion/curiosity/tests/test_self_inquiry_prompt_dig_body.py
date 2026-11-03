"""Self Atlas replacement, Patch A: the records section points at the live dig kit first."""
from orion.curiosity.self_inquiry_prompt import DIG_BODY_PARAGRAPH, build_self_inquiry_prompt

RUN = "abcd12"


def _text(**over) -> str:
    kwargs = dict(question=None, run_id=RUN, graph_enabled=False, repo_root="/repo")
    kwargs.update(over)
    return build_self_inquiry_prompt(**kwargs)


def test_records_section_renders_the_dig_body_paragraph() -> None:
    text = _text()
    for line in DIG_BODY_PARAGRAPH:
        assert line in text
    for cmd in ("orion.dig body", "orion.dig body <organ_id>", "orion.dig inventory now",
                "orion.dig compare <date> now", "orion.dig body-economy"):
        assert f"python -P -m {cmd}" in text
    assert "python -m orion.dig" not in text, "always -P: plain -m runs the sandbox copy"


def test_dig_tools_come_before_the_raw_repo_pointer_inside_the_records_section() -> None:
    text = _text()
    records = text.index("WHERE THE RECORDS ARE")
    dig = text.index("python -P -m orion.dig body", records)
    repo = text.index("/repo/", records)
    assert records < dig < repo
    assert "README" in text[repo - 200:repo + 200], "raw-repo pointer is kept"


def test_dig_paragraph_renders_on_the_graph_enabled_path_too() -> None:
    text = _text(graph_enabled=True)
    assert "python -P -m orion.dig body" in text
    assert text.index("YOUR FIRST TOOL CALL") < text.index("python -P -m orion.dig body")
