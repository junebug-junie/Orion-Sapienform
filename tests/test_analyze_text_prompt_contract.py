from pathlib import Path


def test_analyze_text_prompt_disallows_tool_and_url_fabrication() -> None:
    text = Path("orion/cognition/prompts/analyze_text_prompt.j2").read_text(encoding="utf-8")
    assert "Do not invent tools, tool names, APIs, or documentation links." in text
    assert "Do not include URLs." in text


def test_text_analysis_prompt_disallows_tool_and_url_fabrication() -> None:
    text = Path("orion/cognition/prompts/text_analysis_prompt.j2").read_text(encoding="utf-8")
    assert "Do not invent tools, tool names, APIs, or documentation links." in text
    assert "Do not include URLs." in text
