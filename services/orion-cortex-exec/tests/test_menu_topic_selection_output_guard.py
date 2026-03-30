from __future__ import annotations

from app.executor import _sanitize_menu_topic_selection_reply


def test_menu_topic_selection_output_guard_strips_fabricated_tool_tokens() -> None:
    raw = "Let's use `hm_mesh_continuity` next. hm_mesh_continuity is the right tool."
    cleaned = _sanitize_menu_topic_selection_reply(raw)
    assert "hm_mesh_continuity" not in cleaned


def test_menu_topic_selection_output_guard_strips_placeholder_example_urls() -> None:
    raw = "Read docs at https://example.com/mesh and https://placeholder.local/docs please."
    cleaned = _sanitize_menu_topic_selection_reply(raw)
    assert "example.com" not in cleaned
    assert "placeholder.local" not in cleaned
