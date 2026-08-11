from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.cognition.chat_history_compactor.digest import trim_chat_history_compactor_input
from orion.cognition.planner.prompt_renderer import PromptRenderer
from orion.schemas.discussion_window import DiscussionWindowResultV1, DiscussionWindowTurnV1

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
TEMPLATE_NAME = "chat_history_compactor_digest_v1.j2"


def _renderer() -> PromptRenderer:
    return PromptRenderer(PROMPTS_DIR)


def test_rendered_prompt_carries_topic_coverage_instruction() -> None:
    """Inspectable evidence that the topic-coverage contract text actually
    reaches the digest LLM's real prompt, not just the .j2 source file --
    the LLM-quality question of whether it *follows* the instruction still
    needs an LLM-in-the-loop eval (tracked separately), but this closes the
    gap of the instruction silently not making it into the rendered prompt."""
    text = _renderer().render(TEMPLATE_NAME, {"metadata": {}})
    assert "Cover every distinct topic/decision" in text
    assert "Breadth over depth" in text


def test_rendered_prompt_surfaces_truncation_flag_for_a_long_turn() -> None:
    """When a real long turn gets word-boundary-truncated by
    trim_chat_history_compactor_input, the render must actually carry the
    `truncated`/`turn_content_truncated` markers into the LLM-visible JSON,
    not just compute them and drop them before the prompt is built."""
    long_prompt = "one two three four five " + ("word " * 400)
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 8, 11, 4, 0, tzinfo=timezone.utc),
            correlation_id="corr-1",
            prompt=long_prompt,
            response="short response",
        )
    ]
    window = DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 8, 11, 0, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 8, 11, 23, 59, tzinfo=timezone.utc),
        turn_count=1,
        turns=turns,
        transcript_text="ignored",
    )
    digest_input = trim_chat_history_compactor_input(window)
    assert digest_input["turn_content_truncated"] is True  # sanity: fixture actually truncates

    text = _renderer().render(
        TEMPLATE_NAME, {"metadata": {"chat_history_compactor_input": digest_input}}
    )
    assert '"turn_content_truncated": true' in text
    assert '"truncated": true' in text
    assert "do not guess or invent" in text
