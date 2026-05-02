from __future__ import annotations

from app.pipelines.chat_corpus_builder.renderer import render_pageindex_markdown
from app.pipelines.chat_corpus_builder.types import ClaimResolutionRecord, EpisodeRecord, TurnBlockRecord


def test_renderer_outputs_episode_markdown_shape() -> None:
    markdown = render_pageindex_markdown(
        episodes=[
            EpisodeRecord(
                episode_id="chat-episode-20260425-001",
                start_at="2026-04-25T14:31:00+00:00",
                end_at="2026-04-25T14:40:00+00:00",
                turn_ids=["t1"],
                top_anchors=["journal_entry_index", "orion-pageindex"],
                confidence=0.82,
                episode_label="journal_entry_index / orion-pageindex",
                episode_summary="summary",
            )
        ],
        blocks=[
            TurnBlockRecord(
                turn_id="t1",
                created_at="2026-04-25T14:31:00+00:00",
                user_problem_block="journal_entry_index is empty",
                assistant_answer_block="The fix is to backfill.",
                command_or_code_block="docker compose run",
                log_or_error_block="candidate_count=0",
                optional_reasoning_summary_block="",
            )
        ],
        claims=[
            ClaimResolutionRecord(
                claim_id="claim-1",
                episode_id="chat-episode-20260425-001",
                claim_text="The fix is to backfill.",
                status="confirmed",
                resolution_text="Confirmed: Backfill fixed it.",
                evidence_turn_ids=["t1"],
                status_reason="Derived from evidence",
                last_status_at="2026-04-25T14:40:00+00:00",
            )
        ],
    )
    assert "# Orion Chat Corpus" in markdown
    assert "### Episode: journal_entry_index / orion-pageindex" in markdown
    assert "##### Decision / Resolution" in markdown
