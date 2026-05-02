from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .types import ClaimResolutionRecord, EpisodeRecord, TurnBlockRecord


def render_pageindex_markdown(
    *,
    episodes: list[EpisodeRecord],
    blocks: list[TurnBlockRecord],
    claims: list[ClaimResolutionRecord],
) -> str:
    lines = ["# Orion Chat Corpus", ""]
    block_by_turn = {block.turn_id: block for block in blocks}
    claims_by_episode: dict[str, list[ClaimResolutionRecord]] = defaultdict(list)
    for claim in claims:
        claims_by_episode[claim.episode_id].append(claim)
    for episode in sorted(episodes, key=lambda item: item.start_at):
        day = _day(episode.start_at)
        lines.append(f"## {day}")
        lines.append("")
        lines.append(f"### Episode: {episode.episode_label}")
        lines.append(f"- episode_id: {episode.episode_id}")
        lines.append(f"- top_anchors: {', '.join(episode.top_anchors)}")
        lines.append(f"- confidence: {episode.confidence}")
        if claims_by_episode.get(episode.episode_id):
            outcome = claims_by_episode[episode.episode_id][0].resolution_text
            lines.append(f"- outcome: {outcome}")
        lines.append("")
        for turn_id in episode.turn_ids:
            block = block_by_turn.get(turn_id)
            if not block:
                continue
            lines.extend(
                [
                    f"#### Turn: {block.created_at}",
                    f"- turn_id: {turn_id}",
                    "",
                    "##### User Problem",
                    block.user_problem_block or "(empty)",
                    "",
                    "##### Assistant Answer",
                    block.assistant_answer_block or "(empty)",
                    "",
                ]
            )
            if block.log_or_error_block:
                lines.extend(["##### Log / Evidence", block.log_or_error_block, ""])
            if block.command_or_code_block:
                lines.extend(["##### Command / Code", block.command_or_code_block, ""])
            for claim in claims_by_episode.get(episode.episode_id, []):
                if turn_id in claim.evidence_turn_ids:
                    lines.extend(
                        [
                            "##### Decision / Resolution",
                            f"{claim.status.upper()}: {claim.resolution_text}",
                            "",
                        ]
                    )
    return "\n".join(lines).rstrip() + "\n"


def _day(iso_dt: str) -> str:
    return datetime.fromisoformat(iso_dt).date().isoformat()
