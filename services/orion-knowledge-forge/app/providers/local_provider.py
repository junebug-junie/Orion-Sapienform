from __future__ import annotations

from app.providers.base import IdeationProvider


class LocalIdeationProvider(IdeationProvider):
    """Deterministic offline provider for tests and local smoke."""

    async def run(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        _ = system_prompt, max_tokens
        return "\n".join(
            [
                "# Ideation proposal (local provider)",
                "",
                "> Proposals are not canonical truth. Pending human review only.",
                "",
                "## Current shape",
                "Knowledge Forge v1 exposes corpus read/search/compile and review accept/reject.",
                "",
                "## Arsonist critique",
                "- Ideation is stubbed locally; verify Anthropic path before trusting output.",
                "- Ensure write gates stay off in production until review workflow is exercised.",
                "",
                "## Missing questions",
                "- Which v1.1 surfaces need Hub wiring first?",
                "- What is the minimum MCP tool set worth exposing?",
                "",
                "## Proposed v-next",
                "- Keep ideation behind provider interface; default local in CI.",
                "- Write only to reviews/pending when explicitly enabled.",
                "",
                "## Files likely to touch",
                "- services/orion-knowledge-forge/app/ideation/*",
                "- services/orion-knowledge-forge/app/routers/ideation.py",
                "",
                "## Non-goals",
                "- Silent mutation of accepted claims/specs/decisions",
                "- GraphDB/RDF integration",
                "- Vector search",
                "",
                "## Acceptance checks",
                "- POST /v1/ideation/run returns structured result with local provider",
                "- write disabled leaves artifact_path null",
                "- write enabled creates reviews/pending/ideation-*.md only",
                "",
                "## Task echo",
                user_prompt.strip() or "(empty task)",
            ]
        )
