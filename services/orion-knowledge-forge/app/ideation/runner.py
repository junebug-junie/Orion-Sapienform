from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.api_schemas import IdeationRunRequestV1, IdeationRunResultV1
from app.ideation.prompts import (
    STATIC_KNOWLEDGE_FORGE_CONTRACT,
    build_user_prompt,
    detect_monorepo_root,
)
from app.ideation.writer import assert_safe_artifact_path, build_artifact_path, write_review_artifact
from app.providers.base import IdeationProvider
from app.providers.local_provider import LocalIdeationProvider
from app.service import KnowledgeForgeService
from app.settings import Settings


def build_provider(settings: Settings) -> IdeationProvider:
    provider = settings.knowledge_forge_ideation_provider.lower()
    if provider == "anthropic":
        from app.providers.anthropic_provider import AnthropicIdeationProvider

        return AnthropicIdeationProvider.from_settings(
            api_key=settings.anthropic_api_key,
            model=settings.knowledge_forge_anthropic_model,
        )
    if provider == "local":
        return LocalIdeationProvider()
    raise ValueError(f"unknown ideation provider: {provider}")


class IdeationRunner:
    def __init__(self, settings: Settings, forge_service: KnowledgeForgeService) -> None:
        self.settings = settings
        self.forge_service = forge_service

    async def run(self, request: IdeationRunRequestV1) -> IdeationRunResultV1:
        if not self.settings.knowledge_forge_ideation_enabled:
            raise ValueError("ideation is disabled")

        provider = build_provider(self.settings)
        corpus_root = self.forge_service.root
        monorepo_root = detect_monorepo_root(corpus_root)
        status = self.forge_service.status()
        status_summary = (
            f"claims={status.claim_count} specs={status.spec_count} "
            f"pending_reviews={status.pending_review_count}"
        )
        user_prompt = build_user_prompt(
            task=request.task,
            mode=request.mode.value,
            input_paths=list(request.input_paths),
            corpus_root=corpus_root,
            monorepo_root=monorepo_root,
            status_summary=status_summary,
        )
        content = await provider.run(
            system_prompt=STATIC_KNOWLEDGE_FORGE_CONTRACT,
            user_prompt=user_prompt,
            max_tokens=request.max_tokens,
        )

        now = datetime.now(timezone.utc)
        run_id = f"ideation-{now.strftime('%Y-%m-%d')}-{uuid4().hex[:8]}"
        warnings: list[str] = []
        artifact_path: str | None = None

        if request.write_review:
            if not self.settings.knowledge_forge_ideation_write_enabled:
                warnings.append("write disabled: KNOWLEDGE_FORGE_IDEATION_WRITE_ENABLED is false")
            else:
                written = write_review_artifact(
                    corpus_root,
                    task=request.task,
                    mode=request.mode.value,
                    content=content,
                    run_id=run_id,
                )
                assert_safe_artifact_path(corpus_root, written)
                artifact_path = str(written.relative_to(corpus_root))
        else:
            # dry-run path check for observability
            probe = build_artifact_path(corpus_root, task=request.task, now=now)
            assert_safe_artifact_path(corpus_root, probe)

        summary = _first_nonempty_line(content)
        return IdeationRunResultV1(
            run_id=run_id,
            provider=self.settings.knowledge_forge_ideation_provider,
            model=self.settings.knowledge_forge_anthropic_model
            if self.settings.knowledge_forge_ideation_provider == "anthropic"
            else "local",
            status="proposed",
            summary=summary,
            content=content,
            artifact_path=artifact_path,
            warnings=warnings,
            usage={},
        )


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:240]
    return text.strip()[:240] or "ideation complete"
