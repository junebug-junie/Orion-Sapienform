from __future__ import annotations

import os

from anthropic import APIConnectionError, APIStatusError, APITimeoutError, AsyncAnthropic

from app.providers.base import IdeationProvider


class AnthropicIdeationProvider(IdeationProvider):
    def __init__(self, *, api_key: str, model: str, timeout_seconds: float = 120.0) -> None:
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)
        self.model = model

    @classmethod
    def from_env(cls, *, model: str, timeout_seconds: float = 120.0) -> AnthropicIdeationProvider:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when KNOWLEDGE_FORGE_IDEATION_PROVIDER=anthropic")
        return cls(api_key=api_key, model=model, timeout_seconds=timeout_seconds)

    async def run(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        try:
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except APITimeoutError as exc:
            raise TimeoutError("Anthropic ideation request timed out") from exc
        except (APIConnectionError, APIStatusError) as exc:
            raise RuntimeError(f"Anthropic ideation request failed: {exc}") from exc

        parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts)
