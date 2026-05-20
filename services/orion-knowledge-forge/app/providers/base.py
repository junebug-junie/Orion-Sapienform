from __future__ import annotations

from abc import ABC, abstractmethod


class IdeationProvider(ABC):
    @abstractmethod
    async def run(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        """Run ideation and return model text output."""
