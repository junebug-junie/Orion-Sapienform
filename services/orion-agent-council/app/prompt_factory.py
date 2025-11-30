# services/orion-agent-council/app/prompt_factory.py

"""
Shim module to preserve older imports like `from app.prompt_factory import PromptFactory`.

Internally we now keep the implementation under `app.core.prompt_factory`.
"""

from .core.prompt_factory import PromptContext, PromptFactory

__all__ = ["PromptContext", "PromptFactory"]
