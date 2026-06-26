"""Shared completion-token budget for memory_graph_suggest (ctx-aware, env-configurable)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SuggestTokenBudgetConfig:
    """Align ctx_tokens with atlas-fast ctx_size in config/llm_profiles.yaml."""

    ctx_tokens: int = 4096
    prompt_overhead_tokens: int = 1800
    min_completion_tokens: int = 768
    max_completion_tokens: int = 4096
    chars_per_token_estimate: int = 3
    min_prompt_tokens_estimate: int = 400


def _positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _lookup_value(mapping: Mapping[str, Any] | Any | None, *keys: str) -> Any:
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None
    for key in keys:
        value = getattr(mapping, key, None)
        if value is not None:
            return value
    return None


def suggest_token_budget_config_from_mapping(
    mapping: Mapping[str, Any] | Any | None,
) -> SuggestTokenBudgetConfig:
    lookup = lambda *keys: _lookup_value(mapping, *keys)
    return SuggestTokenBudgetConfig(
        ctx_tokens=_positive_int(
            lookup("MEMORY_GRAPH_SUGGEST_CTX_TOKENS", "memory_graph_suggest_ctx_tokens"),
            4096,
        ),
        prompt_overhead_tokens=_positive_int(
            lookup(
                "MEMORY_GRAPH_SUGGEST_PROMPT_OVERHEAD_TOKENS",
                "memory_graph_suggest_prompt_overhead_tokens",
            ),
            1800,
        ),
        min_completion_tokens=_positive_int(
            lookup(
                "MEMORY_GRAPH_SUGGEST_MIN_COMPLETION_TOKENS",
                "memory_graph_suggest_min_completion_tokens",
            ),
            768,
        ),
        max_completion_tokens=_positive_int(
            lookup("MEMORY_GRAPH_SUGGEST_MAX_TOKENS", "memory_graph_suggest_max_tokens"),
            4096,
        ),
        chars_per_token_estimate=_positive_int(
            lookup("MEMORY_GRAPH_SUGGEST_CHARS_PER_TOKEN", "memory_graph_suggest_chars_per_token"),
            3,
        ),
        min_prompt_tokens_estimate=_positive_int(
            lookup(
                "MEMORY_GRAPH_SUGGEST_MIN_PROMPT_TOKENS_ESTIMATE",
                "memory_graph_suggest_min_prompt_tokens_estimate",
            ),
            400,
        ),
    )


def suggest_token_budget_config_from_env() -> SuggestTokenBudgetConfig:
    return suggest_token_budget_config_from_mapping(os.environ)


def completion_budget_for_transcript(
    transcript: str,
    *,
    config: SuggestTokenBudgetConfig | None = None,
) -> int:
    cfg = config or suggest_token_budget_config_from_env()
    est_prompt_tokens = max(
        cfg.min_prompt_tokens_estimate,
        len(transcript or "") // cfg.chars_per_token_estimate,
    )
    ctx_budget = cfg.ctx_tokens - est_prompt_tokens - cfg.prompt_overhead_tokens
    return max(
        cfg.min_completion_tokens,
        min(cfg.max_completion_tokens, ctx_budget),
    )
