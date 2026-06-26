from __future__ import annotations

from orion.memory_graph.suggest_runner import build_memory_graph_suggest_options
from orion.memory_graph.suggest_token_budget import (
    SuggestTokenBudgetConfig,
    completion_budget_for_transcript,
)


def test_build_memory_graph_suggest_options() -> None:
    opts = build_memory_graph_suggest_options()
    assert opts["no_write"] is True
    assert opts["skip_autonomy_context"] is True
    assert opts["structured_output_schema_name"] == "SuggestDraftV1"
    assert isinstance(opts["structured_output_schema"], dict)
    assert opts["structured_output_method"] == "json_object_schema"
    assert 768 <= opts["max_tokens"] <= 4096


def test_completion_budget_clamps_for_long_transcript() -> None:
    cfg = SuggestTokenBudgetConfig()
    long_transcript = "x" * 9000
    budget = completion_budget_for_transcript(long_transcript, config=cfg)
    assert budget == cfg.min_completion_tokens


def test_completion_budget_formula() -> None:
    cfg = SuggestTokenBudgetConfig(
        ctx_tokens=4096,
        prompt_overhead_tokens=1800,
        min_completion_tokens=768,
        max_completion_tokens=4096,
        chars_per_token_estimate=3,
        min_prompt_tokens_estimate=400,
    )
    # 1500 chars -> 500 est prompt; 4096 - 500 - 1800 = 1796
    assert completion_budget_for_transcript("x" * 1500, config=cfg) == 1796
