from __future__ import annotations

from app.profiles import LlamaCppConfig
from app.thinking_policy import resolve_thinking_launch_policy


def test_explicit_reasoning_budget_not_overwritten():
    cfg = LlamaCppConfig(
        chat_template_kwargs={"enable_thinking": False},
        reasoning_budget=8192,
    )
    pol = resolve_thinking_launch_policy(cfg, {"--reasoning-budget", "--chat-template-kwargs"})
    assert pol.effective_reasoning_budget == 8192
    assert pol.require_jinja is True  # kwargs present → jinja for template path


def test_explicit_reasoning_budget_nonzero_no_kwargs_no_jinja_from_policy():
    cfg = LlamaCppConfig(reasoning_budget=8192)
    pol = resolve_thinking_launch_policy(cfg, {"--reasoning-budget"})
    assert pol.effective_reasoning_budget == 8192
    assert pol.require_jinja is False


def test_implicit_budget_zero_when_kwargs_false():
    cfg = LlamaCppConfig(chat_template_kwargs={"enable_thinking": False})
    pol = resolve_thinking_launch_policy(cfg, {"--reasoning-budget", "--chat-template-kwargs"})
    assert pol.effective_reasoning_budget == 0
    assert pol.require_jinja is True


def test_no_implicit_budget_when_flag_missing():
    cfg = LlamaCppConfig(chat_template_kwargs={"enable_thinking": False})
    pol = resolve_thinking_launch_policy(cfg, {"--chat-template-kwargs"})
    assert pol.effective_reasoning_budget is None
    assert pol.require_jinja is True
