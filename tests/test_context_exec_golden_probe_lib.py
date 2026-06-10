from __future__ import annotations

import pytest

from scripts.context_exec_probe_lib import assert_hub_context_exec_routing


def test_rejects_chat_general_without_context_exec():
    with pytest.raises(AssertionError, match="chat_general"):
        assert_hub_context_exec_routing(
            {
                "session_id": "s1",
                "text": "[Error: llamacpp timed out after waiting]",
                "raw": {"verb": "chat_general", "final_text": "[Error: llamacpp timed out after waiting]"},
            },
            probe_name="denver",
        )


def test_accepts_agent_runtime_with_context_exec_step():
    assert_hub_context_exec_routing(
        {
            "session_id": "s2",
            "raw": {
                "verb": "agent_runtime",
                "final_text": "Belief provenance for Denver",
                "steps": [
                    {
                        "step_name": "context_exec",
                        "result": {
                            "ContextExecService": {
                                "structured": {"context_exec": {"mode": "belief_provenance"}},
                                "runtime_debug": {"engine": "context_exec", "context_exec_attempted": True},
                            }
                        },
                    }
                ],
            },
            "routing_debug": {
                "options": {
                    "agent_runtime_engine": "context_exec",
                    "context_exec_mode": "belief_provenance",
                    "execution_depth": 2,
                }
            },
        },
        probe_name="denver",
        expected_mode="belief_provenance",
    )


def test_agent_runtime_without_context_exec_step_fails():
    with pytest.raises(AssertionError, match="ContextExecService did not run"):
        assert_hub_context_exec_routing(
            {
                "session_id": "s3",
                "raw": {
                    "verb": "agent_runtime",
                    "final_text": "Some agent answer",
                    "metadata": {
                        "auto_route": {
                            "execution_depth": 2,
                            "reason": "heuristic:default+context_exec_investigation",
                        }
                    },
                    "steps": [
                        {"step_name": "agent_chain", "result": {"AgentChainService": {"text": "x"}}},
                    ],
                },
            },
            probe_name="denver",
        )


def test_rejects_llm_timeout_even_if_verb_not_chat_general():
    with pytest.raises(AssertionError, match="LLM timeout"):
        assert_hub_context_exec_routing(
            {
                "raw": {"verb": "agent_runtime", "final_text": "[Error: llamacpp timed out after waiting]"},
            },
            probe_name="repo",
        )
