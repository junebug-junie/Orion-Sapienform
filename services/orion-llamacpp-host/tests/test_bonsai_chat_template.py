"""Bonsai (agent-gpu2) chat template: the GGUF's embedded template 500'd every cortex-exec agent step.

Root cause (2026-10-07): Ternary-Bonsai-2-27B-PQ2_0.gguf's embedded ``tokenizer.chat_template`` ends
its reverse scan for the last user query with ``raise_exception('No user query found in messages.')``
when no plain ``user`` turn exists. cortex-exec's ``_build_hop_messages``
(services/orion-cortex-exec/app/executor.py) sends a verb step as ``[{"role": "system", ...}]`` when
the step carries no context messages -- the gateway logged ``msgs=1`` on all 10 Bonsai agent calls
since 10-06, and llama-server raised on every one. The Qwen3.8-27B Q4 template (same family, Unsloth
build) has no such raise and renders the same request.

Fixtures are the two embedded templates, read byte-for-byte from the GGUF headers on circe
(the Q4 one also equals its live ``/props`` ``chat_template``).

These render with Python jinja2, not llama.cpp's own engine; the server's error text and line
match the jinja2 reproduction exactly, but the real-engine render is a post-deploy check.
"""
from __future__ import annotations

import json
from pathlib import Path

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

HOST = Path(__file__).resolve().parents[1]
REPO = HOST.parents[1]
FIXTURES = HOST / "tests" / "fixtures" / "chat_templates"
SHIPPED = REPO / "config" / "chat_templates" / "ternary-bonsai-2-27b.jinja"
BONSAI_EMBEDDED = FIXTURES / "ternary-bonsai-2-27b-pq2_0.embedded.jinja"
Q4_EMBEDDED = FIXTURES / "qwen3.8-27b-ud-q4_k_xl.embedded.jinja"
RAISE_BLOCK = (
    "{%- if ns.multi_step_tool %}\n"
    "    {{- raise_exception('No user query found in messages.') }}\n"
    "{%- endif %}\n"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}]
CALL = {"type": "function", "function": {"name": "read_file", "arguments": {"path": "/tmp/a.txt"}}}

# Real request shapes. The first is what cortex-exec sends for an agent verb step (msgs=1).
SHAPES = {
    "system_only": [{"role": "system", "content": "You are Orion. Write today's note."}],
    "system_assistant_tool": [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "content": "file body"},
    ],
    "user_chat": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
    "tool_round_trip": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "read /tmp/a.txt"},
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "content": "file body"},
        {"role": "assistant", "content": "It says: file body"},
        {"role": "user", "content": "thanks"},
    ],
}


def _template(path: Path) -> jinja2.Template:
    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda value, **_kw: json.dumps(value, ensure_ascii=False)
    return env.from_string(path.read_text(encoding="utf-8"))


def _render(path: Path, messages, *, enable_thinking=False, tools=None) -> str:
    kwargs = {"messages": messages, "add_generation_prompt": True, "tools": tools}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    # The profile's server-side kwargs (config/llm_profiles.yaml) ride on every request.
    kwargs.update(reasoning_effort="xhigh", preserve_thinking=True)
    return _template(path).render(**kwargs)


def test_embedded_bonsai_template_raises_on_the_cortex_exec_system_only_step():
    with pytest.raises(jinja2.exceptions.TemplateError, match="No user query found in messages."):
        _render(BONSAI_EMBEDDED, SHAPES["system_only"])
    # ...and the same check fires with no user turn at all, whatever else is in the list.
    with pytest.raises(jinja2.exceptions.TemplateError, match="No user query found in messages."):
        _render(BONSAI_EMBEDDED, SHAPES["system_assistant_tool"])


def test_q4_template_renders_the_request_bonsai_rejected():
    out = _render(Q4_EMBEDDED, SHAPES["system_only"])
    assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_shipped_template_is_the_embedded_one_minus_only_the_raise():
    embedded = BONSAI_EMBEDDED.read_text(encoding="utf-8")
    shipped = SHIPPED.read_text(encoding="utf-8")
    assert embedded.count(RAISE_BLOCK) == 1
    assert "No user query found" not in shipped.split("{#- Orion:")[0]
    # Everything else -- tool-call XML format, thinking blocks, reasoning-effort checks -- is Bonsai's own.
    assert shipped.split("\n{#- Orion:")[0] == embedded.replace(RAISE_BLOCK, "")


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("enable_thinking", [False, True, None])
@pytest.mark.parametrize("tools", [None, TOOLS])
def test_shipped_template_renders_every_real_shape(shape, enable_thinking, tools):
    out = _render(SHIPPED, SHAPES[shape], enable_thinking=enable_thinking, tools=tools)
    assert out.rstrip().endswith("<|im_start|>assistant\n<think>") or out.endswith("</think>\n\n")
    if tools:
        assert "<tools>" in out and "<function=example_function_name>" in out
    if enable_thinking is False:
        assert out.endswith("<think>\n\n</think>\n\n")


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("tools", [None, TOOLS])
def test_shipped_template_renders_byte_identical_to_q4_on_these_shapes(shape, tools):
    """The fix does not invent a new prompt: on every shape the agent lane sends, Bonsai now gets
    exactly the prompt the Q4 27B (which serves these requests successfully) gets."""
    assert _render(SHIPPED, SHAPES[shape], tools=tools) == _render(Q4_EMBEDDED, SHAPES[shape], tools=tools)


def test_tool_round_trip_keeps_bonsai_tool_call_format():
    out = _render(SHIPPED, SHAPES["tool_round_trip"], tools=TOOLS)
    assert "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n</function>\n</tool_call>" in out
    assert "<tool_response>\nfile body\n</tool_response>" in out
