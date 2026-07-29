"""Regression coverage for the substrate.inspect/summarize/observe grounding fix.

Diagnosed live 2026-07-28/29: these three prompts previously gave the model
only a bare target_kind/target_id/allowed_scope and an instruction to avoid
"invented telemetry values" -- with no real data to ground anything in, that
instruction was self-contradicting by construction. Confirmed in production
(``substrate_dispatch_results``) that the model was fabricating specific
technical claims (transformer "attention heads", token-throughput deltas)
that don't correspond to any real signal in this substrate.

The fix wires ``ProposalCandidateV1.motivating_dimensions``/``priority_score``/
``risk_score`` (already real, already computed in
``orion/proposals/builder.py::_build_candidate()``) through
``orion/execution_dispatch/envelopes.py::build_cortex_request_envelope()``'s
context dict, which flows unmodified all the way to this service's Jinja
render context (see ``executor.py::_render_prompt`` / ``_prompt_render_ctx``).
This test renders the real templates the same way ``call_step_services`` does
and checks the real numbers actually show up -- and that omission of
``motivating_dimensions`` (e.g. an older persisted envelope) degrades
gracefully instead of crashing the render.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(EXEC_ROOT) not in sys.path:
    sys.path.append(str(EXEC_ROOT))


def _load_executor_module():
    app_dir = EXEC_ROOT / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec_lane"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(app_dir.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(app_dir)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.executor", executor_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_template(name: str) -> str:
    return (ROOT / "orion" / "cognition" / "prompts" / name).read_text(encoding="utf-8")


def _base_ctx(**overrides) -> dict:
    ctx = {
        "target_kind": "node",
        "target_id": "node:atlas",
        "allowed_scope": "inspect_only",
        "proposal_id": "proposal:inspect:node:atlas",
        "decision_id": "policy.decision:proposal:inspect:node:atlas:substrate_policy.v1",
        "self_state_id": "self_state:v1:abcd",
    }
    ctx.update(overrides)
    return ctx


def test_substrate_inspect_renders_real_motivating_dimensions():
    executor_module = _load_executor_module()
    template = _load_template("substrate_inspect.j2")
    ctx = _base_ctx(
        motivating_dimensions={"resource_pressure": 0.652, "cpu_pressure": 0.41},
        priority_score=0.71,
        risk_score=0.05,
    )
    prompt = executor_module._render_prompt(template, ctx)
    assert "REAL TELEMETRY" in prompt
    assert "resource_pressure: 0.652" in prompt
    assert "cpu_pressure: 0.410" in prompt
    assert "priority_score: 0.710" in prompt
    assert "risk_score: 0.050" in prompt
    # The old self-contradicting instruction (no real data + "no invented
    # telemetry values") is gone in favor of an enforceable instruction now
    # that real data is present.
    assert "no invented telemetry values" not in prompt


def test_substrate_summarize_renders_real_motivating_dimensions():
    executor_module = _load_executor_module()
    template = _load_template("substrate_summarize.j2")
    ctx = _base_ctx(
        target_kind="capability",
        target_id="capability:orchestration",
        allowed_scope="summarize_only",
        motivating_dimensions={"resource_pressure": 0.3, "attention_load": 0.2},
        priority_score=0.3,
        risk_score=0.1,
    )
    prompt = executor_module._render_prompt(template, ctx)
    assert "REAL TELEMETRY" in prompt
    assert "resource_pressure: 0.300" in prompt
    assert "attention_load: 0.200" in prompt


def test_substrate_observe_renders_real_motivating_dimensions():
    executor_module = _load_executor_module()
    template = _load_template("substrate_observe.j2")
    ctx = _base_ctx(
        target_id="node:substrate.route",
        motivating_dimensions={"resource_pressure": 0.05},
        priority_score=0.05,
        risk_score=0.02,
    )
    prompt = executor_module._render_prompt(template, ctx)
    assert "REAL TELEMETRY" in prompt
    assert "resource_pressure: 0.050" in prompt


def test_substrate_prompts_degrade_gracefully_without_motivating_dimensions():
    """A context missing motivating_dimensions/priority_score/risk_score
    (e.g. an older persisted request_envelope, or any other caller of these
    templates that hasn't been updated) must not crash the render -- it
    should fall back to an explicit "no reading" statement, not raise.
    """
    executor_module = _load_executor_module()
    for name in ("substrate_inspect.j2", "substrate_summarize.j2", "substrate_observe.j2"):
        template = _load_template(name)
        ctx = _base_ctx()
        prompt = executor_module._render_prompt(template, ctx)
        assert "REAL TELEMETRY" in prompt
        assert "unavailable" in prompt
        assert "none recorded for this target" in prompt


def test_substrate_prompts_degrade_gracefully_with_explicit_none_dimensions():
    """Same as the missing-key case above, but for an explicit ``None`` value
    (as opposed to the key being entirely absent from ctx) -- Jinja's
    ``default`` filter only substitutes on ``Undefined`` unless told to also
    treat ``None`` as falsy via ``default(fallback, true)``. Caught in code
    review: the first cut of these templates used a bare
    ``motivating_dimensions | default({})``, which would have raised
    ``AttributeError: 'NoneType' object has no attribute 'items'`` on an
    explicit ``None`` (not currently reachable from the real producer, since
    ``ProposalCandidateV1.motivating_dimensions`` is never ``None`` through
    pydantic, but worth locking down given how easily ctx dicts get
    mutated/merged elsewhere in this service).
    """
    executor_module = _load_executor_module()
    for name in ("substrate_inspect.j2", "substrate_summarize.j2", "substrate_observe.j2"):
        template = _load_template(name)
        ctx = _base_ctx(motivating_dimensions=None, priority_score=None, risk_score=None)
        prompt = executor_module._render_prompt(template, ctx)
        assert "REAL TELEMETRY" in prompt
        assert "unavailable" in prompt
        assert "none recorded for this target" in prompt
