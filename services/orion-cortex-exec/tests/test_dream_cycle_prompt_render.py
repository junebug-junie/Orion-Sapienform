"""dream_cycle.j2 through the real exec renderer: a sleep-started dream leads with
that sleep's replay; a hand-started one renders the old memory-only prompt."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "orion" / "cognition" / "prompts" / "dream_cycle.j2"


def _load_executor_module():
    app_dir = EXEC_ROOT / "app"
    package_name = "orion_cortex_exec_dream_prompt_render"
    app_package_name = f"{package_name}.app"
    for name, path in ((package_name, app_dir.parent), (app_package_name, app_dir)):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.executor", app_dir / "executor.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _render(ctx):
    return _load_executor_module()._render_prompt(TEMPLATE_PATH.read_text(encoding="utf-8"), ctx)


SLEEP = {
    "cycle_id": "dc-abc123", "started_at": "2026-10-09T06:27:00Z", "pressure": 13.26, "overdue": False,
    "replay": ["metacog: transport:rpc_timeout on the gateway", "resonance: ring_quiet"],
}


def test_a_sleep_started_dream_leads_with_the_replay():
    prompt = _render({"memory_digest": "MEMORIES", "metadata": {"dream_trigger": {"mode": "standard", "sleep": SLEEP}}})
    assert "TONIGHT'S SLEEP" in prompt and "dc-abc123" in prompt and "13.26" in prompt
    assert "- metacog: transport:rpc_timeout on the gateway" in prompt and "- resonance: ring_quiet" in prompt
    assert prompt.index("TONIGHT'S SLEEP") < prompt.index("MEMORIES")
    assert "overdue" not in prompt.split("TASK")[0]
    assert '"narrative"' in prompt  # the output contract is unchanged


def test_a_hand_started_dream_renders_the_memory_only_prompt():
    for ctx in ({"memory_digest": "MEMORIES"},
                {"memory_digest": "MEMORIES", "metadata": {"dream_trigger": {"mode": "standard", "sleep": None}}},
                {"memory_digest": "MEMORIES", "metadata": {}}):
        prompt = _render(ctx)
        assert "TONIGHT'S SLEEP" not in prompt
        assert "MEMORY BUNDLE (rendered recall; primary grounding):\nMEMORIES" in prompt
        assert "Synthesize a dream narrative from the memory bundle and mode." in prompt
