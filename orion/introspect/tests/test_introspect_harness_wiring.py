"""The harness attaches orion-introspect only when it should, and the model never sets the binding."""
import json

import pytest

from orion.fcc import mcp_config
from orion.fcc.mcp_config import McpPreflightError
from orion.harness import fcc_motor as motor
from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import make_thought
from orion.introspect.binding import introspect_binding_for_turn
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.introspect import IntrospectToolBindingV1
from orion.schemas.reading import ReadingToolBindingV1

READING = ReadingToolBindingV1(invocation_context="curiosity", parent_run_id="run-9", parent_trace_id="trace-9")
BUS = "redis://100.92.216.81:6379/0"
FCC_ENV = {
    "GITHUB_PAT": "x", "FIRECRAWL_API_KEY": "y",
    "AITOWN_CONVEX_URL": "http://convex.test", "AITOWN_ADMIN_KEY": "k", "AITOWN_WORLD_ID": "w",
}


@pytest.fixture
def harness_env(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(mcp_config, "_TMP_ROOT", tmp_path)
    monkeypatch.setattr(mcp_config, "_probe_convex_version", lambda *a, **k: None)
    monkeypatch.setattr(mcp_config, "_probe_convex_auth", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", lambda _path: dict(FCC_ENV))
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_INTROSPECT_ENABLED", "true")
    monkeypatch.setenv("HARNESS_AITOWN_ENABLED", "false")
    monkeypatch.setenv("ORION_BUS_URL", BUS)
    for key in ("HARNESS_FCC_GITNEXUS_ENABLED", "HARNESS_FCC_CONTEXT_MODE_ENABLED",
                "HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "HARNESS_AITOWN_CONVEX_URL"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _servers(**kwargs):
    path = motor._maybe_render_mcp_config(correlation_id="corr-introspect", **kwargs)
    return json.loads(path.read_text())["mcpServers"]


def test_attached_with_flag_and_binding(harness_env):
    server = _servers(reading_binding=READING)["orion-introspect"]
    assert server["type"] == "stdio"
    assert server["args"] == ["-P", "-m", "orion.introspect.mcp_server"]
    assert set(server["env"]) == {"ORION_BUS_URL", "PYTHONPATH", "ORION_INTROSPECT_BINDING"}
    assert server["env"]["ORION_BUS_URL"] == BUS
    binding = IntrospectToolBindingV1.model_validate_json(server["env"]["ORION_INTROSPECT_BINDING"])
    assert binding == IntrospectToolBindingV1(
        invocation_context="curiosity", parent_run_id="run-9", parent_trace_id="trace-9", memory_allowed=True,
    )


def test_ai_town_turns_get_memory_disallowed(harness_env):
    harness_env.setenv("HARNESS_AITOWN_ENABLED", "true")
    servers = _servers(reading_binding=READING)
    assert "orion-aitown" in servers
    binding = IntrospectToolBindingV1.model_validate_json(servers["orion-introspect"]["env"]["ORION_INTROSPECT_BINDING"])
    assert binding.memory_allowed is False


def test_absent_without_flag_or_binding(harness_env):
    assert "orion-introspect" not in _servers()
    harness_env.setenv("HARNESS_FCC_INTROSPECT_ENABLED", "false")
    assert "orion-introspect" not in _servers(reading_binding=READING)


def test_reading_only_turns_stay_empty(harness_env):
    assert _servers(reading_binding=READING, reading_only=True) == {}


def test_render_refuses_memory_access_alongside_ai_town(harness_env, tmp_path):
    risky = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
    )
    with pytest.raises(McpPreflightError) as exc:
        mcp_config.render_mcp_config(
            correlation_id="c", fcc_env=dict(FCC_ENV), tmp_dir=tmp_path,
            include_aitown=True, introspect_binding=risky, introspect_bus_url=BUS,
        )
    assert exc.value.error_code == "fcc_introspect_outward_memory"


def test_render_requires_bus_url(harness_env, tmp_path):
    binding = introspect_binding_for_turn(READING)
    with pytest.raises(McpPreflightError) as exc:
        mcp_config.render_mcp_config(
            correlation_id="c", fcc_env=dict(FCC_ENV), tmp_dir=tmp_path, introspect_binding=binding,
        )
    assert exc.value.error_code == "fcc_introspect_bus_missing"


def test_binding_requires_master_mcp_flag(harness_env):
    harness_env.delenv("HARNESS_FCC_MCP_ENABLED")
    assert introspect_binding_for_turn(READING) is None


def test_introspect_calls_count_as_context_gathering():
    assert motor.classify_step_tool_kind("mcp__orion-introspect__reading_results") == "context_gathering"


def _prefix(**kwargs):
    return compile_harness_prefix(
        make_thought(imperative="What did you learn from that article?"),
        repair_overlay=HarnessRepairOverlayV1(),
        **kwargs,
    )


def test_brief_present_only_when_server_attached(harness_env):
    harness_env.delenv("ORION_GITHUB_OWNER", raising=False)
    harness_env.delenv("ORION_GITHUB_REPO", raising=False)
    prompt = _prefix(reading_binding=READING)
    assert "orion-introspect" in prompt
    assert "reading_results" in prompt
    assert "the answer is unknown" in prompt
    assert "not settled beliefs" in prompt
    assert "orion-introspect" not in _prefix(reading_binding=READING, reading_only=True)
    assert "orion-introspect" not in _prefix()
    harness_env.setenv("HARNESS_FCC_INTROSPECT_ENABLED", "false")
    assert "orion-introspect" not in _prefix(reading_binding=READING)


def test_brief_tells_orion_to_search_by_meaning():
    from orion.introspect.brief import introspect_brief_lines

    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
    )
    text = " ".join(introspect_brief_lines(binding))
    assert "query=" in text and "similarity" in text
    assert "answer is unknown" in text


def test_brief_covers_dreams_as_experiences_not_facts():
    from orion.introspect.brief import introspect_brief_lines
    from orion.schemas.introspect import IntrospectToolBindingV1

    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=False,
    )
    [dreams_line] = [line for line in introspect_brief_lines(binding) if line.startswith("dreams ")]
    assert "not facts" in dreams_line
    assert "items=[] means no dream matched" in dreams_line
    assert "a tool error means the answer is unknown" in dreams_line
    assert "never report it as not having dreamed" in dreams_line
    assert "'pull requests', not 'a dream about pull requests'" in dreams_line
    assert "kind=narrative returns the nightly dream narratives" in dreams_line
    assert "kind=hypothesis the offered sleep-cycle hypotheses" in dreams_line
