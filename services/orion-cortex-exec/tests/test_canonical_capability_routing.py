"""Canonical phrase → semantic capability verb (supervisor rails before agent-chain)."""

from app.supervisor import _select_canonical_operational_tool
from orion.schemas.agents.schemas import ToolDef


def _toolset_stub(*ids: str) -> list[ToolDef]:
    return [
        ToolDef(
            tool_id=i,
            description=i,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            execution_mode="capability_backed",
            requires_capability_selector=True,
            preferred_skill_families=[],
            side_effect_level="none",
        )
        for i in ids
    ]


def test_canonical_routes_hub_prompts_to_new_semantic_verbs():
    tools = _toolset_stub(
        "answer_current_datetime",
        "inspect_gpu_status",
        "show_biometrics_snapshot",
        "list_biometrics_recent_readings",
        "inspect_docker_container_status",
        "send_operator_notification",
        "show_landing_pad_metrics",
        "assess_mesh_presence",
    )
    assert _select_canonical_operational_tool("What time is it right now?", tools).tool_id == "answer_current_datetime"
    assert _select_canonical_operational_tool("Show NVIDIA GPU status on this node.", tools).tool_id == "inspect_gpu_status"
    assert _select_canonical_operational_tool("Show the current biometrics snapshot.", tools).tool_id == "show_biometrics_snapshot"
    assert _select_canonical_operational_tool("Show the 10 most recent biometrics readings.", tools).tool_id == "list_biometrics_recent_readings"
    assert (
        _select_canonical_operational_tool('Send a notification to operators saying "test alert from Orion".', tools).tool_id
        == "send_operator_notification"
    )
    assert _select_canonical_operational_tool("Show the landing pad metrics snapshot.", tools).tool_id == "show_landing_pad_metrics"
    assert (
        _select_canonical_operational_tool("Show Docker container status on this node.", tools).tool_id
        == "inspect_docker_container_status"
    )
