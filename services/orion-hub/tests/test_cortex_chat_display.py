from __future__ import annotations

from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult

from scripts.cortex_chat_display import hub_effective_chat_text


def test_hub_effective_chat_text_prefers_longer_nested_final_text():
    short = "short"
    long_text = "x" * 100
    cr = CortexClientResult(
        ok=True,
        mode="brain",
        verb="skills.mesh.tailscale_mesh_status.v1",
        status="success",
        final_text=long_text,
    )
    resp = CortexChatResult(cortex_result=cr, final_text=short)
    assert hub_effective_chat_text(resp) == long_text


def test_hub_effective_chat_text_uses_top_when_longer():
    top = "y" * 50
    nested = "z" * 10
    cr = CortexClientResult(
        ok=True,
        mode="brain",
        verb="chat_general",
        status="success",
        final_text=nested,
    )
    resp = CortexChatResult(cortex_result=cr, final_text=top)
    assert hub_effective_chat_text(resp) == top
