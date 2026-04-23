from __future__ import annotations

import asyncio

from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
from scripts.api_routes import handle_chat_request


class _TurnEffectClient:
    async def chat(self, req, correlation_id=None):
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_general",
            status="success",
            final_text="ok",
            metadata={
                "turn_effect": {"turn": {"novelty": 0.17}},
                "turn_effect_evidence": {"phi_before": {"novelty": 0.11}},
                "turn_effect_status": "present",
                "autonomy_summary": {"stance_hint": "maintain stable direct response"},
            },
            correlation_id=correlation_id or "corr-1",
        )
        return CortexChatResult(cortex_result=result, final_text="ok")


def test_handle_chat_request_exports_turn_effect_into_result_payload() -> None:
    payload = {"mode": "brain", "messages": [{"role": "user", "content": "hello"}]}
    out = asyncio.run(handle_chat_request(_TurnEffectClient(), payload, "sid-1", no_write=True))
    assert out["turn_effect"]["turn"]["novelty"] == 0.17
    assert out["turn_effect_status"] == "present"
    assert out["spark_meta"]["turn_effect"]["turn"]["novelty"] == 0.17

