"""Live contract smoke for system-only Cortex Exec prompts on Qwen3.5 metacog.

Run from repo root with ORION_BUS_URL set to the Tailscale bus URL. Off Athena,
also set ORION_LLM_GATEWAY_URL to Athena's Tailscale gateway URL:
    PYTHONPATH=services/orion-cortex-exec:. python services/orion-cortex-exec/evals/smoke_system_only_metacog.py

The operator lease is last in the GPU pool queue. It never calls the worker directly.
"""

from __future__ import annotations

import os

from app.executor import _build_hop_messages
from orion.evals.leased_llm import operator_lane_sync


def main() -> None:
    messages = _build_hop_messages(prompt="Reply with the single word OK.", ctx_messages=None)
    if [message["role"] for message in messages] != ["system", "user"]:
        raise RuntimeError("Cortex Exec did not add the service task turn")
    with operator_lane_sync(
        "metacog", holder="operator:system-only-metacog-smoke",
        bus_url=os.environ["ORION_BUS_URL"], queue_wait_sec=120,
    ) as lane:
        reply = lane.chat({"messages": messages, "temperature": 0, "max_tokens": 16, "stream": False}, timeout_sec=45)
        if lane.role != "fast":
            raise RuntimeError(f"metacog was served by unexpected pool role {lane.role!r}")
        model = str(reply.get("model") or "").casefold()
        if "qwen3.5" not in model:
            raise RuntimeError(f"metacog was served by unexpected model {model!r}")
        answer = (reply["choices"][0]["message"].get("content") or "").strip()
        if not answer:
            raise RuntimeError("Qwen3.5 served an empty system-only task response")
        print(f"PASS role_shape=system,user pool_role={lane.role} response_nonempty=true")


if __name__ == "__main__":
    main()
