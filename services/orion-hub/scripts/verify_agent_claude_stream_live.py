"""Live smoke: Hub agent-claude WS streams claude_step + final llm_response.

Usage:
  PYTHONPATH=services/orion-hub:. python services/orion-hub/scripts/verify_agent_claude_stream_live.py \\
      --ws ws://127.0.0.1:8080/ws \\
      --text "list files in services/orion-hub/scripts" \\
      --fcc-model-label MODEL_HAIKU
Exit 0 iff: >=1 claude_step frames AND final llm_response non-empty.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import websockets


async def run(ws_url: str, text: str, fcc_label: str, timeout: float) -> int:
    steps = 0
    final = ""
    session_id = f"agent-claude-{uuid.uuid4().hex[:8]}"
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({
            "mode": "agent-claude",
            "text": text,
            "session_id": session_id,
            "fcc_model_label": fcc_label,
            "claude_session_id": None,
            "resume": False,
        }))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                d = json.loads(raw)
                if d.get("kind") == "claude_step":
                    steps += 1
                    step = d.get("step") or {}
                    print(f"  step type={step.get('type')}")
                elif d.get("llm_response"):
                    final = str(d.get("llm_response"))
                    break
                elif d.get("error"):
                    print("error:", d.get("error"), d.get("error_code"))
                    break
        except asyncio.TimeoutError:
            print("timeout waiting for frames")

    print(f"steps={steps} final_len={len(final)}")
    print("final_text:", final[:500])
    ok = steps >= 1 and bool(final.strip())
    print("AGENT_CLAUDE_PASS" if ok else "AGENT_CLAUDE_FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8080/ws")
    ap.add_argument("--text", default="list files in services/orion-hub/scripts")
    ap.add_argument("--fcc-model-label", default="MODEL_HAIKU")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    return asyncio.run(run(args.ws, args.text, args.fcc_model_label, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
