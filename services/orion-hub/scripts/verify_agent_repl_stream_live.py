"""Gate 2 live proof: drive the Hub chat WS in agent mode and assert live step frames.

Usage:
  ./venv/bin/python services/orion-hub/scripts/verify_agent_repl_stream_live.py \
      --ws ws://127.0.0.1:8080/ws \
      --text "what would happen if we changed the orion-hub runtime?"
Exit 0 iff: >=2 agent_step frames received AND a final llm_response arrives.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import websockets


async def run(ws_url: str, text: str, timeout: float) -> int:
    steps = 0
    final = ""
    session_id = f"gate2-{uuid.uuid4().hex[:8]}"
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({
            "mode": "agent",
            "text": text,
            "session_id": session_id,
        }))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                d = json.loads(raw)
                if d.get("kind") == "agent_step":
                    steps += 1
                    s = d.get("step", {})
                    print(f"  step #{s.get('step_index')} {s.get('tool_id')} {s.get('duration_ms')}ms")
                elif d.get("llm_response"):
                    final = str(d.get("llm_response"))
                    break
        except asyncio.TimeoutError:
            print("timeout waiting for frames")

    print(f"steps={steps} final_len={len(final)}")
    print("final_text:", final[:500])
    ok = steps >= 2 and bool(final.strip())
    print("GATE2_PASS" if ok else "GATE2_FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8080/ws")
    ap.add_argument("--text", default="what would happen if we changed the orion-hub runtime?")
    ap.add_argument("--timeout", type=float, default=650.0)
    args = ap.parse_args()
    return asyncio.run(run(args.ws, args.text, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
