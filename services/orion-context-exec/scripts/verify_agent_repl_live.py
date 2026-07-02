"""Gate 1 live proof: run agent_repl against the live stack and print the step trace.

Usage:
  ./venv/bin/python services/orion-context-exec/scripts/verify_agent_repl_live.py \
      --url http://127.0.0.1:8096 \
      --text "what would happen if we changed the orion-hub runtime?"
Exit 0 iff: status==ok, >=2 steps, >=1 ok step, clean non-error final_text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8096")
    ap.add_argument("--text", default="what would happen if we changed the orion-hub runtime?")
    ap.add_argument("--timeout", type=float, default=650.0)
    args = ap.parse_args()

    body = {
        "text": args.text,
        "mode": "agent_repl",
        "llm_profile": "agent",
        "permissions": {"read_repo": True, "read_recall": True},
    }
    started = time.time()
    resp = requests.post(f"{args.url}/context-exec/run", json=body, timeout=args.timeout)
    elapsed = time.time() - started
    resp.raise_for_status()
    run = resp.json()

    steps = run.get("verb_trace") or []
    ok_steps = [s for s in steps if s.get("status") == "ok"]
    print(
        f"status={run.get('status')} mode={run.get('mode')} elapsed={elapsed:.1f}s "
        f"steps={len(steps)} ok_steps={len(ok_steps)}"
    )
    for s in steps:
        print(f"  [{s.get('step_index')}] {s.get('callable')} status={s.get('status')} "
              f"dur_ms={s.get('duration_ms')} in={str(s.get('input_summary'))[:80]!r} "
              f"out={str(s.get('output_summary'))[:80]!r}")
    print("final_text:")
    print(run.get("final_text"))
    print("runtime_debug:", json.dumps(run.get("runtime_debug", {}), indent=2)[:1500])

    error_markers = ("Error in code parsing", "Connection refused", "llamacpp failed", "Reached max steps")
    final_text = str(run.get("final_text") or "").strip()
    ok = (
        run.get("status") == "ok"
        and len(steps) >= 2
        and len(ok_steps) >= 1
        and bool(final_text)
        and not any(m in final_text for m in error_markers)
    )
    print("GATE1_PASS" if ok else "GATE1_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
