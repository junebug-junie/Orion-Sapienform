#!/usr/bin/env python3
"""
End-to-end diagnostics for Hub deterministic Skill Runner (catalogue time skill).

Phases:
  0) Offline: build_chat_request() → CortexChatRequest + route debug (no bus).
  1) HTTP: GET /api/session, GET /api/debug/skill-runner-deterministic (if present), POST /api/chat.
  2) WebSocket: same payload shape as Hub UI omitting `mode` (deterministic path).

Run from repo root::

  ORION_HTTP_TIMEOUT=120 ORION_WS_WAIT=120 \\
    python scripts/trace_hub_skill_runner_e2e.py

If you use ``python scripts/...``, the script strips ``sys.path[0]`` (the ``scripts/``
directory) so stdlib imports used by ``websockets`` are not shadowed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import httpx


def _ensure_sys_path_stdlib_safe() -> None:
    """When launched as ``python scripts/this.py``, sys.path[0] is ``.../scripts`` and can
    shadow stdlib modules during later imports (e.g. websockets → uuid → platform)."""
    here = str(Path(__file__).resolve().parent)
    while sys.path and sys.path[0] in ("", here):
        sys.path.pop(0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def phase0_offline(prompt: str) -> None:
    print("\n======== PHASE 0: offline build_chat_request ========")
    root = _repo_root()
    hub = root / "services" / "orion-hub"
    code = f"""
import importlib.util, json, sys
from pathlib import Path
root = Path({str(root)!r})
hub = root / "services" / "orion-hub"
mod_path = hub / "scripts" / "cortex_request_builder.py"
sys.path.insert(0, str(root))
import os
os.chdir(str(hub))
spec = importlib.util.spec_from_file_location("hub_cortex_request_builder_trace", mod_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
prompt = {json.dumps(prompt)}
payload = {{
    "skill_runner_origin": True,
    "skill_runner_lane": "deterministic",
    "verbs": [],
    "use_recall": False,
    "recall_mode": None,
    "recall_profile": None,
    "recall_required": False,
    "packs": [],
}}
req, dbg, _ur = m.build_chat_request(
    payload=payload,
    session_id="trace-session",
    user_id="trace-user",
    trace_id="trace-corr-offline",
    default_mode="brain",
    auto_default_enabled=False,
    source_label="hub_trace_script",
    prompt=prompt,
    messages=[{{"role": "user", "content": prompt}}],
)
dumped = req.model_dump(mode="json")
subset = {{k: dbg.get(k) for k in (
    "skill_runner_deterministic", "skill_runner_lane_requested", "skill_runner_catalogue_verb",
    "verb", "mode", "recall_enabled", "recall_profile", "packs",
)}}
print(json.dumps({{"cortex_chat_request": dumped, "route_debug_subset": subset}}, indent=2)[:12000])
"""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(hub),
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        print("PHASE0 subprocess failed", proc.returncode)
        print(proc.stderr)
        return
    print(proc.stdout)


def _summarize_ws(obj: Dict[str, Any]) -> Dict[str, Any]:
    raw = obj.get("raw") if isinstance(obj.get("raw"), dict) else {}
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    llm = obj.get("llm_response")
    ft = raw.get("final_text")
    return {
        "keys": sorted(obj.keys()),
        "state": obj.get("state"),
        "error": obj.get("error"),
        "error_code": obj.get("error_code"),
        "llm_response_type": type(llm).__name__,
        "llm_response_len": len(str(llm or "")),
        "llm_response_head": str(llm or "")[:240],
        "raw_final_text_len": len(str(ft or "")),
        "raw_final_text_head": str(ft or "")[:240],
        "raw_metadata_keys": sorted(meta.keys())[:40] if meta else [],
        "skill_result_preview": str(meta.get("skill_result"))[:400] if meta.get("skill_result") is not None else None,
    }


def phase1_http(base: str, prefix: str, prompt: str, timeout: float) -> str | None:
    print("\n======== PHASE 1: HTTP ========")
    base = base.rstrip("/")
    prefix = (prefix or "").rstrip("/")
    root = f"{base}{prefix}"
    sid: str | None = None
    with httpx.Client(timeout=timeout) as client:
        r0 = client.get(f"{root}/api/session")
        print("GET /api/session", r0.status_code, r0.text[:500])
        if r0.is_success:
            try:
                sid = str((r0.json() or {}).get("session_id") or "").strip() or None
            except Exception:
                sid = None
        if not sid:
            print("WARN: no session_id; subsequent calls may fail.")

        rprobe = client.get(
            f"{root}/api/debug/skill-runner-deterministic",
            params={"prompt": prompt},
        )
        print("GET /api/debug/skill-runner-deterministic", rprobe.status_code, rprobe.text[:800])

        body = {
            "messages": [{"role": "user", "content": prompt}],
            "skill_runner_origin": True,
            "skill_runner_lane": "deterministic",
            "verbs": [],
            "use_recall": False,
            "recall_mode": None,
            "recall_profile": None,
            "recall_required": False,
            "packs": [],
        }
        headers = {}
        if sid:
            headers["X-Orion-Session-Id"] = sid
        t0 = time.perf_counter()
        try:
            r1 = client.post(f"{root}/api/chat", json=body, headers=headers)
        except httpx.ReadTimeout as e:
            print("POST /api/chat READ TIMEOUT", repr(e), "elapsed_s", round(time.perf_counter() - t0, 2))
            print(
                "DIAGNOSIS: HTTP handler blocked on `await cortex_client.chat(...)` past client timeout. "
                "Same root cause as WS: bus RPC to orion-cortex-gateway not completing."
            )
            return sid
        elapsed = time.perf_counter() - t0
        print("POST /api/chat", r1.status_code, "elapsed_s", round(elapsed, 3), "len", len(r1.content))
        txt = r1.text
        print(txt[:16000])
        if r1.is_success:
            try:
                j = r1.json()
                print(
                    "--- http summary ---",
                    json.dumps(
                        {
                            "text_len": len(str(j.get("text") or "")),
                            "error": j.get("error"),
                            "error_code": j.get("error_code"),
                            "correlation_id": j.get("correlation_id"),
                            "mode": j.get("mode"),
                            "routing_debug_verb": (j.get("routing_debug") or {}).get("verb")
                            if isinstance(j.get("routing_debug"), dict)
                            else None,
                        },
                        indent=2,
                    ),
                )
            except Exception as exc:
                print("json parse error", exc)
    return sid


async def phase2_ws(ws_url: str, prompt: str, session_id: str | None, wait_sec: float) -> None:
    print("\n======== PHASE 2: WebSocket ========")
    import websockets

    payload: Dict[str, Any] = {
        "text_input": prompt,
        "disable_tts": True,
        "no_write": False,
        "use_recall": False,
        "recall_mode": None,
        "recall_profile": None,
        "recall_required": False,
        "packs": [],
        "verbs": [],
        "skill_runner_origin": True,
        "skill_runner_lane": "deterministic",
        "surface_context": {"surface": "hub_desktop", "input_modality": "typed"},
    }
    if session_id:
        payload["session_id"] = session_id
    # Intentionally omit "mode" (browser deterministic path).

    print("WS URL", ws_url)
    print("WS send keys", sorted(payload.keys()))
    deadline = time.monotonic() + wait_sec
    n = 0
    saw_chat_reply = False
    saw_error = False
    try:
        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            await ws.send(json.dumps(payload))
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, min(5.0, wait_sec)))
                except asyncio.TimeoutError:
                    continue
                n += 1
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"--- ws frame {n} non-json len={len(raw)} ---")
                    print(raw[:2000])
                    continue
                print(f"--- ws frame {n} ---")
                print(json.dumps(_summarize_ws(obj), indent=2))
                if obj.get("error") or obj.get("error_code"):
                    saw_error = True
                    print("full error payload keys", list(obj.keys()))
                if (str(obj.get("llm_response") or "").strip()) or obj.get("workflow"):
                    saw_chat_reply = True
                    print("STOP: got assistant-visible payload")
                    break
    except OSError as e:
        print("WS connect failed", repr(e))

    print("\n======== WS SUMMARY ========")
    print("frames_seen", n, "saw_chat_reply", saw_chat_reply, "saw_error", saw_error)
    if not saw_chat_reply and not saw_error:
        print(
            "DIAGNOSIS: Hub did not emit an llm_response within wait window. "
            "This matches a stalled or slow `cortex_client.chat` RPC (bus → gateway → orch → exec). "
            "Phase 0 shows the built CortexChatRequest is valid; fix downstream (bus connectivity, "
            "orion-cortex-gateway, orch, exec, TIMEOUT_SEC, CORTEX_GATEWAY_RPC_TIMEOUT_SEC)."
        )


def _http_to_ws_url(base: str, prefix: str) -> str:
    u = urlparse(base)
    scheme = "wss" if u.scheme == "https" else "ws"
    host = u.netloc
    path = (prefix or "").rstrip("/") + "/ws"
    return f"{scheme}://{host}{path}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-base", default=os.environ.get("ORION_HUB_BASE", "http://127.0.0.1:8080"))
    ap.add_argument("--path-prefix", default=os.environ.get("ORION_HUB_PREFIX", ""))
    ap.add_argument(
        "--prompt",
        default="What time is it right now?",
        help="Exact Skill Runner catalogue prompt",
    )
    ap.add_argument("--http-timeout", type=float, default=float(os.environ.get("ORION_HTTP_TIMEOUT", "180")))
    ap.add_argument("--ws-wait", type=float, default=float(os.environ.get("ORION_WS_WAIT", "60")))
    ap.add_argument("--skip-ws", action="store_true")
    ap.add_argument("--skip-http", action="store_true")
    args = ap.parse_args()

    _ensure_sys_path_stdlib_safe()

    phase0_offline(args.prompt)

    sid: str | None = None
    if not args.skip_http:
        sid = phase1_http(args.hub_base, args.path_prefix, args.prompt, args.http_timeout)

    if not args.skip_ws:
        ws_url = _http_to_ws_url(args.hub_base, args.path_prefix)
        asyncio.run(phase2_ws(ws_url, args.prompt, sid, args.ws_wait))


if __name__ == "__main__":
    main()
