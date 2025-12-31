from __future__ import annotations

"""
Minimal bus harness to prove Client → Bus → Orch → Exec → Workers → Bus → Client.

Usage:
  python scripts/bus_harness.py brain "hello world"
  python scripts/bus_harness.py agent "plan a trip"
  python scripts/bus_harness.py tap  (pattern-subscribe to orion:*)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List
from uuid import uuid4

# Ensure repository root is on sys.path so orion modules are importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())

# Load harness-specific env (first .env, then .env_example) without overriding existing env vars.
_load_env_file(ROOT / "scripts" / ".env")
_load_env_file(ROOT / "scripts" / ".env_example")

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
from orion.schemas.cortex.contracts import (
    CortexClientContext,
    CortexClientRequest,
    RecallDirective,
)


def _service_ref(name: str, version: str, node: str | None) -> ServiceRef:
    return ServiceRef(name=name, version=version, node=node)


async def _rpc_request(
    *,
    bus_url: str,
    channel: str,
    reply_prefix: str,
    req: CortexClientRequest,
    service_name: str,
    service_version: str,
    node: str | None,
    timeout: float,
    ) -> dict:
    bus = OrionBusAsync(url=bus_url)
    await bus.connect()
    try:
        corr = uuid4()
        reply = f"{reply_prefix}:{corr}"
        env = BaseEnvelope(
            kind="cortex.orch.request",
            source=_service_ref(service_name, service_version, node),
            correlation_id=corr,
            reply_to=reply,
            payload=req.model_dump(mode="json"),
        )
        msg = await bus.rpc_request(channel, env, reply_channel=reply, timeout_sec=timeout)
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        payload = decoded.envelope.payload
        return payload if isinstance(payload, dict) else decoded.envelope.model_dump(mode="json")
    except TimeoutError as te:
        raise TimeoutError(
            f"RPC timeout waiting on reply_channel={reply} (request_channel={channel}, bus={bus_url}). "
            "Ensure orion-cortex-orch is running and subscribed to the intake channel."
        ) from te
    finally:
        await bus.close()


async def _tap(bus_url: str) -> None:
    bus = OrionBusAsync(url=bus_url)
    await bus.connect()
    print(f"[tap] subscribing to orion:* on {bus_url}")
    async with bus.subscribe("orion:*", patterns=True) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            decoded = bus.codec.decode(msg.get("data"))
            env = decoded.envelope
            print(
                json.dumps(
                    {
                        "channel": msg.get("channel") or msg.get("pattern"),
                        "kind": env.kind,
                        "correlation_id": str(env.correlation_id),
                        "reply_to": env.reply_to,
                    }
                )
            )


def _build_request(mode: str, text: str, args: argparse.Namespace) -> CortexClientRequest:
    recall = RecallDirective(
        enabled=not args.disable_recall,
        required=args.require_recall,
        mode=args.recall_mode,
        time_window_days=args.time_window_days,
        max_items=args.max_items,
    )
    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content=text)],
        session_id=args.session_id,
        user_id=args.user_id,
        trace_id=args.trace_id,
        metadata={"harness": True},
    )
    return CortexClientRequest(
        mode=mode,
        verb=args.verb,
        packs=args.packs,
        options={"temperature": args.temperature, "max_tokens": args.max_tokens},
        recall=recall,
        context=ctx,
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orion bus harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--bus-url", default="redis://100.92.216.81:6379/0")
        p.add_argument(
            "--channel",
            default=os.getenv("ORCH_REQUEST_CHANNEL", os.getenv("CORTEX_REQUEST_CHANNEL", "orion-cortex:request")),
            help="Request channel for cortex-orch (defaults to ORCH_REQUEST_CHANNEL/CORTEX_REQUEST_CHANNEL env or orion-cortex:request)",
        )
        p.add_argument(
            "--reply-prefix",
            default=os.getenv("ORCH_RESULT_PREFIX", "orion-cortex:result"),
            help="Reply channel prefix (defaults to ORCH_RESULT_PREFIX env or orion-cortex:result)",
        )
        p.add_argument("--service-name", default="bus-harness")
        p.add_argument("--service-version", default="0.0.1")
        p.add_argument("--node", default="harness")
        p.add_argument(
            "--timeout",
            type=float,
            default=float(os.getenv("HARNESS_RPC_TIMEOUT_SEC", 120)),
            help="RPC timeout in seconds (defaults to HARNESS_RPC_TIMEOUT_SEC or 120s to match LLM step budgets)",
        )
        p.add_argument("--verb", default="chat_general")
        p.add_argument("--packs", nargs="*", default=["executive_pack"])
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--session-id", default="harness-session")
        p.add_argument("--user-id", default="harness-user")
        p.add_argument("--trace-id", default=None)
        p.add_argument("--disable-recall", action="store_true")
        p.add_argument("--require-recall", action="store_true")
        p.add_argument("--recall-mode", default="hybrid")
        p.add_argument("--time-window-days", type=int, default=90)
        p.add_argument("--max-items", type=int, default=8)

    brain = sub.add_parser("brain", help="Brain chat through Orch/Exec/LLM")
    add_common(brain)
    brain.add_argument("text")

    agent = sub.add_parser("agent", help="Agentic path via AgentChainService")
    add_common(agent)
    agent.add_argument("text")

    tap = sub.add_parser("tap", help="Subscribe to orion:* traffic")
    tap.add_argument("--bus-url", default="redis://100.92.216.81:6379/0")

    return parser.parse_args(argv)


async def _main(argv: List[str]) -> int:
    args = _parse_args(argv)
    if args.cmd == "tap":
        await _tap(args.bus_url)
        return 0


    # Pre-flight correlation notice
    preview_corr = str(uuid4())
    preview_reply = f"{args.reply_prefix}:{preview_corr}"
    print(
        json.dumps(
            {
                "preview_correlation": preview_corr,
                "request_channel": args.channel,
                "reply_channel": preview_reply,
                "timeout_sec": args.timeout,
            },
            indent=2,
        ),
        file=sys.stderr,
    )

    req = _build_request(args.cmd, args.text, args)
    try:
        payload = await _rpc_request(
            bus_url=args.bus_url,
            channel=args.channel,
            reply_prefix=args.reply_prefix,
            req=req,
            service_name=args.service_name,
            service_version=args.service_version,
            node=args.node,
            timeout=args.timeout,
        )

        print(
            json.dumps(
                {
                    "correlation_id": str(payload.get("correlation_id") or "n/a"),
                    "reply_prefix": args.reply_prefix,
                    "request_channel": args.channel,
                },
                indent=2,
                default=str,
            ),
            file=sys.stderr,
        )

        print(json.dumps(payload, indent=2, default=str))
        return 0
    except TimeoutError as te:
        print(f"[bus-harness] {te}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
