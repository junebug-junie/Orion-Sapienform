"""Opt-in real Claude WebFetch smoke; no reading queue or memory writes.

Run inside a container with Claude and ~/.fcc/.env, passing a candidate FCC URL.
Only WebFetch is exposed. Timeout terminates the subprocess group.
"""

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path

from dotenv import dotenv_values


async def run(args):
    env = os.environ.copy()
    config = dotenv_values(Path.home() / ".fcc/.env")
    env.update(
        ANTHROPIC_BASE_URL=args.base_url,
        ANTHROPIC_AUTH_TOKEN=config.get("ANTHROPIC_AUTH_TOKEN") or "fcc",
        ORION_FCC_SUBPROCESS="1",
    )
    env.pop("ANTHROPIC_CUSTOM_HEADERS", None)
    env.pop("ANTHROPIC_API_KEY", None)
    prompt = (
        "Use WebFetch exactly once on https://arxiv.org/abs/2310.19279. "
        "Ask it for the paper title and two concrete claims from the abstract. "
        "Do not search, retry, or fetch other URLs. Report the tool's answer, "
        "or report its error honestly."
    )
    command = [
        "claude",
        "-p",
        prompt,
        "--model",
        "llamacpp/agent",
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        "3",
        "--tools",
        "WebFetch",
        "--allowedTools",
        "WebFetch",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--setting-sources",
        "",
    ]
    proc = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        cwd="/tmp",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        limit=4 * 1024 * 1024,
    )
    started = time.monotonic()
    receipts = []
    tool_ids = set()
    wrong_tool = False
    final_error = False
    final_text = ""
    try:
        async with asyncio.timeout(args.timeout):
            async for line in proc.stdout:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                content = item.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_use":
                            tool_ids.add(block["id"])
                            wrong_tool |= (
                                block.get("name") != "WebFetch"
                                or block.get("input", {}).get("url")
                                != "https://arxiv.org/abs/2310.19279"
                            )
                            print(
                                json.dumps(
                                    {
                                        "elapsed": round(time.monotonic() - started, 1),
                                        "tool": block.get("name"),
                                        "input": block.get("input"),
                                    }
                                ),
                                flush=True,
                            )
                        if block.get("type") == "tool_result":
                            receipts.append(block)
                            print(
                                json.dumps(
                                    {
                                        "elapsed": round(time.monotonic() - started, 1),
                                        "receipt": block,
                                    }
                                ),
                                flush=True,
                            )
                if item.get("type") == "result":
                    final_error = bool(item.get("is_error"))
                    final_text = str(item.get("result") or "")
                    print(
                        json.dumps(
                            {
                                "result": item.get("result"),
                                "error": item.get("is_error"),
                            }
                        ),
                        flush=True,
                    )
            await proc.wait()
    finally:
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
    title = "information dynamics of our brains in dynamically driven disordered superconducting loop networks"
    valid = [
        r
        for r in receipts
        if r.get("tool_use_id") in tool_ids
        and not r.get("is_error")
        and title in str(r.get("content", "")).lower()
    ]
    if (
        proc.returncode
        or wrong_tool
        or final_error
        or title not in final_text.lower()
        or len(tool_ids) != 1
        or len(valid) != 1
    ):
        raise SystemExit(
            "FAIL: expected exactly one successful, substantive WebFetch receipt"
        )
    print(
        "PASS: real Claude WebFetch returned source content and a grounded final reply",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    asyncio.run(run(parser.parse_args()))
