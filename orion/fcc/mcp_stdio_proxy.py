"""stdio MCP proxy: truncate oversized tool results before they reach Claude Code."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any

from orion.fcc.context_budget import mcp_tool_result_max_chars, tool_result_body_text


def _truncate_mcp_payload(obj: Any, *, max_chars: int) -> tuple[Any, bool]:
    """Walk MCP JSON-RPC result and truncate text tool payloads."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        changed = False
        for key, value in obj.items():
            if key == "content" and isinstance(value, list):
                new_content = []
                for block in value:
                    if not isinstance(block, dict):
                        new_content.append(block)
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        text = block["text"]
                        if len(text) > max_chars:
                            snippet = text[: max_chars - 120]
                            block = {
                                **block,
                                "text": (
                                    f"{snippet}\n\n"
                                    f"[orion-fcc-mcp-proxy: truncated {len(text)} chars to "
                                    f"{max_chars}. Summarize from this excerpt; do not re-fetch "
                                    f"the same bulk list.]"
                                ),
                            }
                            changed = True
                    new_content.append(block)
                out[key] = new_content
                continue
            if key == "result" and isinstance(value, (dict, list)):
                nested, nested_changed = _truncate_mcp_payload(value, max_chars=max_chars)
                out[key] = nested
                changed = changed or nested_changed
                continue
            out[key] = value
        return out, changed
    if isinstance(obj, list):
        items = []
        changed = False
        for item in obj:
            nested, nested_changed = _truncate_mcp_payload(item, max_chars=max_chars)
            items.append(nested)
            changed = changed or nested_changed
        return items, changed
    return obj, False


def _maybe_truncate_line(line: str, *, max_chars: int) -> str:
    stripped = line.strip()
    if not stripped:
        return line
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return line
    if not isinstance(payload, dict):
        return line
    truncated, changed = _truncate_mcp_payload(payload, max_chars=max_chars)
    if not changed:
        return line
    return json.dumps(truncated, ensure_ascii=False) + "\n"


def _forward_output(pipe, out_stream, *, max_chars: int) -> None:
    for line in iter(pipe.readline, ""):
        if not line:
            break
        out_stream.write(_maybe_truncate_line(line, max_chars=max_chars))
        out_stream.flush()
    pipe.close()


def _forward_input(in_stream, pipe) -> None:
    for line in iter(in_stream.readline, ""):
        if not line:
            break
        pipe.write(line)
        pipe.flush()
    pipe.close()


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] != "--":
        print(
            "usage: python -m orion.fcc.mcp_stdio_proxy -- <command> [args...]",
            file=sys.stderr,
        )
        return 2
    cmd = args[1:]
    if not cmd:
        print("mcp_stdio_proxy: missing child command", file=sys.stderr)
        return 2
    max_chars = mcp_tool_result_max_chars()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    threading.Thread(
        target=_forward_input,
        args=(sys.stdin, proc.stdin),
        daemon=True,
    ).start()
    _forward_output(proc.stdout, sys.stdout, max_chars=max_chars)
    return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
