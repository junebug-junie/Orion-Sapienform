"""Opt-in replay of one real compaction request: tools-stripped shape vs flattened shape.

Run inside the FCC container (needs the pinned upstream and llm-gateway on the network):

    docker cp services/orion-fcc/orion_fcc_messages_transport.py orion-athena-fcc:/tmp/flat/
    docker cp services/orion-fcc/evals/compaction_replay.py orion-athena-fcc:/tmp/flat/
    docker exec orion-athena-fcc python /tmp/flat/compaction_replay.py --line <N>

<N> is the 0-based line of a `TRACE api.request.received` record in /tmp/fcc-server.log
whose last user turn is the CLI's compaction prompt. Passes when the flattened shape
returns a <summary> block and no tool call written as text.
Live 2026-10-30 (line 288): stripped -> tool call as text in 3 s; flattened -> summary in 39 s.
"""

import argparse
import ast
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.models.anthropic import MessagesRequest  # noqa: E402
from orion_fcc_messages_transport import is_compaction_request, prepare_request  # noqa: E402


def ask(url, request, max_tokens):
    body = request.model_dump(mode="json", exclude_none=True)
    body.pop("thinking", None)
    body.update(stream=False, max_tokens=max_tokens)
    sent = urllib.request.Request(
        url, json.dumps(body).encode(), {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    )
    with urllib.request.urlopen(sent, timeout=900) as response:
        out = json.load(response)
    return "\n".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/tmp/fcc-server.log")
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--url", default="http://llm-gateway:8210/v1/messages")
    parser.add_argument("--max-tokens", type=int, default=6000)
    args = parser.parse_args()

    with open(args.log, errors="ignore") as log:
        record = json.loads(next(line for n, line in enumerate(log) if n == args.line))
    snapshot = record["snapshot"]
    request = MessagesRequest.model_validate(snapshot if isinstance(snapshot, dict) else ast.literal_eval(snapshot))
    if not is_compaction_request(request):
        sys.exit("that log line is not a compaction request")

    results = {}
    for name, shaped in (
        ("stripped", request.model_copy(update={"tools": None, "tool_choice": None})),
        ("flattened", prepare_request(request)),
    ):
        started = time.time()
        text = ask(args.url, shaped, args.max_tokens)
        ok = "<summary>" in text and "<tool_call>" not in text
        results[name] = ok
        print(f"{name}: {time.time() - started:.0f}s summary_ok={ok}")
        print("  " + text[:400].replace("\n", " | "))
    sys.exit(0 if results["flattened"] else 1)


if __name__ == "__main__":
    main()
