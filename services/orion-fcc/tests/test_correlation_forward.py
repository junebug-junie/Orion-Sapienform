"""The harness turn id survives the FCC hop (regression, 2026-10-29).

Claude Code sends ``X-Orion-Correlation-Id`` on every model call. FCC builds a new upstream
request to llm-gateway, and before this fix it dropped the header: live, all 17 harness rows in
``llm_call_usage`` had ``correlation_id`` NULL and the gateway logged ``corr=-``.

These run the REAL patched FCC app (``api.app``, real llama.cpp provider) against a stub
upstream, in a fresh process each, so settings and provider caches are this test's own.
"""

import json
import os
import subprocess
import sys
import unittest

PROBE = r'''
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

seen = []
SSE = "".join(
    f"event: {k}\ndata: {json.dumps({'type': k, **v})}\n\n"
    for k, v in [
        ("message_start", {"message": {"id": "m", "type": "message", "role": "assistant", "content": [],
                                       "model": "agent", "stop_reason": None, "stop_sequence": None,
                                       "usage": {"input_tokens": 3, "output_tokens": 0}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {}),
    ]
).encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"data":[{"id":"agent"}]}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
        seen.append((payload["messages"][0]["content"], self.headers.get("X-Orion-Correlation-Id")))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(SSE)))
        self.end_headers()
        self.wfile.write(SSE)


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
model = "llamacpp/agent"
os.environ.update(LLAMACPP_BASE_URL=f"http://127.0.0.1:{srv.server_address[1]}/v1", MODEL=model,
                  MODEL_SONNET=model, MODEL_HAIKU=model, MODEL_OPUS=model, ANTHROPIC_AUTH_TOKEN="")

from fastapi.testclient import TestClient
import api.app as appmod

app = appmod.create_app() if hasattr(appmod, "create_app") else appmod.app


def call(client, tag, headers, stream):
    r = client.post("/v1/messages", headers=headers, json={
        "model": "claude-sonnet-4", "max_tokens": 8, "stream": stream,
        "messages": [{"role": "user", "content": tag}]})
    assert r.status_code == 200, (r.status_code, r.text[:300])


with TestClient(app) as client:
    call(client, "stream", {"X-Orion-Correlation-Id": "turn-stream"}, True)
    call(client, "nonstream", {"X-Orion-Correlation-Id": "turn-json"}, False)
    call(client, "none", {}, True)
    call(client, "long", {"X-Orion-Correlation-Id": "t" * 300}, True)
    # Concurrent turns must each carry their own id, never a neighbour's.
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: call(client, f"c{i}", {"X-Orion-Correlation-Id": f"turn-{i}"}, True),
                      range(8)))
print(json.dumps(seen))
'''


def run_probe():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120,
                         env={**os.environ})
    if out.returncode != 0:
        raise AssertionError(out.stderr[-3000:])
    return dict(json.loads(out.stdout.strip().splitlines()[-1]))


class CorrelationForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seen = run_probe()

    def test_streamed_turn_forwards_its_id(self):
        self.assertEqual(self.seen["stream"], "turn-stream")

    def test_non_streamed_turn_forwards_its_id(self):
        self.assertEqual(self.seen["nonstream"], "turn-json")

    def test_no_header_sends_none(self):
        self.assertIsNone(self.seen["none"])

    def test_long_id_is_clipped_to_the_gateway_limit(self):
        self.assertEqual(self.seen["long"], "t" * 128)

    def test_concurrent_turns_never_swap_ids(self):
        for i in range(8):
            self.assertEqual(self.seen[f"c{i}"], f"turn-{i}")

    def test_forward_is_installed_when_the_app_imports(self):
        probe = ("import api.app\n"
                 "from providers.transports.anthropic_messages.transport import AnthropicMessagesTransport as T\n"
                 "assert T._request_headers.orion_correlation_forward")
        subprocess.run([sys.executable, "-c", probe], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
