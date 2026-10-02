"""The #27148 bleed canary (scripts/probe_slot_bleed.py) against a fake llama.cpp server: it must
call a leak a leak, refuse a wrong or busy worker, and never report PASS on a run that could not
have seen a leak."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_slot_bleed.py"
CANARY = re.compile(r"CW-[A-Z0-9]{8}")


def _load():
    spec = importlib.util.spec_from_file_location("probe_slot_bleed", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


class Fake:
    """llama.cpp's /props, /slots, /v1/chat/completions. ``mode``: honest | leak | cache | short | mute."""

    def __init__(self, mode="honest", model="/models/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", slots=2, busy=False):
        self.mode, self.model, self.slots, self.busy = mode, model, slots, busy
        self.bodies: list[dict] = []
        self.previous: str | None = None
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _json(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/props":
                    self._json({"model_path": fake.model, "total_slots": fake.slots,
                                "default_generation_settings": {"n_ctx": 131072}})
                elif self.path == "/slots":
                    self._json([{"id": i, "is_processing": fake.busy} for i in range(fake.slots)])
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body)
                text = json.dumps(body["messages"])
                own = CANARY.findall(text)[0]
                words = len(text.split())
                reply = own
                if fake.mode == "leak" and fake.previous:
                    reply = f"{own}\n{fake.previous}"
                if fake.mode == "mute":
                    reply = "I cannot find any codeword."
                fake.previous = own
                prompt_n = 100 if fake.mode == "short" else int(words * 1.3)
                cache_n = 4000 if fake.mode == "cache" else 3
                self._json({"choices": [{"message": {"content": reply}}],
                            "timings": {"cache_n": cache_n, "prompt_n": prompt_n}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    servers = []

    def make(**kw):
        servers.append(Fake(**kw))
        return servers[-1]
    yield make
    for s in servers:
        s.close()


def _run(fake_server, tmp_path, **kw):
    mod = _load()
    args = ["--url", fake_server.url, "--out", str(tmp_path / "out"), "--rounds", "1", "--pace-sec", "0",
            "--idle-wait-sec", "0", "--seed", "7", *kw.pop("extra", [])]
    code = mod.main(args)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    return code, summary


def test_honest_worker_passes_with_every_phase_and_long_prompts(fake, tmp_path):
    server = fake()
    code, summary = _run(server, tmp_path)
    assert (code, summary["verdict"]) == (0, "PASS"), summary
    assert set(summary["phases"]) == {"seed", "sequential", "pinned_A", "pinned_B", "simultaneous",
                                      "tool_turn", "shared_prefix"}
    assert summary["prompt_tokens_min"] >= 4500
    # Tool-call turns really carry an assistant tool_call and a tool result with the canary.
    tool_bodies = [b for b in server.bodies if b.get("tools")]
    assert tool_bodies and all(any(m.get("tool_calls") for m in b["messages"]) for b in tool_bodies)
    assert all(CANARY.search(next(m["content"] for m in b["messages"] if m["role"] == "tool")) for b in tool_bodies)
    assert {b.get("id_slot") for b in server.bodies} >= {0, 1}
    # Thinking off per request (Bonsai rejects reasoning_effort none/high; enable_thinking false works).
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in server.bodies)
    lines = (tmp_path / "out" / "requests.jsonl").read_text().splitlines()
    assert len(lines) == summary["requests"] > 10


def test_a_foreign_canary_is_a_leak(fake, tmp_path):
    code, summary = _run(fake(mode="leak"), tmp_path)
    assert (code, summary["verdict"]) == (1, "LEAK")
    assert summary["leaks"] and summary["leaks"][0]["foreign_convs"]


def test_cache_reuse_past_the_header_on_unrelated_prompts_is_a_leak(fake, tmp_path):
    code, summary = _run(fake(mode="cache"), tmp_path)
    assert (code, summary["verdict"]) == (1, "LEAK")
    assert summary["cache_over_header"]
    assert all(r["phase"] != "shared_prefix" for r in summary["cache_over_header"])


@pytest.mark.parametrize("kw,why", [
    ({"model": "/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"}, "does not contain"),
    ({"slots": 1}, "total_slots=1"),
    ({"busy": True}, "busy"),
])
def test_refuses_the_wrong_worker_or_a_busy_one(fake, tmp_path, kw, why):
    server = fake(**kw)
    code, summary = _run(server, tmp_path)
    assert (code, summary["verdict"]) == (2, "REFUSED") and why in summary["why"]
    assert server.bodies == []   # not one request against real work or the wrong model


@pytest.mark.parametrize("mode,why", [("short", "under 4500 tokens"), ("mute", "own codeword")])
def test_a_run_that_could_not_see_a_leak_is_inconclusive_not_pass(fake, tmp_path, mode, why):
    code, summary = _run(fake(mode=mode), tmp_path)
    assert (code, summary["verdict"]) == (3, "INCONCLUSIVE") and why in summary["why"]


def test_an_unreachable_worker_is_refused(tmp_path):
    mod = _load()
    code = mod.main(["--url", "http://127.0.0.1:9", "--out", str(tmp_path / "o"), "--idle-wait-sec", "0"])
    assert code == 2
