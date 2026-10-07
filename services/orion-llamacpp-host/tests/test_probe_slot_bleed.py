"""The #27148 bleed canary (scripts/probe_slot_bleed.py) against a fake llama.cpp server: it must
call a leak a leak, refuse a wrong or busy worker, and never report PASS on a run that could not
have seen a leak."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
import zlib
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


def _tok(text: str) -> list[int]:
    return [zlib.crc32(w.encode()) for w in text.split()]


class Fake:
    """llama.cpp's /props, /slots, /apply-template, /tokenize, /v1/chat/completions, plus the pool's
    /v1/pool on the same port. Honest replies report cache_n = the true common prefix with an earlier
    prompt (what a correct server reuses). ``mode``: honest | leak | cache | stale | short | mute |
    error500 | crash | busy_later | pool_busy_later | no_tokenize."""

    def __init__(self, mode="honest", model="/models/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", slots=2, busy=False,
                 pool_leases=None):
        self.mode, self.model, self.slots, self.busy = mode, model, slots, busy
        self.pool_leases = list(pool_leases or [])
        self.bodies: list[dict] = []
        self.seen: list[list[int]] = []
        self.previous: str | None = None
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
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
                elif self.path == "/v1/pool":
                    self._json({"leases": fake.pool_leases})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/apply-template":
                    if fake.mode == "no_tokenize":
                        return self._json({"error": "nope"}, 501)
                    return self._json({"prompt": json.dumps([body["messages"], body.get("tools")])})
                if self.path == "/tokenize":
                    return self._json({"tokens": _tok(body["content"])})
                with fake.lock:
                    fake.bodies.append(body)
                    n = len(fake.bodies)
                if fake.mode == "error500" and n % 5 == 0:
                    return self._json({"error": "boom"}, 500)
                if fake.mode == "crash" and body.get("tools"):
                    return self._json({"choices": "not-a-list-of-dicts"})
                if fake.mode == "busy_later" and n >= 3:
                    fake.busy = True
                if fake.mode == "pool_busy_later" and n >= 3:
                    fake.pool_leases = [{"lease_id": "h1", "role": "agent-gpu2", "status": "granted"}]
                text = json.dumps([body["messages"], body.get("tools")])
                toks = _tok(text)
                with fake.lock:
                    lcp = 0
                    for other in fake.seen:
                        k = 0
                        for a, b in zip(toks, other):
                            if a != b:
                                break
                            k += 1
                        lcp = max(lcp, k)
                    fake.seen.append(toks)
                own = CANARY.findall(json.dumps(body["messages"]))[0]
                reply = own
                if fake.mode == "leak" and fake.previous:
                    reply = f"{own}\n{fake.previous}"
                if fake.mode == "mute":
                    reply = "I cannot find any codeword."
                fake.previous = own
                prompt_n = 100 if fake.mode == "short" else int(len(toks) * 1.3)
                cache_n = lcp
                if fake.mode == "cache":
                    cache_n = 4000
                if fake.mode == "stale" and lcp > 100:   # only the shared-prefix pair: a stale tail
                    cache_n = lcp + 50
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
    args = ["--url", fake_server.url, "--pool-url", fake_server.url, "--out", str(tmp_path / "out"),
            "--rounds", "1", "--pace-sec", "0",
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
    assert len(lines) == summary["requests"] == summary["expected"] == 15
    # The shared-prefix pair is checked against its real common prefix, not skipped.
    recs = [json.loads(l) for l in lines]
    sp = [r for r in recs if r["phase"] == "shared_prefix"]
    assert sp[1]["true_lcp"] > 1000 and sp[1]["cache_n"] == sp[1]["true_lcp"]
    # Evidence is owner-only: a real leak would put private text in it.
    assert (tmp_path / "out").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "out" / "requests.jsonl").stat().st_mode & 0o777 == 0o600


def test_a_foreign_canary_is_a_leak(fake, tmp_path):
    code, summary = _run(fake(mode="leak"), tmp_path)
    assert (code, summary["verdict"]) == (1, "LEAK")
    assert summary["leaks"] and summary["leaks"][0]["foreign_convs"]


def test_cache_reuse_past_the_true_prefix_on_unrelated_prompts_is_a_leak(fake, tmp_path):
    code, summary = _run(fake(mode="cache"), tmp_path)
    assert (code, summary["verdict"]) == (1, "LEAK")
    assert summary["cache_over_lcp"]


def test_a_stale_tail_on_the_shared_prefix_pair_is_a_leak_even_unechoed(fake, tmp_path):
    code, summary = _run(fake(mode="stale"), tmp_path)
    assert (code, summary["verdict"]) == (1, "LEAK") and not summary["leaks"]
    assert {r["phase"] for r in summary["cache_over_lcp"]} == {"shared_prefix"}


@pytest.mark.parametrize("kw,why", [
    ({"model": "/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"}, "does not contain"),
    ({"slots": 1}, "total_slots=1"),
    ({"busy": True}, "slot is processing"),
    # A durable run between tool steps: every slot idle, but the pool says the seat is held.
    ({"pool_leases": [{"lease_id": "h1", "role": "agent-gpu2", "status": "granted"}]}, "pool leases active"),
])
def test_refuses_the_wrong_worker_or_a_busy_one(fake, tmp_path, kw, why):
    server = fake(**kw)
    code, summary = _run(server, tmp_path)
    assert (code, summary["verdict"]) == (2, "REFUSED") and why in summary["why"]
    assert server.bodies == []   # not one request against real work or the wrong model


@pytest.mark.parametrize("mode,why", [
    ("short", "under 4500 tokens"), ("mute", "own codeword"), ("error500", "failed"), ("crash", "failed"),
    ("busy_later", "real work overlapped"), ("pool_busy_later", "real work overlapped"),
    ("no_tokenize", "common prefix"),
])
def test_a_run_that_could_not_see_a_leak_is_inconclusive_not_pass(fake, tmp_path, mode, why):
    code, summary = _run(fake(mode=mode), tmp_path)
    assert (code, summary["verdict"]) == (3, "INCONCLUSIVE") and why in summary["why"]


def test_an_unreadable_pool_is_refused(fake, tmp_path):
    server = fake()
    mod = _load()
    code = mod.main(["--url", server.url, "--pool-url", "http://127.0.0.1:9", "--out", str(tmp_path / "o"),
                     "--idle-wait-sec", "0"])
    assert code == 2 and server.bodies == []


def test_an_unreachable_worker_is_refused(tmp_path):
    mod = _load()
    code = mod.main(["--url", "http://127.0.0.1:9", "--out", str(tmp_path / "o"), "--idle-wait-sec", "0"])
    assert code == 2
