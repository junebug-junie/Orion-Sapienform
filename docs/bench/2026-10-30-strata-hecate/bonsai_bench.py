"""Bonsai 2 27B bake-off bench on circe. stdlib only. Writes JSONL + prints a summary."""
import glob, os, json, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8017/v1/chat/completions"
OUT = open(os.path.join(os.environ.get("BENCH_DIR", "/tmp/bonsai-test"), "results.jsonl"), "a")
# Servers without llama.cpp's `timings` block (Strata): stream, and derive prefill speed from time to
# first token and decode speed from the rest. cache_n stays 0, so prompt_n is the whole prompt.
STREAM = os.environ.get("BENCH_STREAM") == "1"
API_KEY = os.environ.get("BENCH_API_KEY", "")
MODEL = os.environ.get("BENCH_MODEL", "")


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _chat_stream(body, tag):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    if MODEL:
        body["model"] = MODEL
    t = time.time()
    first, content, reasoning, finish, usage = None, [], [], None, {}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=_headers())
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.time()
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = ch.get("finish_reason") or finish
    end = time.time()
    first = first or end
    pn, gn = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    tm = {"prompt_n": pn, "cache_n": 0, "predicted_n": gn, "ttft_s": round(first - t, 3),
          "prompt_per_second": pn / max(first - t, 1e-6), "predicted_per_second": max(gn - 1, 0) / max(end - first, 1e-6)}
    msg = {"content": "".join(content), "reasoning_content": "".join(reasoning)}
    rec = {"tag": tag, "wall_s": round(end - t, 2), "finish": finish, "timings": tm, "usage": usage,
           "reasoning_chars": len(msg["reasoning_content"]), "content": msg["content"][:1500], "tool_calls": None}
    OUT.write(json.dumps(rec) + "\n"); OUT.flush()
    return rec, msg


def chat(messages, max_tokens=4096, effort="medium", tag="", **extra):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
            "chat_template_kwargs": (effort if isinstance(effort, dict) else {"enable_thinking": False} if effort == "none" else {"reasoning_effort": effort}), "cache_prompt": True, **extra}
    kw = body["chat_template_kwargs"]
    body["reasoning_effort"] = "none" if kw.get("enable_thinking") is False else kw.get("reasoning_effort", "medium")
    if STREAM:
        return _chat_stream(body, tag)
    t = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=_headers())
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    wall = time.time() - t
    msg = d["choices"][0]["message"]
    tm = d.get("timings", {})
    rec = {"tag": tag, "wall_s": round(wall, 2), "finish": d["choices"][0].get("finish_reason"),
           "timings": tm, "usage": d.get("usage"),
           "reasoning_chars": len(msg.get("reasoning_content") or ""),
           "content": (msg.get("content") or "")[:1500], "tool_calls": msg.get("tool_calls")}
    OUT.write(json.dumps(rec) + "\n"); OUT.flush()
    return rec, msg


def line(rec):
    t = rec["timings"]
    return (f"{rec['tag']:<28} cache={t.get('cache_n')} prompt={t.get('prompt_n')} "
            f"pp={t.get('prompt_per_second', 0):.1f}t/s gen={t.get('predicted_n')} "
            f"tg={t.get('predicted_per_second', 0):.2f}t/s wall={rec['wall_s']}s finish={rec['finish']}")


def parallel(n, fn):
    res = [None] * n
    ths = [threading.Thread(target=lambda i=i: res.__setitem__(i, fn(i))) for i in range(n)]
    t = time.time()
    [th.start() for th in ths]; [th.join() for th in ths]
    return res, time.time() - t


def phase_single():
    print("## single")
    r, _ = chat([{"role": "user", "content": "Say hello in five words."}], 256, "none", "cold-hello")
    print(line(r)); print("  ->", r["content"][:200])
    r, _ = chat([{"role": "user", "content": "Tell me the weirdest thing you ever heard."}], 5000, "medium", "weirdest-medium")
    print(line(r), "reasoning_chars", r["reasoning_chars"]); print("  ->", r["content"][:400])


SANITY = [
    ("math", "What is the sum of all positive integers n < 100 such that n^2 + n + 41 is NOT prime? Give just the number at the end."),
    ("logic", "Alice is taller than Bob. Carol is shorter than Bob. Dave is taller than Alice. Who is the second shortest? One word answer at the end."),
    ("code", "Write a Python function `merge_intervals(xs)` that merges overlapping [start, end] intervals and returns them sorted. Code only."),
    ("orion", "In two sentences: what is the difference between a metric that varies and a metric that can return to rest?"),
]


def phase_sanity():
    print("## sanity (medium effort)")
    for tag, q in SANITY:
        r, _ = chat([{"role": "user", "content": q}], 8192, "medium", f"sanity-{tag}")
        print(line(r), "reasoning_chars", r["reasoning_chars"]); print("  ->", r["content"][-600:].replace("\n", " | "))
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the repo",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    r, _ = chat([{"role": "user", "content": "Open services/orion-llamacpp-bonsai-host/README.md and tell me which port it uses."}],
                4096, "medium", "sanity-toolcall", tools=tools)
    print(line(r)); print("  -> tool_calls:", json.dumps(r["tool_calls"])[:300])


def phase_concurrency():
    print("## concurrency (effort none, 512 forced tokens, ignore_eos)")
    for n in (1, 2, 4):
        res, wall = parallel(n, lambda i: chat([{"role": "user", "content": f"Write a long story about lighthouse keeper #{i}."}],
                                              512, "none", f"conc{n}-{i}", ignore_eos=True)[0])
        tg = [r["timings"]["predicted_per_second"] for r in res]
        agg = sum(r["timings"]["predicted_n"] for r in res) / wall
        print(f"n={n} per-run tg={[round(x, 2) for x in tg]} aggregate={agg:.2f} t/s wall={wall:.1f}s")


def corpus(chars):
    buf = []
    for p in sorted(glob.glob("/mnt/scripts/Orion-Sapienform/docs/*.md")):
        buf.append(open(p, errors="ignore").read())
        if sum(map(len, buf)) > chars:
            break
    return "".join(buf)[:chars]


def phase_longctx():
    print("## long context (~30K tokens, single)")
    doc = corpus(110_000)
    r, _ = chat([{"role": "user", "content": doc + "\n\nList three distinct services named in the text above. Short answer."}],
                2048, "none", "longctx-30k")
    print(line(r)); print("  ->", r["content"][:300].replace("\n", " | "))
    r, _ = chat([{"role": "user", "content": doc + "\n\nList three distinct services named in the text above. Short answer."}],
                2048, "none", "longctx-30k-repeat")
    print(line(r), "(repeat: expect cache hit)")


def agent_loop(i, turns=6):
    """Growing tool-loop conversation, like a curiosity run: each step resends the whole history."""
    doc = corpus(8000 * (i + 1))[-8000:]
    msgs = [{"role": "system", "content": "You are investigating a repository. Be brief."},
            {"role": "user", "content": f"Run {i}: summarise what matters in these notes, one step at a time.\n\n{doc}"}]
    out = []
    for t in range(turns):
        r, m = chat(msgs, 512, "none", f"agent{i}-t{t}")
        out.append(r)
        msgs.append({"role": "assistant", "content": m.get("content") or ""})
        msgs.append({"role": "user", "content": f"Tool result {t}:\n" + corpus(4000 * (t + 2))[-3000:] + "\nContinue."})
    return out


def phase_agent4():
    print("## 4 concurrent agent-style loops (6 steps each)")
    res, wall = parallel(4, agent_loop)
    for runs in res:
        for r in runs:
            print(line(r))
    total_prompt = sum(r["timings"]["prompt_n"] for rr in res for r in rr)
    total_cache = sum(r["timings"].get("cache_n", 0) for rr in res for r in rr)
    print(f"reprocessed prompt tokens={total_prompt} reused(cache)={total_cache} "
          f"reuse={total_cache / max(1, total_cache + total_prompt):.1%} wall={wall:.1f}s")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[f"phase_{name}"]()
