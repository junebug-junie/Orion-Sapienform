"""Bonsai 2 27B bake-off bench on circe. stdlib only. Writes JSONL + prints a summary."""
import glob, json, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8017/v1/chat/completions"
OUT = open("/tmp/bonsai-test/results.jsonl", "a")


def chat(messages, max_tokens=4096, effort="medium", tag="", **extra):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
            "chat_template_kwargs": ({"enable_thinking": False} if effort == "none" else {"reasoning_effort": effort}), "cache_prompt": True, **extra}
    t = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
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
