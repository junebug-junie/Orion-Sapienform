"""Context-depth sweep: one conversation grows to ~60K tokens. Per turn: prefill/decode speed,
recall of a fact planted in turn 0 (far) and one planted in the newest chunk (near)."""
import json, sys, time
sys.path.insert(0, "/tmp/bonsai-test")
import bonsai_bench as b

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8017/v1/chat/completions"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "bonsai"
EFFORT = sys.argv[3] if len(sys.argv) > 3 else "none"
b.URL = URL
b.OUT = open(f"/tmp/bonsai-test/depth-{LABEL}-{EFFORT}.jsonl", "w")

corpus = b.corpus(400_000)
CHUNK = 22_000  # chars, ~5.5K tokens
FAR = "The access code for the east archive is PELICAN-4471."
msgs = [{"role": "system", "content": "You are reading a long set of notes. Answer questions exactly and briefly."},
        {"role": "user", "content": FAR + "\n\n" + corpus[:CHUNK] + "\n\nAcknowledge in one word."}]
r, m = b.chat(msgs, 1024, EFFORT, f"{LABEL}-depth-t0")
msgs.append({"role": "assistant", "content": m.get("content") or ""})
for t in range(1, 12):
    near = f"The courier for batch {t} is named Oriel-{t * 37}."
    chunk = corpus[t * CHUNK:(t + 1) * CHUNK]
    msgs.append({"role": "user", "content": chunk + f"\n\n{near}\n\nQuestion: what is the access code for the east archive, "
                 f"and who is the courier for batch {t}? Answer as: CODE; NAME"})
    try:
        r, m = b.chat(msgs, 2048, EFFORT, f"{LABEL}-depth-t{t}")
    except Exception as e:  # context full ends the sweep
        print(f"t{t} stop: {e}"); break
    ans = (m.get("content") or "").strip()
    tm = r["timings"]
    depth = tm.get("cache_n", 0) + tm.get("prompt_n", 0)
    print(f"t{t} depth={depth} prompt_new={tm.get('prompt_n')} pp={tm.get('prompt_per_second', 0):.0f} "
          f"tg={tm.get('predicted_per_second', 0):.2f} far={'PELICAN-4471' in ans} near={f'Oriel-{t * 37}' in ans} "
          f"reasoning_chars={r['reasoning_chars']} ans={ans[:80]!r}", flush=True)
    msgs.append({"role": "assistant", "content": ans})
