"""Reasoning at depth: multi-hop facts planted at 10% / 50% / 90% of a long filler context.
Each depth gets fresh random values, so answers cannot be memorised. Scored exactly.

usage: reason_depth.py URL LABEL [thinking_kwargs_json]
"""
import json, random, re, sys, time
sys.path.insert(0, "/tmp/bonsai-test")
import bonsai_bench as b

URL, LABEL = sys.argv[1], sys.argv[2]
KW = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"reasoning_effort": "medium"}
b.URL = URL
out = open(f"/tmp/bonsai-test/reason-{LABEL}.jsonl", "w")
CHARS_PER_TOKEN = 3.5
DEPTHS = [8_000, 24_000, 40_000, 50_000]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CITIES = ["Marrow", "Quillon", "Ashgrove", "Tessel", "Varn", "Oakhollow"]
NAMES = ["Tessaly Brin", "Oswin Hale", "Mirela Cask", "Jory Vend", "Anik Solde", "Petra Lume"]
corpus = b.corpus(260_000)


def build(depth, rng):
    vault = f"K{rng.randint(10, 99)}"
    city, keeper = rng.choice(CITIES), rng.choice(NAMES)
    badge = rng.randint(1000, 9999)
    a = rng.randint(800, 2000)
    delta = rng.randint(100, 600)
    day0 = rng.randrange(7)
    shift = rng.randint(2, 3)
    early = [f"Vault {vault} is kept in the city of {city}.",
             f"Shipment A weighed {a} kg.",
             f"The quarterly review is scheduled for {DAYS[day0]}."]
    mid = [f"The keeper of every vault in {city} is {keeper}."]
    late = [f"{keeper}'s badge number is {badge}.",
            f"Shipment B weighed {delta} kg more than shipment A.",
            f"Correction: the quarterly review moved {shift} days later than originally scheduled."]
    n = int(depth * CHARS_PER_TOKEN)
    text = corpus[:n]
    cut = lambda f: text.rfind("\n", 0, int(len(text) * f)) + 1 or int(len(text) * f)
    i10, i50, i90 = cut(0.10), cut(0.50), cut(0.90)
    doc = (text[:i10] + "\n" + " ".join(early) + "\n" + text[i10:i50] + "\n" + " ".join(mid) + "\n"
           + text[i50:i90] + "\n" + " ".join(late) + "\n" + text[i90:])
    qs = [("chain3", f"What is the badge number of the keeper of vault {vault}?", str(badge)),
          ("arith", "What is the combined weight of shipment A and shipment B, in kg?", str(a + a + delta)),
          ("update", "On which day of the week is the quarterly review now held?", DAYS[(day0 + shift) % 7])]
    return doc, qs


rng = random.Random(20260930)
score = {}
for depth in DEPTHS:
    doc, qs = build(depth, rng)
    for tag, q, want in qs:
        msgs = [{"role": "user", "content": doc + "\n\nUsing only the notes above, answer: " + q
                 + "\nEnd your reply with a line of the form 'ANSWER: <value>'."}]
        t = time.time()
        try:
            rec, m = b.chat(msgs, 12000, KW, f"{LABEL}-{depth}-{tag}")
        except Exception as e:
            print(f"{depth} {tag} ERROR {e}", flush=True); continue
        content = m.get("content") or ""
        got = re.findall(r"ANSWER:\s*(.+)", content)
        got = got[-1].strip().rstrip(".").replace(",", "") if got else ""
        ok = want.lower() in got.lower()
        tm = rec["timings"]
        score.setdefault(depth, []).append(ok)
        row = {"depth_target": depth, "tag": tag, "want": want, "got": got, "ok": ok, "finish": rec["finish"],
               "prompt_tokens": tm.get("cache_n", 0) + tm.get("prompt_n", 0), "gen": tm.get("predicted_n"),
               "tg": tm.get("predicted_per_second"), "reasoning_chars": rec["reasoning_chars"], "wall": rec["wall_s"]}
        out.write(json.dumps(row) + "\n"); out.flush()
        print(f"{LABEL} depth~{row['prompt_tokens']} {tag:<6} want={want:<10} got={got[:20]!r:<22} ok={ok} "
              f"finish={rec['finish']} gen={tm.get('predicted_n')} tg={tm.get('predicted_per_second', 0):.1f}", flush=True)
tot = sum(sum(v) for v in score.values()); n = sum(len(v) for v in score.values())
print(f"{LABEL} SCORE {tot}/{n} by depth: " + ", ".join(f"{d}:{sum(v)}/{len(v)}" for d, v in score.items()), flush=True)
