"""Harder reasoning at depth: decoys, a mid-text reassignment, 4-hop chains, unit conversion, date math.
Facts are spread across the context at fixed fractions; each depth gets fresh random values.

usage: reason_hard.py URL LABEL [thinking_kwargs_json]
"""
import json, random, re, sys
sys.path.insert(0, "/tmp/bonsai-test")
import bonsai_bench as b

URL, LABEL = sys.argv[1], sys.argv[2]
KW = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"reasoning_effort": "medium"}
b.URL = URL
b.OUT = open(f"/tmp/bonsai-test/hard-raw-{LABEL}.jsonl", "w")
out = open(f"/tmp/bonsai-test/hard-{LABEL}.jsonl", "w")
CPT = 3.5
DEPTHS = [8_000, 24_000, 40_000, 50_000]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CITIES = ["Marrow", "Quillon", "Ashgrove", "Tessel", "Varn", "Oakhollow", "Brisk", "Calder"]
NAMES = ["Tessaly Brin", "Oswin Hale", "Mirela Cask", "Jory Vend", "Anik Solde", "Petra Lume", "Corin Ash", "Ilse Marr"]
corpus = b.corpus(260_000)


def build(depth, rng):
    cities = rng.sample(CITIES, 3)
    keepers = rng.sample(NAMES, 4)             # 3 original keepers + 1 replacement
    vaults = [f"K{v}" for v in rng.sample(range(10, 99), 3)]
    badges = rng.sample(range(1000, 9999), 4)
    target = rng.randrange(3)                  # which vault the chain question asks about
    lbs = rng.randint(1000, 3000)              # shipment A in pounds
    b_kg = rng.randint(300, 900)
    c_less = rng.randint(50, 250)
    day0 = rng.randrange(7)
    s1, s2 = rng.randint(2, 4), rng.randint(1, 3)
    facts = {
        0.05: [f"Vault {vaults[i]} is kept in the city of {cities[i]}." for i in range(3)],
        0.20: [f"The keeper of every vault in {cities[i]} is {keepers[i]}." for i in range(3)]
              + [f"Shipment A weighed {lbs} pounds."],
        0.40: [f"{keepers[i]}'s badge number is {badges[i]}." for i in range(3)]
              + [f"The quarterly review is scheduled for {DAYS[day0]}."],
        0.60: [f"Shipment B weighed {b_kg} kg.", f"Update: the review moved {s1} days later than scheduled."],
        0.75: [f"As of this quarter, {keepers[3]} replaced {keepers[target]} as keeper of every vault in {cities[target]}.",
               f"{keepers[3]}'s badge number is {badges[3]}."],
        0.90: [f"Shipment C weighed {c_less} kg less than shipment B.",
               f"Second update: the review moved a further {s2} days later.",
               "For all conversions, use 1 pound = 0.4536 kg."],
    }
    n = int(depth * CPT)
    text = corpus[:n]
    pieces, prev = [], 0
    for frac in sorted(facts):
        cut = text.rfind("\n", 0, int(len(text) * frac)) + 1 or int(len(text) * frac)
        pieces += [text[prev:cut], "\n" + " ".join(facts[frac]) + "\n"]
        prev = cut
    doc = "".join(pieces) + text[prev:]
    total_kg = round(lbs * 0.4536 + b_kg + (b_kg - c_less))
    other = (target + 1) % 3
    qs = [
        ("chain4", f"What is the badge number of the current keeper of vault {vaults[target]}?", str(badges[3])),
        ("decoy", f"What is the badge number of the current keeper of vault {vaults[other]}?", str(badges[other])),
        ("arith", "What is the combined weight of shipments A, B and C in kg, rounded to the nearest whole kg?",
         str(total_kg)),
        ("date", "On which day of the week is the quarterly review now held?", DAYS[(day0 + s1 + s2) % 7]),
        ("who", f"Who kept the vaults in {cities[target]} before the most recent change?", keepers[target]),
    ]
    return doc, qs


def matches(want, got):
    g = got.lower().replace(",", "").strip()
    if want.isdigit():
        nums = re.findall(r"\d+", g)
        return bool(nums) and abs(int(nums[0]) - int(want)) <= (1 if len(want) > 3 else 0)
    return want.lower() in g


rng = random.Random(4242)
score = {}
for depth in DEPTHS:
    doc, qs = build(depth, rng)
    for tag, q, want in qs:
        msgs = [{"role": "user", "content": doc + "\n\nUsing only the notes above, answer: " + q
                 + "\nEnd your reply with a line of the form 'ANSWER: <value>'."}]
        try:
            rec, m = b.chat(msgs, 12000, KW, f"{LABEL}-{depth}-{tag}")
        except Exception as e:
            print(f"{depth} {tag} ERROR {e}", flush=True); continue
        found = re.findall(r"ANSWER:\s*(.+)", m.get("content") or "")
        got = found[-1].strip().rstrip(".") if found else ""
        ok = matches(want, got)
        tm = rec["timings"]
        score.setdefault(depth, []).append(ok)
        row = {"depth_target": depth, "tag": tag, "want": want, "got": got, "ok": ok, "finish": rec["finish"],
               "prompt_tokens": tm.get("cache_n", 0) + tm.get("prompt_n", 0), "gen": tm.get("predicted_n"),
               "tg": tm.get("predicted_per_second"), "reasoning_chars": rec["reasoning_chars"], "wall": rec["wall_s"]}
        out.write(json.dumps(row) + "\n"); out.flush()
        print(f"{LABEL} depth~{row['prompt_tokens']} {tag:<6} want={want:<14} got={got[:24]!r:<26} ok={ok} "
              f"finish={rec['finish']} gen={tm.get('predicted_n')}", flush=True)
tot = sum(sum(v) for v in score.values()); n = sum(len(v) for v in score.values())
print(f"{LABEL} HARDSCORE {tot}/{n} by depth: " + ", ".join(f"{d}:{sum(v)}/{len(v)}" for d, v in score.items()), flush=True)
