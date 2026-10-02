import json, sys, time, urllib.request
UP = sys.argv[1]
words = [f"item{i} alpha beta gamma delta;" for i in range(800)]
base = " ".join(words)
def run(tag, prompt):
    while json.load(urllib.request.urlopen(UP + "/slots"))[0]["is_processing"]: time.sleep(2)
    r = json.load(urllib.request.urlopen(urllib.request.Request(UP + "/completion", data=json.dumps({"prompt": prompt, "n_predict": 4, "cache_prompt": True}).encode(), headers={"content-type": "application/json"})))
    t = r["timings"]; print(f"{tag:28s} cache_n={t.get('cache_n')} prompt_n={t['prompt_n']} prompt_ms={t['prompt_ms']:.0f}", flush=True)
run("cold", base)
run("identical", base)
run("append sentence", base + " And one more sentence here.")
mid = words[:]; mid[400] = "EDITED midpoint token;"
run("edit at 50%", " ".join(mid))
run("restore original", base)
late = words[:]; late[760] = "EDITED late token;"
run("edit at 95%", " ".join(late))
