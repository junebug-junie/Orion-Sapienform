"""Replay captured Claude Code requests against llama-server /v1/messages with
max_tokens=1, in different body-shaping variants, and report prompt tokens
actually processed per step (llama-server timings via /slots is not exposed,
so we read usage + wall time; llama logs are pulled separately)."""
import json, sys, time, urllib.request, copy
UP = "http://100.112.254.99:8015"
def hoist(body):  # current gateway behavior
    msgs = body["messages"]; system = list(body.get("system") or []); conv = []
    for m in msgs:
        if m.get("role") == "system":
            c = m["content"]; add = [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
            if system and add: system.append({"type": "text", "text": "\n\n"})
            system.extend(add)
        else: conv.append(m)
    return {**body, "system": system, "messages": conv}
def inplace(body):  # proposed: mid-conversation system -> user turn, in position
    conv = []
    for m in body["messages"]:
        if m.get("role") == "system":
            c = m["content"]; txt = c if isinstance(c, str) else "".join(b.get("text", "") for b in c)
            conv.append({"role": "user", "content": [{"type": "text", "text": "<system-reminder>\n" + txt + "\n</system-reminder>"}]})
        else: conv.append(m)
    return {**body, "messages": conv}
def idle():
    return not json.load(urllib.request.urlopen(UP + "/slots"))[0]["is_processing"]
if __name__ == "__main__":
    variant = sys.argv[1]; kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    for i in range(1, 5):
        b = json.load(open(f"rec/{i:03d}_msg_in.json"))
        b = hoist(b) if variant == "hoist" else inplace(b)
        b["stream"] = False; b["max_tokens"] = 1
        b.pop("thinking", None); b.pop("context_management", None); b.pop("output_config", None)
        if kwargs: b["chat_template_kwargs"] = kwargs
        while not idle(): time.sleep(2)
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(UP + "/v1/messages", data=json.dumps(b).encode(), headers={"content-type": "application/json"}), timeout=600))
        print(variant, kwargs, "step", i, "wall %.1fs" % (time.time() - t), "usage", r.get("usage"), flush=True)
