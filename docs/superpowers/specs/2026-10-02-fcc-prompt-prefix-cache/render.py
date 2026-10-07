"""Mirror llama.cpp b10398 server_chat_convert_anthropic_to_oai, then render
via the server's own /apply-template. Usage: render.py <port> <fwd.json>..."""
import json, sys, urllib.request
def conv(body):
    msgs = []
    s = body.get("system")
    if s is not None:
        if isinstance(s, str): sc = s
        else: sc = "".join(b.get("text", "") for b in s if b.get("type") == "text")
        msgs.append({"role": "system", "content": sc})
    for m in body["messages"]:
        c = m.get("content")
        if not isinstance(c, list): msgs.append(m); continue
        tc, cc, tr, rc = [], [], [], ""
        for b in c:
            t = b.get("type")
            if t == "text": cc.append(b)
            elif t == "thinking": rc += b.get("thinking", "")
            elif t == "tool_use": tc.append({"id": b.get("id",""), "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}), separators=(",", ":"))}})
            elif t == "tool_result":
                rcn = b.get("content")
                if isinstance(rcn, str): txt = rcn
                elif isinstance(rcn, list): txt = "".join(x.get("text", "") for x in rcn if x.get("type") == "text")
                else: txt = ""
                tr.append({"role": "tool", "tool_call_id": b.get("tool_use_id",""), "content": txt})
        if cc or tc or rc:
            nm = {"role": m["role"]}
            nm["content"] = cc if cc else ""
            if tc: nm["tool_calls"] = tc
            if rc: nm["reasoning_content"] = rc
            msgs.append(nm)
        msgs.extend(tr)
    out = {"messages": msgs}
    if body.get("tools"):
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description",""), "parameters": t.get("input_schema", {})}} for t in body["tools"]]
    if "chat_template_kwargs" in body: out["chat_template_kwargs"] = body["chat_template_kwargs"]
    return out
def render(port, oai):
    r = urllib.request.Request(f"http://100.112.254.99:{port}/apply-template", data=json.dumps(oai).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))["prompt"]
if __name__ == "__main__":
    port = sys.argv[1]
    for f in sys.argv[2:]:
        p = render(port, conv(json.load(open(f))))
        open(f + f".{port}.prompt.txt", "w").write(p)
        print(f, len(p))
