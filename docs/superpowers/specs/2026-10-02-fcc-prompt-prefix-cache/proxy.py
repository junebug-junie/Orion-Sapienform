"""Recording Anthropic proxy: claude -> here -> llama-server /v1/messages.
Applies the gateway's only body transform (system-message hoist) and records
raw + forwarded bodies and the raw SSE response."""
import json, os, sys, threading, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
UP = os.environ.get("UPSTREAM", "http://100.112.254.99:8015")
OUT = os.environ.get("OUT", "/tmp/fccprobe/rec")
os.makedirs(OUT, exist_ok=True)
lock = threading.Lock(); n = [0]

def normalize(body):  # copy of services/orion-llm-gateway/app/anthropic_passthrough.py
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        return dict(body)
    def blocks(c):
        if isinstance(c, str): return [{"type": "text", "text": c}] if c else []
        if c is None: return []
        return list(c)
    system = blocks(body.get("system")); conv = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            add = blocks(m.get("content"))
            if system and add: system.append({"type": "text", "text": "\n\n"})
            system.extend(add)
        else: conv.append(m)
    return {**body, "system": system, "messages": conv}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, data, ctype="application/json"):
        self.send_response(code); self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_HEAD(self): self._send(200, b"")
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, json.dumps({"data": [{"id": "llamacpp/agent", "type": "model", "display_name": "agent"}], "has_more": False}).encode())
        self._send(200, b"{}")
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        body = json.loads(raw or b"{}")
        path = self.path.split("?")[0]
        with lock:
            n[0] += 1; i = n[0]
        tag = "count" if "count_tokens" in path else "msg"
        fwd = normalize(body) if tag == "msg" else body
        open(f"{OUT}/{i:03d}_{tag}_in.json", "w").write(json.dumps(body))
        open(f"{OUT}/{i:03d}_{tag}_fwd.json", "w").write(json.dumps(fwd))
        req = urllib.request.Request(UP + path, data=json.dumps(fwd).encode(), method="POST",
                                     headers={"content-type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=900)
            code = resp.status
        except urllib.error.HTTPError as e:
            resp = e; code = e.code
        self.send_response(code)
        self.send_header("content-type", resp.headers.get("content-type", "application/json"))
        self.end_headers()
        with open(f"{OUT}/{i:03d}_{tag}_resp.txt", "wb") as f:
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk: break
                f.write(chunk); self.wfile.write(chunk); self.wfile.flush()
        self.close_connection = True

ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PORT", "18999"))), H).serve_forever()
