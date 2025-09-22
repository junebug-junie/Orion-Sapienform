from __future__ import annotations
import os, time, json, typing as T
import httpx

class OrionLLMClient:
    """
    Minimal client for the Orion Brain Service (Ollama router).
    - generate(prompt, options, stream=False, system=None) -> str | iterator[str]
    - chat(messages, options, stream=False) -> str | iterator[str]
    """
    def __init__(self,
                 base_url: str | None = None,
                 model: str | None = None,
                 connect_timeout: float | None = None,
                 read_timeout: float | None = None,
                 retries: int = 2,
                 backoff: float = 0.25,
                 headers: dict | None = None):
        self.base_url = (base_url or os.getenv("ORION_BRAIN_URL") or "http://localhost:8088").rstrip("/")
        self.model = model or os.getenv("ORION_MODEL") or "mistral:instruct"
        self.retries = retries
        self.backoff = backoff
        ct = float(os.getenv("ORION_CONNECT_TIMEOUT", connect_timeout or 10))
        rt = float(os.getenv("ORION_READ_TIMEOUT", read_timeout or 600))
        self.timeout = httpx.Timeout(connect=ct, read=rt, write=rt, pool=ct)
        self.headers = headers or {"content-type":"application/json"}
        self._client = httpx.Client(timeout=self.timeout, headers=self.headers)

    def close(self):
        self._client.close()

    # ---------- public API ----------
    def generate(self, prompt: str, options: dict | None = None, stream: bool = False,
                 system: str | None = None, mistral_inst_wrap: bool = True):
        """Call /generate. If system is provided and mistral_inst_wrap=True, wrap with [INST] template."""
        if system and mistral_inst_wrap:
            prompt = f"<s>[INST] {system}\n\n{prompt} [/INST]"
        payload = {"model": self.model, "prompt": prompt, "options": options or {}, "stream": bool(stream)}
        if stream:
            return self._stream("/generate", payload)
        return self._json("/generate", payload).get("response","")

    def chat(self, messages: list[dict], options: dict | None = None, stream: bool = False):
        """Call /chat with an array of {role,content} messages."""
        payload = {"model": self.model, "messages": messages, "options": options or {}, "stream": bool(stream)}
        if stream:
            return self._stream("/chat", payload)
        data = self._json("/chat", payload)
        # Some Ollama builds return {"message":{"role":"assistant","content":"..."}}
        if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content","")
        return data.get("response","")

    # ---------- internals ----------
    def _json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(url, json=payload)
                if 200 <= r.status_code < 300:
                    return r.json()
                if 500 <= r.status_code < 600:
                    raise RuntimeError(f"Server error {r.status_code}: {r.text[:200]}")
                # 4xx: raise with details
                r.raise_for_status()
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise
        raise last_err  # pragma: no cover

    def _stream(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        def iter_lines():
            last_err = None
            for attempt in range(self.retries + 1):
                try:
                    with self._client.stream("POST", url, json=payload) as resp:
                        resp.raise_for_status()
                        for raw in resp.iter_lines():
                            if not raw:
                                continue
                            try:
                                obj = json.loads(raw)
                                # Ollama NDJSON lines include "response" tokens
                                if "response" in obj and obj.get("done") is not True:
                                    yield obj["response"]
                            except Exception:
                                # if it's not JSON, yield raw
                                try:
                                    yield raw.decode("utf-8", "ignore")
                                except Exception:
                                    yield str(raw)
                        return
                except Exception as e:
                    last_err = e
                    if attempt < self.retries:
                        time.sleep(self.backoff * (2 ** attempt))
                        continue
                    raise
        return iter_lines()
