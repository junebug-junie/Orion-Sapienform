# app/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os, asyncio, time, uuid, json, itertools
import httpx
from typing import List, Dict, Optional, Tuple, AsyncIterator
from app.redis_bus import publish_event, publish_bus_out
1
# ---- config ----
BACKENDS = [b.strip() for b in os.getenv("BACKENDS","").split(",") if b.strip()]
SELECTION_POLICY = os.getenv("SELECTION_POLICY","least_conn")
HEALTH_INTERVAL = int(os.getenv("HEALTH_INTERVAL_SEC","5"))
CONNECT_TIMEOUT = int(os.getenv("CONNECT_TIMEOUT_SEC","10"))
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT_SEC","600"))
PORT = int(os.getenv("PORT","8088"))

# ---- helpers (safe bus emits) ----
async def _emit_event(kind: str, fields: Dict):
    try:
        await publish_event(kind, fields)
    except Exception:
        pass

async def _emit_bus(topic: str, content: Dict):
    try:
        await publish_bus_out(topic, content)
    except Exception:
        pass

# ---- state ----
class Backend:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.healthy: bool = False
        self.last_latency_ms: float = 0.0
        self.inflight: int = 0
        self.last_error: Optional[str] = None

class BrainRouter:
    def __init__(self, urls: List[str]):
        self.backends: Dict[str, Backend] = {u: Backend(u) for u in urls}
        self._rr = itertools.cycle(list(self.backends.keys())) if self.backends else None
        self._lock = asyncio.Lock()

    def list(self) -> List[Dict]:
        return [{
            "url": b.url, "healthy": b.healthy,
            "latency_ms": b.last_latency_ms, "inflight": b.inflight,
            "last_error": b.last_error
        } for b in self.backends.values()]

    async def register(self, url: str):
        async with self._lock:
            if url not in self.backends:
                self.backends[url] = Backend(url)
                self._rr = itertools.cycle(list(self.backends.keys()))
        return True

    async def deregister(self, url: str):
        async with self._lock:
            self.backends.pop(url, None)
            self._rr = itertools.cycle(list(self.backends.keys())) if self.backends else None
        return True

    def pick(self) -> Optional[Backend]:
        healthy = [b for b in self.backends.values() if b.healthy]
        if not healthy:
            return None
        if SELECTION_POLICY == "round_robin":
            for _ in range(len(healthy)*2):
                key = next(self._rr)
                b = self.backends.get(key)
                if b and b.healthy:
                    return b
            return healthy[0]
        # default least_conn; tie-break by latency
        return sorted(healthy, key=lambda b: (b.inflight, b.last_latency_ms))[0]

router = BrainRouter(BACKENDS)
app = FastAPI(title="Orion Brain Service")

# ---- models ----
class GenerateBody(BaseModel):
    model: str
    prompt: str
    options: Optional[dict] = None
    stream: Optional[bool] = False
    # optional extras:
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatBody(BaseModel):
    model: str
    messages: List[Dict]
    options: Optional[dict] = None
    stream: Optional[bool] = False
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

# ---- health checking task ----
async def probe_backend(b: Backend):
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=10)) as client:
            r = await client.get(f"{b.url}/api/tags")
            b.healthy = r.status_code == 200
            b.last_error = None if b.healthy else f"HTTP {r.status_code}"
    except Exception as e:
        b.healthy = False
        b.last_error = str(e)[:200]
    finally:
        b.last_latency_ms = (time.perf_counter() - t0) * 1000.0

async def health_loop():
    while True:
        await asyncio.gather(*(probe_backend(b) for b in router.backends.values()))
        await asyncio.sleep(HEALTH_INTERVAL)

@app.on_event("startup")
async def startup():
    await asyncio.gather(*(probe_backend(b) for b in router.backends.values()))
    asyncio.create_task(health_loop())

# ---- utility: forwarders ----
async def _forward_stream(path: str, payload: dict, trace_id: str) -> AsyncIterator[bytes]:
    b = router.pick()
    if not b:
        yield (json.dumps({"error":"No healthy backends"}) + "\n").encode("utf-8")
        return
    await _emit_event("route.selected", {"trace_id": trace_id, "backend": b.url, "policy": SELECTION_POLICY})
    b.inflight += 1
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            async with client.stream("POST", f"{b.url}{path}", json=payload) as resp:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
    finally:
        b.inflight -= 1

async def _forward_json(path: str, payload: dict, trace_id: str) -> Tuple[int, Dict, str]:
    """
    Returns (status_code, data_dict, backend_url). Never returns JSONResponse.
    """
    b = router.pick()
    if not b:
        raise HTTPException(503, "No healthy backends")
    await _emit_event("route.selected", {"trace_id": trace_id, "backend": b.url, "policy": SELECTION_POLICY})
    b.inflight += 1
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = await client.post(f"{b.url}{path}", json=payload)
            status = r.status_code
            try:
                data = r.json()
            except Exception:
                txt = (r.text or "")[:500]
                data = {"error": "backend_invalid_json", "body": txt}
            return status, data, b.url
    finally:
        b.inflight -= 1

# ---- endpoints ----
@app.get("/health")
async def health():
    return {"ok": True, "backends": router.list()}

@app.get("/stats")
async def stats():
    return {"policy": SELECTION_POLICY, "backends": router.list()}

@app.get("/backends")
async def backends_get():
    return {"backends": router.list()}

@app.post("/backends/register")
async def backends_register(body: Dict):
    url = (body or {}).get("url")
    if not url:
        raise HTTPException(400, "Provide 'url'")
    await router.register(url)
    return {"ok": True}

@app.post("/backends/deregister")
async def backends_deregister(body: Dict):
    url = (body or {}).get("url")
    if not url:
        raise HTTPException(400, "Provide 'url'")
    await router.deregister(url)
    return {"ok": True}

@app.get("/models")
async def models():
    results = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=10)) as client:
        for b in router.backends.values():
            if not b.healthy: 
                continue
            try:
                r = await client.get(f"{b.url}/api/tags")
                if r.status_code == 200:
                    results.append({"backend": b.url, "models": r.json()})
            except Exception:
                pass
    return {"results": results}

@app.post("/generate")
async def generate(body: GenerateBody, request: Request):
    payload     = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    trace_id    = payload.get("trace_id") or str(uuid.uuid4())
    user_id     = payload.get("user_id")
    session_id  = payload.get("session_id")
    model_name  = payload.get("model")
    return_json = bool(payload.get("return_json", False))
    t0 = time.time()

    await _emit_event("request.received", {
        "trace_id": trace_id, "endpoint": "/generate",
        "user_id": user_id, "session_id": session_id, "model": model_name
    })

    try:
        if body.stream:
            async def wrapped_stream():
                idx = 0
                async for chunk in _forward_stream("/api/generate", payload, trace_id):
                    if not return_json:
                        yield chunk
                        continue
                    line = (chunk.decode("utf-8","ignore")).strip()
                    if not line: 
                        continue
                    # pass-through NDJSON if it's already JSON; otherwise wrap
                    try:
                        json.loads(line)
                        yield (line + "\n").encode("utf-8")
                    except json.JSONDecodeError:
                        yield (json.dumps({"id": trace_id, "delta": line, "index": idx}) + "\n").encode("utf-8")
                        idx += 1
                lat_ms = int((time.time() - t0) * 1000)
                await _emit_bus("llm.response", {
                    "trace_id": trace_id, "user_id": user_id, "session_id": session_id,
                    "model": model_name, "latency_ms": lat_ms
                })
                await _emit_event("request.completed", {"trace_id": trace_id, "latency_ms": lat_ms, "status": "ok"})
                if return_json:
                    yield (json.dumps({"event":"end","id":trace_id,"latency_ms":lat_ms}) + "\n").encode("utf-8")

            return StreamingResponse(wrapped_stream(), media_type="application/x-ndjson")

        # ---- non-stream path ----
        status, data, backend = await _forward_json("/api/generate", payload, trace_id)
        lat_ms = int((time.time() - t0) * 1000)
        final_text = data.get("response") or data.get("output") or data.get("text") or ""
        usage = data.get("usage") or {}

        await _emit_bus("llm.response", {
            "trace_id": trace_id, "user_id": user_id, "session_id": session_id,
            "model": model_name, "latency_ms": lat_ms, "usage": usage, "text": final_text
        })
        await _emit_event("request.completed", {
            "trace_id": trace_id, "latency_ms": lat_ms, "status": "ok", "usage": usage
        })

        if return_json:
            return {
                "id": trace_id, "model": model_name, "created": int(time.time()),
                "latency_ms": lat_ms, "usage": usage, "response": final_text,
                "meta": {"trace_id": trace_id, "backend": backend}
            }
        # return backend JSON with original status
        return JSONResponse(status_code=status, content=data)

    except Exception as e:
        await _emit_event("request.error", {"trace_id": trace_id, "error": str(e)[:500]})
        return JSONResponse(status_code=500, content={"error":"internal_error","trace_id":trace_id})

@app.post("/chat")
async def chat(body: ChatBody, request: Request):
    payload     = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    trace_id    = payload.get("trace_id") or str(uuid.uuid4())
    user_id     = payload.get("user_id")
    session_id  = payload.get("session_id")
    model_name  = payload.get("model")
    return_json = bool(payload.get("return_json", False))
    t0 = time.time()

    await _emit_event("request.received", {
        "trace_id": trace_id, "endpoint": "/chat",
        "user_id": user_id, "session_id": session_id, "model": model_name
    })

    try:
        if body.stream:
            async def wrapped_stream():
                idx = 0
                async for chunk in _forward_stream("/api/chat", payload, trace_id):
                    if not return_json:
                        yield chunk
                        continue
                    line = (chunk.decode("utf-8","ignore")).strip()
                    if not line: 
                        continue
                    try:
                        json.loads(line)
                        yield (line + "\n").encode("utf-8")
                    except json.JSONDecodeError:
                        yield (json.dumps({"id": trace_id, "delta": line, "index": idx}) + "\n").encode("utf-8")
                        idx += 1
                lat_ms = int((time.time() - t0) * 1000)
                await _emit_bus("llm.response", {
                    "trace_id": trace_id, "user_id": user_id, "session_id": session_id,
                    "model": model_name, "latency_ms": lat_ms
                })
                await _emit_event("request.completed", {"trace_id": trace_id, "latency_ms": lat_ms, "status": "ok"})
                if return_json:
                    yield (json.dumps({"event":"end","id":trace_id,"latency_ms":lat_ms}) + "\n").encode("utf-8")

            return StreamingResponse(wrapped_stream(), media_type="application/x-ndjson")

        status, data, backend = await _forward_json("/api/chat", payload, trace_id)
        lat_ms = int((time.time() - t0) * 1000)
        await _emit_bus("llm.response", {
            "trace_id": trace_id, "user_id": user_id, "session_id": session_id,
            "model": model_name, "latency_ms": lat_ms, "text": (data.get("response") or "")
        })
        await _emit_event("request.completed", {"trace_id": trace_id, "latency_ms": lat_ms, "status": "ok"})

        if return_json:
            return {
                "id": trace_id, "model": model_name, "created": int(time.time()),
                "latency_ms": lat_ms, "response": data.get("response") or "",
                "meta": {"trace_id": trace_id, "backend": backend}
            }
        return JSONResponse(status_code=status, content=data)

    except Exception as e:
        await _emit_event("request.error", {"trace_id": trace_id, "error": str(e)[:500]})
        return JSONResponse(status_code=500, content={"error":"internal_error","trace_id":trace_id})
