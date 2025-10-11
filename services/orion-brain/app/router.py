import asyncio, itertools, time, httpx
from typing import Dict, List, Optional
from fastapi import HTTPException
from fastapi import APIRouter
from app.health import check_gpu_runtime, check_ollama_backend, wait_for_redis
from app.config import SELECTION_POLICY, CONNECT_TIMEOUT, READ_TIMEOUT, HEALTH_INTERVAL

router = APIRouter()

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
        return [
            {
                "url": b.url,
                "healthy": b.healthy,
                "latency_ms": b.last_latency_ms,
                "inflight": b.inflight,
                "last_error": b.last_error,
            }
            for b in self.backends.values()
        ]

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
            for _ in range(len(healthy) * 2):
                key = next(self._rr)
                b = self.backends.get(key)
                if b and b.healthy:
                    return b
            return healthy[0]
        return sorted(healthy, key=lambda b: (b.inflight, b.last_latency_ms))[0]

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

async def health_loop(router: BrainRouter):
    while True:
        await asyncio.gather(*(probe_backend(b) for b in router.backends.values()))
        await asyncio.sleep(HEALTH_INTERVAL)

@router.get("/health/gpu")
async def health_gpu():
    await wait_for_redis()
    gpu_status = await check_gpu_runtime()
    ollama_status = await check_ollama_backend()
    return {"gpu": gpu_status, "ollama": ollama_status}
