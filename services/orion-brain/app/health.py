# app/health.py
from fastapi import APIRouter
import subprocess, shutil, asyncio, json, httpx
from app.config import BACKENDS, CONNECT_TIMEOUT, READ_TIMEOUT

router = APIRouter()

# Pick the first backend as the default Ollama endpoint (fallback to localhost)
DEFAULT_BACKEND = BACKENDS[0] if BACKENDS else "http://llm-brain:11434"


async def check_gpu_runtime(backend_url: str = DEFAULT_BACKEND):
    """
    Return GPU availability — preferring backend (Ollama) GPUs over local nvidia-smi.
    """
    backend_err = None

    # Try remote backend (Ollama or other)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = await client.get(f"{backend_url}/api/tags")
            if r.status_code == 200:
                info = {"available": True, "backend": backend_url}
                # Try to get version info
                try:
                    r2 = await client.get(f"{backend_url}/api/version")
                    info["version"] = r2.json().get("version", "unknown")
                except Exception:
                    info["version"] = "unknown"
                return info
    except Exception as e:
        backend_err = str(e)

    # Fallback: local GPU check
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            gpus = []
            for line in out.splitlines():
                name, total, free = [s.strip() for s in line.split(",")]
                gpus.append({
                    "name": name,
                    "memory_total_gb": round(float(total) / 1024, 2),
                    "memory_free_gb": round(float(free) / 1024, 2),
                })
            return {"available": True, "local": True, "gpus": gpus}
        except Exception as e:
            return {"available": False, "reason": f"local GPU error: {e}"}

    return {"available": False, "reason": backend_err or "no GPU detected"}


async def check_ollama_backend(backend_url: str = DEFAULT_BACKEND):
    """Check if the LLM backend (Ollama) is alive and list models."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            version_url = f"{backend_url}/api/version"
            tags_url = f"{backend_url}/api/tags"

            version_r = await client.get(version_url)
            tags_r = await client.get(tags_url)

            return {
                "reachable": True,
                "version": version_r.json().get("version", "unknown") if version_r.status_code == 200 else "n/a",
                "models": tags_r.json() if tags_r.status_code == 200 else [],
                "backend_url": backend_url,
            }
    except Exception as e:
        return {"reachable": False, "error": str(e), "backend_url": backend_url}


@router.get("/health/gpu")
async def health_gpu():
    """Expose combined GPU + backend runtime health."""
    gpu = await check_gpu_runtime()
    ollama = await check_ollama_backend()
    return {"gpu": gpu, "ollama": ollama}


async def startup_checks():
    """Run initial health checks at startup."""
    gpu = await check_gpu_runtime()
    ollama = await check_ollama_backend()
    print(f"✅ Startup GPU check: {json.dumps(gpu, indent=2)}")
    print(f"✅ Startup Ollama check: {json.dumps(ollama, indent=2)}")
