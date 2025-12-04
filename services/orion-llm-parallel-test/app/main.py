import httpx
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from settings import settings

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas-router")

app = FastAPI(title="Atlas Single-Node Controller")

class ChatRequest(BaseModel):
    prompt: str
    # We keep 'backend' optional so old scripts don't break, 
    # but we ignore it since we only have one engine now.
    backend: str | None = None 
    max_tokens: int = 512
    temperature: float = 0.7

@app.get("/")
def health_check():
    return {
        "status": "Atlas Online",
        "mode": "Single Engine (llama.cpp)",
        "target": settings.llamacpp_url
    }

@app.post("/chat")
async def chat_router(request: ChatRequest):
    # 1. Log the Request
    logger.info(f"Received request. Forwarding to Atlas Engine (llama.cpp)...")

    # 2. Build Payload (OpenAI Compatible)
    payload = {
        "model": settings.model_alias, # llama.cpp usually ignores this, but it's required by spec
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature
    }

    # 3. Forward Request inside the Docker Network
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(settings.llamacpp_url, json=payload)
            
            # Handle non-200 responses from the engine
            if resp.status_code != 200:
                logger.error(f"Engine Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Engine Error: {resp.text}")
            
            return resp.json()

    except httpx.ConnectError:
        logger.error("Connection Refused. Is llm-gpu-sync-test-llama-cpp running?")
        raise HTTPException(status_code=502, detail="Atlas Engine is unreachable. Check container logs.")
    except httpx.ReadTimeout:
        logger.error("Read Timeout. The model is too slow or stuck.")
        raise HTTPException(status_code=504, detail="Model timed out processing the request.")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=False
    )
