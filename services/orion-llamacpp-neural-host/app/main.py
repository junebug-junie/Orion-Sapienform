import logging
import os
import contextlib
import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from app.settings import settings
from app.profiles import LLMProfile, LlamaCppConfig

try:
    from orion.core.bus.bus_service_chassis import ChassisConfig, BaseChassis
except ImportError:
    # Fallback if orion lib not available (should be available in docker)
    ChassisConfig = None
    BaseChassis = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-host")

# ----------------------------------------------------------------------
# Bus / Chassis
# ----------------------------------------------------------------------

class NeuralHostChassis(BaseChassis):
    async def _run(self) -> None:
        # We don't have a specific consumption loop, just keep alive
        while not self._stop.is_set():
            await asyncio.sleep(1.0)

# ----------------------------------------------------------------------
# Helpers (Model Resolution)
# ----------------------------------------------------------------------

def _ensure_model_file(model_path: str, dl: Optional[LlamaCppConfig]) -> None:
    p = Path(model_path)
    if p.exists():
        return

    if dl is None or not dl.repo_id or not dl.filename:
        # If model_path is absolute and no dl info, we can't do anything
        raise FileNotFoundError(f"Model not found and no download spec available: {model_path}")

    Path(dl.model_root).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s/%s -> %s", dl.repo_id, dl.filename, dl.model_root)
    hf_hub_download(
        repo_id=dl.repo_id,
        filename=dl.filename,
        local_dir=dl.model_root,
        local_dir_use_symlinks=False,
        token=settings.hf_token,
    )

    if not p.exists():
        raise FileNotFoundError(f"Download completed but model still missing at: {model_path}")

def _resolve_runtime(profile: LLMProfile) -> Tuple[str, LlamaCppConfig]:
    if profile.llamacpp is None:
        raise RuntimeError(
            f"Profile '{profile.name}' backend=llamacpp requires a 'llamacpp:' block"
        )

    cfg = profile.llamacpp

    # Apply overrides
    if settings.llamacpp_ctx_size_override is not None:
        cfg.ctx_size = settings.llamacpp_ctx_size_override
    if settings.llamacpp_n_gpu_layers_override is not None:
        cfg.n_gpu_layers = settings.llamacpp_n_gpu_layers_override
    # We ignore port/host overrides here as they are for the server binding,
    # but we bind uvicorn based on docker compose usually.
    # However, cfg.n_gpu_layers and ctx_size are critical for Llama() init.

    # Model path
    if settings.llamacpp_model_path_override:
        model_path = settings.llamacpp_model_path_override
    else:
        if cfg.filename:
            model_path = str(Path(cfg.model_root) / cfg.filename)
        elif profile.model_id.endswith(".gguf") and profile.model_id.startswith("/"):
            model_path = profile.model_id
        else:
             raise RuntimeError(f"Profile '{profile.name}' missing filename or direct path")

    return model_path, cfg

# ----------------------------------------------------------------------
# State & Lifespan
# ----------------------------------------------------------------------

class AppState:
    llm: Optional[Llama] = None
    chassis: Optional[NeuralHostChassis] = None

state = AppState()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Profile
    logger.info(f"Resolving profile: {settings.llm_profile_name}")
    try:
        profile = settings.resolve_profile()
    except Exception as e:
        logger.error(f"Failed to resolve profile: {e}")
        raise e

    model_path, cfg = _resolve_runtime(profile)
    logger.info(f"Selected model path: {model_path}")

    # 2. Download/Ensure Model
    _ensure_model_file(model_path, cfg)

    # 3. Initialize Llama
    # "Native Neural Extraction" requires embedding=True
    logger.info(f"Loading Llama model... n_gpu_layers={cfg.n_gpu_layers}, ctx={cfg.ctx_size}")
    try:
        state.llm = Llama(
            model_path=model_path,
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=cfg.ctx_size,
            embedding=True,
            verbose=True
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    # 4. Start Bus Heartbeat
    if BaseChassis:
        logger.info("Starting Bus Chassis...")
        chassis_cfg = ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            instance_id=settings.instance_id,
            bus_url=settings.bus_url,
        )
        state.chassis = NeuralHostChassis(chassis_cfg)
        await state.chassis.start_background()
    else:
        logger.warning("Orion Bus lib not found; skipping heartbeats.")

    yield

    # Cleanup
    if state.chassis:
        await state.chassis.stop()

app = FastAPI(title="Orion Neural Host", lifespan=lifespan)

# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: float = 0.8
    max_tokens: int = 128
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    if not state.llm:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        # Neural Projection requires the full response to generate the embedding.
        raise HTTPException(status_code=400, detail="Streaming is not supported by Neural Host.")

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    # 1. Call create_chat_completion
    response = state.llm.create_chat_completion(
        messages=messages_dicts,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=request.stop,
        stream=False
    )

    # 2. Extract content
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""

    # 3. Introspection: Embed the content
    spark_vector = []
    if content:
        try:
            # create_embedding returns {'object': 'list', 'data': [{'embedding': [...]}], ...}
            embed_resp = state.llm.create_embedding(content)
            if embed_resp and "data" in embed_resp and len(embed_resp["data"]) > 0:
                spark_vector = embed_resp["data"][0]["embedding"]
        except Exception as e:
             logger.error(f"Embedding failed: {e}")
             spark_vector = []

    # 4. Injection
    response["spark_vector"] = spark_vector

    return response

@app.get("/health")
def health():
    if not state.llm:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "ok"}
