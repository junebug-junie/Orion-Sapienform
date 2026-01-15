import logging
import os
import contextlib
import asyncio
import uuid
from typing import List, Optional, Any, Union, Tuple
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from intention import IntentionForCausalLM

from app.settings import settings
from app.profiles import LLMProfile, LlamaColaConfig

try:
    from orion.core.bus.bus_service_chassis import ChassisConfig, BaseChassis
except ImportError:
    # Fallback if orion lib not available (should be available in docker)
    ChassisConfig = None
    BaseChassis = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llama-cola-host")

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

def _ensure_model_path(model_path: str, dl: Optional[LlamaColaConfig], repo_id: Optional[str]) -> None:
    if not settings.ensure_model_download:
        logger.info("Skipping model download check (ENSURE_MODEL_DOWNLOAD=false)")
        return

    p = Path(model_path)
    if p.exists():
        return

    if dl is None or not repo_id:
        raise FileNotFoundError(f"Model not found and no download spec available: {model_path}")

    Path(model_path).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s -> %s", repo_id, model_path)
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_path,
        local_dir_use_symlinks=False,
        revision=settings.llama_cola_revision_override or dl.revision,
        token=settings.hf_token,
    )

    if not p.exists():
        raise FileNotFoundError(f"Download completed but model still missing at: {model_path}")

def _resolve_runtime(profile: LLMProfile) -> Tuple[str, LlamaColaConfig, Optional[str]]:
    if profile.llama_cola is None:
        raise RuntimeError(
            f"Profile '{profile.name}' backend=llama-cola requires a 'llama_cola:' block"
        )

    cfg = profile.llama_cola

    repo_id: Optional[str] = None

    if settings.llama_cola_model_path_override:
        model_path = settings.llama_cola_model_path_override
    elif cfg.model_path:
        model_path = cfg.model_path
    elif profile.model_id and Path(profile.model_id).exists():
        model_path = profile.model_id
    else:
        repo_id = cfg.repo_id or profile.model_id
        if not repo_id:
            raise RuntimeError(f"Profile '{profile.name}' missing repo_id or local model_path")
        model_path = str(Path(cfg.model_root) / repo_id.replace("/", "--"))

    return model_path, cfg, repo_id


def _format_messages(messages: List["ChatMessage"], tokenizer: Any) -> str:
    formatted = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                formatted,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.warning("Failed to apply chat template; falling back to raw prompt.")
    prompt_lines = [f"{item['role']}: {item['content']}" for item in formatted]
    prompt_lines.append("assistant:")
    return "\n".join(prompt_lines)

# ----------------------------------------------------------------------
# State & Lifespan
# ----------------------------------------------------------------------

class AppState:
    llm: Optional[IntentionForCausalLM] = None
    tokenizer: Optional[Any] = None
    device: Optional[torch.device] = None
    chassis: Optional[NeuralHostChassis] = None

state = AppState()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 0. Wait if requested
    if settings.wait_for_model_seconds > 0:
        logger.info(f"Waiting {settings.wait_for_model_seconds}s before loading model...")
        await asyncio.sleep(settings.wait_for_model_seconds)

    # 1. Load Profile
    logger.info(f"Resolving profile: {settings.llm_profile_name}")
    try:
        profile = settings.resolve_profile()
    except Exception as e:
        logger.error(f"Failed to resolve profile: {e}")
        raise e
    
    model_path, cfg, repo_id = _resolve_runtime(profile)
    logger.info(f"Selected model path: {model_path}")

    # 2. Download/Ensure Model
    _ensure_model_path(model_path, cfg, repo_id)
    
    # 3. Apply GPU overrides (before Llama init)
    if settings.cuda_visible_devices_override:
        logger.info(f"Overriding CUDA_VISIBLE_DEVICES={settings.cuda_visible_devices_override}")
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices_override
    elif profile.gpu.device_ids:
        # Also respect profile-level pinning if override not set
        dev_ids = ",".join(str(i) for i in profile.gpu.device_ids)
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={dev_ids} from profile")
        os.environ["CUDA_VISIBLE_DEVICES"] = dev_ids

    # 4. Initialize CoLA
    logger.info("Loading CoLA model...")
    try:
        state.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        state.llm = IntentionForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
        )
        state.llm.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state.llm.to(device)
        state.device = device
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    # 5. Start Bus Heartbeat
    if BaseChassis:
        logger.info("Starting Bus Chassis...")
        chassis_cfg = ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            instance_id=settings.instance_id,
            bus_url=settings.bus_url,
        )
        # Note: ChassisConfig currently doesn't support 'enforce_catalog' directly in constructor
        # based on memory/inspection of bus_service_chassis, but we can set env var or
        # trust OrionBusAsync to pick it up from env if it supports it.
        # The env var ORION_BUS_ENFORCE_CATALOG is already in os.environ by virtue of docker env.
        
        state.chassis = NeuralHostChassis(chassis_cfg)
        await state.chassis.start_background()
    else:
        logger.warning("Orion Bus lib not found; skipping heartbeats.")

    yield

    # Cleanup
    if state.chassis:
        await state.chassis.stop()

app = FastAPI(title="Orion Llama CoLA Host", lifespan=lifespan)

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


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    if not state.llm or not state.tokenizer or not state.device:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        # Neural Projection requires the full response to generate the embedding.
        raise HTTPException(status_code=400, detail="Streaming is not supported by Neural Host.")

    prompt = _format_messages(request.messages, state.tokenizer)
    inputs = state.tokenizer(prompt, return_tensors="pt").to(state.device)

    state.llm.set_action_sampling(greedy=False, tau=2.0)
    with torch.no_grad():
        state.llm.reset_action_info()
        outputs = state.llm.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=request.max_tokens,
            do_sample=False,
        )
        action_indices, _ = state.llm.get_action_info(prob=True)

    generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
    text_output = state.tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    response = {
        "id": f"cola-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": request.model or settings.llm_profile_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_output[0] if text_output else "",
                },
                "finish_reason": "stop",
            }
        ],
        "action_indices": action_indices.tolist() if action_indices is not None else [],
        "text_output": text_output,
    }

    return response


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    if not state.llm or not state.tokenizer or not state.device:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = request.input if isinstance(request.input, list) else [request.input]
    data = []
    for idx, text in enumerate(inputs):
        embedding: List[float] = []
        if text:
            try:
                encoded = state.tokenizer(str(text), return_tensors="pt").to(state.device)
                state.llm.set_action_sampling(greedy=False, tau=2.0)
                with torch.no_grad():
                    state.llm.reset_action_info()
                    _ = state.llm.generate(
                        **encoded,
                        use_cache=True,
                        max_new_tokens=1,
                        do_sample=False,
                    )
                    action_indices, _ = state.llm.get_action_info(prob=True)
                if action_indices is not None:
                    embedding = action_indices[0].tolist()
            except Exception as e:
                logger.error(f"Action extraction failed: {e}")
        data.append({"object": "embedding", "embedding": embedding, "index": idx})

    return {"object": "list", "data": data, "model": request.model or "llama-cola"}

@app.get("/health")
def health():
    if not state.llm:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "ok"}
