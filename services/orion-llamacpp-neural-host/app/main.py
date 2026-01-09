# services/orion-llamacpp-neural-host/app/main.py
import logging
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from llama_cpp import Llama

from app.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-host")

app = FastAPI(title="Orion Neural Host")

# Initialize Llama with embedding=True
logger.info(f"Loading model from {settings.model_path} with embedding=True...")
try:
    llm = Llama(
        model_path=settings.model_path,
        n_gpu_layers=settings.n_gpu_layers,
        n_ctx=settings.n_ctx,
        embedding=True,
        verbose=True
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Raise to crash container so it restarts or alerts
    raise e

# Request schemas (simplified OpenAI compatible)
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

@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        # Neural Projection requires the full response to generate the embedding.
        # We explicitly disable streaming support for this host.
        raise HTTPException(status_code=400, detail="Streaming is not supported by Neural Host (requires full text for embedding).")

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    # 1. Call create_chat_completion
    response = llm.create_chat_completion(
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
             # llm.embed(content) returns a list of floats (for single string)
             spark_vector = llm.embed(content)
        except Exception as e:
             logger.error(f"Embedding failed: {e}")
             spark_vector = []

    # 4. Injection
    # Inject spark_vector into the response JSON
    # create_chat_completion returns a dict, so we can modify it directly
    response["spark_vector"] = spark_vector

    return response

@app.get("/health")
def health():
    return {"status": "ok"}
