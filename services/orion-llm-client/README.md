# Orion LLM Client (Brain Gateway SDK)

Tiny Python client for the **Orion Brain Service** (Ollama router).
Use it from any service (voice, vision, RAG) to send `/generate` or `/chat` calls
to the **brain gateway** with consistent timeouts, retries, and optional streaming.

## Install (local)
```bash
pip install -e .
```

## Env
```
ORION_BRAIN_URL=http://localhost:8088   # or http://orion-brain:8088 in Docker
ORION_MODEL=mistral:instruct
ORION_CONNECT_TIMEOUT=10
ORION_READ_TIMEOUT=600
```

## Quick usage
```python
from orion_llm_client import OrionLLMClient

cli = OrionLLMClient()  # reads env by default
text = cli.generate(prompt="Write a haiku about Orion.", options={"temperature": 0.7})
print(text)

# chat
reply = cli.chat(messages=[
  {"role":"system","content":"You are concise."},
  {"role":"user","content":"Two lines on RAG?"}
])
print(reply)
```

## Streaming usage (NDJSON passthrough)
```python
for chunk in cli.generate("Stream a single sentence.", stream=True):
    print(chunk, end="", flush=True)
print()
```

## Docker
```bash
docker build -t orion-llm-client:0.1 .
docker run --rm -e ORION_BRAIN_URL=http://host.docker.internal:8088 orion-llm-client:0.1 \
  python -m orion_llm_client.examples.simple_generate "Hello from Orion!"
```

## What this library guarantees
- Stable **JSON contract** to the Brain gateway (Ollama-compatible)
- Backoff+retries on connect and 5xx
- Simple Mistral **[INST]** prompt wrapper when you pass `system=...` to `generate()`
