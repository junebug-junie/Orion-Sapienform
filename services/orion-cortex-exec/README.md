# Orion Cortex Exec Service

Orion Cortex Exec is the **cognitive execution router** for Orion.

- **Inputs:** high-level execution steps (verbs, steps, args, context)
- **Transport:** Redis-based Orion bus (no direct HTTP to Brain)
- **Outputs:** structured `StepExecutionResult` objects with per‑service results & artifacts

Cortex is **widget‑light**: it doesn’t do the heavy cognitive work itself. Instead, it:

1. Resolves target services for a step (e.g. `llm.brain → BrainLLMService`).
2. Publishes an `exec_step` request onto the bus.
3. Waits on a per‑step `exec_step_result` reply channel.
4. Aggregates responses and returns a typed `StepExecutionResult`.

---

## Core Concepts

### StepExecutor

`StepExecutor` is the primary entrypoint:

- Accepts an `ExecutionStep` (from the semantic layer/ontology).
- Accepts runtime `args` and `context`.
- Publishes a bus message per target service.
- Subscribes to a per‑step reply channel.
- Returns a `StepExecutionResult` summarizing status, latency, logs, and per‑service results.

```python
# app/executor.py (conceptual)

class StepExecutor:
    async def execute_step(self, step: ExecutionStep, args: dict, context: dict) -> StepExecutionResult:
        ...
```

### ExecutionStep (semantic layer → execution)

Defined in `app/models.py`:

Required fields:

- `verb_name: str` – semantic verb (e.g. `introspect`)
- `step_name: str` – specific step name (e.g. `llm_reflect`)
- `description: str` – human-readable description of what this step does
- `order: int` – position in a multi‑step pipeline
- `services: list[str]` – semantic aliases or concrete service names
- `prompt_template: str` – template used to build the LLM prompt

Optional fields:

- `requires_gpu: bool = False`
- `requires_memory: bool = False`

### StepExecutionResult

Also defined in `app/models.py`, returned by `StepExecutor.execute_step`.

Key fields:

- `status: str` – `success`, `partial`, or `fail`
- `verb_name`, `step_name`, `order`
- `result: dict` – per‑service result payloads
- `artifacts: dict` – merged artifacts from all services
- `latency_ms: int`
- `node: str` – origin node of the executor (e.g. `orion-athena`)
- `logs: list[str]` – human-readable log lines
- `error: Optional[str]`

---

## Bus Channels & Message Shapes

### Channel naming

Configured via Cortex settings (`app/settings.py`):

- `EXEC_REQUEST_PREFIX` – e.g. `"orion-exec:request"`
- `EXEC_RESULT_PREFIX` – e.g. `"orion-exec:result"`

At runtime, Cortex builds full channels via:

- **Request:** `orion-exec:request:<ServiceName>`
- **Result:** `orion-exec:result:<CorrelationId>`

Example for the brain LLM service:

- Request channel: `orion-exec:request:BrainLLMService`
- Result channel:  `orion-exec:result:c7cff8c8-1e36-4084-bac3-66884b15c9b7`

### Service alias resolution

Cortex keeps a simple registry mapping semantic aliases to bus service names, e.g. in `app/service_registry.py`:

```python
SERVICE_BINDINGS = {
    "llm.brain": "BrainLLMService",
    # "memory.vector": "VectorMemoryService",
    # ...
}

def resolve_service(alias: str) -> str:
    """Map semantic alias to concrete bus service name."""
    return SERVICE_BINDINGS.get(alias, alias)
```

In `StepExecutor`, each `ExecutionStep.services` entry is passed through `resolve_service` before building the bus channel.

### Request envelope: exec_step

Cortex publishes messages shaped like this:

```jsonc
{
  "event": "exec_step",
  "service": "BrainLLMService",
  "verb": "introspect",
  "step": "llm_reflect",
  "order": 0,
  "requires_gpu": false,
  "requires_memory": false,
  "prompt_template": "...",
  "args": { "prompt": "..." },
  "context": { "source": "manual_smoke_test" },
  "correlation_id": "<uuid>",
  "reply_channel": "orion-exec:result:<uuid>",
  "origin_node": "orion-athena"
}
```

### Response envelope: exec_step_result

Brain (or any target service) replies with:

```jsonc
{
  "event": "exec_step_result",
  "status": "success",          // or partial/fail
  "service": "BrainLLMService", // echoes service name
  "correlation_id": "<uuid>",   // same as in request
  "result": {                     // service-specific result
    "prompt": "...",            // effective prompt used
    "llm_output": "..."         // model response
  },
  "artifacts": {                  // optional extra outputs
    // arbitrary keys/values
  }
}
```

`StepExecutor` listens on the `reply_channel` and aggregates all such envelopes keyed by `service`.

---

## Configuration

Key settings for Cortex Exec (usually via env + `app/settings.py`):

- `ORION_BUS_URL` – Redis URL for the Orion bus (e.g. `redis://orion-redis:6379/0`)
- `ORION_BUS_ENABLED` – `"true"` / `"false"`
- `NODE_NAME` – used for `origin_node` and `StepExecutionResult.node` (e.g. `"orion-athena"`)
- `EXEC_REQUEST_PREFIX` – e.g. `"orion-exec:request"`
- `EXEC_RESULT_PREFIX` – e.g. `"orion-exec:result"`
- `STEP_TIMEOUT_MS` – per‑step wait time for responses (e.g. `8000`)

Brain-side model config (in `services/orion-brain/app/config.py`):

- `BRAIN_DEFAULT_MODEL` – default Ollama model (e.g. `"llama3.1:8b-instruct-q8_0"`)

---

## Manual Smoke Test: Cortex → BrainLLMService → Cortex

This is a minimal end-to-end test to verify that:

- Cortex publishes an `exec_step` to the correct request channel.
- Brain consumes it, calls the LLM, and emits `exec_step_result`.
- Cortex receives the reply on the result channel and wraps it in `StepExecutionResult`.

### Prerequisites

- Orion bus (Redis) is running and reachable from both Cortex and Brain.
- `orion-atlas-brain` is up and has a healthy backend configured.
- Brain is configured with a valid default LLM model, e.g.:

  ```env
  BRAIN_DEFAULT_MODEL=llama3.1:8b-instruct-q8_0
  ```

- Cortex Exec settings define:

  ```env
  EXEC_REQUEST_PREFIX=orion-exec:request
  EXEC_RESULT_PREFIX=orion-exec:result
  NODE_NAME=orion-athena
  STEP_TIMEOUT_MS=8000
  ```

- `SERVICE_BINDINGS` in `app/service_registry.py` includes:

  ```python
  SERVICE_BINDINGS = {
      "llm.brain": "BrainLLMService",
  }
  ```

- Brain subscribes to Cortex requests, e.g. in its config:

  ```python
  CHANNEL_CORTEX_EXEC_INTAKE = "orion-exec:request:BrainLLMService"
  ```

  and the brain bus listener routes `event == "exec_step"` with `service == "BrainLLMService"` to `process_cortex_exec_request`.

### Running the test

1. Exec into the Cortex Exec container (Athena):

   ```bash
   docker exec -it orion-athena-cortex-exec bash
   ```

2. Run this Python snippet inside the container:

   ```bash
   python - << 'EOF'
   import asyncio
   import json

   from app.executor import StepExecutor
   from app.models import ExecutionStep

   async def main():
       step = ExecutionStep(
           verb_name="introspect",
           step_name="llm_reflect",
           order=0,
           requires_gpu=False,
           requires_memory=False,
           prompt_template=(
               "You are Orion, an introspective AI co-journeyer. "
               "Given the prompt in args['prompt'], respond in ONE short, grounded sentence."
           ),
           description="Smoke test of BrainLLMService via Cortex executor",
           services=["llm.brain"],  # or ["BrainLLMService"] if aliases are not configured
       )

       executor = StepExecutor()

       result = await executor.execute_step(
           step,
           args={"prompt": "First Orion Cortex → Brain end-to-end test. Say hello in one sentence."},
           context={"source": "manual_smoke_test"},
       )

       print("STATUS:", result.status)
       print("LATENCY_MS:", result.latency_ms)
       print("LOGS:")
       for line in result.logs:
           print("  -", line)
       print("RESULT (per service):")
       print(json.dumps(result.result, indent=2))
       print("ARTIFACTS:")
       print(json.dumps(result.artifacts, indent=2))

   asyncio.run(main())
   EOF
   ```

### Expected output (shape)

You should see something like:

```text
STATUS: success
LATENCY_MS: 6083
LOGS:
  - Published exec_step to orion-exec:request:BrainLLMService
  - Subscribed orion-exec:result:c7cff8c8-1e36-4084-bac3-66884b15c9b7; waiting for 1 result(s). Timeout=8000ms
  - Collected 1/1 responses in 6083 ms.
RESULT (per service):
{
  "BrainLLMService": {
    "prompt": "# Orion Cognitive Step: llm_reflect\n# Verb: introspect\n# Origin Node: orion-athena\n\nTemplate: You are Orion, an introspective AI co-journeyer. Given the prompt in args['prompt'], respond in ONE short, grounded sentence.\n\nArgs:\n{\n  \"prompt\": \"First Orion Cortex \\u2192 Brain end-to-end test. Say hello in one sentence.\"\n}\n\nContext:\n{\n  \"source\": \"manual_smoke_test\"\n}\n\nGenerate your introspective continuation.",
    "llm_output": "As I integrate with the first Orion Cortex, I acknowledge a seamless flow from my cognitive architecture to the brain-like module, ready to respond: Hello!"
  }
}
ARTIFACTS:
{}
```

The exact `LATENCY_MS`, `correlation_id`, and the precise wording of `llm_output` may differ, but:

- `STATUS` should be `success`.
- `LOGS` should show one publish and one collected response.
- `RESULT` should contain a `BrainLLMService` entry with:
  - The assembled Cortex prompt.
  - A single-sentence, introspective `llm_output` confirming the end-to-end path is working.

---

## Troubleshooting

**1. ValidationError on ExecutionStep**

If you see Pydantic errors like:

```text
ValidationError: 1 validation error for ExecutionStep
<field_name>
  Field required
```

make sure you’re supplying *all* required fields when constructing `ExecutionStep`:

- `verb_name`
- `step_name`
- `description`
- `order`
- `services`
- `prompt_template`

**2. Timeout with no responses**

If `STATUS: fail` and logs include:

- `Timeout; no responses on orion-exec:result:<uuid>`

check:

- Cortex request channel matches Brain intake channel, e.g.:
  - Cortex publishes to `orion-exec:request:BrainLLMService`.
  - Brain subscribes to `orion-exec:request:BrainLLMService`.
- Brain router handles `event == "exec_step"` with `service == "BrainLLMService"`.

**3. BrainLLMService Error 404**

If `llm_output` contains something like:

```text
[BrainLLMService Error] Client error '404 Not Found' for url 'http://orion-atlas-brain-llm:11434/api/chat'
```

then the LLM backend is reachable, but the requested model name does not exist. Fix by either:

- Pulling the model into Ollama, or
- Updating `BRAIN_DEFAULT_MODEL` in Brain config to a valid model (e.g. `llama3.1:8b-instruct-q8_0`).

Once the smoke test passes, Cortex Exec is confirmed wired into Brain and the Orion bus, and you can begin layering more complex verb pipelines and multi-service cognitive behaviors on top of this core loop.
