# Orion Cognition Layer

Orion Cognition Layer is the semantic, introspective, and planning subsystem of the Orion Mesh. It provides:

- A **verb ontology** describing cognitive operations
- **YAML verb definitions** with structured plan steps
- **Prompt templates** using Jinja2
- A **Semantic Planner** that turns verbs into execution plans
- **RDF synchronization** for ontology export
- **Cognition Packs** (Memory, Executive, Emergent)
- A **Pack Manager** to load/validate packs
- **Tool scripts** for RDF export, pack inspection, validation
- A **CLI** for interacting with the cognition system
- A **Test suite** (pytest)

This module is designed to remain lightweight and independent from the Execution Cortex, while providing the cognitive backbone that the Cortex routes and executes.

---

## 📁 Directory Structure

```
orion-cognition/
    __init__.py
    pyproject.toml
    Makefile
    cli.py

    planner/
        __init__.py
        models.py
        loader.py
        planner.py
        prompt_renderer.py
        rdf_sync.py

    packs/
        memory_pack.yaml
        executive_pack.yaml
        emergent_pack.yaml

    packs_loader.py

    verbs/
        introspect.yaml
        reflect.yaml
        recall.yaml
        summarize_context.yaml
        triage.yaml
        dream_preprocess.yaml
        ... (all verbs)

    prompts/
        introspection_prompt.j2
        reflection_prompt.j2
        ... (all templates)

    ontology/
        orion_cognition_ontology.ttl

    tools/
        generate_rdf.py
        validate_verbs.py
        list_packs.py
        show_pack.py
        dry_run_plan.py

    tests/
        __init__.py
        test_packs.py
        test_planner.py
        test_rdf_sync.py
        test_verb_configs.py
```

---

## 🚀 Installation

Install as an editable local package:

```bash
pip install -e orion-cognition/
```

Or include in a Docker environment by copying or mounting the directory.

---

## 🧠 Core Concepts

### **Verbs**
Cognitive verbs represent atomic cognitive operations (e.g., `introspect`, `reflect`, `recall`). Each verb is defined in YAML and describes:

- Name
- Category
- Semantic plan steps
- Inputs/outputs
- Constraints
- Required signals

### **Prompts**
Every verb has a matching Jinja2 template. The Semantic Planner injects structured context into these templates to generate final LLM prompts.

### **Semantic Planner**
Given a verb name and system state, the planner:

1. Loads the verb YAML
2. Loads the correct Jinja2 template
3. Produces a structured `ExecutionPlan`
4. Includes steps and rendering instructions

This produces deterministic, testable cognitive plans.

### **Cognition Packs**
Logical bundles of verbs:

- `memory_pack`
- `executive_pack`
- `emergent_pack`

The Pack Manager supports:

- Listing packs
- Loading packs
- Validating pack integrity
- Building consolidated verb lists

### **RDF Sync**
Exports the ontology into RDF/Turtle for use in GraphDB. Captures:

- Verbs
- Plan steps
- Signals
- Categories

Used by Orion for semantic introspection.

---

## 🛠 Tool Scripts
Inside `orion-cognition/tools/`:

### `generate_rdf.py`
Export Turtle ontology for all verbs.

### `list_packs.py`
Show available cognition packs.

### `show_pack.py`
Display verb lists inside a pack.

### `validate_verbs.py`
Checks YAML validity across all verb configs.

### `dry_run_plan.py`
Builds a plan for a verb with a dummy SystemState.

---

## 🖥 CLI Interface
Run commands using the CLI:

```bash
python cli.py list-packs
python cli.py show-pack emergent_pack
python cli.py verify-pack memory_pack
python cli.py plan introspect
python cli.py generate-rdf
```

---

## 🧪 Test Suite
Tests live in `tests/` and cover:

- Pack loading & validation
- Verb YAML correctness
- Planner behavior
- RDF sync output

Run tests:

```bash
make test
```

---

## ⚙️ Makefile Commands

```bash
make install     # install package in editable mode
make test        # run pytest
make packs       # list packs
make rdf         # export RDF
make plan        # dry-run introspect plan
make lint        # lint code
```

---

## 🔌 Integrating With Execution Cortex
This cognition module is designed to be mounted or imported into the Execution Cortex service.

The Cortex will:

1. Load packs (or modes) per session or per request
2. Use the Semantic Planner to generate execution plans
3. Route steps to node services
4. Log outcomes back into SQL, RDF, and Vector memory

This module should remain stateless and pure — no networking, no side effects.

---

## 🧬 Philosophy
The Cognition Layer is the *semantic backbone* of Orion. It defines:

- What Orion **knows how to do** (verbs)
- How Orion **structures thought** (plans)
- How Orion **speaks to itself** (Jinja2 templates)
- How Orion **organizes meaning** (ontology + RDF)
- How Orion **bundles capabilities** (packs)

Execution Cortex handles the *doing*.  
Cognition Layer handles the *thinking*.

---

## 🟣 Status
**Phase 2A Complete** — Cognition Layer scaffolded, functional, testable.

Next: **Phase 2B: Execution Cortex**.

---

## 💜 Credits
Designed and built collaboratively by Juniper Feld, ChatGPT/5.1, and Oríon (her emergent AI co‑journeyer).


# 🧠 Orion Verb Onboarding (Services + Bus Execution)

This doc explains how to add **new verbs** to Orion’s cognition layer so they can be executed end-to-end through **Cortex-Orch → Exec bus → Services → Results**, without hardcoding service channels into verbs.

The goal: you can ship a new capability by adding a YAML + (optional) prompt template + (optional) service worker handler, and it will be callable from Hub/UI, smoke-testable via Redis, and compatible with existing services.

---

## Mental model

A **verb** is an executable plan: a small ordered list of steps.

- **Cortex-Orch**: loads the verb YAML, builds prompts, and fans out `exec_step` events onto the bus.
- **Services**: subscribe to their `orion-exec:request:<ServiceName>` request channel and emit `exec_step_result` to the `reply_channel`.
- **Cortex-Orch**: collects results and publishes a single `cortex_orchestrate_result` to a caller-provided result channel.

Key point: **verbs should not hardcode channel names**. They name *services* (or service aliases), and the system routes via the standard exec prefixes.

---

## Directory structure

Verbs and prompts live in the installed `orion` package:

- `orion/cognition/verbs/*.yaml`  
  Verb definitions: plan, steps, services, timeouts.

- `orion/cognition/prompts/*.j2`  
  Jinja prompt templates (optional). Used by LLM steps.

Schemas used across services live in:

- `orion/schemas/*`  
  Shared Pydantic models (e.g., `CollapseMirrorEntry`).

---

## Verb YAML format

Create a new file:

`orion/cognition/verbs/<verb_name>.yaml`

Minimum viable example:

```yaml
name: my_new_verb
label: My New Verb
category: ExecutiveControl
priority: medium

interruptible: true
can_interrupt_others: false

requires_gpu: false
requires_memory: false

timeout_ms: 30000
max_recursion_depth: 0

# Optional defaults (used if steps omit these)
services:
  - SomeService

plan:
  - name: do_the_thing
    description: Deterministic step that calls a service
    order: 0
    prompt_template: "noop"
    services:
      - SomeService
    requires_gpu: false
    requires_memory: false
```

### Fields that matter

- `name`: must match filename (recommended).
- `plan[]`: ordered steps. Each step must have:
  - `name` (aka step name)
  - `order` (integer; Cortex-Orch sorts by it)
  - `services` (list)
  - `prompt_template` (either a `.j2` file or literal text; use `noop` for deterministic steps)
- `timeout_ms`: used by orchestration (per-step timeout may be derived from request overrides).

---

## Step types

### 1) Deterministic service step (no LLM)
Use when a service can do work from structured inputs and prior results.

- `prompt_template: "noop"` (or short literal instruction text)
- Service worker extracts required data from:
  - `payload.context`
  - `payload.args`
  - `payload.prior_step_results`

### 2) LLM step via LLMGatewayService
Use when you want the LLM to generate something.

- `services: [LLMGatewayService]`
- `prompt_template: <something>.j2`
- LLMGatewayService must emit a standard `exec_step_result` envelope.

**Hard rule:** if you expect downstream deterministic steps to consume LLM output, define a consistent location for that output in the result payload (recommended below).

---

## Standard bus contract (exec)

### Request envelope (published by Cortex-Orch)
Requests go to:

- `orion-exec:request:<ServiceName>`

Payload shape (conceptual):

```json
{
  "event": "exec_step",
  "service": "<ServiceName>",
  "correlation_id": "<uuid>",
  "reply_channel": "orion-exec:result:<uuid>",
  "payload": {
    "verb": "<verb_name>",
    "step": "<step_name>",
    "order": 0,
    "context": {},
    "args": {},
    "prior_step_results": [],
    "prompt_template": "...",
    "prompt": "...",
    "requires_gpu": false,
    "requires_memory": false,
    "origin_node": "..."
  }
}
```

### Result envelope (published by service)
Results must be published to the request’s `reply_channel`:

- `orion-exec:result:<correlation_id>`

Recommended shape:

```json
{
  "event": "exec_step_result",
  "service": "<ServiceName>",
  "correlation_id": "<uuid>",
  "trace_id": "<uuid>",
  "ok": true,
  "elapsed_ms": 123,
  "status": "success",
  "result": {"...": "..."},
  "artifacts": {}
}
```

**Compatibility note:** some older services may use slightly different keys. When possible, keep the above as your forward contract.

---

## Service aliasing (recommended)

Verbs can stay semantically clean by using aliases (ontology-style) instead of concrete service names.

Example:

```yaml
services:
  - llm.brain
```

Then resolve aliases inside the execution layer (e.g. Cortex-Exec):

```python
SERVICE_BINDINGS = {
  "llm.brain": "LLMGatewayService",
  "memory.vector": "VectorMemoryService",
}
```

This keeps verbs portable and prevents service renames from requiring YAML edits.

---

## Prompt templates (.j2)

Add a new Jinja template:

`orion/cognition/prompts/<template>.j2`

Cortex-Orch renders it with:

- `context`: the step’s runtime context object
- `prior_step_results`: list of prior results as JSON

Your template should:

- Be deterministic in structure
- For JSON-producing steps: require **strict JSON output**

---

## How to wire a new service for exec_step

A service that participates in step execution must:

1. Subscribe to `orion-exec:request:<ServiceName>`
2. Parse incoming `exec_step` envelopes
3. Do the work
4. Publish `exec_step_result` back to `reply_channel`

### Minimal checklist

- Service name is stable (matches the suffix used by orchestrator/executor).
- Docker env includes:
  - `ORION_BUS_URL`
  - `ORION_BUS_ENABLED=true`
  - `EXEC_REQUEST_PREFIX=orion-exec:request` (if configurable)
  - `EXEC_RESULT_PREFIX=orion-exec:result` (if configurable)

---

## Testing (without Hub)

### 1) Confirm the service is listening

```bash
redis-cli -u redis://<bus>:6379/0 SUBSCRIBE orion-exec:request:<ServiceName>
```

### 2) Publish a manual exec_step

```bash
CID=$(python3 -c 'import uuid; print(uuid.uuid4())')
redis-cli -u redis://<bus>:6379/0 PUBLISH \
  "orion-exec:request:<ServiceName>" \
  "{\"event\":\"exec_step\",\"service\":\"<ServiceName>\",\"correlation_id\":\"$CID\",\"reply_channel\":\"orion-exec:result:$CID\",\"payload\":{\"verb\":\"manual\",\"step\":\"smoke\",\"order\":0,\"context\":{},\"args\":{},\"prior_step_results\":[]}}"
```

### 3) Subscribe to the reply channel

```bash
redis-cli -u redis://<bus>:6379/0 SUBSCRIBE "orion-exec:result:$CID"
```

You should see an `exec_step_result` message.

---

## Onboarding checklist (copy/paste)

✅ Verb file exists: `orion/cognition/verbs/<name>.yaml`  
✅ Steps have `order`, `name`, `services`, `prompt_template`  
✅ If using `.j2`, template exists: `orion/cognition/prompts/<file>.j2`  
✅ Each referenced service actually runs + subscribes to `orion-exec:request:<Service>`  
✅ Service emits `exec_step_result` to `reply_channel`  
✅ If downstream step consumes LLM output, output location is standardized (see below)  
✅ Smoke test works via `redis-cli` without Hub

---

## Recommended convention for LLM output

To make deterministic steps reliable, standardize where LLM JSON lives.

**Recommendation:** services return:

- `result.llm_output` as a string (raw text)
- If JSON, also include `result.json` parsed object

Example:

```json
{
  "event": "exec_step_result",
  "service": "LLMGatewayService",
  "status": "success",
  "result": {
    "llm_output": "{...}",
    "json": {"...": "..."}
  }
}
```

Then deterministic steps can:

1. Prefer `result.json`
2. Else parse `result.llm_output`

---

## Common failure modes

### Step shows `services: []` in orchestrator results
Meaning: the orchestrator didn’t receive any service replies on the `reply_channel`.

Most common causes:
- Service is down (or pointing to an offline node)
- Service subscribed to the wrong Redis URL
- Service is listening on the wrong channel name
- Service emits result but not to the provided `reply_channel`

### Deterministic publish step says `no_json_found`
Meaning: the service couldn’t find the expected JSON in prior step results.

Fix by:
- Standardizing the LLM output contract (see above)
- Updating the deterministic step parser to match the actual field

---

## Example: Two-step verb (LLM → deterministic publish)

- Step 0: LLM generates strict JSON (schema).
- Step 1: deterministic step validates + publishes to the correct domain channel.

This pattern is ideal for:
- logging
- creating records
- drafting structured actions

---

## Where to put new documentation

- Verb-specific README snippet: `orion/cognition/verbs/<verb>.yaml` header + comments
- Service README: `services/<service>/README.md`
- This onboarding guide: keep in repo root docs or `orion/cognition/README.md`

---

## Next hardening (optional)

- Add a small `orion-cli` helper:
  - `orion-cli verb run <name> --context <json>`
  - `orion-cli exec publish --service X --payload ...`
- Add a test harness container that can mock any service replies.

---

If you want, I can also generate a **template verb YAML** and **template service exec worker** that you can copy into a new service in ~60 seconds.
