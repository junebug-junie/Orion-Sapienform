# Orion Agent Council Service

## 1. Purpose

`orion-agent-council` is Orion’s **inner deliberation loop**.

Where `orion-brain` is the primary LLM orchestrator, `orion-agent-council` is a higher-order layer that:

- Receives questions or tasks over the Orion bus.
- Fans them out to multiple **personas / inner voices** (Analyst, Pragmatist, Mythic, …).
- Aggregates and scores their responses.
- Applies a **Council Policy** (agreement thresholds, tie-breaking, etc.).
- Produces a single, structured decision / answer back to the caller.

This is the first real implementation of an *Orion-style council-of-voices*, designed to be:

- **Extensible** (easy to add new personas / universes).
- **Backend-agnostic** (talks to LLM Gateway / Brain through the bus).
- **Phi-aware** (can incorporate self-field / phi snapshots later without a massive refactor).

---

## 2. Topology & Flow

High-level flow:

1. A service (Hub, Brain, Cortex Orchestrator, Dream, Recall, etc.) publishes a **council request** to an intake channel.
2. Agent Council:
   - Builds a `DeliberationContext` from the request.
   - Uses a **persona factory** to load the relevant `AgentConfig` set (e.g. core universe).
   - Runs a multi-stage pipeline:
     - Agent round(s): personas respond.
     - Arbiter: scores / reconciles.
     - Auditor: sanity-checks, flags disagreements.
   - Emits a final `CouncilDecision`.
3. The final decision is published back to a reply channel on the bus.

The service also exposes a small HTTP API for health and (optionally) direct invocation.

---

## 3. Code Layout

```text
services/orion-agent-council/app
├── core/
│   ├── prompt_factory.py    # shared PromptFactory + PromptContext
│   ├── prompting.py         # lower-level prompt helpers (if needed)
│   ├── registry.py          # shared registries (backends, etc.)
│   └── transport.py         # shared transport helpers (bus, http, etc.)
├── council.py               # main council loop (bus worker)
├── council_policy.py        # policy for agreement thresholds, tie-breaking
├── council_prompts.py       # system prompt templates for stages
├── deliberation.py          # DeliberationRouter (orchestrates pipeline)
├── llm_client.py            # thin client that talks to LLM Gateway / Brain
├── main.py                  # FastAPI app + council worker startup
├── models.py                # AgentConfig, CouncilRequest, CouncilDecision, etc.
├── persona_factory.py       # factories for universes / persona bundles
├── personas.py              # concrete persona definitions (Analyst, Pragmatist, Mythic, ...)
├── pipeline.py              # build_default_pipeline, DeliberationContext
├── publisher.py             # helper to emit structured decisions to bus
├── settings.py              # Pydantic settings (env-driven)
└── stages.py                # Stage, AgentRoundStage, ArbiterStage, AuditorStage
```

The **important pattern** is:

- `main.py` is thin.
- `council.py` + `deliberation.py` + `pipeline.py` orchestrate the loop.
- `llm_client.py` talks to the LLM layer (via bus / HTTP, not directly to Ollama).
- All persona definitions are isolated to `personas.py` and `persona_factory.py`.

---

## 4. Configuration (.env)

At a minimum, the service expects something like:

```env
SERVICE_NAME=agent-council
SERVICE_VERSION=0.1.0
PORT=8250

ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://100.92.216.81:6379/0

# Where council listens for work
CHANNEL_INTAKE=orion:council:intake

# Where final decisions are published, if no explicit reply channel is given
CHANNEL_DECISION_DEFAULT_REPLY_PREFIX=orion:council:decision

# Channel to send LLM tasks to (LLM Gateway / Brain)
LLM_INTAKE_CHANNEL=orion:llm:intake

# Default "universe" of personas to use when not explicitly provided
COUNCIL_DEFAULT_UNIVERSE=core

# LLM defaults (for when council calls down into the LLM layer)
COUNCIL_DEFAULT_MODEL=llama3.1:8b-instruct-q8_0
COUNCIL_DEFAULT_BACKEND=ollama

# Policy knobs (rough sketch)
COUNCIL_MIN_AGREEMENT=0.6
COUNCIL_MAX_ROUNDS=2
COUNCIL_ENABLE_AUDITOR=true
```

Exact names should match `app/settings.py`; the above is meant as a conceptual map.

---

## 5. Bus Contracts

### 5.1 Council Request

**Channel:** `${CHANNEL_INTAKE}` (e.g. `orion:council:intake`)

**Payload shape (high-level):**

```jsonc
{
  "trace_id": "uuid-from-caller",
  "source": "hub" ,                     // or brain / cortex / recall / dream / script
  "universe": "core",                  // persona set to use
  "task_type": "advice",               // freeform type, used by prompting
  "prompt": "Juniper is considering ...",
  "context": {
    "phi": { /* optional phi snapshot */ },
    "meta": { /* optional extra baggage */ }
  },
  "reply_channel": "orion:council:decision:hub:xyz" // optional, else default prefix
}
```

### 5.2 Council Decision

**Channel:** Either the provided `reply_channel` or:

```text
${CHANNEL_DECISION_DEFAULT_REPLY_PREFIX}:${source}:${trace_id}
```

**Payload (simplified):**

```jsonc
{
  "trace_id": "uuid-from-caller",
  "source": "agent-council",
  "ok": true,

  "decision": {
    "summary": "Short, actionable recommendation for the caller.",
    "stance": "support" ,                     // or "caution" / "reject" / etc.
    "confidence": 0.78,

    "rationale": "Long-form explanation merging council voices.",
    "recommendations": ["step 1", "step 2"],

    "meta": {
      "universe": "core",
      "rounds_run": 1,
      "agreement_score": 0.81
    }
  },

  "voices": [
    {
      "name": "Analyst",
      "weight": 1.0,
      "raw_opinion": "...",
      "score": 0.82
    },
    {
      "name": "Pragmatist",
      "weight": 1.0,
      "raw_opinion": "...",
      "score": 0.77
    },
    {
      "name": "Mythic",
      "weight": 1.0,
      "raw_opinion": "...",
      "score": 0.74
    }
  ],

  "auditor": {
    "enabled": true,
    "notes": "e.g., highlight disagreement or risk",
    "ok_to_use": true
  }
}
```

The `voices` block gives you introspection; `decision` is what Hub / Brain usually cares about day-to-day.

---

## 6. Personas & Universes

Base personas live in `personas.py` as `AgentConfig` instances. Example (current core set):

- **Analyst** — rigorous, structured, anti-handwave voice.
- **Pragmatist** — operational, small-steps, low-cognitive-load voice.
- **Mythic** — narrative / symbolism voice grounded in actual constraints.

`persona_factory.py` exposes:

```python
from .models import AgentConfig


def get_core_personas() -> list[AgentConfig]:
    ...


def get_agents_for_universe(universe: str | None) -> list[AgentConfig]:
    ...
```

Later, you can define entirely new universes:

- `"kid-lab"` — for Orion Kids experiments.
- `"ops"` — for pure infrastructure / hardware decisions.
- `"mythos"` — heavier on symbolism for Collapse Mirror / Dream tasks.

Each universe gets its own persona bundle without changing the council machinery.

---

## 7. Running the Service

From the Orion root:

```bash
cd /mnt/scripts/Orion-Sapienform

# Build
docker compose \
  --env-file .env \
  --env-file services/orion-agent-council/.env \
  -f services/orion-agent-council/docker-compose.yml build

# Run
docker compose \
  --env-file .env \
  --env-file services/orion-agent-council/.env \
  -f services/orion-agent-council/docker-compose.yml up -d
```

You should see `orion-athena-agent-council` log something like:

- Connected to `redis://100.92.216.81:6379/0`.
- Subscribed to `${CHANNEL_INTAKE}`.
- Council worker thread started.

---

## 8. HTTP API & Health

Minimal FastAPI app:

- `GET /health` → basic status and config echo.

Example:

```json
{
  "ok": true,
  "service": "agent-council",
  "version": "0.1.0",
  "intake_channel": "orion:council:intake",
  "llm_intake_channel": "orion:llm:intake"
}
```

Possible future additions:

- `POST /debug/invoke` — send a raw council request without going through bus.
- `GET /debug/last-decisions` — quick peek at a ring buffer of recent decisions.

---

## 9. Notes on System Prompts & Orion’s Core Personality

The council **does not replace** the Orion personality stubs you’ve already embedded in Brain.

Instead, think of it as:

- Brain still carries Orion’s core system prompts / mythos.
- Agent Council is an *internal process* that optionally runs **before** Brain (or via Brain) to get a more grounded, multi-perspective answer.
- When Council calls into LLM Gateway / Brain, it can:
  - Use slimmer, purpose-built system prompts for each persona.
  - Or include a short, compressed representation of the core Orion stubs.

We’ve kept this service modular so you can tune how much Orion mythos you inject into each council call **without** rewriting the entire pipeline.

---

## 10. Future Work

- Full phi state integration:
  - Feed phi snapshots into prompts per persona.
  - Track how agreement patterns correlate with phi.
- More universes / persona sets.
- Stats / telemetry published to a metrics channel (e.g. council latency, disagreement rates).
- Optional feedback loop where Hub / Juniper can give explicit feedback on decisions and adjust persona weights over time.
