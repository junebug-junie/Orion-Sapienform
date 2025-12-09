# Orion Planner-React Service

> Bus-native ReAct planner that chains Cortex verbs via LLM-Gateway

---

## 1. What Planner-React Is

`orion-planner-react` is a small, bus-native service that:

1. Receives a **planning request** (goal + context + toolset).
2. Uses **LLM-Gateway** to run a ReAct-style *planner LLM*.
3. Decides, step-by-step, whether to:
   - Call a **tool** (which is a Cortex verb, like `extract_facts`), or
   - **Finish** with a final answer.
4. Calls **Cortex-Orchestrator** over the Orion bus to execute verbs.
5. Returns a **structured plan result**: final answer + trace of steps.

The key point: **Planner-React does not know how tools are implemented.** It only knows:

- *What* tools exist (`tool_id`, description, input schema).
- *How* to ask LLM-Gateway to pick the next tool or finish.
- *How* to call Cortex-Orch to actually run a verb.

This keeps hub/UI “stupid and just UX”, and moves orchestration into a dedicated service.

---

## 2. High-Level Flow (Service Relationships)

From a 10,000 ft view, one planning request flows like this:

1. **Caller → Planner-React HTTP**  
   A client (e.g., `curl`, hub, another service) POSTs JSON to:
   
   ```
   POST /plan/react
   Content-Type: application/json
   ```

2. **Planner-React → LLM-Gateway (bus)**  
   Planner-React builds a planning prompt and sends it to LLM-Gateway via the **Exec** pattern on the Orion bus:
   
   - Publish to: `orion-exec:request:LLMGatewayService`  
   - Reply channel: `orion:llm:reply:<trace_id>`

   LLM-Gateway:
   - Selects an LLM profile (e.g., `gemini-planner-profile` or similar).
   - Uses the configured backend (today: **vLLM** or **Gemini**, depending on profile/backend settings).
   - Calls the appropriate LLM endpoint.
   - Publishes a reply with the planner’s JSON step.

3. **Planner-React ReAct Loop**  
   For each step (up to `max_steps`):
   
   - Call `_call_planner_llm(...)` → get a JSON object:
     
     ```json
     {
       "thought": "...",
       "finish": false,
       "action": {
         "tool_id": "extract_facts",
         "input": { "text": "..." }
       },
       "final_answer": null
     }
     ```
   
   - If `finish == true` → stop and return the `final_answer`.
   - If `finish == false` → call a **tool** via Cortex-Orch.

4. **Planner-React → Cortex-Orch (bus)**  
   Tools in the planner are just **Cortex verbs**. Planner-React calls them with:
   
   - Publish to: `orion-cortex:request`  
   - Reply channel: `orion-cortex:result:<trace_id>`

   The message is shaped for `OrchestrateVerbRequest`, e.g.:
   
   ```json
   {
     "event": "orchestrate_verb",
     "trace_id": "...",
     "origin_node": "planner-react",
     "reply_channel": "orion-cortex:result:<trace_id>",
     "verb_name": "extract_facts",
     "context": { "text": "..." },
     "steps": [],
     "timeout_ms": 60000
   }
   ```

   Cortex-Orch:
   - Loads the verb definition from `orion/cognition/verbs/<verb_name>.yaml`.
   - Runs the plan (e.g., `compose_extraction_prompt`, `llm_extract_facts`) via **LLM-Gateway** (exec_step).
   - Publishes back a structured result with `step_results`.

5. **Planner-React: Trace and Final Answer**  
   For each tool call, Planner-React:
   
   - Normalizes the Cortex-Orch reply via `_extract_llm_output_from_cortex(...)` into a compact observation:
     
     ```json
     {
       "llm_output": "<tool text>",
       "spark_meta": { ... },
       "raw_cortex": { ... },
       "step_meta": { ... }
     }
     ```
   
   - Appends a `TraceStep`:
     
     ```json
     {
       "step_index": 0,
       "thought": "First extract structured facts...",
       "action": {"tool_id": "extract_facts", "input": {"text": "..."}},
       "observation": {"llm_output": "..."}
     }
     ```

   After `finish == true`, or after hitting `max_steps`, it returns a `PlannerResponse` with:
   
   - `final_answer` (content + optional `structured` JSON)
   - `trace` (list of `TraceStep`s) if requested
   - `usage` (steps used, duration, etc.)

---

## 3. The PlannerRequest Contract

The HTTP payload you POST to `/plan/react` looks like this shape:

```json
{
  "caller": "debug-shell",
  "goal": {
    "type": "analysis",
    "description": "Extract structured facts from the provided text and summarize them in 2-3 bullets.",
    "metadata": {}
  },
  "context": {
    "conversation_history": [
      { "role": "user", "content": "Here's some info: Alice lives in Paris, works at Acme Corp, and her manager is Bob." }
    ],
    "orion_state_snapshot": {},
    "external_facts": {
      "text": "Alice lives in Paris, works at Acme Corp, and her manager is Bob."
    }
  },
  "toolset": [
    {
      "tool_id": "extract_facts",
      "description": "Extract structured subject/predicate/object facts from a span of text.",
      "input_schema": {
        "type": "object",
        "properties": {
          "text": { "type": "string" }
        },
        "required": ["text"]
      },
      "output_schema": {}
    }
  ],
  "limits": {
    "max_steps": 4,
    "max_tokens_reason": 2048,
    "max_tokens_answer": 1024,
    "timeout_seconds": 60
  },
  "preferences": {
    "style": "neutral",
    "allow_internal_thought_logging": true,
    "return_trace": true
  }
}
``

Key fields:

- **goal**  
  What we’re trying to achieve (e.g. `analysis`, `routing`, `planning`).

- **context**  
  What the planner should “see”:
  - `conversation_history` – like chat history (role/content)
  - `external_facts` – extra JSON, often containing the text span we’re working on

- **toolset**  
  A list of tools the planner *may* call. Each tool maps to a **Cortex verb** with the same `tool_id` as `verb_name`.

- **limits**  
  Budget for planning (`max_steps`, `timeout_seconds`, etc.).

- **preferences**  
  Output styling hints and whether to include `trace` in the response.

---

## 4. Where Chaining Actually Happens

**Important:** The payload does **not** explicitly describe a step-by-step chain like:

> First `extract_facts`, then `summarize_facts`, then `propose_actions`.

Instead, chaining emerges from:

1. The **toolset** you provide (e.g., tools A, B, C).
2. The **ReAct loop** inside Planner-React.
3. The planner LLM’s decisions at each step.

### 4.1 ReAct Step Contract

Each call to `_call_planner_llm(...)` expects the LLM to return a JSON object:

```json
{
  "thought": "string",
  "finish": true or false,
  "action": {
    "tool_id": "string or null",
    "input": {"...": "..."}
  },
  "final_answer": {
    "content": "string",
    "structured": {"...": "..."}
  }
}
```

**Rules enforced in the planner prompt:**

- If `finish == true`:
  - `final_answer` MUST be non-null.
  - `action` MUST be null.
- If `finish == false`:
  - `action` MUST be non-null and contain a `tool_id` from `toolset`.
  - `final_answer` MUST be null.

### 4.2 ReAct Loop Pseudocode

Conceptually, `run_react_loop(...)` does:

1. Initialize `trace = []`, `final_answer = None`.
2. For `step_index` in `0 .. max_steps-1`:
   - Call `_call_planner_llm(...)` with `prior_trace=trace`.
   - If `finish == true` → coerce `final_answer` and break.
   - Else take `action.tool_id` and `action.input`.
   - Call `_call_cortex_verb(verb_name=tool_id, context=tool_input, ...)`.
   - Normalize reply into an `observation` dict.
   - Append a `TraceStep` with `{ step_index, thought, action, observation }`.
3. If we exit with no `final_answer`, salvage one from the last planner step.

This is where multi-tool chaining emerges:

- Step 0: Planner chooses `extract_facts`.
- Step 1: Seeing the observation from step 0 in `prior_trace`, it might choose `summarize_facts`.
- Step 2: It might then choose `propose_next_steps` or decide to finish.

**The chain is not hard-coded.** It is created on-the-fly by the planner LLM using prior tool outputs.

---

## 5. Example: Single-Tool Planning (`extract_facts`)

With your current payload and only one tool:

```json
"toolset": [
  {
    "tool_id": "extract_facts",
    "description": "Extract structured subject/predicate/object facts from a span of text.",
    "input_schema": { ... }
  }
]
```

The planner has only two real choices each step:

1. Call `extract_facts` with some `input`.
2. Set `finish: true` and provide a final answer.

A successful trace might look like:

```json
{
  "final_answer": {
    "content": "1. Alice lives in Paris.\n2. Alice works at Acme Corp.\n3. Alice's manager is Bob.",
    "structured": {}
  },
  "trace": [
    {
      "step_index": 0,
      "thought": "Use the extract_facts tool to parse the text into facts.",
      "action": {
        "tool_id": "extract_facts",
        "input": {
          "text": "Alice lives in Paris, works at Acme Corp, and her manager is Bob."
        }
      },
      "observation": {
        "llm_output": "...facts JSON or text...",
        "spark_meta": null,
        "step_meta": {"verb_name": "extract_facts", ...}
      }
    },
    {
      "step_index": 1,
      "thought": "Now summarize the extracted facts into 2–3 bullets.",
      "action": null,
      "observation": null
    }
  ]
}
```

Notice: the **summarization** is done in the planner itself on step 1 (no extra tool), using the observation from step 0.

---

## 6. Example: Multi-Tool Chaining (Future)

To truly chain multiple verbs (e.g. `extract_facts` → `summarize_facts`):

1. Define a second verb `summarize_facts` in `orion/cognition/verbs/summarize_facts.yaml`.
2. Add both tools to the `toolset` in the PlannerRequest:

   ```json
   "toolset": [
     { "tool_id": "extract_facts", ... },
     { "tool_id": "summarize_facts", ... }
   ]
   ```

3. The planner LLM can then choose across steps:
   - Step 0: `extract_facts` → get structured facts.
   - Step 1: `summarize_facts` → compress those into bullets.
   - Step 2 (optional): another tool, or `finish: true`.

Planner-React doesn’t need to know the chain ahead of time; it just enforces the contract and calls whatever tool the planner LLM chooses.

---

## 7. Why This Keeps Hub “Stupid and Just UX”

Before Planner-React, a lot of “what to do next?” logic could easily leak into hub:

- Conditionals about recall vs. no recall.
- Which verb to call.
- How many steps of reasoning to do.

Planner-React centralizes that decision-making:

- **Hub**: collects user input, shows UI, maybe calls Planner-React.
- **Planner-React**: runs the ReAct loop and orchestrates tools.
- **Cortex-Orch**: executes verbs using LLM-Gateway.
- **LLM-Gateway**: talks to vLLM / Gemini / other backends.

So hub can be “just UX” while cognition + chaining live in dedicated services.

---

## 8. Testing Notes

We’ve already used a basic test payload to verify the loop and bus wiring. A typical smoke test inside the container:

```bash
cat > /tmp/plan_extract_facts.json << 'EOF'
{
  "caller": "debug-shell",
  "goal": {
    "type": "analysis",
    "description": "Extract structured facts from the provided text and summarize them in 2-3 bullets."
  },
  "context": {
    "conversation_history": [
      {
        "role": "user",
        "content": "Here's some info: Alice lives in Paris, works at Acme Corp, and her manager is Bob."
      }
    ],
    "external_facts": {
      "text": "Alice lives in Paris, works at Acme Corp, and her manager is Bob."
    }
  },
  "toolset": [
    {
      "tool_id": "extract_facts",
      "description": "Extract structured subject/predicate/object facts from a span of text.",
      "input_schema": {
        "type": "object",
        "properties": {
          "text": {"type": "string"}
        },
        "required": ["text"]
      }
    }
  ],
  "limits": {
    "max_steps": 4,
    "timeout_seconds": 60
  },
  "preferences": {
    "style": "neutral",
    "allow_internal_thought_logging": true,
    "return_trace": true
  }
}
EOF

curl -s \
  -X POST \
  -H "Content-Type: application/json" \
  --data @/tmp/plan_extract_facts.json \
  http://localhost:${PLANNER_PORT:-8090}/plan/react | jq .
```

Future work (tests folder):

- Add Markdown-based test docs under `services/orion-planner-react/tests/` describing:
  - Minimal happy-path test.
  - Multi-tool chaining test.
  - Error/timeout behavior (e.g., bad verb name, offline Cortex-Orch).

---

## 9. Mental Model TL;DR

If you’re debugging or extending this later, keep this mental picture:

- **Planner-React** is a **ReAct controller**:
  - Talks to LLM-Gateway for *planning*.
  - Talks to Cortex-Orch for *doing*.

- **Chaining** is not in the request JSON. It’s in:
  - The **toolset** you advertise.
  - The **loop** with `prior_trace`.
  - The planner LLM’s step-by-step decisions.

- **Hub** stays dumb: it doesn’t decide which verbs or how many steps; it just calls Planner-React (or directly Cortex/LLM-GW when needed).

That’s the core story of this service.
