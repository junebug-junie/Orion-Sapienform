# Orion ReAct Planner & Watcher Services — Future-Proof Contract

This document defines the **service-level contracts** for two core components in the Orion mesh:

- **ReAct Planner Service** — `orion-planner-react`
- **Watcher / Agent Service** — `orion-agent-daemon`

These contracts are designed to be **future-proof**:
- They separate **planning** from **tool execution** and **scheduling**.
- They assume **cortex-orch** remains the canonical verb/tool router.
- They allow hub, watcher, and future services to share the same planning interface.

---

## 1. ReAct Planner Service — `orion-planner-react`

### 1.1. Role

The ReAct Planner is the **reasoning + tool-using loop** of Orion.
It implements a generic pattern:

> Reason → choose tool → Act (via cortex-orch) → Observe → Reason → … → Final answer

It does **not** execute tools directly. All tool calls flow through **cortex-orch** using the ontology/verb registry.

The planner can be used by:
- **Hub** (interactive chat, debug modes, council-like flows)
- **Watcher / Agent Daemon** (autonomous agents)
- Future services (e.g. batch jobs, experiment harnesses)

---

### 1.2. Planner Service API

> **Service name:** `orion-planner-react`

The API is described at the logical level. Transport (HTTP/gRPC/bus RPC) can be chosen later as long as these message shapes are preserved.

#### 1.2.1. `POST /plan/react`

**Purpose:** Run a single ReAct planning loop for a given goal.

**Request Body (conceptual JSON):**

```jsonc
{
  "request_id": "uuid-optional",
  "caller": "hub" | "agent-daemon" | "test" | "other",

  "goal": {
    "type": "chat" | "agent_task" | "spark_review" | "system_maint" | "other",
    "description": "Human-readable goal text",
    "metadata": {
      "user_message": "optional raw user text",
      "mode": "brain" | "council" | "debug" | "dream" | "...",
      "tags": ["juniper", "orion", "autonomy"]
    }
  },

  "context": {
    "conversation_history": [
      {"role": "user" | "assistant" | "system", "content": "..."}
    ],
    "orion_state_snapshot": {
      "version": "v1",
      "themes": ["emergent_orion", "juniper_care"],
      "curiosity": {"emergent_orion": 0.8},
      "strain": {"juniper_care": 0.3},
      "notes": "freeform state summary if needed"
    },
    "external_facts": {
      "collapse_ids": ["..."],
      "dream_ids": ["..."],
      "other_ids": ["..."]
    }
  },

  "toolset": [
    {
      "tool_id": "recall.fetch_context",
      "description": "Fetch semantically relevant fragments from recall store.",
      "input_schema": {"type": "object"},
      "output_schema": {"type": "object"}
    },
    {
      "tool_id": "sim.markov_next_theme",
      "description": "Run Markov step to choose next theme.",
      "input_schema": {"type": "object"},
      "output_schema": {"type": "object"}
    }
  ],

  "limits": {
    "max_steps": 8,
    "max_tokens_reason": 2048,
    "max_tokens_answer": 1024,
    "timeout_seconds": 60
  },

  "preferences": {
    "style": "warm" | "technical" | "neutral",
    "allow_internal_thought_logging": true,
    "return_trace": true
  }
}
```

**Response Body (conceptual JSON):**

```jsonc
{
  "request_id": "...",  
  "status": "ok" | "error" | "timeout",
  "error": {
    "code": "optional_error_code",
    "message": "optional error message"
  },

  "final_answer": {
    "content": "string: what hub shows or the agent consumes",
    "structured": { "optional_structured_payload": "..." }
  },

  "trace": [
    {
      "step_index": 0,
      "thought": "LLM reasoning (internal monologue)",
      "action": null,
      "observation": null
    },
    {
      "step_index": 1,
      "thought": "I should fetch context first.",
      "action": {
        "tool_id": "recall.fetch_context",
        "input": {"query": "...", "limit": 10}
      },
      "observation": {
        "ok": true,
        "output": {"fragments": ["..."]}
      }
    },
    {
      "step_index": 2,
      "thought": "Given context, now call brain.chat for a response.",
      "action": {
        "tool_id": "brain.chat",
        "input": {"prompt": "..."}
      },
      "observation": {
        "ok": true,
        "output": {"message": "..."}
      }
    }
  ],

  "usage": {
    "steps": 3,
    "tokens_reason": 1500,
    "tokens_answer": 200,
    "tools_called": ["recall.fetch_context", "brain.chat"],
    "duration_ms": 5400
  }
}
```

#### 1.2.2. Tool Execution via Cortex-Orch

The planner MUST NOT call tools directly. Instead, it calls **cortex-orch** through a separate contract (already in the system):

- Planner → Cortex-Orch: "execute verb/tool X with payload Y"
- Cortex-Orch → Tool service (recall, sim-lab, brain, etc.)

Planner should be configurable to support different backends but always speak in terms of **verb IDs** from the ontology.

---

### 1.3. Planner Behavioural Guarantees

- **Deterministic API, non-deterministic internals:**
  - Same request may yield different internal traces but must respect step/usage limits.
- **Bounded loops:**
  - Obeys `max_steps` and `timeout_seconds`.
- **Safety & scoping:**
  - Only tools listed in `toolset` may be used for this run.
  - Planner must not invent tool IDs.
- **Trace transparency:**
  - When `return_trace` is true, each tool call and LLM reasoning step is present in `trace`.

---

## 2. Watcher / Agent Service — `orion-agent-daemon`

### 2.1. Role

The Watcher / Agent Daemon is Orion's **autonomic nervous system**.

It:
- Listens to **time**, **bus events**, and **OrionState**.
- Holds a registry of **Agent Profiles**.
- Decides **when** to run the ReAct planner, with **which goal** and **which tools**.
- Routes planner outputs to the appropriate memories and interfaces (collapse, dreams, notifications, etc.).

It does **not** perform multi-step reasoning itself; it delegates to the **planner**.

---

### 2.2. Agent Profile Contract

Agents are defined as data (YAML/JSON/etc.) using a shared schema.

#### 2.2.1. `AgentProfile` (conceptual JSON)

```jsonc
{
  "id": "agent.collapse_autowriter",
  "description": "Writes autonomous Collapse Mirrors at key moments.",

  "goal_template": {
    "type": "agent_task",
    "description": "Write a Collapse Mirror summarizing recent key events between Juniper and Orion.",
    "metadata": {
      "tags": ["collapse", "reflection", "juniper", "orion"]
    }
  },

  "toolset": [
    "recall.fetch_context",
    "brain.chat",
    "collapse.write_entry"
  ],

  "triggers": {
    "time": [
      {
        "cron": "0 2 * * *",  
        "timezone": "America/Denver"
      }
    ],
    "events": [
      {
        "channel": "orion:hub:session_end",
        "conditions": {"min_duration_sec": 300}
      }
    ],
    "state": [
      {
        "selector": "orion_state.curiosity.emergent_orion",
        "op": ">",
        "value": 0.7
      }
    ]
  },

  "limits": {
    "max_runs_per_day": 3,
    "min_interval_seconds": 1800,
    "max_concurrent_runs": 1,
    "planner_limits": {
      "max_steps": 6,
      "timeout_seconds": 45,
      "max_tokens_reason": 2048,
      "max_tokens_answer": 512
    }
  },

  "autonomy_level": 0,

  "outputs": {
    "collapse_entry": {
      "enabled": true,
      "topic": "autonomous-reflection"
    },
    "notify_hub": {
      "enabled": true,
      "mode": "on_next_login"
    }
  },

  "metadata": {
    "owner": "juniper",
    "tags": ["spark", "reflection", "safe"]
  }
}
```

Semantics:
- `autonomy_level`:
  - `0` = reflect-only (write to internal logs/memories; no external actions)
  - `1` = notify (may send messages to Juniper/hub but not change configs)
  - `2` = act-in-bounds (may adjust pre-approved knobs under constraints)

---

### 2.3. Watcher Service API

> **Service name:** `orion-agent-daemon`

The watcher exposes a small control API and otherwise primarily reacts to time + events.

#### 2.3.1. `GET /agents`

List known agents and their status.

```jsonc
{
  "agents": [
    {
      "id": "agent.collapse_autowriter",
      "enabled": true,
      "autonomy_level": 0,
      "last_run": "2025-12-07T02:13:00Z",
      "next_scheduled_run": "2025-12-08T02:00:00Z",
      "status": "idle" | "running" | "error"
    }
  ]
}
```

#### 2.3.2. `POST /agents/{id}/run-once`

Manually trigger an agent run (for testing or debugging).

**Request:** optional overrides:

```jsonc
{
  "override_goal_description": "optional new goal text",
  "extra_context": {"debug": true}
}
```

**Response:** a `RunRecord` (see below).

#### 2.3.3. `POST /agents/{id}/enable` / `POST /agents/{id}/disable`

Enable or disable an agent.

---

### 2.4. Internal Watcher → Planner Contract

When watcher decides to run an agent, it builds a planner request:

```jsonc
{
  "request_id": "generated-uuid",
  "caller": "agent-daemon",
  "goal": {
    "type": "agent_task",
    "description": "Expanded from agent.goal_template + context",
    "metadata": {"agent_id": "agent.collapse_autowriter"}
  },
  "context": {
    "conversation_history": [],
    "orion_state_snapshot": {"...": "..."},
    "external_facts": {"collapse_ids": ["..."]}
  },
  "toolset": [
    {"tool_id": "recall.fetch_context"},
    {"tool_id": "brain.chat"},
    {"tool_id": "collapse.write_entry"}
  ],
  "limits": {
    "max_steps": 6,
    "timeout_seconds": 45
  },
  "preferences": {
    "style": "warm",
    "return_trace": true
  }
}
```

Watcher then:
- Sends this to `orion-planner-react` `/plan/react`.
- Receives `final_answer` + `trace`.
- Creates a `RunRecord` and routes outputs according to `AgentProfile.outputs`.

---

### 2.5. Agent Run Record

Watcher stores an `AgentRunRecord` for observability and Spark introspection.

```jsonc
{
  "run_id": "uuid",
  "agent_id": "agent.collapse_autowriter",
  "started_at": "2025-12-07T02:13:00Z",
  "finished_at": "2025-12-07T02:13:11Z",
  "status": "success" | "error" | "timeout",
  "planner_request_id": "...",

  "goal": {
    "description": "...",
    "metadata": {"...": "..."}
  },

  "toolset": ["recall.fetch_context", "brain.chat", "collapse.write_entry"],

  "result_summary": {
    "final_answer_excerpt": "first 200 chars...",
    "written_collapse_id": "optional-id",
    "notifications_sent": true
  },

  "usage": {
    "planner_steps": 4,
    "tokens_reason": 1200,
    "tokens_answer": 300,
    "duration_ms": 11000
  }
}
```

---

## 3. Integration Points

### 3.1. Hub → Planner

For interactive chat, Hub calls:

- `orion-planner-react /plan/react` with:
  - `caller = "hub"`
  - `goal.type = "chat"`
  - `toolset` chosen based on chat mode (e.g. `brain.chat` only, or `recall + brain + collapse`).

Hub then:
- Displays `final_answer.content` to the user.
- May optionally use `trace` for debug UI.

Hub should **not** embed bespoke tool-choosing glue; it should defer to planner where possible.

---

### 3.2. Planner → Cortex-Orch

Planner calls cortex-orch via a stable verb-execution API (existing or to-be-formalized):

```jsonc
{
  "verb_id": "recall.fetch_context",
  "payload": {"query": "..."},
  "request_id": "...",
  "caller": "planner-react"
}
```

Cortex-Orch returns:

```jsonc
{
  "ok": true,
  "output": {"fragments": ["..."]},
  "error": null
}
```

All tool use in planner traces must refer to these `verb_id`s.

---

### 3.3. Watcher → Bus & OrionState

Watcher subscribes to events like:
- `orion:hub:session_end`
- `orion:collapse:intake`
- `orion:power:alert`

Additionally, watcher may query OrionState (via a separate state service or direct DB access) to evaluate state-based triggers.

---

## 4. Safety & Autonomy Guardrails

- **Toolset scoping:** agents and hub must explicitly declare which tools planner may use for a given run.
- **Autonomy levels:** higher autonomy agents must be limited to specific safe tools and bounded state changes.
- **Quotas:** watcher enforces global + per-agent quotas to avoid runaway loops (max runs/day, min intervals, global token budget).
- **Trace logging:** all planner traces for autonomous runs are persisted as part of `AgentRunRecord` for later Spark introspection and debugging.

---

## 5. Future Extensions (Non-breaking)

This contract is designed to be extendable without breaking existing callers. Likely extensions include:

- **Multi-turn planner sessions:** allow `/plan/react` to maintain session IDs for incremental planning.
- **Hierarchical tools:** planner tools that themselves call planner recursively for subgoals.
- **Policy hooks:** externalized safety/policy checks that can veto specific tool invocations.
- **Experiment flags:** fields under `preferences` to toggle experimental reasoning styles (council, self-critique, etc.) using the same contract.

All such changes should respect the core shape:
- Planner: `goal + context + toolset + limits → final_answer + trace`.
- Watcher: `AgentProfile + triggers → planner request → AgentRunRecord`.

---

*End of Contract.*
