# LLM Services and Agentic Flow

This note summarizes how Orion routes language-model work across brain, agent, and council modes. It focuses on the bus-mediated hops, who owns each step, and how recall and planning interplay before the LLM gateway is invoked.

## Service Roles

| Service | Role | Notes |
| --- | --- | --- |
| Cortex-Orch | Validates client envelopes, builds a mode-specific plan, forwards to Cortex-Exec. | The only entry point for clients. |
| Cortex-Exec | Executes plans step-by-step, invoking workers over the bus and aggregating results. | Owns recall gating and step ordering. |
| PlannerReactService | Produces a short plan/intent for agent mode. | Runs before the agent chain. |
| AgentChainService | Executes the agentic step (tool use, ReAct reasoning) and produces the agent answer. | Second step in agent plans. |
| LLM Gateway | Performs direct chat/completion for brain mode, or tools inside the agent chain. | Receives `llm.chat.request`. |
| RecallService | Supplies retrieved context when enabled. | Skipped when recall is disabled. |
| CouncilService (stub) | Reserved for supervisor-style flows. | Not used by brain/agent today. |

## High-Level Flow (All Modes)

```mermaid
flowchart LR
    Client[["Client / Hub"]]
    subgraph Bus["Redis Bus"]
        Orch["Cortex-Orch"]
        Exec["Cortex-Exec"]
        Planner["PlannerReactService"]
        AgentChain["AgentChainService"]
        LLM["LLM Gateway"]
        Recall["RecallService"]
        Council["CouncilService - stub"]
    end

    Client -->|"cortex.orch.request"| Orch
    Orch -->|"cortex.exec.request"| Exec

    Exec -->|"llm.chat.request"| LLM

    Exec -->|"agent.planner.request"| Planner
    Planner -->|"agent.planner.result"| Exec
    Exec -->|"agent.chain.request"| AgentChain
    AgentChain -->|"agent.chain.result"| Exec

    Exec -->|"council.request"| Council

    Exec -->|"recall.query.request"| Recall
    Recall -->|"recall.query.result"| Exec

    Exec -->|"cortex.exec.result"| Orch
    Orch -->|"cortex.orch.result"| Client
```

## Mode-Specific Sequences

### Brain Mode (single LLM step)

```mermaid
sequenceDiagram
    participant Client
    participant Orch as Cortex-Orch
    participant Exec as Cortex-Exec
    participant Recall as RecallService
    participant LLM as LLM Gateway

    Client->>Orch: cortex.orch.request (mode=brain, verb=chat_general)
    Orch->>Exec: cortex.exec.request (plan: llm_chat_general)
    opt recall.enabled
        Exec->>Recall: recall.query.request
        Recall-->>Exec: recall.query.result
    end
    Exec->>LLM: llm.chat.request
    LLM-->>Exec: llm.chat.result
    Exec-->>Orch: cortex.exec.result (final_text, steps)
    Orch-->>Client: cortex.orch.result
```

### Agent Mode (planner + agent chain)

```mermaid
sequenceDiagram
    participant Client
    participant Orch as Cortex-Orch
    participant Exec as Cortex-Exec
    participant Planner as PlannerReactService
    participant Agent as AgentChainService
    participant Recall as RecallService

    Client->>Orch: cortex.orch.request (mode=agent, packs=[...])
    Orch->>Exec: cortex.exec.request (plan: planner_react -> agent_chain)
    opt recall.enabled
        Exec->>Recall: recall.query.request
        Recall-->>Exec: recall.query.result
    end
    Exec->>Planner: agent.planner.request
    Planner-->>Exec: agent.planner.result (intent/plan)
    Exec->>Agent: agent.chain.request (messages + planner guidance)
    Agent-->>Exec: agent.chain.result (final agent answer)
    Exec-->>Orch: cortex.exec.result (steps, final_text)
    Orch-->>Client: cortex.orch.result
```

### Council Mode (stubbed)

```mermaid
sequenceDiagram
    participant Client
    participant Orch as Cortex-Orch
    participant Exec as Cortex-Exec
    participant Council as CouncilService

    Client->>Orch: cortex.orch.request (mode=council)
    Orch->>Exec: cortex.exec.request (plan: council)
    Exec->>Council: council.request
    Council-->>Exec: council.result (currently stubbed error)
    Exec-->>Orch: cortex.exec.result
    Orch-->>Client: cortex.orch.result
```

## How the Chain Hangs Together

* **Orch builds the plan** based on `mode` + `verb` + `packs`; it never calls workers directly.
* **Exec enforces ordering** and passes correlation IDs, reply channels, and recall directives to every worker.
* **Recall is optional**: when disabled, exec skips the recall hop entirely; when required, failures surface as structured errors instead of silent fallback.
* **Planner → AgentChain** is the agentic path: PlannerReact produces intent, AgentChain executes it (tool use, ReAct reasoning), and only AgentChain’s result becomes the agent reply.
* **LLM Gateway is shared**: brain mode calls it directly; agent mode may call it indirectly within AgentChain/tool steps.
* **Council remains isolated** behind its own `council.request`/`council.result` channel, so future supervisor flows do not contaminate brain/agent traffic.

## Common Observability Hooks

* Every hop logs the intake channel, reply channel, correlation ID, and elapsed time.
* Exec surfaces per-step `logs` (RPC emits/returns) and `latency_ms` in `StepExecutionResult`.
* Tap the bus with `python scripts/bus_harness.py tap --bus-url redis://...` to watch:
  * `cortex.orch.request`, `cortex.exec.request`
  * `recall.query.request` (if enabled)
  * `agent.planner.request`, `agent.chain.request`
  * `llm.chat.request`
