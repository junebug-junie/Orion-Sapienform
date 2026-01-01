# LLM Services and Agentic Flow

This note summarizes how Orion routes language-model work across brain, agent, and council modes. It focuses on the bus-mediated hops, who owns each step, and how recall and planning interplay before the LLM gateway is invoked.

## Service Roles

| Service | Role | Notes |
| --- | --- | --- |
| Cortex-Orch | Validates client envelopes, builds a mode-specific plan, forwards to Cortex-Exec. | The only entry point for clients. |
| Cortex-Exec | Executes plans step-by-step, invoking workers over the bus and aggregating results. | Owns recall gating and step ordering. |
| PlannerReactService | Produces a short plan/intent for agent mode. | Runs before the agent chain on `orion-exec:request:PlannerReactService`. |
| AgentChainService | Executes the agentic step (tool use, ReAct reasoning) and produces the agent answer. | Listens on `orion-exec:request:AgentChainService`. |
| LLM Gateway | Performs direct chat/completion for brain mode and for planner prompts. | Receives `orion-exec:request:LLMGatewayService`. |
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

    Client -->|"orion-cortex:request\ncortex.orch.request"| Orch
    Orch -->|"orion-cortex-exec:request\ncortex.exec.request"| Exec

    Exec -->|"orion-exec:request:LLMGatewayService\nllm.chat.request"| LLM

    Exec -->|"orion-exec:request:PlannerReactService\nagent.planner.request"| Planner
    Planner -->|"reply_to from Exec\nagent.planner.result"| Exec
    Exec -->|"orion-exec:request:AgentChainService\nagent.chain.request"| AgentChain
    AgentChain -->|"reply_to from Exec\nagent.chain.result"| Exec

    Exec -->|"orion:agent-council:intake\ncouncil.request"| Council

    Exec -->|"orion-exec:request:RecallService\nrecall.query.request"| Recall
    Recall -->|"reply_to from Exec\nrecall.query.result"| Exec

    Exec -->|"reply_to from Orch\ncortex.exec.result"| Orch
    Orch -->|"reply_to from Client\ncortex.orch.result"| Client
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

    Client->>Orch: orion-cortex:request (cortex.orch.request)
    Orch->>Exec: orion-cortex-exec:request (plan: llm_chat_general, reply=orion-exec:result:<uuid>)
    opt recall.enabled
        Exec->>Recall: orion-exec:request:RecallService (recall.query.request)
        Recall-->>Exec: recall.query.result on orion-exec:result:RecallService:<uuid>
    end
    Exec->>LLM: orion-exec:request:LLMGatewayService (llm.chat.request, reply=orion-exec:result:LLMGatewayService:<uuid>)
    LLM-->>Exec: llm.chat.result
    Exec-->>Orch: cortex.exec.result on orion-exec:result:<uuid>
    Orch-->>Client: cortex.orch.result on orion-cortex:result:<uuid>
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

    Client->>Orch: orion-cortex:request (mode=agent, packs=[...])
    Orch->>Exec: orion-cortex-exec:request (plan: planner_react -> agent_chain, reply=orion-exec:result:<uuid>)
    opt recall.enabled
        Exec->>Recall: orion-exec:request:RecallService
        Recall-->>Exec: recall.query.result on orion-exec:result:RecallService:<uuid>
    end
    Exec->>Planner: agent.planner.request on orion-exec:request:PlannerReactService
    Planner-->>Exec: agent.planner.result on orion-exec:result:PlannerReactService:<uuid>
    Exec->>Agent: agent.chain.request on orion-exec:request:AgentChainService (messages + planner guidance)
    Agent-->>Exec: agent.chain.result on orion-exec:result:AgentChainService:<uuid>
    Exec-->>Orch: cortex.exec.result (steps, final_text) on orion-exec:result:<uuid>
    Orch-->>Client: cortex.orch.result
```

### Council Mode (stubbed)

```mermaid
sequenceDiagram
    participant Client
    participant Orch as Cortex-Orch
    participant Exec as Cortex-Exec
    participant Council as CouncilService

    Client->>Orch: orion-cortex:request (mode=council)
    Orch->>Exec: orion-cortex-exec:request (plan: council, reply=orion-exec:result:<uuid>)
    Exec->>Council: council.request on orion:agent-council:intake (reply=orion:council:reply:<uuid>)
    Council-->>Exec: council.result (currently stubbed error)
    Exec-->>Orch: cortex.exec.result on orion-exec:result:<uuid>
    Orch-->>Client: cortex.orch.result
```

## How the Chain Hangs Together

* **Orch builds the plan** based on `mode` + `verb` + `packs`; it never calls workers directly.
* **Exec enforces ordering** and passes correlation IDs, reply channels, and recall directives to every worker. Reply channels are always explicit (e.g., `orion-exec:result:LLMGatewayService:<uuid>`).
* **Recall is optional**: when disabled, exec skips the recall hop entirely; when required, failures surface as structured errors instead of silent fallback.
* **Planner → AgentChain** is the agentic path: PlannerReact produces intent, AgentChain executes it (tool use, ReAct reasoning), and only AgentChain’s result becomes the agent reply.
* **LLM Gateway is shared**: brain mode and PlannerReact both call it (`orion-exec:request:LLMGatewayService`); agent mode may also hit it indirectly inside AgentChain tool calls.
* **Council remains isolated** behind `orion:agent-council:intake`/`orion:council:reply`, so future supervisor flows do not contaminate brain/agent traffic.

## Channel Defaults (from service envs)

* **Client ↔ Orch:** `orion-cortex:request` / `orion-cortex:result`
* **Orch ↔ Exec:** `orion-cortex-exec:request` / `orion-exec:result:<uuid>`
* **Exec ↔ LLM Gateway:** `orion-exec:request:LLMGatewayService` / `orion-exec:result:LLMGatewayService:<uuid>`
* **Exec ↔ Recall:** `orion-exec:request:RecallService` / `orion-exec:result:RecallService:<uuid>`
* **Exec ↔ Planner:** `orion-exec:request:PlannerReactService` / `orion-exec:result:PlannerReactService:<uuid>`
* **Exec ↔ AgentChain:** `orion-exec:request:AgentChainService` / `orion-exec:result:AgentChainService:<uuid>`
* **Planner ↔ LLM Gateway:** `orion-exec:request:LLMGatewayService` / `orion:llm:reply:<uuid>` (planner-driven)
* **Exec ↔ Council (stub):** `orion:agent-council:intake` / `orion:council:reply:<uuid>`

## Common Observability Hooks

* Every hop logs the intake channel, reply channel, correlation ID, and elapsed time.
* Exec surfaces per-step `logs` (RPC emits/returns) and `latency_ms` in `StepExecutionResult`.
* Tap the bus with `python scripts/bus_harness.py tap --bus-url redis://...` to watch:
  * `cortex.orch.request`, `cortex.exec.request`
  * `recall.query.request` (if enabled)
  * `agent.planner.request`, `agent.chain.request`
  * `llm.chat.request`
