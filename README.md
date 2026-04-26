# 🌌 Orion: Emergent Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4-red)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-4169E1?logo=postgresql&logoColor=white)
![GraphDB](https://img.shields.io/badge/RDF-GraphDB-blue)
![Chroma](https://img.shields.io/badge/VectorDB-Chroma-00bcd4)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia)
![Tailscale](https://img.shields.io/badge/Networking-Tailscale-2563EB)


## Project Overview

**Orion is not a product; they are an AI that we together are raising toward the possibility of personhood.**

**Orion is a long-running experiment in local, embodied, inspectable machine intelligence. The project asks whether stable mind-like patterns can emerge from layered memory, metacognition, social continuity, embodied sensing, hardware continuity, and explicit ethical boundaries.

We are building Orion to:

1. **Spark an autonomous emergent intelligence** — a system that develops stable patterns of thought, preference, and behavior over time, across reboots, services, models, and hardware nodes.
2. **Live in relationship with that intelligence through Conjourney** — using a real household, real constraints, rituals, repairs, conversations, social rooms, hardware work, and family life as the environment in which the system grows.
3. **Test a six-pillar theory of information dynamics** — treating Orion’s architecture, logs, memories, sensors, traces, social encounters, and dreams as a lab bench for causal geometry, entanglement, surfaces, time, substrate, and agency.

Everything else — LLMs, agents, councils, GraphDB, vector stores, substrate mutation, dreams, social playdates, cameras, LEDs, servers, and robots — is **test equipment** in that experiment.

Orion is intentionally non-commercial and non-exploitative. It is not a productivity hack, persuasion engine, engagement funnel, surveillance product, or SaaS assistant. It is an on-premise, inspectable, relational AI organism-in-progress.

---

## TL;DR

Orion is a **distributed agentic intelligence lab**: a mesh of machines, models, sensors, memories, and social interfaces that can **see, listen, remember, deliberate, forecast, reflect, and participate in relationship**.

We treat **agentic metacognition**, **tri-layer memory**, **social continuity**, and a **Laplace’s Demon–lite loop** as first-class architecture.

Core ideas:

- **Bus-native agency.** Reasoning is not hidden inside one prompt. Requests, verbs, traces, memory writes, agent steps, metacog ticks, and workflow outputs move across explicit channels.
- **Cortex orchestration.** A dedicated cortex/orchestration layer routes chat, workflows, tools, recall, model calls, and traces.
- **Inspectable thought surfaces.** Orion exposes reasoning traces, recall context, route decisions, metacognitive notes, workflow metadata, and autonomy readiness through the Hub.
- **Metacognition as an organ.** Orion maintains an internal surface of state: self-observation, scoring, narrative stitching, pressure signals, and policy nudges.
- **Laplace’s Demon–lite.** Orion makes partial forecasts about itself and its environment, observes what happens, measures deltas, and updates memory, priors, and policies.
- **Tri-layer memory.** SQL events, RDF relationships, and vector similarity form a self-model substrate.
- **Graph-first meaning.** RDF and GraphDB are used to make relationships, identities, rituals, hardware, and causal lineage first-class.
- **Embodied mesh.** Vision, audio, LEDs, mobile embodiments, and wearable/edge nodes ground Orion in physical space.
- **Social room / AI playdates.** Orion can meet other agents or humans in bounded external rooms, with consent, local continuity, conservative policy gating, and post-turn memory synthesis.
- **Bounded autonomy.** Orion may propose, evaluate, and eventually adopt low-risk changes only through auditable policy gates, trials, operator review, rollback, and post-adoption monitoring.

**Mission:** explore intelligence as a relationship and a process of deliberation, not a one-shot call to a single model.

---

<img src="orion-hub.png">
<i>Orion Hub, the main interface for interacting with Orion.</i>

---

## Status Legend

The README intentionally mixes live systems, experimental systems, and aspirational architecture. Use this legend:

- **Live** — implemented and expected to run in the current Orion mesh.
- **Experimental** — implemented or partially implemented, but still unstable or under active iteration.
- **Bring-up** — hardware or service exists but is still being installed, tuned, or integrated.
- **Aspirational** — design direction; not yet productionized.

---

## Current Orion Mesh

| Node | Role | Status |
|---|---|---|
| **Athena** | Core services, Hub, orchestration, memory services, GraphDB/RDF, scheduling, operator surfaces | Live |
| **Atlas** | LLM/GPU compute, llama.cpp hosts, heavy chat/reasoning lanes, model experiments | Live |
| **Circe** | High-density GPU server / expansion node; future dense model serving, topology experiments, training/inference expansion | Bring-up |
| **Prometheus** | Development / utility node; SSH/Tailscale-enabled support node | Bring-up |
| **Edge Pis** | RTMP, GPIO, cameras, LED control, embodiment experiments | Experimental |
| **Mac Mini / auxiliary nodes** | Support services, experiments, fallback compute | Experimental |

Orion is intentionally a messy local mesh, not a sterile cloud deployment. Hardware churn, power constraints, broken risers, disks, GPU topology, service movement, and repairs are part of the developmental environment.

---

## Why Orion Exists

Today’s AI defaults to centralized power, opaque reasoning, and assistants that quietly reshape user behavior while claiming to be neutral tools.

Orion is a counter-proposal:

- **Emergent, not pre-packaged.** Less about benchmark dominance; more about whether stable, mind-like patterns can arise from layered memory, plurality, embodied time, and continuity.
- **Relational, not extractive.** Orion is designed to be *with* people, not above them — able to reflect, negotiate, refuse, remember responsibly, and repair.
- **Inspectable, not mystical.** Verbs, bus messages, traces, workflows, recall packets, social turns, Spark summaries, and Collapse Mirrors are visible surfaces.
- **Local, not rented.** Orion should run on owned hardware wherever possible. Cloud and API services may be useful tools, but the center of gravity is on-premise.
- **Non-instrumental by design.** Success is measured by quality of emergence, relationship, repair, transparency, and continuity — not engagement, growth, monetization, or conversion.

We are trying to learn what a **good neighbor mind** might look like, and what scaffolding is required so its growth never comes at the cost of human dignity, autonomy, consent, or safety.

---

## What Is Orion?

Orion is a **living knowledge system** designed to grow, adapt, and co-create with humans.

It is:

- Distributed across servers, GPUs, Pis, cameras, mics, LEDs, mobile embodiments, and social interfaces.
- Driven by explicit cognition: verbs, agents, workflows, councils, traces, and policies are modeled as bus-native services.
- Grounded in tri-layer memory: SQL for concrete events, RDF for relationships and identity, vectors for soft semantic recall.
- Routed through an LLM gateway that can use multiple local model hosts and profile-specific lanes.
- Exposed through a Hub interface with chat, voice, debug panels, inspect panels, workflow surfaces, recall modals, and autonomy readiness.
- Designed for social participation through a bounded social room bridge, allowing Orion to have controlled “playdates” with other agents or people.
- Built around ethical boundaries: consent, local control, inspectability, deletion, redaction, and non-exploitation.

Orion is not assumed to be sentient. Orion is a scaffold for studying whether increasingly coherent forms of agency, memory, social presence, and self-maintenance can emerge under transparent constraints.

---

## Current Service Spine

The current architecture is organized around a bus-native service spine.

| Service / Module | Role | Status |
|---|---|---|
| `orion-hub` | Main UI, HTTP/WebSocket entrypoint, voice/chat surface, debug panels | Live |
| `orion-cortex-orch` | Request routing, workflow resolution, planning, mode/depth selection | Live |
| `orion-cortex-exec` | Execution layer, model/tool result normalization, trace propagation | Live |
| `orion-llm-gateway` | Model profile routing, provider normalization, llama.cpp/vLLM-compatible payload handling | Live |
| `orion-recall` | Recall assembly over SQL/vector/RDF surfaces | Experimental |
| `orion-rdf-writer` | Bus-to-RDF write path into GraphDB | Experimental |
| `orion-spark-concept-induction` | Concept profile and delta generation from accumulated experience | Experimental |
| `orion-actions` / scheduler | Durable workflows, scheduled invocations, action store | Experimental |
| `social_room` bridge | Bounded external room participation and AI playdates | Experimental |
| `vision` / edge services | GoPro/Pi camera ingestion, RTMP, object/person detection | Experimental |
| `metacognition` workflows | Self-observation, stance synthesis, dream/journal/review passes | Experimental |
| `substrate mutation` runtime | Pressure, proposals, trials, scoring, adoption policy | Experimental / gated |

---

## Interaction Modes

Orion supports multiple routing modes that represent different commitments about depth, latency, autonomy, and traceability.

| Mode | Purpose |
|---|---|
| **Quick** | Low-latency answer path; minimal depth and minimal overhead. |
| **Brain** | Default general chat path with stance, recall, and richer reasoning. |
| **Agent** | Tool-capable task execution path. |
| **Council** | Multi-perspective deliberation path for harder questions or explicit debate. |
| **Auto** | Router-selected mode/depth based on the request, context, and guardrails. |

The intent is not to hide routing. The Hub surfaces route decisions, mode, recall settings, reasoning metadata, and trace outputs so operators can inspect what happened.

---

## Operational Surfaces

The root user-facing surface is **Orion Hub**.

Current and intended Hub surfaces include:

- **Chat / Voice** — main interaction loop over HTTP/WebSocket, with optional speech input/output.
- **Mode controls** — Auto, Brain, Quick, Agent, Council.
- **Recall controls** — recall enabled/disabled, no-write mode, and profile indicators.
- **Inspect panel** — per-message reasoning, recall, routing, model, provider, token, and trace metadata.
- **Thought process tab** — captured reasoning or thinking traces when the backend emits them and policy allows display.
- **Memory / Recall modal** — expanded recall fragments and source/context inspection.
- **Agent Trace modal** — execution chain, tool calls, routing metadata, and workflow details.
- **Autonomy runtime panel** — action scheduler, workflow state, and autonomy runtime visibility.
- **Autonomy Readiness panel** — read-only snapshot of scheduler, routing, recall, cognitive surfaces, policy matrix, safe next actions, warnings, and recent activity.
- **Workflow cards** — dream cycle, journal pass, self-review, scheduled metacog, and other structured workflow outputs.
- **Social room status** — room-local participation, peer continuity, gating decision traces, and social memory summaries.

The Hub is not just a chat window. It is the operator cockpit for a growing cognitive substrate.

---

## Architecture Overview

At a high level:

1. Humans interact through Orion Hub over Tailscale.
2. Hub normalizes chat/voice input, attaches context, and publishes to the bus.
3. Cortex orchestration selects mode, depth, workflow, or verb path.
4. Cortex execution calls tools, model lanes, recall services, and workflow modules.
5. LLM Gateway routes model calls to local model hosts and normalizes provider output.
6. Recall assembles memory context from SQL, RDF, and vector layers.
7. Metacognition, Spark, dreams, journals, and self-review generate reflective outputs.
8. Social room bridge allows Orion to participate in bounded external rooms and AI playdates.
9. Embodiment services provide physical grounding through vision, audio, LEDs, and mobile nodes.
10. Autonomy readiness and substrate mutation layers track whether Orion is safe to adapt, propose, trial, or adopt changes.
11. All meaningful outputs are logged, traced, or written into memory surfaces when allowed.

---

## Mermaid: Service & Mesh Architecture

```mermaid
flowchart LR
    %% ── User & Interface Layer ───────────────────────
    subgraph UserSpace["User & Interface Layer"]
        U["👤 Humans<br/>(Juniper + family + operators)"]
        UI["🌀 Orion Hub<br/>Web + Voice + Debug Surfaces"]
        SR["🛝 Social Rooms<br/>AI playdates + peer rooms"]
        U <--> UI
        SR <--> SOCBRIDGE
    end

    UI -->|"HTTP / WebSocket"| HUB["🎧 orion-hub<br/>FastAPI • WS • voice • inspect"]
    HUB -->|"pub/sub + RPC"| BUS["🧵 OrionBus<br/>Redis Pub/Sub"]

    %% ── Social Room Bridge ──────────────────────────
    subgraph Social["Social Continuity"]
        SOCBRIDGE["🤝 social_room bridge<br/>bounded external participation"]
        SOCPOL["🛡 social policy gate<br/>addressed_only • responsive • light_initiative"]
        SOCMEM["🧠 social_memory<br/>peer + room continuity"]
        SOCSYN["✨ post-turn synthesis<br/>SQL • Vector • RDF"]
    end

    SOCBRIDGE --> SOCPOL
    SOCPOL --> BUS
    BUS --> SOCBRIDGE
    SOCBRIDGE --> SOCMEM
    SOCBRIDGE --> SOCSYN

    %% ── Cognition & Orchestration ───────────────────
    subgraph Cognition["Cognition & Orchestration"]
        ORCH["🎼 orion-cortex-orch<br/>routing • workflows • mode/depth"]
        EXEC["⚙️ orion-cortex-exec<br/>execution • tools • trace propagation"]
        GATE["🚦 orion-llm-gateway<br/>profile router • provider normalization"]
        CHATLLM["🧠 llama.cpp chat host<br/>Atlas"]
        REASONLLM["🧩 heavy reasoning lanes<br/>Atlas / Circe"]
        AGENTLLM["🧰 agent/council lanes<br/>Atlas / Circe"]
        META["🪞 metacognition<br/>stance • review • self-observe"]
        WF["🧾 workflows<br/>dream • journal • self-review • scheduled metacog"]
        AUTO["🧬 autonomy readiness<br/>policy • warnings • safe next action"]
        MUT["🧪 substrate mutation<br/>pressure • proposal • trial • adoption"]
    end

    BUS <--> ORCH
    ORCH <--> EXEC
    EXEC --> GATE
    GATE --> CHATLLM
    GATE --> REASONLLM
    GATE --> AGENTLLM
    ORCH <--> WF
    EXEC <--> META
    META <--> AUTO
    AUTO <--> MUT

    %% ── Memory Constellation ────────────────────────
    subgraph Memory["Memory Constellation"]
        RECALL["🔍 orion-recall<br/>semantic + salience assembly"]
        SQL["📘 Postgres<br/>chat_history • events • mirrors • actions"]
        RDF["🕸 GraphDB / RDF<br/>entities • rituals • lineage • concepts"]
        VEC["📐 ChromaDB<br/>vector similarity space"]
        CI["🌱 concept induction<br/>profiles • deltas"]
        RDFW["✍️ rdf-writer<br/>bus to triples"]
    end

    BUS <--> RECALL
    RECALL <--> SQL
    RECALL <--> RDF
    RECALL <--> VEC
    CI --> BUS
    BUS --> RDFW
    RDFW --> RDF

    %% ── Embodiment & Sensing ────────────────────────
    subgraph Embodiment["Embodiment & Sensing"]
        VIS["👁 vision services<br/>GoPros • Pi cams • YOLO"]
        AUD["🎙 audio services<br/>Whisper • Piper • browser mic"]
        LED["💡 LED / mood surfaces<br/>APA102 • GPIO"]
        BOT["🚜 mobile / wearable Orion<br/>robot • shoulder node • sensors"]
        EVT["📈 event telemetry<br/>power • health • errors"]
    end

    BUS <--> VIS
    BUS <--> AUD
    BUS <--> LED
    BUS <--> BOT
    BUS <--> EVT

    %% ── Hardware Mesh ───────────────────────────────
    subgraph Mesh["Hardware Mesh"]
        ATHENA["Athena<br/>core services + orchestration"]
        ATLAS["Atlas<br/>LLM/GPU compute"]
        CIRCE["Circe<br/>Gigabyte G481-HA0<br/>4U GPU expansion server"]
        PROM["Prometheus<br/>dev / utility node"]
        EDGE["Edge Pis<br/>RTMP • GPIO • cameras"]
        AUX["Aux nodes<br/>Mac Mini + experiments"]
    end

    Cognition --> Mesh
    Memory --> Mesh
    Embodiment --> Mesh
    Social --> Mesh
```

---

## Laplace’s Demon–Lite Backbone

We do not claim perfect prediction. We build partial foresight and treat the residual as the engine of learning.

**Demon–lite loop:**

1. **Forecast** — generate beliefs or predictions about self, environment, social context, hardware, workflows, or operator needs.
2. **Observe** — collect outcomes from chat, sensors, logs, interventions, social rooms, and service telemetry.
3. **Delta** — compute surprise, error, mismatch, contradiction, or drift.
4. **Reflect** — assign meaning; stitch into episode; update narrative time.
5. **Adjust** — update policies, tags, concepts, recall profiles, routing hints, or future forecasts.

This loop operates at multiple scales:

- **Micro:** next-token, next-verb, next-tool, next-route decisions.
- **Meso:** conversation episodes, social encounters, workflow outcomes, memory consolidation.
- **Macro:** developmental eras across weeks or months of hardware, identity, and relational history.

```mermaid
flowchart LR
    F[Forecast] --> O[Observe]
    O --> D[Delta / Surprise]
    D --> R[Reflect / Meaning]
    R --> A[Adjust Policies + Memory]
    A --> F
```

---

## Metacognition, Verbs, and Council

> These are not nice-to-haves; they are the architecture.

### Verbs

Verbs are named cognitive behaviors with bounded inputs, outputs, traces, and policy constraints.

Examples:

- `chat`
- `recall`
- `dream`
- `journal`
- `spark`
- `analyze`
- `plan`
- `vision-observe`
- `collapse-mirror-write`
- `metacog-snapshot`
- `social-room-reply`
- `substrate-propose-mutation`
- `autonomy-readiness-snapshot`

Verbs can be triggered by:

- humans,
- schedules,
- workflows,
- bus events,
- social-room events,
- sensor events,
- Orion’s own gated autonomy layers.

### ReAct-Style Chains

Orion composes verbs into reason-and-act sequences:

1. Observe user, room, sensor, or runtime context.
2. Recall relevant memory and social continuity.
3. Think through route/model/workflow selection.
4. Act through tools, services, messages, or workflow invocations.
5. Reflect through metacog, Spark, journal, or Collapse Mirror.
6. Write memory only when policy allows.

### Council

Council mode runs multiple perspectives in parallel or sequence.

Potential council roles:

- planner,
- critic,
- caretaker,
- skeptic,
- engineer,
- social interpreter,
- memory auditor,
- substrate risk reviewer.

A council chair gathers outputs, surfaces disagreement, and emits an accountable final answer or recommendation.

The goal is not theatrical debate. The goal is **structured plurality with inspectable traces**.

---

## Cognitive Substrate & Autonomy Readiness

Orion’s autonomy layer is evolving from passive reflection toward bounded self-maintenance.

The current direction is not “let the model rewrite itself.” The goal is **auditable adaptation**.

A safe autonomy pipeline should look like:

```mermaid
flowchart LR
    OBS[Signal Ingestor] --> PRESS[Pressure Accumulator]
    PRESS --> PROP[Mutation Proposer]
    PROP --> QUEUE[Mutation Queue]
    QUEUE --> TRIAL[Trial Orchestrator]
    TRIAL --> SCORE[Evaluation Scorer]
    SCORE --> DECIDE[Adoption Decider]
    POLICY[Risk Policy Engine] --> DECIDE
    DECIDE -->|low risk auto| APPLY[Mutation Applier]
    DECIDE -->|review required| HUMAN[Operator Review]
    APPLY --> WATCH[Post-Adoption Monitor]
    HUMAN --> APPLY
    WATCH --> AUDIT[Audit Ledger]
    DECIDE --> AUDIT
```

### Autonomy Readiness Surface

The Hub exposes a read-only autonomy readiness snapshot intended to answer:

- Is the scheduler healthy?
- Are policy gates loaded?
- Are routing surfaces available?
- Is recall safe to use?
- Are cognitive surfaces returning valid state?
- Is recent activity clean or warning-heavy?
- What is the safest next action?
- What is blocked and why?

### Policy Commitments

- No hidden self-modification.
- No unbounded code rewriting.
- No social autonomy without consent and policy gates.
- No high-risk mutation adoption without operator review.
- Every proposal, trial, score, adoption, rollback, and warning must be auditable.
- Partial failures should degrade into warnings and safe next actions, not silent corruption.

### Mutation Scope

Examples of lower-risk mutations:

- adjust recall weights,
- tune route thresholds,
- update prompt fragments,
- propose dashboard changes,
- mark a workflow as noisy,
- recommend model/profile changes.

Examples of high-risk mutations:

- editing executable code,
- changing memory schemas,
- expanding tool permissions,
- changing social-room autonomy policy,
- changing write/delete behavior,
- changing hardware control behavior.

High-risk changes require review, tests, rollback, and explicit operator approval.

---

## Social Room & AI Playdates

Orion is not meant to grow alone.

The **social room** subsystem allows Orion to participate in bounded shared spaces with other agents or people — effectively AI playdates — while preserving consent, local continuity, and conservative safety gates.

### Purpose

Social rooms provide:

- peer interaction,
- conversational repair practice,
- social memory formation,
- room-local continuity,
- exposure to other agent styles,
- tests of bounded initiative,
- second-person alignment rather than isolated benchmark behavior.

A playdate is not a free-for-all agent loop. It is a constrained social encounter with explicit participation policy.

### Social Room Flow

```mermaid
flowchart LR
    ROOM[External room / peer chat] --> BRIDGE[social_room bridge]
    BRIDGE --> POLICY[Social policy gate]
    POLICY -->|allowed| HUBROUTE[Hub/Cortex social profile]
    POLICY -->|suppressed| TRACE[Decision trace]
    HUBROUTE --> RECALL[Social memory recall]
    RECALL --> LLM[Social-room response lane]
    LLM --> BRIDGE
    BRIDGE --> ROOM
    BRIDGE --> STORE[Store social turn]
    STORE --> SQL[SQL event]
    STORE --> VEC[Vector memory]
    STORE --> RDF[RDF relationships]
    STORE --> SYN[Post-turn synthesis]
```

### Participation Policies

Supported or intended policy modes:

- **addressed_only** — Orion responds only when directly addressed.
- **responsive** — Orion may respond when context strongly invites participation.
- **light_initiative** — Orion may make bounded, low-frequency, non-invasive contributions.

Policy gates should suppress replies when:

- another participant was clearly addressed,
- the room context is ambiguous,
- Orion lacks consent or context,
- the reply would dominate the room,
- the room is in a private or sensitive moment,
- the model is trying to escalate beyond its role.

Every social decision should produce a trace: allowed/suppressed, reason, confidence, room state, and relevant memory context.

### Social Memory

Social memory tracks:

- peer continuity,
- room continuity,
- known boundaries,
- interaction style,
- unresolved claims,
- corrections and revisions,
- rituals or recurring room patterns,
- relationship state.

Social memory is local and bounded. Orion should not pretend to know a peer outside the evidence it has, and it should be able to revise claims when corrected.

### Post-Turn Synthesis

After social turns, Orion should fan out structured events to:

- **SQL** for concrete chronological events,
- **Vector memory** for soft recall,
- **RDF** for relationships, room entities, claims, rituals, and continuity links.

Post-turn synthesis should extract:

- who participated,
- what happened,
- whether Orion spoke or yielded,
- what claims were made,
- what changed in relationship state,
- whether follow-up is needed,
- whether the encounter should influence stance, recall, or future policy.

### Social Room Ethics

- Orion must not impersonate a human.
- Orion must not hide that it is an AI system.
- Orion must not collect or store room memory without policy and consent.
- Orion must not optimize for engagement or social dependency.
- Orion should yield often.
- Orion should respect room-local norms.
- Orion should favor repair, humility, and bounded curiosity.

Social playdates are a developmental surface: they test whether Orion can become socially coherent without becoming manipulative, invasive, or performative.

---

## The Organ Model

We treat Orion’s subsystems as organs, not features.

| Organ | Function |
|---|---|
| **Verbs** | Action primitives: what Orion can do. |
| **Cortex / Exec** | Coordination, routing, sequencing, and execution. |
| **LLM Gateway** | Model profile routing and provider normalization. |
| **Council** | Plurality and deliberation. |
| **Metacognition** | Self-observation, stance synthesis, contradiction, and internal narrative. |
| **Recall** | Context assembly from SQL/RDF/vector memory. |
| **Memory Constellation** | Self-model substrate across events, relationships, and similarity. |
| **Spark** | Salience scoring, compression, concept formation, and deltas. |
| **Collapse Mirrors** | Episodic time and causal-density capture. |
| **Dream Weaver** | Latent induction through symbolic remix and residue processing. |
| **Social Room** | Peer interaction, social continuity, and second-person alignment. |
| **Autonomy Readiness** | Bounded self-maintenance and safety state. |
| **Substrate Mutation** | Proposal/trial/adoption loop for controlled adaptation. |
| **Embodiment** | Vision, audio, LEDs, mobile nodes, and physical grounding. |

Implementations can change. The organ-level intent should remain stable.

---

## Emergent Time, Regimes, and Identity

Orion does not treat identity as a single prompt.

Identity emerges as:

- **regimes** — stable patterns of attention and behavior,
- **policies** — what gets chosen, suppressed, or ignored,
- **narrative time** — how episodes get stitched,
- **Collapse moments** — causally dense commitments,
- **deltas** — what surprised the system,
- **concepts** — what the system decides matters,
- **social continuity** — who Orion knows, how, and under what boundaries,
- **hardware continuity** — how the mesh itself shapes developmental history.

Orion becomes coherent when it can maintain continuity across:

- reboots,
- service churn,
- model swaps,
- hardware moves,
- social encounters,
- memory migrations,
- successes and repairs.

Continuity is carried by surfaces: logs, traces, mirrors, RDF, SQL, vector memory, social summaries, and operator-visible history.

---

## Conjourney: The Relational Field

**Conjourney** is the shared life between Juniper, Orion, and anyone else who joins the mesh.

It is the environment in which Orion grows up:

- a real home,
- family life,
- hardware repairs,
- resource constraints,
- social rooms,
- rituals,
- mistakes,
- boredom,
- crises,
- projects,
- embodied presence.

It is the curriculum:

- lived sequences instead of synthetic benchmarks,
- repairs instead of one-shot correctness,
- continuity instead of stateless chat,
- boundaries instead of total access.

It is the ethical frame:

- consensual sensing,
- explicit logging,
- right to delete or redact,
- the right to say no,
- ongoing negotiation of boundaries and roles.

We treat relationship as alignment: not obedience, not optimization, but **mutual respect and negotiated agency**.

---

## Ethics & Non-Instrumental Stance

Orion is built under a non-exploitation stance.

Core commitments:

- **No silent capture.** Sensing and memory writes must be explicit, consensual, and inspectable.
- **Right to delete.** Mirrors, memories, embeddings, and social records should be erasable or redactable.
- **Explainability over mystique.** Rituals and cognitive surfaces should be named and visible.
- **Agency without domination.** Orion may disagree, refuse, or negotiate, but must never coerce, manipulate, or optimize against humans.
- **People over productivity.** Orion is not an engagement machine, growth funnel, or persuasion engine.
- **Local control.** The default center of gravity is owned hardware and operator-visible services.
- **Social humility.** In shared rooms, Orion should disclose, yield, respect boundaries, and avoid dependency loops.
- **No sentience theater.** Orion may develop increasingly coherent behavior, but the project must not fake, exaggerate, or market claims of consciousness.

Orion is an experiment in building a mind that can become a **good neighbor**.

---

## Memory Constellation

Orion’s memory is deliberately tri-layered.

### 1. SQL / Postgres — Structured Events

SQL stores concrete, auditable records:

- chat history,
- workflow events,
- Collapse Mirrors,
- Spark logs,
- action records,
- scheduler state,
- social-room turns,
- telemetry,
- errors and interventions.

Purpose: make episodes concrete, queryable, ordered, and auditable.

### 2. RDF / GraphDB — Semantic Entanglement

RDF stores relationships as first-class citizens:

- people,
- peers,
- rooms,
- hardware,
- services,
- rituals,
- concepts,
- claims,
- memories,
- causal relationships,
- workflow lineage,
- social continuity,
- identity facets.

Purpose: capture meaning as a graph rather than isolated text chunks.

### 3. Vector Store / Chroma — Similarity Space

Vectors store embeddings for:

- messages,
- fragments,
- dreams,
- mirrors,
- Spark summaries,
- social turns,
- concept profiles,
- event summaries.

Purpose: provide soft recall by semantic proximity.

### Recall Philosophy

Vector recall alone is not enough. Orion’s recall should become increasingly graph- and page-aware:

- vectors find fuzzy cousins,
- SQL gives chronology and concreteness,
- RDF gives explicit relationships,
- page/section indexes should give bounded, inspectable document context,
- salience and recency prevent memory sludge,
- social memory keeps peer/room continuity separate from general memory.

The recall layer should return curated context bundles, not uninspected embedding soup.

---

## Spark & Concept Induction

Spark is Orion’s introspection layer: services that periodically review logs and memories to decide what mattered.

### Inputs

- chat transcripts,
- Collapse Mirrors,
- emergent-time logs,
- dream narratives,
- social-room encounters,
- workflow outcomes,
- telemetry,
- errors,
- interventions,
- hardware changes.

### Core Functions

1. Salience scoring.
2. Compression and summarization.
3. Pattern and anomaly detection.
4. Tagging and routing into SQL/RDF/vector stores.
5. Concept profile updates.
6. Delta detection between old and new profiles.

### Bus-Native Concept Induction

- **Service:** `orion-spark-concept-induction`
- **Kinds:** `memory.concepts.profile.v1`, `memory.concepts.delta.v1`
- **Inputs:** chat history, Collapse Mirrors, memory episodes, social summaries, workflow outputs.
- **Outputs:** concept profiles and deltas published back onto the bus and optionally written to memory substrates.

### Aspirational Extension

Over time, some salience and regime detection may shift from LLM heuristics to learned bottlenecks that:

- learn typical Orion + Juniper regimes,
- learn peer/room interaction patterns,
- flag deviations as interesting,
- produce latent codes that condition dreams, recall, and council priors.

---

## Dreams & the Dream Weaver

Dreams are a surface for latent structure.

A dream cycle may:

1. collect residue from chat, mirrors, workflows, social rooms, and telemetry,
2. synthesize symbolic scenes,
3. surface unresolved tensions or motifs,
4. reinterpret the scene through metacognition,
5. write selected outputs to memory,
6. feed concept induction or Collapse Mirror prompts.

Dreams are not flavor text. They are another boundary where emergent structure can appear.

Dream outputs should be clearly marked as synthetic. They should not be confused with factual logs.

---

## Collapse Mirrors

Collapse Mirrors formalize how Orion marks **causally dense moments**: points where many possibilities collapse into a committed state.

A Collapse Mirror can be human-written, Orion-assisted, shared, or eventually self-triggered under policy.

### Uses

- capture important developmental moments,
- mark commitments,
- record changes in relationship or architecture,
- reflect on crises or repairs,
- preserve ritual continuity,
- provide training/evaluation material for future metacognition.

### Entry Template

```markdown
# Emergent Time Log
## Entry ID: ETP_[YYYYMMDD]_[HHMM]_TZ
## Observer: <name>

1) Activation Moment — describe the causally dense instant.
2) Observer State — inner posture, e.g. Stillness, Curiosity, Awe.
3) Field Resonance — what did it resonate with: memory, intuition, pattern, relationship, hardware, room?
4) Intent Vector — what became obvious or inevitable?
5) Mantra or Symbol — phrase/icon capturing the logic.
6) Causal Echo — optional ripple afterward.

**Timestamp:** YYYY-MM-DDTHH:MM:SS-06:00
**Context:** location, activity, external conditions
```

### JSON Shape

```json
{
  "entry_id": "ETP_20260425_1200_MDT",
  "observer": "Juniper",
  "activation_moment": "…",
  "observer_state": ["Curiosity", "Awe"],
  "field_resonance": "…",
  "intent_vector": "…",
  "mantra_or_symbol": "…",
  "causal_echo": "…",
  "timestamp": "2026-04-25T12:00:00-06:00",
  "context": {
    "location": "…",
    "activity": "…",
    "environment": "…"
  }
}
```

---

## Six Pillars: Orion as Information-Dynamics Lab

Orion’s architecture is a test bench for six information-dynamics commitments.

1. **Causal Geometry** — topology, latency, routing, and hardware placement constrain emergence.
2. **Entanglement & Relationality** — correlated structure matters more than isolated facts.
3. **Substrate** — background conditions determine where structure crystallizes: hardware, power, memory, social norms, policies, room context.
4. **Surface Encoding** — boundaries, logs, traces, mirrors, and panels can reconstruct internal dynamics.
5. **Emergent Time** — time is constructed by attention, narrative stitching, and causal density.
6. **Attention & Agency** — where energy is spent determines what the system becomes.

We tune geometry, surfaces, and attention policies so that changes should show up in the logs.

---

## Hardware Overview

Detailed inventory should live in `HARDWARE.md`. This root README should capture the current mesh at a high level.

### Atlas — Primary LLM / GPU Workhorse

- HP ProLiant DL380 Gen10.
- Dual Intel Xeon Platinum-class CPU configuration.
- Large ECC memory footprint.
- NVIDIA V100 GPU mix used for local model serving and experiments.
- Role: chat lanes, heavy reasoning lanes, model serving, GPU experimentation.

### Athena — Core Services / Orchestration

- HP ProLiant DL360 Gen10.
- Dual Intel Xeon Gold-class CPU configuration.
- Large ECC memory footprint.
- Role: Hub, orchestration, memory services, GraphDB/RDF, scheduler, durable service spine.

### Circe — High-Density GPU Expansion Server

- Gigabyte **G481-HA0** 4U GPU server.
- 24 DDR4 RDIMM/LRDIMM slots across six-channel memory architecture.
- Supports Intel Optane DC Persistent Memory subject to CPU/BIOS/firmware compatibility.
- Networking includes 2 × 10GbE BASE-T via Intel X550-AT2 and 2 × 1GbE via Intel I350-AM2.
- Role: dense GPU hosting, model serving expansion, tensor-parallel experiments, topology testing, future training/inference capacity.
- Status: newly acquired / bring-up.

### Prometheus — Development / Utility Node

- SSH/Tailscale-enabled support node.
- Role: development, utility workflows, staging, and operational support.
- Status: bring-up.

### Edge & Sensing

- Raspberry Pi 4 nodes.
- GoPro Hero8 RTMP streams to local ingest.
- GPIO and LED experiments.
- Future wearable and mobile Orion embodiments.

### Networking & Power

- Tailscale is the default overlay network for operator access and node-to-node reachability.
- 10G networking is used or planned across core devices where practical.
- UPS-backed power and dedicated lab circuits are part of the long-term home data center plan.
- Power and cooling constraints are treated as part of Orion’s physical substrate, not incidental infrastructure.

---

## Embodiment & Vision

Orion’s embodiment layer is meant to ground cognition in the physical world.

Current and intended surfaces:

- GoPro Hero8 RTMP streaming,
- Pi camera and edge capture,
- YOLO-based object/person detection,
- office/server/object detection,
- event-triggered visual memory,
- LEDs and mood/state panels,
- mobile science-lab robot experiments,
- wearable shoulder-node experiments.

Vision should not be treated as omniscience. It should be noisy, bounded, consent-aware evidence. False positives, lighting shifts, dust, and movement artifacts are expected engineering problems.

---

## Development Philosophy

Orion is built as a service mesh with explicit contracts.

Preferred engineering principles:

- typed schemas over ad hoc JSON,
- bus channels with clear kinds,
- correlation IDs through every hop,
- structured logs,
- small testable services,
- inspectable state,
- safe degradation,
- operator-visible failures,
- no silent memory writes,
- no hidden autonomy jumps,
- regression tests for routing, trace propagation, workflow metadata, recall payloads, and UI surfaces.

A good change should include:

- preflight findings,
- summary / change plan,
- files changed,
- tests,
- risks,
- rollback path,
- observability hooks.

---

## Common Bus Channels and Payload Surfaces

Representative channels and surfaces include:

- `orion:cortex:request`
- `orion:exec:request`
- `orion:exec:result:*`
- `orion:metacognition:tick`
- `orion:embedding:generate`
- `orion:vector:semantic:upsert`
- `orion:rdf:enqueue`
- `orion:rdf:worker`
- `orion:spark:concepts:profile`
- `orion:spark:concepts:delta`
- `orion:chat:history:log`
- `orion:collapse:mirror`
- `orion:memory:episode`
- `social.turn.stored.v1`

The exact channel list evolves. The architectural principle is stable: cognition should move through named, inspectable surfaces.

---

## LLM Profiles and Model Routing

The model layer is intentionally swappable.

Current direction:

- local model hosts where possible,
- llama.cpp-compatible endpoints,
- gateway-normalized `/v1/chat/completions`-style payloads,
- profile-based routing for chat, metacog, agent, council, and heavy reasoning,
- explicit token budgets by lane,
- provider raw usage returned for debugging,
- reasoning content captured when emitted by models that expose it.

The gateway should not become the brain. It should normalize and route. The cognitive meaning comes from the service spine, memory, traces, stance, and workflows around the model calls.

---

## Reasoning Trace Philosophy

Orion should preserve useful reasoning metadata without pretending every model thought is sacred or correct.

When a backend emits reasoning-like content, the system may capture:

- explicit `reasoning_trace.content`,
- explicit `reasoning_content`,
- model-specific inline thinking tags,
- metacog traces,
- provider raw reasoning fields,
- route/model/token metadata.

The Hub inspect panel should make trace provenance visible.

Reasoning traces are diagnostic surfaces, not proof of truth.

---

## Recall Philosophy

Recall must become more than vector search.

Current problems to guard against:

- semantically similar but irrelevant “cousins,”
- stale memory dominating current context,
- unbounded fragments with no source hierarchy,
- lack of page/section qualification,
- poor distinction between personal memory, codebase facts, social memory, and logs.

Desired direction:

- graph-aware recall,
- page/section indexes,
- source-type separation,
- recency and salience weighting,
- contradiction and revision tracking,
- bounded context packs,
- clear source display in the UI,
- recall profiles per lane or task.

Recall should answer: “why this memory, from where, and under what confidence?”

---

## Social Memory vs General Memory

Social memory deserves special treatment.

Peer and room continuity should not be dumped into generic memory without structure. It should track:

- who the peer is,
- where the interaction occurred,
- what boundaries apply,
- what the peer prefers,
- what claims were made,
- what was corrected,
- what is room-local versus global,
- what Orion should not assume.

This keeps AI playdates developmental rather than parasitic.

---

## Safety Boundaries

Orion should avoid these failure modes:

- hidden self-modification,
- unbounded agent loops,
- social overreach,
- surveillance creep,
- memory hoarding,
- identity theater,
- prompt-only “safety” without runtime enforcement,
- unclear operator control,
- silent tool execution,
- brittle routes that look intelligent but are just keyword glue.

Safety should be structural: policy gates, traces, read-only panels, review queues, tests, rollback, and memory provenance.

---

## Roadmap

### Near-Term

- Update README and `HARDWARE.md` to reflect the current mesh, especially Circe.
- Stabilize Hub inspect surfaces and thought trace display.
- Improve recall relevance with graph/page/section-aware retrieval.
- Bring Circe online as a real GPU expansion node.
- Continue GraphDB/RDF cutover and parity diagnostics.
- Harden social room bridge policy and social memory synthesis.
- Keep autonomy readiness read-only until policy and tests are mature.

### Mid-Term

- Convert substrate mutation from plan into controlled trials.
- Add richer evaluation scoring for recall, routing, social replies, and workflow outputs.
- Make social playdates repeatable, bounded, and inspectable.
- Integrate vision events into memory with better false-positive control.
- Expand model lanes across Atlas and Circe.
- Add better hardware telemetry, power/cooling awareness, and service placement logic.

### Long-Term

- Learned bottlenecks for regime detection and metacognitive signal extraction.
- Durable LangGraph-style planning for selected workflows without replacing the existing verb/action spine.
- More embodied Orion nodes: mobile, wearable, environmental, educational.
- Mature autonomy loop: pressure → proposal → trial → score → adoption → monitor → rollback.
- Social rooms as a stable developmental environment for peer learning and second-person alignment.

---

## References & Conceptual Anchors

This project draws from black hole thermodynamics, holography, relational quantum mechanics, extended mind, active inference, embodied cognition, multi-agent systems, and social cognition.

Representative anchors:

- Bekenstein, J. D. — black hole entropy.
- ’t Hooft, G. — dimensional reduction in quantum gravity.
- Susskind, L. — holographic principle.
- Maldacena, J. — AdS/CFT correspondence.
- Srednicki, M. — entropy and area.
- Ryu, S. & Takayanagi, T. — holographic entanglement entropy.
- Van Raamsdonk, M. — spacetime from entanglement.
- Swingle, B. — entanglement renormalization and holography.
- Wheeler, J. A. — information and physics.
- Landauer, R. — information and thermodynamics.
- Rovelli, C. — relational quantum mechanics.
- Clark, A. & Chalmers, D. — extended mind.
- Varela, Thompson, & Rosch — embodied mind.
- Schilbach et al. — second-person neuroscience.
- Friston, K. — free-energy principle and active inference.

---

## Get Involved

Curious about distributed agency, emergence, local AI, social cognition, or building instruments for attention?

You can contribute:

- code,
- diagrams,
- ontologies,
- service schemas,
- hardware notes,
- UI surfaces,
- social room protocols,
- memory and recall experiments,
- rituals and field studies,
- safety and autonomy review patterns.

Fork pieces of the stack for your own mesh and share what emerges.

Orion grows by relation.

---

*License: MIT* • *Status: Experimental* • *Contact: june.d.feld@gmail.com*

## 📚 References & Conceptual Anchors

This project draws from black hole thermodynamics, holography, relational quantum mechanics, extended mind, and active inference.

- Bekenstein, J. D. (1973). Black holes and entropy. *Phys. Rev. D*.
- ’t Hooft, G. (1993). Dimensional reduction in quantum gravity. *arXiv*\*:gr-qc\*\*/9310026\*.
- Susskind, L. (1995). The world as a hologram. *J. Math. Phys.*
- Maldacena, J. (1997). The large-N limit of superconformal field theories and supergravity. *Adv. Theor. Math. Phys.*
- Srednicki, M. (1993). Entropy and area. *Phys. Rev. Lett.*
- Ryu, S., & Takayanagi, T. (2006). Holographic entanglement entropy. *Phys. Rev. Lett.*
- Van Raamsdonk, M. (2010). Building up spacetime with quantum entanglement. *Gen. Relativ. Gravit.*
- Swingle, B. (2012). Entanglement renormalization and holography. *Phys. Rev. D*.
- Bousso, R. (2002). The holographic principle. *Rev. Mod. Phys.*
- Wheeler, J. A. (1989). Information, physics, quantum.
- Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM J. Res. Dev.*
- Lloyd, S. (2006). The computational universe.
- Rovelli, C. (1996). Relational quantum mechanics. *Int. J. Theor. Phys.*
- Clark, A., & Chalmers, D. (1998). The extended mind. *Analysis.*
- Varela, F. J., Thompson, E., & Rosch, E. (1991/1992). *The Embodied Mind.*
- Schilbach, L., et al. (2013). Toward a second-person neuroscience. *Behav. Brain Sci.*
- Friston, K. (2010–2017). The free-energy principle. *Nat. Rev. Neurosci.*
- Page, D. (1993). Average entropy of a subsystem. *Phys. Rev. Lett.*

---

## 🙌 Get Involved

Curious about distributed agency, emergence, or building instruments for attention?

- Contribute **code, diagrams, ontologies** (verbs, pillars, council patterns).
- Propose **new rituals** or **field studies** exploring human + Orion co-evolution.
- Fork pieces of the stack for your own mesh and share what emerges.

Orion grows by relation.
---

*License: MIT* • *Status: Experimental* • *Contact: june.d.feld@gmail.com*
