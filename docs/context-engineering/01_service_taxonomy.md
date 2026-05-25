# Service Taxonomy

## Services are not taxa

A single service can have multiple roles.

Classify service **ports/contracts**, not whole services.

Example:

```text
orion-hub
  GET /api/substrate/self-state/latest
    role: operator_read_surface

  POST /api/chat
    role: operator_ingress

  POST /api/proposal/review
    role: control_plane
```

## Port roles

Use these roles in `SERVICE_PORTS.yaml`.

### `organ_ingress`

A port that observes the world, body, infrastructure, operator, model behavior, or internal cognition and turns it into substrate-legible events.

Typical trace responsibility: high.

Examples:

- biometrics samples;
- vision frame observations;
- world pulse observations;
- security watcher events;
- power telemetry;
- operator input ingress;
- audio/STT ingress.

### `reducer`

A port or runtime that consumes substrate traces and commits state deltas or reduction receipts.

Typical primary contract:

```text
StateDeltaV1
ReductionReceiptV1
```

### `substrate_frame_runtime`

A runtime that consumes committed substrate state or typed frames and emits a compiled typed frame.

Examples:

- field digester;
- attention runtime;
- self-state runtime;
- proposal runtime;
- policy runtime;
- execution dispatch runtime;
- feedback runtime;
- future consolidation runtime.

Primary contract: typed frame.

Trace responsibility: lifecycle and lineage traces for frame creation.

### `transport_infrastructure`

Message movement and channel semantics.

Examples:

- bus service;
- bus tap;
- channel catalog enforcement.

Trace responsibility: transport anomalies, delivery lag, channel violations, backpressure, schema rejection.

### `persistence_infrastructure`

Durable storage and projections.

Examples:

- SQL writer;
- SQL DB;
- vector DB;
- state journaler;
- GraphDB/RDF writer.

Trace responsibility: write committed, write failed, schema decode failed, projection lag, duplicate suppressed.

### `operator_read_surface`

Read-only human/UI/debug surfaces.

Trace responsibility: usually low. Optional traces for substrate-relevant inspections.

### `operator_ingress`

Human input entering the mesh.

Trace responsibility: high, but redacted.

Examples:

- chat message received;
- operator approval/denial;
- manual override requested;
- review submitted.

### `runtime_engine`

Executes plans, chains, councils, workflows, or routing decisions.

Trace responsibility: route selected, plan started, step completed, step failed, result emitted, blocked.

### `model_host`

Hosts inference endpoints.

Trace responsibility: model load, request accepted/rejected, latency, token pressure, failure, capacity. Do not emit raw prompts or completions by default.

### `external_gateway`

Translates between Orion and an external protocol/system.

Trace responsibility: ingress events, egress attempts, consent/policy boundaries, external failure.

### `effector`

Produces external effects.

Examples:

- TTS speech;
- outbound social message;
- service restart;
- power control;
- file write;
- operator notification.

Trace responsibility: high. Must pass through proposal/policy/dispatch unless explicitly manual/operator-driven.

### `control_plane`

Mutates configuration, approvals, policy state, runtime state, or operator-reviewed decisions.

Trace responsibility: high. Policy and evidence required.

## Role-to-contract matrix

| Role | Native contract | Trace required? | Downstream stage |
|---|---|---:|---|
| organ_ingress | service-specific observation | yes | Layer 1–3 |
| reducer | state delta / receipt | yes for lifecycle | Layer 3 |
| substrate_frame_runtime | typed frame | yes for lifecycle/lineage | Layers 4–11 |
| transport_infrastructure | bus/message envelope | telemetry only | Layer 1/10/11 if relevant |
| persistence_infrastructure | write/projection | telemetry only | Layer 1/10/11 if relevant |
| operator_read_surface | HTTP/UI read | usually no | no direct stage |
| operator_ingress | operator event | yes | Layer 1/8/10 |
| runtime_engine | plan/step/result | yes | Layer 1–3 or 9–10 |
| model_host | inference endpoint | telemetry only | Layer 1/10/11 if relevant |
| external_gateway | protocol bridge | yes for ingress/egress | Layer 1/9/10 |
| effector | external effect | yes | Layer 9/10 |
| control_plane | mutation/approval | yes | Layer 8/10 |

## Classification workflow

For each service:

1. List ports/contracts.
2. Assign each port a role.
3. Identify meaningful transitions.
4. Decide whether substrate traces are primary, lifecycle, telemetry, or not needed.
5. Decide whether a reducer is needed.
6. Decide whether field/attention/self/proposal/policy/dispatch/feedback/consolidation should consume the outputs.

## Warning signs

A service needs taxonomy hardening if:

- it has a POST route that mutates state but no control-plane classification;
- it emits bus messages without a documented channel contract;
- it performs external effects without policy/dispatch lineage;
- it stores artifacts without source/evidence refs;
- it is described only by prose README claims;
- agents must inspect code to infer basic role and contract.