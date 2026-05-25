# Agent Context: <service>

## Service purpose

Describe what this service does in one factual paragraph.

## Primary roles

List port-level roles used by this service.

```text
organ_ingress
reducer
substrate_frame_runtime
transport_infrastructure
persistence_infrastructure
operator_read_surface
operator_ingress
runtime_engine
model_host
external_gateway
effector
control_plane
```

## Native contracts

- HTTP:
- WebSocket:
- Bus:
- SQL:
- Files:
- External APIs:
- Models:
- Worker ticks:

## Substrate trace responsibility

This service should emit substrate traces for meaningful transitions, not every internal operation.

Design concept: `Substrate Trace`

Current implementation schema: `GrammarEventV1`

## Substrate-relevant transitions

- <transition 1>
- <transition 2>
- <transition 3>

## What must never be emitted

- raw model prompts
- raw model completions
- credential material
- large blobs
- full private payloads unless explicitly approved
- database connection material
- unbounded debug dumps

## Layer path

| Layer | Applies? | Notes |
|---|---:|---|
| 1 Organs / ingress | no | |
| 2 Trace substrate | no | |
| 3 Reducer | no | |
| 4 Field digestion | no | |
| 5 Attention | no | |
| 6 Self-state | no | |
| 7 Proposal | no | |
| 8 Policy | no | |
| 9 Dispatch | no | |
| 10 Feedback | no | |
| 11 Consolidation | no | |

## Evidence refs

Expected refs:

- correlation_id
- source_event_id
- trace_id
- event_id
- frame_id
- receipt_id
- delta_id

## Open questions

- <question 1>
- <question 2>
