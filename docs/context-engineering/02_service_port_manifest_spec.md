# Service Port Manifest Spec

Each service should declare substrate-facing ports in `services/<service>/SERVICE_PORTS.yaml`.

The manifest is intentionally small. It tells an agent what contracts exist, what role each port plays, and where substrate traces belong.

## Required shape

```yaml
service: orion-example
status: draft

ports:
  - name: example_ingress
    role: organ_ingress
    native_contract: http_json
    substrate_trace_required: true
    trace_roles:
      - example_request_received
      - example_request_validated
      - example_result_emitted
    downstream_layers: [1, 2, 3]
    evidence_refs:
      - correlation_id
      - source_event_id
    redaction:
      never_emit:
        - raw_model_prompt
        - credential_material
        - full_payload_blob
```

## Fields

`service` is the repository service name, usually matching `services/<service>`.

`ports[].name` is a stable local name such as `chat_ingress`, `proposal_review`, `state_read_api`, `bus_publish_loop`, or `feedback_worker_tick`.

`ports[].role` must be one of:

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

`ports[].native_contract` is the local operational format, such as `http_json`, `websocket_json`, `redis_channel`, `sql_table`, `file`, `model_api`, `audio_stream`, or `internal_worker_tick`.

`ports[].substrate_trace_required` is true when the port produces transitions that matter to future cognition.

`ports[].trace_roles` lists meaningful transition names. Use names like `operator_review_submitted`, `policy_decision_rejected`, `dispatch_candidate_blocked`, `sql_write_failed`, or `bus_schema_validation_failed` rather than function names.

`ports[].downstream_layers` lists which Layers 1–11 are relevant.

`ports[].evidence_refs` lists expected identifiers such as `correlation_id`, `session_id`, `source_event_id`, `frame_id`, `proposal_id`, `policy_decision_id`, `dispatch_id`, `receipt_id`, `delta_id`, or `otel_trace_id`.

`ports[].redaction.never_emit` lists local safety boundaries. Prefer references and ids over large or private content.

## Example: Hub

```yaml
service: orion-hub
status: draft

ports:
  - name: substrate_read_api
    role: operator_read_surface
    native_contract: http_json
    substrate_trace_required: false
    notes: Read-only projection of typed substrate frames.

  - name: chat_ingress
    role: operator_ingress
    native_contract: websocket_json
    substrate_trace_required: true
    trace_roles:
      - operator_message_received
      - operator_message_routed
    downstream_layers: [1, 2, 3, 10, 11]
    evidence_refs:
      - session_id
      - message_id
      - correlation_id
    redaction:
      never_emit:
        - raw_model_prompt
        - full_user_message_text
        - credential_material

  - name: proposal_review
    role: control_plane
    native_contract: http_json
    substrate_trace_required: true
    trace_roles:
      - operator_review_submitted
      - operator_approval_granted
      - operator_approval_denied
    downstream_layers: [1, 2, 3, 8, 10, 11]
    evidence_refs:
      - proposal_id
      - policy_frame_id
      - review_decision_id
```

## Example: SQL writer

```yaml
service: orion-sql-writer
status: draft

ports:
  - name: grammar_event_writer
    role: persistence_infrastructure
    native_contract: sql_insert
    substrate_trace_required: true
    trace_roles:
      - grammar_event_persisted
      - grammar_event_persist_failed
      - schema_decode_failed
      - write_lag_observed
    downstream_layers: [1, 2, 10, 11]
    evidence_refs:
      - event_id
      - table
      - error_class
    redaction:
      never_emit:
        - database_connection_material
        - full_payload_blob
```

## Acceptance criteria

A manifest is acceptable when an agent can answer:

- what ports exist;
- which ports emit traces;
- which semantic roles are expected;
- which layers are affected;
- what identifiers provide lineage;
- what data should be represented only by safe references.