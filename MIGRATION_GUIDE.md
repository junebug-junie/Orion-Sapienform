# Orion Bus & Schema Migration Guide

This repo is converging on the canonical bus + envelope contract defined in `docs/bus-contracts.md` and `docs/contracts.md`. Use this guide while migrating services.

## Canonical Envelope
- Model: `orion.core.bus.bus_schemas.BaseEnvelope` (or typed `Envelope[PayloadModel]`).
- Fields:
  - `schema`/`schema_id="orion.envelope"` and `schema_version="2.0.0"`
  - `id` (unique per message)
  - `correlation_id` (stable across a flow; never regenerate downstream)
  - `kind` (e.g., `llm.chat.request`, `cortex.exec.result`)
  - `reply_to` (set for RPC calls so the callee knows where to respond)
  - `source` (`ServiceRef` with `name`, optional `node`, `version`, `instance`)
  - `causality_chain` (append-only lineage; preserve when replying/forwarding)
  - `payload` (typed Pydantic model or strict dict)

## Bus Client Patterns
- Use `orion.core.bus.async_service.OrionBusAsync` for async publish/subscribe/RPC.
- Codec: always encode/decode through `orion.core.bus.codec.OrionCodec`.
- Intake:
  1. Subscribe with `bus.subscribe(...)`.
  2. For each message, `decoded = bus.codec.decode(msg["data"])`.
  3. Drop/telemetry on `decoded.ok` is false.
  4. Validate/parse payload into the typed model from `orion/schemas/*`.
- Outbound publish: build an envelope with preserved `correlation_id`, `reply_to`, and `causality_chain` (use `derive_child` when appropriate); publish via `bus.publish(channel, envelope)`.
- RPC: use `await bus.rpc_request(request_channel, envelope, reply_channel=..., timeout_sec=...)`.

## Kinds & Channels
- Respect kind names and request/reply channels from `docs/bus-contracts.md`.
- Reply channels are caller-owned (`<prefix>:<uuid>`). Do not invent new channel names; expose them as settings with defaults pulled from `.env_example` and `docker-compose.yml`.

## Settings Provenance Checklist
For every setting a service uses:
1) Add/update the field in `app/settings.py` with validation/defaults.
2) Ensure `.env_example` documents the variable.
3) Ensure `docker-compose.yml` wires the variable in the service `environment:` block.

## Per-Service Migration Checklist
- [ ] Remove legacy `orion.core.bus.service.OrionBus` imports/usages.
- [ ] Switch to `OrionBusAsync` (or chassis helpers) with `OrionCodec`.
- [ ] Decode every inbound message into `BaseEnvelope`/`Envelope`.
- [ ] Validate payloads using typed models from `orion/schemas/*` (or strict local Pydantic models where no shared schema exists).
- [ ] Preserve `correlation_id`, `reply_to`, and `causality_chain` on replies.
- [ ] Emit responses/notifications as envelopes with the correct `kind` and `source`.
- [ ] Update `settings.py` + `.env_example` + `docker-compose.yml` for any env var changes.
- [ ] Run `python -m compileall services/<service>/app` and the smallest available smoke/build check (e.g., `docker compose -f services/<service>/docker-compose.yml build <service>`).
