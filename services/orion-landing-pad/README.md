# 🛰️ Orion Landing Pad

The Landing Pad is an ephemeral working-memory surface for Orion. It ingests Titanium envelopes from the bus, normalizes them into `PadEventV1`, builds rolling state frames, emits high-salience signals, and exposes a lightweight RPC + HTTP debug surface.

## Capabilities
- Titanium-compliant envelopes using the shared `BaseEnvelope`.
- Reducer registry with safe fallback for unknown kinds.
- Salience scoring + pulse publishing for high-salience events.
- Rolling frame aggregation with tensor hash projections.
- RPC over the bus (`orion.pad.rpc.request.v1`).
- HTTP debug API for quick inspection.

## Running locally
1. Copy `.env_example` to `.env` and adjust values.
2. `docker compose -f services/orion-landing-pad/docker-compose.yml --env-file services/orion-landing-pad/.env up --build`
3. HTTP health: `curl http://localhost:$PORT/health`

## Debug APIs
- `GET /health`
- `GET /frame/latest`
- `GET /frames?limit=10`
- `GET /events/salient?limit=20`
- `GET /tensor/latest`

## Bus contracts
See `docs/landing_pad.md` for full channel/kind contracts and onboarding guidance.
