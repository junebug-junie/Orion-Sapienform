# Unified Cognitive Substrate — Phase 21: Control-Plane Parity and Wiring Verification

## Why this phase exists

Phase 20c established Postgres-first parity for policy profile and telemetry comparison paths, but queue/control-plane durability and wiring verification still needed explicit hardening.

Phase 21 completes parity and verification work without changing cognitive semantics.

## Boundary contract (unchanged)

- **GraphDB**: semantic/cognitive substrate state and semantic reads.
- **Postgres**: operational/control-plane durability (queue, policy profile lifecycle, telemetry, operator comparison reads).

## What Phase 21 hardens

1. **Queue control-plane parity**
   - `GraphReviewQueue` now supports Postgres-first persistence/read path.
   - sqlite remains bounded fallback/dev support.
   - source posture is explicit (`postgres`, `sqlite`, `fallback`, `memory`) with surfaced errors.

2. **Operator source honesty**
   - Hub substrate queue/telemetry/calibration/comparison payloads now surface concrete control-plane source posture and degradation signals.

3. **Bus/schema/sql-writer wiring verification**
   - Explicit test coverage now verifies feedback/control-plane-adjacent wiring exists across:
     - `orion/bus/channels.yaml`
     - `orion/schemas/registry.py`
     - `services/orion-sql-writer/app/settings.py`
     - `services/orion-sql-writer/.env_example`
     - `services/orion-sql-writer/docker-compose.yml`
   - This prevents silent routing regressions.

4. **Deployment parity polish**
   - SQL-writer env/compose defaults include feedback channel and kind route mapping.
   - Hub env/compose include control-plane Postgres selection variables and bounded sqlite queue fallback variable.

## Safety posture

- Advisory-only policy comparison semantics are unchanged.
- No autonomous policy mutation was introduced.
- No semantic substrate responsibility was moved out of GraphDB.

## Remaining known hardening gap

- Table bootstrap is still runtime-owned for low blast-radius parity. Migration-framework ownership can be introduced later without changing contracts.
