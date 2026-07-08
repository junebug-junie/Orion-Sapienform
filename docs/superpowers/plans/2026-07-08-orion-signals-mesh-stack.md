# Orion Signals Mesh Stack — Implementation Plan

> **Goal:** Add `services/orion-signals/` — a thin orchestration layer for the organ-signal mesh (causal spine), not a new cognition service.

**Architecture:** Machine-readable roster (`roster.v1.yaml`) + deterministic launcher scripts. Each producer keeps its own `docker-compose.yml` and `.env`. The stack passes shared `ORION_BUS_URL` and selects tiers via profile. Hub is documented as an external operator dependency (host networking).

**Non-goals:** New adapters, new bus channels, mega-compose merging all Dockerfiles, bundled duplicate Redis when bus-core already runs.

---

## Task 1 — Scaffold `services/orion-signals/`

**Files to create:**

| Path | Role |
|------|------|
| `services/orion-signals/roster.v1.yaml` | Tier → service list (core, tier1, tier2, routing) |
| `services/orion-signals/.env_example` | Shared operator contract: `ORION_BUS_URL`, `PROJECT`, `NODE_NAME`, `SIGNALS_TIER` |
| `services/orion-signals/README.md` | Operator guide: tiers, launch, smoke, hub dependency |
| `services/orion-signals/scripts/up.sh` | Start services for selected tier in dependency order |
| `services/orion-signals/scripts/down.sh` | Stop services for selected tier (reverse order) |
| `services/orion-signals/scripts/smoke.sh` | Health checks: bus ping, gateway `/health`, gateway `/signals/active` |

**Roster tiers (from prior design session):**

### `core`
- `orion-bus` → `bus-core`
- `orion-signal-gateway` → `orion-signal-gateway` (+ OTEL sidecars from its compose)

### `tier1` (extends core — primary causal chain + chat)
- `orion-biometrics` → `biometrics`
- `orion-equilibrium-service` → `equilibrium-service`
- `orion-collapse-mirror` → `collapse-mirror`
- `orion-cortex-exec` → `cortex-exec` (and chat lane if separate service exists in compose)
- `orion-recall` → `recall`
- `orion-spark-introspector` → `spark-introspector`
- `orion-memory-consolidation` → (read compose for service name)

### `tier2` (homeostatic consumer)
- `orion-spark-concept-induction` → (read compose for service name)

### `routing` (optional — full turn correlation)
- `orion-cortex-gateway` → `cortex-gateway`
- `orion-cortex-orch` → `cortex-orchestrator` or actual service name
- `orion-llm-gateway` → `llm-gateway`

**Roster entry shape:**
```yaml
- id: biometrics
  compose_dir: orion-biometrics
  compose_service: biometrics
  organ_ids: [biometrics]
  required_env: [ORION_BUS_URL]
```

**Script behavior:**
- Resolve repo root from script location
- Load `services/orion-signals/.env` if present
- Require `ORION_BUS_URL` (no default to bus-core/redis hostnames — operator must set Tailscale IP per AGENTS.md)
- For each roster entry: `docker compose --env-file <service>/.env --env-file services/orion-signals/.env -f services/<compose_dir>/docker-compose.yml up -d <compose_service>`
- Skip service if compose file or service name missing (log warning, don't fail whole stack unless `required: true`)
- `up.sh` accepts tier arg: `core|tier1|tier2|routing|full` where `full` = core+tier1+tier2+routing

**Redis dedupe note in README:** When `core` includes both bus and signal-gateway, set gateway `ORION_BUS_URL` to `redis://bus-core:6379/0` on `app-net` OR use external Tailscale bus — do not run gateway's bundled `orion-redis` alongside `bus-core`. Document `SIGNAL_GATEWAY_SKIP_BUNDLED_REDIS=1` or compose profile if gateway supports it; otherwise document manual step to not start `orion-redis` service.

**Acceptance:**
- All files exist
- `bash -n scripts/*.sh` passes
- README lists exact restart commands

---

## Task 2 — Gate tests

**Files:**
- `services/orion-signals/tests/test_roster.py`
- `services/orion-signals/tests/test_scripts.py`

**Tests:**
1. `roster.v1.yaml` parses; every `compose_dir` has `services/<dir>/docker-compose.yml`
2. Every `compose_service` exists in that compose file (string match)
3. Tier `tier1` includes biometrics, equilibrium, collapse-mirror, cortex-exec, recall, spark-introspector, memory-consolidation
4. Tier `core` includes bus-core and orion-signal-gateway
5. Scripts are executable syntax-valid
6. `.env_example` contains `ORION_BUS_URL`

Run: `pytest services/orion-signals/tests -q`

---

## Task 3 — Cross-link (optional, same commit as Task 2 if small)

Add one line to root `README.md` under observability/signals pointing to `services/orion-signals/README.md`.

---

## Verification gate (AGENTS.md)

```bash
git diff --check
pytest services/orion-signals/tests -q
bash -n services/orion-signals/scripts/*.sh
docker compose --env-file services/orion-signals/.env_example -f services/orion-signals/../orion-bus/docker-compose.yml config >/dev/null  # if applicable
```

No `.env` committed. Sync local `.env` from `.env_example` if template added.
