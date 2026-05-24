# PR: Node-scoped biometrics grammar ingress

**Branch:** `feat/biometrics-node-grammar-ingress`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-biometrics-node-grammar-ingress`

## Summary

- Publishes per-node `GrammarEventV1` traces on `orion:grammar:event` from existing biometrics sample/summary/induction (not cluster aggregate).
- YAML node catalog canonicalizes aliases (e.g. `prometheous` → `prometheus`) and attaches role, capability, and `expected_online` context atoms.
- Grammar publish failures are non-fatal; normal biometrics telemetry continues.

## Architecture

```text
collect_biometrics()
  → BiometricsSampleV1
  → BiometricsPipeline.update()
  → BiometricsSummaryV1 + BiometricsInductionV1
  → publish sample/summary/induction (unchanged)
  → NodeCatalog.resolve(node)
  → build_biometrics_node_grammar_events()
  → publish grammar.event.v1 × 13 per tick on orion:grammar:event
```

Trace unit: **one node, one observed biometrics moment** (`biometrics.node:{node_id}:{timestamp}`).

## Changes by area

### `config/biometrics/node_catalog.yaml`
- Canonical nodes: atlas, athena, circe, prometheus with Tailscale aliases and typo aliases.

### `services/orion-biometrics`
- `app/node_catalog.py` — resolve raw node → `NodeProfile`
- `app/grammar_emit.py` — five atoms, six edges, trace lifecycle
- `app/main.py` — optional grammar publish after induction
- `app/settings.py` — `PUBLISH_BIOMETRICS_GRAMMAR`, `GRAMMAR_EVENT_CHANNEL`, `NODE_CATALOG_PATH`
- `tests/` — 10 unit tests (catalog + emitter schema/trace shape)
- `Dockerfile` / `docker-compose.yml` — catalog mount + env
- `README.md`, `.env_example`

### `orion/bus/channels.yaml`
- `orion-biometrics` added as producer on `orion:grammar:event`

### `scripts/smoke_biometrics_grammar.sh`
- Host unit tests + optional container pytest + redis tap instructions

## Test plan

- [x] `PYTHONPATH=services/orion-biometrics:. pytest services/orion-biometrics/tests/ -q` → **10 passed**
- [ ] `redis-cli SUBSCRIBE orion:grammar:event` after biometrics tick → trace_id `biometrics.node:*`, `provenance.source_service=orion-biometrics`
- [ ] sql-writer ingests `grammar.event.v1` (if stack running)

## Verification evidence

```
10 passed in 0.23s (worktree, repo .venv)
```

Runtime bus subscribe: **UNVERIFIED** in CI/agent environment (no live biometrics container + redis tap in this session).

## Acceptance criteria

| # | Criterion | Met |
|---|-----------|-----|
| 1 | Atlas trace `biometrics.node:atlas:...` | Yes (emitter + catalog tests) |
| 2 | Separate Athena trace per deployment | Yes (per `NODE_NAME` / sample.node) |
| 3 | Circe `expected_online=false` | Yes (catalog + availability atom test) |
| 4 | `prometheous` → `prometheus` | Yes |
| 5 | Valid GrammarEventV1 fields only | Yes (literal membership tests) |
| 6 | `payload_ref` only, no blob fields | Yes |
| 7 | Grammar failure non-fatal | Yes (`try/except` in `publish_metrics`) |
| 8 | No new organ | Yes |
| 9 | No cluster grammar source | Yes |
| 10 | Distinct capability surfaces | Yes (athena vs atlas test) |

## Non-goals (confirmed out of scope)

- New organ, mesh-average grammar, cluster-as-source, UI, signal-registry redesign.
