# PR: Harden biometrics substrate delta seam for field digestion

**Branch:** `feat/biometrics-substrate-delta-seam-hardening`  
**Base:** `feat/biometrics-substrate-closed-loop-v1`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-biometrics-substrate-delta-seam-hardening`  
**Head:** `f7d17930` (3 commits)

## Summary

Hardens the committed substrate delta seam so `orion-field-digester` can consume `ReductionReceiptV1` / `StateDeltaV1` without duplicate perturbations on replay. Replaces UUID-ish `receipt_id` / `delta_id` with deterministic SHA-256-derived identities, fixes Hub debug to return **node-scoped** emission/receipt lineage (not global latest), and adds receipt lookup for digester prep.

## Problem

- Reducers minted `rcpt_{uuid}` / `delta_{uuid}` — replay/retry produced logically identical deltas with new IDs.
- Hub `GET /api/substrate/biometrics-node/{node_id}/latest` returned the latest **global** organ emission and reduction receipt, not the chain for that node.

## Solution

### Deterministic IDs (`orion/substrate/ids.py`)

- `stable_hash_id(prefix, parts)` → `{prefix}_{sha256[:24]}`
- `stable_delta_id(...)` — reducer, projection, target, operation, sorted `caused_by_event_ids`
- `stable_receipt_id(...)` — reducer, sorted event buckets, optional `emission_id`

Wired in `node_reducer.py` and `pressure_reducer.py` (noop receipts included).

**Digester note:** Pressure receipts include `emission_id` in the preimage (organ still creates a new `emission_id` per invocation). Dedupe across emissions should use **`delta_id`**, not `receipt_id`.

### Node-scoped lineage (`orion/substrate/biometrics_loop/lineage.py`)

- `receipt_touches_node` / `emission_touches_node` / `state_deltas_for_node`
- Hub and `BiometricsSubstrateStore` scan the last 50 rows and filter by node (replacing global `LIMIT 1`).

### Hub API

| Route | Change |
|-------|--------|
| `GET /api/substrate/biometrics-node/{node_id}/latest` | Node-filtered emission/receipt + `committed_state_deltas` |
| `GET /api/substrate/receipts/{receipt_id}` | **New** — load committed receipt by id |

### Runtime store

- `latest_receipt_for_node`, `load_receipt`
- `latest_emission_for_node` uses shared lineage helper

Existing `ON CONFLICT (receipt_id) DO NOTHING` on insert remains the persistence idempotency guard.

## Architecture (unchanged loop)

```text
orion-biometrics → grammar_events → substrate-runtime
  → node_biometrics_projection → biometrics_pressure → active_node_pressure_projection
  → committed ReductionReceiptV1 + StateDeltaV1 (now stable IDs)
```

## Test plan

- [x] `PYTHONPATH=. pytest tests/test_substrate_deterministic_ids.py tests/test_substrate_lineage.py tests/test_node_biometrics_reducer.py tests/test_node_pressure_reducer.py tests/test_biometrics_pipeline.py orion/grammar/tests/test_biometrics_substrate_schemas.py services/orion-hub/tests/test_substrate_biometrics_debug_api.py -q` → **28 passed**
- [ ] Live stack: replay same grammar event / runtime poll — receipt row count does not grow for same logical delta
- [ ] Hub: `curl .../biometrics-node/atlas/latest` shows atlas emission/receipt when a newer global receipt exists for another node

## Verification evidence

```
28 passed in ~1.5s (targeted substrate + hub debug suite)
```

Code review (subagent): **Approved** — spec compliant; added negative lineage hub test post-review.

## Non-goals (confirmed)

- No field digester service, tensor math, new organ, grammar redesign, collector rewrite, UI polish, autonomy actions, cluster aggregate source, `debug_trace` parsing.

## Files touched

| Area | Files |
|------|--------|
| IDs | `orion/substrate/ids.py` |
| Reducers | `node_reducer.py`, `pressure_reducer.py` |
| Lineage | `lineage.py`, `store.py`, `substrate_biometrics_routes.py` |
| Tests | `test_substrate_deterministic_ids.py`, `test_substrate_lineage.py`, reducer/hub tests |
| Docs | `services/orion-substrate-runtime/README.md` |

## Operator notes

No new env vars or SQL migration. Redeploy `orion-substrate-runtime` and Hub after merge. Existing receipts with random IDs remain in DB; new writes use deterministic IDs.
