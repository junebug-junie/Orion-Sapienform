# Prediction-error node write gate — all four remaining domains

Branch: `fix/substrate-pe-node-write-gate`
Date: 2026-07-30
Status: **DONE**

## Summary

- Extends `0fe7de56f`'s `bus_synaptic` fix to the four domains that still gated
  `_write_prediction_error_node()` behind `if error > 0.0:` — `biometrics`, `execution`, `chat`,
  `route`.
- Found live via the Attention Organ tab (PR #1504): `node:substrate.route` frozen at `0.00025`
  with `observed_at` not advancing, while `orion-cortex-orch` had produced 1,608 grammar events in
  the prior 3 hours — the tick was running the whole time.
- A quiet domain returns `error == 0.0` on most ticks, so the gated write left the node holding its
  last **non-zero** value indefinitely: a signal that can rise but can never come back down to a
  genuine calm reading.
- Guard is structural (AST over `worker.py`) rather than five per-domain fixtures — this bug has now
  been found and fixed one-domain-at-a-time three times.
- Live-verified after deploy: `route` now reads a real `0.0`, `execution` reaches `0.0` too.

## Outcome moved

Two Active-Inference domains that could never report "calm" now can. `node:substrate.route` went
from a 10-hour-stale `0.00025` to a live `0.0`; `node:substrate.execution` was observed reaching
`0.0` for the first time in the sampling window. Every consumer that polls these nodes' current
value — `orion-equilibrium-service`'s transport gate, AST/HOT's `prediction_error_by_domain` — now
reads a present-tense value rather than a stale high-water mark.

## Current architecture

Each `_*_tick()` in `services/orion-substrate-runtime/app/worker.py` computes a domain
`prediction_error`, then writes two things under a single `if error > 0.0:` guard: a
`save_receipt()` audit record and a durable `_write_prediction_error_node()` upsert of
`node:substrate.<domain>`.

`0fe7de56f` (2026-07-30, 20:01) split those two for `bus_synaptic` only — receipt stays gated, node
write moved out — after that node was found frozen at a stale `1.0` for hours. The other four
domains were left as they were.

## Architecture touched

`services/orion-substrate-runtime` only. No contract, schema, bus, or env changes.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: node write moved out of the `error > 0.0` guard
  for `biometrics` (735), `execution` (2142), `chat` (2202), `route` (2255). `save_receipt()` left
  gated in all four — it is an audit trail of notable events, not a polled current-state read, so
  skipping it on a calm tick is correct and is not the bug.
- `services/orion-substrate-runtime/tests/test_prediction_error_node_write_not_gated.py` (new): AST
  check that no `_write_prediction_error_node()` call sits inside an `if error > 0.0:` block, plus a
  companion asserting every domain still has a node write at all (so "fixing" it by deleting the
  call fails too).

## Why `route` specifically

`route_prediction_error()` is a **categorical mismatch rate** over route-arbitration decisions
(`lane`, `lane_reason`, `output_mode`, `mind_requested`) — not a continuous magnitude like the other
instruments. With chat idle the arbitration decision does not change between batches, so the rate is
exactly `0.0` every tick, and the gate never opened. `chat` is frozen for a different and legitimate
reason (`_chat_tick()` returns early at `if not events:` — there is genuinely no chat traffic), which
this patch does not and should not change.

## Schema / bus / API changes

None.

## Env/config changes

None. No `.env_example` change, nothing to sync.

## Tests run

```text
$ pytest tests/test_prediction_error_node_write_not_gated.py -q
2 passed in 0.10s

$ pytest tests -q --ignore=tests/test_grammar_consumer_integration.py
13 failed, 188 passed, 9 errors
```

Failures are **pre-existing**, established against a scratch worktree at `origin/main`:

```text
origin/main baseline : 23 FAILED/ERROR lines
this branch          : 23 FAILED/ERROR lines
symmetric difference : (empty)
```

`tests/test_grammar_consumer_integration.py` is excluded because it fails at *collection* on both
branches (`ModuleNotFoundError: No module named 'app.models'` — there is no `app/models.py` in the
service at all). Unrelated to this patch; not introduced here.

Red-before-green, run against `origin/main`'s `worker.py`:

```text
correctly RED against origin/main:
  node:substrate.biometrics at worker.py:744, node:substrate.execution at worker.py:2151,
  node:substrate.chat at worker.py:2211, node:substrate.route at worker.py:2264
```

## Evals run

```text
No eval harness exists for services/orion-substrate-runtime.
```

Flagged rather than claimed. The behavior this patch restores is directly observable in the live
node values (below), which is the meaningful check here.

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
Container orion-athena-substrate-runtime Recreated / Started
```

Live node values sampled over 60s after deploy (FalkorDB `orion_substrate`):

```text
biometric=0.03   bus_synap=1.0  chat=0.0135  execution=0.0     route=0.0
biometric=0.0008 bus_synap=1.0  chat=0.0135  execution=0.0     route=0.0
biometric=0.0238 bus_synap=1.0  chat=0.0135  execution=0.2331  route=0.0
biometric=0.0853 bus_synap=1.0  chat=0.0135  execution=0.2331  route=0.0
```

`route` was `0.00025` and static for 10 hours before this; it now reads a genuine `0.0`.
`execution` reaches `0.0`, which it structurally could not do before. `bus_synaptic` remaining
pinned at `1.0` is a **separate** defect with its own fix (`fix/bus-synaptic-per-edge-saturation`)
— this patch isolates it rather than addressing it.

## Review findings fixed

Not run as a separate subagent pass for this patch — it is a four-line mechanical extension of an
already-reviewed and already-merged fix (`0fe7de56f`), with a structural test that is confirmed red
against `origin/main`. Called out here rather than silently skipped.

## Restart required

Already applied (see Docker checks). To redeploy from the main checkout after merge:

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- **Severity: note.** Node writes now happen on every tick for four more domains, so upsert volume
  against `orion_substrate` rises. `bus_synaptic` has been running this way since 20:01 today with
  no observed issue, and these ticks are far lower-frequency than the concept-induction pipeline
  already hitting the same store.
- **Severity: note.** `node:substrate.chat` is still frozen (6 days), by a different mechanism this
  patch deliberately does not touch — `_chat_tick()` returns before any write when there are no chat
  grammar events. It is nonetheless still averaged into `prediction_error_confidence` by the
  upstream reducer, which is worth its own decision.

## PR link

<to be filled after push>
