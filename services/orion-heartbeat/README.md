# orion-heartbeat (v0)

A small, real tensor-network substrate (matrix product state, via `quimb`) that
tests whether Orion's grammar-event stream exhibits holographic-style
boundary/bulk entanglement structure. Read-only, additive, publishes nothing
to any existing consumer.

Full design record, including the pivots that got here (three prior brainstorm
rounds, a shelved 2026-05-01 research charter, and why the first two attempts
at "Phase 0" were corrected before landing on this):
`docs/superpowers/specs/2026-07-24-spark-field-holographic-lattice-design.md`.

## What it does

1. Subscribes to the existing `orion:grammar:event` stream (`GrammarEventV1`)
   — no bespoke per-organ ingestion, reuses what
   `services/orion-substrate-runtime` already standardized.
2. Filters to five confirmed-live organs (chat/`orion-hub`,
   biometrics/`orion-biometrics`, execution/`orion-cortex-exec`,
   transport/`orion-bus`, route/`orion-cortex-orch`) — see
   `app/substrate/routing.py`'s `ORGAN_SITE_MAP`.
3. Routes each `atom_emitted` event's atom onto one of 10 MPS sites (5
   boundary, one per organ; 5 bulk, touched only via entanglement
   propagation) and applies a small local absorb-and-entangle update,
   bond-dimension capped at 4.
4. Periodically computes the MPS's bipartite entanglement entropy at every
   cut along the chain — the boundary/bulk cut (site 5) is the headline H1
   result: how much the boundary and bulk are actually entangled, the real
   (not statistical) signature of holographic-style encoding.

## What it deliberately does not do (v0 scope)

- No active-inference free-energy minimization (the 2026-05-01 charter's
  original update rule) — a much simpler local entangling update is used
  instead; see `app/substrate/mps_state.py`'s module docstring.
- No literal "partial trace + max-entropy completion + quantum fidelity" (the
  charter's literal H1 formula) — confirmed this session to be either
  near-tautological for a pure global MPS state (boundary/bulk reduced
  density matrices share an identical spectrum) or computationally too
  expensive for a tick loop at reasonable subset sizes. Uses the MPS's native,
  cheap bipartite entanglement entropy instead; see
  `app/substrate/reconstruction.py`'s module docstring for the full
  reasoning.
- No H2 (cross-organ mutual information), H3 (intervention propagation), H4
  (predictive surprise), shadow-comparison against `orion/spark/orion_tissue.py`,
  ablation baseline, or formal pre-registration process.
- No modification to `FieldStateV1`, `orion-field-digester`, or
  `orion/spark/orion_tissue.py` — this is a wholly separate, additive
  consumer of an existing stream.
- No `SelfStateV1` dependency anywhere.

## Run

```bash
cp services/orion-heartbeat/.env_example services/orion-heartbeat/.env
python scripts/sync_local_env_from_example.py orion-heartbeat
```

Then via `scripts/safe_docker_build.sh` (per CLAUDE.md; do not call `docker
compose` directly from the shared checkout):

```bash
scripts/safe_docker_build.sh orion-heartbeat up -d --build
curl -fsS http://localhost:7251/health
curl -fsS http://localhost:7251/h1
```

`/h1` returns `{"ok": false, "reason": "no_h1_computed_yet"}` until
`HEARTBEAT_H1_INTERVAL_SEC` (default 30s) has elapsed since start.

## Debug surfaces

- `GET /health` — service status + absorption/skip counters + substrate bond
  dimension/norm sanity checks.
- `GET /h1` — latest H1 result: full entanglement-entropy profile, the
  boundary/bulk headline number, its ratio against the bond-dimension's
  theoretical maximum, and an explicit verdict (`redundant` / `concentrated`
  / `mixed`) — the verdict thresholds are provisional, not calibrated against
  any real baseline yet (v0 has no precedent run the way the phi encoder's
  thresholds did).
