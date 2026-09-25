# Substrate lattice audit (transport lane, M1–L11) — 2026-09-22

Status: audit committed 2026-09-25 alongside the config cleanup it motivated
(branch `chore/substrate-lattice-config-cleanup`). Live numbers below were pulled
from production Postgres (`conjourney`) and live container env on 2026-09-22/23
unless another date is given.

## Arsonist summary

The transport lane is the only lane with a full proof ladder, and it is mostly a
pipe carrying nothing. The bottom (grammar atoms, M1–M2) is healthy and busy. The
middle (M3 reducer, M4 field) runs but mostly watches two narrow world_pulse
streams, so it reads near-zero almost all the time. The top (L7–L11) fires on a
timer regardless of what the bottom says, and the one learning signal at the end
(the `transport_contract_drift_loop` motif) needs a reading 40x higher than
anything seen in a week. Around it sat two config files read by nothing, plus
unread keys in two more, one of which (`dimension_weights`) had a hand-copied
mirror in the Hub keyed on a channel retired two months earlier. Config being present was standing in
for behavior.

## Current architecture

Per rung, with a verdict.

| Rung | What it is | Evidence | Verdict |
|------|-----------|----------|---------|
| M1–M2 | `bus.transport:*` grammar atoms from orion-bus, persisted by sql-writer | ~108k bus atoms/day | **Good.** |
| M3 | `transport_bus_reducer` → `substrate_transport_bus_projection` | Live, but 7 of 9 channels watch only the two `orion:stream:world_pulse:*` streams in `BUS_OBSERVER_STREAMS`; `observer_failure_pressure` and `contract_pressure` exactly 0 for 7 days | **Alive, mostly blind.** |
| M4 | `capability:transport` field vector | 7-day `pressure` mean 0.039, max 0.354; ≥0.25 in 0.98% of 121k ticks; never ≥0.5 | **Alive, low-dynamic-range.** |
| M5 | Attention frame (capability targets novelty-scored) | Dead 2026-09-20 21:58Z → 2026-09-23 (`FieldStateV1` `extra="forbid"` schema skew rejected frames); now fixed | **Live again; the outage was invisible to every flag.** |
| L6 | `SelfStateV1.transport_integrity` | Producer `orion-self-state-runtime` and `config/self_state/` deleted 2026-07-22 (`bcc72f6a0`) | **Gone.** Hub route/UI already dropped it; docs and smoke script still described it (fixed here). |
| L7 | Transport proposal templates | Templates have empty `dimensions`; fired ~1,200/h regardless of state | **Timer, not a response.** |
| L8–L10 | Policy / dispatch (`dispatch_read_only`) / feedback | Running | **Plumbing only; nothing transport-specific to decide on.** |
| L11 | `transport_contract_drift_loop` motif | Needs `contract_pressure ≥ 0.70`; observed 7-day max 0.018–0.022; never fired in 30 days. The motif (`orion/consolidation/motif.py`) reads M4's `contract_pressure`, which the topology fills from `catalog_drift_pressure` — not M3's reducer `contract_pressure` (exactly 0 all week) | **Unreachable.** |

Cross-cutting findings:

- **Dead topology edge.** `config/field/orion_field_topology.v1.yaml` maps
  `capability:transport → capability:orchestration` on `stream_backlog_pressure`,
  a channel never written on `capability:transport`. The edge carries nothing.
- **Three vocabularies for one signal.** The reducer says `stream_backlog_pressure`,
  the policy says `bus_synaptic_pressure`, the field says `pressure`. Related: the
  Hub's contract gate reads M4 `contract_pressure`, which the topology fills from
  `catalog_drift_pressure`, not from the reducer's own `contract_pressure`.
  `transport_healthy_idle` (consolidation policy) still keys on the retired
  `stream_backlog_*` census.
- **Heartbeat** holds an MPS (matrix product state), not MCPs; 5 hardcoded organ
  sites; emits nothing.
- **Grammar producer catalog was wrong.** `orion:grammar:event` in
  `orion/bus/channels.yaml` listed orion-vision-retina/-edge/-window (no grammar
  code, zero rows in 60 days) and omitted orion-substrate-runtime (12.7k rows/7d).
  Live 7-day `grammar_events` by `source_service`: biometrics 812k, bus 321k,
  cortex-exec 266k, harness-governor 46k, substrate-runtime 12.7k, cortex-orch
  11.4k, hub 1.6k, gpu-pool 5.
- **Unread config** under `config/substrate-lattice/`:
  - `action_ceiling_policy.v1.yaml` — zero loaders.
  - `grammar_producer_registry.v1.yaml` — zero loaders, and wrong (biometrics and
    cortex-orch "planned" though live; retired channel names; missing hub,
    harness-governor, substrate-runtime, gpu-pool).
  - `transport_lattice_policy.v1.yaml` `dimension_weights` — read by no process.
    Its only "application" was `_TRANSPORT_CHANNELS` in
    `services/orion-hub/scripts/substrate_lattice_routes.py`, a hand copy keyed on
    retired `stream_backlog_pressure`, so the simulator silently used hardcoded
    thresholds. Worse, the simulator read M3's top level, which has no pressure
    fields, so it always saw 0.0. Same for the Lattice Values panel and the M3
    card. `required_windows` and `healthy_idle` were also unread.
  - `gate_policy.v1.yaml` — only `freshness` and `evidence` are read;
    `confidence`, `lineage`, `action_ceiling` were not.

Candidate organs for the next lane, ranked: llm-gateway, sql-writer, recall,
vision self-report, equilibrium, heartbeat H1 verdict.

## Missing questions

1. Should `capability:transport → capability:orchestration` carry `pressure`
   (bus_synaptic) instead of the dead channel? That changes orchestration
   pressure live, so it needs the metric quality gate first, not a rename.
2. What should L7 transport templates actually respond to? Filling `dimensions`
   changes proposal-loop behavior — proposal mode first.
3. Is the L11 0.70 threshold wrong, or is `contract_pressure` the wrong input?
   Its input has been exactly 0 for a week; lowering the bar on a flat signal
   would make a motif out of noise.
4. Should `BUS_OBSERVER_STREAMS` widen past the two world_pulse streams, or
   should the 7 census channels be retired in favor of bus_synaptic?

## Proposed schema / API changes

Shipped on this branch:

- Deleted `config/substrate-lattice/action_ceiling_policy.v1.yaml` and
  `config/substrate-lattice/grammar_producer_registry.v1.yaml`.
- `transport_lattice_policy.v1.yaml`: removed `dimension_weights`,
  `required_windows`, `healthy_idle`. `channels:` is now the single source.
- `gate_policy.v1.yaml`: removed the unread `confidence`/`lineage`/`action_ceiling` keys.
- Hub `substrate_lattice_routes.py`: deleted `_TRANSPORT_CHANNELS`. The simulator,
  Lattice Values panel and simulator inputs read `channels:` from the YAML.
  Values come from where they actually live: `bus_synaptic_pressure` from M4
  `capability:transport.pressure`, the rest as max over M3's per-bus rows whose
  own `observed_at` is fresh (so a stopped or phantom bus row cannot hold the max).
  **Salience is now the strongest reading among channels at or above their watch
  threshold** (0.0 if none), not a weighted sum. Unmeasured channels (stale/missing
  layer) never promote and are listed, never read as 0.0. `/transport/latest`
  gains `lattice_channels`; `/transport/simulate` gains `promoted_channels`,
  `unmeasured_channels`, `ignored_thresholds`, `channel_values`.
- Hub lane rail: biometrics/execution no longer say "planned"; route lane added.
- `orion/bus/channels.yaml` `orion:grammar:event` producers fixed to match
  reality, gated by `tests/test_grammar_event_producer_catalog.py` (static AST
  scan of `GrammarProvenanceV1(source_service=...)` sites), wired into
  `orion-static-gates` CI.
- L6 removed from `docs/transport_substrate_proof_ladder.md` and the full-stack
  smoke script; ladder doc records the production flag state.

Sibling PRs (other agents):

| Branch | Fixes |
|--------|-------|
| `fix/substrate-ladder-liveness-gate` | Freshness + schema-skew gate across the ladder (the M5 outage class) |
| `fix/transport-rpc-timeout-phantom-node` | Phantom `bus:rpc_timeout` / `node:rpc_timeout` from the rpc-timeout grammar marker |
| `feat/route-lane-field-digestion` | Route lane into field digestion |
| `feat/llm-gateway-grammar-lane` | llm-gateway as a grammar producer (top-ranked candidate organ) |

## Files likely to touch

Done here: `config/substrate-lattice/*`, `services/orion-hub/scripts/substrate_lattice_routes.py`,
`services/orion-hub/static/{substrate-lattice.html,js/substrate-lattice.js}`,
`orion/bus/channels.yaml`, `tests/test_grammar_event_producer_catalog.py`,
`docs/transport_substrate_proof_ladder.md`, `scripts/smoke_orion_bus_transport_full_stack.sh`.

Open items would touch `config/field/orion_field_topology.v1.yaml`,
`config/proposals/proposal_policy.v1.yaml`, `config/consolidation/consolidation_policy.v1.yaml`,
`services/orion-bus/.env_example`.

## Non-goals

- Not deleting `orion/schemas/self_state.py`: 8+ modules still import it.
- Not deleting the untracked, gitignored `services/orion-self-state-runtime/`
  leftover in the primary checkout (only `.env` and `__pycache__`); that is an
  operator `rm -rf` and needs Juniper's say-so.
- No change to what any runtime service computes. Every change here is config
  nobody read, Hub display/simulator, catalog, docs, or tests.

## Acceptance checks

- `rg "grammar_producer_registry|action_ceiling_policy|dimension_weights" config services orion`
  finds only explanatory comments and the unrelated proposal-policy `dimension_weights`
  (`config/proposals`, `orion/proposals`) — no loader.
- `pytest services/orion-hub/tests/test_substrate_lattice_routes.py services/orion-hub/tests/test_substrate_lattice_hub_tab.py`
  green, including kill-means-kill regressions (no `_TRANSPORT_CHANNELS`, no weights, UI has no hardcoded policy).
- `pytest tests/test_grammar_event_producer_catalog.py` green, and red when a
  phantom producer is re-added or a real one removed (verified by mutation).
- `services/orion-mind` recall resolver tests green (still reads `bus_synaptic_pressure` rungs).
- After Hub restart: Lattice Values shows four channels with real values and
  their `value_source`; simulator inputs render from the policy.

## Open items — not fixed by any PR, with verdicts

| Item | Verdict |
|------|---------|
| Dead `capability:transport → capability:orchestration` edge on `stream_backlog_pressure` | **Keep open; needs the metric quality gate.** Rewiring to `pressure` changes orchestration pressure live. Deleting the edge is the cheap honest step if no one wants to run the gate. |
| L7 transport templates with empty `dimensions`, firing ~1,200/h | **Keep open; needs proposal mode.** Changes proposal-loop behavior. |
| L11 `transport_contract_drift_loop` threshold 0.70 vs observed max 0.018–0.022 | **Keep open.** Its input is really catalog drift (via M4), not contract mismatches. Decide which signal the motif should read (vocabulary item below) before touching the bar; lowering it on a near-flat signal makes a motif out of noise. |
| M3 observer covers only two world_pulse streams | **Keep open.** Either widen `BUS_OBSERVER_STREAMS` or retire the 7 census channels in favor of bus_synaptic — pick one, not both. |
| Hub contract gate reads M4 `contract_pressure` (fed by `catalog_drift_pressure`); pressure gate compares M4 `reliability_pressure` (= max(observer_failure, 1 − delivery_confidence)) against the `observer_failure_pressure` threshold | **Keep open; vocabulary bug.** Display-only. This branch relabels both gate reasons with what they actually read, but the gate overlay and the Lattice Values panel (M3 per-bus values) can still disagree under the same channel name until the three vocabularies are unified. |
| `transport_healthy_idle` motif keyed on retired `stream_backlog_*` | **Keep open.** Same census retirement decision as above. |
