# catalog_drift_pressure: mesh-wide fix (item 3, Idea 1)

Status: **implemented, this session.** Resolves the Idea-1-vs-Idea-2 fork named as the
"Recommended starting point" in `docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md`
-- the original brainstorm doc's own audit was never actually run; this doc is that audit plus
the fix.

## Arsonist summary

`catalog_drift_pressure` (`orion/substrate/transport_loop/extract.py::compute_transport_pressures()`)
has read `uncataloged_stream_count / streams_observed` since it was introduced -- `streams_observed`
caps at the size of `BUS_OBSERVER_STREAMS` (2 real Redis Streams today), so this channel is
structurally incapable of representing general bus health regardless of how correctly it's wired.
This was already named as "scope dishonesty" in two prior design docs
(`2026-07-22-transport-bus-signal-quality-measurement-design.md`,
`2026-07-23-fcc-motor-field-digester-signals-design.md`'s Appendix item 3) and never fixed.

A separate, already-completed fix decoupled the similarly-named `contract_pressure` from
`catalog_drift_pressure` (they used to be a literal alias) -- confirmed via `git`/test-comment
archaeology, not assumed. That fix did not touch `catalog_drift_pressure`'s own narrow-scope bug at
all; the two are unrelated gaps that happen to share a channel-name-adjacency.

## The audit (resolving the fork)

Grepped every real consumer of `contract_pressure` (the capability-level channel
`catalog_drift_pressure` feeds via `config/field/orion_field_topology.v1.yaml`'s
`node:athena -> capability:transport` edge). Found genuinely wide downstream policy coupling, wider
than the parent doc's "closer to a bug fix" framing suggested:

- `config/attention/field_attention_policy.v1.yaml` -- hardcoded salience weights (0.85, 0.90)
- `config/proposals/proposal_policy.v1.yaml` -- 4 hardcoded gating thresholds (0.50-0.65)
- `orion/consolidation/motif.py` -- `min_contract_pressure: 0.70` motif-trigger gate
- `orion/self_state/transport.py` / `builder.py` -- self-state narrative construction
- `services/orion-hub/scripts/substrate_lattice_routes.py` -- `watch_at` gate thresholds

This is real, calibrated policy surface -- attention, proposal generation, and consolidation are
cognition-loop-adjacent, not an isolated telemetry metric. Flagged to Juniper explicitly before
proceeding (per this repo's CLAUDE.md section 0A proposal-mode discipline for anything touching
cognition-adjacent systems), who chose to proceed now.

**Mitigating factor found during the audit**: Phase 2's real census (this session, earlier) measured
`undeclared_active=0` at real baseline. The new mesh-wide formula and the old narrow formula both
read ~0 under normal conditions -- the swap is unlikely to destabilize existing threshold behavior
day-to-day, only to become honestly non-zero when something is *actually* wrong mesh-wide (which the
old formula could never detect at all, since it only ever sees 2 channels).

## The fix

Replace `uncataloged_stream_count / streams_observed` with
`undeclared_active_count / catalog_size`, sourced from `orion.bus.census.compute_census()` over
`orion.bus.velocity.scan_active_channels()`'s real SCAN of the full `orion:bus:velocity:*`
namespace against the actual ~264-entry declared catalog -- Phase 1+2's own machinery, previously
built but never wired to a consumer.

**Cross-service wiring** (contract -> producer -> consumer, this repo's own prescribed sequencing):

1. **Contract**: `TransportBusStateV1` (`orion/schemas/transport_projection.py`) gains
   `undeclared_active_count: int | None = None` and `catalog_size: int = 0`. `None` (not `0`) means
   "not measured this tick" -- must stay distinguishable from a real, honest zero (this repo's "no
   empty-shell cognition" rule).
2. **Producer**: `services/orion-bus/app/bus_observer.py::_fetch_redis_snapshot()` runs
   `scan_active_channels()` + `compute_census()` when `BUS_OBSERVER_CENSUS_ENABLED=true` (gated --
   the SCAN cost at real mesh scale was an explicitly named, unmeasured question in the parent
   design doc). `ObserverRollup` threads the result through a new
   `BusTransportGrammarCollector.record_bus_census_computed()` atom (`bus_census_computed`),
   counts-only, same "bounded rollups, never per-message content" convention as every other atom in
   that file.
3. **Consumer**: `extract.py`'s `_ATOM_ROLES` gains `bus_census_computed`; the event-parsing loop
   populates the new state fields; `compute_transport_pressures()` prefers the mesh-wide formula
   when available, **falling back to the old narrow formula when not measured** (flag off, or the
   scan itself failed) -- not falling back to `0.0`, which would misrepresent "not measured" as
   "confirmed no drift."

**Live-verified before enabling** (the parent doc's own named missing question, "is a Redis SCAN
over the full namespace, run on a schedule, cheap enough at real mesh scale?"): ran
`scan_active_channels()` + `compute_census()` directly against the real production bus --
`scan_active_channels()`: 154 active channels, 19ms. `compute_census()`: 182ms (208 declared-silent
entries checked against wildcards), `undeclared_active=0`. Total ~200ms against a
`BUS_OBSERVER_POLL_INTERVAL_SEC=10s` default -- cheap, no concern. `BUS_OBSERVER_CENSUS_ENABLED=true`
shipped as the real `.env_example`/`.env` default on the strength of this measurement, not left
gated indefinitely.

## Non-goals

- Not touching `contract_pressure`'s own (already-fixed, unrelated) computation.
- Not changing `catalog_drift_pressure`'s decay mode in field-digester (`NODE_DECAY_CHANNELS`) --
  unaffected, only the upstream source formula changed, not the channel's shape or consumption mode.
- Not building Idea 2 (an additive `bus_channel_undeclared_pressure` channel) -- the fork resolved
  in favor of fixing the existing channel at its source, per the parent doc's own metric-quality-gate
  independence check (shipping both would have been the exact redundancy that gate exists to catch).

## Acceptance checks

- Live-data sanity check: done (SCAN/census cost measured against real traffic before enabling).
- Backward compatibility: confirmed via test (`test_falls_back_to_old_formula_when_census_not_available`)
  -- with the flag off or the atom absent, behavior is byte-identical to pre-fix.
- Decay-mode audit: not needed -- mode unchanged, only the source formula changed.
