# Retire `transport_prediction_error()` / `node:substrate.transport` — `bus_synaptic` is the real transport signal now — design spec

Status: **IMPLEMENTED, 2026-07-26.** Juniper gave explicit go-ahead after this proposal-mode doc.
`services/orion-substrate-runtime/app/worker.py::_transport_tick()`'s prediction_error write was
removed; `endogenous_curiosity.py`'s generic node iteration already carries `bus_synaptic` with no
new wiring, confirming this doc's own core claim. See `services/orion-substrate-runtime/README.md`'s
"RETIRED, 2026-07-26" section and the AST/HOT design spec's matching revision for the live writeup.
Recommended-next-patch steps 1-4 below were all completed; `transport_prediction_error()` was kept
(not deleted) per step 3's "cheapest" option, unused by any live caller.

Original proposal-mode status, preserved below for the record: **design mode, not implemented.**
Written per CLAUDE.md's proposal-mode rule (this touches
`orion/substrate/endogenous_curiosity.py`, a rung-5 autonomy-adjacent cognition-loop consumer).
Juniper explicitly asked for this doc before any code changes, after flagging that the old
`transport` Active-Inference domain is illegitimate and `bus_synaptic` is the real signal.

## Arsonist summary

`orion/substrate/prediction_error.py::transport_prediction_error()` was already known-broken and
already excluded from `attention_self_model.py`'s `ACTIVE_INFERENCE_DOMAINS` (see that module's
2026-07-24 comment: flat 0.0 for 100% of an 8h window). What wasn't previously surfaced clearly:
it is not merely excluded and dormant — `services/orion-substrate-runtime/app/worker.py:2112`
still runs a live tick writing `node:substrate.transport`'s `prediction_error` metadata every
cycle, and `orion/substrate/endogenous_curiosity.py` (rung 5, self-seeded curiosity candidates)
generically reads *every* substrate node's `prediction_error` metadata field, so this dead
instrument keeps competing for real candidate-budget slots in a live, enabled consumer.
`bus_synaptic_prediction_error()` (PR #1377, floor-bias fixed in PR #1391) is structurally the
signal `transport` was supposed to be — mesh-wide coverage of real communication anomalies,
where the old instrument only diffed 3 fields from a narrow 2-Redis-Streams census already
documented ~1000x miscalibrated. This spec proposes retiring the old instrument and its tick,
relying on `endogenous_curiosity.py`'s existing generic node-iteration to pick up `bus_synaptic`
automatically — no new wiring needed there.

## Current architecture

- **`transport_prediction_error(prev, curr)`** (`orion/substrate/prediction_error.py:71-98`):
  diffs `stream_backlog_health`, `delivery_confidence`, `stream_backlog_pressure` between two
  `TransportBusProjectionV1` snapshots, `min(1.0, mean(deltas) / 0.30)`. `TransportBusProjectionV1`
  is fed by `BUS_OBSERVER_STREAMS`, which watches exactly 2 real Redis Streams — a structurally
  narrow slice of the mesh, already measured ~1000x miscalibrated against real operating range
  (`docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md`, PR #1275).
- **Live tick**: `services/orion-substrate-runtime/app/worker.py:2112` calls
  `transport_prediction_error()` every cycle and writes the result to `node:substrate.transport`'s
  `prediction_error` metadata via the same shared `_write_prediction_error_node()` helper all six
  domains use.
- **Two contradictory live measurements exist for this node**, both real, from different windows
  -- itself evidence of an unreliable instrument, not a single clean failure mode:
  - `attention_self_model.py`'s 2026-07-24 comment: flat **0.0** for 100% of an 8h window.
  - `endogenous_curiosity.py`'s 2026-07-16 comment: pinned at **1.0** across all 1,428 persisted
    candidate sets over a 24h window (path live since 2026-07-02).
  - Freshly reconfirmed today (2026-07-26, 2h/3528-tick window, direct Postgres query): flat
    **0.0** again. So this instrument alternates between long stuck-low and long stuck-high
    periods -- never meaningfully tracking real transport conditions in either state.
- **`endogenous_curiosity.py`'s `_prediction_error_candidates()`** (line 132) is fully generic: it
  iterates whatever `nodes: Sequence[Any]` its caller passes and treats any node with a
  `metadata["prediction_error"]` above `min_prediction_error` (default 0.55) as a curiosity
  candidate. It does **not** hardcode `node:substrate.transport` or any specific domain name.
- **The real caller**: `worker.py::_endogenous_curiosity_tick()` (line ~1774) passes
  `nodes=list(store.snapshot().nodes.values())` -- literally every node in the live substrate
  graph store. Gated by `s.enable_endogenous_curiosity` (confirmed live/true per the 2026-07-16
  comment's own evidence -- "path live since 2026-07-02, not dormant") and
  `s.endogenous_curiosity_kill_switch` (must be false). The `EndogenousCuriosityConfig(enabled=True,
  ...)` constructed inside this method is not a bypass of the module's own default-off flag --
  it's redundant-but-safe, since the caller already gated on `s.enable_endogenous_curiosity` before
  reaching that line.
- **Consequence of genericity**: `node:substrate.bus_synaptic` already flows into this same
  candidate-generation path today, automatically, with zero code changes -- it's written via the
  identical `_write_prediction_error_node()` helper and lives in the same substrate graph store
  snapshot. The floor-bias fix (PR #1391) already improved its candidate-worthiness (previously it
  would have carried a permanent ~0.27+ floor into this same 0.55-threshold comparison; now it
  genuinely rests near 0 and only crosses 0.55 on real spikes).
- **Guardrails already in place on this consumer** (from the module's own docstring): master flag +
  kill switch, hard per-cycle budget (`HARD_BUDGET_CEILING = 8`), candidates target
  `concept_graph` only (never the strict self/relationship zone, never autonomy directly), output
  is signals-only -- anything that proposes real change goes through rung-6 governance
  (trials + rollback). A bad candidate from a broken `transport` node cannot directly cause harm on
  its own; it wastes budget slots that a real signal (like `bus_synaptic`) could otherwise win.
- **Out of scope / separate system**: `TransportBusProjectionV1`'s raw `stream_backlog_health` /
  `stream_backlog_pressure` / `delivery_confidence` fields feed a much wider set of consumers
  unrelated to Active-Inference prediction-error scoring -- field-digester's
  `services/orion-field-digester/app/tensor/channels.py`, `orion/consolidation/motif.py`,
  `orion/field_coherence.py`, `services/orion-hub/scripts/substrate_lattice_routes.py`'s UI, and
  `catalog_drift_pressure` (a *different*, already-fixed metric under the same "transport" umbrella,
  PR #1373 -- unrelated to this instrument). None of that is touched by this proposal.

## Missing questions

1. Does `worker.py:2112`'s transport tick do anything else useful besides feeding
   `transport_prediction_error()` -- i.e. does removing the tick orphan any other consumer of
   `node:substrate.transport`'s existence (not just its `prediction_error` field)? Grep at
   proposal-mode time found no other consumer of that specific node id besides the two named here,
   but this should be re-confirmed immediately before implementation, not assumed stale from this
   doc.
2. Root cause of the flat-0.0 vs. pinned-1.0 alternation is not diagnosed here -- only observed.
   Given the instrument is being retired rather than repaired, this is treated as a non-goal, but
   worth naming: it suggests `TransportBusProjectionV1`'s underlying `stream_backlog_health` field
   is itself a discretized/toggling signal (schema default paths in
   `orion/substrate/transport_loop/extract.py` show three literal values: 1.0, 0.0, 0.5 -- not a
   continuous measurement), which would explain both failure modes without needing two separate
   root causes.
3. Should `node:substrate.transport` be deleted from the live graph store, or just stop being
   written to going forward (stale node ages out via existing decay/pruning)? Deleting live data
   is a bigger, separate decision from stopping a tick.
4. Does `min_prediction_error=0.55` (endogenous curiosity's own threshold) still make sense once
   `bus_synaptic` is a realistic candidate source, or was it implicitly tuned against a world where
   only the pinned-at-1.0 `transport` node ever crossed it? Not evaluated here.

## Proposed schema / API changes

- **Remove** `transport_prediction_error()` from `orion/substrate/prediction_error.py` (or leave
  the pure function in place, unused, if there's value in keeping it importable for historical
  replay scripts -- `scripts/analysis/measure_transport_bus_signal_history.py` and
  `scripts/analysis/measure_transport_biometrics_prediction_error_correlation.py` both reference it
  by name for measurement purposes, not live consumption; deleting the function would break those
  scripts' imports).
- **Stop** the `worker.py:2112` tick call so `node:substrate.transport`'s `prediction_error`
  metadata stops being refreshed. Existing node ages out under the same staleness decay
  `endogenous_curiosity.py` already applies to every node (`_prediction_error_staleness_decay`).
- **No change needed** to `endogenous_curiosity.py` itself -- its generic node iteration already
  picks up `bus_synaptic` once the `transport` node stops dominating.
- **No change** to `attention_self_model.py` -- `transport` is already excluded from
  `ACTIVE_INFERENCE_DOMAINS` there; this proposal only stops the tick that feeds its raw dict entry
  (which was already documented as reported "honestly" via the confidence basis string, not
  silently dropped).

## Files likely to touch

- `services/orion-substrate-runtime/app/worker.py` (remove or flag-gate the tick call at line 2112)
- `orion/substrate/prediction_error.py` (mark `transport_prediction_error()` deprecated or remove,
  pending answer to Missing Question 1's script-import check)
- `orion/substrate/tests/test_prediction_error.py` (update/remove `transport_prediction_error` tests)
- `services/orion-substrate-runtime/tests/` (remove/update the transport-tick test, if one exists
  covering this specific call)
- `orion/substrate/endogenous_curiosity.py` (docstring/comment update only -- the 2026-07-16
  "Live-confirmed" comment referencing the old pinned-at-1.0 finding should be updated to note it's
  resolved, not deleted, so the historical incident stays traceable)
- `services/orion-substrate-runtime/README.md` (dated revision section, matching this session's
  established pattern)
- `docs/superpowers/specs/2026-07-22-l6-self-model-ast-hot-active-inference-design.md` (note the
  `transport` domain is now fully retired, not merely excluded)

## Non-goals

- Not touching `TransportBusProjectionV1` / `stream_backlog_health` / `stream_backlog_pressure` /
  `delivery_confidence` themselves, or any of their consumers in field-digester, motif detection,
  or the hub UI -- that's a separate, much larger raw-pressure-channel system with its own
  independently-tracked health issues, out of scope here.
- Not touching `catalog_drift_pressure` (already fixed, PR #1373, unrelated metric).
- Not root-causing *why* `TransportBusProjectionV1`'s fields toggle between discrete values --
  named as a missing question, not solved here.
- Not re-tuning `endogenous_curiosity`'s `min_prediction_error=0.55` threshold -- named as a
  missing question, not solved here.
- Not deleting the live `node:substrate.transport` row outright (vs. letting it age out) --
  deferred to implementation-time judgment per Missing Question 3.

## Acceptance checks

1. `grep -rn "transport_prediction_error" orion/ services/ scripts/` after the patch shows only
   the (possibly-deprecated-but-kept) function definition and the two measurement scripts that
   reference it for historical replay -- no live tick call remains.
2. `docker logs orion-athena-substrate-runtime` post-deploy shows the transport tick no longer
   firing (or firing behind an explicit off-by-default flag, if a flag-gate approach is chosen
   over outright removal).
3. Live query against `substrate_field_state` post-deploy: `node:substrate.transport`'s
   `prediction_error` no longer updates (its `generated_at` in the field snapshot stays frozen at
   the last pre-patch tick), while `node:substrate.bus_synaptic` continues updating normally.
4. `endogenous_curiosity.py`'s persisted candidate sets (`substrate_endogenous_curiosity_candidates`
   or equivalent store) show `bus_synaptic`-sourced candidates appearing over a real post-deploy
   window, not just `transport`-sourced ones as before -- confirms the generic node-iteration claim
   in this doc, not just an assumption.
5. Full test suite for touched files passes.

## Recommended next patch

1. Confirm Missing Question 1 (no other live consumer of `node:substrate.transport`'s existence)
   via a fresh repo-wide grep immediately before implementation, not reused from this doc.
2. Remove the `worker.py:2112` tick call (simplest, most aligned with "kill means kill" -- no
   fallback to the broken signal).
3. Decide via a quick confirm with Juniper: keep `transport_prediction_error()` as an importable,
   unused pure function (cheapest, keeps the two measurement scripts working) vs. delete it and fix
   those two scripts' imports (cleaner, slightly more invasive).
4. Update `endogenous_curiosity.py`'s 2026-07-16 comment to note resolution, run the full test
   suite, run the acceptance checks above against live data, write the dated README/doc revisions,
   PR per the usual template.
