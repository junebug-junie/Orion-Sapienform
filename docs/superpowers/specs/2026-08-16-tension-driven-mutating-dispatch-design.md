# Tension-driven mutating dispatch — wiring deviation to real action

Status: implemented, tested, live-config-verified. Ships with its code, same convention as
`docs/superpowers/specs/2026-08-14-field-deviation-tension-sensing-design.md`.

## Arsonist summary

The sensing-layer spec (2026-08-14) built a real, scale-free, deviation-based tension signal —
48.30% admission vs the drives program's 0.064%, a genuine rest point, no hand-authored
cross-channel weights — and deliberately stopped there: *"No bus publication, no schema
registration, no prompt/Mind wiring, no action. Wiring tension to anything that acts is a
separate patch requiring proposal sign-off."*

This is that patch. Juniper's explicit instruction set the shape: no shadow gating (anything
built goes live), no operator-approval gate anywhere in the new path, and real mutation — while
using the perception ladder's `look_at_camera` skill only as a *consumer*, not touching any
perception-ladder file (a different agent owns that workstream).

## Current architecture (before this patch)

- `orion.attention.tension` (merged, read-only): `DeviationGate`, `FieldTensionCompetition`,
  `deviation_pressure` had no live producer — the offline measurement scripts rebuilt a gate
  from scratch on every run, which cannot carry state across a live service's restarts.
- `orion/proposals/scoring.py::PRESSURE_DIMENSIONS`: a closed 4-element set
  (execution/resource/reasoning/reliability_pressure), each backed by a per-tick EWMA precision
  baseline (`services/orion-field-digester/app/digestion/precision.py`) that only tracks
  dimensions `field_pressures()` can produce.
- `orion/execution_dispatch/policy.py::template_to_cortex` (shipped 2026-08-13, PR referenced in
  that file's own comment): a per-template cortex-route override, already generalized past the
  one-verb-per-kind cap — shipped inert (`{}` in the live config) with its own test asserting
  exactly that.
- `skills.runtime.image_prune.v1` and `skills.runtime.docker_prune_stopped_containers.v1`:
  already-shipped mutating verbs (image_prune landed 2026-08-13 as "Orion's second mutating
  action") but **no live template routed to either** — built, never wired. Their own gates are
  NOT equivalent (see §4's disclosed asymmetry, added after code review): image_prune has a real
  measured decline threshold, docker_prune_stopped_containers does not.
- `skills.perception.look_at_camera.v1`: a read-only verb (reads `orion-vision-window`'s
  existing passive projection, does not trigger a capture) with `allowed_scope` never declared
  anywhere, so it was reachable by nothing.

The gap this patch closes was not "build a mutating action" — two already existed unwired. It
was "give Orion's internal deviation state a way to become a proposal, and the proposal a route
to a real, already-gated verb."

## The design

### 1. A live producer for the deviation gate

`services/orion-field-digester/app/digestion/tension.py::update_tension_pressure()` runs once
per digestion tick (wired into `run_digestion_tick()`, after decay/diffusion/suppression, before
`update_dimension_precision_baseline()`):

1. Rehydrate a `FieldTensionCompetition` from `FieldStateV1`'s persisted baseline dicts.
2. Run one real tick through it (`state.node_vectors` fed directly as the tension package's
   `field_json["node_vectors"]` shape).
3. Write `state.tension_deviation_pressure` (the admitted-deviation scalar) and dump the gate's
   updated baselines back onto `state`.

**Persistence contract**, mirroring `dimension_precision_ewma*`'s existing pattern exactly:
`DeviationGate.export_baselines()`/`import_baselines()` (new, pure — no I/O) give a
`(node_id, channel) -> (mu, var, count)` dict; the producer flattens it onto
`FieldStateV1.tension_baseline_mu/_var/_n` with an ASCII unit-separator key
(`f"{node_id}\x1f{channel}"` — node_ids already contain colons and dots). Bounded cardinality (4
nodes × 33 channels = 132 keys, live 2026-08-14) — **not** unbounded per-event growth like the
`evidence_event_ids` TOAST incident CLAUDE.md records; a new node or channel adds a bounded
number of keys, ticks do not. Verified: a producer rehydrated from a persisted-dict snapshot
behaves byte-identically to one that was never torn down
(`test_baseline_persisted_on_state_matches_a_continuous_in_process_run`).

### 2. A scalar that survives contact with PRESSURE_DIMENSIONS

`orion.attention.tension.competition.deviation_pressure(tick)`: the largest single admitted
(channel, node) excess this tick, in z-units past `z_threshold`, saturated at a disclosed
`DEVIATION_PRESSURE_SATURATION = 10.0` and clamped to `[0, 1]`. 0.0 on a quiet tick is a real
"nothing admitted" reading, matching the sensing layer's own "honest limit" design note — never
a fabricated absence.

`orion/field/pressure.py::field_pressures_with_provenance()` injects it directly
(`dims["deviation_pressure"] = clamp01(field.tension_deviation_pressure)`), the same "derived key
outside the generic channel merge" precedent `recent_perturbation_count` already uses — it is not
a raw channel routed through `CHANNEL_DIMENSION_MAP`, so it carries no `DimensionProvenance`
entry (nothing to attribute a channel-merge winner to). **Unlike the other 4 dimensions, it is
always present**, even on an empty field — disclosed and tested
(`test_field_pressures_reports_deviation_pressure_even_on_an_empty_field`), and 4 pre-existing
tests in `test_field_pressure_provenance.py` and `test_dimension_precision_baseline.py` were
updated to assert the new, correct set rather than silently drift.

`orion/proposals/scoring.py::PRESSURE_DIMENSIONS` gains `"deviation_pressure"`, with a **derived**
`DIMENSION_PRECISION_MIN_VARIANCE` floor (`6.165e-4`, 1% of its own measured population variance)
— not borrowed from another dimension, per this repo's own recorded lesson that borrowed
calibrated constants silently re-break across domains.

### 3. Metric quality gate (CLAUDE.md 0A, run before wiring)

Recorded in full in `orion/proposals/scoring.py`'s own comment (restated compactly here):

1. **Provenance** — traced to `update_tension_pressure()`'s own docstring, not assumed.
2. **Independence** — per-`(node, channel)` z-score admission vs the other 4 dimensions' single
   aggregate-value EWMA: different population, different granularity.
3. **Theory anchor** — deviation-from-adapted-baseline admission (standard change detection) +
   Borda rank aggregation (de Borda 1770), both already named in the sensing-layer spec.
4. **Live-data sanity**, 41,973 real `substrate_field_state` ticks (24h, 2026-08-16):

   | | Value |
   |---|---|
   | Nonzero ticks | 49.9% |
   | Distinct values | 18,699 |
   | Population variance | 6.164936e-02 |
   | Mean | 0.1134 |
   | Median | **0.0000** |
   | Max | 1.0000 |
   | Rest→active rises | **4,357** |
   | Producer liveness | refreshed |
   | Decay artifact | none |

   The median-exactly-0.0 line is the bar the `bus_synaptic_prediction_error` incident failed
   (a permanent ~0.27 floor that could never read calm); 4,357 rest→active rises is the bar
   the `node:substrate.route` incident failed (decayed-unopposed, not recurring). Both checked
   by hand against this metric's own numbers, not assumed from the sensing-layer patch's clean
   bill.
5. **Existing mechanism** — searched: the 4 existing `PRESSURE_DIMENSIONS` track a single
   aggregate value's own EWMA; nothing already tracked a per-(node, channel) deviation baseline
   (the sensing-layer spec's own words: *"This patch does not duplicate the existing one"*).
6. **Reversibility** — removing the `PRESSURE_DIMENSIONS` entry (and its floor, kept in sync by
   the module's existing assertion) reverts every downstream reader to today's graceful
   `.get(..., 0.0)` degradation. No schema migration, no data loss elsewhere.

### 4. Three tension-driven templates, three routes, no operator gate

`config/proposals/proposal_policy.v1.yaml` — fixed targets, not attention-bound (binding to the
tension Borda winner is real, separate follow-up work — a second `target_binding` literal plus a
resolver change — deferred rather than built speculatively for this patch):

| Template | Kind | Dimensions | `required_policy_gate` | Route |
|---|---|---|---|---|
| `observe_tension_via_camera` | inspect | `deviation_pressure: 0.55` | read_only | `skills.perception.look_at_camera.v1`, `inspect_only` |
| `prune_dangling_images` | maintain | `deviation_pressure: 0.35`, `resource_pressure: 0.35` | read_only | `skills.runtime.image_prune.v1`, `maintenance_bounded` |
| `prune_stopped_containers` | maintain | `deviation_pressure: 0.35`, `resource_pressure: 0.25` | read_only | `skills.runtime.docker_prune_stopped_containers.v1`, `maintenance_bounded` |

`required_policy_gate: read_only` on the two mutating templates is the same precedent
`prune_build_cache` already set: the field is not a safety claim, the safety comes from three
independent gates in `execution_dispatch_policy.v1.yaml` (`mode.allow_mutating_dispatch` — ON
since 2026-08-12; the `maintenance_bounded` scope check; a third gate that is **not symmetric
between the two routes**, disclosed rather than glossed over after code review: `image_prune`
declines (`declined_nothing_to_reclaim`) unless disk is genuinely ≥75% AND there are genuinely
dangling images — a real measured threshold. `docker_prune_stopped_containers` has no such
count or resource-pressure floor of its own: in execute mode it prunes every exited,
non-label-protected container it finds (optionally age-filtered by `until`), gated only by its
own env flag (`SKILLS_ALLOW_MUTATING_RUNTIME_HOUSEKEEPING`) and label/age filters. An earlier
draft of this doc claimed the two skills had equivalent gates ("real stopped-container count");
that was wrong and is corrected here, not merely softened. **No `operator_review` anywhere in
this path**, per Juniper's explicit instruction — that fact is unchanged by the asymmetry above,
but the asymmetry is a real, larger blast radius for `prune_stopped_containers` specifically than
this doc first represented.

`observe_tension_via_camera` needed no new gate at all: `inspect_only` is unconditionally allowed
in `orion/execution_dispatch/builder.py`'s `scope_allowed` check, independent of
`mode.allow_mutating_dispatch` — verified by a dedicated test
(`test_camera_route_is_unaffected_by_the_mutating_flag`).

`config/execution_dispatch/execution_dispatch_policy.v1.yaml`'s `template_to_cortex` (shipped
inert 2026-08-13) gets its first three real entries. Timeouts staggered against each skill's own
budget, same discipline as `prune_build_cache`'s: `prune_dangling_images` at 720s (matches
`IMAGE_PRUNE_TIMEOUT_SEC=600s` < verb `timeout_ms=660s` < this), `prune_stopped_containers` at
60s (verb `timeout_ms=30s`), `observe_tension_via_camera` at 40s (verb `timeout_ms=20s`).

### 5. "Use the video work as a consumer, don't step on the other agent"

`observe_tension_via_camera` is the wiring for this instruction: it names
`skills.perception.look_at_camera.v1` by reference and scores on `deviation_pressure`, so the
camera-look competes harder when the field is genuinely deviating. Nothing in
`orion/cognition/verbs/`, `services/orion-cortex-exec/app/verb_adapters.py`,
`services/orion-vision-*`, `config/vision_*`, or `config/field/orion_field_topology.v1.yaml` is
touched — the perception ladder's own files are untouched by this patch.

## Files

- `orion/attention/tension/deviation_gate.py` — `export_baselines()`/`import_baselines()`.
- `orion/attention/tension/competition.py` — `DEVIATION_PRESSURE_SATURATION`,
  `deviation_pressure()`, `FieldTensionCompetition.export_baselines()`/`import_baselines()`.
- `orion/schemas/field_state.py` — `tension_baseline_mu/_var/_n`, `tension_deviation_pressure`.
- `services/orion-field-digester/app/digestion/tension.py` (new) — the live producer.
- `services/orion-field-digester/app/tensor/update_rules.py` — wired before the precision update.
- `orion/field/pressure.py` — `deviation_pressure` injected into `field_pressures_with_provenance()`.
- `orion/proposals/scoring.py` — `PRESSURE_DIMENSIONS` + `DIMENSION_PRECISION_MIN_VARIANCE`.
- `orion/proposals/templates.py` — `"maintain"` added to the `ProposalKind` type hint (hygiene;
  the cast was always a runtime no-op) + copy for the 3 new templates.
- `config/proposals/proposal_policy.v1.yaml` — 3 templates, `dimension_weights.deviation_pressure`,
  `limits.max_suppressed` 10 → 20 (see Corrections below).
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml` — 3 `template_to_cortex` entries.
- Tests: `orion/attention/tension/tests/test_deviation_gate.py` +
  `test_competition.py` (persistence + `deviation_pressure` unit tests),
  `services/orion-field-digester/tests/test_tension_pressure_baseline.py` (new),
  `tests/test_tension_driven_dispatch_gating.py` (new), plus updates to 5 pre-existing test files
  where this patch's behavior change was real and needed a new, correct assertion rather than a
  stale one (`test_field_pressure_provenance.py`, `test_dimension_precision_baseline.py`,
  `test_cortex_route_resolution.py`, `test_proposal_policy_loader.py`, `test_proposal_scoring.py`).

## Corrections found by this patch's own test suite

- **`limits.max_suppressed` was already tight, this patch made it visibly so.** 10 was sized
  against roughly 10-13 templates; this patch takes `proposal_templates` to 16, and
  `test_gate_also_suppresses_external_candidates` (a pre-existing test) caught its own fixture
  candidate falling out of the truncated `suppressed_candidates` list — not because anything
  broke, but because 3 more real templates legitimately outranked it for a fixed 10 slots on a
  quiet tick. Raised to 20. `suppressed_candidates` exists so a quiet tick's candidates stay
  inspectable ("Quiet, not lost" — `orion/proposals/builder.py`'s own comment); a cap too tight
  to hold the real template count defeats that purpose silently.
- **`deviation_pressure`'s always-present behavior is a real, disclosed asymmetry** against the
  other 4 `PRESSURE_DIMENSIONS`, which are genuinely absent when no channel maps to them this
  tick. Four pre-existing tests asserted an exact-empty-or-exact-4-key `dims` result and needed
  updating, not working around — each fix states why in its own comment rather than silently
  loosening the assertion.

## Corrections found by code review

- **`_pressure_dimension_ids()`'s empty-`dimensions` fallback silently widened.** Adding
  `deviation_pressure` to `PRESSURE_DIMENSIONS` meant every already-shipped template with an
  honestly-empty `dimensions: {}` (5 of them: `inspect_bus_channel_catalog`,
  `summarize_transport_contract_drift`, `watch_transport_backpressure`,
  `inspect_field_topology_catalog`, `inspect_attended_target`) would start scoring
  urgency/confidence partly off field-deviation tension they were never audited to react to —
  the exact independence-check CLAUDE.md 0A requires tracing to *every* metric already in the
  model, missed in the first pass. Fixed by decoupling the fallback into its own
  `_LEGACY_EMPTY_DIMENSIONS_FALLBACK` constant, frozen to the original 4 dimensions those 5
  templates have always meant — future `PRESSURE_DIMENSIONS` growth no longer silently
  re-scores them.
- **Two mutating skills do not have equivalent gates**, and this doc originally implied they
  did. `image_prune` declines on a real measured threshold; `docker_prune_stopped_containers`
  does not — see §4 and the yaml comments it now points to. Corrected in both config files and
  this doc, not just noted here.
- **The live producer re-parsed static YAML config on every ~2s digestion tick.**
  `FieldTensionCompetition()`'s default `directions` field reloads
  `config/attention/channel_direction_map.yaml` from disk on every construction, and
  `update_tension_pressure()` constructed a fresh one every call. Fixed with a module-level
  `@lru_cache(maxsize=1)` around `load_direction_map()` — the gate baselines still rehydrate
  fresh from `state` every tick (that part is correctness-load-bearing), only the static
  direction map is now cached.

## Non-goals

- No attention-binding to the tension Borda winner. `orion/proposals/builder.py`'s
  `target_binding` mechanism is untouched; the 3 new templates use fixed targets. Binding is real,
  separate follow-up work, named but not built.
- No widening of `hard_blocks` in `execution_dispatch_policy.v1.yaml`. `network_call`/`file_write`
  stay blocked; this patch's two mutating routes clear the existing bar without needing either.
- No perception-ladder file touched. `look_at_camera` is referenced by verb name only.
- No change to `mode.allow_mutating_dispatch`'s existing ON state's justification — this patch
  adds routes behind the same already-open gate, it does not re-litigate opening it.

## Acceptance checks

1. `orion/attention/tension/tests/` — 88 → 100+ tests pass, including new persistence round-trip
   tests proving a rehydrated gate behaves identically to a continuously-running one.
2. `services/orion-field-digester/tests/test_tension_pressure_baseline.py` — producer tests,
   including a direct cross-check against an in-process continuous `FieldTensionCompetition` run.
3. `tests/test_tension_driven_dispatch_gating.py` — the same three-gate proof
   `test_maintenance_dispatch_gating.py` established for `prune_build_cache`, applied to the two
   new mutating routes, plus proof the read-only camera route is unaffected by the mutating flag.
4. `tests/test_cortex_route_resolution.py::test_live_config_declares_exactly_the_shipped_
   template_routes` — pins the live config to exactly the 3 new entries (was `== {}`,
   deliberately updated, same "visible step, not silent drift" precedent that test's own
   docstring already established).
5. End-to-end smoke (not committed, run by hand): a synthetic field warmed through 10 steady
   ticks then a real spike produces `tension_deviation_pressure == 1.0`, `action_warrant_gate ==
   "warranted"`, `prune_dangling_images` proposed at priority 0.32, and `resolve_cortex_route()`
   resolving it to `skills.runtime.image_prune.v1` under `maintenance_bounded` — the full chain,
   traced by hand, not asserted from a unit boundary alone.

## What this does and does not establish

It establishes that Orion's internal deviation state can now become a real, routed, mutating
proposal without an operator-approval step anywhere in the path — the gap the whole tension arc
was chartered to close. It does **not** establish that any of the 3 new templates' scoring
weights, saturation constant, or `max_suppressed: 20` are correctly calibrated; all are disclosed,
uncalibrated starting values in the same style as every other constant shipped in this arc, to be
revisited against real post-deploy dispatch data.
