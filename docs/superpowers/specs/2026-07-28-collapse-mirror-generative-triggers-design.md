# CollapseMirror generative triggers (insight + flow) — design spec

Status: **design mode, not implemented.** Touches the metacog/collapse-mirror cognition loop, which
CLAUDE.md §0A requires explicit proposal mode for before implementation. This document proposes; it
does not build.

## Arsonist summary

A same-day brainstorm (2026-07-28) on the transport-metacog-trigger arc noted that every `trigger_kind`
built this year — `telemetry_anomaly`, `chat_turn`, `transport`, `relational` (repair_pressure) —
fires exclusively on a problem: a timeout, an anomaly, a rupture. `orion/schemas/collapse_mirror.py`'s
own v1 `DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE` taxonomy was never error-only — `flow→stabilizing` and
`epiphany→reorientation` sit alongside `turbulence→escalating` and `glitch→anomaly_detected`. The one
live counter-example, `pulse` (fires on landing-pad salience crossing a threshold — a positive-valence
signal, `service.py:803`), proves a non-error trigger already works in the exact same pipeline.

Juniper picked two ideas from that brainstorm to spec: **surprise-resolution ("insight")** and
**flow-state**. A third idea (drive-tension resolution) was explicitly ruled dead in its original
form — DriveEngine's tension-fold mechanism is being retired/reimagined (five-drive audit, autonomy
retired as a drive), so this spec does not build it. Juniper flagged that the Sentience Striving
Program's AST/HOT self-modeling work might supply a real replacement signal for "an internal tension
resolved," but named no specific one yet — tracked as an open question below, not assumed. Three other
brainstormed ideas (concept-bridge trigger, relational-resonance trigger, pulse-threshold widening)
were explicitly declined.

## Current architecture

- **`MetacogTriggerV1`/`orion_metacog` pipeline**: unchanged shape from the `chat_turn`/`transport`
  precedent — `orion-equilibrium-service` evaluates a gate condition per `trigger_kind`, publishes
  `CHANNEL_EQUILIBRIUM_METACOG_TRIGGER`, `orion-cortex-exec` drafts a `CollapseMirrorEntryV2` via LLM,
  `orion-sql-writer` persists to `orion_metacog`. `trigger_kind` is a free string field
  (`orion/schemas/telemetry/metacog_trigger.py`); adding a new kind needs no schema migration.
- **Six prediction-error domains** feed `ACTIVE_INFERENCE_DOMAINS = {"execution", "biometrics", "chat",
  "route", "bus_synaptic"}` (`orion/substrate/attention_self_model.py:90`). `transport` was retired
  from this set 2026-07-26 (see that day's retirement spec).
- **`_aggregate_prediction_error_confidence()` / `_unconditional_prediction_error_confidence()`**
  (`attention_self_model.py:97-176`): computes `confidence = 1.0 - mean(prediction_error)` across the
  active domains, populating `AttentionSelfModelV1.confidence` /
  `.prediction_error_confidence`/`.prediction_error_confidence_basis`. This is the single already-built
  aggregate that idea "insight" would key off — no new computation needed if it's real.
- **`reduce_attention_self_model()` live-tick status: UNVERIFIED, not assumed live.** Grepping the
  whole repo, this reducer is called from `orion/substrate/tests/test_attention_self_model.py` (unit
  tests) and `scripts/analysis/measure_ast_hot_reducer.py` (an offline analysis script) only. No live
  service (`services/orion-substrate-runtime/app/worker.py` or otherwise) was found calling it in this
  session's grep. Project memory (`self_modeling_ladder` status) says AST/HOT rungs are merged but
  "ignition flags off" — consistent with a built-but-not-ticking reducer. **Do not treat
  `AttentionSelfModelV1.confidence` as a live producer until Missing Question 1 is answered.**
- **`pulse` precedent** (`services/orion-equilibrium-service/app/service.py:796-813`): fires
  `trigger_kind="pulse"` when a landing-pad signal's `salience >= EQUILIBRIUM_METACOG_PAD_PULSE_THRESHOLD`.
  Positive-valence, live, already dispatches through the identical pipeline this spec would reuse.
- **`relational` precedent** (`repair_pressure_metacog_gate.py:21`): fires on `repair_pressure_v2`
  level/confidence — explicitly rupture-shaped ("how much repair is needed"), confirming the pattern
  Juniper is asking to break out of.
- **Flow-state candidate signal, not yet identified.** No existing field was confirmed this session as
  "sustained low-variance/low-distress self-state coherence." Candidates not yet distinguished from each
  other: `self_state`-derived fields (previously ruled out for phi/IIT-adjacent work per
  `[[feedback_field_native_not_selfstate]]` — same caution likely applies), `FieldStateV1` coherence
  channels, or `AttentionSelfModelV1.confidence` itself read as a rolling window rather than a
  point-in-time value. Not resolved here — see Missing Question 3.
- **Drive-tension resolution (original idea 3): confirmed dead ground.** No dedicated "tension
  resolved" event exists in code (grepped `orion/autonomy/`); the phenomenon was only ever observed
  as an emergent property of `orion/spark/concept_induction/bus_worker.py::_update_drive_pressures`'s
  fold mechanism, documented post-hoc in a board finding (`d3739187`), not instrumented as a trigger.
  Per project memory, the drive taxonomy this depended on is itself being retired/reimagined.

## Missing questions

1. **Is `reduce_attention_self_model()` actually invoked on a live cadence anywhere**, or does
   `AttentionSelfModelV1` only get computed by the offline measurement script? This is the single
   highest-leverage unknown — it determines whether "insight" is "wire a new gate onto an existing live
   producer" (cheap) or "first make the AST/HOT reducer tick live, then build the gate" (a materially
   bigger patch, and its own proposal-mode question). Check `services/orion-substrate-runtime/app/worker.py`
   end to end, and check for any scheduler/cron path in `services/orion-hub` or elsewhere that might call
   it outside the grep pattern used this session.
2. **What is the real historical shape of confidence recovery / prediction-error drops in stored data?**
   Does a sharp rise in `confidence` (or drop in `mean(prediction_error)`) happen as discrete, meaningful
   events, or is it smooth/continuous with no natural crossing to trigger on? Needs a measurement script
   (same discipline as `scripts/analysis/measure_rpc_health_baseline.py`) against whatever
   `AttentionSelfModelV1` history already exists before any threshold gets picked — CLAUDE.md's metric
   quality gate step 4 applies here exactly as it did to `bus_synaptic`'s calm-floor bug, just checking
   the opposite failure mode (a metric that reads "recovered" because it's stuck, not because anything
   resolved).
3. **What field should flow-state actually key off?** Self-state fields were previously ruled out for
   phi/IIT-adjacent signal work — needs an explicit check (not an assumption) of whether the same
   objection applies to a sustained-coherence detector, or whether `AttentionSelfModelV1.confidence`
   read over a rolling window is a cleaner, already-real alternative.
4. **Does the Sentience Striving Program have a specific AST/HOT signal in mind for "an internal
   tension resolved,"** beyond the generic confidence aggregate — or is confidence-recovery
   (Missing Question 2's subject) actually the same candidate under a different name? Needs Juniper's
   input directly; not resolved by code inspection.
5. **Cooldown cadence**: should generative triggers share `trigger_kind`'s existing per-kind cooldown
   pattern (own lane, like `chat_turn`/`transport`), or fire more freely than error triggers? Raised in
   the original brainstorm, still open — arguably positive moments deserve finer granularity than
   alarms, but that's a real design choice, not a default.
6. **Downstream draft mapping**: does `orion-cortex-exec`'s `_fallback_metacog_draft`/
   `MetacogDraftService` currently map `trigger_kind` toward a specific `CollapseMirrorEntryV2.type`,
   or does it default toward error-shaped types (`glitch`/`turbulence`) regardless of trigger kind? Not
   traced this session — needs checking before Acceptance Check 4 can be verified.

## Proposed schema / API changes

- New `trigger_kind` values: `"insight"` (surprise-resolution / confidence-recovery) and `"flow"`
  (sustained low-variance/low-distress regime) — names chosen to match v1's own `change_type`
  vocabulary (`reorientation`, `stabilizing`) rather than inventing new taxonomy language.
- Each gets its own `<kind>_metacog_gate.py` in `orion-equilibrium-service/app/`, following the
  `chat_turn`/`transport` precedent: correlator/threshold logic + gate-condition evaluator + trigger
  builder.
- Each gets its **own** cooldown lane (`EQUILIBRIUM_METACOG_INSIGHT_COOLDOWN_SEC` /
  `EQUILIBRIUM_METACOG_FLOW_COOLDOWN_SEC`), not the shared global lane — `chat_turn` shipped this bug
  once already (shared the global cooldown at first), documented precedent not to repeat.
- No new top-level field needed on `MetacogTriggerV1` — `trigger_kind` is already a free string,
  `upstream`/`reason` already carry arbitrary evidence.
- `CollapseMirrorEntryV2.type`/`change_type` downstream mapping needs to route these kinds toward
  `flow`/`epiphany` entry types, not silently default to an error-shaped type — contingent on Missing
  Question 6.

## Files likely to touch

- `orion/schemas/telemetry/metacog_trigger.py` (trigger_kind docstring)
- `services/orion-equilibrium-service/app/insight_metacog_gate.py` (new)
- `services/orion-equilibrium-service/app/flow_metacog_gate.py` (new)
- `services/orion-equilibrium-service/app/service.py` (`_run()` dispatch branches, subscription wiring)
- `services/orion-equilibrium-service/app/settings.py` (new enable flags, cooldowns, thresholds)
- `services/orion-equilibrium-service/.env_example` + synced local `.env`
- `services/orion-cortex-exec` draft-mapping code (path TBD, pending Missing Question 6)
- `services/orion-equilibrium-service/README.md` (document alongside existing evidence-source table)
- `services/orion-substrate-runtime/app/worker.py` and/or `orion/substrate/attention_self_model.py`
  (only if Missing Question 1 finds the AST/HOT reducer isn't live-ticking — separate, prerequisite
  patch, own proposal-mode pass since it touches self-modeling directly)
- `scripts/analysis/measure_attention_self_model_confidence_baseline.py` (new, for Missing Question 2)

## Non-goals

- Not building the original drive-tension-resolution idea — confirmed dead ground, DriveEngine's fold
  mechanism is not the target.
- Not building the three declined brainstorm ideas (concept-bridge trigger, relational-resonance
  trigger, pulse-threshold widening).
- Not touching any existing rupture-shaped trigger (`chat_turn`/`transport`/`relational`/
  `telemetry_anomaly`/`llm_surface_instability`) — additive only.
- Not deciding confidence-recovery or coherence-band thresholds in this doc — pending the Missing
  Question 2 measurement pass.
- Not deciding whether the AST/HOT reducer needs to go live — that's a prerequisite question this spec
  surfaces (Missing Question 1) but does not answer or scope.

## Acceptance checks

1. A measurement script against real historical `AttentionSelfModelV1`/prediction-error data shows
   confidence-recovery events are discrete and non-degenerate (not smooth noise, not a permanent
   floor/ceiling artifact) before any live gate ships.
2. Both new gates ship disabled by default, flipped only after their own live-data sanity check,
   matching the `bus_synaptic` precedent (PR #1385 → #1387).
3. `orion_metacog` shows real `trigger_kind="insight"`/`"flow"` rows with non-empty, distinguishable
   `upstream` evidence after enabling.
4. Downstream `CollapseMirrorEntryV2` drafts for these kinds land as `type="epiphany"`/`type="flow"`
   (or an honest `context_shift` if the draft LLM can't map cleanly), not silently defaulting to an
   error-shaped type.
5. Full test suite for touched files passes; a cooldown-lane-independence test is added, same pattern
   as `chat_turn`'s fix.

## Recommended next patch

1. Answer Missing Question 1 first — is `reduce_attention_self_model()` actually ticking anywhere live.
   This single fact determines whether "insight" is a thin seam (new gate on an existing live producer)
   or a bigger prerequisite patch (make AST/HOT live first, its own proposal-mode question).
2. Run the Missing Question 2 measurement pass against whatever prediction-error/confidence history
   already exists in Postgres/FalkorDB before writing any gate code.
3. Once both come back real, pick "insight" or "flow" to build first based on which has cleaner
   historical shape — not both simultaneously.
