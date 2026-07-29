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
- **`reduce_attention_self_model()` live-tick status: ANSWERED, now live (Missing Question 1
  resolved).** `docs/superpowers/specs/2026-07-29-ast-hot-reducer-live-ticking-design.md` +
  PR #1459 (merged) made `services/orion-substrate-runtime/app/worker.py`'s
  `_attention_broadcast_tick()` call `reduce_attention_self_model()` on every ~30s broadcast tick
  (gated `SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED`, now `true` in this environment), persisting
  each result to a new, exclusively-owned `substrate_attention_self_model` table. Review found the
  design doc's original premise was itself wrong — `_brain_frame_tick()` already called the same
  reducer live with a narrower input set (`field_frame=None`, no trend) — corrected in that doc; the
  two live call sites are intentionally left un-unified. **`AttentionSelfModelV1.confidence` /
  `.prediction_error_confidence` may now be treated as a live producer.**
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

1. **ANSWERED 2026-07-29 — yes, now live.** See the "live-tick status" update above and
   PR #1459. Not a blocker anymore.
2. **ANSWERED 2026-07-29, with a real-data finding that changes Acceptance Check 1's shape.**
   `scripts/analysis/measure_attention_self_model_confidence_baseline.py` (PR #1463, merged) ran
   against real `substrate_attention_self_model` history once it cleared its own 200-tick/2h trust
   floor (1,524 rows / 14.4h span at run time). Findings:
   - Both fields pass the metric-quality-gate sanity check: 100% coverage, no ceiling/floor/zero-variance
     degeneracy. `confidence` ranged 0.30–0.90 (mean 0.886); `prediction_error_confidence` ranged
     0.597–0.974 (mean 0.900).
   - At the design doc's own untested anchor thresholds (low=0.5, high=0.8), **zero recovery events
     fired** — not because the signal is smooth, but structurally: `prediction_error_confidence` never
     once dropped to 0.5 in the observed window. The anchor thresholds don't match this metric's actual
     observed range in this environment.
   - Re-run at thresholds matched to the metric's real range (low=0.70, high=0.90): **3 genuine
     recovery events fired over 13.93h**, ticks-to-cross = 1, 3, 12 (mean 5.3, median 3).
   - **Verdict: leans smooth, not discrete.** A median 3-tick crossing (each tick ~30s, so ~90s median)
     is not a sharp single-tick jump — recoveries unfold gradually. This means a gate keyed on a single-
     tick threshold crossing (the pattern every existing `trigger_kind` uses) would either fire on noise
     mid-climb or need a sustained-duration condition instead of a point crossing. **Acceptance Check 1
     ("confidence-recovery events are discrete") does not currently hold as literally stated** — the
     recommended next patch below adjusts for this rather than treating it as a blocker.
   - Thresholds are still not independently calibrated beyond "matches observed range" — before any
     gate ships, a longer window (the current run covers only the table's first ~14h of existence)
     should be re-measured to confirm 0.70/0.90 hold up, not just this window's specific 3 events.
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

1. **Revised 2026-07-29 per real measurement (see Missing Question 2 findings):** confidence-recovery
   events are non-degenerate (not a permanent floor/ceiling artifact) but lean smooth, not discrete —
   the gate condition must account for a multi-tick gradual crossing, not assume a sharp single-tick
   jump. Original wording ("shows...events are discrete") does not hold as literally stated; superseded
   by the sustained-transition condition in Recommended next patch item 4.
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

1. ~~Answer Missing Question 1~~ — done, PR #1459. `reduce_attention_self_model()` is live.
2. ~~Run the Missing Question 2 measurement pass~~ — done, PR #1463 + this update. Real finding:
   confidence-recovery is smooth (median 3-tick crossing), not a sharp single-tick jump.
3. Re-run the confidence-baseline script against a longer window (days, not the current ~14h) before
   picking final thresholds — this update's 3 events are real but from a young table; confirm 0.70/0.90
   holds up rather than locking in a 14h sample.
4. Because the signal leans smooth, scope the "insight" gate's condition as a sustained low→high
   transition over N ticks (or a rolling-window derivative), not a single-tick `>= high_threshold`
   crossing — the existing `chat_turn`/`transport`/`relational` gates are all single-tick-crossing
   gates and would misfire on this signal's own gradual-climb shape if copied unmodified.
5. Missing Questions 3–6 (flow-state field choice, Sentience Striving Program signal-naming, cooldown
   cadence, downstream draft mapping) remain open — still needs Juniper's input or further tracing
   before "insight" or "flow" gate code gets written.
