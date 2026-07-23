# Collapse-mirror / metacog redesign (brainstorm, not yet a proposal)

Date: 2026-07-18

Status: **ideation stage.** This captures a real design conversation between Juniper and Claude
so it doesn't evaporate into chat history. It is not implementation-ready, and per this repo's
CLAUDE.md, changes to cognition loops need explicit proposal mode (capability change, data
touched, privacy boundary, trace, failure mode, rollback) before any code — none of that has
been decided yet. Treat every idea below as "worth trying," not "agreed."

## Why this exists

Collapse-mirror started as a ritual shared between Juniper and Orion — a founding piece of the
base schema (`orion/schemas/collapse_mirror.py`'s `CollapseMirrorEntryV2`: `trigger`,
`observer_state`, `field_resonance`, `emergent_entity`, `summary`, `mantra`, `what_changed`). The
idea was: something triggers a causally-dense moment, a waveform collapse gets marked, and it can
be traced later. Over time this morphed into a metacognitive exercise trying to simulate a
reverie-like experience, and it drifted into telemetry land — the numeric self-report side
(`numeric_sisters`: `valence`/`arousal`/`clarity`/`overload`/`risk_score`) became the dominant
concern, and the ritual/narrative side got diluted.

This conversation started from a different problem: the mood-arc felt-state-trajectory
autoencoder (`orion/mood_arc/` once PR #1189 merges, still `scripts/fit_mood_arc_encoder.py`
until then) needs an independent criterion signal to validate whether it learned real structure,
not just noise. The old corpus's `valence` field was reused for this originally and turned out to
be circular (it was re-deriving `orion-field-digester`'s own decay filter). Juniper explicitly
ruled out reinventing a valence-shaped substitute computed from the new field channels ("no
valence in this new world"). Investigating alternatives is what surfaced that collapse-mirror's
own `numeric_sisters` has the exact same disease, one level removed — which is the actual subject
of this doc.

## Root problems, confirmed against real code (not assumed)

**1. `numeric_sisters` is contaminated by the old phi/self-state pipeline.**
`services/orion-cortex-exec/app/executor.py:3841-3864`: before the LLM generates its
`numeric_sisters` self-report, the metacog prompt is built with
`phi_hint = spark_phi_hint(spark_snap)` — pulled from `SparkStateSnapshotV1`, phi's own live
output — and formatted directly into the prompt as
`"φ={valence_band}-{valence_dir}, energy={energy_band}, clarity={coherence_band},
overload={novelty_band}"`. The LLM is handed phi's own bands as context, then asked to produce
its own valence/arousal/clarity/overload numbers — which it substantially echoes rather than
independently generates. This is downstream of the exact same `_phi_from_self_state()` pipeline
that produced the original invalidated `mood_arc_corpus.v1`'s `valence` field, just laundered
through an LLM call in between. **Confirmed live: 8,481 `collapse_mirror` rows in Postgres, 6,579
with a real non-null valence (mean 0.48, range -0.8 to 1.93) — real data, genuinely contaminated
at the source.**

**2. There is no real endogenous "causally dense moment" detector.**
`_metacog_trigger_lineage()` (executor.py:698-717) only ever sets `trigger_kind` to `"chat_turn"`
(any turn with a correlation id), `"heartbeat"` (a scheduled tick), or `"unknown"`. Nothing
gates *whether* something notable happened before firing — it fires unconditionally, on a
schedule. `causal_density`/`is_causally_dense` scoring
(`orion/collapse/service.py::apply_causal_density_to_entry`) only runs *after* an entry already
exists, as a post-hoc label, not as a gate deciding whether to create one. Phi was supposed to be
the thing that decided "this moment matters" — and per problem 1, that pipeline doesn't work for
this purpose either.

## Design direction: ground triggers and snapshots in real turn artifacts, not self-rated numbers

A normal cortex-exec chat turn already computes a pile of concrete, real artifacts before
anything reaches metacog: `llm_uncertainty`, `autonomy_slice`, `grounding_capsule`,
`reasoning_content`/`reasoning_trace`, `structured_output_diagnostics`, step-level friction
(`failed_step_count`/`started_step_count` — literally what feeds `execution_friction` already).
None of this needs inventing; it's already sitting in `ctx` by end-of-turn and already flows to
substrate-runtime via the execution_run/grammar pipeline (see the reasoning_load fix, PR #1177,
for the exact mechanism this reuses).

Proposed shape:
- **Trigger** = a real condition on those artifacts, not "every turn fires." e.g. uncertainty
  crossed a threshold, a step failed/retried, `grounding_capsule` flagged a conflict, autonomy
  mode changed, or `metacog_traces` are non-empty (the turn already did deep reasoning worth
  marking). Most routine turns produce nothing; friction/ambiguity produces a real entry — which
  is what "causally dense" was supposed to mean from the start.
- **State-snapshot** = read those same artifacts directly instead of asking an LLM to rate its
  own valence. "Uncertainty was high," "a step failed," "autonomy mode changed" are facts already
  computed, not a vibe check.

This is not a new retrospective mechanism — it's wiring collapse-mirror's trigger and snapshot to
*read from* what a turn already computed, instead of running a second LLM call to re-examine
itself introspectively. Less machinery than exists today, not more.

### Honest limits of this direction (Juniper's "operational exhaust" pushback, confirmed correct)

Per-component deterministic success/fail codes escape the *phi* circularity completely (a real,
independent win), but they don't fully escape a *different* circularity: a step failing is
mechanically the same information as `execution_friction`/`failure_pressure`, already in the
mood-arc encoder's input channels — just less aggregated. This is **not** the same rigor as a
truly independent signal (see the earlier investigation into GPU power draw / network I/O as
candidates, which are genuinely causally disconnected from execution load). What this direction
*does* buy: the encoder only ever sees a flat per-tick aggregate; it has no concept of *which*
phase failed, in what sequence, alongside what else. A phase-aware, sequential, structured signal
tests a different and harder question than the flat scalar can — "does the encoder's structure
track *which kind* of thing is happening and *when*," not "is this independent of substrate
load." Worth keeping. Don't mistake it for full independence.

## Proposed schema field remapping (first pass, not final)

| Field | Current meaning | Proposed meaning |
|---|---|---|
| `trigger` | free-text trigger label | which specific thing fired the entry (e.g. `"step:grounding_capsule failed"`); an ordered list of every component's code for the turn lives in the snapshot as supporting evidence, not squeezed into this one field — a turn is a sequence of phases, not one atomic event |
| `observer_state` | unclear/overloaded | severity classification (nominal/degraded/critical), thresholded off real numbers already computed (uncertainty, retry count) |
| `field_resonance` | a phi-derived narrative string (`"φ:{vb}-{vd}, energy:{eb}..."`) | **repurposed**, not severity: a structural/topological cross-reference — which *other* parts of the system this event echoes across (e.g. "touched: reasoning, grounding, autonomy") |
| `emergent_entity` | more poetic/emergent (seen in code as e.g. `"Fallback Baseline"`) | phase of the unified run — workable, but this converts a previously-emergent field into something purely mechanical. Note that trade-off explicitly if taken; a deterministic scaffold field is fine, just don't do it by accident |
| `summary` / `mantra` | authored narrative | stays authored/non-deterministic — this is the actual ritual half, not something to mechanize |
| `what_changed` | one field, mixed | split: a structured, computed diff (deterministic) plus `what_changed_summary` as the authored gloss on it |

## New lineage/provenance shape (Juniper's "came from X, produces Y, impacts [Z,A,...]")

Proposed addition, not yet fleshed into a real schema:

```
provenance:
  source: <what fired it, e.g. "cortex_exec.step_friction", "field_digester.anomaly_detector", "chat.correction_detected">
  produces: <what kind of artifact this entry is, e.g. "friction_event", "anomaly_flag", "redirection_marker">
  impacts: [<downstream things this is about>, e.g. "execution_trajectory", "field_channel:reasoning_load", "relationship_thread"]
```

## Trigger-type taxonomy (widening beyond chat_turn/heartbeat)

Heartbeat's unconditional-schedule shape should probably die, but the *reason* it exists —
visibility into autonomous/idle-time activity when no chat turn is happening — shouldn't die with
it. Proposed replacement: split into condition-gated types instead of one unconditional tick.

- **`chat_turn`** (existing, keep) — gated on the turn-artifact conditions above instead of firing
  unconditionally.
- **`substrate_grammar`** (new) — execution_loop/harness-governor-level events (the same pipeline
  touched during the reasoning_load fix, PR #1177), gated on friction/uncertainty during
  non-chat activity — the replacement for heartbeat's coverage.
- **`telemetry_anomaly`** (new) — **this already exists and just needs wiring, not building.** The
  anomaly detector shipped today (`detect-anomalies`, PR #1185) is exactly a real, deterministic
  "something notable happened" detector for field-channel data. A flagged anomalous window is a
  legitimate trigger condition as-is.
- **`relational`** (new) — **also already exists and just needs wiring.** `orion/memory/
  turn_change_classify.py`'s live `SHIFT: NONE|TOPIC|STANCE|REPAIR` appraisal (see below) is the
  trigger condition — something notable in the conversation/relationship itself, orthogonal to
  execution mechanics, already computed today for recall-routing purposes.

## The relational axis (Juniper: "really hard to quantify," asked to keep riffing)

Component success/fail codes, taken alone, are still pure telemetry — just symbolic instead of
numeric. The ritual quality was never about "did step 4 fail," it was about something notable
happening *between* Juniper and Orion.

**First pass at this (retracted).** Word-list sentiment markers and embedding-distance topic-shift
detection were proposed and correctly rejected by Juniper as keyword cathedrals — pattern-matching
dressed up as detection, with no theory behind either category, exactly what CLAUDE.md section 0A
bans. Recorded here so the mistake isn't repeated, not as a live proposal.

**Real theory instead of a heuristic: rupture-and-repair.** From relational psychology (Safran &
Muran) — an established framework for relationship micro-events, not invented for this purpose. A
*rupture* is a moment of misalignment/friction in a relationship; a *repair* is the move that
addresses it. This conversation's own corrections ("no i meant," "stretch. do better," "call me
out on my bullshit") are textbook rupture-confrontation instances — real exemplars, not
hypotheticals.

**Then found: this is already built, running, and better-designed than what was proposed.**
`orion/memory/turn_change_classify.py` is a live classifier, currently feeding
`orion/memory/retrieval_intent.py`'s recall-routing decisions (`derive_retrieval_intent()`), that
compares the current turn against a session-window baseline and outputs exactly four categories:

```
SHIFT: NONE | TOPIC (subject) | STANCE (identity/beliefs/relationship framing) | REPAIR (correction/recovery)
```

Scored via `enum_scores_from_top_logprobs` — real per-token logprob confidence, not an argmax
guess — producing `novelty_score`, per-category `shift_scores`, and an overall `confidence`
(`build_turn_change_appraisal()`). `REPAIR (correction/recovery)` *is* the rupture-repair
construct, already implemented under a different name. `TOPIC` directly answers Juniper's question
("did we also build a classifier... that figures out topic shift") — yes, already live. This is
zero-shot with tight category definitions rather than literally few-shot (no embedded examples in
the prompt today), but it's already theory-anchored and already scored, which is what actually
mattered.

**Revised design: reuse this classifier directly, build nothing new for topic-shift or
rupture-repair.** `appraisal.shift_kind == "REPAIR"` or `"TOPIC"` (novelty-floor-gated, matching
`retrieval_intent.py`'s own `shift_novelty_mismatch()`/`reconcile_novelty_with_shift()` pattern for
handling low-confidence shifts) becomes the `relational` trigger type's actual condition. No new
classifier, no new prompt, no new theory to invent — wire the existing signal into collapse-mirror
the same way `telemetry_anomaly` wires in the existing anomaly detector.

**This also answers the `causal_density`/severity question Juniper flagged as unresolved.**
`novelty_score` and `confidence` from this same appraisal are exactly the severity/impact
magnitude that was missing — `causal_density.score` should be sourced from these, not invented as
a new metric. `shift_scores` (per-category confidence, not just the winning label) could inform
`field_resonance`'s "which axes this event touched" if a turn scores nontrivially on more than one
shift kind at once.

**Still open, not resolved by this discovery:** `STANCE` is a third category (identity/beliefs/
relationship framing) neither Juniper nor this doc had considered before finding the code — worth
deciding whether it belongs in collapse-mirror's trigger set alongside REPAIR/TOPIC, ignored, or
handled differently. Silence/re-engagement is separately covered by `discussion_window`'s
90-minute gap (see above) and doesn't need this classifier at all.

## Explicitly rejected / ruled out

- **Reusing `collapse_mirror.numeric_sisters` as-is for anything validation-related.** Contaminated
  at the source (see Root problem 1). Confirmed via live Postgres data, not assumed.
- **Any new hand-composited "valence-shaped" scalar computed from `field_channel_corpus.v1`'s own
  channels.** Juniper's explicit, standing decision ("no valence in this new world") — this
  applies here too, not just to the mood-arc corpus directly.
- **GPU power draw / network I/O as criterion signals**, despite being genuinely uncorrelated
  mechanically with the encoder's inputs (confirmed: `services/orion-biometrics/app/metrics.py`
  collects both, `grammar_emit.py` never surfaces them into any field channel) — ruled out not for
  being circular, but for being the wrong *kind* of signal: substrate health, not felt/relational
  quality. Kept in this doc as a reference point for what "actually independent" looks like when
  evaluating other candidates.
- **Word-list lexical sentiment markers and embedding-distance topic-shift detection.** Keyword
  cathedrals — no theory behind either category. Superseded by reusing
  `turn_change_classify.py`'s already-theory-anchored `SHIFT` appraisal instead.

## Open questions (need a real proposal pass, not decided here)

- What capability actually changes? (Collapse-mirror's trigger condition + snapshot content, at
  minimum. Scope of `numeric_sisters` — deprecated, repurposed, or removed — undecided.)
- What data is touched? (`collapse_mirror` Postgres table, `orion/schemas/collapse_mirror.py`,
  `orion/collapse/service.py`, the metacog-prompt-building code in
  `services/orion-cortex-exec/app/executor.py`, and read access to `orion/memory/
  turn_change_classify.py`'s appraisal output — currently only consumed by
  `orion/memory/retrieval_intent.py` for recall routing. Adding a second consumer needs checking
  whether that appraisal is already reliably available at the point in the turn where collapse-
  mirror would need it, or only computed later/conditionally.)
- Does `STANCE` (identity/beliefs/relationship framing) belong in collapse-mirror's trigger
  set alongside `REPAIR`/`TOPIC`, get ignored, or handled as a fourth, distinct trigger condition?
  Not decided — found late, not yet discussed.
- Privacy boundary? Not assessed yet.
- What trace would prove any of this actually worked? Not designed yet — likely something like
  "entries now correlate with real friction/uncertainty events, verified against a known
  incident, the same way the anomaly detector was verified against the real pre-fix corpus
  period."
- Failure mode if this goes wrong, and how to roll back? Not assessed.
- Is `heartbeat` actually safe to remove, or does something else depend on its unconditional
  firing today? Not checked.

## Metric quality gate, applied (CLAUDE.md section 0A)

Per the new repo-wide metric quality gate (`CLAUDE.md`, added same day, PR #1193), every metric
proposed above gets run through the same 6-step checklist before being trusted further. Doing that
here, retroactively, as the first real application of the gate.

**`collapse_mirror.numeric_sisters` (the thing being replaced) — fails the gate, which is exactly
why it's being replaced.**
1. Provenance: traced to `services/orion-cortex-exec/app/executor.py:3841-3864` —
   `spark_phi_hint()`. ✅ traced.
2. Independence: fails. Directly fed the same `SparkStateSnapshotV1` bands the old, already-
   invalidated `mood_arc_corpus.v1` used. Not independent of the thing it needs to validate.
3. Theory anchor: none stated at the point of construction — an LLM asked to rate its own
   valence/arousal with no named framework.
4. Live-data check: done — 8,481 real Postgres rows, 6,579 non-null, real variance. Passes on its
   own, but irrelevant once #2 fails.
5. Existing-mechanism check: n/a, this is the thing already built (the problem, not a candidate).
6. Reversibility: cheap — nothing downstream depends on `numeric_sisters` yet, it was never wired
   into anything beyond its own storage.

**`orion/memory/turn_change_classify.py`'s `SHIFT` appraisal (for the `relational` trigger) —
passes.**
1. Provenance: traced to `build_turn_change_prompt()`/`build_turn_change_appraisal()`. ✅
2. Independence: no causal link to field-channel/execution telemetry — it's a comparison between
   the current turn's text and a session-window baseline, a completely different computational
   path. ✅
3. Theory anchor: `REPAIR (correction/recovery)` maps directly to rupture-and-repair (Safran &
   Muran); `TOPIC`/`STANCE` are named, defined categories in the prompt itself, not vibes. ✅
4. Live-data check: **done** (2026-07-18, implementation pass). `chat_history_log.spark_meta`,
   7-day window: 74 real appraisals -- TOPIC 33 (avg confidence 0.87, avg novelty 0.98), NONE 24,
   STANCE 17, REPAIR 0. Non-degenerate distribution. **Zero REPAIR events in this window** --
   either genuinely rare (corrections are infrequent) or worth re-checking over a longer window
   once the relational trigger has run live for a while. Not blocking: TOPIC alone already
   validates the wiring.
5. Existing-mechanism check: this *is* the existing-mechanism check succeeding — found before
   building a duplicate. ✅
6. Reversibility: cheap — read-only consumption of an existing signal, no schema change to the
   producer.

**`detect-anomalies` (PR #1185, for the `telemetry_anomaly` trigger) — passes, already
live-verified.**
1. Provenance: traced and built this session — `score_windows()`/`recon_loss()`. ✅
2. Independence: n/a in the same sense — this isn't being used to validate the mood-arc encoder,
   it's a consumer of the encoder's own output as a QA tool. Different role, independence
   question doesn't apply the same way.
3. Theory anchor: reconstruction-error-as-anomaly-score is a standard, named technique, not
   invented for this purpose. ✅
4. Live-data check: done in the same session — 1,166/9,103 real pre-fix windows flagged. ✅
5. Existing-mechanism check: n/a, newly built, but check ran first (nothing else in the repo did
   this).
6. Reversibility: cheap — pure CLI tool, no schema commitment.

**`discussion_window`'s 90-minute gap (for silence/re-engagement) — passes.**
1. Provenance: traced to `orion/discussion_window/sql_fetch.py:18`,
   `_DEFAULT_CONTIGUITY_MAX_GAP_SEC = 5400`. ✅
2. Independence: a wall-clock gap between turns, no relationship to substrate/execution
   telemetry at all. ✅
3. Theory anchor: session-boundary/contiguity is a standard, named concept in dialogue systems,
   not invented here. ✅
4. Live-data check: not re-verified in this doc (Juniper recalled the value correctly from memory,
   confirmed against the actual constant, but hasn't been checked against current live gap
   distributions).
5. Existing-mechanism check: this is the same "found before reinventing" story as the SHIFT
   classifier. ✅
6. Reversibility: cheap — reusing an existing constant, not touching its producer.

**Net result of running the gate for real**: 3 of 4 candidate mechanisms pass; the one that fails
(`numeric_sisters`) is exactly the one this whole redesign exists to replace, which is a decent
sign the gate works rather than rubber-stamping everything. Two open items before calling the
passing ones fully cleared: pull real, current `shift_kind` distributions (item 4 for the SHIFT
classifier), and re-check the 90-minute gap against current live conversation gap patterns (item 4
for `discussion_window`).

## Next step

Whenever this moves from ideation to a real patch: full Design-mode writeup per CLAUDE.md
(current architecture, missing questions, proposed schema/API changes, files touched, non-goals,
acceptance checks), then explicit proposal-mode sign-off before any code, since this is a
cognition-loop change.

## Implementation, first slice (2026-07-18, branch `feat/collapse-mirror-metacog-trigger-redesign`)

Per Juniper's direct go-ahead to implement, built the narrowest slice of this doc that didn't
depend on any of the still-open questions above: the **relational trigger**, wired end to end.

- `services/orion-equilibrium-service/app/relational_metacog_gate.py` (new): reads a
  `turn_change_appraisal` and fires `trigger_kind=relational` for REPAIR/TOPIC above a confidence
  floor. STANCE intentionally excluded (still undecided, see above).
- `services/orion-equilibrium-service/app/service.py`: subscribes to the existing
  `orion:chat:history:spark_meta:patch` channel (already published by `orion-memory-consolidation`
  for every classified turn) as an additional consumer -- confirmed it's a real fan-out channel,
  not single-consumer, via `orion-sql-writer` and `orion-spark-introspector` already reading it.
- `orion/collapse/service.py`: `_relational_evidence_score()` reads the relational trigger's
  `novelty_score`/`confidence` back off the entry's own telemetry (stamped there by
  `_apply_metacog_system_fields` during the draft step, `services/orion-cortex-exec/app/executor.py`)
  and blends it into `causal_density` alongside phi-evidence -- exactly the "source causal_density
  from the appraisal itself" idea above, not a new invented metric.
- `orion/bus/channels.yaml`: registered `orion-equilibrium-service` as a consumer of
  `orion:chat:history:spark_meta:patch` -- also fixed a pre-existing gap where
  `orion-spark-introspector` was already consuming that channel live but was missing from the
  registry.
- Live-data check for the metric quality gate done against real Postgres data (see above);
  README and `.env_example` updated; 9 new tests (`test_relational_metacog_gate.py`, 3 new cases
  in `test_apply_causal_density_to_entry.py`), 40 tests green across both touched packages.

**Deliberately not built in this slice** (still genuinely open, per this doc's own "Still open"/
"Open questions" sections above -- building them now would mean inventing answers to questions
this doc explicitly left for Juniper):
- `telemetry_anomaly` trigger: this doc's framing of it as "already exists and just needs wiring"
  turned out to be optimistic on inspection. `detect-anomalies` (PR #1185) is a CLI tool only --
  no live service runs it on a schedule or publishes its output. Wiring it as a real trigger means
  building a new live anomaly-scoring loop, not just subscribing to something that already emits
  events. Correction recorded here so the next pass doesn't re-assume it's a thin wire-up.
- `numeric_sisters`/phi_hint contamination fix (root problem 1): diagnosis only in this doc, no
  proposed fix -- needs its own design pass on the prompt-construction side of
  `services/orion-cortex-exec/app/executor.py`.
- Schema field remapping, `substrate_grammar` trigger, provenance schema, heartbeat removal,
  STANCE handling: all still open per this doc's own "Open questions" section, untouched.

## Implementation, second slice (2026-07-18, branch `feat/metacog-real-artifact-model`)

Per Juniper's direct go-ahead to implement the real-artifact model, built the full split: a new,
independent metacog schema/table sourced from live turn artifacts, with `collapse_mirror` now
strict-lane-only for good.

- **`collapse_mirror` is strict-lane-only.** `services/orion-dream/app/aggregators_sql.py::
  fetch_recent_sql_fragments()` now filters `WHERE ... AND lower(trim(observer)) = 'juniper'`
  (a SQL-level approximation of `orion/schemas/collapse_mirror.py::_observer_is_juniper()`,
  minus diacritics-stripping -- a known, accepted simplification). No history migrated; the
  8,481 existing rows stay exactly where they are.
- **New schema: `orion/schemas/metacog_entry.py::MetacogEntryV1`.** Independent of
  `CollapseMirrorEntryV2` -- no shared base class, no reused field names carrying old baggage.
  Drops `numeric_sisters` entirely, no replacement. `state` (`MetacogRealState`) holds only real,
  live-computed artifacts: `biometrics`, `turn_effect`/`turn_effect_evidence`,
  `substrate_eventfulness_score`/`_reasons`, `llm_uncertainty`, `reasoning_excerpt`, and
  `repair_pressure` (level/confidence/evidence/behavior_applied). `snapshot_kind` is a real
  `Literal["baseline", "confirmed_dense"]` -- learned from `collapse_mirror.snapshot_kind`'s 38
  distinct garbage free-text values in production after years as an unconstrained string.
  `causal_density` (`orion/metacog/service.py::compute_causal_density`) blends only real
  artifacts (repair_pressure, substrate_eventfulness, a severity read off turn_effect) -- there is
  no self-report leg in this model, so nothing to blend it against.
- **New table: `orion_metacog`** (`services/orion-sql-writer/app/models/metacog_entry.py`), a
  genuinely different table from `collapse_mirror`, not a `_v2`. New channel
  `orion:metacog:sql-write`, registered in `orion/bus/channels.yaml` and wired into
  `orion-sql-writer`'s `MODEL_MAP`/`DEFAULT_ROUTE_MAP`/subscribe list (settings default AND
  `.env_example`, since the env alias replaces the JSON route map wholesale in live deployments).
- **`numeric_sisters` contamination fix (root problem 1, finally addressed).** The enrich prompt
  (`orion/cognition/prompts/log_orion_metacognition_enrich.j2`) no longer instructs the LLM to
  derive `numeric_sisters` from phi bands -- that section, the "NUMERIC SISTERS" mapping table,
  and the "RISK SCORE" section are deleted from the template; the allowed-keys/forbidden-keys
  lists and example JSON were updated to match. `phi_hint`/`spark_phi_narrative` computation in
  `MetacogContextService` is left in place (still used by the fallback draft's narrative text and
  the draft/enrich prompts' scene-setting "SPARK φ INTERPRETATION" guidance) -- just no longer fed
  into anything that becomes a numeric self-rating.
- **relational trigger re-sourced to repair_pressure_v2, replacing `turn_change_classify`.**
  `services/orion-hub/scripts/pre_turn_appraisal_wiring.py` now forwards the full typed evidence
  breakdown (`evidence_kind`/`score`/`confidence` per detector, not just a count) and publishes a
  new `orion:repair_pressure:appraisal` envelope (Option A: new channel, chosen over inlining this
  into cortex-exec) whenever the repair_pressure paradigm actually ran.
  `services/orion-equilibrium-service/app/relational_metacog_gate.py` (subscribed to
  `orion:chat:history:spark_meta:patch`, keyed on `turn_change_classify`'s SHIFT appraisal) is
  deleted; `repair_pressure_metacog_gate.py` replaces it, keyed on `level`/`confidence` floors
  over the new channel. `trigger_kind="relational"` is unchanged -- same conceptual category,
  different evidence source.
- **cortex-exec retarget.** `MetacogDraftService`/`MetacogEnrichService` keep using a
  `CollapseMirrorEntryV2`-shaped scratch object internally -- that machinery (LLM patch schemas,
  prompt-budget enforcement, the uncertainty probe) stays as-is, since it's just plumbing for the
  authored summary/mantra/what_changed narrative loop (the "ritual half," explicitly meant to stay
  non-deterministic). Only `MetacogPublishService` changed: it now builds a real `MetacogEntryV1`
  from `ctx`'s live artifacts plus the scratch object's authored text, computes `causal_density`
  inline via `orion.metacog.service.compute_causal_density`, and publishes to
  `orion:metacog:sql-write` instead of `orion:collapse:sql-write`. The old self_state-blended
  `apply_causal_density_to_entry` call is gone from this path (still used, unchanged, by the
  strict-lane Hub form via `collapse_verbs.py`).
- Tests: new schema tests (`orion/schemas/tests/test_metacog_entry.py`), new causal-density
  scoring tests (`orion/metacog/tests/test_service.py`), new equilibrium gate tests
  (`services/orion-equilibrium-service/tests/test_repair_pressure_metacog_gate.py`, replacing the
  deleted `test_relational_metacog_gate.py`), new sql-writer wiring-shape tests
  (`services/orion-sql-writer/tests/test_metacog_entry_sql_shape.py`), new hub wiring tests
  (`services/orion-hub/tests/test_pre_turn_appraisal_wiring.py` additions), a dream-scope
  regression test (`services/orion-dream/tests/test_aggregators_sql_collapse_scope.py`), and a
  rewritten cortex-exec publish-lane test
  (`services/orion-cortex-exec/tests/test_metacog_publish_lane.py::
  test_publish_builds_metacog_entry_from_real_artifacts_no_self_report`, replacing the old
  self_state-blend assertion this patch obsoletes).

**Open question, not answered in this slice:** should `orion/memory/turn_change_classify.py` be
retired? It is no longer used by metacog -- repair_pressure_v2 replaced it there in this same
patch -- but it still feeds `orion/memory/retrieval_intent.py`'s recall-routing
(`derive_retrieval_intent()`), which this patch does not touch. A later pass should look at
whether that one remaining use case still justifies its own LLM-probe classifier, or whether it
could be consolidated/simplified now that its other consumer is gone. Deliberately not answered
here.

## Correction pass (2026-07-18, same branch): the first slice dropped real design content

The first pass of this slice (above) kept the "no self-report" and "repair_pressure_v2" wins but
flattened or dropped several structural pieces this doc's own "Proposed schema field remapping"
and "New lineage/provenance shape" sections had already worked out. Root cause: the dispatch spec
for that pass cited this doc by line number but then paraphrased its content from memory instead
of transcribing it -- the gap was between what this doc says and what got *told* to the
implementing pass, not something lost during execution. Caught via direct review pushback, not
by any gate in this process, which is itself worth noting.

What was missing and what got fixed, each backed by data already sitting in `ctx` with no new
producer needed:

- **`trigger` as a sequence, not one code.** This doc's proposal: `trigger` names the one specific
  thing that fired the entry, with a fuller ordered list of every component's code for the turn
  living in the snapshot as supporting evidence. `executor.py`'s own `logs`/`merged_result`
  (accumulated across every step of the turn, "ok <- X" / "error <- X" / "skip <- X (...)") already
  *is* that ordered sequence. Now wired into `MetacogWhatChanged.evidence`, capped to the last 20
  entries, 200 chars each. `trigger_kind`/`trigger_reason` (from the upstream `MetacogTriggerV1`)
  were already a reasonable fit for "the one specific thing" and were left as-is.
- **Severity, repurposing `observer_state`.** This doc: nominal/degraded/critical, thresholded off
  real numbers already available (uncertainty, retry/failure count). Added
  `MetacogEntryV1.severity`, computed by `orion.metacog.service.compute_severity()` off two real,
  independent signals: a count of unambiguous failure-prefixed lines in `logs` this turn (`fail`/
  `error`/`exception` only -- an earlier draft of this fix wrongly counted routine `exec ->` start
  markers as failures, caught by a test regression before it shipped, see
  `test_severity_degraded_on_single_failed_step` and friends) and `orion-llm-gateway`'s own real
  logprob-margin telemetry (`mean_top1_margin`, `low_margin_token_count` --
  `services/orion-llm-gateway/app/llm_uncertainty.py`, not a self-rating). Thresholds are starting
  defaults, same calibration caveat as `causal_density`'s weights.
- **Topology, repurposing `field_resonance`.** This doc: which other parts of the system this event
  echoes across ("touched: reasoning, grounding, autonomy"), not a repeat of severity. Added
  `MetacogEntryV1.touches`, computed by `compute_touches()` -- mechanically names which `state`
  sub-fields are actually populated (`repair_pressure` present -> `"relational"`, etc.). No new
  signal, just naming what's already on the entry.
- **`provenance.source`/`impacts`, dynamic not hardcoded.** This doc gave varied, specific examples
  (`"cortex_exec.step_friction"`, `"chat.correction_detected"`). The first pass shipped a constant
  (`source="cortex_exec.metacog_pipeline"`, `impacts=[]`) on every single entry -- a schema-valid
  field carrying zero information, exactly the pattern CLAUDE.md's no-empty-shell-cognition rule
  bans. `compute_provenance()` now derives `source` from the real `trigger_kind` and `impacts` from
  the same `touches` list above (a channel-name/relationship-thread/execution-trajectory mapping per
  touch), so two different entries actually produce two different provenance records.
- **`emergent_entity`-as-phase-of-run:** left genuinely open. This doc's own framing of this one was
  a caution ("notice you're doing it on purpose, not by accident"), not a decision -- not filling it
  in guessed.

New tests: `orion/metacog/tests/test_service.py` (severity/touches/provenance, 13 new cases) and
`services/orion-cortex-exec/tests/test_metacog_publish_lane.py::
test_publish_severity_and_touches_reflect_failures_and_repair_pressure` (adversarial-ctx
integration case: real repair_pressure evidence plus the existing real-artifact case now assert on
`touches`/`severity`/`provenance` instead of only checking the response didn't crash).

## Metric quality gate, actually run this time (2026-07-19, same branch)

CLAUDE.md section 0A requires 6 steps per new signal, findings recorded. The prior passes on this
branch did steps 1 (provenance) and 5 (existing-mechanism) for some of `MetacogRealState`'s fields,
ad hoc, and never step 2 (independence) or step 4 (live-data sanity) for any of them, and never
wrote it down. Run properly this time, against real containers/Postgres, not code-tracing alone.

**Finding, serious: `repair_pressure`'s confidence structurally floors at exactly 0.0 on ordinary
turns.** `orion/substrate/appraisal/paradigms/repair_pressure_v2.py:91-97`: confidence is
`min()` over only the evidence kinds that scored `> 0.5`; if none do, confidence is set to exactly
`0.0`, by construction, not a low-but-nonzero read. The only real appraisal found anywhere (docker
logs, `orion-athena-cortex-exec`, corr `9899932f-...`): `level=0.087 confidence=0.000`. The
relational trigger's gate (`repair_pressure_metacog_gate.py`) fires on `level >= threshold and
confidence >= confidence_floor` (0.7 default) -- with confidence structurally zero unless some
evidence kind scores strongly, this gate may rarely or never open on ordinary conversation. n=1,
so "never" isn't proven, but this is a real, structural reason to doubt the thing shipped as
"already live and working" two commits ago, not a calibration nit.

**Correction to an earlier claim in this same doc:** "repair_pressure_v2 ... already runs live on
every real chat turn today" was asserted from `ENABLE_PRE_TURN_APPRAISAL=true` being set in the
running hub container. Real log evidence: **1 firing in 88 real chat turns over 7 days** (~1%), not
every turn. Config-on is not proof of frequency -- exactly the "runtime truth beats config truth"
rule this doc otherwise applies to `telemetry_anomaly`.

**Zero durable observability, found while trying to get a longer look-back window.** Docker's
logging driver on every cortex-exec container is `json-file` -- ephemeral, wiped by today's fleet
restart (22:41:41). No centralized log store exists (a "vector" container stack turned out to be
ChromaDB for recall embeddings, unrelated, a false lead). `repair_pressure`'s appraisal was never
persisted to Postgres either (`chat_history_log.client_meta`/`spark_meta` checked across 30 days:
zero `substrate_effect_summary` keys found). One real sample, ever, no way to look further back --
not a search-window problem, an actual absence-of-retention problem.

**Fix shipped:** `repair_pressure_appraisal_log`, a new standalone Postgres table
(`services/orion-sql-writer/app/models/repair_pressure_appraisal.py`), fed by a new consumer on the
already-existing `orion:repair_pressure:appraisal` channel (`orion-sql-writer` added as a second
consumer alongside `orion-equilibrium-service`; `orion/bus/channels.yaml`'s `schema_id` also
corrected from `MetacogRepairPressure`, which doesn't carry `correlation_id`, to the real dedicated
schema `RepairPressureAppraisalV1`, `orion/schemas/repair_pressure_appraisal.py`). Logs **every**
appraisal, gated or not -- the point is visibility into the real distribution, not just the ones
that happened to cross a threshold nobody has validated yet. Deliberately a standalone insert-only
table, not a patch onto `chat_history_log` the way `turn_change_appraisal` works: `repair_pressure`
computes pre-turn, before that row necessarily exists, and `orion-sql-writer`'s existing
`_apply_spark_meta_patch` errors and drops the patch if the target row is missing
(`services/orion-sql-writer/app/worker.py:732-745`) -- a patch-shaped fix would have silently lost
data on exactly the turns this table needs to capture.

**Bug caught while adding this table, unrelated to `repair_pressure` itself:** `severity`/`touches`
were added to `MetacogEntryV1` in the prior correction pass, but the matching SQLAlchemy columns on
`orion-sql-writer`'s `MetacogEntry` table were never added. `_row_dict`'s generic column-name filter
silently drops any payload key without a matching column -- both fields were being dropped on every
real insert into `orion_metacog`. Fixed (two new columns) and regression-tested
(`test_severity_and_touches_columns_exist_and_do_not_get_silently_dropped`).

**Still open, not resolved by this pass:** whether `confidence_floor=0.7` is the right threshold
given confidence's structural floor at 0.0 needs a real window of `repair_pressure_appraisal_log`
data to answer, not more code-tracing -- that's exactly what this table now makes possible. Revisit
once there's been enough live chat volume (at ~1 turn/hour of `repair_pressure` paradigm firings
observed so far, this needs days, not hours). `substrate_eventfulness_score`, `turn_effect`,
`biometrics`, `llm_uncertainty`, and `reasoning_excerpt` did not get the same full 6-step treatment
in this pass -- provenance was traced for all of them earlier in this conversation, but independence
(especially `turn_effect` vs `substrate_eventfulness_score`, both partly derived from `self_state`)
and live-data sanity checks were not completed for any of them due to the same short
container-uptime limitation. Worth a follow-up pass once `repair_pressure_appraisal_log` has
accumulated enough history to make a "did adding persistence actually help" comparison meaningful.

## `telemetry_anomaly` trigger: shipped, then arsonist-audited (2026-07-21)

PR #1224 wired the third trigger from this doc's taxonomy: `telemetry_anomaly`, sourced from a
trained autoencoder (`orion/mood_arc/fit_encoder.py`) scoring `field_channel_corpus.v1` (the
per-channel field-digester corpus, 17 selected channels, itself "Item 1 v2" per
`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`). Two live deploy bugs were found
and fixed same-session (PR #1228: a module-level `scripts.fit_phi_encoder` import crash-looped
`orion-field-digester`; PR #1231: the new env keys were never added to either service's
`docker-compose.yml` `environment:` list, so `.env` changes silently never reached the containers).
The pipeline is now proven live end-to-end, twice: once organically, once via a forced synthetic
publish, confirmed all the way into a real `orion_metacog` row (then deleted, since it was
synthetic-input-driven).

**Arsonist summary.** Real math, wired through a real, now-proven pipeline, trained on a corpus
whose own documentation was stale in ways that already changed the trusted feature set, scoring
against a threshold nobody validated, still carrying the exact decay-filter contamination risk that
killed the *prior* mood-arc encoder attempt (see "Correction, 2026-07-13 post-Item-2-spike" in the
roadmap spec), with no per-channel attribution to interpret a firing and zero consumers reading what
it produces. It works, in the narrow sense that data moves end to end. Whether it means anything was
unverified in four independent ways at once while it sat live in production.

| Component | Status | Verdict |
|---|---|---|
| Bus plumbing (digester -> equilibrium -> orch -> sql-writer -> table) | Proven live twice | **KEEP** |
| Corpus (`field_channel_corpus.v1`) | Real, growing -- but 3/17 trained channels (`memory_pressure`, `thermal_pressure`, `disk_pressure`) documented "folded-away, never produced" in `services/orion-field-digester/README.md`'s 2026-07-16 glossary are demonstrably alive and varying as of 2026-07-21 live data pulls | **RE-AUDIT** the glossary against current data before trusting it further |
| Decay-contamination check (`ceiling_ratio`, the AR(1)-surrogate comparison) | Computed for the trained artifact (0.240, directionally good -- real loss ~1/4 of a decay-only baseline) but `MoodArcEncoderManifestV1`'s own docstring says this field has **no calibrated pass/fail threshold across multiple runs yet** | **UNRESOLVED** -- this is the specific check built to catch the v1 failure mode |
| Alert threshold (`recon_error_p95 * 3.0`) | Carried over from the CLI's own default, never independently validated. The one real (non-forced) firing traced to a container-restart cold-start artifact; a 44-minute, 23-window sustained episode found in the historical `detect-anomalies` run traced to a genuine low-activity lull (cpu/thermal/execution/reasoning all *lower* than baseline, corroborated by only 9 real chat turns in that window) rather than a crisis | **UNCALIBRATED** -- real signal confirmed (regime shifts, not noise), but "anomalous" currently conflates "unusually busy" and "unusually quiet" into one undifferentiated flag |
| Per-channel attribution | Does not exist for this signal. `orion/self_state/builder.py::evidence_for_dimension()` already ships the identical pattern (rank contributing channels, top-N, human-readable strings) for `SelfStateV1` dimensions -- overlooked, not designed around, when this trigger was built | **BUILD** -- cheap: `xhat - x` is already computed per-window in `forward()`, just averaged away across channels; reshaping to `(window_size, n_fields)` and averaging over time only (not channels) gives per-channel error with no retrain |
| Upstream field-digester bugs (`contract_pressure`/`catalog_drift_pressure` exact-duplicate pair, several one-way-ratcheted channels, at least one merge-polarity masking bug) | Documented in the same stale 2026-07-16 glossary, never re-checked against the current 17 selected channels specifically | **UNVERIFIED** whether any of it still contaminates this model's inputs |
| Consumers of `orion_metacog` (any trigger_kind, not just this one) | Zero. Checked directly: no query against the table anywhere in the codebase. `orion-dream`'s own code has a comment claiming entries "now live in `orion_metacog`" with no actual read backing it | Every trigger this doc's taxonomy has shipped (`relational`, `baseline`, now `telemetry_anomaly`) writes into a table nothing reads. **Do not treat "wired a new trigger type" as progress toward the mission until at least one real consumer exists** |

**Self-audit of the session that shipped this** (kept here because the pattern is likely to
recur): status was reported as "done"/"verified" at least three times before it actually was --
after a deploy where the feature was silently still off (env keys never reached the container via
docker-compose), after finding one live firing that turned out to be a restart artifact, and before
per-channel attribution was checked against prior art already in the repo. A synthetic test row was
left sitting in the live `orion_metacog` table until asked directly whether it was fake, despite
already having flagged the risk before creating it. None of these were caught by re-running tests or
re-reading code -- all were caught by direct, repeated pushback demanding live evidence over
plumbing-works-therefore-it's-fine reasoning. Read as a standing instruction for this class of work,
not a one-off.

**Path to bulletproof, in dependency order (not yet started, see brainstorm output in the
2026-07-21 conversation for full detail):**
1. Re-audit the 17 trained channels against *current* live data (ratchet/duplicate/quantization
   checks), correcting the stale README glossary -- fast, zero-risk, and determines whether steps
   below are worth doing on the current channel set or should wait for a pruned one.
2. Build per-channel attribution (cheap, no retrain, reuses `evidence_for_dimension`'s shape).
3. Retrain excluding whatever step 1 finds contaminated; re-run the two-tier gate.
4. Establish a real `ceiling_ratio` threshold via multiple training seeds/slices instead of one
   uncalibrated number.
5. Add direction-awareness (elevated vs. depressed relative to learned normal) so "busy" and
   "quiet" stop looking identical.
6. Only then: wire one real consumer (`orion-dream` is the cheapest, already-half-intended seam)
   -- wiring a consumer to an unverified signal launders the verification gap one hop downstream
   instead of closing it.

Explicitly undecided: whether `FIELD_CHANNEL_ANOMALY_ENABLED` should revert to `false` in the live
`.env` while 1-4 above are outstanding, rather than staying live-and-unverified by default. Flagged,
not resolved, as of this section.

## `chat_turn` trigger: implementation spec (2026-07-22, design mode, not yet approved)

Status: **spec, not proposal-mode sign-off.** Per this doc's own "Next step" section and CLAUDE.md's
cognition-loop rule, this needs explicit go-ahead before code. Written now because the design
conversation that produced it (2026-07-22, arsonist-audited three times) is long enough that it needs
to live somewhere durable before it evaporates, same reason this whole doc exists.

**Read the "Correction" subsection below "Roadmap: Gate 1 and beyond" first** if skimming — an
earlier pass in this same section concluded a real orion-unified bug needed fixing before this could
be built ("Gate 1"); that conclusion was checked further and found wrong. There is no orion-unified
bug and no Gate 1. The correlator/gate design is simpler than originally specced (2 channels, not 7)
and is ready to react to as-is.

### Arsonist summary

Two arsonist passes already ran against pieces of this design during the conversation that produced
it (not yet against the whole thing end to end — that's the ask this section exists to satisfy).
Findings already folded in below, not re-litigated:

- The original claim that `MetacogDraftService`/`EnrichService`/`PublishService` run inline,
  unconditionally, on every chat turn was **wrong** — verified they're referenced in exactly one
  place repo-wide, `orion/cognition/verbs/log_orion_metacognition.yaml`. `chat_turn` as a trigger
  kind is genuinely unbuilt, not "gate something already firing." Nothing to remove.
- Two of five originally-proposed gate conditions (`grounding_capsule` conflict, `autonomy_mode`
  changed) have **zero producing code** anywhere. Carried forward from this doc's own earlier
  brainstorm prose without checking. Dropped from the gate; the raw fields they'd read
  (`grounding_capsule`, `autonomy_slice`) still get captured as context, just don't drive fire/no-fire.
  Reader ownership for the eventual real derived signal, if built, is out of scope for this pass.
  Note per the "Trigger-type taxonomy" section above: this whole doc's *original* 2026-07-18 framing
  of `chat_turn` as "existing, keep — gated on the turn-artifact conditions above instead of firing
  unconditionally" was itself describing pre-redesign `collapse_mirror` behavior, not anything in the
  current `orion_metacog`/`MetacogEntryV1` system — there was never a working `chat_turn` trigger in
  the redesigned system to gate. See "Files likely to touch" below for the trace this is grounded in.
- A field-by-field schema plan (typed `Optional` sub-models on `MetacogRealState`, mirroring the
  existing `MetacogRepairPressure` pattern, populated from `ctx["trigger"]["upstream"]`) was proposed
  and then found to be **premature**: `orion_metacog` has zero real consumers anywhere in the
  codebase (independently re-verified twice), and `ctx["trigger"]["upstream"]` is genuinely
  discarded by the `log_orion_metacognition` entry-building path specifically — but a **second,
  previously-untraced consumer** (`services/orion-actions/app/main.py::_handle_journal_metacog`,
  subscribed to the same `orion:equilibrium:metacog:trigger` channel) already reads the full
  `upstream` dict into a live journal-composition LLM prompt. The "evidence is discarded" framing
  was true for the metacog-entry pipeline specifically, false as a blanket statement. This spec does
  **not** propose the `MetacogRealState` schema extension — see "Non-goals."
- A real, independent bug was found in passing: `orion/bus/channels.yaml`'s
  `orion:equilibrium:metacog:trigger` registry entry is missing `orion-actions` from
  `consumer_services`, even though it's a live subscriber. Contract drift, CLAUDE.md section 6.
  Cheap, real, unrelated to whether the rest of this spec ships — worth its own small fix regardless.

### Correction (2026-07-22, same conversation): `post_turn_closure` is not a bug, and there is no "Gate 1"

The full arsonist pass below (kept for its still-valid findings) concluded `orion:substrate:
post_turn_closure`'s failure to publish on 5 of 6 real exit paths in `handle_harness_run_request`
was an integrity gap in `orion-unified` needing its own upstream fix ("Gate 1") before the `chat_turn`
trigger could be built. **That conclusion was wrong, and there is no orion-unified bug to fix.**

Checked the actual code and PR history for all 5 non-emitting paths, not just asserted the pattern
from one:

- **Path 4** (`SubstrateAppraisalUnavailableError`): confirmed in writing —
  `docs/superpowers/pr-reports/2026-07-16-substrate-finalize-degrade-passthrough-pr.md`: *"Deliberately
  did not attempt turn_outcome_molecule/post_turn_closure: emit_turn_outcome_molecule's
  substrate_appraisal/reflection params are non-optional... fabricating placeholder ones to force the
  emission would be exactly the empty-shell-cognition anti-pattern this repo prohibits... Scope was
  confirmed with Juniper before implementation."* In-code comment at `bus_listener.py:339-344` matches
  verbatim.
- **Paths 1 and 2** (refused, no draft): both short-circuit *before* the finalize chain runs at all —
  5a/5b never executed, structurally identical "nothing real to report" situation as path 4.
- **Path 3** (`HarnessFinalizeFailedError`, only 5c failed): the *only* path with real data available
  — `exc.partial.substrate_appraisal`/`.reflection` genuinely exist, because 5a and 5b succeeded
  before 5c broke — and it is, correspondingly, the *only* one of the 6 paths that does emit closure.
- **Path 5** (generic `except Exception` from the finalize chain, `bus_listener.py:378-394`): read the
  actual branch — no partial-data recovery at all, nothing analogous to path 3's `exc.partial`.
  Genuinely nothing real to report, same as 1/2/4.
- **Path 6** (outer `_handle_bus_message` handler): even further removed from the finalize chain,
  same story.

Five for five, one consistent rule, correctly applied every time: emit closure only when real
`substrate_appraisal`/`reflection` data exists to compute `surprise_resolved` from honestly; never
fabricate it. There is no asymmetry to fix, no oversight, no "Gate 1." This was already someone
else's correct application of the exact "no empty-shell cognition" rule this whole design conversation
has been leaning on — I mischaracterized it as a bug because I only checked one of the five paths in
detail before generalizing.

**The actual fix is entirely on the `chat_turn` design's side, and it simplifies the correlator
significantly**: stop treating `post_turn_closure` as the "turn is over, evaluate now" trigger. It
was never meant to mean that — it means "turn is over *and* real reflection happened," a narrower and
deliberately rarer thing. `orion:harness:run:artifact` (`HarnessRunV1`) is the actual "turn is over"
signal — confirmed to publish on all 6 real exit paths, always carrying honest data
(`compliance_verdict`, `exit_code`, `grounding_status`, `finalize_degraded_reason`) even when the
finalize chain never got far enough to produce anything richer.

Better still: `HarnessRunV1` already embeds `substrate_appraisal: SubstrateFinalizeAppraisalV1 | None`
and `reflection: FinalizeReflectionV1 | None` directly as fields, populated whenever they exist (paths
3 and the full-success path). That makes `orion:harness:verdict:artifact` (`HarnessVerdictMoleculeV1`,
which just wraps the same `FinalizeReflectionV1`) and `orion:substrate:turn_outcome`
(`HarnessTurnOutcomeMoleculeV1`, whose `alignment_verdict`/`surprise_level_at_draft` are re-derived
from the same appraisal+reflection) **redundant re-publications of data `run:artifact` already
carries** — not additional signal, just the same facts under a second name. `orion:substrate:
post_turn_closure` itself only ever references those same objects by ID (`verdict_molecule_id`,
`outcome_molecule_id`), never carries new data of its own.

**Net effect: the correlator collapses from 7 subscribed channels down to 2** — `orion:thought:
artifact` (for `disposition`/`boundary_register`/`grounding_capsule`/`autonomy_slice`, which
`HarnessRunV1` doesn't carry) and `orion:harness:run:artifact` (for everything else, always present,
richer when the finalize chain got further). No waiting on a signal that might never arrive, no TTL-
leak risk from an absent closure, no redundant subscriptions to three channels re-publishing the same
underlying facts. This is simpler *and* more correct than the original 7-channel design — see the
revised "Proposed schema / API changes" below for the corrected gate conditions and `upstream` shape.

The three findings below the original "Gate 1" framing (cost/fan-out risk, evidence thinness, gate
condition #10 having no real basis) are still real and still apply — kept as-is, just no longer gated
behind a nonexistent upstream fix.

- **Cost/fan-out risk, not previously priced in**: `orion-actions`' `_handle_journal_metacog` is a
  live, already-subscribed consumer of every `MetacogTriggerV1`, firing its own LLM call per trigger
  with a dedupe key that doesn't meaningfully rate-limit (hashes on `timestamp`+`reason`, both
  different per firing). Shipping `chat_turn` means every fired turn costs a second LLM call
  somewhere else in the system, silently. Needs a real answer before this ships.
- **Evidence thinness**: already corrected directly in "Proposed schema / API changes" below —
  `disposition_reasons`/`alignment_notes`/`grounding_status` added to the `upstream` payload.
- **Gate condition (failed tool-step count) has no real basis**: `GrammarReceiptV1.summary` is free
  text with no structured success/fail marker — dropped from the condition list below pending real
  design, not carried forward as if it exists.

**Forest, not just trees**: the broader shape of this paradigm is unchanged by any of this — reuse the
existing `telemetry_anomaly`/`relational` dispatch mechanism, ground every gate condition in one real
schema field, keep `MetacogRealState` untouched until a real consumer exists, capture
`grounding_capsule`/`autonomy_slice` as context without inventing derived judgments for them. What
changed is narrower and better: fewer channels, no phantom upstream blocker, a correlator that fires
reliably instead of one keyed to a signal that structurally can't arrive on the turns that matter most.

See also `docs/superpowers/specs/orion-semantic-self-indexing-design-spec.md` §5.2 for the
independently-authored source of the 12-step `orion-unified` pipeline trace this whole section is
built on.

<details>
<summary>Original "Roadmap: Gate 1 and beyond" section (2026-07-22, superseded by the correction
above — kept for the record, not as current guidance)</summary>

A full end-to-end arsonist pass (2026-07-22, against this entire section, not a narrow claim within
it — see "Arsonist summary" above for the earlier, narrower passes) found one **disqualifying**
architectural problem: the design fires the correlator/gate on `orion:substrate:post_turn_closure`,
treating it as "this turn is finalized, safe to evaluate." Traced `services/orion-harness-governor/
app/bus_listener.py::handle_harness_run_request` path by path — **that event does not publish on at
least 5 real exit paths**, including a refused request (`compliance_verdict="refused"`), a failed
draft (`"failed"`), and the generic exception handler. `orion:harness:run:artifact` (`HarnessRunV1`)
*does* publish on every one of those paths. This isn't a parameter to tune (the open TTL/threshold
questions below) — it's a wrong signal choice, and it means the design as originally specced would
systematically fail to fire on exactly the worst turns two of its own 8 gate conditions
(`compliance_verdict != "completed"`, non-zero `exit_code`) exist to catch.

That finding reframes the whole effort. **This is not a bug in the `chat_turn` trigger design — it's
a real, pre-existing integrity gap in `orion-unified`'s own turn-completion signaling**, independent
of metacog entirely. Any future consumer that ever wants a reliable "this turn is over, evaluate it"
hook — not just this one — hits the same gap. Fixing it belongs upstream of, and before, anything
metacog-specific gets built on top of it.

So the path forward is staged, not abandoned:

- **Gate 1 (prerequisite, not yet scoped as its own patch): fix `orion-unified`'s turn-completion
  signal integrity.** Either make `orion:substrate:post_turn_closure` (or a real equivalent) publish
  on every exit path `orion:harness:run:artifact` already covers, or establish `orion:harness:run:
  artifact` itself as the canonical "turn is over" signal for any future consumer, metacog or
  otherwise. This needs its own investigation (why were the closure paths built asymmetrically —
  intentional fail-open design per the in-code "empty-shell cognition" comment on one of them, or an
  oversight on the others?) before a fix is designed. Out of scope for this document to resolve by
  itself; flagged here as the actual next concrete step.
- **Gate 2 and beyond: everything else already specced below** — the correlator, the 8 gate
  conditions, the `upstream` payload, the `chat_turn` trigger_kind registration — unchanged in shape,
  but blocked on Gate 1 landing first.

</details>

### Current architecture

**The pipeline this trigger targets** — the real, verified `orion-unified` Hub chat lifecycle
(`docs/superpowers/specs/orion-semantic-self-indexing-design-spec.md` §5.2), not the older
direct-cortex-request path. Twelve steps, traced against real code (file:line citations in "Files
likely to touch" below), not the spec doc's prose:

1. `execute_unified_turn()` emits surface observation, optionally runs pre-turn appraisal.
2. Hub builds `HubAssociationBundleV1` (substrate broadcast + trajectory slice + repair bundle).
   **Local-only** — never published.
3. Hub sends `StanceReactRequestV1` to the Thought service over bus RPC.
4. Thought service returns `ThoughtEventV1` — **published in full** on `orion:thought:artifact`.
   Contains `disposition` (`proceed`/`defer`/`refuse`), `disposition_reasons`, `boundary_register`,
   `grounding_capsule` (`GroundingCapsuleV1`), `autonomy_slice` (`AutonomySliceV1 | None`).
   Currently zero real subscribers to this channel besides a diagnostic tracer.
5. Hub honors defer/refuse or builds `HarnessRunRequestV1`. **Local-only.**
6. `HarnessGovernorClient` sends the request over bus RPC. Per-step FCC tool-call events publish to
   `orion:harness:run:step` — real, structured, but pure ephemeral pub/sub (WS relay + liveness
   only, nothing persists the sequence). **Not subscribed by this design** — no gate condition
   depends on it (see correction above and "Proposed schema / API changes" below).
7. Governor maps the repair overlay (`HarnessRepairOverlayV1`). **Local-only** — distinct from the
   already-published `orion:repair_pressure:appraisal` summary from step 1/2 (same underlying
   repair_pressure paradigm data, two different downstream fates).
8. `compile_harness_prefix()` builds the FCC motor prompt string. **Local-only.**
9. `run_fcc_turn()` runs the real Claude/FCC tool loop. Per-step grammar receipts publish to
   `orion:grammar:event` (same `GrammarEventV1` mechanism already feeding
   `execution_friction`/`failure_pressure` elsewhere in the system). `HarnessDraftMoleculeV1` itself
   is **local-only** beyond one RPC hop. **Not subscribed by this design**, same reason as step 6.
10. `run_harness_finalize_chain()` — substrate appraisal (5a) and integrative reflection (5b) run
    only when the finalize chain gets this far (paths 1/2/4/5/6 in the correction above never reach
    here at all, or exit before 5b completes); when they do, both objects ride whole inside step
    11's `HarnessRunV1.substrate_appraisal`/`.reflection` fields — that's this design's actual source
    for them. `HarnessVerdictMoleculeV1` (`orion:harness:verdict:artifact`) and
    `HarnessTurnOutcomeMoleculeV1` (`orion:substrate:turn_outcome`) are separately published here too,
    but confirmed to be **redundant re-publications of the same `FinalizeReflectionV1`/
    `SubstrateFinalizeAppraisalV1` objects already embedded in `run:artifact`** — not subscribed by
    this design (see correction above).
11. Governor returns `HarnessRunV1` — **published in full** on `orion:harness:run:artifact`
    (`compliance_verdict`, `grounding_status`, `exit_code`, `finalize_degraded_reason`, `recall_debug`,
    `memory_digest`, `substrate_appraisal`, `reflection`) — **on every one of the 6 real exit paths,
    always with honest data even when the finalize chain barely ran**. This is the design's actual
    "turn is over, evaluate now" signal. `HarnessPostTurnClosureV1`
    (`orion:substrate:post_turn_closure`) only fires on 1 of those 6 paths (deliberately — see
    correction above) and only ever references the verdict/outcome objects by ID, carrying no data of
    its own — **not subscribed by this design**.
12. Hub persists chat-history/chat-turn envelopes to Postgres (`chat_history_log`, via
    orion-sql-writer) and a Spark introspection candidate. **The only durable step in the whole
    pipeline** — everything from step 4 through 11 is Redis pub/sub with no persistence; gone the
    instant nothing is subscribed.

**The existing metacog-entry mechanism** (already live, already correct, reused as-is):
`orion-equilibrium-service`'s trigger-kind gate files (`telemetry_anomaly_metacog_gate.py`,
`repair_pressure_metacog_gate.py`) publish `MetacogTriggerV1` to `orion:equilibrium:metacog:trigger`.
`orion-cortex-orch`'s `dispatch_metacog_trigger()` (`orchestrator.py:813`) consumes it and dispatches
a fresh, independent plan run of `log_orion_metacognition.yaml`
(`MetacogContextService` → `MetacogDraftService` → `MetacogEnrichService` → `MetacogPublishService`),
which builds and writes a real `MetacogEntryV1` row to `orion_metacog` via orion-sql-writer. This
mechanism is unconditionally reused, not modified, by this spec.

**The plumbing gap this spec does NOT fix (see Non-goals)**: `MetacogPublishService`'s `state`
construction (`executor.py:3649-3664`) reads generic, current-snapshot `ctx` values
(`ctx["metadata"]["substrate_effect_summary"]` for `repair_pressure`, plus `biometrics`/
`turn_effect`/`substrate_eventfulness_score`/`llm_uncertainty`) — never `ctx["trigger"]["upstream"]`,
the specific evidence that caused the firing trigger. Confirmed this is true for `telemetry_anomaly`
today: its rich per-channel attribution (`top_channels`, `deviation_direction`, `attribution_ok`)
never reaches the `orion_metacog` row or the draft/enrich narrative LLM call (draft template only
renders `trigger.trigger_kind`/`reason`/`pressure`/`zen_state`). It *does* reach a real consumer via
a separate path — `orion-actions`' journal-compose prompt — just not this one.

### Missing questions (open, need Juniper's call before implementation)

1. **Correlator placement and lifetime.** Proposed: a new module in
   `services/orion-equilibrium-service/app/chat_turn_metacog_gate.py` (matching the file-per-
   trigger-kind convention) subscribes to `orion:thought:artifact` and `orion:harness:run:artifact`
   only (see correction above — the other 5 originally-planned channels are either redundant or
   unused by any current gate condition), accumulates by `correlation_id` in a short-TTL Redis hash,
   and evaluates/fires as soon as `run:artifact` arrives for a known `correlation_id` — no waiting on
   a second signal, since `run:artifact` is now confirmed to be the actual "turn is over" event on
   every real path. TTL still needs a real value (long enough to span `thought:artifact` arriving
   well before a slow FCC tool loop finishes; short enough not to leak entries for turns whose
   `thought:artifact` arrives but `run:artifact` genuinely never does — e.g. a mid-flight
   `HarnessRunCancelV1`). Not yet measured against real turn-duration data.
2. **Threshold value** for the one remaining numeric gate condition
   (`substrate_appraisal.surprise_level >= X`) — no existing config to inherit from; would need to be
   a new equilibrium-service setting, tuned against real data the same way `telemetry_anomaly`'s
   `threshold_multiplier` was.
3. ~~Whether the redundant surprise signal gets deduplicated correctly~~ — **moot as of the
   correction above.** `turn_outcome`/`post_turn_closure` are no longer subscribed at all; there is
   only one surprise field now (`run_artifact.substrate_appraisal.surprise_level`), nothing to
   deduplicate.
4. **Whether `orion_metacog` should keep growing at all**, given zero real consumers and a sibling
   table (`orion_metacognitive_trace`, fed by a different producer path,
   `kind == "metacognitive.trace.v1"` per `orion-vector-writer`) that already has a real reader. Not
   answered here — flagged for Juniper, matches this doc's own "path to bulletproof" pattern of
   naming undecided things instead of quietly picking one.
5. **Whether the cheap prompt-template fix (below) ships in the same patch or a separate one.**

### Proposed schema / API changes

**No new Pydantic schema.** The correlator holds the real, already-registered types as-is —
deliberately not inventing a new bus contract for the accumulator, since it's ephemeral internal
state, not a durable artifact. Simplified as of the correction above — down to the 2 channels that
are actually needed, not 7:

```python
# services/orion-equilibrium-service/app/chat_turn_metacog_gate.py (new)
class ChatTurnEvidenceAccumulator:
    correlation_id: str
    thought_event: ThoughtEventV1 | None = None
    run_artifact: HarnessRunV1 | None = None   # .substrate_appraisal, .reflection nested when present
```

**Gate conditions, each pinned to one real field, no invented booleans — all read off just these two
objects:**

| Condition | Field | Type |
|---|---|---|
| Deferred or refused | `thought_event.disposition != "proceed"` | `Literal["proceed","defer","refuse"]` |
| Boundary hit | `thought_event.boundary_register is True` | `bool` |
| Misaligned/uncertain (only evaluable when reflection ran) | `run_artifact.reflection is not None and run_artifact.reflection.alignment_verdict != "aligned"` | `Literal["aligned","misaligned","uncertain"] \| None` |
| Strain unresolved (only evaluable when reflection ran) | `run_artifact.reflection is not None and run_artifact.reflection.strain_unresolved is True` | `bool \| None` |
| Surprise crossed threshold (only evaluable when 5a ran) | `run_artifact.substrate_appraisal is not None and run_artifact.substrate_appraisal.surprise_level >= X` (threshold TBD, question 2) | `float [0,1] \| None` |
| Compliance short of complete | `run_artifact.compliance_verdict != "completed"` | `Literal["completed","partial","failed","refused"]` |
| Non-zero exit | `run_artifact.exit_code not in (0, None)` | `int \| None` |
| Degraded (infra unavailable, not a content failure — see correction above, path 4) | `run_artifact.finalize_degraded_reason is not None` | `str \| None` |
| Failed/retried tool steps — **not built, no existing basis.** Originally described as reusing `compute_severity`'s `non_ok_step_count` pattern; verified false — that function takes a pre-computed int, and `GrammarReceiptV1.summary` (the only field this could read) is free text with no structured success/fail marker (`summarize_harness_step`, `orion/harness/fcc_motor.py`). Dropped until a real classifier is designed. | n/a — not a real condition today |

**`MetacogTriggerV1.upstream` shape when fired** — carries the fired condition list (same
multi-value shape as `telemetry_anomaly`'s `top_channels`) plus the full accumulated evidence, so a
downstream reader never has to re-derive what happened:

```python
upstream={
    "fired_conditions": [...],          # e.g. ["disposition=defer", "alignment_verdict=misaligned"]
    "disposition": thought_event.disposition,
    "disposition_reasons": thought_event.disposition_reasons,   # the machine verdict alone is
                                                                  # near-unreadable without this
    "compliance_verdict": run_artifact.compliance_verdict,
    "grounding_status": run_artifact.grounding_status,
    "exit_code": run_artifact.exit_code,
    "finalize_degraded_reason": run_artifact.finalize_degraded_reason,
    # Present only when the finalize chain got this far (paths 3 / full-success in the correction
    # above) -- None on paths 1/2/4/5/6, and the upstream reader must handle that, not assume presence.
    "alignment_verdict": run_artifact.reflection.alignment_verdict if run_artifact.reflection else None,
    "alignment_notes": run_artifact.reflection.alignment_notes if run_artifact.reflection else None,
    "strain_unresolved": run_artifact.reflection.strain_unresolved if run_artifact.reflection else None,
    "surprise_level": run_artifact.substrate_appraisal.surprise_level if run_artifact.substrate_appraisal else None,
    "grounding_capsule": thought_event.grounding_capsule,   # raw context, not a fire condition
    "autonomy_slice": thought_event.autonomy_slice,          # raw context, not a fire condition
}
```

**`ROLE_TO_PRESSURE_KIND`-equivalent registration**: `MetacogTriggerV1.trigger_kind`'s docstring
enum (`orion/schemas/telemetry/metacog_trigger.py`) needs `"chat_turn"` added to its documented
list — it's currently absent even as a name (the only `"chat_turn"`-producing code today is the
orphaned `_metacog_trigger_lineage()` helper, whose output never reaches a persisted column).

**Channel registry fix (independent, ships regardless)**: `orion/bus/channels.yaml`'s
`orion:equilibrium:metacog:trigger` entry gets `orion-actions` added to `consumer_services`.

### Files likely to touch

- `services/orion-equilibrium-service/app/chat_turn_metacog_gate.py` (new) — correlator + gate.
- `services/orion-equilibrium-service/app/settings.py` — new threshold config (question 2).
- `services/orion-equilibrium-service/app/service.py` — wire the new subscriptions (mirrors how
  `telemetry_anomaly_metacog_gate.py` gets invoked from `service.py` today).
- `orion/schemas/telemetry/metacog_trigger.py` — docstring enum update only, no field changes.
- `orion/bus/channels.yaml` — register `orion-equilibrium-service` as a consumer of
  `orion:thought:artifact` and `orion:harness:run:artifact` (both already carry a wildcard `["*"]`
  consumer list today, so this may already be satisfied — verify before assuming an edit is needed),
  plus the independent `orion-actions` fix on `orion:equilibrium:metacog:trigger`.
- `orion/cognition/prompts/log_orion_metacognition_draft.j2` /
  `log_orion_metacognition_enrich.j2` — optional, if the prompt-evidence-cue fix ships in this
  patch (question 5): render `trigger.upstream`'s `fired_conditions`/key fields as an evidence-cue
  block, same style as the existing `spark_state_json`/`turn_effect_json` blocks ("evidence cues
  only, do not paste verbatim").
- Tests: new `services/orion-equilibrium-service/tests/test_chat_turn_metacog_gate.py`, matching
  `test_telemetry_anomaly_metacog_gate.py`'s shape (per-condition unit tests, dedup test, TTL/
  correlation test).

Sources for every file:line citation above, verified against real code during this conversation:
`orion/hub/turn_orchestrator.py`, `orion/hub/association.py`,
`services/orion-hub/scripts/thought_client.py`, `services/orion-thought/app/bus_listener.py`,
`orion/schemas/thought.py`, `services/orion-harness-governor/app/bus_listener.py`,
`orion/harness/runner.py`, `orion/harness/finalize.py`, `orion/schemas/harness_finalize.py`,
`services/orion-substrate-runtime/app/finalize_appraisal_listener.py`,
`services/orion-cortex-orch/app/orchestrator.py`, `services/orion-cortex-exec/app/executor.py`,
`orion/cognition/verbs/log_orion_metacognition.yaml`, `orion/schemas/metacog_entry.py`,
`orion/metacog/service.py`, `services/orion-actions/app/main.py`,
`orion/journaler/worker.py`, `orion/bus/channels.yaml`.

### Non-goals

- **No `MetacogRealState` schema extension.** Explicitly deferred pending either a real consumer of
  `orion_metacog` existing, or a decision that `orion_metacog` should be retired in favor of
  `orion_metacognitive_trace`. Building typed sub-models for a table nothing reads is the CLAUDE.md
  0A "schema without consumer" pattern this doc's own metric-quality-gate section exists to block.
- **No "grounding_capsule conflict" or "autonomy_mode changed" derived signal.** Zero producing code
  exists for either. Not designed here; a future pass, with its own metric-quality-gate run, if
  ever wanted.
- **No change to the existing `telemetry_anomaly`/`relational` gates or the `log_orion_metacognition`
  plan's draft/enrich/publish logic** beyond the optional prompt-template evidence-cue addition
  (question 5) — this spec reuses that machinery, doesn't modify it.
- **No removal of anything** — there is no existing inline chat-turn metacog path to remove (see
  arsonist summary). This is pure addition.
- **Not building a synchronous request/response mechanism.** An earlier pass in this design
  conversation proposed one, based on an incorrect assumption that draft/enrich needed to run
  inside the same live turn. Verified false — `log_orion_metacognition` is already a fully
  independent, freshly-dispatched plan for every trigger kind that uses it today. Fire-and-forget,
  same as `telemetry_anomaly`/`relational`.

### Acceptance checks

- A chat turn where `thought_event.disposition == "defer"` produces exactly one `MetacogTriggerV1`
  with `trigger_kind="chat_turn"`, `fired_conditions` containing `"disposition=defer"`, published
  once.
- A chat turn where none of the real conditions fire produces zero trigger events — no entry
  written, matching the whole point of gating instead of firing unconditionally.
- The correlator does not leak: a turn whose `thought_event` arrives but whose `run_artifact` never
  does (e.g. a mid-flight `HarnessRunCancelV1`) does not hold its Redis-hash entry past the TTL.
- **The actual regression test that matters most**: a chat turn that hits a refused/failed/crashed
  exit path (paths 1/2/4/5/6 in the correction above) still produces a `run_artifact` event, and the
  gate still correctly fires on `compliance_verdict`/`exit_code`/`finalize_degraded_reason` for it —
  confirming the corrected design actually catches the turns the original, `post_turn_closure`-keyed
  design would have silently missed.
- A chat turn where the finalize chain fully ran also correctly evaluates the
  `alignment_verdict`/`strain_unresolved`/`surprise_level` conditions, sourced from
  `run_artifact.reflection`/`.substrate_appraisal` directly (not from a separate subscription).
- Live-data check, same standard as `telemetry_anomaly`'s bulletproofing pass: pull real
  `orion_metacog` rows with `trigger_kind="chat_turn"` post-deploy and confirm `upstream` actually
  reflects the real turn (not degenerate/always-the-same-value) before calling this verified.
- `orion/bus/channels.yaml` validation passes with the new subscriptions and the `orion-actions`
  registry fix. Correction (2026-07-22 arsonist pass): `scripts/check_bus_channels.py`, cited here
  originally and in this repo's own CLAUDE.md §11, **does not exist**. Real mechanisms present:
  `scripts/check_single_consumer_channels.py` (confirmed no conflict — neither new channel is
  `single_consumer: true`) plus the pytest catalog tests (`tests/test_channel_prefix_guardrail.py`,
  `test_*_bus_catalog.py`).

### Recommended next patch

No upstream fix needed (see correction above — there is no "Gate 1"). Two small, independently-
shippable pieces that don't wait on any open question: the channel registry fix (`orion-actions`
added to `orion:equilibrium:metacog:trigger`'s `consumer_services` — real, free, zero design risk),
and the prompt-template evidence-cue change (question 5) for `telemetry_anomaly`/`relational`'s
already-live triggers, which delivers real narrative-quality improvement today on already-shipped
triggers.

The `chat_turn` gate itself is now real, scoped, and only blocked on the cost/fan-out question
(`orion-actions`' journal path) having a real answer, on top of the pre-existing open questions 1-4
(TTL, threshold, and the `orion_metacog`-vs-`orion_metacognitive_trace` fate question — question 3
is now moot). This section is the spec to react to, not yet a green light.
