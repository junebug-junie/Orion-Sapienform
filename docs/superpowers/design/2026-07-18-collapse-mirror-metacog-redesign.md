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
4. Live-data check: **not yet done in this doc.** Confidence/novelty distributions for real,
   recent `shift_kind` values haven't been pulled. Open item before treating this as settled.
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
