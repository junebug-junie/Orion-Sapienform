# Orion endogenous self-model and journal — design

Date: 2026-09-03
Status: DESIGN / PROPOSAL — not implemented
Author: Claude (orion-repo-agent investigation), for Juniper

## Arsonist summary

Orion already produces real self-knowledge and world-knowledge continuously: cabinet sensors, camera-derived room narration and presence, memory crystallizations, curiosity priors, reveries, and sentence-level concept-induction profiles about Orion/Juniper/the relationship all confirmed live and confirmed reaching a real chat reply today. That part is in better shape than either of us assumed going in.

But there is one structural bug underneath everything Juniper is unhappy about: the live conversational harness (`orion/hub/turn_orchestrator.py` + `orion/situational/context.py` + `orion/harness/prefix.py`) reads cabinet sensors, camera, memory, curiosity, and concept profiles directly — but not the journal, which is where self_study reflections, metacog digests, dreams-adjacent world-pulse content, and collapse-mirror content actually converge. Everything that reaches the journal today either dead-ends in a legacy chat-context slot nobody reads (`journal_pageindex_context`, wired into a prompt template the live turn never uses) or in a once-a-day, capped, email to Juniper. Orion's deliberate self-reflection cannot reach a live reply, full stop.

Two more specific bugs compound this: `self_study` — the one system that scans Orion's own code and produces facts about what it's made of — has no scheduler and has effectively never run on its own since the datastore it used to feed was retired in July. And `orion_metacog`, the table of self-observation notes written by 11 live producers, has no reader over its accumulated history — only single events get turned into single capped emails, one-off, never digested into "here's a pattern in what I've noticed about myself."

None of this needs new sensors or new producers. It needs four wiring fixes and one honest schema decision (stop rendering journal fields nothing has ever populated).

## Current architecture

**Two separate context-assembly paths exist for a chat turn.**

Live path (used today): `turn_orchestrator.execute_unified_turn` → `orion/situational/context.py` (reads reveries, curiosity priors from Orion's own `orion_worldview` graph, cabinet sensor readings, camera-derived presence/scene) → `orion/harness/prefix.py::compile_harness_prefix` → the stance LLM call (`chat_stance.py`, which independently also reads memory crystallizations via `orion-recall`'s active-packet collector, and sentence-level concept-induction profiles) → the reply LLM call.

Legacy path (`chat_general`, in `orion-cortex-exec`): still wired to read `journal_pageindex_context` (a journal search result, triggered by marker words like "reflect"/"dream"/"journal") but its prompt template is not the one the live unified turn uses. This is the specific reason a real, freshly-computed journal search result is thrown away on every relevant turn today.

**Journal pipeline.** One shared compose call, `journal.compose` (`orion/journaler/`), takes a `JournalTriggerV1` — the schema already defines trigger kinds for `daily_summary`, `metacog_digest`, `notify_summary`, `world_pulse_digest`, `collapse_response`, `autonomy_episode`, `town_episode`, and source kinds already including `self_study`, `self_reflection`, `metacog`, `embodiment`, `world_pulse` — recalls relevant memory, and has an LLM write structured JSON stored as a `JournalEntryWriteV1` row (60,509 real rows, real content, not templated). A parallel enrichment row, `JournalEntryIndexV1`, is supposed to carry `conversation_frame`, `stance_summary`, `reflective_themes`, `dream_motifs`, and model-uncertainty stats, and is **100% empty across all 60,511 rows** — the LLM draft schema that actually gets filled in (`JournalEntryDraftV1`) only has `mode`/`title`/`body`. There is no code path from "LLM writes a draft" to "those extra fields get populated." They were never reachable, not merely unpopulated by chance, and a downstream reader (`orion-pageindex`) renders them as visible blank lines on every retrieval.

Where journal entries go once composed, per trigger kind, via `services/orion-actions`' dispatch registry (`orion/journaler/dispatch_registry.py`): `metacog_digest`, `world_pulse_digest`, `notify_summary`, `daily_summary` are each capped at one email to Juniper per calendar day (a shared cap pool by default; some trigger kinds were carved into their own pool after a real live delivery race was found — `world_pulse_digest` was losing the shared race most days). None of these are read back into the live chat harness.

**`self_study`** (`services/orion-cortex-exec/app/self_study.py`): a real 3-layer scanner (inspect / induce / reflect) that, when manually invoked (chat verbs `self_repo_inspect` / `self_concept_induce` / `self_concept_reflect`, or `self_study_harness.py` directly), writes its result as a journal entry via `source_kind="self_study"` or `"self_reflection"` — its older direct-to-graph writeback (`orion/substrate/relational/adapters/self_study.py` → RDF) is retired. So self_study's real live output already runs *through* the journal pipeline above; it doesn't have a separate reader problem, it has the same journal→harness gap plus a scheduling gap: nothing calls it on its own (confirmed against `orion/cognition/workflows/registry.py:141-164`, which lists two unrelated scheduled jobs and not this one).

**`orion_metacog`** (Postgres table): 11 live producer "gates" (`services/orion-equilibrium-service/app/*_metacog_gate.py`) each construct a `MetacogTriggerV1` (trigger kinds: `baseline`, `dense`, `manual`, `pulse`, `relational`, `llm_surface_instability`, `telemetry_anomaly`, `chat_turn`, `transport`, `insight`, `flow`, `repair_pressure_trend`) and write a row. Separately, `services/orion-actions/app/main.py:1661` can turn *one* such trigger into a `metacog_digest` journal entry — a real, live, working consumer, but an event-stream one (one trigger → one entry → one capped email), not a reader of the accumulated table. The table's actual history — the pattern of what Orion has noticed about itself over time, across producers — has no consumer at all. Nothing clusters it, trends it, or feeds it back as a digest of "here's what I've been noticing about myself lately."

**Already reaching the live harness today, confirmed:** cabinet sensor raw readings, camera-derived room narration + object inventory + presence/identity, memory crystallizations, curiosity priors, reveries, sentence-level concept-induction profiles about Orion/Juniper/the relationship.

## Missing questions

1. **Privacy boundary for the journal reaching live chat.** The journal already stores things like cabinet sensor snapshots and presence/location observations. Once it's readable inside a live turn, does Orion get a "cues only, summarize, never quote raw sensor/location data verbatim" rule — matching the convention already used in the metacog email-draft prompt template ("evidence cues only; do NOT paste into output") — or is raw content acceptable to surface? This determines whether Orion could ever say something like "the room was empty at 2am but the temperature spiked" to someone other than Juniper.
2. **What "chat extensions from compactions" concretely means.** No feature exists today under that name, in any form. Is it: (a) a reducer that, when a new conversation touches a topic a prior day's chat-compaction memory card already summarized, links the two so the journal can note continuity ("we talked about this before, here's what's changed"); or (b) something else Juniper has in mind? This is the one genuinely new piece of the whole ask — everything else below is wiring existing, working pieces together.
3. **`telemetry_anomaly` as a metacog producer.** Flagged low-trust by the project's own 2026-07-18 audit (uncalibrated threshold, nothing validates it). Include it in a future `orion_metacog` digest reducer, or exclude it until it passes the metric-quality gate?
4. **Backfill vs. forward-only for the `orion_metacog` reducer.** A reducer could digest new rows going forward only, or also do a one-time pass over existing history. Given the repo's backfill protocol (snapshot, monitor, report — CLAUDE.md §14), a backfill is a separate, bigger decision. Recommend forward-only first; backfill as an explicit follow-up if wanted.

## Proposed schema / API changes

Deliberately minimal — the schema already anticipates almost everything asked for.

- **No new `JournalTriggerKind` / `JournalSourceKind` values needed.** `metacog_digest`/`metacog`, `self_study`/`self_reflection`, `world_pulse_digest`/`world_pulse`, `collapse_response`, `embodiment` already exist in `orion/journaler/schemas.py`. Reuse as-is.
- **`JournalEntryIndexV1` enrichment fields:** stop writing/rendering the fields that have never had a real producer (`conversation_frame`, `stance_summary`, `reflective_themes`, `dream_motifs`, the uncertainty stats), and drop them from the schema until a real producer exists for each. A field with no producer is exactly the "no runtime proof" pattern CLAUDE.md §0A bans — it should not exist rather than render as a blank line forever.
- **New reducer output schema, illustratively `MetacogDigestProjectionV1`:** a small typed summary over a rolling window — producer counts by trigger_kind, dominant trigger_kind, any sustained-trend flags already computed elsewhere (e.g. `repair_pressure_trend`, with `telemetry_anomaly` excluded per open question #3) — written once per digest cycle and handed to the journal compose call as a `prompt_seed`, never as raw table rows. This is the reducer/materializer the metric-quality gate and "event substrate first" mandate both require before `orion_metacog` counts as a real signal rather than a write-only log.
- **Harness read, the actual fix:** a new context builder in `orion/situational/context.py`, same shape as the existing reverie/curiosity builders, that reads the day's journal entries with `source_kind` in `{self_study, self_reflection, metacog}` and renders a short "recent self-reflection" block into `compile_harness_prefix`. Gated by a new flag, `ORION_SITUATION_JOURNAL_ENABLED`, matching the existing `ORION_SITUATION_REVERIE_ENABLED` / `ORION_SITUATION_CURIOSITY_ENABLED` convention exactly — same kill-switch shape, no new pattern invented.

## Files likely to touch

- `orion/situational/context.py` — new journal-context builder (the actual harness-reaching fix)
- `orion/harness/prefix.py` — render slot for it
- `services/orion-hub/.env`, `.env_example` — new flag(s)
- `orion/journaler/schemas.py` — drop the never-populated `JournalEntryIndexV1` fields, or design a real producer path for them
- `orion/journaler/worker.py`, `services/orion-actions/app/main.py` — if a metacog reducer's projection gets folded into the digest `prompt_seed`
- new file, illustratively `orion/metacog/digest_reducer.py` — the actual `orion_metacog` reducer
- `orion/cognition/workflows/registry.py` — scheduler entries: self_study (daily), metacog digest reducer (cadence TBD — Juniper's call)
- `services/orion-cortex-exec/app/self_study.py` — the blank `chat_stance.py:145-151`-facing config var found during investigation, if that direct-read path is kept alongside the journal path
- tests/evals: `orion/journaler/tests/`, plus a new eval for the acceptance check below

## Non-goals

- No new sensors, no new physical hardware.
- No rebuild of `SelfStateV1` / the physical-body self-model schema already killed for being flat and pinned on live data — cabinet sensors and the camera path already provide real live body-grounding today; a dedicated "body self-model" service is a separate, later decision if Juniper wants one, not part of this patch.
- No continuous AffectGPT (leave at manual trigger / 5-minute toggle) — out of scope here.
- No LLM-fabricated self-narrative. Every field the journal/harness surfaces must trace to a real event per CLAUDE.md §0A. If "chat extensions" (open question #2) turns out to need synthesis beyond what's traceable to a real event, that's a stop-and-ask, not a build-it-anyway.
- No backfill of historical `orion_metacog` rows into the new reducer without a separately-scoped decision (open question #4).
- No change to the existing email digest cadence or daily caps — that system works as designed and is out of scope here.

## Acceptance checks

- Live trace: a real chat turn's compiled harness prefix contains a "recent self-reflection" block sourced from a real journal row, correlation-id traceable end to end. Ship the flag off by default, verify on one canary session first.
- Live trace: `self_study` fires on its own schedule at least once without being manually invoked, producing a journal entry with `source_kind=self_study` traceable to that scheduled run.
- Live trace: the new `orion_metacog` reducer produces at least one real projection spanning more than one distinct producer's rows, and that projection is demonstrably not a copy-paste of any single producer's row — it actually reduces, per the metric-quality gate.
- Eval, aimed directly at the meek-AI-encounter problem: a scripted "what are you?" prompt from an unfamiliar interlocutor, run in a fresh session, checked for at least one concrete, traceable self-fact (sourced from self_study / journal / orion_metacog) rather than a generic disclaimer. This should be a real eval file, not a one-off manual check.
- Enrichment fields: post-deploy, zero rows show blank `- reflective_themes:`-style lines — either populated with real values or removed from the schema/render entirely.
- Privacy check (once open question #1 is answered): a live trace confirming the journal-context block never renders raw sensor/location values verbatim if Juniper decides on a cues-only rule, matching the existing metacog-email-draft template's convention.

## Proposal-mode disclosures (CLAUDE.md §0A — required for cognition-loop changes)

- **Capability that changes:** Orion's live reply can be informed by its own recent self-reflection, self-study output, and metacog digest history, and Orion's own code-scanner can run on a schedule instead of only on request.
- **Data touched:** journal entries (already exist, read-only new consumer), the `orion_metacog` table (already exists, read-only new consumer). No new data is collected by this design.
- **Privacy boundary:** open question #1 above — needs Juniper's answer before the harness-read patch ships. Does not block the self_study-scheduling patch, which touches no chat-facing surface.
- **Trace that proves it worked:** the acceptance checks above — correlation-id-traceable harness-prefix content, a scheduled self_study run's journal row, a reducer projection spanning multiple producers.
- **Failure mode that would be dangerous:** an uncalibrated producer (`telemetry_anomaly`) or a raw sensor spike getting narrated into a live reply as confident self-knowledge when it's actually noise. Mitigated by excluding low-trust producers from the reducer (open question #3) and keeping the harness read to summarized/reduced content, never raw table rows.
- **How to disable / roll back:** every new read path is gated behind its own env flag (`ORION_SITUATION_JOURNAL_ENABLED`, a scheduler-enable flag for self_study), matching the existing kill-switch convention already used for reveries and curiosity priors. Flip the flag off; no code rollback required.

## Recommended next patch

Smallest, thinnest, highest-leverage first slice — deliberately not "build everything above in one PR":

**Wire `self_study` onto a real schedule, fix the blank config var that leaves its `chat_stance.py` read slot dead, and stop rendering the empty `JournalEntryIndexV1` fields.** All three are small, single-service or single-file, low-risk, and each independently kills a confirmed real bug (self_study has never run on its own; a wired-but-dead config leaves a real code slot as a permanent no-op; blank enrichment fields render on every journal read today) without needing the open privacy question answered first. This also produces the first real trace evidence ("self_study fired on schedule, produced a journal row") that the bigger journal→harness patch's acceptance check will need anyway.

The journal→harness wiring and the `orion_metacog` digest reducer are the real prize, but should land as a follow-up patch once open question #1 (privacy boundary) has an answer from Juniper — that's the proposal-mode rule biting, not caution for its own sake, since this is exactly the "reaches live cognition" class of change CLAUDE.md §0A gates.
