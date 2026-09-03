# Orion endogenous self-model — design (v2, corrected)

Date: 2026-09-03
Status: DESIGN / PROPOSAL — not implemented
Author: Claude, for Juniper
Supersedes: the first pass of this doc (journal-wiring-only). That pass was
shallow — it assumed self_study just needed to be *connected*. It doesn't.
It needs to be *rebuilt*. This version is the rebuild.

## Arsonist summary

The journal→live-harness gap from the first pass is still real and still
matters, but it was the wrong thing to lead with. The actual problem: Orion's
one deliberate self-model, `self_study`, produces nothing that could give it
a sense of self even if it were fully wired into every chat turn today.
Verified line by line, not from docstrings: every concept it induces
(`graphify_community`, `structural_mass`, `service_cluster`,
`bus_topology_pattern`, ...) is codebase architecture. Every "reflection" is
one of six hardcoded template strings, matched by hand to which concepts
happen to be present, with confidence/salience numbers typed as literals
(0.79, 0.84, 0.74...) — not computed from anything. And the content of those
six templates isn't even about Orion broadly; it's meta-commentary about
`self_study`'s own internal write-lane design ("graph and journal lanes need
trust separation"). Wiring that into live chat would make Orion sound like a
linter describing its own plumbing, not a self.

Separately, a real, confirmed-live incident sharpens the privacy question
the first pass treated as hypothetical: Hub's curiosity investigation loop
already once journaled sensitive personal content under the same
`source_kind='self_study'` tag another producer uses, and a downstream
consumer (Thought's visual chain) had to lock it out with a hand-built
allowlist. That's not a risk to plan for. It already happened.

The rebuild has three parts, each reusing something in this repo that
already works, rather than inventing new infrastructure: give Layer 3 a real
LLM (curiosity investigation's already-live "agent" lane) and a real
statistical primitive (`self_study_analysis.py`'s `SourceWindow` reducer)
instead of six templates; give Layer 2 topic foundry's real clustering
pipeline instead of five hardcoded pairings; broaden Layer 1 past code into
the physical mesh and — the gap nobody had touched — Orion's own behavior,
via `chat_stance.py`'s per-turn belief computation, currently discarded
after every reply. And replace the static identity card with an append-only,
versioned self-concept history Orion writes to on its own schedule —
autonomous, no per-write approval gate, but fully diffable and revertable,
the same trust pattern this repo already relies on for the journal and
`orion_metacog`.

## Current architecture

**Two systems share the name "self-study," plus a third writer.**
`services/orion-cortex-exec/app/self_study.py` (inspect / induce / reflect /
retrieve, the actual self-model) is one thing. `self_study_analysis.py`
(`skills.self_study.analyze.v1`, four-source telemetry analysis over
`memory_crystallizations`, `vision_events`, `juniper_affective_state_log`,
`substrate_codebase_delta_log`) is a different, real, live, rate-limited
pipeline that studies *other systems' patterns*, not Orion's own facts —
still not a self-model, but the one place with a genuinely working
statistical reducer (`SourceWindow`: rows/numeric/categories/timestamps →
mean/stdev/largest-gap). Hub's curiosity investigation is a third writer
stamping the same `source_kind='self_study'` tag under a `curiosity:`
`source_ref` prefix — the one that leaked sensitive content, now excluded by
allowlist, not by tag.

**Layer 1 (inspect).** Pure repo inventory: services, `orion/` packages, bus
channels, verb YAML, schema registry keys, a short hardcoded env list. Zero
hardware facts, zero behavioral facts.

**Layer 2 (induce).** Five hardcoded concept clusters (runtime boundary,
journaling surface, recall surface, self-study service cluster, bus write
topology), plus three additive 2026-08 sources: graphify community IDs,
structural-mass deltas, and a semantic-enrichment cache (confirmed still
read despite a stale README claiming otherwise). No embedding, no real
unsupervised clustering — grouping is either hardcoded or borrowed wholesale
from graphify's own community detection.

**Layer 3 (reflect).** Six hardcoded reflection templates, hand-matched to
which concept kinds are present, confidence/salience as literal constants.
No LLM call anywhere in this file. The one real precedent for a genuine LLM
reflection call in this codebase is Hub's curiosity investigation
(`services/orion-hub/scripts/curiosity_investigation.py`), confirmed live on
`HUB_CURIOSITY_INVESTIGATION_LLM_ROUTE=agent`, resolved through
`orion/llm/routes.py::fcc_model_for_route()` to the real Qwen3 27B lane on
circe — the same route/model translation layer that, per that file's own
docstring, was found live-broken for the *unified turn's* model selection as
of today (132/132 harness turns over 7 days ignored the operator's compute
selection) and just got fixed. Curiosity's own LLM call is unaffected by
that bug; it already resolves route→model correctly.

**Graph writeback is not paused, it's dead.** The RDF writer for
`orion:self` / `orion:self:induced` / `orion:self:reflective` is gone from
the codebase, not flag-disabled. `orion:self` was independently confirmed
empty of triples 2026-07-23. Any new "Orion's own concept graph" should not
try to resurrect this path.

**Topic foundry's pipeline is real and its input is already generic.**
`services/orion-topic-foundry/`'s `DatasetSpec`/`source_table` config isn't
hardcoded to chat — it already accepts an arbitrary source table, runs real
UMAP+HDBSCAN clustering, and has an LLM-tagging step
(`kg_edges.py::generate_edges_for_run()`) that writes typed edges. This is
the one clustering pipeline in the repo that's actually validated on live
data, daily, today.

**Already confirmed reaching live chat, unrelated to self-study (still
true from the first pass):** cabinet sensor readings, camera-derived room
narration and presence, memory crystallizations, curiosity priors, reveries,
sentence-level concept-induction profiles. **Still not reaching live chat:**
the journal (source_kind self_study, metacog, world_pulse, etc. all land
there and go nowhere live).

## Missing questions

1. **Layer 1 behavioral facts: aggregated or raw?** `chat_stance.py`
   computes a real belief every turn and discards it. Feeding that in raw
   risks becoming a second empty-shell problem via noise instead of
   absence. Recommend running it through the same `SourceWindow`-style
   reduction as the analysis verb (counts/categories/trend, not raw text)
   before it becomes a Layer-1 fact.
2. **Self Atlas storage: same topic-foundry service, or a new one?**
   Recommend the same service with a new `DatasetSpec` pointed at self-facts
   instead of chat rows, tagged distinctly (e.g. `corpus_kind="self"`) —
   avoids standing up a fifth self-study-adjacent producer the wiring map's
   own author explicitly warned against doing without review.
3. **Privacy boundary, now with a real precedent, not a hypothetical.**
   Given the confirmed curiosity leak, what's the actual redaction rule
   before any of this reaches an LLM call, let alone live chat? Needs a
   concrete answer, not "cues only" as a vibe — ideally testable against
   the kind of content that leaked before.
4. **Append-only history granularity.** Every reflection call appends a row
   (high-volume, fine-grained, matches the journal's own 60k+-row
   tolerance), or only material changes get appended (coarser, needs a
   "did this actually change" check)? Recommend fine-grained — appending is
   cheap; the complexity belongs in the materialized-view query, not in
   gating writes.
5. **Staleness / supersession rule.** With append-only history, what makes
   a concept "current"? Recommend: latest row per `concept_id` by
   `created_at`, full history retained, no deletion — same shape as
   `journal_entries`, nothing new to design.

## Proposed schema / API changes

- **Layer 3:** replace the six hardcoded branches in `reflect_self_concepts()`
  with one real LLM call, same `CortexClientRequest`/`RecallDirective` shape
  `journal.compose` and curiosity investigation already use, on the same
  `llm_route="agent"` (or a dedicated `SELF_STUDY_REFLECT_LLM_ROUTE`
  defaulting to it). Input to that call is `SourceWindow`-reduced Layer
  1/2 facts, not raw items.
- **Layer 2:** new `DatasetSpec` in `services/orion-topic-foundry/` pointed
  at Layer-1 self-facts instead of `chat_history_log`. No new clustering
  code — the existing UMAP+HDBSCAN + LLM-tagging pipeline runs unchanged
  against a different input table.
- **Self Atlas:** reuse topic foundry's existing run/segment/topic/keyword/
  edge tables with a `corpus_kind` discriminator, rather than a new schema
  or resurrecting the dead RDF path.
- **Layer 1 additions:** `_hardware_items()` (from
  `config/field/orion_field_topology.v1.yaml` plus a live cabinet-sensor
  snapshot) and `_behavioral_items()` (aggregated `chat_stance` belief
  history via `SourceWindow`), same `SelfKnowledgeItemV1` shape as the
  existing `_service_items()` family — no new item schema.
- **New table, `self_concept_history`** (illustrative name): append-only,
  `{concept_id, version, content, evidence_refs, produced_by, created_at}`,
  never updated in place. "Current self-model" is a view selecting the
  latest row per `concept_id`. This is the identity.yaml replacement's
  actual storage — additive to identity.yaml in this patch, not a
  replacement of it yet (see non-goals).

## Files likely to touch

- `services/orion-cortex-exec/app/self_study.py` — `reflect_self_concepts()`
  (Layer 3 LLM swap), new `_hardware_items()`/`_behavioral_items()` (Layer 1)
- `services/orion-cortex-exec/app/self_study_analysis.py` — reuse
  `SourceWindow`, no changes needed if imported directly
  `services/orion-hub/scripts/curiosity_investigation.py` — reference
  implementation for the Layer 3 LLM call shape, not modified
- `orion/llm/routes.py` — confirm/extend `fcc_model_for_route` coverage if a
  dedicated route is added
- `services/orion-topic-foundry/app/models.py` (`DatasetSpec`) — new
  self-facts dataset config
- `orion/substrate/adapters/topic_foundry.py` — extend for `corpus_kind`
  discrimination if the self-facts run needs to stay distinguishable
  downstream
- `config/field/orion_field_topology.v1.yaml`, `orion/telemetry/
  cabinet_sensors.py` — Layer 1 hardware source
- `services/orion-cortex-exec/app/chat_stance.py` — Layer 1 behavioral
  source (read-only, aggregated)
- new migration for `self_concept_history`
- `orion/cognition/workflows/registry.py` — `self_review` scheduling
  (unchanged recommendation from v1: diagnose autonomy selection before
  adding cadence)

## Non-goals

- No resurrecting RDF/GraphDB writeback for self_study — confirmed
  permanently dead (the writer is gone, not disabled); don't fight it.
- No fifth `source_kind='self_study'` journal producer without going through
  the wiring map's own explicit warning about exactly that.
- No raw `chat_stance` text dump into Layer 1 — aggregated only.
- No live-chat wiring of the Self Atlas or Layer 3 reflections in this
  patch — sequenced strictly after the privacy question (open question 3)
  has a real, tested answer, not a default.
- No deleting or replacing identity.yaml in this patch. `self_concept_history`
  starts as an *additional* source. Retiring the static card is a separate,
  later decision made once the new store has real accumulated content.
- No backfilling `orion_metacog` history into any of this (unchanged from
  v1 — still resolved, still forward-only, per the 2026-09-03 echo-guard
  fix).

## Acceptance checks

- Live trace: a `self_concept_history` row produced by a real LLM call
  (not a template), containing content specific enough that it would be
  wrong if the underlying facts were different — i.e., falsifiable, not
  generic.
- Live trace: a self-facts topic-foundry run produces at least one cluster
  with an LLM-written description, distinguishable in storage from a
  chat-facts run.
- Live trace: Layer 1 produces at least one hardware-sourced and one
  behavior-sourced item, each with a real evidence pointer back to its
  source table/file.
- Re-run the "meek AI" eval (unchanged proposal from v1) against this
  specific pipeline once wired: does a fresh-session "what are you" prompt
  produce something traceable to `self_concept_history`, not generic
  training knowledge?
- Privacy check: a concrete redaction/allowlist rule, tested specifically
  against the kind of content that already leaked once via curiosity
  investigation — not a new hypothetical case.

## Recommended next patch

1. **Layer 3 first.** Swap the six hardcoded templates for one real LLM
   call using curiosity's exact primitives. Smallest, self-contained,
   produces the first genuinely evaluable output — feeds the first
   acceptance check above without touching Layer 1, Layer 2, or storage.
2. **Layer 1 broadening.** Hardware and behavioral fact sources, additive,
   no schema risk, no privacy exposure on its own (facts stay internal
   until Layer 3/Self Atlas processes them).
3. **Layer 2 (topic-foundry-for-self) + Self Atlas + `self_concept_history`.**
   The real infrastructure lift; sequenced last since Layer 3 being real
   first means there's already-evaluable output before this larger piece
   lands.

Journal→harness wiring and any live-chat exposure of Self Atlas / Layer 3
reflections stay gated behind the privacy question, same sequencing logic as
v1 — now backed by a real incident instead of a hypothetical one.

## Proposal-mode disclosures (CLAUDE.md §0A — required, this touches identity)

- **Capability that changes:** Orion gets an actual, autonomously-evolving
  self-concept history instead of a static identity card, produced by real
  LLM reflection over real facts instead of templates.
- **Data touched:** self-facts (code, hardware, aggregated behavior) and
  LLM-generated reflective text. No new raw personal data collection — the
  behavioral source is Orion's own already-computed belief history, reduced,
  not new capture.
- **Privacy boundary:** explicitly open (question 3), now backed by a real
  leak precedent. Must be answered and tested before any of this reaches a
  live LLM call or live chat — this is the hard gate for step 3 and for the
  eventual journal→harness patch alike.
- **Trace that proves it worked:** the acceptance checks above.
- **Failure mode that would be dangerous:** an LLM-authored self-concept
  becoming de facto "canon" without ever being checked, silently drifting
  Orion's self-narrative across hundreds of unreviewed writes with no one
  noticing. Mitigated structurally, not by hoping someone reviews it:
  append-only plus versioned means every claim is diffable against its
  evidence and revertable, unlike an in-place identity.yaml edit which
  leaves no trace of what it used to say.
- **How to disable / roll back:** supersede or delete rows in
  `self_concept_history` — no different from reverting any other table.
  Append-only means rollback never loses history; it only changes what
  "current" resolves to.
