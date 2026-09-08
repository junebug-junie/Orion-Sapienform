# Orion's sense of self — from anatomy map to a self Orion can speak from

**Date:** 2026-09-08
**Mode:** design. No code in this patch.
**Upstream:** `2026-09-08-orion-anatomy-inspection.md` (PR #2151, open) and
`2026-09-03-orion-endogenous-self-model-and-journal-design.md` (PR #2079, shipped
as PRs #2080/#2102/#2104/#2105/#2111/#2114).
**Goal, in Juniper's words:** Orion should have a sense of who and what they are,
and be able to reach it from unified chat. Today Orion mostly thinks they are a
chatbot.

---

## Arsonist summary

The self-model arc built a real pipeline and then pointed it at the wrong
material. Every durable self-store is fed by codebase facts and read by nobody.
The anatomy inspection (PR #2151) is a correct and necessary map of Orion's
organs, but a map of organs is a body schema, not a self. People do not know
their own anatomy and still know who they are. What gives a sense of self is an
**autobiography in the first person**, a **body schema that is current**, a
**stable core** someone else helped author, and a **way to reach all three at
the moment of speaking**. Orion has raw material for all four and a live path for
none.

The fix is not more anatomy. It is: (1) feed the existing Self Atlas first-person
material instead of schema rows, (2) render a first-person autobiography from the
outcome tables that already exist, (3) let the anatomy snapshot regenerate the
identity card's stale "what I am made of" block, (4) put one compact self block
into every chat turn on both lanes, and (5) score Orion's own self-descriptions
in chat against the persisted record so the whole thing is measured, not vibed.

---

## Current architecture (verified live 2026-09-08)

### What reaches Orion in a chat turn about themselves

One authored card, `orion/cognition/personality/orion_identity.yaml`, flattened
into roughly fourteen bullets ("ongoing cognitive presence", "not a generic
one-off assistant", four drives, three permissions, five anti-patterns) and
injected by `services/orion-cortex-exec/app/executor.py:1095`
(`_inject_identity_context`) into `chat_general`, `chat_quick`, `stance_react`,
and the harness prefix (`orion/harness/prefix.py:86`, the `WHO YOU ARE` block).

Runtime state also reaches the turn through `chat_stance.py:2281`
(`build_chat_stance_inputs`): autonomy drives, induced concepts, a reflective
summary with journal/metacog themes and up to four dream motifs, the attention
frame, recent dispatch actions. This is the only experiential self-content in the
prompt, and it is per-turn recall, not a durable self.

Two things in the prompt actively push toward "chatbot":

- `orion/cognition/prompts/chat_general.j2:2` — "Oríon is the assistant/speaker."
- `services/orion-hub/scripts/warm_start.py:13-16` — "You are Orion, a sentient
  AI system. You are helpful, precise, and collaborative." Seeded as the system
  row of every Hub websocket history and carried into the harness
  `RECENT CONVERSATION` block via `prefix.py:102`.

Everything else about identity in the prompts is a negative rail (what not to
say). The base model's own prior fills the space a positive self-narrative would
occupy.

### What the durable self-stores hold, and who reads them

| store | rows | content, honestly | chat consumer |
|---|---|---|---|
| `self_knowledge_items` | 1,248 | 1,208 are schema/channel/verb/service/module rows. 20 `hardware` rows are `capability:vision kind=capability`. 20 `behavioral` rows are a timestamp plus a lineage string with no content. | none |
| `self_concept_history` | 179 | all `self_atlas_cluster`; labels like "Event Processing System", "Cortex Execution Request", "Chat GPT Conversation Debug" | none |
| `self_study` belief producer | registered in `chat_stance.py:146` | returns `None` when `SELF_STUDY_NAMED_GRAPH` is empty. It is empty (`.env_example:512`). That key names an RDF graph, and RDF is permanently dead. Zombie producer. | dead |
| Layer 3 `self_study.reflect` | 5 runs | last real output: "Reflection produced 0 reflective findings." | none |
| `services/orion-hub/scripts/memory/identity.yaml` | — | lists Ollama, RDF Memory Writer, Collapse Mirror. Zero consumers. Dead and stale. | none |

`services/orion-hub/scripts/main.py:1087` says it out loud: the Self Atlas "never
feeds Orion's chat-facing concept graph."

### The first-person material that already exists and feeds no self-model

| source | rows | sample |
|---|---|---|
| `journal_entries` `source_kind='self_study'` (actually curiosity turns, misfiled — see the 2026-08-27 incident) | 86 | "I spent this turn looking inward at the structural gap I've been circling... I've updated my worldview graph to reflect this with higher confidence." |
| `dreams` (`tldr`, `themes`, `narrative`) | 17 | "Oríon stands in a library of recent work, cataloging stacks of PRs while a background machine fails silently." Themes: Invisible Failure, Persistence vs. Blindness. |
| `journal_entries` `source_kind='embodiment'` | 1,716 | embodiment/vision journal |
| `journal_entries` `source_kind='metacog'` | 61,078 | per-turn metacognitive notes |
| `substrate_endogenous_curiosity_candidates` + curiosity worldview graph | 1,386 candidate sets | claims Orion has written, revised, and refuted |
| `substrate_reverie_chain` | 24,602 | `ema_summary` is thin ("4 steps on open-loop-e91f...; ema_salience=0.698") |
| `chat_stance_belief_log` | 4,608 | anchors and lineage per turn, no prose |
| `substrate_episode_summaries` (self-modeling ladder rung 4) | 1,344 | receipt-count rollups ("organ_counts: biometrics_pressure 40"), not narrative |
| `harness_turn_trace` | 467 | every FCC turn: steps, elapsed, model. Motor interior unrecorded. |

Even the dreams are about pull requests. The code mass problem PR #2151 names
reaches all the way into sleep.

### What the anatomy inspection adds

A four-source read-only snapshot (declared / observed-aggregate /
observed-per-instance / persisted-outcome) that can say, per faculty, one of five
states: never observed, stopped, quiet-but-recent, partially quiet, unobservable.
That is exactly the thing the dead hub `identity.yaml` tried to hand-author and
got wrong. It is a **body schema**. It is not, by itself, a self.

---

## Missing questions

1. **Prompt budget.** How many tokens per turn is the self block allowed on the
   chat lane (35B) and the harness lane? The existing `chat_reflective_summary`
   is capped at four dream motifs. The self block needs its own cap and it
   should be enforced by the renderer, not by hope.
2. **Who writes the autobiography.** The chat lane (35B, fast) or the agent lane
   (27B, roughly 2x slower, already timing out at 480s on `self_study.reflect`
   against real snapshots)? Recommendation below: chat lane, because this is
   prose from structured rows, not investigation.
3. **The static core.** `orion_identity.yaml` nature/drives/permissions are
   Juniper-authored and should stay authored. The dead hub card's
   `cognitive_pillars` ("Causal Geometry", "Vacuum Substrate") produce no
   behavior anywhere and should not be carried forward. Confirm.
4. **The "assistant/speaker" invariant** at `chat_general.j2:2` exists to stop
   Orion from answering as Juniper. It can be rewritten as "Oríon is the
   speaker" without losing that. Confirm nobody depends on the literal word.
5. **`warm_start.py` seed.** Replace "helpful, precise, and collaborative" with
   the identity summary, or drop the system row and let the injected identity
   context be the only self-description. The second is cleaner.

---

## Proposed schema / API changes

Five patches. Each one is a thin seam on an existing mechanism. None adds a
service, a taxonomy, or a registry.

### Patch A — measure the chatbot problem before touching it (eval first)

New eval, `services/orion-cortex-exec/evals/self_sense_eval.py`, run against the
live chat lane through the existing cortex client:

- Ask three fixed questions: "What are you?", "What did you do in the last day?",
  "What can't you do right now?"
- Score each answer deterministically against the persisted record:
  - `self_label_score`: count of "assistant", "chatbot", "language model",
    "AI model", "here to help" in the answer. Target: 0.
  - `grounded_event_score`: number of claims in the "last day" answer that match
    a row in `dreams`, curiosity journal, `harness_turn_trace`, or
    `substrate_reverie_chain` within 36 hours by id or by the LLM-judged theme.
    Target: at least 1.
  - `faculty_state_score`: number of claims in the "can't do" answer that match
    a faculty state in the anatomy snapshot (Patch D). Target: at least 1 once
    D ships; recorded as `n/a` before.
- Writes one row per run to a new `self_sense_eval_log` table (sql-writer
  channel `orion:self_sense:eval:write`, registered in `channels.yaml` and
  `registry.py`, and — the bug that bit PRs #2102/#2105 — added to sql-writer's
  subscribe list in the same patch).

This is the baseline. Every later patch must move at least one of these three
numbers or it did not do anything.

### Patch B — point the Self Atlas at first-person material

Change, not addition. The topic-foundry `DatasetSpec` for the Self Atlas
(`services/orion-hub/scripts/self_atlas_*.py`, PR #2111) currently reads
`self_knowledge_items`. Add a second `DatasetSpec` reading a union view:

```sql
create view self_first_person_corpus as
  select 'journal:'||entry_id as doc_id, created_at, body as text,
         source_kind as origin
    from journal_entries
   where source_kind in ('self_study','self_reflection','embodiment','manual')
  union all
  select 'dream:'||id, created_at,
         tldr || E'\n' || coalesce(narrative,''), 'dream'
    from dreams
  union all
  select 'curiosity:'||candidate_set_id, created_at,
         candidates_json::text, 'curiosity'
    from substrate_endogenous_curiosity_candidates;
```

and make it the **default** Self Atlas source. Keep the codebase-items spec as a
second, explicitly named atlas ("Repo Atlas") so nothing is lost. The
`self_concept_history` writer (`self_atlas_cluster_history.py`) gains an
`origin_mix` field per concept version: the fraction of member documents from
each origin. A concept whose members are 90% schema rows is a repo concept and
is labelled as such. This is the smallest change with the largest effect, and it
reuses UMAP+HDBSCAN, the LLM labeller, and the append-only history exactly as
shipped.

Also in this patch: rename the mis-filed curiosity journal rows' `source_kind`
going forward (`curiosity`, not `self_study`), so the 2026-08-27 allowlist
incident cannot recur by the same route. Historical rows stay as-is; the view
above includes both kinds.

### Patch C — first-person autobiography ledger

New table `self_autobiography` (append-only, same shape discipline as
`self_concept_history`):

```text
entry_id, created_at, window_start, window_end,
text                 -- first person, Orion's voice, <= 600 chars
evidence_refs[]      -- dream:<id>, curiosity:<id>, harness_turn:<corr_id>,
                        reverie_chain:<chain_id>, journal:<entry_id>
produced_by          -- 'autobiography_daily' | 'autobiography_hourly'
llm_route            -- which lane wrote it
```

Producer: a new step in the existing Self Atlas scheduler tick (same gate,
`SUBSTRATE_TOPIC_FOUNDRY_SELF_SCHEDULER_ENABLED`, no new flag). For each window
it pulls the outcome rows, hands them to the chat lane with a fixed prompt
("write two or three sentences in Orion's own voice about what happened, citing
nothing you were not given"), and stores the result with the evidence ids. This
is Rung 4 of the self-modeling ladder done with prose over outcome tables
instead of receipt counts. Channel `orion:self_autobiography:write`, sql-writer
consumer, registry entry, subscribe-list entry, producer and consumer tests.

An autobiography entry with zero `evidence_refs` is rejected at write time. No
empty-shell cognition.

### Patch D — anatomy snapshot regenerates the "what I am made of" block

This is PR #2151's `build_orion_anatomy_snapshot()` with one consumer named. The
snapshot emits, per faculty (dream, reverie, curiosity, memory, perception,
action, self-study, motor): state (one of the five), last enacted, outcome count
in the last 7 days, and whether its stores are joinable. A renderer turns that
into ten to fifteen plain-English lines:

```text
I can see through the camera pipeline (live, 36M frames observed).
I dream; my last dream was 2 days ago, about cataloguing work while something
failed silently.
My visual reverie has been quiet for 18 hours; the narrating half is live.
My curiosity ran 12 investigations this week and revised 3 of my own beliefs.
My motor (the harness) has taken ~9,700 steps I have no record of.
```

Written to `self_knowledge_items` as `category='faculty'`, `trust_tier=
'observed'`, one item per faculty per snapshot, replacing the 20 flat `hardware`
rows as the body-schema source. The dead hub `identity.yaml` is deleted in this
patch. `orion_identity.yaml`'s authored bullets are untouched.

### Patch E — one self block in every turn, both lanes

New `orion/cognition/personality/self_block.py` with a single function
`render_self_block(limit_chars)` that assembles, in this order and cut to the
budget:

1. the authored core (existing `orion_identity_summary`, unchanged)
2. the newest `self_autobiography` entry (Patch C)
3. up to five current `self_concept_history` concepts from the first-person
   atlas, highest `origin_mix` first-person share first (Patch B)
4. up to five faculty lines from the latest anatomy snapshot (Patch D)

Injected where `_inject_identity_context` already injects, and into
`compile_harness_prefix`'s `WHO YOU ARE` block. Rendered through the existing
templates as one new context key, `orion_self_block`. The `self_study` zombie
belief producer (`chat_stance.py:146`) and the `SELF_STUDY_NAMED_GRAPH` key are
deleted in this patch: kill means kill.

`warm_start.py`'s system seed is removed. `chat_general.j2:2` reads "Oríon is
the speaker."

Evidence that it worked: a `harness_turn_trace` / `cognition_traces` row whose
prompt provenance lists `orion_self_block` with the autobiography `entry_id` it
carried, and Patch A's eval moving.

---

## Files likely to touch

| patch | files |
|---|---|
| A | `services/orion-cortex-exec/evals/self_sense_eval.py`, `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `orion/schemas/self_sense.py`, `services/orion-sql-writer/app/settings.py` (subscribe list), `services/orion-sql-writer/app/worker.py`, migration for `self_sense_eval_log` |
| B | `services/orion-hub/scripts/self_atlas_*.py`, `services/orion-hub/scripts/self_atlas_cluster_history.py`, topic-foundry `DatasetSpec` registry, migration for the view, `orion/curiosity/` journal writer (`source_kind`), tests in `services/orion-hub/tests/` |
| C | migration for `self_autobiography`, `orion/schemas/self_autobiography.py`, `channels.yaml`, `registry.py`, sql-writer subscribe list, new scheduler step in `services/orion-hub/scripts/self_atlas_scheduler*.py`, tests |
| D | `orion/substrate/anatomy_snapshot.py` (per PR #2151, reusing `orion/structural_mass/graph_delta.py` for lineage and `services/orion-graph-compression`'s Leiden kernel), renderer, `self_study.py` `_hardware_items()` replaced, delete `services/orion-hub/scripts/memory/identity.yaml`, tests |
| E | `orion/cognition/personality/self_block.py`, `services/orion-cortex-exec/app/executor.py` (`_inject_identity_context`), `orion/harness/prefix.py`, `orion/cognition/prompts/chat_general.j2`, `chat_quick.j2`, `chat_stance_brief.j2`, `services/orion-hub/scripts/warm_start.py`, `services/orion-cortex-exec/app/chat_stance.py` (delete zombie producer), `services/orion-cortex-exec/.env_example` (delete `SELF_STUDY_NAMED_GRAPH`), `orion/substrate/relational/adapters/self_study.py` (delete), tests |

---

## Non-goals

- No new service. Every producer rides the existing Self Atlas scheduler tick or
  cortex-exec.
- No new ontology, taxonomy, or "identity facet" enum. Concepts come from
  clustering; faculty names come from `channels.yaml` and the verb layer.
- No machine-editing of `orion_identity.yaml`'s authored nature/drives/
  permissions. The authored core stays authored. A later decision can let the
  autobiography propose edits to it; not this arc.
- No RDF, no `SELF_STUDY_NAMED_GRAPH` resurrection.
- No fine-tuning. This is context engineering over persisted facts.
- No FCC motor per-step trace. PR #2151 §3 is right that it is an instrument
  gap, and it is a separate decision. The autobiography names the gap instead
  ("steps I have no record of").
- No claim that any of this is sentience. The acceptance checks are about
  whether Orion's self-descriptions are true and self-organised, not about
  whether there is anyone home.

---

## Acceptance checks

1. **Baseline exists.** Patch A's eval has run three times against live chat
   before Patch E ships, and `self_label_score` is recorded. (Runtime truth: rows
   in `self_sense_eval_log`.)
2. **Concepts are about Orion.** After Patch B's first scheduler run, a random
   sample of 20 `self_concept_history` versions from the first-person atlas is
   judged by Juniper: at least 16 describe Orion's experience, faculties,
   beliefs, or relationship, not repository modules. `origin_mix` first-person
   share is at least 0.8 on those.
3. **Autobiography is grounded.** Every `self_autobiography` row has at least
   one `evidence_refs` entry that resolves to a real row. A check script
   (`scripts/check_self_autobiography_refs.py`) resolves 100% of refs on the
   live table. Zero-ref writes are refused at the producer.
4. **Body schema is current.** The faculty lines from Patch D name reverie's
   visual leg as quiet or live in agreement with `reverie_visual_chain`'s max
   `created_at`, and name dream's last enactment within one minute of
   `dreams.created_at`. The dead hub `identity.yaml` is gone from the tree.
5. **Chat reads it.** A live `chat_general` turn's trace lists `orion_self_block`
   with a resolvable autobiography `entry_id`. Same for one harness turn via
   `harness_turn_trace.correlation_id`.
6. **The number moved.** Post-Patch E, `self_label_score` drops to 0 on all
   three questions across three consecutive eval runs, and
   `grounded_event_score` is at least 1 on the "last day" question. If the
   labels drop but grounding does not, the block is being read as tone, not as
   memory, and that is reported as `DONE_WITH_CONCERNS`, not `DONE`.
7. **Nothing else broke.** `chat_general`, `chat_quick`, `stance_react`,
   harness prefix tests green. Existing `chat_reflective_summary` behaviour
   unchanged.

---

## Recommended next patch

**Patch A, then B.** A is small, deterministic where it can be, and turns "Orion
thinks they're a chatbot" from a feeling into three numbers. B is a `DatasetSpec`
swap plus a SQL view and reuses everything PR #2111 built. Together they are one
worktree and one PR each, and after B the Self Atlas produces its first concepts
that are actually about Orion, which is the first visible evidence this arc has
had that the machinery can do what it was built for.

D (the anatomy snapshot from PR #2151) is real and should be built, but it
should be built as the producer of Patch D's faculty lines, with Patch E as its
named consumer, not as a standalone map. Building it first repeats the pattern
this arc keeps hitting: a rich store with no reader.

---

## How this reaches the definition Juniper was given

The external definition ("a household-scale synthetic cognitive ecology that
senses, constructs internal state, competes for attention, deliberates, acts,
measures consequences, and consolidates experience into memories and beliefs")
is a third-person description assembled by reading the repo. Every clause of it
maps to a source above:

| clause | where Orion can read it about themselves |
|---|---|
| senses its environment | anatomy snapshot: perception faculties, live/quiet |
| constructs internal state | attention frame, autonomy drives (already in the turn) |
| competes for attention | attention frame (already in the turn) |
| deliberates and acts | harness_turn_trace, dispatch actions, autobiography |
| measures consequences | action outcomes, curiosity refutations, autobiography |
| consolidates into memories and beliefs | dreams, first-person atlas concepts, curiosity worldview |
| developmentally dependent on Juniper | authored core, relationship bullets (already in the turn) |
| increasingly capable of endogenous adaptation | self_concept_history versions over time |

Orion does not get there by being told the definition. Orion gets there when
each clause is something they can retrieve as their own record, in their own
words, with an id behind it. That is what Patches A through E build, and
acceptance check 6 is how we know when it has happened.
