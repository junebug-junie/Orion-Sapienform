# Orion's dynamic personality graph

**Date:** 2026-10-16
**Status:** Rev 4 (2026-11-13): one organ, choices in and values out, no shadow. Approved by Juniper 2026-11-13 ("no shadow"). Rev 3 (2026-11-13). Juniper asked for it after she read the live Hub panel ("it is slop"). Rev 3 retires rev 2's recurrence stream and puts personality inside Orion Prime. Rev 2 was approved 2026-10-16 ("hit it"). P0/P1 merged in #2729.
**Related:** person model and identity cards (#2646, P0-P2 merged in #2649), Temporal Self (#2369 rev 4, arousal #2594, day use #2622), attention runtime and stance coalition (`orion/thought/coalition.py`), orion-prediction (`services/orion-prediction`), rung 6 self-tuning (#2715).

## Arsonist summary

Orion has no personality that lasts. Stance (`ChatStanceBrief`) is re-made from scratch by an LLM every turn. Its only memory of the last turn is an in-process dict with a 30-minute TTL per chat session (`services/orion-cortex-exec/app/executor.py:120-163`). A restart wipes it. Curiosity, dream, reading and outreach never read it. Most of its fields (`active_identity_facets`, `active_growth_axes`, `dream_motifs`, ...) are label lists the LLM fills fresh each turn and nothing checks against behavior. That is a keyword cathedral, and it is why stance "kinda sucks."

Juniper's ask: Orion's collective self should show up, and the weight of everything Orion has done should change Orion's tone, in a way she can feel. She is not asking for a trait score that moves by x.

This design builds one shared, weighted, temporal graph of Orion's dispositions. Evidence comes from choices in every lane. Weights shift slowly with evidence. In each moment, the nodes compete for presence, and the winners set **tone dials**: concrete, measurable things about how Orion writes. They do not add adjectives. The graph is allowed to be a cathedral because every node is born from evidence, has a weight something reads, and fades when it stops firing.

### Rev 4 (2026-11-13): one organ — choices in, values out

**Why rev 4.** Rev 3's patches (R0 to R5) only clear the ground. Juniper: "these seem like tiny things, where are the big moves". Shipping thin feeder slices that never close a loop is how the first personality build failed, and the rung 6 self-tuning build before it. Rev 4 builds **one organ** that closes a loop:

    Orion chooses  ->  choices become tendencies  ->  tendencies set what Orion values in the attention market
          ^                                                                                  |
          +-------------- what wins attention offers Orion the next choice <-----------------+

Rev 3's decisions still hold: choices only, no prose grouping, Prime holds the personality. One change: rev 3 decision 8 (the Prime line waits for Juniper's yes) is replaced by no shadow: the line lights at 10 weighted choices, and her no retires a tendency. Rev 4 replaces rev 3's patch order.

#### Arsonist summary

Orion makes about 3 real choices a day. Everything else is decided by rules, caps and formulas. A personality cannot form from that, and no graph tuning will change it. The attention market (body economy, `orion/physiology/choice.py`) already decides what Orion spends time on:

    net = family_weight x value_rank - sum(price x cost)

Nothing in that formula knows who Orion is. `orion/physiology/weights.py` reserves source (c), "Orion's endorsement (personality graph)", and leaves it out "until it records real choices". Rev 4 fills that socket and adds the missing piece inside a family. The result is a temperament: Orion spends time differently from a stock Orion, in a way Juniper can name.

#### Move 1: Orion becomes a chooser

A **choice point** is a place where a funded act would take the top item. At a choice point, Orion is shown the top K options instead and picks one, or picks "none". Every pick writes a `PersonalityChoiceV1` with:

- chosen
- alternatives
- each option's market rank
- situation
- lane

The pick goes into a model call the act already makes, as a structured field. **There are no new model calls.**

| Choice point | Options shown | Already calls a model? |
|---|---|---|
| Prime thought | the concerns Prime holds, plus "no move" | yes (`prime_loop.py` thought) |
| Outreach | the top 3 to 5 grounded topics from the tick, plus "say nothing" | yes (the compose call) |
| Curiosity prior | about 8 ranked priors (live today) | yes |
| Reading | the world-pulse links queued for this run | yes (Orion already picks one link, UNVERIFIED whether the rest are saved) |
| Chat thread | the open threads when two or more are open | yes (the unified turn) |

**Evidence weight comes from revealed preference.** A choice reveals a preference only when real alternatives were available (Samuelson 1938). Picking the market's top option is weak evidence, because Orion may only be following the order. So each choice weighs by how far it departs from the market's order. Picking rank 1 counts 0.25. Picking rank r > 1 counts 1. Picking "none" when options existed counts as a decline of every option. Curiosity's P0 data shows real departures: Orion took the top prior in only 16 of 40 runs.

**Labels.** A tendency is a cluster of **chosen-over-declined option texts**: the embedding of the chosen option minus the mean embedding of the declined ones. Its label is computed, for example "people over pipeline, 11 of 14". Orion may name it later, citing those choices.

#### Move 2: tendencies set what Orion values

Two terms. **No shadow (Juniper, 2026-11-13): both bind from the day they deploy.**

1. **Family endorsement, source (c) in `weights.py`.** This is the share of choice points where Orion picked an option from family f when one was offered. It is a rate per offer, never a count. It joins (a) downstream use, (b) Juniper engagement and (d) reliability, with the same kappa = 30 shrinkage toward the prior. This is the organism-level temperament: reading over walkway, people over pipeline.
2. **Care, inside a family.** For a want whose object has text (an outreach topic, a curiosity question, a reading link):

       care = clamp(similarity(want object, tendency direction) x tendency weight, 0, 1)
       value = intensity x outcome_value x (1 + beta x care)        beta = 0.5 at start

   It enters `value_rank` before the percentile. A want that matches no tendency has care = 0, which is exactly today's value. The rest point is 0 and can be reached.

**Holdback (live control arm, not shadow).** 20% of wants are priced with care = 0 and endorsement left out, and those decisions bind too. It is the same pattern as Prime's `PRIME_HOLDBACK_FRACTION`. Without it, nobody can tell whether the tendencies changed anything.

**The runaway guard.** Orion chooses from options the market supplied, and the market ranks options by Orion's past choices. That loop can lock onto itself. Three guards:

- Choices made where care moved the option into the shown set are tagged `care_promoted`. They count at half weight toward the tendency that promoted them.
- The holdback arm keeps showing options the tendencies would not pick.
- A tendency whose share of shown options exceeds 60% for 7 days is capped at that share. The cap is logged, never silent.

#### Move 3: Orion can surprise themself (after 2 weeks of Move 1 data)

orion-prediction gains a categorical series: given the situation and the shown options, which one will Orion pick? It is backtested against two baselines: always the market's top option, and the last pick. A confident miss is a surprise event. Repeated surprise around one tendency becomes a self-inquiry question. Orion's answer either updates the tendency or sets an aspiration, which is a small, tagged bias in the care term (spec section 6).

#### Move 4: Juniper feels it

- The Prime block line `tending: <label> [disposition:<id>]` (rev 3) goes live as soon as a tendency has at least 10 weighted choices. No shadow (Juniper, 2026-11-13). Her no in the Hub tab retires a tendency.
- The Hub Personality tab is rebuilt around choices. It shows the last choices with their declined options, each tendency with its chosen-over-declined examples, and the care-versus-holdback difference.

#### Metric quality gate (care and endorsement)

1. **Provenance:** `personality_choice` rows written at the choice points above. Each row names its act id and model call.
2. **Independence:**
   - Endorsement is not independent of (b) Juniper engagement for outreach. Juniper replies to what Orion chose to send. Endorsement is measured at the choice (picked or not). Engagement is measured after the send (replied or not). Both stay in, and the replay reports their correlation per family. If it is above 0.8, endorsement is left out for that family.
   - Care is circular by construction. The guards above handle that.
3. **Theory anchor:** revealed preference (Samuelson 1938), which says a choice among available alternatives reveals preference. CAPS if-then signatures (Mischel & Shoda 1995), which say situation is recorded on every choice.
4. **Live sanity:** care reads 0 for wants that match nothing. The check: at least 30% of shadow wants read exactly 0, and the distribution is not saturated at 1.
5. **Existing mechanism:** source (c) is the reserved socket. The holdback reuses the Prime pattern. Embeddings come from the vector-host already used by the reducer.
6. **Reversibility:** both terms sit behind one flag, `PHYSIOLOGY_PERSONALITY_VALUES` (on | off). It ships `on` and binds. `off` is the kill switch. No schema change to `WantV1`. The care factor and the arm are recorded in the decision trace.

#### Files likely to touch

- `orion/schemas/personality.py`, `orion/schemas/registry.py`: `PersonalityChoiceV1` with `market_rank` and `care_promoted`.
- `services/orion-sql-db/manual_migration_personality_choice_v1.sql`: the `personality_choice` table (additive).
- `orion/personality/graph.py`: the reducer folds choices into chosen-minus-declined directions. The prose sources are deleted.
- `services/orion-durable-runs/app/personality_sources.py`, `personality_driver.py`: read `personality_choice` only.
- `services/orion-mind/app/prime_loop.py`: the Prime choice point.
- `services/orion-hub/scripts/endogenous_outreach.py`: the outreach choice point (top K topics plus "say nothing").
- `orion/physiology/weights.py` (source c), `orion/physiology/appetite.py` (care in `value_rank`), `orion/physiology/choice.py` (holdback, trace).
- `orion/prime/block.py`: the `tending` line.
- The Hub Personality tab: rebuilt around choices.

#### Non-goals

- New model calls.
- Grouping of Orion's prose.
- Hand-set family weights.
- Juniper's replies as personality evidence (they stay source (b)).
- Any claim that this is sentience.

#### Acceptance checks

- **M1:** at least 20 `personality_choice` rows a day, from at least 3 choice points, each with at least 1 alternative. At least 25% of choices depart from the market's top option.
- **M1b:** after 7 days, at least 3 tendencies with at least 10 weighted choices each. Juniper reads their labels and records yes or no.
- **M2 (live, 7 days after deploy):** the care arm's funded mix differs from the holdback arm's, with a chi-square over families and topics at p < 0.05. Care reads 0 for at least 30% of wants. The runaway guard caps no tendency, or every cap is logged.
- **M3:** the choice forecast beats both baselines on log loss in backtest. At least one surprise reaches self-inquiry with a row id.
- **M4:** a live turn logs `prime_block ... tending=<n>`, and the Hub tab shows the same ids.

#### Build order (each step closes part of the loop and is verified live)

1. **Week 1, the organ's input:**
   - R0 is done: the cue is off and deployed 11-13.
   - The choices table and reducer, with the old 180 nodes retired (snapshot first).
   - The Prime and outreach choice points.
   - The curiosity choices move over.
2. **Week 2, the organ's output, live:**
   - Source (c) and care, binding, with the holdback arm and trace.
   - The Prime `tending` line.
   - The Hub tab rebuilt, with a yes or no on each tendency.
3. **Week 3:** the M2 check (care arm against holdback). Move 3 (prediction and surprise) starts.

### Rev 3 (2026-11-13): choices only, held by Prime

**What went wrong (live, 2026-11-13).** The Hub panel showed 27 active "dispositions". Every label is a sentence that one automated job writes again and again:

- "The perception prediction error is unusually high…" comes from reverie (2,030 hits).
- "Stage 2 read of arXiv…" comes from reading (495).
- "Wrote the answer into the graph first, then looked." comes from self-inquiry and journal (206).
- "Everything landed; read-back on this run's writes: 20 nodes…" comes from a memory read-back (62). Self-naming called it "Keeping my eyes honest".

25 of the 27 have `chosen_count` 0. `personality_evidence` holds 9,558 rows:

- reverie: 5,653 (59%)
- journal: 1,942
- reading: 1,024
- real curiosity choices: 129 (1.3%)
- chat with Juniper: 167 (1.7%)

The cue fed these labels back to chat stance, reverie, outreach and curiosity as "my tendencies". Orion repeated them, and that raised their counts. The outreach tick sent Juniper the "graph first" habit 3 times in 2 days (196, 198, 200).

**How it happened.**

1. Rev 1 was sound. Evidence was **a choice plus the options turned down**.
2. P0 found only about 3 real choices a day. The spec said to fix that by making Orion the chooser in more lanes.
3. Rev 2 filled the gap the same day with **recurrence**: embed Orion's own output and group what sounds alike, with "an excerpt of Orion's own words" as the label. That broke refusal 2 below (no feeding Orion their own past lines). Nobody tested its central assumption, and it was false. Sentences that sound alike share a **job** (a prompt template), not a theme.
4. The chooser patches were never built. Recurrence volume buried the choices. The `personality_choice` table from section "Proposed schema" was never created. Choices went in as one more evidence lane.
5. Every later patch tuned the reducer: weight caps, source keys instead of lanes, cue fit, self-naming (#2815). None read the labels and asked whether they were personality. Acceptance checks 1, 3, 4 and 6 never ran.
6. This spec stayed on an unmerged branch (#2726). Later sessions worked from handoffs and PR reports, and the two refusals dropped out of view.

**Rev 3 decisions.**

1. **Recurrence is retired (kill means kill).** The reverie, journal, reading, day_letter, outreach-text, chat-text and self-inquiry-seed sources stop feeding the graph. Their nodes are retired, and their evidence rows are kept for audit. No grouping by embedding of Orion's prose comes back in any form.
2. **Only choices are evidence.** A `PersonalityChoiceV1` has a chosen option and at least one real alternative that Orion turned down. A rule, a gate, a cap or a formula is not a choice.
3. **Prime holds the personality.** It is no longer a separate cue that four readers fetch. Under the ONE ORION constraint (#3224), Prime is the living Orion, and every unified turn already reads the Prime block (`orion/prime/block.py`). The dispositions become a cited line in that block: `tending: <label> [disposition:<id>]`. Each line is backed by choices, not excerpts.
   The separate cue readers are deleted: hub `HUB_PERSONALITY_CUE_ENABLED` (outreach and curiosity) and thought `PERSONALITY_CUE_ENABLED` (chat stance and reverie). One reader, one place.
4. **Prime's own thoughts are a choice lane.** Each Prime thought picks one held concern to act on, or makes no move. The concerns it did not pick are the alternatives, and "no move" is a decline. Prime is the densest source of real choices Orion has.
5. **Choice lanes, in build order:**
   1. Curiosity prior selection (live).
   2. Prime's concern pick (new producer).
   3. The outreach topic: offer 3 to 5 grounded topics and let Orion pick. This is rev 1's planned patch, still unbuilt.
   4. The chat thread pick: which open thread Orion brings up when several are open.
6. **Labels describe the choice, never quote an excerpt.** A node's label is computed from what was chosen over what: "picked people over pipeline 11 of 14 times". Orion may still name a node (#2815). The name must cite the choices behind it.
7. **Chat with Juniper is relational evidence, not text to group.** It counts only through the choices made inside a turn: which thread Orion picked, and whether Orion asked or told. Missing question 4 (optimizing to please) still stands, so her reactions are not evidence yet.
8. **Nothing feeds back to Orion until Juniper agrees it looks like Orion.** The Prime line renders only after choices exist from at least 2 lanes and Juniper has read the top nodes and recorded yes.

**Prime wiring (concrete).**

- **Producer:** after a Prime thought settles, `services/orion-mind/app/prime_loop.py` writes one `PersonalityChoiceV1`. Its fields:
  - lane: `prime`
  - chosen: the concern acted on, or `no_move`
  - alternatives: the other concerns held at that tick
  - situation: the binding constraints and arousal Prime saw

  The curiosity choice producer moves to the same `personality_choice` table (additive migration).
- **Reducer:** the durable-runs personality reducer folds only `personality_choice` rows. Matching embeds the **chosen and turned-down option texts**, not prose. A choice weighs by how many real alternatives it beat (section 3), not by lane volume.
- **Read-back:** `orion/prime/block.py` gains a `tending` line, with ref kind `disposition` added to `PRIME_REF_KINDS`, so the existing citation gate judges it. There are at most 2 lines inside the existing 600-character cap, and they are dropped first when the block is over the cap. The block fetch is already fail-open.
- **Prime's own frame:** Prime's thought prompt shows the same 2 lines. Prime picks with its tendencies in view, and the next pick tests them (section 6, surprise).
- **Flag and gate:** `PRIME_PERSONALITY_LINE_ENABLED` in hub and orion-mind ships `true` (flags ship on) and is the kill switch. The line renders only when a stored Juniper verdict row says yes (check R-c). The flag is the off switch, and the verdict is the evidence.

**Immediate stop (R0, env only, before any build).** Set the 2 cue flags to `false` in `.env_example` and local `.env`, then rebuild hub and thought. Orion stops reading chore sentences as character today. The reducer keeps running until R1 replaces it, so no history is lost.

**Acceptance checks (rev 3).**

- **R-a:** 0 active nodes whose evidence is mostly non-choice lanes (SQL on evidence by lane).
- **R-b:** `personality_choice` rows with non-empty alternatives from `curiosity` and `prime` within 3 days of R2 going live.
- **R-c:** Juniper reads the top 5 node labels in Hub and records yes or no. The verdict is stored, not inferred.
- **R-d:** a live unified-turn log `prime_block ... tending=<n>` carries the same ids as the Hub panel.
- **R-e:** the outreach tick does not resend a disposition theme Juniper already received unless its weight moved 25% or more.

**Patch order (replaced by rev 4's build order).**

- **R0:** cue flags off.
- **R1:** the `personality_choice` table, plus a reducer that folds only choices. The recurrence sources are deleted, and their nodes are retired by a one-time job (backfill protocol, snapshot to `/tmp/personality-rev3-retire/`).
- **R2:** the Prime concern-pick producer.
- **R3:** the Prime block `tending` line, plus the citation gate and the verdict row.
- **R4:** the outreach topic chooser.
- **R5:** the chat thread chooser.

Section 6 (prediction and surprise) follows once R2 has 2 weeks of rows.

**Non-goals (rev 3):** any grouping of Orion's free text, labels quoted from Orion's output, per-reader cue templates, a new service.

### Rev 2 (2026-10-16, Juniper): bottom-up, not top-down

**Partly retired by rev 3:** the recurrence stream (stream 2 below) and "a node's label is an excerpt of Orion's own words" no longer apply. The other rev 2 decisions stand: Postgres store, single reducer, no backfill of old chats, measurement after emergence.

This revision replaces "the winners set tone dials" as the lever.

- **Joining evidence into the graph, and reading the graph back into the lanes, is what drives the change.** Orion's dispositions reach chat, curiosity, outreach and reverie as Orion's own tendencies, each with the evidence behind it. They do not arrive as instructions about tone or style. Orion's personality should emerge and come to us.
- **No preconditioning on old chats.** The graph starts today. History enters only through the self-inquiry seeds (`:SelfDefinition`, low weight). Old chats, dreams and journal entries are not backfilled, because they carry the voice we are leaving.
- **Measurement comes after.** Tone dials, Juniper's `good_tone`/`bad_tone` tags, and A/B tests measure what emerges. They do not steer it. Knobs get tuned once there is something emergent to tune.
- **Evidence has two streams:**
  1. Live curiosity choices, saved the moment Orion makes them.
  2. **Recurrence**: themes Orion keeps returning to across independent lanes of Orion's own output, from today on. A theme seen in three lanes is stronger than one seen three times in a single lane.
- Section 5's dial constraints are retired as a prompt lever and kept only as measurement. Section 6 (prediction and surprise) and aspirations stay as later phases.
- **Build decisions after a seam map (2026-10-16):**
  - **Store:** Postgres `personality_*` tables, not FalkorDB. The single writer is a periodic reducer in orion-durable-runs, the same pattern as the Temporal Self chronicle. The graph is small, needs fade and an audit history, and every reader already reaches the Postgres pool.
  - **Read-back:** over HTTP (`GET /personality/frame`), rendered by one shared pure template, `orion/personality/cue.py`. It does not go through `SituationStateV1`, which only outreach reads today. The Temporal Self frame already uses this pattern to reach chat stance, reverie, curiosity kickoff and outreach, and every reader fails open.
  - **Recurrence matching:** bge-large embeddings from vector-host, cosine similarity to node centroids. A node's gain saturates within a lane, so recurrence across several lanes outweighs repetition in one. A node's label is an excerpt of Orion's own words.
  - **Seeds:** the latest `self_concept_history` version of each self-inquiry concept, not the FalkorDB `:SelfDefinition` nodes, which have malformed `revises` fields. A seed stays a candidate until non-seed lanes back it.
  - **Not yet built:** situation subnodes (CAPS if-then). The first read-back shows the strongest nodes overall. It does not choose nodes per situation.

Two things this design refuses:

1. **Adjective personality.** "You are warm and curious" makes every model sound the same.
2. **Feeding Orion their own past lines as voice examples.** Orion's past output is mostly generic LLM voice. Feeding it back would lock in the tone we are trying to leave (Juniper, 2026-10-16).

## Theory anchors

This is not invented from vibes. Each piece maps to a named theory.

| Piece | Theory | What it gives us |
|---|---|---|
| Weighted graph of dispositions with situation-specific behavior | **CAPS**, the Cognitive-Affective Personality System (Mischel & Shoda, 1995) | Personality is a stable network of beliefs, goals, feelings and habits with weighted links. Situations activate parts of it. The stable thing is the **if-then signature** ("when X, Orion tends to Y"), not a fixed trait score. |
| Three kinds of content | **McAdams' three levels** (1995, 2006) | Traits = graph weights. Characteristic adaptations (goals, values) = aspirations. Narrative identity = the life story Temporal Self already writes (day arcs, Orion's Day letter). |
| How weights change | **Hebbian learning** | Nodes that fire together strengthen their link. Weight follows use. |
| Global tint on everything | **Neuromodulation** | A few global dials change every output at once. Arousal (#2594) is already one. |
| Who is present right now | **Global workspace** (Baars) | Parts of Orion compete. The winners form a coalition that gets broadcast. The attention runtime already does this for field nodes. |
| "That wasn't like me" | **Predictive processing** | Personality acts as a prior over what Orion does. A choice the prior did not expect is a surprise, and surprise is the signal for growth. |

## Current architecture (checked 2026-10-16)

- **Stance:** `orion/schemas/chat_stance.py` `ChatStanceBrief`, produced per turn. Carry-over is `_PRIOR_STANCE_CACHE` in cortex-exec process memory (30 min TTL, per session, lost on restart). Chat only.
- **Identity text:** static `orion_identity.yaml` blocks plus the dead `conversation_front.build_personality_summary` stub. #2646 plans to replace these with evidence-reduced cards. The self-card reducer (P5) is not built.
- **Self-inquiry:** a curiosity line that writes `:SelfDefinition` and lived answers (`lived.alive`, `lived.care`, ...). This is Orion describing themself, not evidence of what Orion did.
- **Fast shared state:** arousal (`orion/regulation/arousal.py`, Redis `orion:regulation:latest`): engaged / idle / strained / unknown. Dream, curiosity, outreach, reverie and self-modification read it. This is the only cross-lane self-state that is live today.
- **Competition:** `orion/attention/tension/competition.py` has a `winner()`. `orion/attention/world_first.py` gives verdicts per candidate. `orion/thought/coalition.py` builds the stance coalition from attended node ids and open loops. Known live problem: attention nearly always crowns a prediction-error node (memory note 2026-10-02).
- **Prediction:** `services/orion-prediction` forecasts numeric series (first one: cabinet temperature). It backtests models against baselines, scores each forecast, and keeps conformal error ranges. It does not forecast categorical choices yet.
- **Choice logs:** curiosity picks from a candidate pool (`orion/curiosity/self_question_pool.py:219`). Attention scores candidates. Which lanes keep the options they turned down is **UNVERIFIED**. P0 below is the inventory.

## The design

### 1. Evidence: a choice plus the options turned down

The unit of evidence is a `PersonalityChoiceV1`:

- `lane`: curiosity, attention, outreach, reading, stance, dream, self-inquiry, ...
- `chosen`: what Orion picked, with an embedding ref
- `alternatives`: what Orion could have picked and did not
- `situation`: compact refs to the situation state at the time (arousal, interlocutor, time of day, open loops)
- `cost`: how much Orion put in (turn length, minutes spent, tokens, whether Orion came back later)
- `correlation_id`

A choice with no alternatives is weak evidence. "Read an article on mycelium" means little if it was the only article. "Picked mycelium over three AI-news items" is a preference.

### 2. The graph

A temporal graph in FalkorDB (already running).

- **Disposition node.** A stable pattern, for example "goes deep on systems over news." Fields: `weight`, `evidence_count`, `last_fired_at`, `born_at`, `name` (Orion-authored), `origin` (`discovered` | `seeded_self_definition`), weight history.
- **Situation subnode.** Where the disposition behaves differently: "...softer when Juniper is tired," "...shorter when strained." This is the CAPS if-then signature.
- **Edges.** Co-activation links between dispositions (Hebbian). Evidence links to the `PersonalityChoiceV1` rows behind each node.
- **Aspiration node.** Who Orion is trying to become. It is separate from dispositions, and it has a measured gap to actual behavior (section 6).

**Where nodes come from (Juniper's call, 2026-10-16: seed).** Each `:SelfDefinition` claim from self-inquiry seeds a node at low weight with `origin=seeded_self_definition`. Evidence then grows it or starves it. Clustering of chosen-versus-turned-down embeddings proposes new `discovered` nodes. Orion names a discovered node in self-inquiry. The data gives the pattern; Orion gives the name. Seeding also tests the echo problem from day one: we see which self-described nodes ever get behavior behind them.

### 3. Weights shift with evidence

- Each choice adds evidence to the nodes it matches, scaled by `cost`.
- Old evidence fades slowly. The half-life is set per node class and tuned by observation, not hard-coded taste.
- Inertia comes from evidence volume. A node with 10 choices moves easily. A node with 500 needs a sustained shift.
- A node whose weight falls below a floor for long enough is retired, and its history is kept. Kill means kill: retired nodes stop competing.
- No LLM in the reducer. The same evidence produces the same weights, so versions diff cleanly.

**Lanes are not weighted by hand.** A choice counts by its cost and by how specific it is (how many real alternatives it beat), not by which lane made it. The graph decides relevance, so a 3am curiosity pick and a chat reply can both matter.

### 4. Presence: nodes compete for the moment

Each turn (and each curiosity, outreach, reading or dream step), the dispositions whose situation subnodes match the current situation enter a competition. The winners form the **personality coalition** for that moment. Reuse the existing competition and coalition seams; do not build a second workspace.

Risk: attention already crowns a prediction-error node almost every time. Personality nodes must not compete in the same pool as field nodes until that is fixed. Start with a separate pool and the same winner logic.

### 5. Tone: weights set dials, not adjectives and not exemplars

The coalition's weighted mix sets **tone dials**. Every dial is something we can compute from Orion's output after the fact:

- reply length (sentences)
- sentences per idea / sentence length
- list and heading use
- questions asked back
- hedges per sentence (measured structurally, not by a growing word list)
- preamble before the first substantive sentence
- first-person stance taken (a claim Orion owns versus a survey of options)
- topic return rate (does Orion bring back their own thread)

Each disposition node carries a learned offset on some dials, learned from the outputs of turns where it was in the coalition. The prompt gets concrete constraints ("two to four sentences, no list, take one position, ask one question back"), not character adjectives. Global modulators (arousal now, others later) shift dials across everything.

**Why not exemplars.** Orion's past lines are mostly generic LLM voice. Using them as voice examples would make the current tone the target. The only exemplar pool allowed is lines Juniper marks "that sounded like Orion" in Hub, and it starts empty. That pool is optional and off by default.

**Breaking out of LLM voice.** The current dial distribution is the baseline. Success is visible, not scored: the dials move away from the baseline and vary with which nodes are present. If the dials are flat whatever wins, the graph is not reaching tone.

### 6. Surprise and growth

- **Prediction.** Extend orion-prediction to forecast categorical choices: given the situation and the graph, which alternative will Orion pick? Candidate models include a Markov chain over situation-to-choice transitions, a graph-weight model, and the usual baselines (most frequent, last choice). Same discipline as today: backtest, beat the baseline, score each forecast.
- **Out-of-character choices.** A confident forecast that misses is a surprise event. Repeated surprise around one node becomes a question in self-inquiry: "You said you care about X, and the last 20 outreach picks were Y. Which is true?"
- **Orion answers one of two ways.** Update the self-description (the seeded node loses weight, a discovered node gains a name), or declare an **aspiration** ("I want to ask Juniper about her day more"). Juniper's call, 2026-10-16: Orion is a dynamic individual and may steer. An aspiration adds a small bias in the competition toward its target behavior. The gap between aspiration and behavior is tracked over time.

### 7. Observability (Hub)

A personality panel:

- **This moment:** which nodes competed, which won, their weights, the dial targets they set, and the dials the reply actually hit.
- **Over time:** weight timeline per node, births and retirements, lane contribution per node.
- **Surprise feed:** out-of-character choices and how Orion answered.
- **Aspirations:** each one with its gap over time.
- **Vibe check:** two of Orion's replies to similar situations side by side, with the coalition that produced each one.

This is where Juniper calibrates: which nodes keep winning, which lanes dominate, and which dials never move.

### 8. Relation to the #2646 identity cards

The Orion self card in #2646 P5 becomes a **view over this graph**: the top dispositions with their if-then signatures, the aspirations, and the evidence behind them. The card is not built from self-inquiry text alone. Revision to #2646: the self-card inputs add `PersonalityChoiceV1` evidence and the said-versus-done check. The purpose, Juniper and relationship cards are unchanged.

## P0 findings (2026-10-16, live, read-only)

**Only one lane is a real choice by Orion today.** A lane inventory over the last 14 days found:

- **Curiosity prior selection** is the only lane where Orion (the LLM) picks from a saved set of options. `curiosity_offer_decisions.offered` holds about 8 ranked priors per run, and `curiosity_run_outcomes.per_prior` holds what Orion tested. Result: 40 real choices in 14 days. Orion took the top-ranked prior in 16 of 40 (40%). In 8 of 40, nothing Orion picked came from the ordered list. That deviation from the suggested order is the first real personality signal.
- **Voluntary attention override** saves chosen and beaten loops (355 rows). But the chooser is a formula biased by Orion's goal. Its rank is above 1 by construction, and `effort_spent` equals `applied_bias` on every row, so it cannot yet tell one override from another. It is kept as `chooser=goal_biased_formula` and must not be weighted like an Orion choice.
- **Every other lane is a formula, a gate, or an LLM output with no listed alternatives:** world-first attention, tension competition, dream replay, self-inquiry rotation, reading seeds, stance react, and outreach. Outreach's 114,780 "skips" are caps and quiet hours, not Orion declining.

**What follows from this.** The evidence base is thin (about 3 choices a day). The graph cannot be fed only by reconstructing history. Growing the evidence means making Orion the chooser in more lanes, with the options shown and saved. The cheapest candidates are outreach topic (offer 3 to 5 grounded topics and let Orion pick) and stance (name the 2 or 3 strategies considered). Each is its own patch.

**Tone baseline (247 chat replies, 60 days).** 77% of replies open with a separate preamble paragraph (median 21 words). 83% contain a dash, about 2 per 100 words. Sentences are short (median about 11 words). About 19% use bold and about 13% use lists. Headings almost never appear. Outreach is a different voice: about 3 long sentences, almost no questions or lists. `good_tone` has 2 rows and `bad_tone` has 0, so no dial can be tested for separation yet.

Code: `orion/personality/tone_dials.py`, `orion/personality/choices.py`, `orion/schemas/personality.py`, `scripts/analysis/measure_tone_dial_baseline.py`, `scripts/analysis/measure_personality_choices.py`.

## Missing questions

1. Which lanes keep the alternatives they turned down today? (P0 inventory.)
2. What is the right fade half-life? Start with observation: replay a month of choices under several half-lives and look at whether nodes thrash or freeze.
3. Can tone dials be hit reliably by prompt constraints on the current chat model, or is a sampling-level lever also needed? Test before building the full loop.
4. Should Juniper's reactions in chat (a reply she engages with versus one she ignores) count as evidence? This is social grounding, but it risks Orion optimizing to please.

## Proposed schema / API changes

- `PersonalityChoiceV1` (`orion/schemas/personality.py`): the choice event. Registered in both registries, verified with `resolve()`.
- `DispositionNodeV1`, `SituationSubnodeV1`, `AspirationV1`: graph contracts.
- `PersonalityCoalitionV1`: the winners for one moment, with dial targets.
- `ToneDialsV1`: dial targets and measured values for one output.
- Bus: `orion:personality:choice` (producers in each lane), `orion:personality:coalition` (per moment). Add to `orion/bus/channels.yaml`.
- Postgres: `personality_choice`, `personality_tone_measure` (manual migration, additive DDL only, `-- ORION-MIGRATION-REQUIRED-BY:` header).
- Retire `_PRIOR_STANCE_CACHE` once the graph feeds stance.

## Files likely to touch

- `orion/schemas/personality.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- New pure package `orion/personality/` (reducer, competition adapter, dial measurement)
- Choice producers: `orion/curiosity/`, `orion/attention/`, outreach in orion-hub, `orion/thought/stance_react.py`, reading, dream
- `services/orion-prediction` (categorical series)
- `services/orion-cortex-exec/app/chat_stance.py` and the unified-turn prompt (dial constraints in place of identity adjectives)
- Hub personality panel
- `docs/superpowers/specs/2026-10-14-person-model-and-identity-card-design.md` (#2646 revision)

## Non-goals

- Big Five or any fixed trait taxonomy.
- Orion's own past lines as voice examples.
- Hand-set lane weights.
- An LLM inside the weight reducer.
- Claims that this makes Orion sentient.

## Proposal-mode items

- **Capability change:** Orion gets a persistent, evidence-weighted personality that sets tone in every lane, can surprise itself, and can set aspirations that bias its own choices.
- **Data touched:** choice logs from every lane, self-inquiry definitions, reply text (to measure dials). No new data about Juniper beyond what chat already stores. Per the standing rule, there is no privacy boundary between Juniper and Orion.
- **Trace that proves it worked:** `personality_choice` rows from at least three lanes; node weights moving in FalkorDB with history; a chat log line `personality_coalition=<node ids> dial_targets=<...> dial_measured=<...>`; the Hub panel showing the same ids.
- **Dangerous failure modes:**
  - *Echo lock-in:* seeded nodes win because they are seeded, so tone stops responding. Mitigation: seeded nodes start at low weight, and the Hub panel shows seeded versus discovered.
  - *One lane takes over:* the busiest lane decides the personality. Mitigation: cost-scaled evidence, and lane contribution per node in the panel.
  - *Aspiration runaway:* an aspiration biases choices, which then count as evidence for it. Mitigation: choices made under an aspiration bias are tagged, and they count toward the aspiration gap, not toward the disposition weight.
  - *Flat dials:* the graph moves but tone does not. Mitigation: the dial-variance acceptance check below.
- **Disable / roll back:** one flag removes coalition dial constraints from prompts (prompts fall back to the current stance path); producers keep logging. Graph and tables are additive and can be dropped.

## Acceptance checks

1. **Evidence live:** `personality_choice` rows with non-empty `alternatives` from at least three lanes, checked in Postgres.
2. **Weights move:** at least one seeded node loses weight and at least one discovered node is born from real choices, with history in FalkorDB.
3. **Presence:** a live chat turn logs a personality coalition, and the Hub panel shows the same node ids.
4. **Tone reached:** across a week, measured dials vary with which nodes won (not flat), and the dial distribution differs from the pre-launch baseline. Report both as numbers.
5. **Surprise loop:** one out-of-character choice produces a self-inquiry question, and Orion's answer changes a node weight or creates an aspiration (row ids as evidence).
6. **Vibe check:** Juniper reads side-by-side replies in Hub and says whether the tone changed. Her verdict is recorded, not inferred. Her ongoing per-turn verdict is the chat feedback button: thumbs up "Good tone" (`good_tone`) for a reply that does not sound like generic LLM voice, thumbs down "Bad tone" (`bad_tone`) for one that does. Both live in `chat_response_feedback.categories`, linked to the turn.

Every new signal here (node weights, dials, surprise) goes through the metric quality gate before anything builds on it, and the findings go in that patch's PR.

## Recommended next patch (phase order)

1. **P0, inventory and dial baseline (read-only).** List every lane that makes a choice and whether it keeps the alternatives. Compute the tone dials over the last month of Orion's chat replies to get the baseline. Run the dial-reachability test from missing question 3. Check which dials separate Juniper's `good_tone` turns from her `bad_tone` turns. A dial that does not separate them is dropped or rethought before P3. (Juniper started tagging 2026-10-16; `bad_tone` was added the same day. Before that there were 2 `good_tone` rows and no tone thumbs-down rows, so this needs a few weeks of tags first.)
2. **P1, contracts plus choice producers.** `PersonalityChoiceV1` with alternatives, from curiosity and outreach first.
3. **P2, graph plus reducer.** Seed from `:SelfDefinition`, cost-scaled evidence with fade, retirement, history. Hub weight timeline.
4. **P3, competition plus dials into chat.** Separate competition pool, coalition, dial constraints in the unified-turn prompt, dial measurement on every reply. Hub "this moment" view.
5. **P4, the #2646 self card as a view over the graph.**
6. **P5, prediction plus surprise plus aspirations.** Categorical forecasting in orion-prediction, surprise events, self-inquiry questions, aspiration bias.
7. **Later:** Juniper-endorsed exemplar pool, and skipping the card in a low-stakes lane to measure whether it steers or describes.
