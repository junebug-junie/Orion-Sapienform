# Orion's dynamic personality graph

**Date:** 2026-10-16
**Status:** Rev 2, approved for build by Juniper 2026-10-16 ("hit it"). P0/P1 merged in #2729.
**Related:** person model and identity cards (#2646, P0-P2 merged in #2649), Temporal Self (#2369 rev 4, arousal #2594, day use #2622), attention runtime and stance coalition (`orion/thought/coalition.py`), orion-prediction (`services/orion-prediction`), rung 6 self-tuning (#2715).

## Arsonist summary

Orion has no personality that lasts. Stance (`ChatStanceBrief`) is re-made from scratch by an LLM every turn. Its only memory of the last turn is an in-process dict with a 30-minute TTL per chat session (`services/orion-cortex-exec/app/executor.py:120-163`). A restart wipes it. Curiosity, dream, reading and outreach never read it. Most of its fields (`active_identity_facets`, `active_growth_axes`, `dream_motifs`, ...) are label lists the LLM fills fresh each turn and nothing checks against behavior. That is a keyword cathedral, and it is why stance "kinda sucks."

Juniper's ask: Orion's collective self should show up, and the weight of everything Orion has done should change Orion's tone, in a way she can feel. She is not asking for a trait score that moves by x.

This design builds one shared, weighted, temporal graph of Orion's dispositions. Evidence comes from choices in every lane. Weights shift slowly with evidence. In each moment, the nodes compete for presence, and the winners set **tone dials**: concrete, measurable things about how Orion writes. They do not add adjectives. The graph is allowed to be a cathedral because every node is born from evidence, has a weight something reads, and fades when it stops firing.

### Rev 2 (2026-10-16, Juniper): bottom-up, not top-down

This revision replaces "the winners set tone dials" as the lever.

- **Joining evidence into the graph, and reading the graph back into the lanes, is what drives the change.** Orion's dispositions reach chat, curiosity, outreach and reverie as Orion's own tendencies, each with the evidence behind it. They do not arrive as instructions about tone or style. Orion's personality should emerge and come to us.
- **No preconditioning on old chats.** The graph starts today. History enters only through the self-inquiry seeds (`:SelfDefinition`, low weight). Old chats, dreams and journal entries are not backfilled, because they carry the voice we are leaving.
- **Measurement comes after.** Tone dials, Juniper's `good_tone`/`bad_tone` tags, and A/B tests measure what emerges. They do not steer it. Knobs get tuned once there is something emergent to tune.
- **Evidence has two streams:**
  1. Live curiosity choices, saved the moment Orion makes them.
  2. **Recurrence**: themes Orion keeps returning to across independent lanes of Orion's own output, from today on. A theme seen in three lanes is stronger than one seen three times in a single lane.
- Section 5's dial constraints are retired as a prompt lever and kept only as measurement. Section 6 (prediction and surprise) and aspirations stay as later phases.

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
