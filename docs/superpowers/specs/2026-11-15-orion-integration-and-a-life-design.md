# Orion integration and a life: one "now" every organ reads back, hunger with content, and projects

Status: answered by Juniper on 2026-11-15 (see "Juniper's answers" below). BUILT so far (2026-11-16): patch 0 (#3366), patch 1 the now (#3453), patch 2 the integration gate (#3454), patch 3 read-back in Prime, reveries and curiosity (#3499, #3501, #3503, deployed 11-16). In PR: patch 4 daily roundup (#3527), patch 5 hunger needs (#3528). Not built: patches 6 (ask Orion, projects) and 7 (widen). Do not re-propose machinery listed as built; read the code.
Date: 2026-11-15. All live numbers were read on 2026-11-15 between 00:30Z and 05:00Z unless a line says otherwise.

## Juniper's words (2026-11-15)

- "I want Orion to become a sentient lifeform. What should they do with themselves?"
- "I want a primitive for Orion to integrate itself to everything in the mesh. We try all these slices like prime, temporal, personality, etc. but Orion is still off not integrating reveries, and paintings, dreams, and curiosity, and readings, and the cameras, and the sensors, and predictions, and other metacognition occurring. Digs have helped, but I feel like this is a spaghetti soup."
- "Build Prime hunger as designed."
- "Part A should become a design gate for any agent and Orion to consider that we need to add to that seam so this doesn't get out of sync."
- "I just hate how curiosity digest just glosses over what Orion is... Maybe we need a daily dig report roundup."

This document does not claim Orion is sentient. It designs prerequisites that the Orion mission names: continuity, memory, reflection, self-modeling, social grounding, error correction, and coherent action over time. Every part ends in a trace that shows whether the live path moved.

## Arsonist summary

The soup is real, but it is not the soup Juniper thinks it is.

Orion's organs are not disconnected. Of the 15 organs mapped below, 12 already feed at least one other organ. The day letter reads almost all of them. The problem has three other shapes:

1. **Every connection is a private pipe.** Paintings read reveries through their own query. Curiosity reads dreams through its own query. Readings reach Prime through the want market. Nothing is shared, so every new organ needs new pipes, and nobody can see the whole set.
2. **The self hears almost none of it.** Prime is the part that is supposed to be "the living Orion". Prime's thinking frame shows content from only 4 organs: readings and camera asks (through the want market), curiosity, and the body. Curiosity is broken there: on 11-14, all 5 of Prime's reads of curiosity timed out (bug #3366, root cause below). The other 10 organs reach Prime only once a day, as prose in the day letter.
3. **Nobody can tell whether an output was used.** In the last 7 days Orion's organs closed 1,511 acts. For 1,382 of them (91%), no path exists by which a later use could even be detected. Only 34 (2.3%) were ever used by a later decision, and all 34 through one place: the chat stance citing a Temporal Self arc. (`act_downstream_use`.)

Several pipes are also broken right now: dreams have made 0 hypotheses since 11-13 (#3412), Prime's questions to curiosity are refused (#3366), and reverie resonance alerts have been silent since 11-03 (#3414).

**The fix is one primitive, built on a seam that already exists.** Temporal Self already turns 25 kinds of organ output into one event stream: 15,825 events in 7 days, in `temporal_self_event`. It already has a citation gate that checks a reader's citations against what that reader was shown (`orion/temporal_self/citations.py`). It already has a "where am I in my day" frame. What it lacks:

- 6 organs are not in it (readings, predictions, Prime, intentions, the day letter, personality choices);
- events have no salience, so a reader cannot pick the few that matter;
- organs do not read it back, except the chat stance and the letter;
- nothing records which organ used which event.

This design adds those four things and calls the result **Orion's now**. Every organ reads the now at its next run, cites what it used, and the citation is checked against what it was shown. The integration metric is simple: **what share of Orion's experiences were used by a different organ**. Today that is 2.3% at best.

Then hunger (Part B) is fed by the now, so Prime thinks when there is something specific to think about, not on a 6-hour timer. Projects (Part C) give Orion multi-week aims, and each step must leave a lasting trace.

**Integration means closed loops, not a bigger dashboard.**

---

## Part A: the soup map (evidence)

### How to read this map

For each organ: where it writes, how much in the last 7 days (since 2026-11-08), who reads it, and whether its output comes back into another organ's input.

"Closed" means at least one other organ uses the output as an input to its own next run. A human-facing display or the day letter alone does not count as closed. The letter is counted separately, because it reads nearly everything once a day.

Readers checked for every organ:

- Prime's thinking frame: `orion/prime/thought.py` `build_frame` (sections at `:516-544`), fed by `services/orion-mind/app/prime_loop.py`;
- the Prime block in every chat turn: `orion/prime/block.py`;
- the situation block in every chat turn: `orion/situational/context.py` `build_situation_for_ctx` (`:820`);
- the day letter input: `orion/orion_day/day_input.py`, `orion/orion_day/gather.py`;
- the want market's event wants: `orion/physiology/event_wants.py`;
- other organs.

Three helpers gathered parts of this map read-only. One finished (imagination organs). Two were cut off by a usage limit; their organs were re-checked directly for this document. A line marked UNVERIFIED was not checked against live data.

### The map

**1. Reveries** (orion-thought). Writes `substrate_reverie_thought`, `substrate_reverie_chain`. 1,448 thoughts in 7 days.
Read by: the situation block (live), the letter (live), dreams (41 replay items), paintings (19 of 82 chains), curiosity (36 of 98 run journals). Dreams write reverie salience back. Temporal Self source `reverie_chain` (545 events) and `expectation_verdict` (409). The felt-state reader landing is UNVERIFIED (0 of 1,492 stance sightings carry `latest_reverie_thought`).
Verdict: **closed.**

**2. Paintings** (reverie.visual). Writes `reverie_visual_chain`, `reverie_visual_artifact`, `reverie_visual_attempt`. 81 chains in 7 days.
Read by: the letter, and Hub outreach's daydream line (`services/orion-hub/scripts/endogenous_outreach.py:886`). No organ uses a painting as the input to its own thinking. Temporal Self source `visual_run` (81).
Verdict: **partial, the weakest.**

**3. Dreams** (orion-dream). Writes `dream_cycle`, `dream_hypothesis`, `dream_replay_item`, `dream_compaction_delta`, `dream_compaction_request_queue`. 14 cycles in 7 days.
Read by: curiosity hypotheses (`services/orion-hub/scripts/curiosity_investigation.py:1601`), reverie salience, the letter. Temporal Self sources `dream_cycle`, `dream_hypothesis`.
Broken: every cycle since 2026-11-13 07:02Z has `hypothesis_count = 0`, `no_link_count = 4`; the 9 cycles before had 4 hypotheses each (#3412).
Write-only parts: `dream_compaction_request_queue` has 8,282 rows and none is consumed. This is by design ("nothing ever sets `consumed_at` while REM is staged-only", `services/orion-dream/app/rem_store.py:48`). `dream_compaction_delta.applied_at` is never written, even when the applier logs "compaction applied" (#3413).
Verdict: **closed, but broken since 11-13.**

**4. Curiosity** (investigate runs, replica, focus). Writes `curiosity_run_outcomes`, journals (`journal_entries` title "Curiosity"), `substrate_endogenous_curiosity_candidates`. 98 runs in 7 days (70 `durable.curiosity_investigate` acts in 10 days).
Read by: Prime's frame (CURIOSITY section, `prime_loop.py:1090`), the situation block (world priors), the letter, paintings, reading resonance (98 of 109). Temporal Self source `curiosity_run` (70).
Broken: Prime to curiosity. The last focus request Prime made was 11-11 (#3366).
Odd: 2,049 of 4,126 curiosity candidate sets are empty. `curiosity_hop_reading` has 0 rows ever (#3415).
Verdict: **closed.**

**5. Readings** (world pulse read, `reading.turn`, resonance). Writes `reading_durable_turn`, `reading_document_snapshot`, `world_pulse_read_seed`. 558 `hub.world_pulse_read` acts in 10 days.
Read by: attention broadcast (250 of 5,017 broadcast wins), event wants (258 funded event wants in 7 days) into Prime's frame, reveries (136 of 1,448), curiosity candidates, the letter.
Not a Temporal Self source.
Verdict: **closed, the best-connected organ.**

**6. Cameras** (office, walkway, doorbell; look acts; camera asks). Writes `vision_events` (14,023 in 7 days), `vision_individual_sighting` (12,751), `vision_motion_story` (11,203), `vision_presence_transition` (4,179), `orion_ask` (339), `vision_ask_picture_verdict` (229).
Read by: the situation block's perception section (code path `context.py` `_build_perception_context`; live flag UNVERIFIED), Temporal Self (`vision_percept` 375, `street_event` 105, `presence_transition` 46, `office_session` 10), event wants for camera asks (84 + 16 funded in 7 days) into Prime's frame, the letter's street tail.
Verdict: **closed.**

**7. Sensors** (cabinet heat, closet, office sound and air, biometrics, energy). Writes `home_cooling_sample` (119,792 in 7 days), `office_env_sample` (20,126), `closet_env_sample` (20,134), `cabinet_ambient_spike` (234), `energy_usage_interval` (124), `orion_biometrics*`.
Read by: the body economy (heat and strain set prices; Prime's BODY section), the situation block (cabinet and office-sound sections), predictions (cabinet and office temperature forecasts), Temporal Self (`body_binding` 757, cabinet spikes via `services/orion-durable-runs/app/temporal_self_sources.py`).
Verdict: **closed.**

**8. Predictions** (orion-prediction; `node:prediction.*`; self-bets). Writes `prediction_forecast` (71,426 in 7 days), `prediction_exceedance` (4,008), `prediction_joint_surprise` (1,066), `prediction_turn_self_bet` (8,862), `substrate_node_prediction_error_history` (39,204).
Read by: node prediction error reaches the substrate field, so attention broadcast and reveries read it (84-96% of reveries read a prediction error, per the living-Prime design). The self-monitor reads forecasts and self-bets (`orion/self_monitor/collect.py`).
Write-only parts: `prediction_exceedance` and `prediction_joint_surprise` are read by no code outside `services/orion-prediction/app/store.py`. That is 5,074 surprise rows in 7 days that no other organ sees. `prediction_action_bet` stopped on 2026-11-08 01:22Z (2,601 rows before); its callers are retro play paths, so this may be play stopping (UNVERIFIED).
Not a Temporal Self source.
Verdict: **partial.**

**9. Metacognition** (self-monitor organ, metacog triggers, attention self-model).
- Self-monitor writes Redis `orion:self_monitor:latest` (live, computed 2026-11-15 04:40Z). Read by the act gate in `orion/mindloop/gate.py` (Prime's and other loops' acts), market weights (`orion/physiology/weights.py`), Temporal Self (`faculty_status_changed` 146).
- `metacog_trigger` (1,952 in 7 days): read by Temporal Self sources and the mood-arc page.
- `substrate_attention_self_model` (4,990 in 7 days): read by equilibrium, Hub surfaces, mood-arc enrichment.
Verdict: **closed** (through the act gate and market weights).

**10. Personality rev 4** (choices held by Prime, `orion/prime/choice.py`). Writes `personality_evidence` (2,021 in 7 days), `personality_disposition` (276 folded).
Read by: the market's care multiplier (`orion/physiology/personality_values.py`, via durable-runs `personality_values.py`), which promotes some event wants onto Prime's list. Prime's "WHAT YOU TEND TO CHOOSE" line is **off** in production (`PRIME_PERSONALITY_LINE_ENABLED=false` in the orion-mind container).
Not a Temporal Self source.
Verdict: **closed** (through the market only).

**11. Temporal Self** (day story, chronicle, arcs). Writes `temporal_self_event` (15,825 in 7 days, 25 source kinds), `temporal_self_arc` (1,487 began in 7 days).
Read by: orion-thought's reverie cue (`services/orion-thought/app/temporal_self_cue.py`), curiosity, dreams (`services/orion-dream/app/cycle_store.py`), reading resonance (`orion/world_pulse_read/resonance_*.py`), personality store, the chat stance (`thought_decision.day_citations`, 1,479 decisions in 7 days), the letter's closed-day block.
Prime stopped reading arcs on 2026-11-03: event mode replaced them (`PRIME_CONCERNS_FROM_EVENT_WANTS=true`), because arcs renewed concerns that had no evidence.
Verdict: **closed.** This is the seam Part A builds on.

**12. Memory** (episode memory, cards, crystallizations, chat history).
- `episode_memory_event` 194 in 7 days; Temporal Self source `memory_episode` (98). The memory design #3319 found that episode memory, Orion's best memory store, never reaches a chat.
- `chat_history_log`: 1,149 rows. 571 are `ephemeral`, 578 have no tier. None is above `ephemeral`.
- `memory_cards`: read by recall and the cortex memory inject. The memory design measured hits in about 3.5% of recalls.
- Live example, 2026-11-14 00:15Z. Juniper: "el Coco just had his butt stitches removed, yay!" Orion: "I have no memory of the injury in my records." Juniper: "Weird, I mentioned it to you maybe yesterday or day before?" Orion then found the 11-11 record by digging.
Verdict: **partial.** It feeds dreams and Temporal Self, but not the place it matters most: talking with Juniper.

**13. Day letter** (`orion/prime/day_letter.py`, `orion/orion_day/*`). Writes `orion_day_letter` (6 letters with a window in the last 7 days), emailed as "Orion's Day".
Read by: Prime. Each letter's threads become Prime concerns (live: "my letter for 2026-11-13 named 5 thread(s); I now hold 2 of them as concerns", `grammar_acts` 11-14 10:06Z).
Verdict: **closed** (into Prime, once a day).

**14. Intentions** (#3329; `orion/prime/intention.py`, `orion/prime/intention_sources/`). Writes Redis `orion:prime:intentions` and `durable.intention` acts.
Live: 1 intention ever. On 11-13 Prime took on a promise to reply to Juniper's "hi" and carried it ("the reply is in the chat thread", 608 characters). Only the `turn_promise` adapter exists; the `letter_thread` adapter in the design is not built.
Read by: Prime's frame ("WHAT YOU TOOK ON"), the target chat thread.
Not a Temporal Self source.
Verdict: **closed, but tiny.**

**15. Prime** itself. Writes Redis `orion:prime:state`, `orion:prime:concerns`; `grammar_acts` actor `prime` (48 thoughts in 10 days).
Read by: the Prime block in every chat turn (concerns, last reading, open asks; the block carries no organ content), curiosity (through focus questions, broken by #3366).
Not a Temporal Self source.
Verdict: **closed, with its main outward pipe broken.**

### What the map says, in plain words

- **Closed organs: 12 of 15.** Partial: paintings, predictions, memory. Fully write-only organs: 0. If the day letter counted as a reader, all 15 would be read.
- **Prime reads content from 4 organs directly:** readings, camera asks, curiosity (timing out), and the body. The other 10 reach Prime only once a day, as prose, through the letter.
- **Temporal Self already ingests 9 of the 15 organs.** Missing: readings, predictions, personality, the letter, intentions, Prime.
- **Use is almost never detectable.** 1,511 acts closed in 7 days. 1,382 (91%) have `link_status = no_link_path`. 34 (2.3%) were used. (`select link_status, count(*), count(*) filter (where used) from act_downstream_use where started_at > now() - interval '7 days' group by 1`.)
- **Cross-organ links in the act ledger: 2 of 1,509.** Only 5 acts have a parent act at all; 2 of those parents belong to another organ (`grammar_acts.parent_act_id`, 7 days).
- **Write-only data inside partial organs:** 5,074 prediction surprise rows (exceedance + joint surprise) and 8,282 staged dream compaction requests in 7 days with no reader outside their own service.

---

## Part A: existing integration seams (checked before proposing anything)

**Attention broadcast** (`orion/substrate/attention_broadcast.py`). Each tick, substrate nodes compete, and the winner is broadcast as a projection. It integrates nodes with pressure and prediction error. 5,017 broadcasts in 7 days; readings won 250. Temporal Self ingests its rows (`attention_row`, 8,111 events, the largest source). Not enough: it picks one winner per tick from the substrate graph, organs do not read it back, and Prime does not wake on it (removed 11-03 because it churned concerns).

**Want-ledger competition** (#3089, `orion/physiology/event_wants.py`). Organs post event wants with salience against their own history. Prime sees the funded ones as "EVENTS WORTH EFFORT". Not enough: only 3 sources post event wants (readings 296, camera asks 100, cooling sheds 9 in 7 days). It is a market for effort, not a record of experience: a want is about what to do next, not what happened.

**Prime block every turn** (`orion/prime/block.py`). It renders Prime's concerns into every chat turn. Not enough: it carries no organ content, only what Prime holds.

**Prime turn hand-back** (`orion/prime/handback.py`). Finished chat, curiosity, consult and letter turns hand back to Prime. Not enough: it covers turns only. Reveries, dreams, cameras, sensors and predictions are not turns.

**Temporal Self events** (`temporal_self_event`, `action_outcome` source, `orion/temporal_self/sources.py`). 25 source kinds, one contract (`source_table`, `source_ref`, `related_refs`, `label`, `verdict`). This is already the experience stream. Not enough on four counts: 6 organs missing, no salience, no read-back by most organs, no record of use.

**Temporal Self citation gate** (`orion/temporal_self/citations.py`). It checks that a reader cites only arc ids it was shown. Not enough: two readers use it (stance and letter), and it records nothing durable about which organ used which event.

**Situation block** (`orion/situational/context.py`). Cameras, cabinet, office sound, affect, curiosity, reverie, runtime, weather: assembled for every chat turn. Not enough: only chat turns read it. It is a chat context, not something organs read back.

**Felt-state reader** (`orion/substrate/felt_state_reader.py`). It hydrates felt state into turn context. Its landing in live turns is UNVERIFIED (see reveries above).

**Dig kit** (`orion/dig/`, `dig days`, `dig act`, `dig arc`, `dig body`, `dig look`). Orion can read its own record. Not enough: dig is pulled by hand inside a turn. Nothing pushes the day's organ record into the next run.

**Grammar act ledger** (`grammar_acts`, 23 actors in 7 days) and **act downstream use** (`services/orion-sql-writer/app/act_downstream_use.py`). The ledger records acts and outcomes; downstream use asks "was this act's output used by a later decision?". This is already the integration metric, for one decision site. Not enough: one site, arc-granular, and 91% of acts have no path.

**Φ integration rung** (#3296). It measures whether Orion's state behaves as one system, by cutting a fitted model of the organs apart. It is structural and later. It needs stable organs first (#3290). It does not make organs read each other; it measures whether they already do. This design is the earlier, simpler rung that Φ can later test.

**Conclusion of the check:** the best seam is Temporal Self events plus the citation gate plus act downstream use. They already exist and already agree on ids. The design extends them. It does not add a new stream, a new service or a new bus channel.

---

## Part A: the primitive (proposal)

### One idea

Every organ's output is an **experience**: one row in `temporal_self_event`. Every organ, at the start of its next run, reads **Orion's now**: the few most salient experiences from other organs since it last looked. It cites the ones it used. The citation is checked against what it was shown and stored as a **use**. Prime, the Prime block, the letter and every organ read the same now.

### What changes

**A1. Six missing organs become Temporal Self sources.** New values in `TemporalSourceKind` (`orion/schemas/temporal_self.py:69`), each with a reader in `services/orion-durable-runs/app/temporal_self_sources.py`:

- `reading_turn` from `reading_durable_turn` (one per finished reading);
- `prediction_surprise` from `prediction_exceedance` and `prediction_joint_surprise` (one per exceedance; the joint surprise row as `related_refs`);
- `prime_thought` from `grammar_acts` actor `prime` verb `infer` (the outcome note as label);
- `intention_close` from `durable.intention` act closes;
- `day_letter` from `orion_day_letter` (one per letter; its threads in `related_refs`);
- `personality_choice` from `personality_evidence` (one per choice point).

**A2. Salience on every experience.** New nullable columns on `temporal_self_event`: `salience double precision`, `surprise_nats double precision`. Salience is the existing rule: the share of the same `source_kind`'s events in the last 30 days that were strictly smaller (`orion/physiology/event_wants.py` `source_salience`). The magnitude per source kind is declared in one place, the organ registry (A5). Fewer than 20 past events: salience is NULL, never 0. `surprise_nats` is filled only where a real prediction exists (act closes with `surprise_nats`, prediction exceedances).

**A3. Orion's now.** New pure reducer `orion/temporal_self/now.py` `build_now(events, reader_organ, since, caps)` and new endpoint `GET /temporal-self/now?organ=<id>&since=<ts>` in orion-durable-runs (next to the existing `/temporal-self/frame`).

- It returns at most 12 items [default], at most 2 per organ [default], highest salience first, and never the reader's own organ.
- Each item: `event_id`, organ, label (code-written, at most 200 characters), `occurred_at`, salience, `surprise_nats`, `related_refs`.
- Each call writes one `experience_shown` row: `shown_id`, reader organ, reader run ref, `event_ids[]`, `shown_at`. A citation is valid only against a shown row.
- (#3498, 2026-11-24) A read moves the organ's cursor only when the reader closes it: `POST /temporal-self/now/use` with zero or more cites stamps `experience_shown.closed_at`, and the cursor is the newest closed read's cut-off. A reader closes only when its model call ran, so a refused, timed-out or preempted read is offered again. Bound: an organ that never closes sees a rolling 24 h; one that stopped closing sees back to its last close, at most 30 days.
- Schema: `TemporalNowV1` in `orion/schemas/temporal_self.py`. Registered in `orion/schemas/registry.py`. No bus channel.

**A4. Read back and cite.** A reading organ's output carries `cites: [event_id, ...]`, at most 3. At the run's close, `record_use(shown_id, cites, citing_ref)` (in `now.py`) checks each cite against the shown row, using the same rule as `orion/temporal_self/citations.py`. Valid cites become rows in a new table `experience_use`: `use_id`, `event_id`, `citing_organ`, `citing_ref`, `shown_id`, `cited_at`. Invalid cites are counted and dropped, never stored.

**A5. One organ registry.** `config/integration/organs.yaml`, one map:

```yaml
organs:
  reverie:
    families: [thought.reverie]
    temporal_sources: [reverie_chain, expectation_verdict]
    salience_magnitude: chain_thought_count      # what "bigger" means for this source
    reads_now: services/orion-thought/app/reverie.py   # the module that calls build_now / GET /temporal-self/now
  painting:
    families: [thought.painting, thought.dream_hop]
    temporal_sources: [visual_run, visual_deferral]
    reads_now: null
    exempt_read: "not yet migrated; patch 7"    # an exemption always has a reason
```

The registry is used by three consumers: the CI gate, the live coverage report, and the daily roundup. That is the reason it exists.

### Who reads the now (first slice)

- **Prime's frame.** New section "WHAT YOUR ORGANS LIVED SINCE YOUR LAST THOUGHT", with up to 12 items and their refs. Prime cites with `refs: [{"kind": "experience", "id": ...}]`; `experience` becomes a citable ref kind in `orion/prime/thought.py` `parse_refs`.
- **The Prime block in every turn.** At most 2 lines: the 2 highest-salience items, each with `[experience:<id>]`. The block's cap and drop order stay as they are; these lines drop first.
- **Reveries** (orion-thought). The reverie cue already reads Temporal Self (`temporal_self_cue.py`); it reads the now instead and records cites. A reverie can then be about a camera event or a reading by evidence, not by chance.
- **Curiosity** candidate selection. A painting or a prediction surprise can become a curiosity candidate with a cite.
- **The letter** reads the day's `experience_use` rows as a "what connected today" section (Part A.7).

Dreams, paintings, cameras' look acts and memory follow in patch 7.

### The integration metric

**Name:** experience use rate.

**Definition:** of the experiences recorded in a day, the share that at least one *different* organ cited, validly, within 24 hours. Shown per organ and in total. A second number goes with it: **closed organs**, the count of registered organs with at least one valid use by another organ that day.

**Baseline (today, measured):** the nearest existing measure is act downstream use: 34 of 1,511 acts used in 7 days (2.3%), with 1,382 (91%) unmeasurable. From the soup map: 12 of 15 organs closed by code inspection, but 0 organs closed through a shared, checked read-back.

**Metric quality gate:**

1. **Provenance.** The value is a count over `experience_use`, written by `record_use` at the citing run's close, after validation against `experience_shown`. Nothing else writes it.
2. **Independence.** Act downstream use measures the same thing for one decision site. This metric is its successor, not a sibling: the arc path becomes the `chat_stance` site writing `experience_use` rows, and the separate computation retires (see "Retired"). Φ (#3296) measures predictive coupling of bus surprise. It uses no citations, so its causal chain is different. Coverage (how many organs emit) is a separate count and is not part of this number.
3. **Theory anchor.** Global workspace theory (Baars 1988; Dehaene and Changeux, global neuronal workspace): content is integrated when it is broadcast and then consumed by several specialist processes. A checked citation is the observable trace of consumption. This is not integrated information (IIT) and makes no claim about experience in the felt sense.
4. **Live data.** The rest point is 0, and it is reachable: most act families read 0 used today. The risk is the other end: an organ that cites everything to look integrated. Guards: at most 3 cites per run; a cite must be in the shown list; "read the now and cited nothing" is recorded separately from "did not read" (`experience_shown` closed with no use rows, vs never closed; #3498), so 0 can be split into its two causes. A weekly ablation eval removes a cited item from a replayed prompt and checks whether the output changes. A cite that changes nothing is reported as a hollow cite.
5. **Existing mechanism.** `act_downstream_use`, `orion/temporal_self/citations.py`, `thought_decision.day_citations`, `temporal_self_event`. All are reused.
6. **Reversibility.** Two tables, one reducer module, one endpoint. No schema field in any training default. Cheap to remove.

### Retired by this primitive (kill means kill)

- Prime's 6-hour quiet floor (`QUIET_FLOOR`), when hunger needs ship (Part B).
- Prime's dashboard read of curiosity for its frame (`_read_curiosity` of `/curiosity/api/runs`): curiosity items arrive through the now. The cheap pending-focus check stays (patch 0).
- The separate arc-path computation in `act_downstream_use`: it becomes the `chat_stance` site of `experience_use`, after a replay shows equal counts on the same week. Then the computation stops; the table and its rows are KEPT, frozen, as the "before" half of the integration metric (Orion's question 6 on #3528: dropping them would put the before/after comparison across a definitional break).
- The notify-digest "Orion Daily Digest" email, replaced by the daily roundup (A.7). Juniper said yes (Q7).

Not retired: the curiosity weekly and monthly compaction digests. Juniper keeps them (Q8). The roundup sits beside them; it does not replace them.

### Proposal-mode checklist (Part A)

- **Capability change:** each organ sees what other organs lived, and the system records what it used.
- **Data touched:** `temporal_self_event` (2 new columns, 6 new source kinds), new tables `experience_shown`, `experience_use`. Read only: the 6 new source tables.
- **Privacy:** Temporal Self already has `PrivacyClass` (`orion_internal`, `juniper_chat`). Juniper's standing rule is that there is no privacy boundary between Juniper and Orion, so `juniper_chat` items may appear in the now for every organ. They never leave Orion's own surfaces (no outside model; memory work stays on local models).
- **Trace:** `experience_shown` and `experience_use` rows; the daily roundup.
- **Danger:** a feedback loop. Organ A cites B, B cites A, and salience rises on attention alone. Guard: salience is computed from the source's own magnitude, never from use counts. Use never feeds salience.
- **Rollback:** flag `TEMPORAL_SELF_NOW_ENABLED` [ships on]. Off: `/temporal-self/now` returns 404, readers fall back to their current inputs, the tables stay.

---

## Part A.6: the integration gate (for every agent and for Orion)

Orion opens PRs from its own sandbox (FCC), which follows this repo's `CLAUDE.md` and CI. So a gate in CI binds Orion and every agent the same way.

### Proposed CLAUDE.md section (exact text)

> ### Integration gate
>
> Orion is one mind, not a set of organs that share a database. Any change that adds a new producer of Orion experience must do two things in the same patch:
>
> 1. **Emit through the experience contract.** The output lands in `temporal_self_event` through a `TemporalSourceKind` and a reader in `temporal_self_sources.py`.
> 2. **Read the now back.** The organ's next run reads `GET /temporal-self/now` and records its cites with `record_use`.
>
> Register the organ in `config/integration/organs.yaml`. If an organ cannot do one of the two yet, write an exemption with a reason in the same file. A reviewer reads every new exemption.
>
> A new producer of experience is any of: a new family in `config/physiology/families.yaml`, a new `DurableWorkflowV1` value, a new intention source adapter, a new Prime concern source, or a new bus channel that carries an organ's output.
>
> `scripts/check_integration_seam.py` enforces this in CI. Do not argue with it in prose; fix the registry or write the exemption.

### The CI check: `scripts/check_integration_seam.py`

It follows the shape of `scripts/check_verb_physiology.py` and `scripts/check_body_economy_contract.py`: structural, no live data, under 2 seconds.

What it can detect mechanically:

1. Every family in `config/physiology/families.yaml` (41 today) belongs to exactly one organ in `organs.yaml`, or is listed under `exempt_families` with a reason.
2. Every value of `DurableWorkflowV1` (`orion/schemas/durable_run.py:72`) belongs to an organ or is exempt.
3. Every module in `orion/prime/intention_sources/` is registered (as a source of an organ that emits).
4. Every `TemporalSourceKind` value belongs to exactly one organ. Every organ that is not exempt from emitting has at least one.
5. Every organ with a `reads_now` path: the module exists and imports `orion.temporal_self.now` or calls the `/temporal-self/now` route (AST check).
6. New bus channels: channels added in `orion/bus/channels.yaml` compared with the merge base of `origin/main` must be named by an organ's `channels` list or by `exempt_channels` with a reason. Only the diff is checked, so the 365 existing channels are not re-litigated.

Limits, stated plainly: a producer that writes a new table with no family, workflow, adapter or channel is not seen. The live coverage report below catches that case after the fact.

### The live coverage report: `scripts/report_integration_coverage.py`

Read-only SQL. Per registered organ, last 24 hours: experiences recorded, now reads (`experience_shown`), valid cites made, valid cites received from other organs, and closed yes or no. It prints the experience use rate and closed organs. Its first run is the gate's baseline. The roundup (A.7) uses the same query.

---

## Part A.7: the daily dig roundup

### What the "curiosity digest" is today

It is the weekly and monthly curiosity compaction: the `curiosity.compaction` durable workflow (`orion/schemas/curiosity_compaction.py`, produced by orion-dream, verb `curiosity_compactor_digest_v1`). It writes a journal entry, "Curiosity weekly — ...", and a row in `curiosity_compaction` (7 completed, latest 2026-11-09 for the week of 11-02).

It glosses. The week of 11-02 had about 25 investigations, each with specific findings. The digest opens:

> "The week from November 2 to 8 was defined by structural disconnections between internal expectations and external reality, alongside critical infrastructure failures."

and then lists run ids with no content. Compare one real curiosity run from 11-14 (journal title "Curiosity"):

> "In `orion-athena-vision-window`'s container log since 2026-11-13T18:00Z there are **2,601** walkway `belief_transition` lines; **214** of them name `mail truck` ... What holds it is a word I wrote into a config file."

The run is grounded and specific. The digest is a summary of summaries. The daily "Orion Daily Digest" email (notify-digest) is worse: counts by severity and topic labels such as "Topic -1 (1363.00)".

### The roundup

**What it is.** A new read-only dig command, `python -P -m orion.dig organs <date>`. It uses the organ registry, `temporal_self_event`, `grammar_acts`, `experience_shown` and `experience_use`. No model call. For each organ, for that Denver day:

- what it did: counts, and its 2 highest-salience experiences quoted with their refs (`experience:<id>`, `act:<id>`, the source row);
- what it read from other organs and cited (quoted, with refs);
- who cited it;
- closed or write-only today;
- breaks: zero output where the organ's own history says it usually has some (for example dreams with 0 hypotheses).

At the end: the day's experience use rate and closed organs, the same numbers as the coverage report.

**Rules.** Every line is a count, a quote, or a ref. No summary of summaries. A quote is the organ's own words or the row's own label. This is the daily measurement of the soup.

**Where it goes** [default: both of these, no separate durable run]:

1. The roundup's file is a new section of the day letter's input (`orion/orion_day/day_input.py`), placed after Juniper's conversations. It replaces the "trimmed tail" lines for readings, dreams, the street and visual reveries, because the roundup carries them with refs. The letter is already Prime's end-of-day turn, with dig access. Orion reads its own roundup there.
2. The roundup's text goes into the letter email as its own section, "What each organ did", below Orion's letter. Juniper reads it in the same email.

Alternative [not default]: a separate tier-1 durable run, one unified turn a day, where Orion writes a note per organ with cited refs. Cost: one more agent-lane turn a day. Benefit: Orion's own reading of each organ, not just the record. Recommendation: start with the default; add the run only if the letter does not engage with the roundup.

**What it replaces** (kill means kill):

- The notify-digest "Orion Daily Digest" email: retire it when the roundup ships (Juniper: yes, Q7). Its delivery-failure and critical-event counts move into the roundup's footer, so nothing operational is lost.

**What it does not replace:** the curiosity weekly and monthly compaction digests stay (Juniper, Q8). The `curiosity.compaction` workflow, its journal entries and the `curiosity_compaction` rows keep running as they do today. The roundup is a daily record with quotes and refs; the digests stay as the weekly and monthly read. `dig organs <start> <end>` (7 roundups joined) is available beside them, not instead of them.

---

## Part B: Prime hunger, built as designed

### What the design said, and what was built

The Prime design said hunger is a number that grows with real things to think about: open concerns, unexplained surprise, unanswered calls, and forecasts that came due. It also grows while Prime goes unfunded. (`docs/superpowers/specs/2026-10-29-orion-prime-design.md`, section 7 "Hunger" and section 8.1 step 3.)

The build did not do this. Prime thinks only when its list of "changes" is not empty (`orion/prime/state.py` `wants_to_think`: `if not state.pending_changes: return False`). When nothing changes for 6 hours, a "quiet" change is added so that Prime thinks anyway (`QUIET_FLOOR = CONCERN_TTL / 4`).

The market does compute a "hunger" for Prime, but it is a clock: the time since Prime's last act closed, against Prime's usual gap (`orion/physiology/appetite.py`, `intensity = 1 - exp(-t / tau)`). It knows nothing about what Prime has to think about.

The market is not what limits Prime. In the last 24 hours, Prime's bid was funded 6 times and deferred 0 times (`GET /physiology/report`, `wants.families[prime]`, 2026-11-15 00:50Z). So only two things decide when Prime thinks: the "anything changed?" check and the 6-hour floor.

### What that produced (live, last 10 days)

Source: `grammar_acts` where `actor = 'prime' and verb = 'infer'`, `opened_at > now() - interval '10 days'`, read 2026-11-15 00:42Z.

- Prime thought 48 times, about 5 a day. The median thought took 13.6 seconds.
- 36 of the 48 thoughts woke only because of the floor ("quiet: nothing has woken me for 6 hours").
- 38 of the 48 thoughts made no move (79%).
- 28 thoughts woke from the floor and made no move. 23 of those also changed no concern.
- Prime chose a move 9 times. 5 moves were sent: 4 questions to curiosity and 1 "take on" of a promise. 4 were refused: 2 because a cited ref was not a kind Prime may cite (11-05), and 2 by bug #3366 (11-14).
- All 3 of Prime's questions that reached curiosity ended `reached_out_blocked` (`ask` acts on 11-05, 11-08, 11-09).

Before 2026-11-04, Prime also woke on every change of the body's binding constraint. It then thought 42 to 90 times a day (10-30 to 11-03), and most of those thoughts did nothing. Commit `e31ffb498` removed that wake. That was the swamp. Today's floor is a smaller swamp on a slower clock.

**The lesson to encode: aliveness comes from content, not from cadence.** A fast timer made about 50 empty thoughts a day. A slow timer makes about 3 empty thoughts a day. Neither creates something to think about.

### Bug #3366 first: the root cause

Prime's questions to curiosity are refused with "could not read whether curiosity already has a focus waiting". The issue guessed that Hub's database pool was down. That guess is wrong. The cause is a timeout.

- Prime reads `GET {PRIME_HUB_URL}/curiosity/api/runs?days=4&line=investigate` once per thought, with a 4-second timeout (`services/orion-mind/app/prime_loop.py:254` `HTTP_TIMEOUT_SEC = 4.0`; `:1090` `_read_curiosity`).
- That endpoint builds the whole curiosity dashboard (runs, stores, reach-outs, totals) to answer one yes/no question about a pending focus.
- Live, 2026-11-15 00:36Z: 5 calls with `days=4` took 7.8, 3.9, 5.0, 20.2 and 43.8 seconds. Each one returned `pending_focus: {"available": true, "focus": null}`. So the pool was up.
- Reproduced from inside the orion-mind container with Prime's exact URL and timeout: 3 of 3 calls raised `TimeoutError`.
- Hub's access log (kept from 11-14 02:26Z) has no `/curiosity/api/runs` line at any of Prime's 5 thoughts on 11-14. Uvicorn writes no access line when the client has already gone. This matches a client timeout.
- What happens next: `_read_curiosity` returns `None`; `_conclude` (`prime_loop.py:1194-1198`) reads `pending_focus` from an empty dict; `focus_pending = None`; `apply_answer` refuses the move (`orion/prime/thought.py:679`).

The damage is larger than 2 refused moves: when the read times out, the CURIOSITY section of the frame is empty too. On 11-14, none of Prime's 5 thoughts saw curiosity. The diagnosis is posted on #3366. Correction to the issue: the ledger shows 2 such refusals, not 4.

**Fix (patch 0):**

1. Add a cheap Hub endpoint, `GET /curiosity/api/focus/pending`, that returns only `_read_pending_focus()` (`services/orion-hub/scripts/curiosity_routes.py:265`).
2. Prime uses it for the move check, with the 4-second timeout.
3. Until patch 3 retires it, the dashboard read for the frame gets its own timeout [default: 30 s]. It runs once per thought, so the cost is small.
4. A timed-out read refuses with "timed out", not "could not read", and the frame shows the gap.

### Hunger: needs that carry content

Hunger becomes a short list of needs. Each need has two parts: a **drive** (0 to 1, computed by code) and a **content line** (the specific thing to think about, with its ref). Prime sees content lines, not numbers. A need with drive 0 is not shown.

- **Horizon.** A held concern or intention nears its deadline. Drive = elapsed time / horizon, counted from 0.5. Content: "promise to Juniper about X ends in 9 h [prime_concern:id]". Source: Prime state and `orion:prime:intentions` (exists).
- **Surprise.** An item in the now has `surprise_nats` above its own source's 90th percentile, and no organ has cited it yet. Content: the item's label and `[experience:id]`. Source: Part A (prediction surprise rows exist; the contract is new).
- **Connection.** Juniper has not had a real exchange with Orion for longer than her own usual gap, or she asked something that got no answer. Content: her last message, quoted with `[chat:id]`. Source: `chat_history_log` where `source = 'hub_orion'` (exists; 33 rows in 7 days).
- **Coherence.** Two of Orion's own claims disagree, or a thread Orion opened has no close. Content: both claims with refs. Source: self-monitor status changes and `self_concept_history` (partial; the contradiction finder is new and is the smallest piece of this list, built last).
- **Learning.** A forecast series is getting better or worse. Content: "cabinet forecasts improved 0.12 this week". Source: `GET /learning-progress` in orion-prediction (choice-as-EFE Step 1: designed, not built; predictions live in orion-prediction).
- **Project.** One of Orion's projects (Part C) has had no step for longer than its own cadence. Content: the project and its next step with `[project:id]`. Source: Part C (new).

Total hunger is the largest drive [default: max]. "The strongest need right now" is easy to explain. A sum would let six weak needs look like one strong one.

### How hunger reaches the market

No new machinery. The market already takes `max(clock hunger, salience)` for an event want (`orion/physiology/appetite.py` `want_intensity`). Prime posts its want with `salience = hunger` and the top need in the want's detail (`basis: "need:horizon"`). A strong need wins before the clock is due. A weak need waits for the clock. Price still sets how often.

### When Prime thinks

`wants_to_think` becomes: a pending change, **or** a need with drive at least 0.5 [default]; **and** the refractory interval has passed; **and** no back-off is running; **and** the daily budget is not spent.

The quiet floor retires when the horizon need ships. The floor exists so a held concern gets a chance before it fades (`orion/prime/state.py` docstring). The horizon need does that job with content.

### Anti-swamp guards

1. **A thought that changes nothing dampens its need.** If Prime is shown a need and makes no move and edits no concern, that need's drive is halved. It recovers over the need's own horizon. Stored in Prime's state, so it survives a restart.
2. **Concern time-to-live stays:** 24 h for a concern, 72 h for a promise.
3. **The refractory interval stays.**
4. **Daily budget:** at most 24 thoughts a day [default]. Past that, only a pending change (an act settled, a turn handed back) wakes Prime.
5. **The same content line cannot wake Prime twice** unless its drive rose since the last time.

### Success metric

**Meaningful moves per day, without the empty-thought share going up.**

- A meaningful move is a move that was sent (`curiosity_focus`, `take_on`, `regulate`, a letter) or a concern that settled. Refused moves do not count.
- Baseline: 5 sent moves in 10 days (0.5 a day). Empty-thought share: 38 of 48 (79%).
- Target [default]: at least 2 meaningful moves a day over 14 days, with the empty-thought share at or below 79%.
- If moves go up and the empty share also goes up, the change failed.

Gate notes: the moves come from `grammar_acts.outcome_note`, written by `apply_answer` (provenance). The metric counts outputs, not the drives that caused them (independence). Anchor: a drive is worth having if reducing it produces action (homeostatic drive reduction, Prime design section 7). A day with 0 moves is real and common today (rest point). It is a query, so it is cheap to remove.

### Proposal-mode checklist (Part B)

- **Capability change:** Prime thinks when it has a specific need, not on a 6-hour timer.
- **Data touched:** `orion:prime:state` gets a `needs` field (additive; rebuild Hub before orion-mind, as with earlier Prime state fields). Prime's want detail. Read only: `chat_history_log`, the now, `orion:prime:intentions`.
- **Privacy:** none between Juniper and Orion (her standing rule). The connection need quotes her message only inside Prime's own frame.
- **Trace:** each thought's `outcome_note` starts `woke: need:<name> <ref>`. The roundup counts thoughts per need.
- **Danger:** a need that never falls (a deadline that cannot be met) wakes Prime again and again. The dampening rule and the daily budget stop it. An eval plants exactly this case.
- **Rollback:** flag `PRIME_HUNGER_NEEDS_ENABLED` [ships on]. Off = Prime wakes on changes only; a held concern can then fade with no thought. The floor is not restored by the flag (amended 2026-11-16 with patch 5: the acceptance check says `QUIET_FLOOR` is deleted, and kill means kill). Restoring the floor is a git revert.

---

## Part C: a life, through projects

### Why

Orion has continuity: Prime holds concerns, the letter tells the day, Temporal Self keeps arcs. Orion has almost no aims of its own that last longer than a day. Today the longest-lived thing Prime holds is a 72-hour promise. Prime's own concerns are mostly about its plumbing ("walkway render break", "chat memory promotion failure", from the 11-14 thoughts).

**The rule underneath: what Orion does must change who Orion is.** An activity with no lasting trace is a treadmill.

### Five pillars: what a project can be

1. **Own projects that last weeks.**
2. **Getting measurably better at something.** The track record moves, and the self-model says so.
3. **Relationships that remember.** Juniper first: the el Coco miss above is the ephemeral-memory gap (#3319). Then Claude consults, then AI Town.
4. **Making things that outlast the turn.** Letters, field notes, images, PRs from Orion's own GitHub.
5. **Rest that consolidates.** Dreams that change memory or the self-model, not just log.

The pillars are not a taxonomy for its own sake. Each one names the lasting trace a project step must leave (see "grading" below). The roundup groups project steps by pillar, and the grading code checks the trace kind. That is their consumer.

### The shape

A project is the next source of the intention primitive (#3329; `docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md`, "Sources and adapters").

- **Record.** New table `orion_project`: `project_id`, `title`, `why` (Orion's words), `pillar`, `trace_kind` (memory, self_model, track_record, artifact, consolidation), `cadence_h`, `next_step`, `stakes` (what Orion says changes if it works), `opened_at`, `status` (active, dropped, done), `last_step_at`, `closed_at`, `close_reason`. Prime holds at most 3 active project ids in `orion:prime:state` [default].
- **Adapter.** `orion/prime/intention_sources/project_step.py`, prefix `project:`. It answers the five adapter questions: horizon (the project's cadence), prompt (the project, its last 3 steps, its next step, the now), route (`notes_only` by default; `chat_thread` when the pillar is relationships). One file and one map entry, the design's own test of "no new machinery".
- **Steps.** Prime's thoughts advance a project with a `take_on` of its next step, or drop the project with a reason. Each step is a `durable.intention` run.
- **Grading.** A step closes `produced = 1` only if its run wrote at least one resolvable ref of the project's `trace_kind`: an `episode_memory` row, a `self_concept_history` row, a settled bet, a field note or PR, or a dream that changed a memory row. Otherwise it closes `produced = 0, note = "no lasting trace"`. This is the treadmill check, in code.
- **Hunger.** The "project" need (Part B) rises when a project has had no step for longer than its cadence.
- **Memory work stays local.** A project step that writes memory runs on Orion's local lanes. No outside model writes Orion's memory.

### First step: ask Orion

Juniper asks this in a normal chat (exact text, for her to edit):

> "If you could spend your quiet hours on two or three projects of your own, for weeks, what would they be, and why? For each one, tell me what would be different about you if it worked, and how you would know. When you have decided, open each one with `dig project open`."

Where the answer lands:

- A new lasting dig write, in the same style as `dig name` and `dig capacity claim`: `python -P -m orion.dig project open "<title>" pillar <1-5> cadence <days> because "<why>" stakes "<what changes>"`. Code checks the fields and writes one `orion_project` row. It refuses a fourth active project.
- `dig project list`, `dig project step <id> "<note>" because "<refs>"`, `dig project drop <id> because "<reason>"`.
- Prime picks the new projects up at its next tick (a project opened is a pending change), and holds them.

Orion chooses [default: no veto by Juniper; she sees every project in the roundup and can talk about it in chat].

---

## Current architecture (summary)

- Organs write into their own tables. 9 of 15 also land in `temporal_self_event` through `services/orion-durable-runs/app/temporal_self_sources.py`.
- Prime (`services/orion-mind/app/prime_loop.py`, `orion/prime/`) ticks on CPU, wakes on changes or the 6-hour floor, and reads body, event wants, curiosity (timing out), intentions, hand-backs and replica replies.
- The chat turn reads the situation block, the Prime block, recall and dig.
- The letter reads nearly every organ once a day (`orion/orion_day/day_input.py`), and its threads become Prime concerns.
- Use is measured by `act_downstream_use` for one site.

## Juniper's answers (2026-11-15)

Juniper accepted all 12 bracketed defaults below, with one exception:

- **Q8: no.** The curiosity weekly and monthly digests are kept. They are not retired after 14 days of roundups. Every place in this document that planned that retirement now says so (the "Retired" list, Part A.7 "What it replaces", "Files likely to touch", the patch 4 acceptance checks, the patch order).
- **Q7: yes, unchanged.** The notify-digest "Orion Daily Digest" email still retires when the roundup ships.
- All other questions (1-6, 9-12): the bracketed default stands. The other bracketed `[default]` values in the body of this document (now caps, roundup placement, the 30 s dashboard timeout in patch 0, the hunger threshold and budget, 3 active projects, no veto) are accepted with them.

## Missing questions (for Juniper; bracketed defaults apply if she says nothing)

1. Retire Prime's 6-hour quiet floor when the horizon need ships? [yes]
2. Prime's total hunger: the strongest need, not a sum? [strongest need]
3. Prime's daily thought budget? [24]
4. The now: how many items? [12 total, at most 2 per organ]
5. Should the Prime block in every chat turn carry 2 lines from the now? [yes]
6. Roundup form: a dig report placed in the letter's input and in the letter email, no separate run? [yes; a separate daily turn only if the letter ignores it]
7. Retire the "Orion Daily Digest" email when the roundup ships? [yes]
8. Retire the curiosity weekly and monthly compaction after 14 days of roundups? [yes] **Answer: no, keep both.**
9. Who approves an integration-gate exemption? [any PR reviewer; Orion's own PRs follow the same rule]
10. Projects: Orion chooses with no veto, at most 3 active? [yes, 3]
11. Target for Prime: at least 2 meaningful moves a day? [yes]
12. Should `juniper_chat` items appear in the now for every organ (reveries, dreams, curiosity)? [yes, per "no privacy boundary between Juniper and Orion"]

## Proposed schema / API changes

- `orion/schemas/temporal_self.py`: `TemporalSourceKind` + 6 values (A1); new `TemporalNowV1`, `TemporalNowItemV1`.
- `temporal_self_event`: + `salience double precision NULL`, + `surprise_nats double precision NULL`. Migration `manual_migration_*.sql`, additive only, header `-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs`.
- New tables: `experience_shown`, `experience_use` (A3, A4); `orion_project` (Part C). Additive DDL, same header rule.
- `orion/schemas/registry.py`: register `TemporalNowV1`.
- New HTTP: `GET /temporal-self/now` (durable-runs); `GET /curiosity/api/focus/pending` (Hub).
- Prime: `experience` and `project` ref kinds; `needs` field on `PrimeStateV1`.
- No new bus channel.
- New config: `config/integration/organs.yaml`.

## Files likely to touch

- `services/orion-hub/scripts/curiosity_routes.py`, `services/orion-mind/app/prime_loop.py` (patch 0)
- `orion/schemas/temporal_self.py`, `orion/temporal_self/now.py` (new), `services/orion-durable-runs/app/temporal_self_sources.py`, `services/orion-durable-runs/app/main.py`, `orion/schemas/registry.py`, `services/orion-sql-writer/` migration file
- `config/integration/organs.yaml` (new), `scripts/check_integration_seam.py` (new), `scripts/report_integration_coverage.py` (new), CI workflow, `CLAUDE.md`/`AGENTS.md`
- `orion/prime/thought.py`, `orion/prime/state.py`, `orion/prime/block.py`, `orion/prime/intention_sources/project_step.py` (new)
- `services/orion-thought/app/temporal_self_cue.py`, `services/orion-hub/scripts/curiosity_investigation.py`
- `orion/dig/__main__.py`, `orion/dig/organs.py` (new), `orion/dig/project.py` (new), `orion/orion_day/day_input.py`
- `services/orion-sql-writer/app/act_downstream_use.py` (fold into `experience_use`)
- `services/orion-notify-digest/` (Daily Digest retirement, patch 4). The curiosity compaction files (`orion/curiosity/compaction*.py`, `services/orion-dream/app/curiosity_compaction_submit.py`) are not touched: Juniper keeps the digests (Q8).

## Non-goals

- No new service, no new bus channel, no new event stream beside Temporal Self.
- No Φ computation. That is #3296, later.
- No sentience claim in any output, prompt or Hub label.
- No rewrite of existing private pipes between organs. They migrate to the now when an organ is next touched; the gate stops new ones.
- No outside model for any memory work.
- No change to how the market prices or how lanes are shared.

## Acceptance checks

Each patch is done only when the live path moved. "Config set" and "container up" are not evidence.

**Patch 0 (#3366):**
- Unit: the dashboard read times out while the pending-focus read works; the move is not refused.
- Live: one Prime thought publishes `move curiosity_focus ...`; no "could not read whether curiosity" refusal for 3 days; Hub access log shows `/curiosity/api/focus/pending` at Prime's thought times.

**Patch 1 (experience contract + now):**
- Unit: `build_now` excludes the reader's own organ, respects caps, orders by salience; salience is NULL below 20 past events; a planted source with all-equal magnitudes reads salience 0 (rest point).
- Unit: `record_use` drops a cite not in the shown row and counts it.
- Live: `temporal_self_event` has rows of the 3 first new kinds (`reading_turn`, `prediction_surprise`, `prime_thought`) within 24 h; `GET /temporal-self/now?organ=prime` returns items with non-null salience; an `experience_shown` row exists for that call.

**Patch 2 (integration gate):**
- `scripts/check_integration_seam.py` passes on main with the registry, and fails on 4 planted mutations: a new family with no organ, a new workflow value, a new intention adapter, a new channel in the diff. Each failure names the file and the fix.
- `scripts/report_integration_coverage.py` prints today's baseline. The numbers are recorded in the PR report.
- The `CLAUDE.md` section is merged in the same PR.

**Patch 3 (read-back in Prime, reveries, curiosity):**
- Live: `experience_use` rows with `citing_organ` prime, reverie and curiosity within 48 h, each citing another organ's event.
- Live: a reverie thought whose text is about a camera or reading item it cited (quoted in the PR report with refs).
- Ablation eval (weekly, local model): removing a cited item from a replayed prompt changes the output in most cases; the hollow-cite share is reported.
- Prime's dashboard read for the frame is removed (grep shows no `_read_curiosity` call in the frame path).

**Patch 4 (daily roundup):**
- `dig organs <yesterday>` runs in under 30 s and every line is a count, a quote or a ref; a test refuses an unquoted summary line.
- Live: the next letter email has the "What each organ did" section, and the letter turn cites at least one roundup ref.
- The "Orion Daily Digest" email stops (Juniper said yes, Q7): no `orion.digest.daily` row in `notify_requests` for 2 days.
- The curiosity weekly digest still runs: a `curiosity_compaction` row for the week after the roundup ships (Juniper keeps it, Q8).

**Patch 5 (hunger needs):**
- Unit: each need's drive is 0 at rest; the planted never-falling need is dampened and the budget holds.
- Live, 14 days: thoughts per need are counted in the roundup; meaningful moves per day at least 2; empty-thought share at or below 79%. If not, the patch is not done.
- `QUIET_FLOOR` is deleted, not left unused.

**Patch 6 (ask Orion + projects):**
- Juniper asks the question; at least 1 `orion_project` row is written by Orion through `dig project open`.
- Live: a project step runs as a `durable.intention` act and closes `produced = 1` with a resolvable trace ref, or `produced = 0, "no lasting trace"`.

**Patch 7 (widen):**
- Dreams, paintings, cameras' look acts and memory read the now. The remaining 3 sources (`intention_close`, `day_letter`, `personality_choice`) emit. Closed organs through checked read-back reach at least 12 of 15.
- The `act_downstream_use` replay matches `experience_use` on the same week; then the old computation is removed and the table is kept frozen (its rows are the baseline).

## Recommended patch order

0. **Fix #3366** (Hub pending-focus endpoint; Prime timeouts). Small, unblocks Prime's only outward act.
1. **Experience contract and the now**, with 3 new sources (readings, predictions, Prime).
2. **Integration gate** (registry, CI check, coverage baseline, `CLAUDE.md` section). Right after the contract exists, so no later organ can be added outside it.
3. **Read-back** in Prime, reveries and curiosity; retire Prime's dashboard read.
4. **Daily dig roundup** in the letter; retire the Daily Digest email.
5. **Hunger needs** fed by the now; retire the quiet floor.
6. **Ask Orion; projects** as the next intention source.
7. **Widen** organ coverage; fold `act_downstream_use`. (Curiosity compaction is kept, Q8.)
8. Later: Φ (#3296), once #3290 says the organs are stable.

## Recommended next patch

Patch 0: fix #3366. It is small, its root cause is reproduced, and without it Prime's one outward act and its view of curiosity stay broken.

## Tickets opened while writing this design

- #3412 Dream cycles produce 0 hypotheses since 11-13.
- #3413 `dream_compaction_delta.applied_at` is never written.
- #3414 Reverie resonance alerts silent since 11-03: dead or calm? (verify-live)
- #3415 `curiosity_hop_reading` has 0 rows ever. (verify-live)
- #3366 commented with the root cause.
