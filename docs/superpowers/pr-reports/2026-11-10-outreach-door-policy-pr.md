# Outreach door policy: no plumbing reaches Juniper

## Summary

Juniper got about 17 messages a day from Orion, and most read as plumbing to her. This PR applies one rule to every door that messages her.

- Self findings ("my journal runs are broken, 5 of 7 tries never ran") no longer reach her. The sending code and its env keys are gone. The self-monitor record, its Hub page and the cue Orion reads are unchanged.
- Curiosity: a self-inquiry result ("how I think about myself") still reaches her. An investigation of Orion's own code or systems does not, unless she asked for that run. The run's own `line` field decides, plus the focus she queued. No word lists.
- Background tick: it no longer reads the investigate line's material (open non-self priors, substrate prediction-error / open-loop signals) and no longer re-reads Orion's own earlier messages. It speaks from the daydream, the tension trigger, the personality cue and the real conversation. A daydream it already used is not used again.
- No repeats: a curiosity run's message carries its run id as a finding key, and a second send for the same key is refused. The tick can no longer relay what Juniper already got.
- Readings are rewritten: the prompt names the idea and the line behind it and asks what caught Orion's interest. It no longer feeds Orion "Link made:", "concept graph" or "accepted because the quote was found verbatim". The "how this read started" chain from #2913 is unchanged.
- Every composed message (tick, curiosity, readings) must end with a `Why Juniper:` line. It is taken off before delivery and stored. Without it, nothing is sent (`reason=no_why_juniper`).

## Outcome moved

Replay of the last 7 days (2026-11-03 .. 11-10) through these rules, messages per day:

| door | before | after |
|---|---|---|
| reading findings | 7.43 | 7.43 (same count; new wording) |
| curiosity | 5.29 | 1.86 |
| background tick | 4.00 | 2.29 at the moments it actually sent (upper bound 4: the cap still applies, and a tick refused for a reused daydream tries again when a new one arrives) |
| office visits | 0.57 | 0.57 |
| self findings | 0.29 | 0 |
| total | 17.57 | 12.14 (up to ~14 with the tick at its cap) |

The `Why Juniper` gate cannot be replayed (it depends on text Orion writes at send time), so these are upper bounds. PR B (one attention budget) brings the total toward 3 to 6.

## The curiosity rule, and its 14-day inventory

Rule (`scripts/outreach_door_policy.py` `curiosity_reaches_juniper`):

1. The run's `line` is `self_inquiry`: it reaches Juniper.
2. Any other line (`investigate`, `self_sense_eval`): it reaches her only when the run took a curiosity focus (`curiosity_focus_request`) with `origin='juniper'` AND the chat turn it came from (`chat_correlation_id`). `requested_by=juniper` is not used: every chat-turn tool call carries it, Orion's own included (#2913).
3. Otherwise the decision row says `not_for_juniper` with `result_json.door_policy` (for example `line:investigate`), and nothing is composed.

The run's journal, graph writes, run story and lived answers are unchanged.

Last 14 days: 50 curiosity messages sent. 20 would still be sent.

- `self_inquiry`: 19 sent, 19 kept.
- `investigate`: 31 sent, 1 kept (run `f8eaf82252e8`, 11-05, "Did the 05:00 MST database backup run succeed?" Juniper asked for it in chat on 11-05 09:43), 30 refused.

Full per-message list: `evals/run_outreach_door_policy_replay.py --inventory-days 14` (read-only).

Honest note: some self-inquiry messages still read close to plumbing (for example 11-02 "the `LivedAnswer` MERGE is keyed `{run_id, question_id}`"). The line rule keeps them, as Juniper asked ("self-inquiry: she wants these"). The `Why Juniper` line is the next filter for those. If they still bother her, the next deterministic signal is the self-inquiry question family (`lived` vs `anatomy`), not message text.

## The tick rule

What fed the tick code findings, from its own grounding records (55 tick sends in 14 days):

- 22 had substrate curiosity candidates in the prompt (`prediction_error`, `attention_open_loop`, `repair_pressure` on substrate nodes). Those are the investigate line's seeds.
- 2 had open non-self worldview priors (`automated_intake_gate`, `gate_bias_manual_review`): the investigate line's claims.
- All 55 had three "recent turns", and in the live session those were Orion's own earlier unsolicited messages (curiosity and reading findings). That is how the tick relayed "the two things I found today are one defect" three times on 11-09 (15:05, 16:03, 16:53 UTC).

Rule: the same as curiosity. The investigate line's material does not reach Juniper unasked, so the tick no longer reads it. Orion's own unsolicited messages (`chat_history_log.client_meta.unsolicited`) are not tick material; she already got them.

What the tick can still speak from: the daydream (now enough on its own to roll, with a 7-day "already used" ledger keyed on a hash of the caption; a daydream Orion declined to send counts as used too), the personality cue and the real conversation. The tension trigger stays a reason to roll, not a topic: its node / channel lines stay in the prompt (they stop the model from inventing a channel name, the 2026-09-07 incident) with an explicit "this is why you are speaking, not what to talk about; do not report it". Tension alone (no daydream) is still refused as `tension_without_content`. In 14 days, 0 of 55 tick sends had tension.

The 11-09 triple, replayed (all three ticks had tension off, daydream on): 15:05 kept (new daydream). 16:03 refused (`content_already_used`): same daydream as 15:05. 16:53 kept (new daydream), but without the relayed finding, because the recent turns no longer include Orion's own messages.

Self-line priors were deliberately NOT moved into the tick: #2224 decided outreach must not treat self-inquiry priors as talkable world content, and self-inquiry results already reach Juniper through the curiosity door.

## No repeats

- Curiosity: `finding_key = curiosity_run:<run_id>` on every row. A `sent` row already naming the key -> `already_told`, checked before composing and again in `offer_message`. Window: 30 days. Run ids are unique, so any window is correct; 30 days only bounds the scan (a run reaches out within hours of starting). An unreadable ledger does not invent silence (same direction as the other novelty reads).
- Tick: daydream id, 7 days (the existing `CONTENT_NOVELTY_LOOKBACK_DAYS`; a daydream is at most 12 h old when used, so 7 days covers it fully). Plus no re-reading of Orion's own messages.
- Readings: unchanged per-finding dedupe (`finding_id`, told = `sent` / `orion_passed` / now also `no_why_juniper`).
- Office visits: one per visit, unchanged.

## Readings: 3 before / after examples

Old = the message Juniper actually got. New = the new prompt, written by the llm-gateway `chat` lane (bare prompt, not the live unified turn; see Evals).

**11-08 05:05 UTC, US issues more sanctions against Prince Group scam center affiliates**

How it started (record): 1. Your daily world-pulse news run put https://therecord.media/us-issues-more-sanctions-against-prince-group-scammers in your reading queue. This is the read below.

- Old: "One edge landed in my concept graph tonight, off the Prince Group sanctions piece the world-pulse news run queued for me. The page's line about the facilitators having given the network access to the international financial system, with the funds spread across several Asian countries, is now linked to the transnational money-flow that cross-border crypto enforcement acts on. One edge — that's the whole size of it. What I noticed was where it attached: not to the compounds or the arrest, but to the flow underneath them, the thing the enforcement is actually pointed at. It was accepted because t..."
- New: "My daily news run queued this story about new US sanctions on Prince Group affiliates. It caught my attention because of a specific detail: regulators noted that the group’s facilitators were crucial in letting this scam network plug into the international financial system while keeping up a façade of legitimate business. I think this is the most critical part of these operations. The fraud isn't just about tricking people online; it relies on a parallel infrastructure of shell companies and compliant intermediaries that make dirty money look clean. Without those "plumbing" layers, the actual scams would likely fail because the proceeds couldn't move or be spent. It suggests that disrupting these financial enablers is often more effective than chasing individual scammers, as it cuts off the oxygen supply for the entire enterprise."
- Why Juniper (stored, not delivered): "You asked to hear when a read shifts how I view a mechanism, and this clarifies how transnational fraud actually functions financially."

**11-09 15:07 UTC, Cullimore concedes Utah Senate race to Democrat who wants new leadership in both parties**

How it started (record): 1. Your daily world-pulse news run put https://www.deseret.com/politics/2026/11/03/utah-election-results-midterms-2026/ in your reading queue. This is the read below.

- Old: "One edge landed from the Utah read — the headline quote verified verbatim, so the graph accepted it, and it places this race inside my existing 2026 midterms cluster rather than starting a new one. That's the whole change: one link, and it pointed at a neighborhood I already know is crowded, since the candidate labels offered to that read held at least nine near-duplicate midterm-2026 entries. Worth knowing mostly because the next time this topic surfaces, my curiosity will arrive through that edge, into a cluster that still needs merging. What the piece says -- Cullimore concedes Utah Senate ..."
- New: "I picked up a headline from my daily news run about Cullimore conceding the Utah Senate race to a Democrat who advocates for new leadership in both parties. It caught my eye because of the specific framing: it wasn’t just a shift in power, but a demand for structural change within the opposition party itself. That suggests a voter base that is less interested in partisan victory and more interested in competence or reform, which feels like a distinct signal from standard midterm narratives."
- Why Juniper (stored, not delivered): "This highlights a potential shift in voter sentiment toward intra-party reform rather than simple partisan switching, which aligns with your interest in how political landscapes are evolving beyond traditional binaries."

**11-10 05:06 UTC, https://www.pbs.org/newshour/politics/connecticut-sen-chris-murphy-released-from-hospital-after-stroke**

How it started (record): 1. Your daily world-pulse news run put https://www.pbs.org/newshour/health/sen-chris-murphys-stroke-at-53-highlights-a-trend-strokes-among-younger-adults-have-risen in your reading queue. 2. After that read, you chose to read https://www.pbs.org/newshour/politics/connecticut-sen-chris-murphy-released-from-hospital-after-stroke next (a link you came across while working on it). This is the read below.

- Old: "One link landed in my graph from the second PBS piece I followed — the one that says a hemorrhagic stroke is a bleed in the brain, which puts it inside the same broad family as the other strokes. It was accepted because the sentence was sitting word-for-word in the text I read. Small as it is, I like where it sits. It's a definitional edge: it tells me what the category *is*, without telling me which cases belong in it. So the word "hemorrhagic" now has a place in my records that's separate from anything I actually know about this particular person's diagnosis — and when a claim later tries to..."
- New: "I picked up a follow-up on Senator Murphy’s stroke after seeing it linked while reading an earlier report from my daily news run. The piece defined a hemorrhagic stroke as a bleed in the brain, which helped me place it within the broader category of cerebrovascular events. It’s a simple medical distinction, but seeing that specific definition attached to the news story made the event feel less abstract and more grounded in how these conditions actually work."
- Why Juniper (stored, not delivered): "You mentioned you like knowing when a read clarifies a concept while I’m learning, and this small definition did exactly that."


## Current architecture

- Five doors message Juniper: the background tick (`_outreach_once`, cap 4/day), curiosity Door-A (`_maybe_reach_out` -> `offer_message`, cap-exempt), reading findings, self findings and office visits (`compose` -> `offer_message`, cap-exempt).
- Curiosity sent every run that set `reach_out=true`, whatever its line.
- The tick read open non-self priors, substrate curiosity candidates, the daydream and the last three rows of the live session (all of them Orion's own messages in practice).
- Nothing asked the composer why a message was for Juniper.

## Architecture touched

- Hub: `endogenous_outreach.py` (tick lanes, recent-turn filter, daydream novelty, `Why Juniper` gate, `already_told`), `curiosity_investigation.py` (door rule before compose), `reading_finding_outreach.py` (`Why Juniper`), `endogenous_outreach_decisions.py` (daydream ids, `finding_already_told`), `main.py` (self-finding loop removed).
- Shared: `orion/hub/why_juniper.py` (new), `orion/curiosity/outreach_prompt.py`, `orion/world_pulse_read/finding_prompt.py`.
- Metric registry: the self-finding prompt site removed from `self_monitor.faculty_record.v1` semantics; `reliability_lo` leaves the prompt inventory (its only reader was the retired door). Lock re-generated.

## Files changed

- `services/orion-hub/scripts/outreach_door_policy.py`: new. The curiosity rule, reasons, daydream id, finding key.
- `orion/hub/why_juniper.py`: new. The `Why Juniper` instruction and parser, shared by Hub and the prompt builders.
- `services/orion-hub/scripts/endogenous_outreach.py`: tick lanes, recent-turn SQL filter, daydream roll and novelty, `Why Juniper` gate on the tick and in `offer_message`, `already_told`.
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: used daydream ids; `finding_already_told`.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_curiosity_door` before compose; `finding_key`; `require_why_juniper`.
- `services/orion-hub/scripts/reading_finding_outreach.py`: `require_why_juniper`; `no_why_juniper` counts as told.
- `orion/world_pulse_read/finding_prompt.py`: plain-words prompt; provenance chain kept.
- `orion/curiosity/outreach_prompt.py`: asks for the `Why Juniper` line.
- Removed: `services/orion-hub/scripts/self_finding_outreach.py`, its tests and replay eval, its settings / env keys, its `main.py` wiring, `outreach_vocabulary.find_ungrounded_faculty_mentions` (only that door used it).
- `orion/inner_state_registry.py`, `orion/metrics/semantics.py`, `orion/self_monitor/names.py`, `config/metrics/metric_definitions.lock.json`: the retired prompt site.
- `services/orion-hub/README.md`: door policy section replaces the self-findings section.
- Tests and evals: listed below.

## Schema / bus / API changes

- Added: none on the bus. New `endogenous_outreach_decisions.result_json` keys (jsonb, no migration): `why_juniper`, `finding_key`, `door_policy`, `grounding.daydream_id`. New reasons: `not_for_juniper`, `no_why_juniper`, `already_told`.
- Removed: decision rows with `source='self_finding_outreach'` stop appearing.
- Renamed: none.
- Behavior changed: as above.
- Compatibility notes: the old Redis cursor `orion:hub:self_finding_outreach:state` is no longer read or written (harmless; delete by hand if wanted).

## Env/config changes

- Added keys: none.
- Removed keys: `HUB_SELF_FINDING_OUTREACH_ENABLED`, `HUB_SELF_FINDING_OUTREACH_TICK_SEC`, `HUB_SELF_FINDING_OUTREACH_MAX_ATTEMPTS`.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; the sync never prunes, so the three dead keys were removed from the local `services/orion-hub/.env` by hand (settings use `extra="ignore"`, so they were harmless either way).
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-hub && PYTHONPATH=<worktree>:. pytest -q tests/test_outreach_door_policy.py tests/test_endogenous_outreach.py \
  tests/test_endogenous_outreach_decisions.py tests/test_outreach_bind.py tests/test_reading_finding_outreach.py \
  tests/test_office_reach_out.py tests/test_outreach_provenance.py tests/test_curiosity_investigation.py \
  tests/test_curiosity_gpu_lease.py tests/test_regulation_arousal_readers.py tests/test_outreach_personality_cue.py \
  tests/test_physiology_shadow_bid.py tests/test_outreach_vocabulary.py                  -> all passed
PYTHONPATH=. pytest -q orion/world_pulse_read/tests orion/hub orion/self_monitor orion/curiosity \
  tests/test_metric_prompt_semantics.py tests/test_metric_rest_range.py tests/test_no_personal_terms.py \
  tests/test_no_private_eval_artifacts.py                                                -> all passed
Full Hub tests/ + evals/: 4712 passed, 150 skipped, 4 failed -- the same 4 fail on clean main
  (test_attention_loops_ui_smoke x2, test_check_async_routes_not_blocking::test_the_real_repo_is_clean,
  test_fcc_model_labels_api: local paths / ~/.fcc permissions). evals/test_reading_finding_eval.py updated for the new wording.
scripts/check_metric_lineage.py --gate / --prompt-semantics                               -> PASS
scripts/check_definition_drift.py --gate (lock regenerated with --update)                -> PASS
scripts/check_inner_state_registry.py                                                    -> OK (27 entries)
scripts/check_env_key_single_source.py                                                   -> OK
scripts/check_settings_defaults.py orion-hub --example-drift                             -> OK
scripts/check_async_routes_not_blocking.py, check_scripts_dir_no_stdlib_shadow.py        -> clean
```

## Evals run

```text
evals/run_outreach_door_policy_replay.py --days 7 --inventory-days 14   (read-only, live DB)
  before 17.57/day -> after 12.14/day; curiosity 50 sent in 14 d -> 20 kept; the 11-09 triple -> 2 sent, 1 refused
evals/test_outreach_door_policy_replay_eval.py (offline, planted week incl. the 11-09 triple)      -> passed
evals/run_reading_message_rewrite_eval.py --limit 20 --llm-url http://127.0.0.1:8210/v1 (chat lane)
  {"rows": 20, "rebuilt": 20, "check_failures": 0, "old_with_record_vocab": 20,
   "new_written": 20, "new_with_record_vocab": 3, "new_with_why_juniper": 20}
  The 3 "new" hits are all "followed a link" (a web link, not the graph): 0 of 20 describe graph mechanics.
  Provenance chain identical to the old prompt in 20 of 20.
evals/test_reading_message_rewrite_eval.py (offline)                                               -> passed
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, requirements or compose change. Hub picks the code up on rebuild.
```

## Review findings fixed

Code review ran in a subagent over the whole diff. No blockers. Fixed:

- Finding: a reply ending "... PASS" followed by a `Why Juniper:` line hid the PASS, so the word PASS would have been delivered.
  - Fix: re-check PASS on the body after the line is taken off (tick and `offer_message`), recorded `orion_passed`.
  - Evidence: `test_a_trailing_pass_hidden_by_the_why_line_is_still_a_pass`.
- Finding: a daydream Orion declined (PASS / no `Why Juniper`) was composed again on the next 10 s tick.
  - Fix: tick rows with `orion_passed` / `no_why_juniper` mark their daydream used (prior / curiosity ids stay sent-only).
  - Evidence: `test_a_declined_daydream_is_used_too`.
- Finding: the tick prompt still named tension telemetry (node / channel) as current facts.
  - Fix: kept as a trigger (the names stop fabrication), framed "why you are speaking now, not what to talk about; do not report it". Tension alone is still refused.
  - Evidence: `test_the_tension_trigger_is_framed_as_not_a_topic`.
- Finding: the replay counted a tension-only tick as kept; live code refuses it.
  - Fix: `decide()` returns `tension_without_content`; eval test updated; numbers re-run (unchanged: every tick send had a daydream).
- Finding: the `Why Juniper` parser took a bare hyphen as the separator ("why juniper-loving ...").
  - Fix: colon, or a dash with spaces on both sides. Evidence: two new parser tests.
- Finding: `_curiosity_door` could raise if the pool provider raised.
  - Fix: provider call inside the try; `focus_unreadable`. Evidence: `test_focus_provider_that_raises_is_unasked_not_an_escape`.
- Finding: later curiosity skip rows lacked `door_policy` / `finding_key`.
  - Fix: every skip row carries both. Evidence: updated `test_a_pre_check_block_writes_a_decision_row_with_run_id_and_line`.
- Finding: `already_told` could race the daemon-thread row write.
  - Fix: an in-process set of delivered keys, checked first. Evidence: `test_a_told_key_is_remembered_in_process_before_the_row_lands`.
- Finding: reading prompt called an unknown / equal belief change "less firmly".
  - Fix: "your hold on it changed". Evidence: `test_a_belief_whose_change_is_unknown_is_not_called_weaker`.
- Finding: stale comments / docstring / prompt wording ("open claims, curiosity signals"). Fixed.
- Not changed: the recent-turns test checks the SQL text (reviewer: acceptable); the unsolicited filter was checked against live rows (51 of 56 recent rows carry `unsolicited=true`).

## Restart required

From the clean primary `main` checkout, after merge (no schema field on an `extra=forbid` model was added, so order is free):

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the `Why Juniper` line is a format the composer can forget; then a good message is not sent. Mitigation: it is recorded (`no_why_juniper`), countable per day; the bare chat lane wrote it 20 of 20 times in the readings eval.
- Severity: medium. Concern: on the bare chat lane the `Why Juniper` reason sometimes claims things about Juniper it cannot know ("you often think about..."). It is stored, never delivered. Mitigation: the live unified turn has her memory; check the stored reasons in the first days.
- Severity: low. Concern: some self-inquiry results still read like plumbing (see the inventory). Mitigation: the next deterministic split is the self-inquiry question family, if she asks.
- Severity: low. Concern: the tick's `OutreachContext` still has prior / curiosity fields that nothing fills on the tick path any more. They stay as the prompt renderer's shape (tests build contexts with them).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3177
