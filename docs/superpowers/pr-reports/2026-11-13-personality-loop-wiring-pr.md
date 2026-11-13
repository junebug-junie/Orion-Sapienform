# Personality loop wiring (rev 4): real option text, care reaches outreach, Prime choices count per family, quiet reducer ticks

Stacks on #3331 (outreach choice point) and #3332 (market values). Merge those first.

## Summary

- Curiosity choices now read as Orion's own words. Hub stores each offered prior's claim on the offer row
  (`curiosity_offer_decisions.offered[].claim`, JSON field, no migration). The durable-runs curiosity sync uses
  it as the option text. Old rows keep the de-slugged id.
- Outreach care is no longer always 0. The binding bid carries the shown topics' text (`object_text=`). When
  the want was funded only through care, the outreach choice is `care_promoted`.
- Prime choices count toward the right family. Each concern option is keyed `<want family>::<concern_id>`
  (`weights.endorsement_option_key`), from the funded event want the concern names. A concern with no event
  keeps its plain id (lane family `prime`).
- Prime choices carry `care_promoted`. `GET /physiology/event-wants` marks a listed want `care_promoted` when
  it is in the top `limit` only through care, or funded only through care. Prime copies the flag onto a
  choice of that concern.
- The reducer stops rewriting every node on every idle tick. A fade-only change is written once the weight
  moved 1% from the stored weight (about every 5 h at the 14-day half-life).
- Juniper's No holds. A choice in the direction of a tendency she said No to counts on that retired node
  instead of founding a new tendency she would be asked about again. It never comes back by itself; a later
  Yes from her revives it. Reads `personality_juniper_verdict` (#3335) defensively: missing table, no answers.

## Outcome moved

- Curiosity option text: "turn start context is stale snapshot" (id) becomes the prior's claim sentence,
  which the reducer embeds. Better directions for every new curiosity choice.
- Outreach care: from structurally 0 (`personality:care=0:reason=no_object_text`) to priced.
- Source (c): Prime picks of a reading concern count for `hub.world_pulse_read`, a camera ask for
  `vision.ask`, not all for `prime`.
- Runaway guard: Prime and outreach choices that only reached Orion through care count at half weight.
- Reducer writes: live replay of the 118 live choice nodes (read-only, `personality_disposition` dump at
  10:37Z): old rule writes 118 on the next idle tick, new rule writes 0; +1 h: 0; +5 h: 118 (the 1% fade).

## Current architecture

- Curiosity offer rows stored only prior ids; `orion/personality/choices.py` de-slugged them.
- Outreach bids had `object_ref="juniper"` and no `object_text`; the market's care term read no text.
- `orion/prime/choice.py` keyed options by concern id; `weights.option_family` mapped them all to `prime`.
- No producer set `PersonalityChoiceV1.care_promoted`.
- `orion/personality/graph.py` `_Tick.result` compared exact snapshots including `weight`, which fade moves
  every tick.

## Architecture touched

- orion-hub: curiosity offer log, outreach binder + tick + shadow bid, outreach choice builder.
- orion-durable-runs: event-wants menu (care_promoted), personality reducer write rule (shared `orion/personality`).
- orion-mind: Prime loop passes event families and care-promoted events to the choice builder.
- Shared: `FundedEventWantV1.care_promoted` (additive), `personality_values.care_funded`.

## Files changed

- `services/orion-hub/scripts/curiosity_offer_decisions.py`: `offered_rows` stores `claim` (<= 400 chars).
- `orion/personality/choices.py`: option text = stored claim, else readable id.
- `services/orion-hub/scripts/outreach_bind.py`: `bind(object_text=)`, sends it with the bid; `care_promoted` on the held want and in `record()`.
- `services/orion-hub/scripts/endogenous_outreach.py`: passes the topics' text to the bid; passes `care_promoted` to the choice.
- `services/orion-hub/scripts/outreach_choice.py`: `outreach_object_text`, `build_choice(care_promoted=)`.
- `services/orion-hub/scripts/physiology_shadow.py`: shadow bid carries `object_text`.
- `orion/physiology/personality_values.py`: `care_funded`.
- `orion/schemas/physiology_want.py`: `FundedEventWantV1.care_promoted`.
- `services/orion-durable-runs/app/event_wants_menu.py`: reads decision `reason`/`warnings`, computes `care_promoted`.
- `orion/prime/choice.py`, `services/orion-mind/app/prime_loop.py`: family keys, `care_promoted`.
- `orion/personality/graph.py`: `_must_write` (+ `GraphParams.fade_write_min_rel_change`); Juniper's answers
  (`fold`/`apply_fade(juniper=)`, `_Tick.absorb`, revive on a later yes).
- `services/orion-durable-runs/app/personality_store.py`: `load_juniper_nodes` (defensive on the missing table).
- `services/orion-durable-runs/app/personality_driver.py`: folds Juniper's retired nodes with the live ones.
- `orion/prime/state.py`, `orion/prime/thought.py`: a new event concern records its want's family and care flag.
- `orion/physiology/choice.py`: `personality:funded_without_care=0|1` on a funded decision with a term.
- `services/orion-durable-runs/app/main.py`: `GET /physiology/event-wants?care=1` opt-in.
- `config/physiology/contract.yaml`: `orion/prime/choice.py` registered (imports `endorsement_option_key` only).
- Tests: `orion/personality/tests/test_graph.py`, `test_choices.py`, `orion/prime/tests/test_prime_choice.py`,
  `orion/physiology/tests/test_personality_values.py`, `services/orion-durable-runs/tests/test_event_wants_menu.py`,
  `test_personality_naming.py` (fade assertions now compare the stored pair faded to now),
  `services/orion-hub/tests/test_outreach_bind.py`, `test_outreach_choice.py`, `test_curiosity_spend_log.py`,
  `services/orion-mind/tests/test_prime_choice_point.py`.
- READMEs: orion-durable-runs, orion-hub.

## Schema / bus / API changes

- Added: `FundedEventWantV1.care_promoted: bool = False` (HTTP contract of `GET /physiology/event-wants`, not a
  bus payload). Sent only with `?care=1`, which the new orion-mind asks for.
- Added: `PrimeConcernV1.event_family`, `event_care_promoted` (Redis `orion:prime:state`), left out of the saved
  state when unset.
- Added: decision warning `personality:funded_without_care=0|1`.
- Added: `curiosity_offer_decisions.offered[].claim` / `stale_offered[].claim` (JSON field, no DDL).
- Behavior changed: Prime `PersonalityChoiceV1` option keys are `<family>::<concern_id>` for event concerns.
  Source (c) then counts a Prime pick for the event's family, and a Prime "no move" (`declined_all`) as a
  decline of every shown concern's family (Orion passed on those events). Only concerns with no event count
  for `prime`.
- Behavior changed: the reducer matches retired nodes Juniper said No to (absorb only) and revives one on her
  later Yes.
- Rollback: an orion-mind built before this PR cannot read a saved Prime state whose concerns carry
  `event_family`/`event_care_promoted`; drop those keys from `orion:prime:state` before rolling mind back.
- Compatibility: `care_promoted` is opt-in (`?care=1`), so any deploy order of orion-mind and orion-durable-runs
  is safe. `PrimeConcernV1` (`extra="forbid"`) gains two fields: Hub reads Prime state for a concern count
  (`parse_prime_redis`), so rebuild orion-hub before orion-mind (the post-pull run's alphabetical order already
  does). Old `personality_choice` rows keep their old keys and text (insert-only).

## Env/config changes

- None. No `.env_example` changed; no sync needed.

## Tests run

```text
orion/personality orion/physiology/tests orion/prime orion/mindloop: 706 passed
services/orion-durable-runs/tests: 914 passed, 136 skipped
services/orion-mind/tests: 367 passed
services/orion-hub/tests (full suite, first commit): 4809 passed, 1 failed, 4 errors -- all in files this PR
  does not touch (test_fcc_model_labels_api env default; test_consult_belief_store_sql needs Postgres)
services/orion-hub outreach/curiosity files after the review fixes: 340 passed
scripts/check_definition_drift.py --gate: PASS (0 changed)
scripts/check_body_economy_contract.py: OK
JUNIPER_NODES_SQL against the live schema (temp verdict table, rolled back): runs, 0 rows
```

## Evals run

```text
orion/prime/evals/run_prime_frame_eval.py: PASS
orion/prime/evals/run_prime_block_eval.py: PASS (0 failing)
orion/prime/evals/run_prime_wake_replay.py: PASS
Live replay (read-only) of personality_disposition, 118 choice nodes: idle-tick writes 118 -> 0
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, compose, requirements or env change. Images rebuild from source on deploy.
```

## Review findings fixed

Code review ran in a subagent on the first commit. No must-fix findings. Fixed:

- Finding: deploy-order window (`FundedEventWantV1` is `extra="forbid"`; durable-runs rebuilds before mind).
  - Fix: `care_promoted` only with `?care=1`; Prime asks for it.
  - Evidence: `test_the_payload_leaves_care_promoted_out_unless_asked`.
- Finding: Prime read family and care flag at choice time, so a concern that outlived its want's place on
  the list fell back to `prime` and lost its flag.
  - Fix: the concern records both when it opens; the live list is only a fallback for old concerns.
  - Evidence: `test_a_new_event_concern_records_its_wants_family_and_care_flag_and_saves_without_them_when_unset`.
- Finding: stale `weights.py` docstring; Prime "no move" now declines event families.
  - Fix: docstring rewritten; behaviour stated above (kept on purpose: passing on every event is a decline).
- Finding: outreach `care_funded` could read an exploration-floor fund as care-only.
  - Fix: the market writes `personality:funded_without_care=0|1` (counts the floor); `care_funded` and the
    event-wants menu read it.
  - Evidence: `test_a_funded_decision_with_a_term_says_whether_it_is_funded_without_care`,
    `test_care_funded_only_when_care_alone_made_the_net_positive`.
- Finding: comment said a retrieval-query fallback exists; it does not. Fix: comment corrected.
- Finding: the first topic filled the 400-char object text. Fix: each topic gets an equal share.
- Finding: restated 400 caps. Fix: the curiosity claim cap imports `CHOICE_OPTION_TEXT_MAX`; the outreach cap
  stays a local constant (the body-economy contract gate refuses a hub import) with a test pinning it to the
  market's cap.
- Finding: `_must_write` reused `history_min_rel_change`. Fix: own `fade_write_min_rel_change`.
- Finding: duplicate event refs read inconsistently. Fix: one first-wins `listed_wants` pass for both maps.
- Finding: no driver-level regression test. Fix: `test_a_quiet_tick_writes_no_node_and_a_one_percent_fade_writes_each_once`.
- Finding: SQL test checked strings only. Fix: `test_read_menu_passes_reason_and_jsonb_warnings_through_to_care_promoted`.
- Second review (Juniper's No commit):
  - Finding: a "yes" revived a node on every tick even after Orion's own rules retired it again.
    - Fix: revive only when her yes is newer than the node's retirement (`juniper_at` read with the answer).
    - Evidence: `test_a_yes_revives_only_a_node_retired_before_it`.
  - Finding: a vetoed node could take a choice a live tendency also matched.
    - Fix: live tendencies first; absorb only what no live tendency matches.
    - Evidence: `test_a_live_tendency_that_matches_keeps_the_choice_over_a_closer_juniper_no`.
  - Finding: `PrimeConcernV1` comment overclaimed rollback safety. Fix: comment states the Hub-first order and
    the rollback step.
  - Finding: tie order differed from #3335's `LATEST_VERDICTS_SQL`. Fix: same order (`verdict_id` ascending).
  - Finding: tick dimension could come from a vetoed node. Fix: live nodes first.
  - Not fixed here (must, other PR): #3335's `record_juniper_verdict` refuses any answer on a retired node, so
    the "later Yes" cannot be written yet. See risks.
- CI: `Body economy contract` refused `orion/prime/choice.py` as an unregistered reader. Fix: registered in
  `config/physiology/contract.yaml` (imports only the key helper).

## Restart required

Deploy order (from the primary main checkout, after #3331 and #3332; no migration in this PR):

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low. Concern: readers of the raw `personality_disposition.weight` column (the market's tendency
  read, the frame) can be up to 1% staler than before. Mitigation: bounded by `history_min_rel_change`; the row
  stays exact as of its `updated_at`.
- Severity: low. Concern: outreach care embeds the joined topics, a blend, not one topic. Mitigation: shown
  topics are ranked by freshness, so care can only change whether the compose is funded, which is what the
  single bid prices.
- Severity: medium. Concern: #3335's `record_juniper_verdict` refuses any answer on a retired tendency
  (`TendencyRetired`), so the Hub cannot write the "later Yes" this reducer honours. Mitigation: the reducer
  side is ready; #3335 needs to accept a yes on a tendency Juniper herself retired.
- UNVERIFIED live: claim text on new curiosity rows, outreach `personality:care=` lines with a reason other
  than `no_object_text`, Prime family keys and `care_promoted` on live rows, and `nodes_written` near 0 on idle
  ticks, Juniper's No absorbing a repeat choice (needs #3335's table). All need the deploy above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3336

🤖 Generated with [Claude Code](https://claude.com/claude-code)
