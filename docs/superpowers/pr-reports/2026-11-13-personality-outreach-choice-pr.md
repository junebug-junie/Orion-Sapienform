# PR report: outreach choice point (personality rev 4) + readable tendency labels

## Summary

- When Orion reaches out unprompted, Orion now picks what to say. The same model call that writes the message shows Orion up to 4 real topics (the newest daydream, and reads waiting to be told), plus "0. Say nothing". Orion answers `Pick: <n>`. There are no new model calls.
- "Say nothing" (`Pick: 0` or PASS) sends nothing. The decision row records `orion_declined`.
- Every readable pick publishes one `PersonalityChoiceV1` (lane `outreach`) on `orion:personality:choice`. This is the input the rev 4 personality organ folds.
- Dedupe is now by subject, not by message text. A subject Juniper got, or that Orion declined, in the last 7 days is not shown again unless it changed. With the choice point on, the tick no longer reads the personality cue, which is how the "graph first" habit reached Juniper 3 times on 11-11..12.
- Separate commit: tendency labels are readable. "turn start context is stale snapshot over camera layer mediation not closeness (+13 more), 21 of 21" becomes "turn start context is stale snapshot, 21 of 21".

## Outcome moved

- The outreach tick becomes a real choice point (spec rev 4, move 1). Before, it composed from whatever single material it had.
- The repeat-send failure is closed in two ways: no free cue material, and a subject ledger.
- Tendency labels on the Hub panel read as phrases, not slugs.

## Current architecture

- `services/orion-hub/scripts/endogenous_outreach.py` `_outreach_once`: gates, then fire on a tension run or a novel daydream, then one compose call over one prompt, then the PASS / Why-Juniper / vocabulary / re-gate checks, then delivery.
- Dedupe was a 7-day ledger of daydream caption hashes. The personality cue (when on) was rendered as free prompt material with no id in that ledger.
- Readings (`reading_finding_outreach.py`) are their own door, with their own compose and spacing.
- `orion/personality/graph.py` built labels from 7-word gists of both sides, plus "(+k more)".

## Architecture touched

- Hub only (one service), plus `orion/personality/graph.py` (shared, pure).
- Producer side of an existing contract. `PersonalityChoiceV1` / `orion:personality:choice` were already registered with orion-hub as a producer (#3321). There is no schema change.

## Files changed

- `services/orion-hub/scripts/outreach_choice.py` (new, pure):
  - topic type
  - subject-ledger fold
  - ranking (freshest first)
  - prompt block
  - `Pick:` parser
  - `PersonalityChoiceV1` builder
- `services/orion-hub/scripts/endogenous_outreach.py`:
  - choice point in `_outreach_once`
  - `_choice_topics`, `_settle_pick`, `_publish_choice`
  - subject claims
  - the cue is not read in choice mode
  - choice prompt
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: `fetch_said_outreach_subjects`. `orion_declined` now marks a daydream used.
- `services/orion-hub/scripts/reading_finding_outreach.py`:
  - `outreach_topics()` (pending reads as tick topics; its spacing gates them)
  - shared claims, with a re-check after the claim
  - tick rows carrying `finding_ids` count as told and start spacing
  - `orion_declined` is told
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`: two keys, plus wiring.
- `services/orion-hub/README.md`: section 4.1.
- `scripts/analysis/outreach_binding_replay.py`: the new reasons count as "reached compose".
- `orion/personality/graph.py`: `gist` (6 words, no dangling tail, no ellipsis), `pair_label` (names 1-2 declined, else the chosen side only), `tidy_label` (old stored labels are tidied on the next fold).
- Tests:
  - `services/orion-hub/tests/test_outreach_choice.py` (new)
  - `tests/test_reading_finding_outreach.py`
  - `orion/personality/tests/test_graph.py`
  - `services/orion-durable-runs/tests/test_personality_loop.py`
- Evals: `services/orion-hub/evals/run_outreach_choice_readout.py` (live readout) and `test_outreach_choice_readout_eval.py`.

## Schema / bus / API changes

- Added: Hub now publishes `personality.choice.v1` (the channel and schema were already registered).
- Added decision reasons: `orion_declined`, `no_pick`, `invalid_pick`. Added row fields: `result_json.choice`. Tick rows also carry `finding_ids` / `seed_id` when a read was sent or declined.
- Behavior changed:
  - A tick PASS among topics is now `orion_declined`, not `orion_passed`. Without topics it is still `orion_passed`.
  - Tendency labels have a new shape.
- Compatibility: the readings door SQL also matches source-less rows with `finding_ids`. Checked live, read-only: it returns 87 rows over 14 days, identical to door-only rows (no existing tick row carries `finding_ids`).

## Env/config changes

- Added keys: `HUB_ENDOGENOUS_OUTREACH_CHOICE_ENABLED=true` (kill switch), `HUB_ENDOGENOUS_OUTREACH_CHOICE_MAX_OPTIONS=4` (1-5).
- `.env_example` updated: yes.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes. Both keys are in the primary checkout's `services/orion-hub/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub tests (-k outreach|reading_finding|door_policy|strain_report|office_reach|personality|provenance|tension) + evals/: 704 passed, 12 skipped
orion/personality/tests + services/orion-durable-runs/tests: 1049 passed, 136 skipped
full services/orion-hub/tests from repo root (as CI): 4784 passed, 1 failed, 4 errors -- all in untouched files needing local DB/env
  (test_fcc_model_labels_api, test_consult_belief_store_sql)
mutation checks: removing the subject filter fails the 3x-repeat and new-version tests; re-enabling the cue in choice mode fails the cue test
gates: check_settings_defaults orion-hub --example-drift, check_env_key_single_source, check_env_template_parity,
  check_metric_lineage --gate / --prompt-semantics, check_definition_drift --gate, check_async_routes_not_blocking,
  check_scripts_dir_no_stdlib_shadow, check_body_economy_contract: all PASS; git diff --check clean
```

## Evals run

```text
evals/test_outreach_choice_readout_eval.py (new, offline): pass
evals/run_outreach_choice_readout.py (live, read-only): run after deploy. Reports choices/day, picks vs declines,
  rank-1 departure share (spec M1 >= 25%), and subjects sent twice inside 7 days (should be 0)
existing outreach replay evals: pass
```

## Docker/build/smoke checks

```text
Not run: no Docker build from this branch (Hub image deps unchanged; no new requirements).
Live read-only: personality_choice table exists (217 curiosity rows); readings SQL predicate verified as above.
```

## Review findings fixed

The code review ran in a subagent. Findings and fixes:

- Finding: a decline at the tick left the read pending, so the readings door sent it anyway.
  - Fix: `orion_declined` rows carry the shown reads' `finding_ids`, the door counts `orion_declined` as told, and an `on_declined` mirror covers the in-process case.
  - Evidence: `test_declining_a_read_at_the_tick_tells_the_readings_door`, `test_a_read_orion_declined_at_the_tick_is_not_sent_by_this_door`.
- Finding: with one topic, a broken Pick line ("Pick: say nothing", "My pick: 2") became an implicit pick.
  - Fix: the parser accepts these forms. A line that looks like a pick but does not parse is never implicit. "Pick 3 things..." is not a pick.
  - Evidence: parametrized parser tests.
- Finding: `build_choice` could raise out of the tick.
  - Fix: guarded. The row records `published: false`.
  - Evidence: `test_a_choice_that_fails_to_build_never_raises_out_of_the_tick`.
- Finding: race where the door claims after the tick released.
  - Fix: the door re-checks told and spacing after its claim.
  - Evidence: `test_a_read_the_tick_told_while_this_door_awaited_its_gates_is_not_composed`.
- Finding: `no_pick` / `invalid_pick` could recompose the same topics on every tick.
  - Fix: both now mark the shown subjects decided.
  - Evidence: fold test.
- Finding: tick compose failures counted against the door's `max_attempts`.
  - Fix: `finding_ids` only go on sent / `no_why_juniper` / declined rows.
  - Evidence: `test_a_tick_compose_failure_on_a_read_does_not_carry_its_finding_ids`.
- Finding: priors and curiosity summaries (no ids) were still rendered next to topics.
  - Fix: they are left out of a choice prompt.
  - Evidence: `test_choice_prompt_leaves_out_material_with_no_subject_id`.
- Finding: the regression test's docstring overclaimed.
  - Fix: reworded. The cue half of the fix is guarded by `test_choice_mode_does_not_read_the_personality_cue`.
- Finding: `claimed_by_tick` stalled later bundles.
  - Fix: it now continues to the next bundle.

## Restart required

Deploy from the clean primary main checkout after merge:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

The durable-runs rebuild picks up the label change. The Hub rebuild picks up the choice point. Order: either; Hub first is fine, since choices published before durable-runs restarts are only lost if durable-runs is down (pub/sub).

## Risks / concerns

- Severity medium. Concern: the model may drop or mangle the `Pick:` line, which gives fewer sends (`no_pick`). Mitigation: a lenient parser, an implicit pick with one topic, `no_pick` visible in the decision log and the readout eval, and the kill switch.
- Severity low. Concern: most ticks will show only 1-2 topics (daydream, plus reads only while their spacing allows). A one-topic choice is speak versus say nothing, which is weaker evidence. Mitigation: more topic sources plug into `topic_sources`.
- Severity low. Concern: tendency labels are word-count gists of slug-derived text. The curiosity offer rows store only prior ids, so there is no richer "plain text" to use. A 6-word cut can still drop a key word.
- UNVERIFIED: a live tick publishing a choice, durable-runs inserting an `outreach` row in `personality_choice`, and the live labels re-rendering. All need a deploy.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3331

🤖 Generated with [Claude Code](https://claude.com/claude-code)
