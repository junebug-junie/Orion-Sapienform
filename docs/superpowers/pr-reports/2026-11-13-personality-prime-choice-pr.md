# Personality rev 4: Prime's choice point and tending lines

## Summary

- Each Prime thought now says which held concern it acts on, or "none", in the same model call it already makes (new `pick` field in the answer). No new model call.
- After the thought, Prime publishes one `personality.choice.v1` on `orion:personality:choice` when it chose among 2 or more concerns, or declined all of them. durable-runs (from #3321) stores it in `personality_choice`.
- The Prime block in every unified turn gains up to 2 lines `- tending: <label> [disposition:<id>]`: what Orion's past choices show Orion tends to pick. Only choice-built tendencies with at least 10 weighted choices are shown. They are the first lines dropped when the block is over its 600-character cap.
- Prime's own thought frame shows the same lines (section WHAT YOU TEND TO CHOOSE), so Prime picks with its tendencies in view.
- `disposition` is now a Prime ref kind, so the existing citation gate judges `[disposition:<id>]` refs.
- New flag `PRIME_PERSONALITY_LINE_ENABLED` (hub and orion-mind) ships `true`. No shadow (Juniper, 2026-11-13).

## Outcome moved

Prime becomes the densest real choice lane for the personality graph (spec rev 4, move 1; rev 3 decision 4). Before this, Prime's pick among its concerns was implicit and never recorded. Now every thought that holds concerns can produce a choice row with the options it turned down, the market's order, and the body situation. The tending line (move 4) closes the read-back half: once a tendency has 10 weighted choices, Orion sees it in every turn and in Prime's own frame.

## Current architecture

- Prime (`services/orion-mind/app/prime_loop.py`) thinks once per funded bid. The answer (`PrimeAnswerV1`) edits concerns and makes at most one move. A curiosity move may name a `concern_id`; nothing recorded that as a choice.
- The Prime block (`orion/prime/block.py`) renders concerns, last thought, asks and body for each unified turn (Hub `orion/hub/turn_orchestrator.py`). It had no personality line.
- #3321 (merged) added the `PersonalityChoiceV1` contract, the `orion:personality:choice` channel, the durable-runs consumer and the choice-only reducer. No Prime producer existed.

## Architecture touched

- `orion/prime/choice.py` (new, pure): options = the concerns the frame showed; pick resolution; market rank; situation.
- `orion/prime/tending.py` (new, pure): selects tendencies from the `/personality/frame` payload.
- `orion/prime/thought.py`: `pick` in the answer and the system prompt (step 4); WHAT YOU TEND TO CHOOSE frame section.
- `orion/prime/block.py`: tending lines, dropped first; `disposition` in `PRIME_REF_KINDS`; `tending=<n>` in `log_fields`.
- `services/orion-mind/app/prime_loop.py`: captures the shown concerns and the market's event order at frame time; publishes the choice after the move (fail-open); reads the tending lines for the frame.
- `orion/hub/turn_orchestrator.py`: `_read_prime_tending` (one GET, 1 s, fail-open) feeding `fetch_prime_block`.

## Files changed

- `orion/prime/choice.py`: Prime choice point (pure).
- `orion/prime/tending.py`: tending selection (pure).
- `orion/prime/thought.py`: `pick` field, prompt step 4, frame section.
- `orion/prime/block.py`: tending lines, ref kind, log field.
- `orion/prime/evals/run_prime_block_eval.py`: tending and choice-point checks on the live 2026-11-12 state; `--durable-runs` for a live read.
- `orion/prime/tests/test_prime_choice.py`: pure tests (choice, tending, block).
- `orion/hub/turn_orchestrator.py`: tending read for the Hub Prime block.
- `services/orion-mind/app/prime_loop.py`: producer + frame read.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: flag + docs.
- `services/orion-mind/tests/test_prime_choice_point.py`: runtime tests (publish, declined_all, holdback, fallback, fail-open, frame lines).
- `services/orion-mind/tests/fixtures/prime_replay_golden.json`, `test_prime_replay.py`: re-recorded golden (see Tests run).
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: flag + docs.
- `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py`: block carries tending lines and logs `tending=1`; the read is fail-open.
- `.github/workflows/orion-prime-tests.yml`: also triggers on `orion/schemas/personality.py`.

## Schema / bus / API changes

- Added: producer of `orion:personality:choice` in orion-mind (already declared in `orion/bus/channels.yaml` by #3321; no contract change).
- Added: optional `pick` field in Prime's model answer (`PrimeAnswerV1`, internal; `extra="ignore"`).
- Behavior changed: `check_prime_citations` now judges `disposition:` refs. A reply citing a disposition id the block did not show is counted as rejected (counted, never blocked).
- Compatibility notes: an answer without `pick` still works (the move's `concern_id` stands in, else no row). The Prime block is unchanged when there are no qualifying tendencies.

## Env/config changes

- Added keys: `PRIME_PERSONALITY_LINE_ENABLED=true` (orion-hub, orion-mind).
- `.env_example` updated: yes (both services), and both `docker-compose.yml` files pass it.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-hub: +PRIME_PERSONALITY_LINE_ENABLED`, `orion-mind: +PRIME_PERSONALITY_LINE_ENABLED`).
- skipped keys requiring operator action: none.

## How each choice is built

- Options: the concerns the frame showed (captured at frame time; a concern added or dropped in the same answer does not change them). Key = concern_id; text = the concern's words (embedded by the reducer).
- `pick` = a shown concern with 2 or more shown: chosen + the others as alternatives. `pick` = "none" or JSON null with 1 or more shown: `declined_all`. A pick that contradicts the move's concern records nothing. No `pick` key: a move's `concern_id` stands in. Otherwise no row; the `prime_thought` log line says why (`choice=no_concerns|one_concern_picked|pick_not_shown|pick_conflicts_move|no_pick|publish_failed`, or `chosen:published|declined_all:published`).
- `market_rank`: Prime holds concerns in insertion order, which is not a ranking. The real order is the body economy's funded-event list (`GET /physiology/event-wants`, highest net first; event mode is on in `.env_example`). Each concern's rank is its event's position in that list among the shown concerns. Concerns whose event is not listed (turn promises, day-letter threads, arc mode) get `None` (weight 1.0 in the reducer). Fewer than 2 listed: all `None`.
- `situation`: `body_read`, `event_mode`, `binding` (comma list), `binding_count`, `strain`, `strain_constraint`. Prime's frame shows no arousal reading; strain is the body-state number it does see, so that is what is recorded.
- `care_promoted`: always false here (the care term does not rank Prime's concerns yet).
- `act_id`: the thought act; `correlation_id`: None (no model-call id reaches the conclude step); `choice_id`: `prime:<thought_id>`.
- Holdback (`PRIME_HOLDBACK_FRACTION`): the arm is drawn after the pick and only decides whether a question is sent. A held-back question still records its choice, with no extra tag. A refused move also keeps its pick: Orion chose; the runtime refused the act.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q orion/prime/tests   -> 186 passed
(services/orion-mind) ... -q tests                                                               -> 365 passed
(services/orion-hub)  ... -q tests/test_turn_orchestrator_cockpit_hops.py                        -> 21 passed
... -q orion/mindloop orion/personality                                                          -> 164 passed
... -q orion/harness/tests/test_prime_block_prefix.py orion/prime                                -> 183 passed
scripts/check_env_template_parity.py      -> PASS
scripts/check_env_key_single_source.py    -> OK
scripts/check_metric_lineage.py --gate    -> exit 0
scripts/check_definition_drift.py --gate  -> PASS
```

Replay golden re-recorded: with the prompt's new step 4 and `"pick"` key mapped back to the old text and the new `choice=` log field removed, every part of every scenario equals the previous golden (checked by script). No replay scenario holds 2 concerns with a pick, so none publishes a choice.

## Evals run

```text
PYTHONPATH=. python orion/prime/evals/run_prime_block_eval.py  -> PASS: 0 failing check(s)
  new: tending_quiet_shown, tending_dropped_first, choice_point_chosen, choice_point_declined
PYTHONPATH=. python orion/prime/evals/run_prime_frame_eval.py  -> PASS
PYTHONPATH=. python orion/prime/evals/run_prime_wake_replay.py -> PASS
```

## Docker/build/smoke checks

```text
Not run: no deploy from a worktree. Live path UNVERIFIED until hub and orion-mind are rebuilt from main.
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on the full diff. All 8 findings fixed:

- Finding (blocker): a malformed `pick` (int, list, object, over 200 chars) failed the whole answer, losing the reading, concern edits and move.
  - Fix: a before-validator turns a non-string pick into a value no concern has and cuts a long one to 200 chars.
  - Evidence: `test_a_malformed_pick_never_fails_the_thought`, `test_a_long_pick_is_cut_not_refused`.
- Finding: an explicit pick that contradicts the move's concern (or "none" with a move about a concern) was recorded as a false decline or departure.
  - Fix: no row, reason `pick_conflicts_move`.
  - Evidence: `test_a_pick_that_contradicts_the_move_records_nothing`.
- Finding: Hub read the tending HTTP frame before the Redis reads (worst case ~2 s).
  - Fix: the read starts as a task and `fetch_prime_block` awaits it after its Redis reads (they overlap); a failed task means no lines.
  - Evidence: `test_fetch_prime_block_awaits_a_tending_task_and_survives_its_failure`; hub tests pass.
- Finding: one placed concern among unranked ones got rank 1 (weighs 0.25 as "followed the market").
  - Fix: ranks only when at least 2 shown concerns are in the market list.
  - Evidence: `test_one_placed_concern_is_not_the_markets_top_over_unranked_ones`.
- Finding (nit): "tending never costs a line" was checked by drop count, which misses a line-boundary cut.
  - Fix: require the block's own refs to be a subset of the refs kept with tending; refs deduped.
  - Evidence: eval `tending_dropped_first` PASS.
- Finding (nit): a label with `[thought:..]` would insert a ref-shaped string the block did not show.
  - Fix: brackets in labels become parentheses.
  - Evidence: `test_tending_label_carries_no_ref_shaped_text`.
- Finding (nit): `correlation_id` copied `choice_id`.
  - Fix: left None (no model-call id reaches `_conclude`; `act_id` names the call).
  - Evidence: `test_a_pick_among_two_concerns_publishes_one_choice_with_market_ranks_and_situation`.
- Finding (nit): Hub fail-open test lacked timeout and non-dict JSON.
  - Fix: both added (timeout returns in < 1 s with the timeout patched to 0.1 s).
  - Evidence: `test_read_prime_tending_is_fail_open`.

## Restart required

Deploy order: durable-runs first (it must already run #3321's consumer and reducer; it does if #3321 is deployed), then orion-mind, then hub.

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Live checks after deploy:

```bash
docker logs orion-mind 2>&1 | grep -o 'prime_thought .*choice=[a-z_:]*' | tail -5
psql ... -c "select choice_id, declined_all, chosen_market_rank, jsonb_array_length(alternatives) from personality_choice where lane='prime' order by occurred_at desc limit 5"
docker logs orion-athena-hub 2>&1 | grep 'prime_block corr' | tail -3   # tending=<n>
```

## Risks / concerns

- Severity: medium. Concern: the model may ignore `pick` at first. Mitigation: the move's `concern_id` stands in; the `choice=no_pick` count on `prime_thought` lines shows how often the pick is missing.
- Severity: medium. Concern: thoughts that hold 0 or 1 concern make few or no rows (the 2026-11-12 fixture held 0; live Redis on 2026-11-13 holds 2, thoughts_total 242). Mitigation: `declined_all` covers 1 concern + "none"; the rate is visible in the logs and in `personality_choice`. M1 (20 rows a day across lanes) is UNVERIFIED.
- Severity: low. Concern: the Hub block now waits on one more GET (1 s cap). Mitigation: it runs concurrently with the Redis reads, so the worst case stays about 1 s.
- Severity: low. Concern: no tendency has 10 weighted choices yet, so the tending line renders nothing at first. That is the intended rest state, not a fault.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3330

🤖 Generated with [Claude Code](https://claude.com/claude-code)
