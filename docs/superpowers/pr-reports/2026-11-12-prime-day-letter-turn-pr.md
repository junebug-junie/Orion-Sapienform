# The day letter is Prime's end-of-day turn (living Prime Part 3)

**This PR contains and depends on #3229 (Prime block), #3230 (turn hand-back) and #3231 (honest letter input).**
It was built on a merge of those three branches. Merge them first; this PR's GitHub diff then shrinks to Part 3
only. To review Part 3 alone now: `git diff 47520a070..HEAD` (47520a070 = the merge of the three base branches).

Design: PR #3224, `docs/superpowers/specs/2026-11-12-living-prime-one-orion-design.md`, Part 3. Juniper signed
off; where the spec's missing questions carry a recommendation, this follows it (a refused bid means no letter
that day and the next letter says so; reminiscence is out of scope).

## Summary

- Prime now writes the day letter itself. When the temporal chronicle has closed yesterday and there is no
  letter, Prime bids a `make` act in the body economy. If the bid is funded, Prime opens the act and orders the
  letter from Hub. If the bid is never funded inside the bid window, there is no letter that day, and the next
  letter is told so.
- The letter is ONE unified turn on the durable turn seam, not two bare cortex verb calls. The turn gets every
  grounding primitive a chat turn has, the Prime block in its prefix (Part 1), and the day's honest input
  (#3231: closed-day block, temporal core, Juniper in full, reveries by theme, self-sense once, trimmed tail).
  It may dig and recall; facts from tools must say where they came from.
- The letter row, journal entry (stable id) and email work as before. `sources` now also holds the turn's
  correlation id, Prime's act id and the missed days the turn was told about (the per-source shares came with #3231).
- The letter's closing threads become Prime concerns (at most 2 per letter, never past Prime's 3-concern cap).
  Curiosity hears about them only through Prime's existing, graded `ask`.
- Retired, kill-means-kill: verbs `orion_day_note_v1` and `orion_day_carry_forward_v1` (their prompts, cortex-exec
  token budgets and verb family), Hub's letter timer (`scripts/orion_day_letter.py`), and the carry-forward
  one-time offer in the curiosity kickoff. A grep gate test keeps them out.
- The Prime block cites `thought:<last_thought_id>` (Part 1 fell back to the thought count).

## Outcome moved

Before: a timer in Hub fired at 08:30 and a bare model call with a one-line identity wrote the letter from a
pile; a second call wrote threads that went straight into the next curiosity run. Nothing in Prime knew a letter
existed. After: the letter is an act of Prime's, priced by the body economy, written by the same unified turn
that speaks in chat, recorded as a graded `make` act, and its threads go back into Prime.

## Current architecture (before this patch)

- `services/orion-hub/scripts/orion_day_letter.py`: timer; gathered the day, submitted `orion_day.letter`, retried
  attempts, emailed the row.
- `services/orion-durable-runs/app/orion_day_graph.py`: `write_note` (verb `orion_day_note_v1`) ->
  `write_carry_forward` (verb `orion_day_carry_forward_v1`) -> `persist`.
- `orion/orion_day/carry_forward.py` + Hub curiosity kickoff: offered `carry_forward_md` once to the next run.
- Prime (`services/orion-mind/app/prime_loop.py`): no part in the letter.

## Architecture touched

- **orion-mind** (`app/prime_loop.py` `_day_letter_*`, pure `orion/prime/day_letter.py`): day close -> make bid ->
  act + order -> settle from `GET {durable}/runs/<run_id>` -> threads to concerns. Own Redis record
  `orion:prime:day_letter`. Kill switch `PRIME_DAY_LETTER_ENABLED` (ON).
- **orion-hub**: `scripts/orion_day_order.py` (status + order endpoints, the only submit path),
  `scripts/orion_day_mailer.py` (email half, unchanged behavior: run-state hook + slow sweep). The letter turn's
  source is not framed as a curiosity run.
- **orion-durable-runs**: `orion_day.letter` = `write_letter` (one harness turn, `harness_smoke_graph` failure
  policy) -> `persist` -> `finish`; the harness turn is cancelled by its checkpoint-derived id on worker death.
- **orion-cortex-exec**: verb special cases and budgets removed.
- **Contracts**: `orion/schemas/orion_day.py`, `orion/schemas/registry.py`, `orion/orion_day/letter_turn.py`.

## Files changed

- `orion/schemas/orion_day.py`: new nodes, letter session/source, `MissedLetterV1`, brief `prime_act_id` /
  `missed_letters`, sources `turn_correlation_id` / `prime_act_id` / `missed_letters`, order/status models.
- `orion/schemas/registry.py`: the four new models registered.
- `orion/orion_day/letter_turn.py` (new): the turn's user message (lifted writing instructions) and the reply split.
- `orion/orion_day/brief.py`: carry-forward TTL removed; Prime fields; context need for one turn.
- `orion/prime/day_letter.py` (new), `orion/prime/block.py`: the end-of-day move; thought ref.
- `services/orion-mind/app/prime_loop.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-durable-runs/app/orion_day_graph.py`, `admission_runtime.py`, `run_acts.py`, `README.md`.
- `services/orion-hub/scripts/orion_day_order.py` (new), `orion_day_mailer.py` (new), `orion_day_letter.py`
  (deleted), `main.py`, `curiosity_investigation.py`, `orion_day_email.py`, template, settings, `.env_example`, README.
- `orion/curiosity/kickoff_prompt.py`, `orion/orion_day/carry_forward.py` (deleted): the offer retired.
- `orion/cognition/verbs/orion_day_*_v1.yaml`, `orion/cognition/prompts/orion_day_*_v1.j2` (deleted),
  `config/physiology/families.yaml`, `services/orion-cortex-exec/*`: verbs retired.
- Tests/evals: listed below. `.github/workflows/orion-day-letter-hub-tests.yml`, `orion-prime-tests.yml`.

## Schema / bus / API changes

- Added: `MissedLetterV1`, `OrionDayLetterOrderV1`, `OrionDayLetterOrderResultV1`, `OrionDayLetterStatusV1`
  (HTTP, registered); Hub `GET /api/orion-day/letters/{date}`, `POST /api/orion-day/letter-run`; brief fields
  `prime_act_id`, `missed_letters`; sources fields `turn_correlation_id`, `prime_act_id`, `missed_letters`;
  status API `orion_day` block: `letter_ready`, `threads`, `turn_correlation_id`, `prime_act_id`, `attempt`.
- Removed: `ORION_DAY_NOTE_VERB`, `ORION_DAY_CARRY_FORWARD_VERB`, brief `carry_forward_ttl_hours`, status
  `note_ready` / `carry_forward_ready` / `llm_attempts`.
- Behavior changed: `orion_day.letter` graph nodes; `carry_forward_md` now holds the letter's own threads;
  `carry_forward_expires_at` written NULL.
- No bus channel added (the order is HTTP; the letter turn hands back on the existing `orion:cockpit:hop`).
- No new `DurableWorkflowV1` value: the existing `orion_day.letter` workflow is reused.
- Compatibility: additive fields on forbid models -> deploy order below. The table is unchanged.

## Env/config changes

- Added keys: orion-mind `PRIME_DAY_LETTER_ENABLED=true`, `PRIME_DAY_LETTER_BID_WINDOW_HOURS=18`,
  `PRIME_DAY_LETTER_TURN_TIMEOUT_SEC=1800`, `PRIME_DAY_LETTER_RUN_DEADLINE_HOURS=12`; orion-hub
  `HUB_ORION_DAY_EMAIL_SWEEP_SEC=900`.
- Removed keys: orion-hub `HUB_ORION_DAY_HOUR_LOCAL`, `HUB_ORION_DAY_MINUTE_LOCAL`, `HUB_ORION_DAY_TICK_SEC`,
  `HUB_ORION_DAY_MAX_ATTEMPTS`, `HUB_ORION_DAY_TIMEOUT_SEC`, `HUB_ORION_DAY_CARRY_FORWARD_TTL_HOURS`,
  `HUB_CURIOSITY_CARRY_FORWARD_ENABLED`; orion-cortex-exec `LLM_ORION_DAY_NOTE_MAX_TOKENS`,
  `LLM_ORION_DAY_CARRY_FORWARD_MAX_TOKENS`.
- Meaning changed: `HUB_ORION_DAY_ENABLED` now gates Prime's orders and the mailer (Hub has no timer).
- `.env_example` updated: orion-mind, orion-hub, orion-cortex-exec.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. orion-mind +4 keys (default
  run); orion-hub +`HUB_ORION_DAY_EMAIL_SWEEP_SEC` (needed `--all-keys`). The script cannot remove keys, so the
  retired keys were removed from the primary checkout's local `.env` files by hand (backups in the session
  scratchpad).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/prime (incl. new test_day_letter.py, test_prime_block.py) + orion/harness/tests/test_prime_block_prefix.py: 145 passed
services/orion-mind/tests (incl. new test_prime_day_letter.py, 11 tests): 255 passed
orion/orion_day/tests (incl. new test_letter_turn.py, test_retired_letter_verbs.py) + orion/curiosity/tests: 399 passed, 29 skipped
services/orion-durable-runs/tests (full suite, -x): passed; test_orion_day_graph.py 19, arc_citations 13, listener_wait 39, verb_text 9
services/orion-durable-runs/tests/test_orion_day_postgres.py: 3 skipped locally (needs ORION_ADMISSION_TEST_DSN; runs in CI)
Hub: test_orion_day_order.py 10, test_orion_day_mailer.py 31, test_orion_day_correction_listener.py 29,
     test_curiosity_investigation.py 153, test_orion_day_what_i_read 3, introspect_listener 24, consult_loop 21: all passed
scripts/check_verb_physiology.py: OK; scripts/check_env_template_parity.py: PASS
```

Acceptance checks for Part 3 that are testable offline, and where each is tested:

- A day closes -> Prime opens a `make` act: `test_a_closed_day_without_a_letter_opens_a_make_act_and_orders_one_turn_with_the_right_inputs`.
- Refused bid -> no letter, next letter told: `test_a_deferred_bid_writes_nothing_and_past_the_window_the_next_letter_says_so`,
  `test_prompt_names_each_missed_day_once`, `test_the_day_before_with_no_row_is_named_even_when_prime_did_not`.
- One turn submitted with the right inputs: `test_the_turn_request_is_the_letter_turn` (durable),
  `test_prompt_holds_instructions_closed_day_block_temporal_core_and_the_whole_record`,
  `test_an_order_builds_the_brief_and_submits_one_run_under_prime_s_ids`.
- Row with shares + correlation id: durable graph happy-path test, `test_letter_run_holds_the_agent_card_for_its_one_turn_and_persists_once` (Postgres, CI).
- Threads become concerns within caps: `test_a_written_letter_closes_the_act_at_1_and_its_threads_become_concerns_within_the_caps`,
  `test_thread_concerns_never_pass_max_concerns`.
- No call to the retired verbs: `orion/orion_day/tests/test_retired_letter_verbs.py` (grep gate over code, config, CI).

## Evals run

```text
orion/orion_day/evals/run_orion_day_eval.py: 8/8 PASS (letter_prompt, grounding, blind, juniper_full, full_text,
  condensation, budget: letter turn needs ~33k < 131k tokens, determinism)
services/orion-hub/evals/run_orion_day_email_eval.py, run_orion_day_reread_eval.py, run_orion_day_correction_eval.py: PASS
orion/prime/evals/run_prime_block_eval.py: PASS
```

## Docker/build/smoke checks

```text
Not run. No container was built or deployed (not asked; deploys go from clean main after merge).
```

Live checks, all UNVERIFIED until deployed (run the morning after deploy; dates are the letter's day):

```sql
-- 1. Prime opened a make act for the day, and how it closed.
SELECT act_id, verb, object_ref, opened_at, closed_at, outcome_value, outcome_note
  FROM grammar_acts WHERE actor = 'prime' AND verb = 'make' ORDER BY opened_at DESC LIMIT 5;
-- 2. The bid(s) the market decided.
SELECT want_id, outcome, reason, decided_at FROM want_ledger
 WHERE family = 'prime' AND verb = 'make' ORDER BY decided_at DESC LIMIT 10;
-- 3. The letter row: shares, the turn's correlation id, Prime's act id, missed days.
SELECT letter_date, run_id, sources->'input_shares', sources->>'turn_correlation_id',
       sources->>'prime_act_id', sources->'missed_letters', emailed_at
  FROM orion_day_letter ORDER BY letter_date DESC LIMIT 3;
-- 4. The turn ran as a real harness turn: its hops are in cockpit_turn_sighting, and the run's
--    completed event says harness_step_count > 0.
SELECT seq, stage, status, ts FROM cockpit_turn_sighting
 WHERE correlation_id = '<sources.turn_correlation_id from 3>' ORDER BY seq;
SELECT payload->'detail'->>'harness_step_count' AS steps, payload->'detail'->'threads' AS threads
  FROM durable_resource_events
 WHERE run_id = 'orion-day-<date>-1' AND payload->>'event' = 'run.completed';
```

```bash
docker logs orion-athena-mind 2>&1 | grep -E "prime_day_letter_(due|funded|ordered|closed|not_written|bid_not_funded)"
docker logs orion-hub 2>&1 | grep -E "orion_day_order |orion_day_emailed"
redis-cli -u "$ORION_BUS_URL" GET orion:prime:day_letter
# Threads became concerns, and the next Prime ask cites one:
redis-cli -u "$ORION_BUS_URL" GET orion:prime:state | python3 -c "import json,sys; s=json.load(sys.stdin); print([c for c in s['concerns'] if (c.get('event_ref') or '').startswith('orion_day_letter:')])"
```

## Review findings fixed

Code review ran in a subagent on `git diff 47520a070..HEAD` (Part 3 only). No blockers; 5 should-fix, 7 nits.

- Finding: a paused or stuck run left the day open forever and blocked every later letter.
  - Fix: past the run deadline + 1 h, a non-terminal run closes the day `not_written` ("still <status> past its deadline").
  - Evidence: `test_a_paused_run_past_its_deadline_closes_the_day_instead_of_blocking_every_later_letter`.
- Finding: missed days were marked "told" when the order was accepted, so a failed run silently dropped them.
  - Fix: marked told only when the letter that carried them is written.
  - Evidence: `test_missed_days_stay_untold_when_the_run_that_carried_them_fails`, `..._are_told_once_the_letter_carrying_them_is_written`.
- Finding: a completed run whose row another run wrote was recorded as a missed day.
  - Fix: `day_letter.done_outcome` maps `persist_outcome == already_written` to `already_written`.
  - Evidence: `test_a_row_another_run_wrote_is_a_letter_not_a_missed_day`.
- Finding: removing `carry_forward_ttl_hours` from the forbid brief would fail an in-flight old-timer run and an old Hub; rollout comments contradicted each other.
  - Fix: field kept as accepted, ignored, excluded on dump (documented deprecated); comments state one order (Hub, durable-runs, orion-mind); deploy step checks `/runs/unfinished` first.
  - Evidence: `test_a_brief_from_the_retired_timer_still_validates_and_its_ttl_is_dropped`.
- Finding: Hub mapped a durable-runs 5xx to a permanent `refused`.
  - Fix: 5xx -> `unavailable` (Prime re-sends); 4xx stays `refused`.
  - Evidence: `test_409_is_exists_other_refusals_are_refused_with_detail` (503 case).
- Nit fixed: an order whose answer was lost is followed if the run exists before Prime gives up (`test_an_order_whose_answer_was_lost_is_followed_when_the_run_exists`).
- Nit fixed: empty days are passed to the next order with their real reason, so Hub never calls them "missed for no known reason".
- Nit fixed: a day not yet closed by the chronicle logs `prime_day_letter_day_not_closed`.
- Nit fixed: order HTTP timeout 90 s -> 45 s (the tick awaits it).
- Nit fixed: the Postgres test now runs Prime's `run_outcome` on the real status API dict (producer/consumer pin, runs in CI).
- Not changed (nits, documented under risks): a Hub `refused`/`already_written` race still closes the make act at 0; a run resumed after its letter checkpoint takes a GPU hold `persist` does not need (harmless, released at finish).

## Restart required

Deploy order across all four PRs, from the clean primary `main` checkout after all four are merged, outside a
running letter (first check `curl -s http://127.0.0.1:8124/runs/unfinished` has no `orion_day.letter` run: a run
checkpointed at the old `write_note` / `write_carry_forward` nodes cannot resume on the new graph):

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

Why this order: the governor first (#3229's request field); Hub before durable-runs (Hub reads the new `sources`
fields with forbid models, and Hub sends the new brief fields only on Prime's order); durable-runs before
orion-mind (Prime's first order must reach the new graph); cortex-exec any time after durable-runs (nothing calls
the retired verbs any more); orion-mind last (Prime starts ordering). Between the Hub deploy and the orion-mind
deploy no letter is written: do them in one sitting.

## Risks / concerns

- Severity: medium. Concern: the letter now goes through the unified turn's finalize step (judge, grounding repair). A
  ~2,000-word letter has never been run through it; finalize may shorten or "repair" it, or judge record facts it
  cannot see. UNVERIFIED until the first live letter. Mitigation: read the first letter next to its `harness_turn_trace`;
  the 400-char floor fails an empty or stub reply as an attempt.
- Severity: medium. Concern: the `make` act is family `prime`, so Prime's 30-minute agent-lane letter shares a priced
  history with its 1-minute chat-lane thoughts (the spec says actor prime). Mitigation: costs are per lane; watch
  `want_ledger` for prime `infer` bids deferring more after deploy.
- Severity: low. Concern: Juniper's constraint 7 acceptance (old vs new letters side by side, her read) is not done
  by this PR; #3231 ships the replay tool. Mitigation: run `run_orion_day_input_replay.py --dump` after merge.
- Severity: low. Concern: a Hub `refused` (flag off) or an `already_written` race still closes the act at 0, which
  counts against `act.prime.make` in the self-monitor. Mitigation: rare; the close note names the cause.
- Severity: low. Concern: between the Hub deploy and the orion-mind deploy no letter is written. Mitigation: deploy in one sitting.

## PR link

(filled after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
