## Summary

Integration patch 3, Prime slice (spec: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md`, Part A). Part of #3425.

- Before each thought, Prime reads Orion's now: what the other organs recorded since Prime last thought. It comes from durable-runs `GET /temporal-self/now?organ=prime&since=...`. The frame gets a new section, WHAT YOUR ORGANS LIVED SINCE YOUR LAST THOUGHT, with up to 12 items. Each item shows its `[experience:<id>]`, its salience, its surprise (when known) and up to 3 refs. Prime keeps the read's `shown_id`.
- `experience` is now a ref kind that Prime can cite: `refs: [{"kind": "experience", "id": ...}]` at the top level of the answer, or inside a move's refs. Experience refs are never forwarded to curiosity, because a curiosity run cannot open one.
- When the thought closes, Prime sends the shown experiences it cited (at most 3) to `POST /temporal-self/now/use` with the `shown_id`. A failure is logged and never fails the thought. The act note records both the proposal and the server's verdict, for example `proposed 1 experience cite(s), now_use stored 1/1`.
- The frame no longer reads Hub's curiosity dashboard (`/curiosity/api/runs`). Curiosity's runs reach Prime through the now. Settling Prime's questions still reads the dashboard. The cheap pending-focus check (patch 0, #3366) stays.
- `config/integration/organs.yaml`: the prime entry sets `reads_now: services/orion-mind/app/prime_loop.py` and drops its `exempt_read`. No other entry was changed.

## Outcome moved

Prime can now see, and cite, what other organs lived. Those organs are reading, curiosity, cameras, prediction surprise, reverie, street and others. Each cite is checked and stored as an `experience_use` row. Before this patch Prime had no checked read-back of other organs. This is the Prime part of the integration metric (experience use rate).

## Current architecture

- Prime's frame read the body, attention or event wants, energy, Hub's curiosity dashboard (slow, 4-44 s) and the pending focus.
- It had no read of Temporal Self experiences beyond arcs.
- Patch 1 (#3453) shipped the now endpoint and `record_use`. Patch 2 (#3454) shipped the integration gate. Prime was exempt from the gate's read-back rule.

## Architecture touched

- **orion-mind Prime loop** (`app/prime_loop.py`):
  - `_read_now`, `_now_since`, `_now_after_call` and `_record_now_use` were added.
  - `_read_curiosity` was removed from `_ask`.
- **Pure thought module** (`orion/prime/thought.py`):
  - `FrameInputs.curiosity` and `FrameInputs.curiosity_error` were removed. `orion_now` was added.
  - New: the now section, `EXPERIENCE_PROMPT`, `parse_refs` (splits experience refs from openable refs) and `shown_cites`.
  - `apply_answer(shown_experience=...)` and `Applied.experience_cites` were added.
- **Organ registry**: the prime entry, as described above.
- **Both servers are unchanged.** durable-runs and Hub get no code change.

## Files changed

- `orion/prime/thought.py`: the now section, experience cites, and a CURIOSITY section that shows only the cheap pending-focus read.
- `services/orion-mind/app/prime_loop.py`:
  - reads the now and remembers the last seen cut-off;
  - reuses an unseen now on a retry after a try that never reached the model;
  - posts the cites with a 10 s timeout;
  - no longer reads the dashboard in the frame.
- `config/integration/organs.yaml`: prime `reads_now`.
- `services/orion-mind/app/settings.py`, `.env_example`, `README.md`: comments say the dashboard is read for settling only; the README documents the now.
- `orion/prime/evals/run_prime_frame_eval.py`: a new `now` check. The frame no longer depends on the dashboard.
- `orion/prime/tests/test_prime_now_cites.py` (new), `orion/prime/tests/test_prime_core.py`: unit tests.
- `services/orion-mind/tests/test_prime_now.py` (new), `tests/test_prime_loop.py`: runtime tests and fake now pages.
- `services/orion-mind/tests/fixtures/prime_replay_golden.json`, `prime_replay_scenarios.py`, `test_prime_replay.py`: re-recorded golden plus one new scenario.
- `tests/test_check_integration_seam.py`: planted tests no longer assume that prime is exempt.

## Schema / bus / API changes

- Added: `experience` ref kind, inside Prime's answer only. It is not added to `orion/refs.py`, so the curiosity dig kit is unchanged.
- Removed: none from the contracts.
- Renamed: none.
- Behavior changed: Prime is now a client of `GET /temporal-self/now` and `POST /temporal-self/now/use`. Both already exist (patch 1).
- Compatibility notes:
  - `PrimeStateV1` is unchanged. The now's cut-off is kept in memory only, so rollback is safe.
  - No bus channel.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: comment only (`PRIME_CURIOSITY_DASHBOARD_TIMEOUT_SEC` is now for settling only).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes.
- Skipped keys requiring operator action: none.
- The flag is durable-runs' `TEMPORAL_SELF_NOW_ENABLED` (ships true). When it is off, the endpoint returns 404 and Prime shows no section.

## Tests run

```text
PYTHONPATH=. pytest -q orion/prime orion/mindloop orion/dig/tests/test_refs_contract.py tests/test_check_integration_seam.py  -> 292 passed
cd services/orion-mind && pytest -q tests                                                                                 -> 412 passed
python scripts/check_integration_seam.py   -> OK (16 organs; 43 families, 18 workflows, 1 intention adapters, 28 source kinds covered)
python scripts/check_env_template_parity.py -> rc 0
```

New tests:
- The section renders from a fake now: ids, salience, surprise, refs, and `since` / `run_ref` in the URL.
- Cites are parsed: top-level, inside a move, `experience:` prefix, duplicates.
- `record_use` receives only shown cites, at most 3, with the `shown_id` and `citing_ref`.
- A now failure leaves no section and the thought still closes with value 1. Covered cases: 404, 503, URLError, timeout and malformed pages.
- A failed POST still closes the thought.
- No `/curiosity/api/runs` GET happens in the frame path, and the pending-focus GET is still made.
- A no-model retry reuses its now, so there is one GET for two tries.
- The next read starts at the last seen `as_of`. A failed read skips nothing.
- A bad experience ref inside a move is a warning, not a refusal.
- A question that quotes a cited experience id is refused with its own reason.

## Evals run

```text
python orion/prime/evals/run_prime_frame_eval.py -> 13/13 PASS, including the new "now" check:
  a full now (12 items, TemporalNowV1-valid, 200-char labels, 8 refs each) gives a 7938-char frame (cap 12000),
  every id is shown, no dashboard runs appear, and the cites kept are 3 of 6 (1 not shown, cap 3)
run_prime_block_eval.py, run_prime_wake_replay.py, run_intention_take_on_replay -> rc 0
```

Replay golden:
- All 29 changed parts of the older scenarios were checked by script. After masking the CURIOSITY section, the `prime_now_unread reason=off` lines and `now_use=none`, they equal the old golden.
- One new scenario, `now_section_and_one_cite`, is recorded last, so no other scenario's ids moved.

## Docker/build/smoke checks

```text
No image build was run. requirements.txt is unchanged and there are no new imports outside the repo package.
Live smoke: UNVERIFIED. This needs the orion-mind rebuild after merge.
Read-only check of live durable-runs (2026-11-16): temporal_self_event has curiosity_run (28 / 2 d), reading_turn,
prediction_surprise and others. experience_shown holds 3 prime rows (inspection reads). experience_use holds 0 rows.
```

## Review findings fixed

The code review ran in a subagent. All findings were fixed:

- Finding: `test_prime_now_cites` used a 16-hex act id, which is invalid, so the test failed.
  - Fix: use a valid 24-hex id, and assert `problems == []`.
  - Evidence: `orion/prime` 252 passed.
- Finding: a malformed experience ref inside a move refused the whole move.
  - Fix: experience-ref problems in a move are now warnings. Only a bad arc/act ref refuses the move.
  - Evidence: `test_a_bad_experience_ref_inside_a_move_is_a_warning_not_a_refusal`.
- Finding: a question that quoted a cited experience id was refused with a misleading "outside refs" message.
  - Fix: a dedicated refusal message, plus a prompt line that says to describe the experience in words.
  - Evidence: `test_a_question_quoting_a_cited_experience_id_is_refused_with_words_that_say_why`.
- Finding: a try that never reached the model wrote extra `experience_shown` rows that read as "read, cited nothing".
  - Fix: the unseen now is reused on the retry (for up to 30 min).
  - Evidence: `test_a_try_that_never_reached_the_model_reuses_its_now_instead_of_reading_again`.
- Finding: `since = last_thought_at - 120 s` skipped experiences after a failed now read.
  - Fix: `since` is the `as_of` of the last now a model actually saw (in memory). After a restart it falls back to the old rule.
  - Evidence: `test_the_next_read_starts_where_the_seen_now_ended_and_a_failed_read_skips_nothing`.
- Finding: the use POST could outlive its 10 s `wait_for`, because the real urlopen timeout was 45 s.
  - Fix: a real 10 s urlopen timeout, and `wait_for` is gone.
  - Evidence: `test_the_real_use_client_has_its_own_short_timeout`.
- Finding: the act note counted cites before the server judged them.
  - Fix: the note now reads `proposed N experience cite(s), now_use stored K/N`.
  - Evidence: an assertion in the cites test, and the new replay scenario.
- Finding: the replay had no scenario with a now section.
  - Fix: added `now_section_and_one_cite`, recorded last.
  - Evidence: older scenarios are byte-identical.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

Run this from the clean primary `main` checkout after merge. durable-runs needs no restart.

## Risks / concerns

- Severity: medium.
  - Concern: the frame lost "investigate runs today X of cap, next eligible". The now does not carry it, so Prime cannot see that curiosity is at its daily cap. This follows the spec's retirement.
  - Mitigation: a cap refusal still reaches Prime through its outcome lines and the organ replies.
- Severity: low.
  - Concern: some `experience_shown` rows still were not seen by a useful thought:
    - a client timeout after the server already stored the row;
    - a preempted thought;
    - an answer that does not parse.
    These rows count as "read, cited nothing".
  - Mitigation: noted on #3498 with a fix for the coverage query.
- Severity: low.
  - Concern: the now's cut-off is in memory only. After a restart, a gap can be skipped if the last read before the restart failed.
  - Mitigation: none in this patch. Making it persistent would change `PrimeStateV1` (extra=forbid), which would make rollback unsafe.
- Severity: medium.
  - Concern: the live path is UNVERIFIED. The patch 3 acceptance check is `experience_use` rows with `citing_organ = prime` within 48 h of deploy, each citing another organ's event.
  - Mitigation: check `select citing_organ, count(*) from experience_use group by 1;` and the `prime_now_read` / `prime_now_use` log lines after the rebuild.
- Issues: commented on #3498. Part of #3425 (not closed: the reverie and curiosity slices are separate PRs). No issues opened or closed.

## PR link

See the PR this report belongs to.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
