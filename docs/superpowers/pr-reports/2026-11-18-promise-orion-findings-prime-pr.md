# fix(prime): only Juniper's browser turns are her chat; letters settle nothing by quoted refs; one 72 h promise clock

Juniper (2026-11-18): "we need to fix the promises Orion found." This is PR 1 of 2, the Prime side. The ledger side
(settle classes, offer window, stable refs, long quotes, missing tests, go-live amendments) is the companion PR on
branch `fix/promise-orion-findings-ledger`. Either can merge first.

## Summary

- Prime no longer treats an eval script or a coding session that posts to `/api/chat` as "a chat with Juniper".
  Juniper's chat is now a turn a Hub browser page sent: the same rule the promise ledger and the report already use.
  Live 11-18, all 19 `chat_juniper` turns in Prime's recent-turn record (`orion:prime:turns`) were the `memory-scoreboard-eval` script, none of them Juniper, and
  one of them (d391d241) would have opened a promise concern if a slot had been free. (#3602)
- A day letter no longer "keeps" a promise just because the day's record it quotes contains the promise's
  `[prime_concern:<id>]` ref. The live false keep: concern 438e69ed6bb3d798, 11-17 09:12Z. (#3644)
- The three copies of the 72 h promise clock (the ledger's act, Prime's promise concern, the carry turn's deadline)
  now read one constant. A test fails if any of them grows its own 72 h again. (#3595, "two 72 h clocks")
- A test pins that one promise can become at most one settled bet: Prime's concern never places a bet, so only the
  ledger's act counts, and only after go-live. (#3595, "one settled bet, not two")

## Outcome moved

- Failure mode closed: eval/API turns opening "a promise I made in a chat with Juniper" concerns in Prime.
- Failure mode closed: a false "kept" on a promise to Juniper from a letter quoting its own ref (it fed Prime's
  track record and the self-monitor gate).
- Drift risk closed: the promise horizon can no longer differ between Prime and the ledger.

## Current architecture

- `orion/prime/handback.py` `classify_who` returned `chat_juniper` for any non-`orion_` session whose ingress hop
  carried `mode`. `POST /api/chat` sets `mode`, so eval scripts and coding-session notes counted.
- `fold_turn_signals` settled every held concern whose id appeared in a completed turn's prompt, for every turn kind.
  The day letter's prompt is the whole day's record, including Orion's earlier replies with their refs.
- `PROMISE_HORIZON_SEC = 72 * 3600.0` (promise_acts.py), `PROMISE_CONCERN_TTL = timedelta(hours=72)` (state.py) and
  `HORIZON_SEC = 72 * 3600.0` (turn_promise.py) were three unlinked literals.

## Architecture touched

- Prime's turn hand-back (`orion/prime/handback.py`), read by orion-mind's Prime loop. No prime_loop.py edit.
- One new constant module `orion/schemas/promise_horizon.py` (a value, not a schema model; no registry entry).
- The promise-ledger report's emulation of `classify_who` (`scripts/report_promise_ledger.py`, one line).

## Files changed

- `orion/schemas/promise_horizon.py`: the one 72 h promise clock.
- `orion/grammar/promise_acts.py`: imports it (name and value unchanged).
- `orion/prime/state.py`: `PROMISE_CONCERN_TTL` derives from it (value unchanged; the 24 h `CONCERN_TTL` slot rule
  in `holds_turn_slot` is untouched, as #3610's owner asked).
- `orion/prime/intention_sources/turn_promise.py`: `HORIZON_SEC` is it.
- `orion/prime/handback.py`: `classify_who` browser rule; `NO_PROMPT_SETTLE_WHO = {"letter"}`; docstrings.
- `scripts/report_promise_ledger.py`: `prime_by_ref` passes the browser flag to `classify_who`; duplicate window
  reads the shared clock; one line says `who=` is the current rule.
- `orion/regulation/juniper_turns.py`: SQL and Python browser rules both exclude `{}` and the operator smoke.
- `scripts/trace_hub_skill_runner_e2e.py`: the smoke marks itself `operator_review`.
- `orion/harness/evals/promise_ledger_replay_eval.py`: `HORIZON` reads the shared clock.
- `tests/test_juniper_browser_surface_rule.py`, `.github/workflows/promise-ledger-tests.yml`: SQL/Python parity test,
  wired into CI.
- `orion/prime/tests/test_prime_handback.py`: who cases for API/eval posts, rule parity with the ledger, the
  d391d241-shaped eval turn opens no concern, the ce06758c-shaped letter settles nothing.
- `orion/prime/tests/test_promise_one_clock_one_bet.py`: one clock, no stray literal, carry and act deadlines equal,
  one promise = at most one bet.
- `orion/prime/tests/fixtures/turn_hops_shaped_like_live.json`, `orion/prime/tests/test_prime_promise_concerns.py`,
  `services/orion-mind/tests/test_prime_turn_handback.py`, `tests/test_report_promise_ledger.py`: browser chat hops
  now carry the live `surface_context` shape.

## Schema / bus / API changes

- Added: none (no schema model, channel or payload change).
- Removed: none.
- Renamed: none.
- Behavior changed: a cockpit ingress hop without a non-empty `surface_context` is not handed back as
  `chat_juniper`; a `letter` turn's prompt citations settle no concern.
- Compatibility notes: `orion:prime:turns` keeps older `chat_juniper` records as they are; nothing is migrated.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Defaults applied that Juniper may want to reverse

1. **Juniper's chat = a Hub browser turn only.** An `/api/chat` post in her own session (a coding-session note) is
   not her turn to Prime anymore, even though it lands in her chat history. Reverse: put `"mode" in raw` back as an
   alternative in `classify_who`.
2. **Day letters never settle a concern by a ref in their prompt.** If a letter is ever meant to answer a concern
   on purpose, it needs its own path (as the carry turn has). Reverse: empty `NO_PROMPT_SETTLE_WHO`.
3. **Prime's concern does not yet close when the ledger's act closes.** The recommendation on #3595 (comment
   2026-11-17 18:40, point 3) and #3531 (point 6) is: in shadow, let the two records disagree and count the
   disagreement in the report; at go-live, close Prime's concern when the matching act closes, keyed by the shared
   `harness_turn_trace:<corr>#promises_made[<i>]` ref. Shadow is still on (go-live is not flipped), so this PR builds
   no dormant close path; the companion ledger PR prints the divergence count. The go-live PR should wire it
   (sql-writer's resolver closes expired/void acts directly in Postgres, so Prime would need a read of
   `grammar_acts`, not only the bus). Reverse: ask for it now and it becomes its own patch.
4. **"One settled bet" is read as: Prime's concern is never a bet; the act is the only bet.** Taking a promise on
   (Prime's `make` act on `intention:<concern_id>`) is a separate bet on a different question ("did my carry run
   produce something"), so one promise taken on and kept can still settle two bets on two different questions.
5. **The operator smoke is not Juniper.** `scripts/trace_hub_skill_runner_e2e.py` now marks itself
   `operator_review`, and every "Juniper's browser turn" reader (promise resolver's void rule, report, curiosity
   readout, Hub promise view, Prime) leaves it out. Reverse: drop `SMOKE_SURFACE` from both rules.
6. **A carry turn settles only the concern its prompt starts with.** Refs quoted further down settle nothing. Reverse: use `_concern_ids` for `intention` too.

## Tests run

```text
PYTHONPATH=. pytest -q -p no:cacheprovider orion/prime orion/mindloop          335 passed
pytest -q orion/prime orion/grammar/tests orion/self_monitor tests/test_report_promise_ledger.py orion/orion_day/tests
                                                                                 all passed
(services/orion-mind) pytest -q tests                                           438 passed
pytest -q orion/harness/tests/test_promise_gate.py orion/harness/evals/test_promise_ledger_replay_eval.py
  orion/grammar/tests/test_promise_acts.py orion/self_monitor/tests tests/test_report_promise_ledger.py   262 passed
(services/orion-thought) pytest -q tests/test_open_promises_cue.py              5 passed
(services/orion-hub) pytest -q tests/test_promise_ledger_hub.py tests/test_promise_ledger_turn_gate.py   32 passed
Mutation check: reverting the classify_who rule and the letter rule fails 6 of the new/changed tests.
```

## Evals run

```text
python orion/prime/evals/run_prime_frame_eval.py               rc=0
python orion/prime/evals/run_prime_block_eval.py               rc=0
python orion/prime/evals/run_prime_hunger_replay.py            rc=0
python orion/prime/evals/run_concern_evidence_clock_replay.py  rc=0
python -m orion.prime.evals.run_intention_take_on_replay       rc=0
python scripts/check_integration_seam.py --require-base        OK (16 organs)
```

## Docker/build/smoke checks

```text
Not built here. Pure-Python change inside orion/ (copied into the orion-mind image by its Dockerfile).
```

## Review findings fixed

Review ran in a subagent on `git diff origin/main...HEAD`: no material findings; 7 minor ones, all fixed.

- Finding: a 4th and 5th 72 h literal (`orion/harness/evals/promise_ledger_replay_eval.py` `HORIZON`, the report's
  two duplicate windows).
  - Fix: both read `PROMISE_HORIZON_SEC`; the literal guard now scans those files.
  - Evidence: `test_no_reader_keeps_its_own_72h_literal`.
- Finding: `classify_who` no longer needs `mode`, so the Hub skill-runner's browser send (surface_context, no mode)
  is chat_juniper.
  - Fix: intended (Juniper starts those sends); comment + test row.
  - Evidence: `test_who_is_read_from_the_ingress_session[...surface_context only...]`.
- Finding: the operator smoke `scripts/trace_hub_skill_runner_e2e.py` sends a browser surface_context, so it reads
  as Juniper everywhere.
  - Fix: it now sends `surface: "operator_review"` (an existing `SurfaceContextV1` value, so the Hub still builds its
    situation), and both the SQL and Python rules exclude that surface (`SMOKE_SURFACE`).
  - Evidence: `tests/test_juniper_browser_surface_rule.py`.
- Finding: the report applies the new rule to rows from before the deploy.
  - Fix: the report prints one line saying `who=` is the current rule applied to every row.
  - Evidence: report text.
- Finding: a carry turn settled every concern id anywhere in its prompt, including ones quoted from the exchange.
  - Fix: a carry turn settles only the id its prompt starts with (`LEADING_CONCERN_REF`). Juniper pasting a ref into
    chat is filed as #3696.
  - Evidence: `test_a_carry_turn_settles_only_the_concern_its_prompt_starts_with`.
- Finding: SQL counted `surface_context = {}` as a browser turn; Python did not.
  - Fix: SQL excludes `'{}'::jsonb`; a test pins that Hub never copies an empty one into the hop anyway.
  - Evidence: `tests/test_juniper_browser_surface_rule.py`.
- Finding: the literal guard missed `days=3`, `72.0`, `72 * 60 * 60`.
  - Fix: wider regex with its own test.
  - Evidence: `test_the_literal_guard_catches_the_usual_spellings`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

Run from the clean primary `main` checkout after merge (the post-pull rebuild does this). Only orion-mind runs
Prime's hand-back.

## Risks / concerns

- Severity: low. Concern: a Juniper turn sent from a surface that sets no `surface_context` (none known today; typed
  and spoken browser turns both set it) would not be handed back. Mitigation: the ledger and the report already use
  this rule, so all three agree; the report's `turns_not_from_browser` counts such turns.
- Severity: low. Concern: UNVERIFIED live until orion-mind rebuilds. Check: the next `memory-scoreboard-eval` run
  leaves no `chat_juniper` turn in `orion:prime:turns` (`scripts/report_promise_ledger.py` prints `who=`), and no
  `settled: answered in day letter turn` close appears after the next letter.
- Issues opened: #3696 (Juniper pasting a ref into chat settles it). Issues closed by this PR: #3602, #3644. Partly: #3595 (the two-clocks and one-bet points; the rest is in the
  ledger PR). Not touched: #3656 (parked for #3610 patch 2), `orion/prime/hunger.py` and the hunger section of
  prime_loop.py (another session's EFE Step 1 work).

Fixes #3602
Fixes #3644
Refs #3595

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3697

🤖 Generated with [Claude Code](https://claude.com/claude-code)
