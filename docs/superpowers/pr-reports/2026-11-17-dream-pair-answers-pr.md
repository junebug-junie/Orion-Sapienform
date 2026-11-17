## Summary

- Every dream sleep now saves what the model said for each idea pair it tested: the pair, the arm,
  how the answer was counted, and its first 300 characters. (`PairAnswerV1`,
  `DreamCycleV1.pair_answers`, stored inside `dream_cycle.cycle_json`.)
- "Copied the input back" (the echo guard) is now its own count, `echo_count`. `no_link_count` now
  means only that the model said `{"link": false}`.
- The sleep log line, the act-ledger close note and the `/cycle` readout show `echo`. The readout
  also lists the pair answers.
- The Hub Dream tab shows an Echo count ("—" for older sleeps) and a "Pair answers" list, escaped.
- Patch 0 of `docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md` (PR #3605). Refs #3412.
  It does not fix #3412 yet; it makes the cause readable.

## Outcome moved

Before: 8 sleeps from 2026-11-13 07:02Z to 11-17 00:14Z each recorded 0 links out of 4. Nothing
showed whether the model declined or echoed, and the gateway logs from that window are gone. After
deploy, the next sleep's `cycle_json->'pair_answers'` shows the answers. #3412 can then be fixed
from evidence.

## Current architecture

`recombine.recombine` asked the model once per pair and counted `no_link` for both an explicit
decline and a claim that only pasted a fragment back. The answer text was thrown away. The cycle row
kept counts only.

## Architecture touched

orion-dream, the shared `orion/schemas/dream_cycle.py` model, and the Hub Dream tab (static JS). No
table, channel or env change.

## Files changed

- `orion/schemas/dream_cycle.py`: `PairAnswerV1`, `PairOutcome`, `echo_count`, `pair_answers`.
- `services/orion-dream/app/recombine.py`: record each pair's outcome and answer; split echo;
  strip NUL and lone surrogates before storing (`_storable`).
- `services/orion-hub/static/js/dream-tab.js`: Echo cell and Pair answers list.
- `services/orion-hub/evals/dream_browser.cjs`: fixture + assertions for both, and the old-row case.
- `services/orion-dream/app/cycle.py`: pass `echo_count` and `pair_answers` into the cycle; log echo.
- `services/orion-dream/app/main.py`: readout carries `echo` and `pair_answers`.
- `services/orion-dream/app/act_emit.py`: close note carries `echo`.
- `services/orion-dream/tests/test_dream_cycle_v2.py`: outcomes per pair, 300-char cap, JSON round
  trip, old rows still load, NUL/surrogate answers stored clean, settings caps fit `MAX_HYPOTHESES`.
- `services/orion-dream/tests/test_physiology_shadow_bid.py`: the bid-invariance view leaves out
  pair answer text (seeded by the random cycle id, same as hypotheses) and keeps the outcomes.

## Schema / bus / API changes

- Added: `PairAnswerV1` (nested in `DreamCycleV1`, not registered alone, same as
  `DreamHypothesisV1`); `DreamCycleV1.echo_count`, `DreamCycleV1.pair_answers` (both defaulted).
- Removed: none. Renamed: none.
- Behavior changed: `no_link_count` no longer includes echoes. Rows before this deploy cannot be
  split; the field comment says so.
- Compatibility notes: no reader strict-validates `cycle_json` outside orion-dream (checked: Hub
  `dream_routes.py` passes it through, Temporal Self reads only `pressure`). Old rows load with
  `pair_answers=[]`, `echo_count=0` (tested).

## Env/config changes

None. `.env_example` unchanged; no sync needed.

## Tests run

```text
services/orion-dream tests+evals: 403 passed, 9 skipped
services/orion-hub tests/test_dream_routes.py + tests/test_dream_hypotheses.py: 30 passed
orion/schemas/tests: 78 passed
node --check dream-tab.js, dream_browser.cjs: ok
Mutation: NUL test fails with _storable removed; browser check fails with the Echo label broken
check_env_template_parity.py: PASS
git diff --check: clean
```

## Evals run

`services/orion-dream/evals` ran with the suite above (included in the 403). No new eval: the proof
is the next live sleep's stored answers.

## Docker/build/smoke checks

Dream tab browser eval (`evals/dream_server.py` fixture server + `evals/dream_browser.cjs`, headless
Chrome): `{"passed":true, ... 15 checks, "requests":28}`. No Docker build: pure Python and static
JS in existing image paths; the live check is after deploy (below).

## Review findings fixed

Review by a subagent (by hand; the skill was not loadable there).

- Finding: a NUL in any model answer becomes `\u0000` in `cycle_json`; Postgres jsonb refuses it
  (checked live with a read-only SELECT), so the sleep's whole transaction would be lost.
  - Fix: `_storable()` strips NUL and replaces lone surrogates before storing.
  - Evidence: `test_an_answer_with_nul_or_lone_surrogate_is_stored_clean`; fails when the fix is removed.
- Finding: the Hub Dream tab would show echoes nowhere (counts would no longer add up to pairs) and
  never drew the stored answers.
  - Fix: Echo cell and Pair answers list in `dream-tab.js`.
  - Evidence: browser eval asserts both, plus "—" and no list on a pre-patch sleep; fails when broken.
- Finding (nit): raising a settings pair cap past `MAX_HYPOTHESES` would make the cycle unsavable.
  - Fix: `test_the_pair_caps_fit_the_stored_lists` locks the sum.
- Not changed (nit): `/dreams/cycle/run` (operator-only, no callers) now returns arm and refs per
  pair; the Hub detail route already shows arm per hypothesis, so nothing new is unblinded.
- Checked sound: outcome order unchanged (old no_link = new no_link + echo); `cycle_json` never
  reaches the blinded introspection, Temporal Self (pressure only) or dig.

## Restart required

Merge, then the post-pull rebuild rebuilds orion-dream. Live check after the next sleep:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select started_at, cycle_json->'echo_count', jsonb_pretty(cycle_json->'pair_answers') from dream_cycle order by started_at desc limit 1"
```

## Risks / concerns

- Severity: low. Concern: `no_link_count` (column and JSON) changes meaning at this deploy,
  2026-11-17; a SQL trend will show a step that is not model behavior. Mitigation: field comment;
  echo is counted apart, so `no_link + echo` equals the old number.
- Issues opened: none. Refs #3412 (stays open until a fix from the stored answers).
