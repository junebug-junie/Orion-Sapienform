## Summary

- Every Zelda goal Orion plans is now a prediction in **orion-prediction**, called an action bet: "this action, in this situation, at this place, works with probability p".
  - The service computes p when the bet is placed.
  - The Hub settles the bet from Retro's own result.
  - The service grades settled bets with its existing `grading.py`.
- **New model `action_beta_backoff`.** A Beta posterior at three levels (skill → situation → place, empirical-Bayes strength 4) and the expected information gain (EIG) of one more try.
  - Shadow baseline `action_skill_rate` (the skill's flat rate) is stored on every bet. A session-paired live verdict serves the baseline when the model is `worse`.
  - Above the place level each place counts as one unit.
- **Game-free action vocabulary** in `orion/embodiment/skills.py`:
  - `traverse`, `engage`, `withdraw`, `acquire`, `enter`, `hold`;
  - situation bands for side, threat and health.
  - Bets pool at the skill level across bodies. That pooling is what carries learning to future games and to a real motor.
- **The model chooses from a scored list.** The Hub quotes every possible move, scores it `value × (p + 2 × EIG)`, and gives the model the top 5. Bets are placed only after Retro accepts the goal.
- **Memory card.** At session end the card gets up to 3 plain fact lines read back from the service, so Orion can talk about what it learned.
- **Retro wall healing.** A learned-blocked position is removed once Link stands on it, so one false bump no longer closes a path forever.

## Outcome moved

Live, 2026-10-23: Orion chose "explore S" 8 times against a solid wall. Every prediction Retro graded was thrown away.

Now, replaying that session's real evidence (`evals/zelda_experience_replay.py`, offline, real model code):

- room 119's south side reads p = 0.145 (place level, 0 of 7);
- untried exits west and east read 0.60;
- the menu never ranks "explore S" first;
- all 8 goals settle.

**Live model eval** (`zelda_planner_replay.py --with-menu`, local GPU, 5 runs): 5/5 chose the top option, exit E toward a room Orion has never visited. 0 dead moves.

The same eval found a bug before the fix. Under raw pooled counts the 7 wall failures made every exit read 0.20, and the model chose `wait` 3 times in 5. The fix is the one-unit-per-place rule.

## Current architecture

- **Retro:** the motor grades every goal (met or refuted); the atlas is a per-save JSON map whose learned blocks never expire.
- **Hub planner:** shows the model a text map, open sides and its last 5 results, with no probabilities.
- **orion-prediction:** forecasts numeric series and grades threshold probabilities. It had no action-conditional predictions.
- **`orion/embodiment`:** the AI Town body only.

## Architecture touched

- **orion-prediction:** a new table, model, endpoints and `/health` field.
- **Contracts** in `orion/schemas/prediction.py`: HTTP only, no bus channel.
- **Hub planner path:** co-play `plan_v2`, leisure, session end, and the memory card.
- **Retro:** the atlas's per-frame observe step.

## Files changed

- `orion/schemas/prediction.py`: the `ActionBet*V1` contracts.
- `orion/embodiment/skills.py` and `tests/test_skills.py`: the vocabulary and the band helpers.
- `services/orion-sql-db/manual_migration_prediction_action_bet_v1.sql`: the new table and 3 indexes (additive; `REQUIRED-BY: orion-prediction`).
- `services/orion-prediction/app/action_bets.py`: model, baseline, EIG, live verdict, expiry, quote/place/settle/live.
- `services/orion-prediction/app/store.py` and `app/main.py`: the store and the endpoints.
- `services/orion-prediction/tests/test_action_bets.py` and `README.md`.
- `services/orion-hub/scripts/retro_experience.py`: the Zelda adapter (options, settle table, menu, client).
- `services/orion-hub/scripts/retro_planner.py`, `retro_routes.py`, `retro_leisure.py`, `retro_memory_export.py`, `retro_session_end.py`: the wiring.
- `services/orion-hub/app/settings.py`, `.env_example`, `README.md`: the flag and the URL.
- `services/orion-hub/tests/retro/test_experience.py`: 71 tests.
- Evals: `services/orion-hub/evals/zelda_experience_replay.py` (new) and `zelda_planner_replay.py` (`--with-menu`). Fixture: `evals/fixtures/zelda_session_c6041e2d_goals.json`, RAM-derived rows only, no ROM bytes.
- `services/orion-retro/app/zelda/atlas.py`, `tests/test_zelda.py`, `README.md`: the healing.
- `docs/superpowers/specs/2026-10-23-embodied-experience-learning-design.md`: the spec (revision 2, plus corrections found in the build).

## Schema / bus / API changes

- **Added** orion-prediction endpoints:
  - `POST /action-bets/quote`
  - `POST /action-bets`
  - `POST /action-bets/{id}/settle`
  - `GET /action-bets/live`
  - `GET /action-bets?session_id=`
- **Added** `/health.action_bets`.
- **Added** table `prediction_action_bet`.
- **Added** the HTTP contracts `ActionBet{Option,QuoteRequest,Quote,Counts,Place,Settle}V1` and `ActionBetV1`.
- **No new bus channel and no schema-registry entry**, because nothing is published.
- **Behavior changed:**
  - the Zelda prompt can carry a scored list;
  - Retro's offers are unchanged;
  - Retro's atlas heals learned blocks.
- **Compatibility:**
  - With `HUB_ACTION_BETS_ENABLED=false`, or the service unreachable, the planner behaves exactly as #2870.
  - orion-prediction refuses to boot without the table. The deploy gate and the post-pull auto-apply handle that: the migration passes `auto_apply_refusals` against the live catalog.

## Env/config changes

- Added keys: `HUB_ACTION_BETS_ENABLED=true`, `HUB_PREDICTION_URL=http://127.0.0.1:8132` (orion-hub).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-hub`: yes. Both keys are in `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest services/orion-hub/tests/retro -q                          171 passed
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_retro_memory_export.py services/orion-hub/tests/test_retro_leisure_listener.py services/orion-hub/tests/test_retro_memory_export_pg.py -q   45 passed, 1 skipped
cd services/orion-prediction && PYTHONPATH=../..:. pytest tests/test_action_bets.py tests/test_grading.py tests/test_live.py -q   (action bets 20 passed)
PYTHONPATH=. pytest services/orion-retro/tests -q                              129 passed
PYTHONPATH=. pytest orion/embodiment/tests -q                                  100 passed
PYTHONPATH=. pytest tests/retro -q                                              27 passed
python3 scripts/check_env_template_parity.py                                    PASS
```

The full orion-prediction suite has 8 failures that also fail on clean `main`. This venv has no `statsmodels`/`prophet` (`test_causal`, `test_events`, `test_backtest`). CI runs with the real requirements.

## Evals run

```text
evals/zelda_experience_replay.py              PASS 5/5 (S place p=0.145; W/E 0.60; menu top exit E; 8/8 settled)
evals/zelda_planner_replay.py --with-menu     PASS 5/5 goals, 0 dead moves (exit E, rank 1, every run)
evals/zelda_planner_replay.py                 PASS 3/3 goals, 0 dead moves (no menu: unchanged)
```

Real SQL on a throwaway Postgres 16 (scratchpad, port 55433, never production):
- migration applied twice;
- place/settle/quote/live run through `PostgresStore`;
- the weighted count query returned east 0.50 (situation level) and south 0.182 (place level, 0 of 7).

## Docker/build/smoke checks

Not run in this branch. The deploy is the post-pull rebuild (below).

## Review findings fixed

The code review subagent ran on the whole branch, including the counting fix.

- **HIGH: atlas healing erased every east-side bump on the next frame.** It healed (c+1, r), but learned blocks are Link positions.
  - Fix: heal only (c, r).
  - Evidence: updated `test_learned_blocked_tile_heals_once_link_stands_on_it` keeps (16, 11).
- **HIGH/MEDIUM: card line 2 was never written with float counts.** It also counted places, not tries.
  - Fix: build it from this session's raw bets.
  - Evidence: `test_learned_lines_are_facts_from_the_bets` uses float `LIVE`.
- **MEDIUM: moves inside a cave were scored as the outside room's sides.**
  - Fix: no place and no adjacency unless `mode == 'play'`.
  - Evidence: `test_a_cave_never_bets_on_the_room_above_and_hearts_count_as_a_pickup`.
- **MEDIUM: heart pickups always settled as a failure.**
  - Fix: `hearts.current` going up counts as gained.
  - Evidence: the same test.
- **MEDIUM/LOW: the verdict and `/live` cost grew with history** (0.82 s per 5,000 bets for the verdict alone).
  - Fix: per-bet p only (`backoff_p`), and a 10 s timeout for the card fetch.
  - Evidence: verdict plus grading on 5,000 bets now takes 0.056 s; `test_backoff_p_matches_backoff_on_many_count_shapes`.
- **LOW: card wording claimed too much.**
  - Fix: "has not led anywhere" instead of "is not a way out", and the average is now `grade()`'s base-rate Brier.
- **LOW: off-menu mislabelling for exit/explore toward one side.**
  - Fix: they share a rank.
  - Evidence: `test_explore_and_exit_toward_one_side_share_a_menu_rank`.
- **LOW: in-process open bets could leak.**
  - Fix: capped at 200; the service expires anything dropped.
- **LOW: a wrong comment** about a missing table in `main.py`. Fixed.
- **Earlier, in the build:**
  - Untried `wait` outranked failing exits on curiosity alone. Fix: value-weighted EIG.
  - Exit and explore duplicates took 2 of 5 menu slots. Fix: one option per side.
  - The spec's rest-state bound was wrong (0.44 × the situation mean). Fix: corrected in the spec.

## CI findings fixed

- `Static repo gates`: `test_schema_skew_discovery` required a declared writer for `ActionBetCountsV1`, `ActionBetQuoteV1` and `ActionBetV1`. Each is now declared as `orion-prediction` in `orion/schema_skew_discovery.py`.
- `prediction`: the job had no `httpx` for FastAPI's `TestClient`. Added to the workflow's install line.
- `native-boundary`: that job has no numpy/scipy, so the offline-replay pytest wrapper skips there (`importorskip('scipy')`). The eval still runs standalone, and in any venv with scipy.
- All 18 checks pass. The prediction job runs the full suite with real `statsmodels`/`prophet`, which confirms the 8 local failures are environment-only.

## Restart required

```bash
git pull   # post-pull rebuild: applies the additive migration, then rebuilds orion-prediction, orion-hub and orion-retro (a running game ends)
```

## Risks / concerns

- **Severity Medium:** the value table is hand-set (new room 1.0, visited 0.3, …).
  - Concern: it decides what Orion prefers.
  - Mitigation: it is labelled in code and in the spec. Learning value is the named next step.
- **Severity Low:** a Hub restart mid-session drops that session's open bets.
  - Concern: they expire `never_settled` after 1 h, not graded. Acceptance check 6's "nothing open 60 s after end" fails after a restart.
- **Severity Low:** a future body without place keys outweighs Zelda's traverse at the skill level (one unit per bet against one per place).
  - Mitigation: noted in the spec; revisit when the second body exists.
- **UNVERIFIED:** live session acceptance (check 6) and cross-body transfer (check 7). They need a played session and a second body.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2878

🤖 Generated with [Claude Code](https://claude.com/claude-code)
