# PR: reverie hygiene (calm skip, honest verdicts, train of thought) + kill metacog hop 0

## Summary

- A calm tick (nothing won attention) no longer starts a reverie chain. Before, it wrote an empty "unknown" chain row and an "unknown" cooldown entry about 190 times a day.
- A loop verdict in the reverie prompt now says who recorded it and when. A verdict older than 24 hours is not shown. A verdict from the automatic decay process is never called a human judgement.
- A chain is now one train of thought. Each step sees the previous thoughts and the "next focus" the last step wrote. The chain stops when the model has nothing further or repeats itself.
- The prompt shows the real measured signal (band, 7-day percentile, trend, value, why it was selected) instead of a boilerplate sentence. It stops calling one node a "coalition".
- The old metacog "hop 0" proposal producer is removed with its flag. It produced 36 candidates in its life and none since 2026-10-08.
- `causal_density` is NOT removed: it is not degenerate on current main (see Risks).

## Outcome moved

Baseline from live Postgres, 2026-10-09..16 (7 days). The expected values apply after deploy. They are UNVERIFIED until a post-deploy query is run.

| Measure | Baseline (7 days) | Expected after deploy |
|---|---|---|
| Chains per day | ~324 (2,269 total) | ~143 (only the 1,000 non-"unknown" chains remain) |
| "unknown" chains per day | ~181 (1,266 total; 1,215 ended `no_coalition`) | 0 |
| `unknown` refractory row | 1 row, re-armed every calm tick | not re-armed (the old row expires) |
| Non-calm chains that ran all 4 steps | 92% (921 / 1,000) | lower; chains stop at `settled` |
| Thoughts starting "The coalition" | 94.8% (3,800 / 4,007) | expected near 0 (the prompt forbids it, and the word is gone) |
| Thoughts claiming "unattended N days" | 4.7% (188 / 4,007) | 0 (no verdict older than 24h reaches the prompt) |
| Thoughts whose loop carried a verdict | 753, all `system:implicit_decay`, 17-37 days old (p50 20.7d) | only verdicts under 24h, attributed by actor |
| `next_focus` populated | 0 / 4,007 | populated on non-final chain steps |
| Metacog hop-0 candidates | 0 in 7 days; 36 lifetime; last 2026-10-08 | producer removed |

Queries used (all read-only):

```sql
select date_trunc('day',created_at)::date, count(*), count(*) filter (where theme_key='unknown')
  from substrate_reverie_chain where created_at > now()-interval '7 days' group by 1;
select count(*), count(*) filter (where interpretation ilike 'The coalition%'),
       count(*) filter (where interpretation ~* 'unattended[^.]*[0-9]+ days|[0-9]+ days[^.]*unattended'),
       count(*) filter (where thought_json->>'next_focus' is not null)
  from substrate_reverie_thought where created_at > now()-interval '7 days';
select count(*) filter (where proposal_frame_json::text like '%proposal:cognitive_hop:%') from substrate_proposal_frames;
```

## Current architecture

- `services/orion-thought/app/chain.py::run_reverie_chain` read the broadcast once and called `run_reverie_once` up to 4 times with only `(chain_id, index)`. No prior thought went into the next prompt. A calm broadcast was not checked before the chain started. Step 0 returned None, and the chain was persisted as theme `unknown`, terminal `no_coalition`. A `pressure_reader` parameter existed but no caller passed it.
- `services/orion-thought/app/store.py::load_recent_loop_outcomes` returned the latest verdict per loop with no actor and no age bound. Loop ids hash the loop label (`orion/substrate/attention/scoring.py`), so a verdict from a past episode attached to every later spike of the same signal. The prompt said every verdict "was recorded by a human closing this loop -- it is ground truth".
- `reverie.py::_open_loops_for_prompt` passed `why_it_matters` (the same boilerplate for every substrate loop) and dropped `magnitude` and `provenance.world_first_*`.
- `ORION_REVERIE_DRIFT_TEMP` was declared in settings and never read.
- `services/orion-proposal-runtime/app/worker.py` ran `orion/metacog/proposal.py::trend_result_to_candidate` behind `ORION_METACOG_HOP_PROPOSE_ENABLED=true`.

## Architecture touched

- orion-thought: chain loop, reverie context, loop-verdict reader.
- Prompt `orion/cognition/prompts/reverie_narrate.j2` (cortex-exec renders it, so cortex-exec needs a rebuild).
- Schema `orion/schemas/reverie.py`: `TerminalReason` gains `settled`.
- Analytics dbt: accepted values and the outcome dimension row for `settled`.
- orion-proposal-runtime: hop-0 branch, store reader, setting, env key, compose line removed.

## Files changed

- `services/orion-thought/app/chain.py`: calm skip with counted log; `ChainContinuity` per step; `settled` stop; `is_repeat_thought`; dead `pressure_reader` removed.
- `services/orion-thought/app/reverie.py`: `ChainContinuity`, `describe_loop_verdict`, `describe_verdict_actor`, `_loop_signal`, `attention_steadiness`; `attended_signal` replaces `coalition_projection` in the reverie context; `chain_step` block; parses `next_focus`/`drift`.
- `services/orion-thought/app/store.py`: `LOOP_VERDICT_MAX_AGE_SEC` (24h) with justification; verdicts carry `actor` and `age_sec`; pure `loop_outcomes_from_rows`.
- `orion/cognition/prompts/reverie_narrate.j2`: line-per-loop source block with the real signal; attributed verdict block; TRAIN OF THOUGHT block; `next_focus`/`drift` on non-final steps; "coalition" wording removed from the reverie branch.
- `orion/schemas/reverie.py`: `settled` terminal reason; doc comment on `next_focus`/`drift`.
- `services/orion-analytics/models/{sources.yml,staging/staging.yml,marts/marts.yml,marts/dim_reverie_outcomes.sql}`: `settled`; honest `pressure_discharged` description; missing `resource_deferred` added to the dimension's accepted values.
- `services/orion-thought/{settings.py,.env_example}`: `ORION_REVERIE_DRIFT_TEMP` removed.
- `services/orion-thought/README.md`: new section.
- `services/orion-thought/tests/*`, `services/orion-thought/evals/test_reverie_chain_continuity_eval.py`, `services/orion-thought/evals/fixtures/reverie_chains_2026-10-16.json`: tests and replay eval.
- `orion/metacog/proposal.py`, `orion/metacog/tests/test_proposal.py`: deleted.
- `services/orion-proposal-runtime/app/{worker.py,store.py,settings.py}`, `.env_example`, `docker-compose.yml`: hop 0 removed. `orion/proposals/builder.py`: comment.
- `services/orion-proposal-runtime/tests/test_metacog_hop0_removed.py`: guard against partial resurrection.

## Schema / bus / API changes

- Added: `TerminalReason` value `settled` (`ReverieChainV1.terminal_reason`).
- Removed: none on the bus. `SpontaneousThoughtV1.next_focus`/`drift` already existed; they now have a producer (the prompt) and a consumer (the next step and the stop rule).
- Behavior changed: the reverie prompt context key `coalition_projection` is now `attended_signal` (reverie branch only; the only consumer is `reverie_narrate.j2`). Open-loop entries carry `signal` or `why_it_matters`, and `outcome` is `{line, verdict, by_human}`.
- Compatibility: all `terminal_reason` readers (Hub reverie routes, orion_day, temporal_self, durable-runs) treat it as a free string. Analytics accepted-values updated in the same patch.

## Env/config changes

- Removed keys: `ORION_REVERIE_DRIFT_TEMP` (orion-thought), `ORION_METACOG_HOP_PROPOSE_ENABLED` (orion-proposal-runtime).
- `.env_example` updated: yes, both.
- Local `.env` synced: `python scripts/sync_local_env_from_example.py` only adds keys, so it reported no change. The two removed keys were deleted by hand from `services/orion-thought/.env` and `services/orion-proposal-runtime/.env` in the primary checkout (backups in the session scratchpad).
- Skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-thought && PYTHONPATH=<worktree> pytest tests evals -q
  690 passed, 32 skipped, 1 failed (test_settings_mind_enrichment::test_mind_enrichment_defaults_off -- also fails on main; reads the host .env)
cd services/orion-proposal-runtime && pytest tests -q            13 passed
PYTHONPATH=. pytest orion/metacog orion/reverie orion/schemas orion/proposals orion/orion_day orion/temporal_self \
  tests/test_dim_reverie_outcomes_covers_terminal_reasons.py tests/test_attention_schema_surface.py tests/test_attention_loop_decay_digest.py -q
  727 passed, 9 failed (orion/reverie/tests/test_proposal.py x8 and orion/schemas/tests/test_context_provenance.py x1 -- identical 9 failures on main)
python scripts/check_env_template_parity.py                       PASS
pytest tests/test_metric_prompt_semantics.py tests/test_dim_reverie_outcomes_covers_terminal_reasons.py   43 passed
python scripts/check_definition_drift.py                          0 changed
```

New regression tests:
- calm tick starts no chain, no LLM step, no refractory entry, counted skip (`test_calm_tick_starts_no_chain_and_no_unknown_refractory`)
- system verdict never labelled human; stale verdict dropped in store and in render; undated verdict dropped
- each step sees the prior thought and hint; `settled` on null `next_focus`; `settled` on a restated thought; final step does not ask for `next_focus`; standalone tick has no chain block
- continuity text cannot ground a thought (hollow guard holds)
- world-first signal replaces boilerplate; reverie prompt does not call one node a coalition
- hop-0 module, flag, and reader gone

## Evals run

```text
cd services/orion-thought && PYTHONPATH=<worktree> pytest evals/test_reverie_chain_continuity_eval.py -q   5 passed
```

Replays 3 real recorded 4-step chains and 1 real chain whose step 1 restated step 0 (Jaccard 0.689) through the real chain loop and the real template. Asserts step k's prompt carries step k-1's thought, no two consecutive prompts are identical, the restating chain now stops at step 2 (`settled`), and the real world-first numbers reach the prompt.

## Docker/build/smoke checks

```text
Not run. No deploy was requested; the live path after deploy is UNVERIFIED.
```

Post-deploy check (read-only):

```sql
select count(*), count(*) filter (where theme_key='unknown'), count(*) filter (where terminal_reason='settled')
  from substrate_reverie_chain where created_at > now()-interval '1 day';
select count(*) filter (where interpretation ilike 'The coalition%'), count(*) filter (where thought_json->>'next_focus' is not null), count(*)
  from substrate_reverie_thought where created_at > now()-interval '1 day';
```

and `docker logs orion-thought 2>&1 | grep reverie_chain_skipped | tail`.

## Review findings fixed

- Finding: lift mode (concern-card prompt, which never asks for `next_focus`) would have cut every chain to one step.
  - Fix: the no-`next_focus` stop applies only when the plain reverie prompt asked for it (`llm_profile != "metacog"`).
  - Evidence: `test_lift_mode_chain_is_not_cut_by_missing_next_focus`.
- Finding: dropping `coalition_stability_score` from the reverie context left a registered prompt site (`inner_state_registry.py`, pinned in `PROMPT_INVENTORY_URNS`) pointing at nothing.
  - Fix: kept it, rendered as words ("how steady your attention is on it: held (0.9)") in `attended_signal.steadiness`. No metric-definition change.
  - Evidence: `test_steadiness_tier_reaches_prompt_as_words`; `check_definition_drift.py`: 0 changed; `tests/test_metric_prompt_semantics.py` 43 passed.
- Finding: the raw world-first `value` reached the prompt with no unit.
  - Fix: removed; only band, 7-day percentile, trend and `world_first_reason` are rendered. Gate record below.
  - Evidence: test asserts `0.0621` is absent.
- Finding: the 24h verdict-bound comment over-claimed; human verdicts already keep a loop out of the coalition for 48h.
  - Fix: comment now says the bound acts in practice on `system:implicit_decay` verdicts, and a human verdict under 24h can only appear on a non-selected loop.
  - Evidence: `store.py` comment.
- Finding: repeat check compared only against the previous step (A-B-A not caught).
  - Fix: compares against every earlier thought in the chain. The restated thought is still published (documented).
  - Evidence: `test_chain_settles_on_a_b_a_restatement`.
- Finding: an undated-verdict test could pass because the DB path raised.
  - Fix: replaced with a pure `loop_outcomes_from_rows` test.
- Finding: shared analytics description was text-chain-specific; blank line between open-loop lines in the template.
  - Fix: neutral wording; blank line removed.
- Finding: `ORION_REVERIE_DRIFT_TEMP` was back in the primary-checkout `services/orion-thought/.env` (main's `.env_example` still carries it, so a sync re-adds it).
  - Fix: removed again by hand. It may come back until this PR is merged; it is harmless (unread).

## Metric quality gate (world-first band / percentile / reason in the reverie prompt)

1. Provenance: `OpenLoopV1.magnitude` is `PredictionErrorMagnitudeV1` from `orion/substrate/prediction_error_magnitude.py` over `substrate_node_prediction_error_history`; `provenance.world_first_reason`/`band` from `orion/attention/world_first.py::judge_candidate`, which selected the loop.
2. Independence: not new signal. It is the reason the attended loop won (the same percentile world-first ranks on). The self-readings block scores the same history; both describe one fact, so it is shown as context, never as a second vote.
3. Theory anchor: the percentile against the signal's own 7 days is the world-first selection rule itself (spec 2026-10-02 reverie prediction-error magnitude proposal); showing it tells the model why this loop was attended instead of boilerplate.
4. Live data: a live broadcast at 2026-10-16T02:27Z carried `band=high, percentile_now=0.900, trend=flat, reason="body unusual for itself (high)"`, and a second loop `trend=rising, percentile_now=0.914`. Rest state exists: `band` has `quiet`/`usual`, and a calm tick has no winner at all (and now no chain).
5. Existing mechanism: the fields already existed on the loop; reverie dropped them. Nothing new was built.
6. Reversibility: prompt text only; no schema, table, or default changes. Delete `_loop_signal` to revert.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-thought orion-cortex-exec orion-proposal-runtime; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

## Risks / concerns

- Severity: low. Concern: `causal_density` was not removed. The 10-14 audit said it is only ever 0 or 0.25. That is no longer true on main: since the 2026-09-24 evidence-mapping change it takes 55-390 distinct values per day (last 14 days), per trigger kind 3-465 distinct values in 7 days, range 0-1, and `is_causally_dense` is true on 15-79 rows per day. It has a live producer and readers. Removing a working metric is out of scope for a hygiene patch. Mitigation: none needed; the "degenerate" note is stale.
- Severity: medium. Concern: the 24h verdict bound means almost no verdict reaches reverie today (over 60 days no human verdict was narrated within 24h of being recorded). That is the honest result, not a loss: every verdict shown before was about an earlier episode. Mitigation: the bound is one constant (`LOOP_VERDICT_MAX_AGE_SEC`) with its justification next to it.
- Severity: low. Concern: the `settled` stop depends on the model returning `next_focus: null`. A model that always writes a next focus still runs to `max_steps`, but each step now sees the prior thought. Mitigation: the deterministic repeat check (Jaccard >= 0.6) is a backstop.
- Severity: low. Concern: `pressure_discharged` keeps its historical key for "a later step yielded nothing". Mitigation: the analytics dimension description now says so.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2707

🤖 Generated with [Claude Code](https://claude.com/claude-code)
