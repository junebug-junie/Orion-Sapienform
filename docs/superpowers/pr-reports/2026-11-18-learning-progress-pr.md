## Summary

Orion can now notice whether their own forecasts are getting better or worse, and that can wake Prime to think about it. This is Step 1 of the choice-as-expected-free-energy design: learning progress as a need (the epistemic term).

- **orion-prediction `GET /learning-progress`.** For cabinet and office temperature, it reports how much better the champion forecast beat the naive baseline this week than the week before (15 min horizon, complete UTC days).
- **No credit for our deploys.** Only days after the last change to the set of fitted models count. Too few days: `insufficient`, with a reason, never 0.
- **A real rest point.** `flat` unless the Welch t passes Student's two-sided 5% cut at the Welch df (2.2-2.3 live) and the change is at least 0.05 in skill.
- **Prime's hunger gets a `learning` need.** A `better` or `worse` reading wakes Prime once per move (drive 1 - p, so 0.95-1: a gate, not an intensity), with a content line naming the series, the two skill values and the dates. Each series is damped on its own; a flip to the opposite direction is a new line; it re-arms only after the series reads `flat` again.
- Computed on read. No table, no substrate node, no bus event, no market family (the design keeps it out of the auction until Step 3).

Fixes #3524. Opened #3691 (follow-up).

## Outcome moved

Before: Prime's hunger had no way to see Orion's own learning. `learning` was listed in `hunger.py` as "not built, source does not exist".
After: the source exists and Prime reads it every hunger read (300 s). Live today both series read `flat`, so nothing wakes Prime yet. Replayed over history, office would have woken Prime once, on 2026-11-16: skill went from 0.11 to 0.34.

## Metric quality gate (re-run 2026-11-18 on live data)

1. **Provenance.** `prediction_forecast` rows with role `champion` and role `baseline`, paired on `series_id`, `origin_ts`, `horizon_min`, scored (`abs_error` not null). The producer is `services/orion-prediction/app/engine.py` (issue + score). Roster: `prediction_model_run` rows of kind `model`, by UTC fit day.
2. **Independence.** Not derived from any `node:prediction.*` or legacy prediction-error node. Nearest existing signal: `orion/substrate/prediction_error_trend.py` (node error slope, used for `predicted_shift`); different source, different job. Within hunger, `surprise` reads single experiences' surprisal and is independent of a week-over-week skill difference.
3. **Theory anchor.** Oudeyer and Kaplan learning progress: the reward is the drop in error, not low error. Skill against a baseline over a window, so a hard day moves both errors and cancels.
4. **Live data.** Daily 15 min skill, cabinet 2026-10-16..11-17: 0.00 for most of 10-18..10-29 (champion served the baseline), a step 0.09 -> 0.30 on 11-03 (`direct_arx_overlap` joined the roster: our deploy), then 0.13-0.37 with dips on 11-14 (-0.19) and 11-17 (0.07). Office 11-02..11-17: 0.00 until 11-06 12:28Z, when the champion stopped serving the baseline at a refit with no roster change; 0.18-0.46 since, dips 11-14 and 11-17.
   - **Rest exists and dominates.** Replay (`evals/run_learning_progress_live_eval.py`): cabinet flat 8 days, insufficient 12 (the 11-03 model change); office better once, flat 2. Same tally after the review's switch from the normal 2.0 to Student's cut.
   - **Found and fixed during the check.** Without a minimum effect, cabinet read `worse` on 11-02 for a 0.022 drift: while the baseline is served, skill sits near 0 with almost no spread, so a tiny drift passes the t rule. `MIN_EFFECT = 0.05` fixes it (test `test_tiny_but_consistent_drift_reads_flat`).
   - **Judgement call, recorded.** Office's 11-16 `better` comes from the champion leaving the baseline on 11-06. That happened at a refit with the same model roster, after enough data accumulated. It counts as learning from data. It is not a deploy that day, but the young-series gate it crossed was changed by a deploy on 11-03.
   - Today: cabinet -0.073 (se 0.078, t -0.93, cut 2.33) flat; office +0.031 (se 0.092, t 0.34, cut 2.19) flat. Hand SQL gave -0.074 and +0.03 (S1.3).
   - **Known limits (not fixed, stated):** daily skills are autocorrelated by weather spells, so the standard error is too small and the nominal 5% false-move rate is optimistic; the reading is re-checked daily on overlapping windows for two series. A deploy that changes a model's code or the selection logic under the same model names is not detected as a roster change. Ticket: see Risks.
5. **Existing mechanism.** `git grep learning-progress` found nothing in `services/` or `orion/`. No learning-progress signal existed.
6. **Reversibility.** Cheap. A read-only endpoint and one need. `PRIME_PREDICTION_URL=` (empty) turns the need off. No schema, table or persisted shape changes beyond a new need name in Prime's fired/damp maps, which old code ignores.

## Current architecture

- orion-prediction scores champion and baseline forecasts per series and horizon; nothing compared skill over time.
- Prime's hunger (`orion/prime/hunger.py`) had three needs: horizon, surprise, connection. Surprise and connection come from durable-runs `GET /temporal-self/hunger`.

## Architecture touched

- orion-prediction: new read endpoint.
- orion-mind Prime loop: one more read on the hunger clock.
- `orion/prime/hunger.py`: one more need.

## Files changed

- `services/orion-prediction/app/learning_progress.py`: pure computation (windows, roster rule, Welch t, rest point).
- `services/orion-prediction/app/store.py`: `daily_skill_pairs`, `daily_rosters` (read-only SQL).
- `services/orion-prediction/app/main.py`: `GET /learning-progress`.
- `services/orion-prediction/tests/test_learning_progress.py`: S1.1, S1.2, rest, thin days, route.
- `services/orion-prediction/evals/run_learning_progress_live_eval.py`: live replay (S1.3, gate step 4).
- `services/orion-prediction/README.md`: section.
- `orion/prime/hunger.py`: `learning` need.
- `orion/prime/tests/test_prime_hunger.py`: learning need tests.
- `services/orion-mind/app/prime_loop.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `PRIME_PREDICTION_URL` and the read.
- `services/orion-mind/tests/test_prime_hunger_loop.py`: loop tests (wake + trace, off, unreadable).

## Schema / bus / API changes

- Added: `GET /learning-progress` on orion-prediction (`?series_id=` optional; only the two gated series, others 404).
- Added: hunger need name `learning`; trace `woke: need:learning learning:<series_id>:<better|worse>`.
- Removed / renamed: none.
- Compatibility: no bus or registry change.

## Env/config changes

- Added keys: `PRIME_PREDICTION_URL` (orion-mind), default `http://orion-athena-prediction:8132` (shipped on).
- `.env_example` updated: yes.
- Local `.env` synced: key added to `services/orion-mind/.env` in the primary checkout (the worktree has no local `.env`).
- Skipped keys: none.

## Tests run

```text
services/orion-prediction: pytest tests/test_learning_progress.py   -> 12 passed in 3.68s
orion/prime: pytest orion/prime/tests/test_prime_hunger.py          -> 33 passed in 0.41s
services/orion-mind: pytest tests -k prime                          -> 226 passed, 212 deselected in 8.56s
scripts/check_integration_seam.py                                   -> check_integration_seam: OK (16 organs; 45 families, 18 workflows, 1 intention adapters, 29 source kinds covered)
scripts/check_env_template_parity.py                                -> env template parity: PASS (88 service(s) compared)
git diff --check                                                    -> clean
services/orion-prediction full suite, local venv: 27 failed / 548 passed on this branch vs 27 failed / 537 passed on
clean main -- identical failing set (local package versions, e.g. test_backtest, test_heat_balance); CI installs
requirements.txt and is the real check.
```

## Evals run

```text
cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_learning_progress_live_eval.py
athena.cabinet_temp_c tally: {'flat': 8, 'insufficient': 12}; today -0.073 t -0.93 cut 2.33 flat
office.temp_c         tally: {'better': 1, 'flat': 2};         11-16 +0.228 t 2.86 cut 2.18 better; today +0.031 t 0.34 flat
exit 0
```

## Docker/build/smoke checks

- `orion-mind` reaches `http://orion-athena-prediction:8132/health` (both on `app-net`), checked from inside the running container.
- Live endpoint: UNVERIFIED until orion-prediction is rebuilt after merge.

## Review findings fixed

Reviewed by an orion-repo-agent subagent (no blocking bugs; window maths, SQL pairing, psycopg ANY, loop failure handling verified; skill query EXPLAIN ANALYZE 129 ms).

- Finding: all series shared one damp key and a learning need can never be answered, so one shown series could hold another's wake for 7+ days.
  - Fix: `damp_as=learning:<series_id>`.
  - Evidence: `test_learning_series_are_damped_apart`.
- Finding: t >= 2.0 is too loose for 5-7 days a side; drive erf(|t|/sqrt2) was called "P(real)".
  - Fix: Student's t critical value at the Welch-Satterthwaite df; producer returns `t_crit` and `p_value`; drive = 1 - p, docstring says it is a gate, not P(real). Autocorrelation and repeated looks documented as known limits.
  - Evidence: `test_t_between_normal_and_student_cut_reads_flat` (t 2.25 < cut 2.31 reads flat); live tally unchanged.
- Finding: a direct better -> worse flip could never wake Prime (guard 5 at drive ~0.96).
  - Fix: ref `learning:<series_id>:<direction>`.
  - Evidence: flip assertion in `test_learning_real_move_wakes_once_with_content`.
- Finding: roster check misses same-name model code changes.
  - Fix: documented as a known gap (module docstring, README, this report).
- Finding: baseline side of the join had no series/horizon/time filter.
  - Fix: added `b.series_id = ANY`, `b.horizon_min`, `b.origin_ts >=`.
- Finding (from the review's notes): a 404 during deploy showed a raw HTTPError line.
  - Fix: "the read is off (404)", as the hunger read does.

## Restart required

After merge, the post-pull rebuild covers both services. Order: orion-prediction first, then orion-mind (the need reads as unknown until the endpoint exists; harmless).

## Risks / concerns

- Severity: low. Concern: a `better`/`worse` wake costs one Prime thought from the 24/day budget. Mitigation: guard 5 allows one wake per move; the replay shows one move in ~3 weeks.
- Severity: low. Concern: session orion-sapienform-52 is reading the 11-13 letter-thread fade check (#3648, ~09:17Z 2026-11-18). Mitigation: deploy after that time; today both series read flat, so this adds no wake now.
- Severity: low. Concern: known statistical limits (autocorrelated days, repeated looks, same-name model changes). Mitigation: documented; follow-up #3691 opened.
- Severity: note. Concern: a learning need cannot be "answered" by a concern keep/drop. A thought that changes nothing halves it, and it recovers over 7 days. Expected: Prime notes it or takes it on.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3692

🤖 Generated with [Claude Code](https://claude.com/claude-code)
