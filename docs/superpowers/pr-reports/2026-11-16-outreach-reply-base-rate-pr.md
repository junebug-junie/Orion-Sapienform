## Summary

- Outreach acts still close on `juniper_replied` (0 or 1). Juniper's vote in `orion/physiology/weights.py` is unchanged.
- At open, the sql-writer now sets the act's expectation (`predicted`) to Juniper's own reply rate for that time of day: on how many of the last 28 days she started a chat turn in the same 10 h window of local clock time. The rate is smoothed (Jeffreys) and uses only days before the act.
- So `error = outcome - predicted` now reads "more or less than usual for this hour". 0 means as usual.
- The outcome row (`substrate_action_outcomes`) carries the rate as `baseline`, its decile as `baseline_bin`, and `observed_delta = outcome - baseline`. The resolver note adds "usual for this time of day 0.62 (17 of 28 days)".
- With under 7 days of history there is no rate, and the act keeps the pooled posterior stamp, as before. Kill switch `SQL_WRITER_ACT_REPLY_BASE_RATE_ENABLED` (ships true).

## Outcome moved

The outreach grade no longer follows Juniper's sleep. Before: the raw grade was 0.30 for acts sent 00:00-06:00 Denver and 0.68 for 12:00-18:00. After, against her base rate: -0.05 to +0.06 across the four 6 h blocks (noise for n=44..139).

## Metric quality gate (live data, read-only, 2026-11-16)

1. **Provenance.** `orion/grammar/reply_base_rate.py:juniper_reply_base_rate` reads `chat_history_log` rows that pass `JUNIPER_TURN_PREDICATE` (her real turns, not outreach rows). The same rows feed the resolver's arm 1.
2. **Independence.** The rate is computed from days before the act, so the act's own outcome is never in it. It is the control for the existing 0/1 grade, not a new signal: it explains the hour, correlation with the outcome 0.31. It does not touch the cost or reliability sources in the weights.
3. **Theory anchor.** Treated vs control at the same baseline: the same contrast `orion/autonomy/contrast.py` uses for world settlement. Did the message move her more than her own rhythm at that hour?
4. **Live sanity.** 303 resolved outreach acts over 30 days, replayed through the shipped module: mean outcome 0.531, mean base rate 0.530, mean residual +0.001, residual sd 0.475. Base rate range 0.09..0.82, so it is not flat and not saturated. Rest state: the residual sits at 0 when outreach does no better than her rhythm, and it reads 0 on average live. Each hour bin has about 60 control days, against 3 to 61 outreach acts per hour. For that reason the rate comes from all days, not from past outreach acts: an act-only rate would have hour 23 empty and five hours with n<=5. Cost: 5 ms per act.
5. **Existing mechanism.** `substrate_action_outcomes.baseline`/`baseline_bin` already existed (world settlement uses them). For acts they were hard-coded to 0, so this patch fills them in. **`substrate_action_effect_posterior.baseline_bin` was NOT used.** For acts, bin 0 is already the pooled row: open stamps it and the self-monitor reads it. A base rate under 0.1 would land in bin 0 and collide with it. Splitting 300 acts into 10 cells would also start every cell cold, and nothing reads per-bin act posteriors.
6. **Reversibility.** Cheap. Nothing in a schema changed. Turning the switch off brings back the pooled stamp. `outcome_value` and the posterior are untouched.

## Current architecture

Hub opens `act.hub.outreach.say|ask` with `juniper_replied` @ 36000 s (`services/orion-hub/scripts/act_emit.py`). The sql-writer act ledger (`orion/grammar/act_ledger.py`) stamped `predicted` from the pooled posterior. The resolver closed the act 0/1 from her next turn start.

## Architecture touched

- `orion/grammar/reply_base_rate.py` (new): a pure base-rate function plus one read-only query.
- `orion/grammar/act_ledger.py`: `open_act` stamps the base rate for outreach acts. `close_act` writes baseline, bin and delta, and adds the note line, only when the open stamped that same rate.
- sql-writer settings, `.env_example`, compose, `main.py`: the kill switch. README section. Doc comments in `act_emit.py` and `self_monitor/adapters.py`.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: for `hub.outreach` acts only, `grammar_acts.predicted` = hourly base rate, `predicted_n` = history days, `cold_start` = false. `error` is hour-relative. Outcome row `baseline`/`baseline_bin`/`observed_delta` are filled in.
- Compatibility: `outcome_value` is unchanged (weights vote). `surprise_nats` is still scored against the pooled posterior. Acts opened before the deploy close as before.

## Env/config changes

- Added keys: `SQL_WRITER_ACT_REPLY_BASE_RATE_ENABLED=true` (orion-sql-writer)
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Skipped keys: none for this service.

## Tests run

```text
pytest services/orion-sql-writer/tests orion/grammar/tests orion/self_monitor -q   -> 1512 passed, 232 skipped
pytest -m integration test_act_ledger_integration.py (+reading cost, workspace binding, camera ask) on a throwaway postgres:16 -> all passed (28 in act ledger file incl. 6 new)
New unit tests: test_reply_base_rate.py (9): hour with no history -> None; hour she always replies -> 0.98, reply error ~0;
  hour she never replies -> 0.017; no leakage after open; window edges; DST; naive inputs; switch; flag ships on.
check_env_template_parity: PASS; check_integration_seam: OK (not a new producer of experience: no new family/workflow/channel)
```

## Evals run

```text
Live replay of the shipped module over 303 outreach acts (read-only): mean residual +0.001, base 0.09..0.82, corr(outcome, base) 0.31.
No standing eval harness for act grading; the live check is #3550 (verify-live).
```

## Docker/build/smoke checks

```text
Not run: no dependency or image change (pure Python in orion/ and sql-writer settings).
```

## Review findings fixed

- Finding: the close recomputed the base rate even when the open had stamped the pooled mean (acts in flight across the deploy, a failed open read, a switch flip). That would store a baseline and an error built on two different expectations.
  - Fix: the close uses the base rate only when it equals the open's `predicted`.
  - Evidence: `test_outreach_opened_without_a_base_rate_keeps_one_expectation_at_close`.
- Finding: the test gaps were a non-UTC server clock and a base-rate read that raises.
  - Fix: `test_base_rate_reads_naive_created_at_on_a_non_utc_server_clock` and `test_a_failing_base_rate_read_does_not_block_the_open`.
- Finding (documented, not changed): the base rate uses `created_at`, which is 1 to 41 min after a turn starts, and it skips errored turns. Surprise is still scored against the pooled posterior. `predicted_n` changes meaning. Self-monitor stated levels now vary by hour. All four are written in the `reply_base_rate.py` and `adapters.py` docstrings. The live replay shows the bias is small (0.530 vs 0.531).

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```
(from the clean primary main checkout after merge)

## Risks / concerns

- Severity: low. Concern: an outreach send with a base rate near 1 that gets no reply now shows a large negative error. Mitigation: that is the intended reading. `outcome_value` and the weights are unchanged.
- Issues opened: #3550 (verify-live). Closed: none. #3392 stays open until #3550's live check passes.

## PR link

(see PR)
