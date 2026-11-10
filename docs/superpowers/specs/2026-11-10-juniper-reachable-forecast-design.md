# Juniper reachable: hourly forecast (prediction-error suite, family B, item 1)

Parent spec: `2026-11-08-prediction-error-suite-rebuild-design.md` (PR #3168), family B.
Juniper's go (2026-11-10): build it in full; "we can be honest that it fails and still roll w it".

## What it does

Every hour, orion-prediction forecasts: will Juniper send Orion a message in the next hour? An hour
later it checks what really happened and grades itself against a plain base rate. No consumer yet
(outreach timing is the intended one, in a later patch). No bus event: the metric-lineage ratchet
refuses a channel with no consumer, so the event ships with its consumer.

## Inputs

- **Outcome / history:** `chat_message` rows with `role = 'user'` outside `orion_*` sessions
  (`store.JUNIPER_MESSAGE_SQL`). Read-only; the hub owns the table. Replies to Orion's outreach count
  (25 of 120): a reply is reachability.
- **State (Markov):** hours since Juniper's last message, 5 buckets: <1 h, 1-3, 3-12, 12-48, >48.
- **Clock:** local hour of day in America/Denver (DST-aware). Day of week dropped: 21 active days
  cannot fill 7 x 24 cells.
- Recency weight 0.5^(age / 14 d): her rate moved ~10x in late October (45-52 messages/week vs 0-6).

## Models (all fit from scratch each hour, from messages strictly before the origin)

| name | role | what |
|---|---|---|
| `reachable_base_rate` | baseline | weighted rate over all past hours, prior 2 % (0.5 in 25 h) |
| `reachable_state` | stored, graded | state-bucket rate, backed off to base (K = 8) |
| `reachable_markov_tod` | served | state x local hour, backed off to state (K = 8) |

## Gate

Log loss per hour. Unit of the test = local day (mean log loss), since hours in a conversation are
not independent. Overlap-aware paired test on the mean (`backtest.paired_win_p`, lag 1), >= 20 days,
Holm across the two subsets. Two subsets:

- **all** hours;
- **cold** hours: no message in the 3 h before the origin (the outreach question: "will she come back?").

Served = model unless `all` is significantly worse; then the baseline. A Wilcoxon / sign test on daily
differences is banned here: quiet days give the model dozens of hair-thin wins, and the sign test
called a cold-hour model "significant" (p < 0.001) whose mean loss was worse.

Also reported: `p_time_of_day_helps` (served vs state-only) and a calibration table.

## Replay result, 2026-07-31..2026-11-10 (evals/run_reachable_replay_eval.py, live data, read-only)

| subset | hours | hours with message | model LL | base LL | skill | p_better (Holm) | verdict |
|---|---|---|---|---|---|---|---|
| all | 2456 | 57 | 0.0767 | 0.0949 | +19 % | 0.040 (0.081) | unclear |
| cold | 2345 | 22 | 0.0497 | 0.0471 | -6 % | 0.91 | unclear (p_worse 0.089) |

- Time of day adds nothing (p 0.58). The whole overall gain is "she is mid-conversation".
- Calibration: under-confident at the top (stated 0.31 -> 0.44 observed; 0.47 -> 0.60).
- **Honest verdict: fails the gate today.** It ships anyway, by Juniper's call, so the live record
  starts accruing. Outreach timing must NOT consume it until `cold` reads `better`.

## Metric quality gate

1. **Provenance:** `chat_message` written by the hub per user turn; reconciled against
   `chat_history_log` non-empty prompts (121 vs 120, one mismatch). `cockpit_turn_sighting` ingress
   was rejected: Juniper's hub session also carries Orion's own outreach prompts.
2. **Independence:** the outcome is Juniper's behaviour, not any substrate node; no existing metric in
   the PE suite reads chat arrival times.
3. **Theory anchor:** human messaging is bursty (inter-message times are heavy-tailed, Barabási 2005),
   so time-since-last is the first-order state; circadian rhythm is the second.
4. **Live sanity:** 2.3 % base rate; per state 48 % / 10 % / 3.6 % / 2.4 % / 0.4 % — not degenerate,
   and the rest state (>48 h quiet) reads ~0.4 %, not a floor.
5. **Existing mechanism:** none forecasts Juniper's arrival. `event_pair` tests A-raises-B, not a forecast.
6. **Reversibility:** one new table, one loop, one endpoint, one flag. Drop table + delete module.

## Files

- `services/orion-prediction/app/reachable.py` (models, verdict, `Reachable` loop)
- `services/orion-prediction/app/store.py`, `app/main.py` (`GET /reachable`, health), `app/settings.py`
- `services/orion-sql-db/manual_migration_prediction_reachable_forecast_v1.sql`
- `services/orion-prediction/.env_example`: `PREDICTION_REACHABLE_ENABLED=true`
- `services/orion-prediction/tests/test_reachable.py`, `evals/run_reachable_replay_eval.py`

## Non-goals

No consumer, no bus event, no outreach change, no household-arrival or turn-lane targets (later B items).

## Acceptance checks

- Live: one row per hour in `prediction_reachable_forecast`, settled ~70 min after its origin.
- `GET :8132/reachable` returns the current p and verdict.
- Re-read verdicts at 20 live days (~2026-12-01); cold `better` is the bar for any consumer.
