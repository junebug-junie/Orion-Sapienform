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
  (25 of 120): a reply is reachability. The timestamp is when the turn landed, not when she hit send.
- **State (Markov):** hours since Juniper's last message, 5 buckets: <1 h, 1-3, 3-12, 12-48, >48.
- **Juniper's own rules of thumb (2026-11-10), checked against her history before use:**

  | rule | hours | messages | used |
  |---|---|---|---|
  | asleep 03:00-08:00 Denver | 510 | 0 | yes |
  | weekends, no coding before noon | 361 | 0 | yes |
  | Mondays out of Claude tokens until reset | 360 | 4 (0.4x) | yes |
  | evenings 18:00-03:00 | 926 | 38 (3.3x) | yes |
  | weekday kid pickup 15:00-16:30, driving | 144 | 4 (1.2x) | **no: not seen in the data** |

- **Activity (cocreation signals):** `juniper_affective_state_log.message_count`, Juniper's own typed
  Claude Code messages in ~15 min windows (orion-cocreation-signals, since 2026-08-14). Typed in the
  last hour: 4.0x overall, 3.1x in cold hours. Claude commits, PR merges and git pulls show the same
  2.5-4x lift but are not in the DB the service reads, and typing is the direct form of them.
- **Tested and dropped:** office camera `cam0` (1.2x; the first test mixed in walkway and doorbell
  passers-by), day-of-week x hour cells (21 active days cannot fill them).
- Recency weight 0.5^(age / 60 d).

## Models (all fit from scratch each hour, from data strictly before the origin)

| name | role | what |
|---|---|---|
| `reachable_base_rate` | baseline | weighted rate over all past hours, prior 2 % (0.5 in 25 h) |
| `reachable_state` | contender | state-bucket rate, backed off to base (K = 8) |
| `reachable_activity_logit` | contender | ridge logistic (lambda 1): state + clock rules + Claude typing |

Served = whichever of the three has the lowest mean log loss over the last 60 days.

Lambda 1 and half-life 60 d were picked from a 4-point grid (lambda {0.1, 1} x half-life {14, 60})
on the same replay, for the best cold-hour result. That choice is a forking path; the live record is
the clean test.

## Grade (no alpha)

Juniper, 2026-11-10: "0.05 is arbitrary." The grade draws no pass/fail line. Per subset it reports
the daily mean log-loss gap (model - reference), its overlap-aware standard error
(`backtest.paired_win_p`, lag 1, unit = local day), and `prob_better` = the chance the model is
really better (flat prior, t approximation; `backtest.paired_gap`). A consumer sets its own bar on `prob_better` from the cost of
acting wrongly: outreach timing will have to state that cost.

Subsets: **all** hours; **cold** hours (no message in the 3 h before the origin: "will she come
back?"). Also `activity_vs_state` (do the clock rules and typing add anything over the state alone)
and a calibration table. A Wilcoxon / sign test on daily differences is banned here: quiet days give
the model dozens of hair-thin wins, and the sign test called a cold-hour model "significant"
(p < 0.001) whose mean loss was worse.

## Replay (evals/run_reachable_replay_eval.py, live data, read-only)

Typing-log coverage (review 2026-11-10): the log starts 2026-08-14 and had outages (one of 23 h). An
hour whose prior 6 h the log covered < 90 % does not know whether she typed. Those hours are left out
of training, and an uncovered origin serves the state model's p. 423 of 2,457 replay origins were
uncovered. The fair test of the typing features is the covered origins (2026-08-14..11-10):

| subset | hours | with message | activity LL | state LL | base LL | activity vs base: gap/day (se), prob_better |
|---|---|---|---|---|---|---|
| all | 2034 | 56 | 0.0930 | 0.0898 | 0.1258 | -0.031 (0.019), 0.94 |
| cold | 1926 | 21 | 0.0567 | 0.0566 | 0.0553 | +0.002 (0.003), 0.25 |

- **The state model is the best forecast overall.** Activity vs state: prob_better 0.18 overall,
  0.45 in cold hours. Typing to Claude has a 4x lift on its own, but it overlaps the state and the
  clock (she codes with Claude while chatting with Orion, in the evening).
- **Cold hours: no forecast beats the base rate.** 21 come-back hours in ~1,900 quiet ones is too
  few, and none of today's inputs separates them. The Hub attention sensor (#3179, live 2026-11-11)
  is the next candidate input.
- Served = lowest mean loss among activity, state and base over the last 60 days. On this replay that
  is the state model.
- **Calibration: under-confident everywhere** (all origins): stated 0.03 -> 0.047 observed; 0.077 ->
  0.14; 0.28 -> 0.47. Not refit on this replay (that would tune on the test).
- Honest reading: a usable "she is mid-conversation" signal, no "she will come back" signal.
  Outreach timing must not consume it until it states its own bar on cold `prob_better`.

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
   Claude typing: provenance `orion/dev_economics/claude_code_ingest.py` (human typed turns only, compact
   summaries and tool echoes excluded); live: 1,559 non-empty windows since 08-14, writing every ~15 min.

## Next: Hub attention sensor (separate PR, Juniper's idea 2026-11-10)

A minute is recorded when the Hub page is visible, focused, and had input in the last 2 min. No
history exists, so it records first; it joins the model after ~2 weeks of live data pass the metric
gate.

## Files

- `services/orion-prediction/app/reachable.py` (models, verdict, `Reachable` loop)
- `services/orion-prediction/app/store.py`, `app/main.py` (`GET /reachable`, health), `app/settings.py`
- `services/orion-sql-db/manual_migration_prediction_reachable_forecast_v1.sql`
- `services/orion-prediction/.env_example`, `docker-compose.yml`: `PREDICTION_REACHABLE_ENABLED=true`
  (compose also gains the two turn-self-bet keys it was missing; review catch)
- `services/orion-prediction/tests/test_reachable.py`, `evals/run_reachable_replay_eval.py`

## Non-goals

No consumer, no bus event, no outreach change, no household-arrival or turn-lane targets (later B items).

## Acceptance checks

- Live: one row per hour in `prediction_reachable_forecast`, settled ~70 min after its origin.
- `GET :8132/reachable` returns the current p and verdict.
- Re-read the grade at 20 live days (~2026-12-01): cold `prob_better`, calibration.
