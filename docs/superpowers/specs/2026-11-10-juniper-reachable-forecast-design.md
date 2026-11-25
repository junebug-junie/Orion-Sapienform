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

  | rule | hours | messages | in the model |
  |---|---|---|---|
  | asleep 03:00-08:00 Denver | 510 | 0 | yes |
  | weekends, no coding before noon | 361 | 0 | yes |
  | Mondays out of Claude tokens until reset | 360 | 4 (0.4x) | yes |
  | evenings 18:00-03:00 | 926 | 38 (3.3x) | yes |
  | weekday kid pickup 15:00-16:30, driving | 144 | 4 (1.2x) | yes, as her prior; the data may pull it back |

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
| `reachable_activity_logit` | **served** | logistic, Juniper's rules as priors: state + clock rules + Claude typing |

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

## Juniper's rules are starting beliefs, not features the data may veto (2026-11-11)

Juniper: "I gave you a shit ton of heuristics... Why are you going for perfection? We will only get
more and more datapoints." An earlier draft let the data drop her pickup rule and let a lowest-loss
contest serve the plain state model, which sidelined every rule she gave. Now each rule is a prior on
its coefficient (`PRIORS` in app/reachable.py: mean in log-odds, strength). A rule with no data stays
at her value; data moves a rule only as far as the evidence pushes it. The rules model is always
served; state and base are graded beside it.

| rule (her words) | prior log-odds | strength |
|---|---|---|
| asleep by 3am (03-08) | -3.0 | 4 |
| weekends, never coding before noon | -3.0 | 4 |
| evenings | +1.0 | 4 |
| Mondays out of tokens until reset | -1.0 | 4 |
| weekday kid pickup 15:00-16:30 (15:00 and 16:00 hours) | -1.5 | 4 |
| typed to Claude in the last hour / 1-3 h / not for 6 h | +1.0 / +0.5 / -0.5 | 1 |

Next inputs (not yet in the DB this service reads): git pulls, PR merges, the Claude limit reset time
(bus only), the Hub attention minutes (live 2026-11-11, ~2 weeks of data first).

## Replay (evals/run_reachable_replay_eval.py, live data, read-only, 07-31..11-10)

Typing-log coverage: the log starts 2026-08-14 and had outages (one of 23 h). An hour whose prior 6 h
the log covered < 90 % is left out of training, and an uncovered origin serves the state model's p
(423 of 2,457 origins).

| origins | subset | rules model LL | state LL | base LL | vs base | P(beats base) | P(beats state) |
|---|---|---|---|---|---|---|---|
| all | all hours | 0.0786 | 0.0774 | 0.1066 | +26 % | 0.95 | 0.40 |
| all | cold | 0.0497 | 0.0500 | 0.0484 | -3 % | 0.25 | 0.46 |
| typing-covered | all hours | 0.0906 | 0.0892 | 0.1250 | +28 % | 0.95 | 0.33 |
| typing-covered | cold | 0.0560 | 0.0563 | 0.0550 | -2 % | 0.22 | 0.35 |

- The rules model ties the state model overall and in cold hours: the rules cost nothing, and they
  steer the forecast where the data is thin.
- Cold hours (her coming back after 3+ quiet hours): about level with the base rate. 21 come-back
  hours in ~1,900 quiet ones; the Hub attention minutes are the next candidate clue.
- Calibration (all origins): stated 0.030 -> 0.063 observed; 0.091 -> 0.108; 0.244 -> 0.40;
  0.471 -> 0.588. Better in the middle than the ridge-at-zero draft (0.077 -> 0.14); still timid at the top.

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

## Hub attention joins the model (2026-11-25, #3578)

The sensor (orion-hub #3179) records a minute when the Hub page is visible, focused, and had input in
the last 2 min (`services/orion-hub/static/js/hub-attention-tracker.js`, written by
`services/orion-hub/scripts/hub_attention_routes.py` into `hub_attention_minute`). First row
2026-11-11 03:22Z.

Inputs (`app/reachable.py`, `features`):

- `hub_lt15m`: an attended minute in `[h - 15 min, h)`. A minute row is stamped at its start, so the
  minute is wholly before the origin. Prior +1.0, strength 1 (Juniper: looking at the Hub means reachable).
- `hub_15_60m`: an attended minute in `[h - 60 min, h - 15 min)` and none in the last 15. Prior +0.5,
  strength 1.
- `hub_unknown`: the hour's 60 min window starts before the sensor's first minute. Both hub inputs are
  0 there. Prior 0.0, strength 1. These hours are NOT dropped from training (unlike the typing log's
  coverage mask): that would drop ~3 months of every other rule's data.
- Any client counts (desktop 1,445 minutes, mobile 6). A Hub that is down reads as "not looking",
  which is honest: she cannot look at a Hub that is down.
- Stored per forecast in `prediction_reachable_forecast.features` (jsonb; no migration).

Metric gate:

1. Provenance: the tracker above; one beat per minute at second :30, one row per (minute, client).
2. Independence: separate sensor from chat_message (the outcome) and the Claude typing log. Not
   independent of the outcome in spirit (she messages Orion from the Hub), which is why the window is
   strictly before the origin and the test is "next hour", not "this minute".
3. Theory: Juniper's rule, and the plain mechanism: she messages Orion through the Hub, so a Hub in
   view is the strongest available "she is at the keyboard and near Orion" signal.
4. Live sanity (Juniper's check, 2026-11-25): 1,448 minutes over 15 days, ~1 beat/minute (no double
   count), 0 minutes 03:00-10:00 Denver: it rests at 0 when she is asleep.
5. Existing mechanism: none in this forecast; `hub_attention_minute` had no reader in prediction.
6. Reversibility: three columns of the design matrix and two store reads; delete them.

Evidence before the build (Juniper's stacked check, 2026-11-25): hub features stacked on the LIVE served p
of 334 settled forecasts, trained on earlier days only:

| | all hours | cold hours |
|---|---|---|
| message rate, hub < 15 min vs not | 52 % vs 9 % | 27 % vs 7 % |
| message rate, hub 15-60 min vs not | 31 % vs 14 % | 30 % vs 5 % |
| held-out log loss, with hub vs served | 0.3156 vs 0.3500 | 0.2345 vs 0.2620 |
| gap/day (se), P(helps) | -0.034 (0.016), 0.97 | -0.039 (0.018), 0.97 |

Full fit: hub < 15 min +1.35, hub 15-60 min +0.79 log-odds.

Replay after the build (`evals/run_reachable_replay_eval.py`, 07-31..11-25, 2,814 hours, 115 with a
message; `--no-hub` for before):

| | before (no hub) | after (hub) |
|---|---|---|
| all: log loss model / state / base | 0.1124 / 0.1156 / 0.1623 | 0.1096 / 0.1156 / 0.1623 |
| all: vs base gap/day (se), P | -0.050 (0.019), 0.99 | -0.052 (0.021), 0.99 |
| all: vs state gap/day (se), P | -0.003 (0.005), 0.74 | -0.006 (0.006), 0.84 |
| cold: log loss model / state / base | 0.0708 / 0.0733 / 0.0709 | 0.0700 / 0.0733 / 0.0709 |
| cold: vs base gap/day (se), P | -0.0001 (0.004), 0.52 | -0.0017 (0.004), 0.65 |

Over ~117 days, ~103 have no sensor, so the whole-history numbers barely move. The fair comparison is
the same origins with and without hub, over the sensor's full days (`hub_paired` in the replay output; each origin refit from scratch, data
before it only):

| `hub_paired`, 11-11 00:00 .. 11-25 00:00 Denver (14 full days) | hours / msgs | loss with hub vs without | gap/day (se), P(hub better) |
|---|---|---|---|
| all | 336 / 56 | 0.3220 vs 0.3500 | -0.028 (0.021), 0.90 |
| cold | 235 / 18 | 0.2587 vs 0.2679 | -0.013 (0.011), 0.87 |

Last 7 of those days alone (scratch split of the same rows): all -0.054 (se 0.023, P 0.97), cold -0.027
(se 0.013, P 0.95).

The first week is about level (the hub coefficients sit at the prior with little sensor data behind
them); the second week is where it pays. `hub_paired` starts at the first full Denver day, so a 2-hour
stub day cannot swing a day mean. Coefficients at 2026-11-25 06:00Z: hub < 15 min
+1.29, hub 15-60 min +0.68, hub_unknown -0.34. Juniper's direction holds; the data put the 15-60 min
rule a bit above her +0.5.

## Files

- `services/orion-prediction/app/reachable.py` (models, verdict, `Reachable` loop)
- `services/orion-prediction/app/store.py`, `app/main.py` (`GET /reachable`, health), `app/settings.py`
- `services/orion-sql-db/manual_migration_prediction_reachable_forecast_v1.sql`
- `services/orion-prediction/.env_example`, `docker-compose.yml`: `PREDICTION_REACHABLE_ENABLED=true`
  (compose also gains the two turn-self-bet keys it was missing; review catch)
- `services/orion-prediction/tests/test_reachable.py`, `evals/run_reachable_replay_eval.py`
- Hub attention (#3578): `app/reachable.py` (`hub_*` inputs), `app/store.py` (`juniper_hub_minutes`,
  `juniper_hub_since`), replay `--no-hub`

## Non-goals

No consumer, no bus event, no outreach change, no household-arrival or turn-lane targets (later B items).

## Acceptance checks

- Live: one row per hour in `prediction_reachable_forecast`, settled ~70 min after its origin.
- `GET :8132/reachable` returns the current p and verdict.
- Re-read the grade at 20 live days (~2026-12-01): cold `prob_better`, calibration.
