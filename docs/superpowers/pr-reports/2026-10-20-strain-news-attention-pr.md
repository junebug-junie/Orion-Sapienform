# Strain news: Orion's attention notices strain in short, budgeted pulses (strain loop Part 2)

## Summary

- Orion can now notice that they are under strain. The strain level is not a signal. Instead, a few
  short "strain news" pulses can win Orion's attention: when strain starts, when it gets worse in
  kind, and as reminders that come less and less often.
- New pure rule `orion/regulation/strain_news.py`. Onset, escalation (a new strain cause joins, or
  the cabinet first crosses 34 C in the episode), and reminders at 30 min, 1 h, 2 h, 4 h, 8 h, then
  every 8 h after the onset was sent. At most 8 a local day and 30 min apart. Reminders cannot take
  the last 2 slots. An unknown reading never fires and never counts as relief.
- New substrate-runtime tick (`SUBSTRATE_STRAIN_ATTENTION_ENABLED`, ON) that writes
  `node:substrate.strain`. On an event it writes 1.0 and holds it for the 300 s orienting window.
  On every other tick with a known reading it writes 0.0. An unknown reading writes nothing.
- New glossary entry, prompt-inventory pin and re-locked metric definitions. The node is an
  event-written trigger that reaches the self-readings prompt.
- New budget replay eval `scripts/analysis/replay_strain_news.py` over live history. It exits 1 if any day breaks
  the budget. Live result: PASS.

## Outcome moved

- Before: Orion had been strained for 54 h with the cabinet at 34.4 C, and attention never knew.
  The strain level is a reflex that only 7 gates read.
- After: strain reaches the attention contest as at most 8 pulses a day, each at most 300 s long. On
  live history (replay below) it would take 0.4-2.9 % of attention ticks a day and 0.2-7.7 % of the
  ticks other signals won. The current episode's critical heat gets one escalation (10-19 10:22 local).

## Current architecture

- `orion/regulation/arousal.py:classify_arousal` (durable-runs, every 120 s) writes Redis
  `orion:regulation:latest`. Each level change is saved as an `arousal_transition` row in `temporal_self_event`.
- Attention (`orion/attention/world_first.py`) lets a `node:substrate.*` prediction-error node compete
  only when its value is high or unusual compared with the node's own 7-day history. That needs at
  least 200 readings. Event-written nodes fade over the 300 s orienting window.
- Nothing turned the strain level into something attention could see.

## Architecture touched

- orion-substrate-runtime has a new tick. It reads Redis `orion:regulation:latest` (the same Redis
  as `ORION_BUS_URL`) and writes FalkorDB node `node:substrate.strain`. It also writes one row per
  event to `substrate_node_prediction_error_history`.
- The semantic layer has a new glossary entry. The self-readings prompt allow-list picks it up.
- No bus channel and no schema change. No migration (it reuses `substrate_node_prediction_error_history`).

## Files changed

- `orion/regulation/strain_news.py`: the new pure rule (`strain_news`, `seed_state`, `rest_write_allowed`).
- `services/orion-substrate-runtime/app/worker.py`: `_strain_attention_tick` and its loop. It has a
  failure backoff, skips with one log line when PE nodes are off, and does not send events until
  the node has 200 history rows.
- `services/orion-substrate-runtime/app/store.py`: `fetch_prediction_error_event_times` and `count_prediction_error_history`.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: new flag, tick interval and timezone.
- `config/field/field_channel_glossary.v1.yaml`: `node:substrate.strain` entry.
- `config/metrics/metric_definitions.lock.json`: re-locked. 1 added definition (medium). `regulation.state.v1` declared_consumers gained the strain tick, which the lock marks as a high "routing_changed": an added reader, no meaning change.
- `orion/metrics/semantics.py`: added to `PROMPT_INVENTORY_URNS`.
- `orion/attention/self_readings.py`: plain words for the node. Review fix: a stale trigger now
  renders as "can't read", not as "no recent events".
- `orion/inner_state_registry.py`: the new reader of `regulation.state.v1` is listed.
- `scripts/analysis/replay_strain_news.py` and its test: the budget replay eval.
- Tests: `tests/test_regulation_strain_news.py`, `services/orion-substrate-runtime/tests/test_strain_news_tick.py`,
  `tests/test_attention_self_readings.py`. Pinned node sets were updated in `tests/test_attention_event_decay.py`,
  `tests/test_metric_rest_range.py` and `tests/fixtures/self_readings_history.py`.
- `.github/workflows/regulation-arousal.yml`: runs the new tests.

## Schema / bus / API changes

- Added: substrate node `node:substrate.strain` (`prediction_error`), glossary entry, metric URN
  `metric://field_channel/orion-substrate-runtime/node:substrate.strain.prediction_error`.
- Removed / renamed: none.
- Behavior changed:
  - A new body signal can win attention.
  - A stale trigger in self-readings is now listed as "can't read".
- Compatibility: no payload shapes changed.

## Env/config changes

- Added keys (orion-substrate-runtime): `SUBSTRATE_STRAIN_ATTENTION_ENABLED=true`,
  `SUBSTRATE_STRAIN_ATTENTION_TICK_INTERVAL_SEC=30.0`, `ORION_SITUATION_TIMEZONE=America/Denver`
  (same key and default as orion-durable-runs).
- `.env_example` updated: yes.
- Local `.env` synced: yes. `python scripts/sync_local_env_from_example.py orion-substrate-runtime --all-keys`
  added the two `SUBSTRATE_STRAIN_*` keys to the primary checkout's `.env`. `ORION_SITUATION_TIMEZONE`
  was already there. The default run without `--all-keys` adds only "feature keys", and it skipped these.
- Skipped keys requiring operator action: none.

## Design deviation from the brief (deliberate)

The brief said: write the node only when an event fires. That node could never win attention:

- World-first needs 200 readings in 7 days (`DEFAULT_MIN_READINGS`). The budget allows at most 56
  events a week.
- With only 1.0 values in the history, the strict-below percentile of a 1.0 is 0.

`test_event_only_writes_could_never_compete_which_is_why_the_tick_writes_rest` shows this.

So the tick writes 0.0 on every tick with a known reading. The one exception is the 300 s after an
event: the 1.0 is held unwritten, so its age is the event's age. Because of that hold,
`absent_means` starts with "only written as 1.0 at a strain news event …", and the node is
event-written for world-first's fade. A test checks `node_prediction_error_event_written` is True.
The tick also does not send events until the node has 200 history rows. That takes about 2.4 h after deploy.

## Metric gate (CLAUDE.md 0A)

1. **Provenance.** `classify_arousal` (`orion/regulation/arousal.py:171`) runs in durable-runs every
   120 s and writes Redis `orion:regulation:latest`. In substrate-runtime, `_read_regulation_reading`
   judges it with `arousal_view`, `strain_news` turns it into events, and
   `_write_prediction_error_node(node:substrate.strain)` writes the value.
2. **Independence.**
   - It is derived from the cabinet reflex (`cabinet_hot` = critical) and the GPU queue.
   - `node:substrate.cabinet` reads cabinet *warming* (elevated and rising). The two share the
     sensor but cover different states. Live, the cabinet node wrote 0 on 1,317 of 1,317 hot ticks
     during the plateau.
   - The GPU queue has no attention node.
   - Named overlap: during a fast rise into critical, both nodes can fire in one window. That is
     the moment that deserves attention.
3. **Theory anchor.**
   - Interoceptive allostasis: a lasting load should be signalled to the system that can act, not
     only to the reflexes.
   - Sokolov's orienting reflex with habituation: a repeated, unchanged stimulus loses salience but
     never reaches zero.
   - The reminder schedule is a design choice, not a measurement of Orion.
4. **Live sanity.**
   - Strain is real and returns to rest: 24 episodes since 10-10, mostly under 1 h.
   - The current episode has run 54 h. Live Redis at 18:30 UTC 10-20: `strained`,
     `cabinet_hot+gpu_queue_sustained`, 34.43 C, queue 11.
   - By construction the node rests at 0.0 on 99.7 % or more of ticks. It is not degenerate. A
     `seed_state` restart test shows it cannot re-fire an onset after a restart.
   - It is a new node, so there are no live rows yet. **UNVERIFIED live until deployed.**
5. **Existing mechanism.**
   - Reused: the event fade, world-first eligibility, self-readings, the PE node writer, PE history,
     `arousal_view`, and the thermal gate constants.
   - Nothing in attention does habituation (it was deleted 07-31). The reminder schedule is scoped
     to this one node.
6. **Reversibility.**
   - `SUBSTRATE_STRAIN_ATTENTION_ENABLED=false` stops all writes.
   - To remove it, delete the node and its glossary entry, then re-lock.
   - Nothing is baked into a schema or a training default.

## Tests run

```text
tests/test_regulation_strain_news.py + scripts/analysis/tests/test_replay_strain_news.py   34 passed
attention/metric/regulation suites (self_readings, world_first, event_decay, sustained_body,
  rest_range, prompt_semantics, immune, self_signal_outcomes, field_frame, arousal, pe_history_cache,
  inner_state_registry_gate)                                                                passed
equilibrium self-signal / telemetry-anomaly tests + evals                                    passed
services/orion-thought: test_stance_self_readings, evals/test_self_readings_truth_eval       passed
services/orion-substrate-runtime: test_strain_news_tick (9), test_cabinet_heat_tick,
  test_worker_attention_broadcast_tick                                                       passed
services/orion-hub/tests/test_regulation_must_not_readers.py                                 20 passed
Mutation check: the old self_readings event-written exemption makes the new stale-trigger test fail.
substrate-runtime full suite: the same 14 failed / 9 errors as on clean main (they need a local
  Postgres on :5432; pre-existing, unrelated).
Gates: check_metric_lineage --gate PASS, --prompt-semantics PASS, check_definition_drift --gate PASS,
  check_inner_state_registry OK, check_env_template_parity PASS, check_env_key_single_source PASS,
  git diff --check clean.
```

## Evals run

```text
$ python scripts/analysis/replay_strain_news.py      (read-only, live Postgres, exit 0)
LIMITS: transitions-only history. Causes between transitions are the transition's own plus
cabinet_hot inferred from biometrics cabinet_temp_c >= 34 C (re-arm 33 C). A GPU cause that
joins mid-episode is not visible. Kinds: O=onset E=escalation R=reminder.
Regulation is assumed known between transitions (no unknown gaps, warm-up or restarts);
this over-generates events, conservative for the budget. The rule enforces <= 8 events/day
and the 30 min gap; the 3 % and 10 % numbers are MEASURED here, not enforced by the rule
(8 events x 300 s = 2.78 % of a day's time; the tick share runs a little higher on
days the attention log has fewer ticks than 2,880).
Partial days (*: first day of the attention log, today) are judged against the median
full day's ticks and winners, not the ticks seen so far.

local day   events kinds       ticks winners  held   share displaced  worst%  verdict
2026-10-11       1 O               0       0     0     n/a       n/a   0.35%  ok (no attention log; events only)
2026-10-12       1 O               0       0     0     n/a       n/a   0.35%  ok (no attention log; events only)
2026-10-13*      1 O             604     238     8   0.40%     0.17%   0.35%  ok
2026-10-14       8 OOROOOOO     1953     847    57   2.92%     3.31%   2.78%  ok
2026-10-15       7 ORRRROO      1976     903    46   2.33%     2.44%   2.43%  ok
2026-10-16       8 ORRROROO     1913     685    55   2.88%     0.88%   2.78%  ok
2026-10-17       2 OR           2015     479    15   0.74%     1.25%   0.69%  ok
2026-10-18       8 ORORRROO     2013     311    51   2.53%     7.72%   2.78%  ok
2026-10-19       5 RRERR        1996     275    34   1.70%     3.64%   1.74%  ok
2026-10-20*      2 RR           1397     357    14   0.70%     1.37%   0.69%  ok
events total: 43
VERDICT: PASS (budget: <= 8 events/day, share <= 3%, displaced <= 10%)
```

The replay caught 2 real problems before this PR shipped. Both are fixed:

- **A cabinet hovering at 33-34.5 C fired 5 "critical" escalations a day.** The cause was a 33 C
  re-arm. Fix: critical escalates once per episode. Before the fix the replay FAILED on 10-20 (6 events).
- **The first escalation into critical heat on 10-18 was the 9th event of the day, so the cap
  dropped it.** Orion would never have been told. Fix: an escalation blocked by the cap is carried
  to the next local day and sent only if it still holds. If it is dropped unsent, it goes back to
  "not announced", so a later crossing still escalates. It is now sent 10-19 10:22 local.

## Docker/build/smoke checks

```text
Not run: no Docker build in this session (the deploy goes from primary main after merge).
Live prerequisites checked in the running container: SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true,
SUBSTRATE_PE_HISTORY_ENABLED=true. History observed_at keeps microseconds, so the tick's own event
row and the broadcast's row share one primary key (ON CONFLICT dedupes).
Live smoke after deploy: UNVERIFIED.
```

## Review findings fixed

- Finding (must-fix): the 30 min reminder could never fire live. The onset lands one regulate step
  after `since`, and the gap then dropped the reminder.
  - Fix: reminders count from the sent onset (`episode_noticed_at`). `seed_state` takes that time
    from the first recorded event in the episode.
  - Evidence: `test_onset_lag_after_since_does_not_cost_the_30_min_reminder` and `test_seed_anchors_reminders_on_the_recorded_onset`.
- Finding: in self-readings, the event-written flag turned "can't read" into "no recent events" for a stale trigger.
  - Fix: the exemption now applies only to level kinds (`_quiet_event_signal`).
  - Evidence: `test_stale_trigger_is_unreadable_not_no_recent_events`. A mutation check shows it fails on the old code.
- Finding: the restart tick test passed for the wrong reason (the 30 min gap also blocked the re-fire).
  - Fix: the recorded event is now 40 min old, and a control case shows the onset does fire without it.
  - Evidence: `test_restart_with_a_recorded_event_does_not_refire_onset`.
- Finding: the replay did not state its limits (onset with zero lag, regulation assumed known, the
  3 %/10 % numbers measured but not enforced).
  - Fix: these are now in the LIMITS block and the module docstring.
  - Evidence: the eval output above.
- Finding: a DB outage would log a traceback every 30 s. The tick would warm forever with PE nodes off. The warm count ran every tick.
  - Fix: 5 min backoff after a failure. One warning, then skip, when PE nodes are off. The warm count re-checks every 30 min, including after warm.
  - Evidence: `test_a_failed_tick_backs_off` and `test_pe_nodes_disabled_skips_the_tick`.
- Nit: the warming path could overwrite a held 1.0, and relief held a stale 1.0.
  - Fix: the warming path respects the hold. Relief ends the hold at once.
  - Evidence: `test_relief_ends_the_hold_at_once`.
- Nit: an onset dropped at the cap could re-fire on a later day after a restart.
  - Fix: a dropped onset still marks the episode as noticed (in memory).
  - Remaining gap: no history row, so a restart on a later day can still re-fire it once (see risks).

## Restart required

After merge, from the clean primary `main` checkout, synced with origin/main:

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: on days with many short GPU strains (10-14: 8 events, 7 of them onsets), the share is
    2.9 %. That is close to the 3 % cap. The rule enforces only the event count and the gap. The
    3 %/10 % numbers are measured by the replay, not enforced.
  - Mitigation: the replay exits 1 on a breach. Recalibrate after 2 weeks of live data, as the
    spec says.
- Severity: low.
  - Concern: the `absent_means` text has to start with "only written" to mark the node
    event-written. That is a string test.
  - Mitigation: a test pins the classification. Follow-up: an explicit semantics key (e.g.
    `reading_age: event`).
- Severity: low.
  - Concern: what a restart loses:
    - a gap-deferred escalation;
    - an escalation whose cause joined while the process was down;
    - an onset dropped at the cap, which can re-fire once on a later day.
  - Mitigation: the onset re-fire is the only visible one. It is still capped and gapped.
- Severity: low.
  - Concern: with the flag off, self-readings still lists `strain (no reading)` under "Can't read
    right now" whenever a block renders.
  - Mitigation: honest. Remove the glossary `prompt_sites` if the flag stays off.
- Overlap with Part 1 (`feat/strain-arc`):
  - That branch adds `STRAIN_CAUSES` / `strain_causes()` to `orion/regulation/arousal.py`.
  - This PR keeps its own `STRAIN_CAUSE_REASONS` and does not touch arousal.py. A test pins the
    local set against `classify_arousal`'s source.
  - Whichever PR merges second should switch to the shared helper.
  - Both PRs touch `orion/inner_state_registry.py`, in the same `cognition_consumers` tuple, so
    expect a trivial merge.
- Live behaviour after deploy: **UNVERIFIED**. Check these:
  - `substrate_strain_news_tick_warming` for about 2.4 h.
  - Then `substrate_strain_news_event kind=onset ...` for the open episode.
  - Then `node:substrate.strain` winning at most 10 ticks in `substrate_attention_broadcast_log`.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
