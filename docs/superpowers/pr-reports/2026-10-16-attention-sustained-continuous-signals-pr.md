## Summary

Two body signals read all the time: how much machine load readings move (`biometrics`) and how
many message-bus links look abnormal (`bus_synaptic`). Each is in the top 10% of its own week
about 10% of the time, whatever happens, so the attention contest gave them a quota, not a
verdict. Now they must stay unusual to win.

- A body signal that reads every tick may win attention only when it has been hot for more than
  300 s of the last 600 s of its own readings, and is hot now. 300 s is the existing orienting
  window (`EVENT_ORIENTING_WINDOW_SEC`); 600 s is the existing episode open window. No new knob.
- One rule (`orion/attention/sustained.py`), three readers: the attention contests (substrate
  broadcast and field contest), the self-readings block (stance and reverie), and the metacog
  episode gate. On the same history at the same time they give the same answer (pinned by a test).
- Dwell is measured from readings (`substrate_node_prediction_error_history`), not frames or polls.
- Perception (camera, the world), event-written signals (execution, chat, codebase, route), the
  cabinet trigger and `world:chat` are unchanged.
- Kill switch `ATTENTION_SUSTAINED_BODY_ENABLED` (default on) in all four services.

## Outcome moved

Replay of the real code path over 7 days of stored broadcast frames (11,616 frames,
2026-10-10 05:57 to 2026-10-16 03:16 UTC). With the rule off, the same path reproduces the stored
winner on 100.0% of frames.

| Winner | Stored (status quo) | New rule (this PR) | Design doc prediction (b) |
|---|---|---|---|
| no winner | 64.4% | 73.9% | 73.9% |
| perception | 8.2% | 8.7% | 8.7% |
| bus_synaptic | 8.1% | 1.6% | 1.8% |
| biometrics | 4.5% | 0.15% | 0.1% |
| execution | 3.3% | 3.6% | 3.6% |
| world:chat | 3.2% | 3.2% | 3.2% |
| codebase | 2.8% | 3.2% | 3.0% |
| cabinet | 2.3% | 2.3% | 2.2% |
| chat (node) | 1.8% | 1.9% | 1.9% |
| route | 1.4% | 1.6% | 1.5% |

Every row is within 0.2 percentage points of the design doc's prediction (acceptance: within 1).
Frames that biometrics or bus_synaptic used to win now go to: no winner 86.4%, perception 4.0%,
execution 3.4%, codebase 3.2%, route 1.4%, chat 1.3%. No continuous win in the replay has a
dwell of 300 s or less.

Frames whose history covers at least 3 days (7,960 frames; the history table keeps only 7 days,
so early frames re-judge their past readings against a shorter week): no winner 72.7%,
bus_synaptic 1.0%, biometrics 0.21%.

## Current architecture

- `orion/attention/world_first.py` `judge_candidate`: a body candidate was eligible at band
  `high`/`unusual` in its bad direction. Event-written sources faded over 300 s. Per-tick sources
  (biometrics, bus_synaptic) had no time rule.
- Self-readings (`orion/attention/self_readings.py`) called the same judge on the same history.
- The metacog episode gate (`orion/metacog/self_signal_episode.py`) already required "hot for
  more than 300 s in 600 s" before opening an episode, counted in 30 s polls. Its constants were
  copies pinned equal by a test.

## Architecture touched

- New stdlib-only module `orion/attention/sustained.py`: `EVENT_ORIENTING_WINDOW_SEC` (moved here,
  re-exported by `world_first` as the same object), `SUSTAIN_DWELL_SEC`, `SUSTAIN_WINDOW_SEC`,
  `hot_dwell_sec()`, `is_sustained()`.
- `world_first`: `node_prediction_error_continuous()` (glossary: internal, `sparsity: per_tick`,
  level/count, not event-written), `reading_hot_flags()` / `sustained_hot_dwell()` (each recent
  reading judged by `judge_candidate` against the node's 7 days), `node_candidate(pe_history=,
  sustain=)`, the judge rule, and two verdict fields (`hot_dwell_sec`, `sustained`) in the trace.
- Episode gate: imports the shared constants; the open comparison is `is_sustained`; a
  continuous signal confirms on its readings dwell (the contest's number); a high-but-not-yet-
  sustained reading is still a hot poll, so pending/calm-clock behaviour is unchanged.
- Callers thread history and the flag: substrate-runtime broadcast tick (in-memory deque),
  attention-runtime field contest (`NodePeHistoryCache.history`), equilibrium episode poll,
  orion-thought self-readings. History is copied only for the continuous nodes.

## Files changed

- `orion/attention/sustained.py`: the one rule (new).
- `orion/attention/world_first.py`: classification, per-reading dwell, judge rule, trace fields.
- `orion/schemas/attention_candidate.py`: `sustain_window_sec`, `hot_dwell_sec` (optional, None =
  today's verdict).
- `orion/attention/self_readings.py`: same rule, `sustain` parameter.
- `orion/metacog/self_signal_episode.py`: shared constants and comparison; readings dwell for
  continuous signals; `SignalObservation.hot_dwell_sec`.
- `orion/substrate/attention_broadcast.py`: `pe_history_by_node_id`, `sustain`.
- `services/orion-substrate-runtime/app/worker.py`, `settings.py`, `.env_example`,
  `docker-compose.yml`: pass history and flag.
- `services/orion-attention-runtime/app/worker.py`, `settings.py`, `.env_example`,
  `docker-compose.yml`: same for the field contest.
- `services/orion-attention-runtime/evals/replay_world_first.py`: `--no-sustain` comparison.
- `services/orion-equilibrium-service/app/self_signal_episode_gate.py`, `service.py`,
  `settings.py`, `.env_example`, `docker-compose.yml`: pass history and flag to the episode gate.
- `services/orion-thought/app/self_readings_cue.py`, `settings.py`, `.env_example`: flag.
- `scripts/analysis/replay_attention_sustained_body.py`: frame replay through the real path.
- `tests/test_attention_sustained_body.py`: new tests (below).
- `tests/fixtures/self_readings_history.py`: the "busy body" bus anomaly is now a sustained
  run at live 60 s cadence; new `bus_spike_rows` (one-reading flicker).
- `tests/test_attention_event_decay.py`: classify `perception_walkway` (pre-existing failure on
  main since the walkway node joined the glossary).
- `services/orion-equilibrium-service/tests/test_self_signal_episode_gate.py`: stub cache gets
  `history()`.

## Schema / bus / API changes

- Added: `AttentionCandidateV1.sustain_window_sec`, `AttentionCandidateV1.hot_dwell_sec`
  (optional, default None). Frame trace candidates gain `sustain_window_sec`, `hot_dwell_sec`,
  `sustained`.
- Removed: none. Renamed: none.
- Behavior changed: biometrics/bus_synaptic need a sustained high to win attention, to read as
  unusual in self-readings, and (unchanged threshold) to open a metacog episode.
- Compatibility notes: `AttentionCandidateV1` is `extra="forbid"` but is built in-process only
  (not deserialized across services); every service that builds one imports the same `orion`
  package, so the fields arrive with the rebuild. No bus channel change, no table, no migration.

## Env/config changes

- Added keys: `ATTENTION_SUSTAINED_BODY_ENABLED=true` in orion-substrate-runtime,
  orion-attention-runtime, orion-equilibrium-service, orion-thought.
- Removed / renamed keys: none.
- `.env_example` updated: yes (all four) plus compose `environment:` (substrate, attention,
  equilibrium; thought uses `env_file`).
- local `.env` synced: yes, `python3 scripts/sync_local_env_from_example.py --all-keys
  orion-substrate-runtime orion-attention-runtime orion-equilibrium-service orion-thought` (the
  default invocation has no `ATTENTION_` prefix and would skip the key; the only change was the
  new key in each `.env`).
- skipped keys requiring operator action: none.

## Metric quality gate: `hot_dwell_sec`

1. **Provenance.** `world_first.sustained_hot_dwell` over the node's own rows in
   `substrate_node_prediction_error_history` (written by substrate-runtime
   `_prediction_error_magnitudes`); each reading judged by `judge_candidate`.
2. **Independence.** Not a new signal: a time filter on the prediction error that already
   competes. No new upstream, never a second vote next to the PE.
3. **Theory anchor.** `EVENT_ORIENTING_WINDOW_SEC` = 300 s from 225 isolated onsets: one event's
   effect on the body is gone by 300 s. "Unusual for longer than one event can explain" is the
   rule the episode gate and outcome association already used.
4. **Live sanity.** Over 2026-10-12..16 at 5-min samples (1,163 per node): bus_synaptic dwell is
   0 s 39% of the time, median 30 s, p90 152 s, max 600 s, above 300 s 3.2% of the time;
   biometrics 0 s 32%, median 40 s, p90 131 s, max 448 s, above 300 s 0.6%. It rests at 0, moves,
   and reaches the threshold for real runs. Not flat, not saturated.
5. **Existing mechanism.** The episode dwell existed; this patch shares it (constants and
   comparison imported, one dwell function) instead of adding a fourth copy.
6. **Reversibility.** One flag; two optional schema fields; trace fields. Nothing persisted as a
   baseline or trained on.

## Tests run

```text
pytest tests/test_attention_sustained_body.py tests/test_attention_self_readings.py \
  tests/test_attention_event_decay.py tests/test_attention_world_first.py \
  tests/test_attention_world_first_field_frame.py tests/test_attention_world_first_parity.py \
  tests/test_self_signal_outcomes.py                                   -> 156 passed (before review fixes); after: 673 passed across
   tests/test_attention_*, self_signal tests, orion/metacog/tests, equilibrium tests
pytest orion/metacog/tests                                             -> 287 passed
pytest services/orion-attention-runtime/tests                          -> 56 passed, 5 skipped
pytest services/orion-equilibrium-service/tests                        -> 238 passed
(cd services/orion-thought) pytest tests/test_stance_self_readings.py evals/test_self_readings_truth_eval.py
                                                                       -> 28 passed, 3 skipped
pytest services/orion-substrate-runtime/tests (ignoring the DB integration file)
                                                                       -> 17 failed, 460 passed;
   the same 17 fail on clean main (need a local Postgres)
scripts: check_env_template_parity PASS, check_definition_drift --gate PASS,
  check_metric_lineage --gate PASS, --prompt-semantics PASS, check_env_key_single_source OK,
  check_service_env_compose_parity (attention, equilibrium OK; substrate's 17 missing keys are
  pre-existing, not this key), check_sentience_instruments --static-only OK, git diff --check clean
```

New tests pin: one-reading spike does not win; hot > 300 s of 600 s wins; exactly 300 s does
not; a calm gap resets; calm now never wins; perception, execution and cabinet unaffected;
missing history and a dwell failure give today's verdict; kill switch restores today's verdict
(candidate and broadcast frame); constants are the same objects; contest, self-readings and
episode gate agree poll by poll on the same history (and with the flag off the episode keeps its
poll-counted dwell).

## Evals run

```text
pytest services/orion-attention-runtime/evals      -> 4 passed
pytest services/orion-equilibrium-service/evals    -> 22 passed
python3 scripts/analysis/replay_attention_sustained_body.py frames.tsv pe.tsv
  -> table above; sustain off reproduces the stored winner 100.0%
```

## Docker/build/smoke checks

```text
Not run: no deploy or restart in this task (instruction). Live path UNVERIFIED.
```

## Review findings fixed

- Finding (should): one hot reading became "sustained" if the writer stopped after it -- the
  hold cap was the 1800 s staleness horizon, longer than the 600 s window.
  - Fix: a reading speaks for at most twice the source's own median write gap
    (`world_first.reading_hold_sec`, derived from the data, no knob).
  - Evidence: `test_a_held_spike_does_not_become_sustained_when_the_writer_stops` (spike then
    0/200/400/900 s of silence: dwell <= 120 s, never eligible).
- Finding (should): the episode gate compared the contest's 600 s-window dwell with the
  operator's `OPEN_DWELL_SEC`; >= 600 would block continuous episodes, < 300 would open them
  before the contest crowned the signal.
  - Fix: continuous signals use the contest's own `is_sustained` threshold; the env keys govern
    poll-counted signals only (documented in code and settings).
  - Evidence: `test_operator_dwell_setting_does_not_split_continuous_episodes_from_the_contest`.
- Finding (should): the dwell rebuilt a 14k-reading week on every ~2 s field-contest tick (~10 ms).
  - Fix: per-node memo keyed on (length, ends, sum of values) plus the 7-day start index.
  - Evidence: 1.5 ms per call on a hit (measured); `test_memo_never_mixes_two_histories_with_the_same_ends`.
- Finding (should): `self_signal_outcomes` uses single-reading "hot".
  - Fix: not changed; documented under Risks (its high hour already needs the poll-counted dwell).
- Finding (nit): substrate-runtime history can lag the reading being judged after a failed write.
  - Fix: `node_candidate` appends the judged reading when it is newer than the history's last row.
  - Evidence: `test_the_dwell_ends_on_the_reading_being_judged_even_if_history_lags`.
- Finding (nit): unreadable-glossary warning on every call. Fix: warned once per node.
- Finding (nit): test gaps (stale writer, real deque wiring, service poll with non-empty history).
  - Fix: `test_substrate_runtime_deque_history_wires_through_the_broadcast`,
    `test_equilibrium_service_poll_passes_real_history`, plus the stale-writer test above. The
    `perception_walkway` line in `test_attention_event_decay.py` fixes a failure already on main.
- Replay re-run after the fixes: table above (sustain off still reproduces 100.0%).

## Restart required

```bash
# from the clean primary main checkout after merge (deploy rule), one service per line:
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: medium. Concern: a real sudden body fault is silent in attention for its first
  5 minutes. Mitigation: hard faults have their own paths (cabinet trigger, guardian checks, RPC
  timeout alerts); a fault that lasts more than 5 minutes still wins.
- Severity: low. Concern: each recent reading is judged against the node's current 7-day window,
  not the window ending at its own time (differs by ~10 of ~14,000 readings). Mitigation: stated
  in the code; the replay matches the doc's prediction.
- Severity: low. Concern: the dwell costs ~10 ms per continuous node when its history changed
  (about once a minute) and ~1.5 ms on a memo hit (14k readings, measured). Mitigation: only the
  two continuous nodes pay it; the memo key includes the sum of every value, so two histories
  with the same ends never share an entry (pinned by a test).
- Severity: low. Concern: `self_signal_outcomes` ("when this is high, my next hour had ...")
  still judges single readings, but its "high hour" already requires the episode gate's
  poll-counted 300 s-in-600 s dwell, so it is a sustained definition measured in polls, not
  readings. Left as is (out of the spec's file list); a follow-up could move it to readings.
- Severity: low. Concern: live acceptance (biometrics <= 0.5% of frames, bus_synaptic <= 3%, no
  body win with `hot_dwell_sec <= 300` over 24 h) is UNVERIFIED until deployed.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
