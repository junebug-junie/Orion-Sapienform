# Self-monitor act gate: retries of the same job always run, and Prime's acts are gated

## Summary

- The act gate no longer stops retries. When a job has already started once, every later retry of
  that same job runs, even while its faculty is held failing. Only a NEW job's first try is refused
  (unless the 6-hour re-test is due). Juniper chose this as "option A" on 2026-11-03.
- For orion-actions journals, "the same job" means the same dedupe key. That key is already written
  on every journal act as `test_before = journal:<dedupe_key>`. One walkway night, one daily window,
  or one trigger's content each has one key. A retry is logged `self_monitor_same_job_retry`, marked
  in its audit row, and spends no re-test.
- The dispatch runtime needs no such rule. A dispatch that was sent leaves the pending set for good,
  and a refused one never started, so there is no "retry of a started dispatch" there.
- Prime (orion-mind) opens its own acts, so the gate now sits in Prime's loop too. Concern bets
  (`predict`), questions (`ask`), slow/quicken signals (`regulate`) and thoughts (`infer`) each ask
  the gate before they open. Today only `act.prime.predict` is held failing.
- Same one-direction rule everywhere: an unknown, stale or absent self-monitor state, or any gate
  error, changes nothing. Tests pin this for the Prime seam as well.

## Outcome moved

Replay on live data (`python -P -m orion.self_monitor.replay`, read-only, 2026-11-03 20:52Z).

walkway_grade is not held broken today (the 7-day window slid past its worst night). To replay the
case Juniper asked about, it is held `broken` for the whole window with the new `--hold` option.
The window is 7.7 days so it covers the same 7 nights as the first report (10-27 to 11-03).

| rule | tries | refused | re-tests | same-job retries | nights | nights that worked | nights lost |
|---|---|---|---|---|---|---|---|
| before (`--no-same-job`) | 29 | 22 | 7 | 0 | 7 | 7 | **3** |
| option A (this PR) | 29 | 0 | 7 | 22 | 7 | 7 | **0** |

Commands:

```text
POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney PYTHONPATH=<worktree> \
  python -P -m orion.self_monitor.replay --days 7.7 --hold act.actions.journal.make.walkway_grade=broken --no-same-job
  act.actions.journal.make.walkway_grade broken reach=actions_journal attempts=29 refuse=22 probe=7 retry=0
    jobs=7 worked=7 lost=3 refused_outcomes={'failed_to_execute': 19, 'worked': 3}
  ... same command without --no-same-job
  act.actions.journal.make.walkway_grade broken reach=actions_journal attempts=29 refuse=0 probe=7 retry=22
    jobs=7 worked=7 lost=0 refused_outcomes={}
```

Prime, with today's real state (7 days, no `--hold`):

```text
  act.prime.predict unreliable reach=prime attempts=42 refuse=31 probe=11 retry=0
    jobs=42 worked=2 lost=1 refused_outcomes={'worked': 1, 'did_not_work': 29, 'open': 1}
```

So with the gate on, Prime would have placed 11 of its 42 concern bets in the last 7 days instead of
all 42. One of the 31 it would not have placed later worked. That is the cost of the rule: the
self-monitor would not have seen that one success. The concern itself would still have been held
and settled; only the bet on the ledger is missing.

## Current architecture

- `orion/self_monitor/gate.py` (PR #3062, live): refuse-only gate, one re-test probe per faculty per
  6 h, probes counted from the act ledger (`grammar_acts`) plus an in-process memo.
- Two seams called it: the dispatch-runtime allocator and orion-actions `_dispatch_journal`.
- Every retry of a journal job was treated as a new attempt. While walkway_grade was held broken,
  a night got one try (the re-test), and the next re-test came after the night had ended.
- Prime opened its acts on its own path (`services/orion-mind/app/prime_loop.py`). The gate could
  not reach `act.prime.predict`, the faculty held unreliable today.

## Architecture touched

- `orion/self_monitor/gate.py`: `act_gate(..., same_job_attempted=)`, `ActGate(job_key_of=)`,
  `decide(..., job_key=)`, `claim_probe(..., job_key=)`, `note_job_started`, `JobMemo`,
  `job_attempted`. The ledger read now also returns `test_before`.
- orion-actions `_dispatch_journal`: passes `journal_job_key(dedupe_key)`.
- orion-mind Prime loop: `build_act_gate`, gate checks at four act openers (details below).
- `orion/self_monitor/replay.py`: same-job rule, per-job counts, `--hold`, `--no-same-job`.

### Design note (spec amendment; the organ spec is not on main, so it lives here)

Rule (option A): the gate stops NEW attempts at a failing faculty. A retry of a job that already
opened an act inside the ledger window (the probe interval, 6 h) is admitted as `same_job_retry`.
It is not a probe and does not spend one. A job gets at most 24 tries inside one window
(`MAX_SAME_JOB_ATTEMPTS`; the live maximum was 13 walkway tries in one night); after that the normal
rule applies, so a stable job key cannot keep a failing faculty running forever. A job whose first try was refused never opened an act,
so its later tries are still new attempts. A job last tried longer ago than the window is a new
job. The chain is bounded by the job key itself: walkway and daily keys carry the date, and the
other journal keys are content hashes of one trigger.

What a refusal means for Prime, verb by verb:

- `predict` (a concern bet): checked before the concern's act opens. A refusal places no bet. The
  concern stays in Prime's state. It keeps its 24-hour fade, can be kept, dropped or settled by a
  question, and counts in Prime's track record as before. Concerns are not an experiment arm, so no
  arm is touched. Prime is told once per concern, as a frame line that does not wake it. When a
  concern whose act never opened ends, a late bet is placed only if the gate admits it as an
  ordinary act at that moment. While the faculty is failing it is not placed, not even as the
  re-test: its outcome is known by then, so it would test nothing. If the self-monitor state has
  gone unknown, stale or healthy by then, the late bet is placed, exactly as with the gate off. The
  re-test is spent only after the open was published, so a failed publish costs nothing.
- `ask` and `regulate` (a move): checked before the holdback arm is drawn. A refusal removes the
  move from both arms alike, so the treated/control comparison stays fair. Prime sees "my move was
  not sent: ...".
- `infer` (a thought): checked before the body-economy bid. A refusal means no bid and no model
  call. The re-test is spent only when the model is actually reached, which is the same rule the
  thought act already follows. Pending changes stay, so Prime still wants to think, and one thought
  per 6 h keeps its concerns from all fading.

## Files changed

- `orion/self_monitor/gate.py`: same-job rule, job memo, `test_before` in the ledger read, docstring.
- `orion/self_monitor/replay.py`: same-job replay, per-job counts, `--hold`, `--no-same-job`, Prime reach.
- `orion/self_monitor/tests/test_self_monitor_gate.py`: same-job tests (pure, ledger, memo, window,
  faculty match, no job notion), replay before/after.
- `services/orion-actions/app/act_emit.py`: `journal_job_key()`, the one place the job key is built.
- `services/orion-actions/app/main.py`: passes the job key, marks same-job retries in the audit.
- `services/orion-actions/app/settings.py`, `.env_example`, `README.md`: describe the rule.
- `services/orion-actions/tests/test_self_monitor_journal_gate.py`: retry admitted while failing and
  marked, new job refused, re-test then its own retry before the ledger catches up.
- `services/orion-execution-dispatch-runtime/app/{worker,store}.py`: docstrings (why no job key).
- `services/orion-mind/app/prime_loop.py`: the Prime gate.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: 3 keys.
- `services/orion-mind/tests/test_prime_self_monitor_gate.py`: 26 tests.
- `.github/workflows/orion-prime-tests.yml`: also runs on `orion/self_monitor/**`.
- `.github/workflows/self-monitor-tests.yml`: also runs on `services/orion-actions/app/act_emit.py`.
- `docs/superpowers/pr-reports/2026-11-03-self-monitor-effects-pr.md`: amended risk notes.

## Schema / bus / API changes

- Added: none on the bus, no table change. The gate's ledger read selects one more existing column
  (`grammar_acts.test_before`).
- Behavior changed: new admit reason `same_job_retry` (log `self_monitor_same_job_retry`, audit flag
  `self_monitor_same_job_retry: true`). Prime logs `prime_act_refused verb=...` and
  `prime_concern_ended_without_bet`.
- Compatibility: `PrimeStateV1` is unchanged (no new persisted field), so a rollback loads Prime's
  saved state.

## Env/config changes

- Added keys (orion-mind): `SELF_MONITOR_ACT_GATE_ENABLED=true`, `SELF_MONITOR_PROBE_INTERVAL_SEC=21600`,
  `SELF_MONITOR_LEDGER_PG_DSN=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney`.
- `.env_example` updated: orion-mind (keys), orion-actions (comment only).
- local `.env` synced with `python scripts/sync_local_env_from_example.py` (the worktree's script, run
  from the primary checkout): yes. All 3 keys verified in `services/orion-mind/.env`; the DSN equals
  the existing `RECALL_TRANSPORT_PG_DSN`. The orion-mind container resolves `orion-athena-sql-db` and
  has psycopg2.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/prime + orion/self_monitor/tests                                     196 passed
rung3 / motor allocator / self_monitor_act_gate / reading pull / retro /
  aitown + services/orion-execution-dispatch-runtime/tests                 302 passed (first commit)
self_monitor_act_gate + rung3 act candidates + dispatch-runtime tests      205 passed (after fixes)
services/orion-mind/tests (incl. 26 new Prime gate tests)                  199 passed
services/orion-actions/tests (incl. journal gate + act_emit)               294 passed
mutation check: re-adding the review's finding-1 bug fails 3 of the new tests
scripts/check_env_template_parity.py                                       PASS
scripts/check_env_key_single_source.py                                     OK
```

## Evals run

```text
orion/self_monitor/replay (live data, numbers above)
orion/prime/evals/run_prime_frame_eval.py      PASS
orion/prime/evals/run_prime_wake_replay.py     PASS
```

## Docker/build/smoke checks

```text
Not run (build only, by instruction). Live behaviour is UNVERIFIED until the three services restart.
```

## Review findings fixed

Review by an orion-repo-agent subagent on the diff against origin/main.

- Finding (material): a concern refused while failing stayed refused after the self-monitor state
  went unknown, stale or healthy. Its late bet was skipped, and the 60 s re-check refused without
  looking. Gate-off code would have placed the bet.
  - Fix: the late-bet check always asks the gate. The re-check throttle now skips only the ledger
    read, and only while the fresh view still says failing; the view is read every time.
  - Evidence: `test_a_refused_concern_that_ends_after_the_state_goes_unknown_or_heals_gets_its_bet`
    (absent, stale, reliable) and `test_the_recheck_throttle_never_outlives_the_failing_state`.
    Re-adding the bug fails 3 tests.
- Finding (material, test passed for the wrong reason): the gate-fault test used an empty ledger,
  where a working gate already admits as the re-test.
  - Fix: a recent bet in the ledger, and the result must equal gate off. Added a view-read-fault test.
  - Evidence: `test_gate_fault_changes_nothing`, `test_view_read_fault_is_unknown_for_every_verb`.
- Finding (minor): a same-job chain was not bounded (a stable key, for example an empty-content
  hash, could keep a failing faculty running).
  - Fix: at most `MAX_SAME_JOB_ATTEMPTS` (24) tries per job per window; the gate logs the try count.
    The replay applies the same cap.
  - Evidence: `test_same_job_admits_are_capped_per_window`, `test_replay_caps_a_stable_job_key_too`.
- Finding (minor): a concern's re-test was spent before its open was published.
  - Fix: spent only after the publish succeeded.
  - Evidence: `test_a_bet_whose_publish_failed_spends_no_probe_and_is_retried`.
- Finding (minor): a refused thought read Postgres every 15 s tick.
  - Fix: same 60 s ledger re-check throttle as concerns (view still read every tick).
  - Evidence: `test_a_refused_thought_is_rechecked_against_the_ledger_at_most_once_a_minute`.
- Nits fixed: replay no longer imports a private helper (`gate.utc`); orion-mind `.env_example` names
  all four gated verbs; a misleading test comment removed.

## Restart required

Not run. From the clean primary main checkout, after merge, when no chat turn is running:

```bash
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
```

The dispatch runtime changed only in docstrings and the shared gate module (one more column read);
its restart is optional.

## Risks / concerns

- Severity: minor.
  Concern: a refused concern bet hides that bet's outcome from the self-monitor. The replay shows 1
  of 31 refused bets would have worked. The faculty can only show it has healed through its re-tests
  (about 4 a day).
  Mitigation: that is the designed rule; `SELF_MONITOR_PROBE_INTERVAL_SEC` sets the re-test rate.
- Severity: minor.
  Concern: fewer `predict` closes on the `prime` family changes its body-economy history a little
  (7 days: 289 infer, 42 predict, 10 ask closes; about 31 fewer predict closes a week).
  Mitigation: `infer` closes dominate the family's rhythm; watch `prime_bid_not_funded` after deploy.
- Severity: minor.
  Concern: without a readable ledger, Prime's gate uses only its in-process memo, so a restart may
  re-test once early.
  Mitigation: the DSN is set; an early re-test only loosens.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3081

🤖 Generated with [Claude Code](https://claude.com/claude-code)
