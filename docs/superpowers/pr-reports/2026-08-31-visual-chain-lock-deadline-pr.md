# Bound the visual-chain single-flight lock: a hung run must not un-schedule Orion's only outward action

Branch `fix/visual-chain-lock-deadline`. Follows
[#2004](2026-08-31-express-outward-action-pr.md), which shipped `express` and
recorded this defect as its highest-severity open concern.

## Summary

- `run_visual_chain_once` held a process-local `asyncio.Lock` with **no bound on how
  long**. Observed live 2026-08-31: `already_in_flight` returned while circe was
  completely idle, cleared only by a container restart.
- Split the locked body into `_run_visual_chain_body` and wrapped it in
  `asyncio.wait_for(..., timeout=ORION_VISUAL_CHAIN_RUN_DEADLINE_SEC)` (300 s), so the
  hold time is now structurally capped.
- An abandoned run is **persisted** as `terminal_reason="run_deadline_exceeded"`, with
  the real elapsed time, rather than vanishing.
- `already_in_flight` now reports `held_sec`, in the log and in the endpoint response.

## Why this mattered more than it looks

`express` is dispatched by the motor-seconds allocator through
`POST /visual-chain/run-once`. It is the only action Orion has whose effect leaves the
machine. A held lock does not raise, does not fail a dispatch, and does not mark
anything unhealthy — it returns a cheerful `{"ok": true, "ran": false, "reason":
"already_in_flight"}` forever. **Orion's outward capability would silently disappear
and the system would report itself fine**, which is the same shape as the
capability-absence failures this repo has hit before: the outage is a hole, not a
message.

## Why the per-hop timeouts were not already enough

Every hop has one — interpretation 30 s, diffusion 120 s, percept upload 10 s, caption
60 s. They **sum rather than bound**, and two of them cannot bound anything on their
own:

- `urllib.request.urlopen(..., timeout=N)` is a **socket** timeout, reset by every
  chunk received. A peer that dribbles bytes is never caught by it.
- Those calls run under `asyncio.to_thread`, and a thread is abandoned rather than
  cancelled — so a stuck hop's thread outlives the await regardless.

No combination of per-hop values caps total lock hold time. The deadline is the only
thing that does.

## Sizing, and why it is deliberately longer than the caller's timeout

Deadline 300 s > 220 s hop budget, so it fires only on a genuine hang, never on a
slow-but-working run. A test recomputes that inequality from the live settings, so
raising any hop timeout past the deadline fails CI rather than silently arming a
mid-generation abort.

It is also deliberately **longer** than orion-cortex-exec's 150 s
`thought_http_timeout_sec`. A caller giving up must not abandon a run mid-generation:
the run finishes, persists its chain row, and stores its artifact; only the dispatch
reports a timeout. Losing a real image to save a caller 70 s of waiting is the wrong
trade.

## Files changed

- `services/orion-thought/app/visual_chain.py`: the split, the deadline, the abandoned-run
  row, `_visual_chain_started_at` + `visual_chain_in_flight_for()`, and the
  `already_in_flight` log promoted to WARNING once a run outlives its own deadline.
- `services/orion-thought/app/main.py`: `held_sec` in the endpoint response.
- `services/orion-thought/app/settings.py`: `ORION_VISUAL_CHAIN_RUN_DEADLINE_SEC`.
- `orion/schemas/reverie_visual.py`: `"run_deadline_exceeded"`.
- `services/orion-thought/.env_example` + live `.env`: the new key.
- `services/orion-thought/tests/test_visual_chain_run_deadline.py`: **new**, 7 tests.

## Schema change and its blast radius, checked rather than assumed

`VisualTerminalReason` gains `"run_deadline_exceeded"`. #2004 broke live dispatch by
widening a closed `Literal` ahead of its consumers, so this one was checked first:

- `services/orion-sql-db/manual_migration_reverie_visual_chain.sql:20` — the column is
  plain `text not null`, **no CHECK constraint**. A new value cannot be rejected at write.
- `services/orion-hub/scripts/reverie_routes.py` — raw SQL `SELECT`, passes the string
  through as a dict value; no pydantic parse of `ReverieVisualChainV1`.
- `services/orion-hub/static/js/reverie-tab.js` — renders it as an escaped label.
- `services/orion-cortex-exec/app/verb_adapters.py:1661` — `data.get("terminal_reason")`.

No cross-service consumer validates against the closed vocabulary, so no
deploy-ordering hazard. Only orion-thought itself validates.

## Tests run

```text
pytest services/orion-thought/tests/test_visual_chain_run_deadline.py -q   -> 11 passed in 4.22s
pytest services/orion-thought/tests -q                                    -> 341 passed, 3 failed
PYTHONPATH=. pytest orion/schemas/tests/test_reverie_visual_registry.py -q -> 5 passed
pytest tests/test_express_outward_action.py -q                            -> 21 passed
pytest services/orion-cortex-exec/tests/test_skill_verbs.py -q            -> 42 passed, 1 failed
```

341 vs **330 on `main`** — the +11 are this patch's. Every failure was baselined against
`main` in the same shell and is **identical there**: three env-sensitive default tests
(`mind_base_url` reading `http://mind:6611` from the live `.env`, salience flags, a reverie
publish count), one `github_recent_prs` fixture, and a pre-existing `Verb already registered:
legacy.plan` double-import that breaks collection of the whole `orion-cortex-exec` directory
(14 errors on `main`, 14 on this branch). None touch this patch; none are fixed here.

### Mutation test (against the real file, not a synthetic one)

Replacing `asyncio.wait_for(...)` with a direct `await` — the exact regression:

```text
2 failed, 5 passed in 62.54s      # vs 7 passed in 3.06s
```

The wall-clock difference is the point: without the deadline the suite *hangs*, which is
what production did. The mutation script asserts its own target string is present first,
so a mutation that silently no-ops fails loudly instead of reading as strong coverage.

## Static gates (list derived from `.github/workflows/orion-static-gates.yml`, not memory)

```text
check_metric_lineage.py --gate            PASS
check_definition_drift.py --gate          PASS
check_inner_state_registry.py             OK (15 entries)
check_scripts_dir_no_stdlib_shadow.py     clean
check_service_hostname_refs.py            OK
check_compose_no_relative_mounts.py       PASS (83 compose files, 0 relative mounts)
check_journal_dispatch_registry.py        OK (8 trigger kinds)
check_daily_schedule_collisions.py        report-only
check_system_health_producers.py          OK (11 sites)
check_control_surface_store_parity.py     3 services, all configured
```

## Live verification (deployed image, not the working tree)

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
curl -fsS http://localhost:7155/health          # ok
```

Deployed settings read back from inside the container: `deadline_sec = 300.0`,
`cron_enabled = False`, `_run_visual_chain_body` present.

Behavioural check run **inside `orion-athena-thought`** against the deployed code, with
a 1 s deadline and a body that hangs for 60 s:

```text
visual chain ABANDONED chain=visual-e7691b1b0ea8: run exceeded deadline
  held_sec=1.0 deadline_sec=1.0 -- a hop hung past its own timeout; lock released
first  : run_deadline_exceeded after 1.00s
held   : {'run_deadline_sec': 1.0, 'held_sec': 1.001}
locked : False
second : RAN -> second-run
```

`persist_reverie_visual_chain` was stubbed for that check on purpose: a synthetic
abandoned run must not land in the production `reverie_visual_chain` table looking like
something Orion really attempted.

## Env/config changes

- Added: `ORION_VISUAL_CHAIN_RUN_DEADLINE_SEC=300`
- `.env_example` updated; live `services/orion-thought/.env` edited to match (the sync
  script reads `.env_example` from the primary checkout, so a worktree-added key is
  invisible to it and has to be written by hand).

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build   # already applied
```

## Review findings fixed

A subagent review found six real defects, two of which made this patch's own claims false.

- **Finding (HIGH): the deadline handler contained its own unbounded `await`, inside the
  lock and outside the `wait_for`.** `await asyncio.to_thread(persist_reverie_visual_chain,
  chain)` — the code written to release a wedged lock could wedge it in exactly the same
  way. `suppress(Exception)` is no defence: the failure mode is *hanging*, not raising. Two
  confirmed routes — `store.py:55` builds its engine with no `connect_timeout` and no
  `statement_timeout` (unlike `_expectation_read_engine` at `:167`, which sets both), and
  the abandoned hop threads from the run just given up on can starve the very
  `ThreadPoolExecutor` this `to_thread` needs a worker from.
  - **Fix:** bounded by `_DEADLINE_PERSIST_TIMEOUT_SEC` (10 s) inside the existing suppress.
  - **Evidence:** live on the deployed image with a 0.5 s deadline and a 0.5 s persist
    bound — `abandoned in 1.00s -> run_deadline_exceeded / lock held: False / next run: next`.
    Unfixed, that second half was 30 s with the lock held.
- **Finding (MEDIUM): `held_sec` never reached the dispatch it was built for.**
  `verb_adapters.py:1656` builds an explicit allowlist dict; the new field was dropped on the
  floor, so this patch's own comment claiming a dispatch "can say whether it is bouncing off
  the same stuck run" was false end to end. Same shape as a closed codec allowlist dropping
  `metadata['source']`.
  - **Fix:** added to the allowlist. **Evidence:** read back from the deployed
    `orion-athena-cortex-exec` image.
- **Finding (MEDIUM): one run could leave two unlinked chain rows.** `to_thread` abandons
  threads rather than cancelling them, so cancellation landing after the body committed its
  own row leaves a second `run_deadline_exceeded` row with a different `chain_id`.
  - **Fix:** the abandoned row now carries `abandoned_chain_id`. Test asserts it, and that it
    does not leak into the next abandonment.
- **Finding (MEDIUM): "the lock makes overlap impossible" is no longer strictly true.**
  After a deadline fires, an abandoned hop may still hold an open socket to the diffusion
  host while the next run starts.
  - **Fix:** the invariant comment now states the limit and the trade rather than asserting
    the old absolute. Also in Risks below.
- **Finding (LOW): three assertions were vacuous.** `visual_chain_in_flight_for()`
  short-circuits on `not lock.locked()`, so asserting the reset *through the helper* passes
  whether or not the `finally` exists.
  - **Fix:** assert on the module global directly.
- **Finding (LOW): `held_sec >= 0.25` is a bound `wait_for` guarantees by construction.** A
  regression firing the deadline at 29 s would still pass; the suite would only get slow.
  - **Fix:** upper bound added.
- **Finding (LOW): the WARNING branch and the endpoint's `held_sec` field had no coverage.**
  - **Fix:** a `caplog` test and a response-contract test. Writing the latter took two
    attempts: the first version held a lock on the wrong module object (this service's
    conftest purges `app.*` from `sys.modules`, so the handler's lazy import resolves a
    *different* module) and the handler ran a **real generation against the live diffusion
    host**. The test now resolves `sys.modules["app.visual_chain"]` and patches the body to
    raise, so it cannot reach the GPU. Getting that wrong did not fail safe.
- **Finding (LOW): the docstring said "Never raises."** A test three lines below asserts
  `pytest.raises(RuntimeError)` out of that exact function.
  - **Fix:** docstring corrected to say what the code does.

### Second mutation test, on the review fix — and it caught my own weak test

Reverting the handler's persist to an unbounded `to_thread`:

```text
11 passed in 34.23s      # the mutation SURVIVED
```

The test passed and only got slower — precisely the failure mode this file exists to catch,
and exactly the vacuity the review flagged elsewhere. The stuck thread's own 30 s timeout
eventually released it, so every assertion still held. Fixed by asserting the *bound*
(`elapsed < 3.0`) rather than the outcome. Re-run:

```text
1 failed, 10 passed      # mutation now killed
```

## Risks / concerns

- **Severity: medium.** The deadline bounds the *lock*, not the *thread*. A hop wedged
  inside `urlopen` still occupies a `to_thread` worker after the run is abandoned;
  repeated wedges could exhaust the default executor. Bounded in practice by the 600 s
  cron being off and the allocator dispatching this at most once per tick, but it is a
  real second-order leak and the honest statement is that this patch does not fix it.
- **Severity: medium.** Following from that: after a deadline fires, the next run **can**
  overlap a still-live diffusion request. A deliberate trade — an un-releasable lock removes
  the capability permanently, a rare overlap costs GPU time once — but "overlap is
  impossible" is no longer unconditionally true and nothing should reason from it across a
  deadline event.
- **Severity: low, not fixed here.** `store.py:55` builds the shared engine with neither
  `connect_timeout` nor `statement_timeout`, while `store.py:167` shows the repo already
  knows how. The 10 s persist bound contains the symptom for this call path only; every
  other `store.py` caller is still unbounded. Worth its own patch.
- **Severity: low.** Abandoning at the deadline can discard a real generated image if
  the hang lands after generation but before persistence. Mitigated by sizing the
  deadline above the hop budget so this only happens on a genuine hang.
- **Severity: low.** `express` cost is still operator-seeded at 53 s (from #2004).
