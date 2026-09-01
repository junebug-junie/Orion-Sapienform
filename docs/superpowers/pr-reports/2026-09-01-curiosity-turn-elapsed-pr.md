# The turn's own wall time never reached the artifact

Status: **DONE_WITH_CONCERNS** — a lost measurement is now durable and honestly
labelled, but it is an upper bound on the leg that actually matters, not that
leg itself. The `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` knob is deliberately
untouched.

## Summary

- `fcc_timeout` is the largest failure mode of the curiosity loop: **13 of 31**
  runs to 2026-09-01 (42%), against 16 grounded (52%).
- Nothing durable recorded how long a turn took. No `harness_turn_trace` rows
  exist for curiosity correlation IDs; `mind_runs` is a different trigger.
- `elapsed_sec` was already computed in `_generate` and returned in its debug
  dict — and dropped at the call site. It survived only for runs that FAILED to
  journal, and was lost for every run that succeeded.
- Now carried into the journal footprint, labelled for what it actually spans.
- Two real robustness defects fixed alongside.

## Why steps could not answer this

| grounding | runs | mean steps | range |
|---|---|---|---|
| grounded | 16 | 69.8 | 25–127 |
| fcc_timeout | 13 | 95.8 | 76–111 |
| fcc_stream_stalled | 1 | 68 | — |

Timed-out runs do **more** steps, not fewer — so they are not hanging. But the
ranges overlap: a 127-step run finished and a 76-step run was killed. Steps
cannot separate "the budget is too small" from "this turn never converged".

## The correction the review forced

I first sold this number as "the quantity the deadline compares against". Wrong.
There are **three nested deadlines**, confirmed against the live containers:

| Budget | Value | Process |
|---|---|---|
| `HARNESS_FCC_TIMEOUT_SEC` | **1600s** | orion-harness-governor |
| `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC` | 2160s | orion-hub |
| `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` | 2700s | orion-hub — this clock |

`fcc_timeout` is emitted by the **governor** at 1600s (`fcc_motor.py:868`),
which then yields its partial draft as an ordinary final frame — the only
reason a timed-out run has a journal at all. The 2700s budget structurally
cannot kill a journaled run: if it fires, `_generate` returns no text and
`_investigate` bails at `empty_generation` before anything is written.

So every entry carrying this number came from a turn where 2700s was slack, and
for an `fcc_timeout` run the FCC leg is a **constant 1600s by construction**.
The number also spans four legs — stance (≤400s), governor queue, FCC, finalize
(≤485s) — so up to ~885s of it is provably not investigation.

Rendered accordingly: `whole turn Ns (stance + harness + finalize)`, after the
grounding label rather than straight after "harness steps", where `in 2699s`
read as the harness leg.

**What it is still good for:** a grounded run's distance from the 1600s FCC
ceiling is real headroom, and it is an upper bound on the FCC leg. Read it as
that, never as the leg itself.

## Review findings fixed

- **Finding (CRITICAL, semantic): measured against a budget that never fires.**
  - Fix: comment and rendered text rewritten to name all three deadlines and
    state which one kills these runs.
  - Evidence: live `printenv` on both containers; mutation "attribute the time
    to the harness leg again" is red.
- **Finding (HIGH): attributed to the harness but includes ≤885s of other legs.**
  - Fix: the legs are named in the rendered text itself.
- **Finding (MEDIUM): the builder sat outside the try guarding the publish.**
  `harness_elapsed_sec` is the first footprint value formatting with `:.0f`,
  so the first that can raise on a bad type; the journal is the only
  persistence for a turn costing up to 1600s of FCC budget.
  - Fix: moved inside. Evidence: `test_a_malformed_duration_cannot_destroy_the_writeup`.
- **Finding (MEDIUM): the invariant's load-bearing half was untested.** The
  hub-side timeout path returned no `elapsed_sec`. It cannot journal today, but
  nothing enforces that — a later change salvaging a partial draft there (what
  `fcc_motor` already does with `accumulated`) would silently drop the number
  for exactly the runs it exists for.
  - Fix: that path now carries elapsed. Evidence:
    `test_the_hub_timeout_path_records_elapsed_even_though_it_cannot_journal`.
- **Finding: population overstated.** "9 of 16 runs" was an undisclosed window
  over the most recent 16 of 31. Corrected to 13/31 (42%), and "dominant
  failure" qualified — grounded is the plurality at 52%.
- **Finding: a test fixture enshrined a misread.** It fed `2699.6` and asserted
  `2700s` — teaching that a turn finishing 0.4s inside a budget reads as exactly
  the budget. Replaced with a non-boundary value and a comment saying why.
- **Reviewer checked and found clean:** the invariant (only one `_generate`
  path returns text, and it always sets elapsed); blast radius (`grep
  "Investigated over"` finds only the two touched files; the one downstream
  reader, `curiosity_routes.py:189`, returns the body verbatim with no
  parsing); and both tests I flagged as possibly vacuous.

## Files changed

- `services/orion-hub/scripts/curiosity_investigation.py`: carry `elapsed_sec`
  into the footprint; builder inside the try; elapsed on the hub timeout path.
- `services/orion-hub/tests/test_curiosity_investigation.py`: 6 tests.

## Deliberately NOT changed

`HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC`. Raising it would be turning a knob,
and the review makes the case sharper: it is the one budget that never binds
here. Raising the wrong one of the three is called out in
`services/orion-harness-governor/.env_example:39-44` as making things strictly
worse.

The prompt-side clock is already wired and verified end to end — the motor
stamps `ORION_TURN_BUDGET_SEC`, `ORION_TURN_DEADLINE_EPOCH` and
`ORION_TURN_STEP_STALL_SEC`; `_budget_section` reads exactly those names and
already instructs reserving the last quarter of the budget for writing. Orion is
told and overruns anyway, which is a convergence question, not a missing
mechanism.

## Tests run

```text
services/orion-hub/tests/test_curiosity_investigation.py .... 103 passed
tests/test_curiosity_worldview.py + atlas + study_material ... 195 passed
6 mutations verified red (each anchor asserted to match exactly once)
10/10 CI static gates PASS
```

## Env/config changes

None. No new keys.

## Restart required

```bash
# Hub only. The footprint is composed in-process per run.
cd /mnt/scripts/Orion-Sapienform
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --force-recreate orion-athena-hub
```

## Risks / concerns

- Severity: medium. Concern: for `fcc_timeout` runs the FCC leg is pinned at
  1600s, so the rendered number's variance is entirely stance/queue/finalize
  overhead. It bounds the leg but cannot isolate it. Mitigation: the text says
  so; the `grounding` label beside it remains the precise discriminator.
- Severity: low. Concern: nothing splits the four legs anywhere in the repo.
  Follow-up below.

## Next patch

Surface the FCC leg's own duration — the number the 1600s ceiling actually
compares against. `fcc_motor` knows it; no frame returns it to the hub, so this
is a producer/consumer contract change (`orion/schemas/registry.py` + a
governor→hub frame field) rather than a one-liner. That is the measurement that
would settle "budget vs. convergence" outright.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2017
