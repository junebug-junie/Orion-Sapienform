# Tell Orion its own clock, and keep the last quarter for writing

## Summary

- The FCC motor stamps `ORION_TURN_BUDGET_SEC`, `ORION_TURN_DEADLINE_EPOCH` and
  `ORION_TURN_STEP_STALL_SEC` into the `claude -p` sandbox at spawn time, so a
  turn can read the deadline the timeout loop is actually enforcing.
- The curiosity kickoff prompt tells Orion where to look, discloses the tighter
  per-step stall wall, and reserves the last quarter of the budget for writing.
- Nodes are written at the moment they are formed rather than at the end.
- No duration is hardcoded anywhere in the prompt, and a test scanning the whole
  assembled prompt fails if anyone writes one in.
- `scripts/check_env_key_single_source.py` + `make check-env-key-single-source`
  turn "one owner per tuned env key" into a failing gate.
- Four stale `HARNESS_FCC_TIMEOUT_SEC=900` copies retired.

## Outcome moved

Run `32b42392f495` — the first curiosity run to complete the whole loop — spent
its entire budget investigating and was killed mid-writeup: `grounding=fcc_timeout`,
`draft_len=66`, one hop of five recorded. The prior that survived had wrong counts
in two independent ways:

| what Orion wrote | what the database says |
|---|---|
| "646 active stance" | **20**. 646 was every active crystallization of every kind. |
| "substantive_shift 599 rejected vs 181 active" | 181 is active **semantic**. Among `stance`, it is 599 rejected vs **0** active. |
| substantive_shift as a rejection filter | It is a `propose` trigger — `orion/memory/consolidation_gate.py:74-87` records why a crystallization was *created*. |

The real separation is total (0/599 of `substantive_shift`-proposed stance survive;
20/57 of `repair_signal`-proposed do), i.e. stronger than what Orion recorded, and
pointing at a different mechanism.

The investigation was sound. The transcription was done against a wall. Neither
change here is a bigger timeout.

## Current architecture

`HARNESS_FCC_TIMEOUT_SEC` was known only to the harness-governor. The prompt said
nothing about time. `_build_subprocess_env` passed the gateway URL, auth token and
curiosity credentials into the sandbox, but nothing about the turn's own deadline.
The write section read as an end-of-turn form.

## Architecture touched

- `orion/harness/fcc_motor.py` — three env keys stamped at spawn.
- `orion/curiosity/kickoff_prompt.py` — new `_budget_section`, inserted before
  `_hops_section` in all three prompt states.
- No bus, schema or HTTP contract touched. No new configured key.

## Files changed

- `orion/harness/fcc_motor.py`: stamp the budget, deadline and stall cap; clear
  them when unknown.
- `orion/curiosity/kickoff_prompt.py`: `_budget_section`; gate the continuation
  note on `writable`; write-at-formation.
- `orion/curiosity/README.md`: §7 subsection on the clock; corrected two claims.
- `scripts/check_env_key_single_source.py`, `Makefile`: the one-owner gate.
- `services/orion-harness-governor/app/settings.py`: `900.0` → `1600.0`.
- `services/orion-hub/app/settings.py`, `orion/llm/routes.py`,
  `services/orion-llm-gateway/README.md`: stale `=900` literals retired.
- `tests/test_curiosity_worldview.py`, `tests/test_env_key_single_source.py`,
  `orion/harness/tests/test_fcc_motor_mcp.py`: coverage.

## Schema / bus / API changes

None. `ORION_TURN_*` are computed per turn, never read from config, never
operator-settable.

## Env/config changes

- Added keys: none configured. Three sandbox-injected values, produced by
  `_build_subprocess_env`.
- `.env_example` updated: no — nothing operator-settable was added.
  `services/orion-harness-governor/.env_example` is raised to 1600 by #1898,
  which this branch is based on.
- local `.env` synced: not applicable, no template changed.
- Behaviour changed: `services/orion-harness-governor/app/settings.py`'s code
  default for `HARNESS_FCC_TIMEOUT_SEC` moves 900 → 1600. Only reachable if the
  key is absent entirely; compose always passes it.

## Tests run

```text
pytest tests/test_curiosity_worldview.py tests/test_curiosity_acl_and_credentials.py \
       tests/test_curiosity_study_material.py tests/test_env_key_single_source.py \
       orion/harness/tests/ -q
  348 passed, 3 failed

pytest services/orion-hub/tests/test_curiosity_investigation.py -q   ->  62 passed
pytest services/orion-harness-governor/tests -q                      ->  18 passed
pytest services/orion-llm-gateway/tests -q                           -> 285 passed

make check-env-key-single-source
  OK: 1 owned env key(s), no drifted copies.
```

The 3 failures (`test_grounding_capsule_consumers.py` x2,
`test_harness_runner.py::test_harness_runner_surfaces_fcc_error_code`) reproduce
identically on unmodified `origin/main`. Pre-existing.

Two measurement traps hit and corrected while running these, both matching known
lessons: running two `tests/` roots in one invocation produces a collection
collision that reported 52 phantom gateway failures (matched runs: 285 passed on
both branch and main), and `tests/` as a whole has 41 pre-existing collection
errors from missing service deps, identical on main.

## Evals run

```text
No eval harness exists for orion/curiosity/. The acceptance check for this
feature is behavioural and takes ~20 runs (does confidence ever go DOWN), and is
recorded in orion/curiosity/README.md §13 rather than as an eval.
```

## Docker/build/smoke checks

```text
./scripts/safe_docker_build.sh orion-harness-governor config
  HARNESS_FCC_TIMEOUT_SEC: "1600"
  HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC: "180"
```

Live check of the stamping, through the real `_build_subprocess_env` and a real
child process running the exact commands the prompt gives Orion:

```text
budget=1600   left=1600   stall=180
--- with no deadline passed ---
budget=no clock   left=no clock   stall=unknown
```

**UNVERIFIED:** that these reach a `claude -p` sandbox inside the deployed
governor container. `orion/` is baked into the image (`COPY orion ./orion`), so
this needs a rebuild plus a real turn, and the daily cap was exhausted.

## Review findings fixed

- Finding: `_budget_section` named the continuation note in all three prompt
  states, but `_outcome_section` — the only place `continue_note` appears — is
  gated on `writable`. A prompt naming a capability the run does not have is the
  exact failure the three-state split exists to prevent.
  - Fix: gated the clause; non-writable states are told to put it in their prose.
    Also fixed the pre-existing "leave yourself a note (below)" pointing at an
    unrendered section.
  - Evidence: `test_the_continuation_note_is_only_named_when_it_can_be_written`,
    `test_a_sixth_hop_points_somewhere_real_in_every_state`.
- Finding: the prompt presented one wall; the per-step stall cap can kill a turn
  earlier while the outer clock still reads generous.
  - Fix: stamped `ORION_TURN_STEP_STALL_SEC` and disclosed it. Corrected
    `README.md`'s "the only number Orion sees is the one the timeout loop is
    enforcing" and named the two walls that remain undisclosed.
  - Evidence: `test_the_per_step_stall_wall_is_disclosed_next_to_the_turn_clock`.
- Finding: `env.pop` does not produce the behaviour its comment claimed —
  `$(( $UNSET - $(date +%s) ))` prints a confident negative and exits 0.
  - Fix: the prompt tests for emptiness and says what `no clock` means; the
    rationale in code and README now says what the clearing really buys (an
    absurd number instead of a plausible one).
  - Evidence: measured `-1787785130`, exit 0.
    `test_the_clock_commands_survive_the_variables_being_unset`.
- Finding: the duration guard's regex missed `a ~26-minute budget`, `roughly 26
  min`, `900 sec`, `26m` — and the first is the phrasing used in this feature's
  own source comments.
  - Fix: widened, and scans the whole assembled prompt in both graph states
    rather than one function.
  - Evidence: mutation-tested, 8/8 phrasings caught.
- Finding: `HARNESS_FCC_TIMEOUT_SEC` live at 1600 while six other places said
  900 — the exact drift this patch's design argument is built on.
  - Fix: `scripts/check_env_key_single_source.py`, wired to
    `make check-env-key-single-source`. Reads the owner file rather than
    hardcoding the number. Four literals retired; the remaining two are in
    `docs/superpowers/` archives and are excluded on purpose.
  - Evidence: mutation-tested against the real `.env_example` — drifting the
    owner to 900 flags both real copies (compose default and `settings.py`).
- Finding: `test_build_subprocess_env_omits_the_deadline_when_it_is_not_known`
  is a near-tautology that passes with the `pop` lines deleted.
  - Fix: relabelled as a signature check; the monkeypatched sibling carries the
    coverage.

## Restart required

Depends on #1898 merging first. Then, from a worktree:

```bash
./scripts/safe_docker_build.sh orion-harness-governor build
./scripts/safe_docker_build.sh orion-harness-governor up -d
./scripts/safe_docker_build.sh orion-hub build
./scripts/safe_docker_build.sh orion-hub up -d
curl -fsS http://127.0.0.1:7156/health
```

Both bake `orion/` into the image (`COPY orion ./orion`), so a restart alone is
not enough.

## Risks / concerns

- Severity: low. Concern: the deployed-container path is UNVERIFIED (see above).
  Mitigation: first run after deploy should be checked for `hops` > 1 and a
  `:Finding` whose counts survive a re-query.
- Severity: low. Concern: `ORION_TURN_DEADLINE_EPOCH` is wall-clock while
  enforcement is monotonic; a host clock step mid-turn would make them disagree.
  Mitigation: rounding-level on a ~26-minute budget, and the alternative is a
  number the sandbox cannot compare against anything.
- Severity: low. Concern: the one-owner gate covers exactly one key today.
  Mitigation: adding a key is a one-line registry entry.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1900
