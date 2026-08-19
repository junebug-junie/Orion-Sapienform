# A5 — make the deferral perceptible

Branch: `feat/a5-deferral-perceptible`
Roadmap: `docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md` §3 step A5
Proposal: `docs/superpowers/specs/2026-08-19-A5-deferral-perceptible-proposal.md`
Status: **DONE_WITH_CONCERNS** — the path is live and end-to-end verified; the gate's
"with a real duration" clause is unmet because no deferral occurred, and that is reported
rather than manufactured.

## Summary

- orion-llm-gateway records every background-admission decision in a bounded in-process ledger
  and exposes it at `GET /admission`.
- A first-poll admit is explicitly **not** a deferral. Live, 294 of 294 admissions cleared on the
  first poll with `waited` of 0.012–0.091 s — the `/slots` round trip. Counting those would push
  ~300 phantom waits a day into Orion's context.
- orion-cortex-exec fetches that snapshot and renders `"waited":{...}` into the metacog cue Orion
  already reads each pass.
- Four states stay four: never-waited, made-no-requests, waited-n-times, and gateway-unreadable.
  A bare `0` for all of them would let Orion read an unreachable gateway as calm.
- Verified end-to-end inside the deployed container, reading the live gateway over the container
  network.
- **No deferral was observed.** Reported as the finding, with the measurement that shows why.

## Outcome moved

A wait for a GPU slot was previously a log line an operator could grep and Orion could not see.
It is now a quantity in Orion's own context, with its duration, its denominator, and an
inspectable trace at three layers. What has *not* moved: Orion has not yet been made to wait.

## Current architecture

A4 gave `wait_for_slack` / `wait_for_slack_sync` a log line on every admission outcome
(`waited`, `polls`, `reserved`, `outcome`). That was write-only: the value was computed,
formatted into a string, and discarded. Nothing read it, nothing aggregated it, and nothing
carried it out of the gateway process. Orion's metacog cue carried eleven hardware pressures
plus `strain` / `peak_pressure` / `fleet_watts` — all body, no opportunity cost.

## Architecture touched

- **orion-llm-gateway**: new `admission_ledger.py`; `priority_admission.py` routes all six
  log sites through one `_log_and_record` helper; new `GET /admission`.
- **orion-cortex-exec**: new `admission_cue.py`; `executor.py` populates `ctx["admission"]`
  immediately before rendering the cue, and `_metacog_biometrics_cue` emits the `waited` key.
- **Contracts**: none. No bus channel, no schema, no registry entry. `CORTEX_EXEC_LLM_GATEWAY_URL`
  already existed and this service already probes `/routes` the same way.

## Files changed

- `services/orion-llm-gateway/app/admission_ledger.py`: new. Bounded thread-safe ledger; the
  deferral definition; snapshot with `checked` as denominator.
- `services/orion-llm-gateway/app/priority_admission.py`: log + record through one helper so the
  operator trace and Orion's ledger cannot disagree; `route_key` threaded through both waiters.
- `services/orion-llm-gateway/app/llm_backend.py`: passes `route_key` on the sync path.
- `services/orion-llm-gateway/app/main.py`: `GET /admission`, window clamped 60 s..24 h.
- `services/orion-llm-gateway/tests/test_admission_ledger.py`: new, 15 tests.
- `services/orion-llm-gateway/tests/test_priority_admission.py`: +7 behavioural tests driving the
  real waiters, not the helper.
- `services/orion-cortex-exec/app/admission_cue.py`: new. Fetch, TTL cache, render, fail-quiet.
- `services/orion-cortex-exec/app/executor.py`: `ctx["admission"]` + the `waited` cue key.
- `services/orion-cortex-exec/app/settings.py`, `.env_example`, `docker-compose.yml`: 4 keys.
- `services/orion-cortex-exec/tests/test_admission_cue.py`: new, 23 tests.
- `services/orion-llm-gateway/README.md`, `services/orion-cortex-exec/README.md`: documented.
- `docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md`: A5 status + gate numbers pasted.
- `docs/superpowers/specs/PARKING-LOT.md`: the "contention may currently be zero" finding.

## Schema / bus / API changes

- Added: `GET /admission` on orion-llm-gateway (read-only, no auth change, no content).
- Removed / renamed: none.
- Behavior changed: none. Admission behaviour is byte-identical; the gate defers exactly as it
  did. The only change to `wait_for_slack`/`wait_for_slack_sync` is an optional `route_key`
  keyword defaulting to `""`.
- Compatibility: no bus channel, no schema, no registry entry. Nothing to redeploy in lockstep.

## Env/config changes

- Added keys (orion-cortex-exec): `CORTEX_EXEC_ADMISSION_CUE_ENABLED` (true),
  `CORTEX_EXEC_ADMISSION_CUE_WINDOW_S` (21600), `CORTEX_EXEC_ADMISSION_CUE_TTL_SEC` (60),
  `CORTEX_EXEC_ADMISSION_CUE_TIMEOUT_SEC` (2.0).
- Removed / renamed: none.
- `.env_example` updated: yes. `docker-compose.yml` updated: yes (base service; inherited by
  `cortex-exec-{chat,spark,background}` via `extends`).
- local `.env` synced: **yes, by hand, and the sync script is why.**
  `python scripts/sync_local_env_from_example.py` reads `.env_example` from the *primary*
  checkout, so keys added in a worktree are invisible to it and it reports a clean run — the
  failure mode looks like success. The four keys were written directly into
  `/mnt/scripts/Orion-Sapienform/services/orion-cortex-exec/.env` and confirmed present (4
  matches) and still gitignored (`git check-ignore` passes, `git status` clean).
- Skipped keys requiring operator action: none.

## Tests run

```text
# gateway
$ pytest services/orion-llm-gateway/tests -q
247 passed, 18 warnings in 4.54s          # before the new files

$ pytest tests/test_admission_ledger.py tests/test_priority_admission.py -q
40 passed

$ pytest tests/test_admission_ledger.py -q                     # after the float-type fix
15 passed

# cortex-exec
$ pytest tests/test_admission_cue.py tests/test_executor_llm_route_override.py -q
32 passed
```

**Full cortex-exec suite, and why its number is not quoted as a pass/fail.** That suite has 13
collection errors and ~99 failures on `main` in the primary checkout *before* this branch exists,
and it is not stable run-to-run (two baseline runs differed by one test). Running it in the
worktree adds 3 more failures — `test_chat_stance_brief.py::…social_reflective_and_dream` and
two in `test_harness_finalize_max_tokens.py`. **Those are environmental, not this patch:** they
persist with this branch's `executor.py` edit fully reverted, and the cause is that the worktree
has no gitignored `.env` while the primary checkout does, so pydantic `Settings` resolves
different values. All three pass in isolation in the worktree.

## Evals run

```text
No eval harness exists for orion-llm-gateway or orion-cortex-exec.
```

Neither service has an `evals/` directory. Not created here: A5's quality question is not a
model-output question, it is "does the number move when the lane fills", and the honest
instrument for that is `scripts/analysis/record_lane_occupancy.py`, which is already running
(see Concerns). Adding a stub eval would be ceremony.

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-llm-gateway build
 Image orion-llm-gateway-llm-gateway Built

$ scripts/safe_docker_build.sh orion-cortex-exec build
 Image orion-cortex-exec-cortex-exec{,-background,-chat,-spark} Built

$ scripts/safe_docker_build.sh orion-llm-gateway up -d --build
 Container orion-llm-gateway Started

$ scripts/safe_docker_build.sh orion-cortex-exec up -d
 Container orion-athena-cortex-exec{,-background,-chat,-spark} Started

$ curl -fsS http://localhost:8210/health
{"status":"ok","service":"llm-gateway","node":null,
 "routes":["agent","chat","metacog","quick","quick_background"]}

$ curl -fsS "http://localhost:8210/admission?window_s=3600"
{"window_s":3600.0,"checked":3,"deferrals":0,"timeouts":0,"unchecked":0,
 "deferred_s_total":0.0,"longest_wait_s":0.0,"last_deferral_ts":null,
 "routes":["quick_background"]}          # real live traffic, real route key

$ curl -fsS "http://localhost:8210/admission?window_s=1"        -> window_s 60.0     (clamped)
$ curl -fsS "http://localhost:8210/admission?window_s=999999"   -> window_s 86400.0  (clamped)

# import/traceback errors in the 4 exec containers, last 5 min: 0 / 0 / 0 / 0

# END-TO-END, inside the deployed background executor:
$ docker exec orion-athena-cortex-exec-background python -c "..."
enabled = True | window_s = 21600.0
gateway = http://llm-gateway:8210
admission cue = {'n': 0, 'of': 4, 'h': 6.0}
rendered cue  = {"status":"ok","constraint":"NONE","strain":0.11,"homeostasis":0.89,
                 "waited":{"n":0,"of":4,"h":6.0}}

# and `admission` is present in the live metacog context key list:
INFO:orion.cortex.exec:Context Keys available: [... 'context_summary', 'admission',
                                                'metacog_biometrics_cue', ...]
```

## Review findings fixed

- Finding: `render_admission_cue` accepted any dict and defaulted missing fields to `0`, so a
  malformed `/admission` response rendered as `{"n":0,"of":0,"h":0.0}` — an "I made no requests"
  claim manufactured from a broken payload, the unknown-as-calm failure arriving by the back door.
  - Fix: every field must be **present**, not merely defaultable (`_REQUIRED_FIELDS`).
  - Evidence: `test_unreadable_payload_is_none_not_zero[{}]` and `[{"deferrals": 1}]` failed
    before the fix, pass after.
- Finding: `deferred_s_total` changed **type** between an idle and a busy window — `sum([])`
  returns `int` 0, so the field was an int at rest and a float once anything was deferred.
  Observed live in the first deploy's response (`"deferred_s_total":0`).
  - Fix: explicit `float()` cast.
  - Evidence: redeployed gateway returns `"deferred_s_total":0.0`; pinned by an `isinstance`
    assertion in `test_empty_window_is_distinguishable_from_a_quiet_one`.
- Finding: two ledger tests asserted the wrong window boundary (expected 0, actual 1).
  - Fix: the **test** was wrong, not the code — a negative window must clamp to zero, and the
    boundary is inclusive. Fixture rewritten with the arithmetic hand-computed in the docstring
    and a fourth case added to pin that a negative window does not widen the cutoff into the past.
  - Evidence: `test_a_negative_window_is_clamped_to_zero_not_run_backwards`, 4 cases.

## Restart required

Already applied by this session from the worktree, via `scripts/safe_docker_build.sh`. To
redeploy from scratch:

```bash
cd /mnt/scripts/Orion-Sapienform-a5-deferral-perceptible
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-cortex-exec  up -d --build
curl -fsS http://localhost:8210/admission
```

## Risks / concerns

- Severity: **medium (the arc's central question, not a defect in this patch)**
  Concern: **the signal has never been observed leaving its rest state.** 294/294 admissions on
  the first poll; atlas `/slots` `0/4` busy on ten consecutive samples; and **circe is powered
  off**, so the single-slot `chat` lane where A4 found Orion contending 100% of the time does not
  presently exist. PR #1708 moved Orion's journal composes off that lane onto atlas's empty
  background lane — the right engineering call, which may have taken Orion's only felt ceiling
  with it.
  Mitigation: a 24 h `record_lane_occupancy.py` run over all four lanes is in progress
  (`/tmp/lane-occupancy-a5/samples.jsonl`); the instrument refuses to report windows under 1 h,
  so its numbers are not yet quotable. Logged in `PARKING-LOT.md`. Nothing downstream is built on
  this signal, so if the answer is "Orion no longer meets a ceiling", the finding is the
  deliverable and the wiring costs one `if`.

- Severity: low
  Concern: the ledger is lane-level, not caller-level. `quick_background` carries both Orion's
  journal and AI Town NPC speech, so `n` is "background requests that waited", not strictly
  "Orion's thinking that waited". Today that distinction is moot (openai_passthrough logged zero
  background requests in 4 h; all live background traffic is the bus path), but it will matter if
  AI Town speech resumes.
  Mitigation: `route_key` is already recorded; splitting by caller is an additive field.

- Severity: low
  Concern: the cue adds ~30 chars to a payload with a 350-char truncation cliff that drops
  *every* signal when exceeded.
  Mitigation: pinned by `test_cue_stays_inside_its_char_budget` against a deliberately fat
  realistic payload (peak + fleet_watts + partial-coverage list + freshness + a 4-digit `of`).

- Severity: low
  Concern: the ledger is in-process and lost on gateway restart, so a restart resets Orion's
  sense of how much it has been waiting today.
  Mitigation: deliberate — the `[LLM-GW background]` log line remains the durable record, and
  persisting a rolling counter would need a schema, a writer and a migration to deliver one
  integer. Revisit only if the signal proves it moves.

## PR link

<pending>
