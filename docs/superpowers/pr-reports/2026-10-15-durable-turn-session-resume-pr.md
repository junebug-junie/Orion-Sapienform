## Summary

- A durable run's harness turn that is killed from outside (cancel, restart, OOM) now resumes its `claude` session on retry (`claude -p --resume <id>`) instead of restarting from step 1.
- New `orion/harness/session_resume.py` decides start-vs-resume per attempt from a registry on the governor's persistent `harness-claude-config` volume.
- Hub forwards `<run_id>:<source_tag>` as `resume_key`; governor request carries it to the motor.
- Governor: an externally killed durable turn (negative exit, not a cut-short code) replies failed (`interrupted_resumable`, no draft on the reply) instead of finalizing a half-written draft.

## Outcome moved

Orion's Day 10-13 complaint "turns that die mid-sentence": 55+ turns in 7 days ended exit -9 with a long draft and an empty final; each retry started from zero.

## Current architecture

`graph.py` documents `harness_turn` as "the only node whose work is not resumable". The motor captured `claude_session_id` but never used it.

## Architecture touched

orion-harness-governor (motor, runner, bus_listener), orion-hub (curiosity_investigation, turn_orchestrator), `HarnessRunRequestV1` (additive `resume_key`, extra=ignore model; old governor ignores it, so deploy order is free).

## Files changed

- `orion/harness/session_resume.py`: decision + registry (flock, per-attempt token, corrupt-value tolerant)
- `orion/harness/fcc_motor.py`: `run_fcc_turn(resume_key)`, argv `--resume`/`--session-id`, `_record_session_end`, ceiling-guard seed
- `orion/harness/runner.py`, `orion/schemas/harness_finalize.py`, `orion/hub/turn_orchestrator.py`: plumb `resume_key`
- `services/orion-hub/scripts/curiosity_investigation.py`: send the key
- `services/orion-harness-governor/app/bus_listener.py`: `interrupted_for_resume`
- `.env_example` / `docker-compose.yml`: `HARNESS_FCC_MAX_SESSION_RESUMES=3`

## Schema / bus / API changes

- Added: `HarnessRunRequestV1.resume_key` (optional). No new channels. `CuriosityTurnRequestV1` unchanged (the Hub already knows the run id).
- Compatibility: old governor ignores the field; old Hub never sends it.

## Env/config changes

- Added keys: `HARNESS_FCC_MAX_SESSION_RESUMES` (governor). `.env_example` updated; local `.env` synced with `scripts/sync_local_env_from_example.py`. No skipped keys.

## Tests run

```text
pytest orion/harness orion/hub          564 passed
pytest services/orion-harness-governor/tests   80 passed
pytest services/orion-hub/tests/test_curiosity_investigation.py   151 passed
scripts/check_env_template_parity.py    PASS
```

## Evals run

No eval harness for the motor. Live CLI proofs (2026-10-14, governor container): fresh process `--resume` recalled prior context; a real transcript ending on an unanswered tool_use resumed and said where it was cut off; unknown id prints `No conversation found`.

## Docker/build/smoke checks

Compose YAML parses with the new key. Deployed path: UNVERIFIED (worktree cannot deploy).

## Review findings fixed

- Finding: timeout/stall/line-limit endings recorded resumable, so the next turn under the key hijacked the session.
  - Fix: only a negative-exit kill is `interrupted`; all other endings close the turn; key includes `source_tag`.
  - Evidence: `test_motor_verdict_endings_close_the_turn...`
- Finding: urgent runs salvage the draft and report success, never retrying.
  - Fix: governor drops the draft from the interrupted reply.
  - Evidence: `test_sigkilled_durable_turn_fails_without_finalizing_and_drops_the_draft_from_the_reply`
- Finding: late `mark_end` from a cancelled attempt overwrote a newer one.
  - Fix: per-attempt token, flock, unique tmp file. Evidence: `test_a_late_end_from_a_cancelled_attempt...`
- Finding: ceiling seed used transcript bytes. Fix: 0.5 discount, used for both gate and seed.
- Finding: corrupt registry value could kill the turn. Fix: tolerant parse, broad catch.
- Finding: tests hand-built the motor result. Fix: real frames through `HarnessRunner`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build   # from clean primary main after merge
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Medium: resumed context may exceed a smaller lane window; guarded by a rough estimate, corrected by the live ceiling guard (a mis-estimate ends as a cut-short, not a loop).
- Medium: outreach turns still bypass durable runs, so they are not resumed (follow-up).
- Low: two concurrent live attempts for one key would share a session id; the governor cancels the old attempt before the retry.
- Post-deploy check: kill a durable turn mid-run, expect log `fcc_session_resume_decision ... resumed=True` and `session_resume` on the trace row.

## PR link

(see PR)
