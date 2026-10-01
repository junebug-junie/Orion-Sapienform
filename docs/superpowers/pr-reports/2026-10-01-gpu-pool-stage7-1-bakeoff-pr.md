## Summary

Stage 7.1 of the GPU pool concurrency arc (spec PR #2442): a tool Juniper runs as one line on circe
to answer the two questions stage 7.2 depends on. Does running two Bonsai conversations at once on
gpu2 (2 slots x 131K, flash attention on) give at least 1.3x the throughput of one? And does
llama.cpp's shared prompt cache ever leak one conversation into another (#27148)?

- `scripts/bench/stage7_1_bakeoff.sh preflight|run|cleanup`: checks gpu2's seat is idle, holds the
  diffusion role so diffusion-host can't load next to the worker, pauses pool actuation, starts Bonsai
  on gpu2 at `--parallel 2`/262144 (card forced, then checked three ways), runs the bench and the canary,
  and on every exit stops the worker, releases the hold and resumes the pool.
- `scripts/bench/stage7_1_client.py` (stdlib only): the depth bench (14K/32K/61K/100K, 1 vs 2 concurrent
  runs), the bleed canary (codewords in content, reasoning and tool calls, plus `cache_n` against the
  true shared token prefix), preflight logic, summary and a draft field-note table.
- `scripts/bench/stage7_1_pool_ctl.py`: pause/resume/hold/release over the bus, run inside the Bonsai
  image because circe has no venv.
- `--with-q4`: an optional second pass on the stock Q4 27B image at 2 x 65K (2 x 131K doesn't fit).
- Field note skeleton with the go/no-go table: `docs/2026-10-01-gpu-pool-stage7-1-bakeoff.md`.

No service code changes.

## Outcome moved

Before this, 7.1 was a spec row with no way to run it safely. Now it is one line on circe that
refuses to start unless gpu2 is empty, and leaves the pool the way it found it. Its result decides
whether 7.2 ships two slots, stays at one, or first has to turn off the prompt cache on every
multi-slot lane.

## Current architecture

- `services/orion-llamacpp-bonsai-host` (#2434/#2447): a manual Bonsai worker. Its profile is 4 x 65K,
  and it can only be pointed at gpu1 or gpu2.
- The wrapper already honours `LLAMACPP_N_PARALLEL_OVERRIDE` / `LLAMACPP_CTX_SIZE_OVERRIDE` (since
  January), so 2 x 131K needs only a compose override file, not a new profile.
- Pool controls `pause_actuation`, `resume_actuation`, `hold`, `release` existed (stage 5.7).
  `scripts/gpu_pool_pause.py` needs the repo venv, which circe doesn't have.
- The previous #27148 probe ran on metacog/fast only (dense 8B, b10398).

## Architecture touched

- No service code, schema, bus channel or env key.
- The pool is driven only through the existing control verbs.
- The worker announces as `LLM_ROLE=bonsai-bakeoff`, so the pool lists it as unclaimed and routes no
  traffic to it.

## Files changed

- `scripts/bench/stage7_1_bakeoff.sh`: the orchestrator. It has a dry-run mode, a lock, an always-run
  cleanup, and signal-safe waits.
- `scripts/bench/stage7_1_client.py`: bench, canary, preflight check, `paused-by`, and summary.
- `scripts/bench/stage7_1_pool_ctl.py`: the pool control verbs (reuses `gpu_pool_pause.parse`).
- `tests/scripts/test_stage7_1_bakeoff.py`: 27 tests. The orchestrator runs in dry-run with stub
  docker, nvidia-smi and curl. Also covers the pass rule, detectors, summary and envelopes.
- `services/orion-gpu-pool/tests/test_stage7_1_bakeoff_sequence.py`: runs the script's exact control
  sequence through the real pool runtime.
- `.github/workflows/orion-gpu-pool-tests.yml`: path filters, plus a step for the tooling tests.
- `services/orion-llamacpp-bonsai-host/README.md`: points to the tool.
- `docs/2026-10-01-gpu-pool-stage7-1-bakeoff.md`: field note skeleton. Same directory convention as
  `docs/2026-09-30-ternary-bonsai2-27b-1xv100-circe.md`; there is no `docs/superpowers/field-notes/` directory.
- `docs/superpowers/pr-reports/2026-10-01-gpu-pool-stage7-1-bakeoff-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none. The tool only uses existing control verbs.
- Compatibility notes: the operator hold is sent with holder `operator:stage7-1-bakeoff` and the pause
  with actor `stage7-1-bakeoff`. Cleanup keys on those names.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none.
- **Finding (not changed here):** circe's `services/orion-llamacpp-bonsai-host/.env` still says
  `BONSAI_CUDA_VISIBLE_DEVICES=0` (chat's card), which predates #2447. The tool forces 2 regardless
  (verified with `docker compose config` on a copy that says 0), but a manual `up` from the README
  would land on gpu0. That `.env` is Juniper's to fix.

## Tests run

```text
PYTHONPATH=. pytest tests/scripts/test_stage7_1_bakeoff.py -q            -> 27 passed
cd services/orion-gpu-pool && pytest tests/test_stage7_1_bakeoff_sequence.py tests/test_stage5_7_enforce.py -q
                                                                          -> 18 passed
pytest services/orion-llamacpp-bonsai-host/tests -q                       -> passed (with the above: 26)
bash -n scripts/bench/stage7_1_bakeoff.sh                                 -> ok
```

## Evals run

```text
Live tool validation against circe metacog (:8012, b10398, 4 x 4096), from athena, kept tiny:
  bench  depths 1000,2500, 64-token decodes, 3 reps: prompts within 1-3% of target, every warm decode
         a cache hit, decodes overlapped 99.8-99.9%. Ratios 0.79-0.87 on a busy live lane: NOT evidence
         about concurrency, only that the harness works.
  canary 31 requests x 2 runs: 0 bleed, 0 cache_n over the true prefix, own-codeword recall 0.96-1.0,
         6 tool calls emitted. metacog returned no reasoning text, so the reasoning channel was not
         exercised there (the canary now calls that WEAK).
Real `preflight` executed on circe (read-only): reached the pool from circe, gpu2 1,248 MiB, OK.
`run` without --yes on circe: changed nothing.
`stage7_1_pool_ctl.py check` on athena: imports + bus OK (sends nothing to the pool).
```

There is no eval harness for a manual bench. The real eval is Juniper's run, which fills the field note.

## Docker/build/smoke checks

```text
docker compose --env-file <copy of .env_example with BONSAI_CUDA_VISIBLE_DEVICES=0> -f bonsai compose
  -f override config  (with BONSAI_CUDA_VISIBLE_DEVICES=2 in the shell)
  -> CUDA_VISIBLE_DEVICES_OVERRIDE "2", LLAMACPP_N_PARALLEL_OVERRIDE "2", LLAMACPP_CTX_SIZE_OVERRIDE "262144",
     LLM_ROLE bonsai-bakeoff, container orion-bench71-bonsai-worker
No container started, no pool pause, nothing touched on gpu2 (Juniper runs the real thing).
```

## Review findings fixed

- Finding (blocker): the pool refuses an operator hold while paused, so pause-then-hold always aborted.
  - Fix: the order is now hold, then pause. After the hold, the script checks that gpu2 has not
    started swapping before it pauses.
  - Evidence: `test_stage7_1_bakeoff_sequence.py` drives the real runtime and pins the refusal.
- Finding (blocker): cleanup decided whether to resume from a local file. A lost pause reply could
  leave the pool paused, and a stale file could resume someone else's pause.
  - Fix: cleanup now resumes only if `/v1/pool` says it was paused by `stage7-1-bakeoff`. The file is
    a fallback for when the pool can't be read, and it is now written before the pause is sent.
    Resume waits up to 60 s and then confirms against pool state.
  - Evidence: `test_a_lost_pause_reply_is_still_resumed`, `test_someone_elses_pause_is_never_resumed`, `test_paused_by`.
- Finding: nothing stopped a second run, or a cleanup started elsewhere, from tearing down a live run.
  - Fix: a `flock` covers both `run` and `cleanup`.
  - Evidence: `test_a_second_run_or_cleanup_is_refused_while_one_holds_the_lock`.
- Finding: a second signal could kill cleanup partway through, and an ssh drop aborts the run.
  - Fix: cleanup ignores INT/TERM/HUP, and the run is launched in tmux.
  - Evidence: `test_sigterm_mid_bench_cleans_up_promptly` (TERM mid-step: cleanup in <30 s, exit 143).
- Finding: if the worker couldn't be stopped, cleanup still released the hold and resumed, which
  invites a ~24 GB load next to a stuck worker.
  - Fix: escalate to `docker kill`. If the container is still there, keep the hold and the pause and
    print the manual steps.
- Finding: adding up each request's own tok/s can pass with no real concurrency.
  - Fix: the verdict now uses combined tokens over the pair's wall span. A depth only counts if the
    two decodes overlap at least 90% and both hit the cache; otherwise it is INVALID and the verdict
    INCOMPLETE.
  - Evidence: `test_pass_rule_pass_fail_incomplete` (staggered case fails), `test_window_overlap`.
- Finding: `summarize` could call a partial run PASS.
  - Fix: planned depths and the planned turn count are saved, and anything that didn't finish is INCOMPLETE.
  - Evidence: `test_summary_never_passes_an_interrupted_run`.
- Finding: `/apply-template` was called without the request's `chat_template_kwargs`.
  - Fix: the same kwargs are now passed. On the live validation, thinking turns showed `cache_n` equal
    to the computed bound.
- Finding: detector 2 cannot see the upstream symptom.
  - Fix: this is now stated in the docstring and the field note. Separately, I added WEAK when no
    thinking turn returned reasoning, a gap the live validation exposed.
- Findings (nits):
  - Fixed: world and urgent-diffusion side effects are documented; the card is now checked in
    `compose config` before `up` and by gpu2 memory growth after boot; the temp reply file is cleaned
    up; a failed release prints recovery steps; repeats went from 2 to 3.
  - Also fixed, from CI: the dry-run tests needed the gitignored service `.env`. Added a
    `BONSAI_ENV_FILE` override.

## Restart required

```text
No restart required (no service code). To run the bake-off on circe (one line, ~60-80 min, in tmux):
git -C /mnt/scripts/Orion-Sapienform fetch -q origin docs/gpu-pool-stage7-1-bakeoff && git -C /mnt/scripts/Orion-Sapienform worktree add -f --detach /mnt/scripts/Orion-Sapienform-bench71 FETCH_HEAD && tmux new -s bench71 '/mnt/scripts/Orion-Sapienform-bench71/scripts/bench/stage7_1_bakeoff.sh run --yes; read -p "done - Enter closes"'
```

## Risks / concerns

- Severity: medium
  - Concern: the pool controls run inside the Bonsai image on circe, and that path has not run there yet.
  - Mitigation: the first step, `check`, imports the exact modules and pings the bus. It aborts before
    anything is held or paused.
- Severity: medium
  - Concern: about an hour of world-model predictions is lost, and diffusion work waits.
  - Mitigation: documented. Run it off-hours.
- Severity: low
  - Concern: the 2 x 131K boot, the VRAM fit with world-model resident, and every result number are
    UNVERIFIED until the run.
  - Mitigation: the tool checks slots, ctx and card before sending load, and dies cleanly otherwise.
- Severity: low
  - Concern: an urgent diffusion request could borrow the hold's slot.
  - Mitigation: no such caller exists today; this is documented.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2463

🤖 Generated with [Claude Code](https://claude.com/claude-code)
