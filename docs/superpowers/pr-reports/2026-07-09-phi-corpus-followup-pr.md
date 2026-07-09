# PR: φ corpus pipeline follow-up — underbuild fixes + corpus hygiene (specs 1 audit + spec 3)

**Status:** IMPLEMENTED, reviewed, tested. Three commits on
`fix/reasoning-telemetry-underbuild-followup`, covering two related but
distinct pieces of φ-corpus follow-up work.

## Summary

1. **Reasoning telemetry underbuild fix** (`bda17650`) — audit of the just-merged
   `feat/reasoning-telemetry-adapter` found two fields marked "reserved" in
   `ReasoningCallV1` that were actually available and never wired
   (`prompt_tokens` sits in the same `usage` dict as `completion_tokens`;
   `thinking_enabled` was hardcoded `False` despite `ctx["chat_template_kwargs"]`
   already being forwarded by `executor.py`). Also fixed a silent env-sync
   allowlist gap that left the prior PR's new keys absent from local `.env`.
2. **φ corpus hygiene Part A** (`8daeecf7`) — capped the unbounded
   `active_execution_trajectory.runs` projection (29,165 runs / 25 MB, no
   eviction ever since May 25) with LRU eviction by `last_updated_at`.
3. **φ corpus hygiene Part B** (`b48a900d`) — added a shared, pure,
   never-raising ingestion-time health gate (`is_corpus_row_healthy`) and
   wired it into both φ-corpus writers, closing the live per-tick writer's
   complete absence of any gate (it wrote every row unconditionally before
   this patch).

Parts A and B were built in parallel by two subagents on disjoint file sets,
each independently caught a real bug during self-review (a tie-break
eviction bug in Part A, fixed with a structural guard + regression test) and
an env-parity gap (`docker-compose.yml` missing the new keys, fixed by the
orchestrator). Full detail per piece is in the two standalone PR docs already
in this branch: `PR_reasoning_telemetry_underbuild_followup.md` and
`PR_phi_corpus_hygiene.md` (repo root) — this file is the single consolidated
report covering the whole branch for PR submission.

## Outcome moved

- φ's `ReasoningCallV1` payload is now data-complete for the fields that have
  a live signal (still gated off by `PUBLISH_REASONING_TELEMETRY=false`).
- The 25 MB/29k-run execution_trajectory backlog self-prunes on the next tick
  after deploy; producer memory/payload size is now bounded going forward.
- New garbage stops entering the φ training corpus at write time instead of
  being silently discovered later at fit time — critically, the live
  spark-introspector writer (the actual high-volume path) is now gated for
  the first time ever.

## Current architecture (before)

- `ReasoningCallV1.prompt_tokens`/`thinking_enabled` were hardcoded
  `None`/`False` at the emit call site despite live data being one hop away.
- `active_execution_trajectory.runs` grew without bound; only the freshest
  ~120s were ever consumed downstream, so 99.98% of the payload was dead
  weight fetched every introspector tick.
- `scripts/backfill_phi_corpus.py` had a partial inline health check;
  `services/orion-spark-introspector/app/worker.py`'s live writer had none.

## Architecture touched

- `services/orion-cortex-exec/app/router.py` — `_provider_completion_meta`
  extracts `prompt_tokens`; new `_thinking_enabled_from_ctx` helper.
- `orion/schemas/telemetry/reasoning.py` — corrected field comments.
- `orion/substrate/execution_loop/{constants,reducer,pipeline}.py` — LRU cap.
- `services/orion-substrate-runtime/{app/settings.py,.env_example,
  app/worker.py,docker-compose.yml}` — cap config, end to end incl. Docker.
- `orion/telemetry/corpus_gate.py` (new) — shared health predicate.
- `scripts/backfill_phi_corpus.py`,
  `services/orion-spark-introspector/app/worker.py` — gate wiring.
- `scripts/sync_local_env_from_example.py` — allowlist fixes (twice, same
  failure mode both times: new `.env_example` keys not covered by the sync
  script's prefix allowlist).

## Files changed

See the two standalone reports (`PR_reasoning_telemetry_underbuild_followup.md`,
`PR_phi_corpus_hygiene.md`) for the full per-file breakdown; this file is the
submission-level summary.

## Schema / bus / API changes

- No shape changes to any schema. `ReasoningCallV1`/`ReasoningActivityV1`,
  `ExecutionTrajectoryProjectionV1`, `InnerStateFeaturesV1` are all unchanged
  — every change here is either data-completeness (fields now populated) or
  a write-path filter/cap, not a contract change.

## Env/config changes

- Added: `EXECUTION_TRAJECTORY_MAX_RUNS=2000`, `EXECUTION_TRAJECTORY_MAX_AGE_SEC=86400`
  (`services/orion-substrate-runtime`).
- No new keys for the reasoning-telemetry fix or corpus-gate Part B.
- `.env_example` files updated; local `.env` synced (confirmed no drift via
  `python scripts/sync_local_env_from_example.py`); `docker-compose.yml`
  passthrough added for the new substrate-runtime keys.

## Tests run

```text
pytest services/orion-cortex-exec/tests/test_router_final_text_assembly.py \
       services/orion-cortex-exec/tests/test_reasoning_emit.py \
       services/orion-cortex-exec/tests/test_cognition_trace_metadata.py -q
  → 50 passed

pytest tests/test_execution_substrate_reducer.py tests/test_execution_substrate_pipeline.py \
       tests/test_execution_projection_schemas.py tests/test_corpus_gate.py -q
  → 33 passed

pytest services/orion-substrate-runtime/tests --ignore=.../test_grammar_consumer_integration.py -q
  → 88 passed, 14 failed (pre-existing, confirmed via git-stash comparison — 87/14 without
    this branch's changes; the +1 pass is new coverage)

pytest services/orion-spark-introspector/tests -q
  → 80 passed, 1 skipped, 1 failed (pre-existing, confirmed via git-stash comparison —
    test_phi_reward_emit.py::test_phi_reward_emitted_when_encoder_ok, unrelated)

pytest services/orion-thought/tests -q
  → 123 passed, 4 pre-existing failures (confirmed via git-stash comparison, unrelated)
```

## Review findings fixed

See per-piece detail in the standalone reports. Headline findings:
- Reasoning telemetry: 0 findings (medium-effort review, clean).
- Corpus hygiene Part A: a subagent self-review caught a real tie-break
  eviction bug (fixed, regression test added); a nested review subagent
  caught a missing `docker-compose.yml` env passthrough (fixed by
  orchestrator).
- Corpus hygiene Part B: a scoped subagent review caught one test-quality
  nit (fixed) and confirmed no blocking issues.
- Orchestrator full-diff pass across all three commits (code-review skill,
  medium effort): no further findings.

## Docker/build/smoke checks

Not run against live containers in this environment. Config validated via
diff review + settings/compose consistency checks. Restart required for the
execution-trajectory caps and corpus gate to take effect (see below); no
restart required for the reasoning-telemetry field fixes (same flag-gated,
default-off behavior as before, just more complete data once enabled).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low, across all three pieces. Reasoning telemetry stays
  flag-gated off. Execution-trajectory caps default well above the
  downstream consume window. The corpus gate only affects new writes, not
  historical data (fit-time filtering still applies there, per spec).
- Branch-hygiene note: this PR bundles three related-but-distinct pieces of
  φ-corpus follow-up work (reasoning-telemetry field fixes + both corpus
  hygiene parts) into one branch/PR rather than three. All three are part of
  the same umbrella initiative (`2026-07-09-phi-truthful-corpus-overview.md`)
  and none conflict, but flagging in case separate PRs are preferred —
  happy to split before merge if so.

## PR link

Branch pushed: `fix/reasoning-telemetry-underbuild-followup`
(commits `bda17650`, `8daeecf7`, `b48a900d`).
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/reasoning-telemetry-underbuild-followup

`gh` is unauthenticated in this session (`gh auth status` fails — no
`GH_TOKEN`/`GITHUB_TOKEN` set), so the PR could not be opened via CLI. Run
`! gh auth login` to authenticate, then:
```bash
gh pr create --title "fix: phi corpus pipeline follow-up (reasoning telemetry + hygiene)" \
  --body-file docs/superpowers/pr-reports/2026-07-09-phi-corpus-followup-pr.md \
  --base main --head fix/reasoning-telemetry-underbuild-followup
```
