# PR report: AI Town Convex engine-recovery fix + data compaction

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1452
Branch: `fix/aitown-convex-compaction`

## Summary

Juniper reported AI Town was broken (couldn't move/interact) and separately
that Orion's in-town dialogue felt transactional, not curious. The dialogue
question was answered inline (root cause: `orion/embodiment/speech.py`'s
`build_speech_prompt()` injects zero personality/curiosity framing and runs
on the fastest/shallowest `chat_quick` lane — a separate, already-understood
tradeoff, not touched by this PR). Investigating the movement/interact
complaint led to two real, unrelated bugs:

1. The engine-recovery diagnostic/recovery tooling from a prior incident
   (`patches/orion-engine-recovery.patch`) had a query bug that made it
   unusable at scale.
2. The actual root cause of the reported lag: self-hosted Convex retains
   unbounded document-revision history with no compaction, and this
   deployment had never been restarted/compacted in ~3 weeks.

## Outcome moved

Live-verified on the running `orion-ai-town` deployment:

| metric | before | after |
|---|---|---|
| `db.sqlite3` size | 23.56GB | 240MB (99% reduction) |
| backend CPU (idle) | 123% | 0.00% |
| backend RSS | 52.87GB | ~1.46GB |
| OCC error rate | ~20-30/min | 0/min |

All 216,460 documents round-tripped intact across every table (verified via
`npx convex data` spot-checks and the import change-summary output).

## Current architecture

`services/orion-ai-town` wraps a16z's self-hosted `ai-town` — Convex backend
(SQLite-backed, single container) + frontend + dashboard. `upstream/` is a
gitignored local clone pinned to a SHA; repo-tracked behavior changes live as
`.patch` files in `patches/`, applied by `scripts/apply_upstream_patches.sh`.
No prior maintenance/compaction tooling existed. The backend container had
been running continuously since world creation (~2026-07-06/07) with no
restarts, and no one had previously needed to invoke the engine-recovery
tooling from `orion-engine-recovery.patch` at any real scale.

## Investigation path (for future reference)

1. `docker compose logs backend` showed 84 unique errors — OCC (optimistic
   concurrency) conflicts on `inputs`/`engines`, plus the built-in "restart
   dead worlds" cron itself erroring repeatedly.
2. Ran the existing `testing:debugEngineState` diagnostic (from the prior
   engine-recovery patch) to confirm a frozen-engine theory — it **timed
   out** ("too many system operations"), which was itself the first real
   finding: the query used `.withIndex(eq engineId only).filter(gt number)`,
   which can't prune the index range. It was scanning the engine's entire
   input history (300k+ rows) instead of just the pending tail, regardless
   of `.take()` limit. Fixed to chain `.gt('number', processed)` directly
   into `withIndex`, matching the engine's own `loadInputs` in
   `convex/engine/abstractGame.ts` (which already did this correctly).
3. With the query fixed, live data showed the engine was **not** actually
   frozen — pending backlog was 0-1, `processedInputNumber` advancing
   normally. The frozen-engine theory (pattern-matched from the *prior*
   incident this patch's comments describe) did not hold up under direct
   verification, and was dropped rather than reported as fixed.
4. User reported continued lag ("characters appearing to move but not,
   chat timing out"). `docker stats` showed the real problem: backend
   container at 123% CPU / 52.87GB RSS, backing a 23.5GB SQLite file.
5. Sampled row counts per table (via a new bounded `tableGrowthSample`
   query, using the auto `by_creation_time` index rather than a full
   `.collect()`) — totaled only ~100-200MB of logical data. Checked the
   singular `world` document directly (new `worldDocSizeProbe` query) —
   4.4KB, no live unbounded-array bug. Conclusion: the 23GB was Convex's own
   retained document-revision history, not deletable app data.
6. Confirmed live: `VACUUM` on the stopped file only recovered ~5%
   (23.56GB → 22.24GB after 14 minutes) — consistent with history being
   *live* data from SQLite's point of view, not reclaimable free space.
7. The only real fix: export (`npx convex export`, no downtime) → stop →
   reset the SQLite file → restart → reimport (`npx convex import
   --replace-all`). First live run of this sequence (manual, no script yet)
   succeeded at shrinking the file but broke the live app: the reset also
   wipes deployed Convex function code and `npx convex env` variables
   (LLM gateway wiring), since both live in the same file. Recovered live by
   redeploying (`npx convex dev --once`) and re-running
   `scripts/wire_llm_gateway.sh`, then manually heartbeating the world back
   to `running`.
8. Built `scripts/compact_convex_data.sh` to encode the *complete* correct
   sequence (including the two gaps found in step 7) so a future/scheduled
   run doesn't repeat the same recovery-by-hand. Wired a daily,
   threshold-gated (5GiB default) host crontab entry.

## Architecture touched

- `patches/orion-engine-recovery.patch` (upstream `convex/testing.ts` diff)
- New `scripts/compact_convex_data.sh`
- `README.md`, `.env_example`
- Host crontab (not part of the git diff)

## Files changed

- `services/orion-ai-town/patches/orion-engine-recovery.patch`: index-range
  fix for `debugEngineState`/`recoverFrozenEngine`; added `tableGrowthSample`
  and `worldDocSizeProbe` read-only sizing probes.
- `services/orion-ai-town/scripts/compact_convex_data.sh`: new — export →
  stop → snapshot+rename → restart → redeploy functions → restore env vars →
  reimport → heartbeat world. Threshold-gated, `--check`/`--force` flags.
- `services/orion-ai-town/tests/test_engine_recovery_patch.py`,
  `test_compact_convex_data_script.py`: new structural gate tests.
- `services/orion-ai-town/README.md`: new "Maintenance: Convex data
  compaction" section.
- `services/orion-ai-town/.env_example`: documented (commented, optional)
  `AITOWN_COMPACT_THRESHOLD_BYTES` / `AITOWN_COMPACT_HEALTH_TIMEOUT_SEC`.

## Schema / bus / API changes

- Added: two internal-only Convex diagnostic queries (`tableGrowthSample`,
  `worldDocSizeProbe`) — not part of any bus/schema contract.
- Removed: none.
- Renamed: none.
- Behavior changed: `debugEngineState`/`recoverFrozenEngine` now take an
  optional `limit` arg and use a real indexed range scan instead of timing
  out.
- Compatibility notes: none needed; these are internal maintenance
  functions, not consumed by the frontend or any other service.

## Env/config changes

- Added keys (optional, commented in `.env_example`):
  `AITOWN_COMPACT_THRESHOLD_BYTES` (default 5GiB, baked into the script),
  `AITOWN_COMPACT_HEALTH_TIMEOUT_SEC` (default 180s, baked into the script).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  yes — no changes applied (new keys are commented/optional, script's
  existing divergence report for unrelated services is pre-existing).
- skipped keys requiring operator action: none.

## Tests run

```
/mnt/scripts/Orion-Sapienform/venv/bin/pytest services/orion-ai-town/tests/ -v
29 passed, 1 failed (test_juniper_blurb_present_in_world_ts — requires the
gitignored upstream/ clone, which doesn't exist in this fresh worktree;
pre-existing/environmental, not a regression from this change).
```

## Evals run

No eval harness exists for this service (infra/ops tooling, not a
cognition-loop change). Live verification against the running deployment
(documented above and in "Docker/build/smoke checks") served as the
functional check.

## Docker/build/smoke checks

Full live run performed against the running deployment during this session
(not a staging/test environment — this repo has no separate staging AI Town
instance):

```
npx convex export --path <job>/export.zip           # no downtime
docker compose stop backend
docker cp <container>:/convex/data/db.sqlite3 <job>/db.sqlite3.pre-compact.bak
docker run --rm -v orion-ai-town_convex-data:/data alpine \
  mv /data/db.sqlite3 /data/db.sqlite3.pre-compact-<ts>
docker compose start backend                         # health check passes
npx convex dev --once                                # function redeploy
npx convex env set --from-file <job>/env.backup       # (manual restore on
                                                        # first run; scripted
                                                        # for future runs)
npx convex import --replace-all -y <job>/export.zip   # 216,460 docs restored
npx convex run world:heartbeatWorld '{"worldId": "..."}'
```

Verified after: `docker compose ps` (healthy), `docker stats` (0.00% CPU,
~1.46GB RSS), `npx convex data worldStatus`/`playerDescriptions` (data
intact), `testing:debugEngineState` (engine ticking,
`processedInputNumber` advancing, 0-1 pending).

Patch chain re-verified from scratch: cloned the pinned upstream SHA fresh
into `/tmp`, applied all 6 tracked patches in `apply_upstream_patches.sh`'s
order (including the regenerated `orion-engine-recovery.patch`) — all apply
cleanly with `git apply --check`.

## Review findings fixed

- Finding: `compact_convex_data.sh`'s rename-old-db step silently swallowed
  failure (`mv ... || true`), so a failed rename would let the script
  proceed as if compaction happened while the backend restarts on the same
  bloated file.
  - Fix: rename failure now aborts the script loudly with explicit
    recovery instructions pointing at the export/backup.
  - Evidence: `scripts/compact_convex_data.sh` step 4/7.
- Finding: hardcoded `VOLUME_NAME="orion-ai-town_convex-data"` assumed
  Compose's default project-name-prefix convention; a `COMPOSE_PROJECT_NAME`
  override would silently target a nonexistent (auto-created, empty) volume
  while the real data volume went untouched.
  - Fix: volume name resolved dynamically from the running backend
    container's actual mount via `docker inspect`.
  - Evidence: `scripts/compact_convex_data.sh`, volume-resolution block.
- Finding: new optional env keys weren't documented in `.env_example`.
  - Fix: added as commented/optional entries with a pointer to the README
    section.
  - Evidence: `services/orion-ai-town/.env_example`.

## Restart required

```
No restart required — the live deployment was already recovered and
verified during this session (backend healthy, functions deployed, env
restored, world running). Merging this branch only updates the tracked
patch/script/docs to match what's already live.
```

## Risks / concerns

- Severity: low
- Concern: the pre-compaction `npx convex env` values were not captured
  before the *first* live compaction run (that capture step was only added
  to the script after discovering the gap mid-session). They were restored
  via `scripts/wire_llm_gateway.sh`'s own default (`LLM_MODEL=quick`), which
  does not match the README's documented default (`LLM_MODEL=chat`). I
  could not confirm which value was actually in use before the compaction.
- Mitigation: worth an operator check of `npx convex env list` against
  README expectations. Low blast radius either way — both are valid gateway
  routes; this only affects which llamacpp lane NPC chat completion uses.
  All future runs of the fixed script capture and restore the exact prior
  values, so this is a one-time gap, not a recurring risk.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1452
