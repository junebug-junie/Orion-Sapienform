# fix(context-exec): make beta gate signal trustworthy

## Summary

Repairs the context-exec beta gate by eliminating cross-test module/settings pollution, clarifying RLM engine runtime-debug semantics, and tightening deterministic fake-engine eval outputs. No proposal types, mutation paths, or approval gates were changed.

## Root causes

1. **Denver vertical slice fixture** (`tests/fixtures/denver_vertical_slice.py`) wiped all `app.*` modules from `sys.modules` and mutated global settings without restore. Downstream tests kept stale `repo_tools`, `FAKE_ORGANS`, and `ContextExecRunner` imports — repo grep returned empty hits and RLM evals failed in full-suite order.
2. **`test_health.py` partial reloads** recreated `app.settings` while leaving `app.runner` / `app.alexzhang_rlm_engine` bound to old settings singletons — monkeypatches and eval harness overrides silently missed the live settings object.
3. **`repo_tools` module-level settings import** cached a pre-reload settings reference, so monkeypatched repo roots in tests were ignored after reload pollution.
4. **Stale runtime-debug contract** conflated requested vs effective engine (`engine` vs `engine_selected`), hiding AlexZhang fallback in eval assertions.
5. **Fake engine belief_provenance** treated any prompt mentioning "denver" as the Denver vertical slice, starving Ogden-seeded fixtures of findings.

## Changes

| Area | Fix |
|------|-----|
| Denver fixture | Settings save/restore; drop `sys.modules` purge; sync `app.runner.settings` |
| `repo_tools.py` | Lazy `settings` reads inside `_repo_root` / `repo_read` |
| `runner.py` | `engine_requested`, `engine_selected`, `fallback_used` in `runtime_debug` |
| `rlm_eval_harness.py` | Contract-aware assertions; fresh runner/settings imports per eval |
| `rlm_engine.py` | Ogden-aware belief provenance; trace autopsy root_cause echo guard |
| Tests | `conftest` autouse settings sync; health reload scoped; ledger intake patches `app.settings.settings` |
| Beta gate script | Split pytest: core suite vs RLM eval fixtures (actionable sections) |

## Test results

### Before (main, full beta gate pytest)

```
15 failed, 120 passed, 1 xfailed
```

Representative failures: repo grep empty hits, alexzhang `runtime_debug.engine` mismatch, fake ogden/trace autopsy quality.

### After (this branch)

| Command | Result |
|---------|--------|
| `pytest services/orion-context-exec/tests/` | **135 passed, 1 xfailed** |
| `bash scripts/context_exec_beta_gate.sh` | **BETA GATE PASS** |
| Focused repo + eval tests | **25 passed, 1 xfailed** |
| Proposal spine tests (schemas, CLI, ledger, review API, Hub) | **all green** |

### Intentional xfail

- `test_rlm_eval_repo_impact_engine_files[fake]` — fake engine greps agent-chain patterns, not engine-selection files (unchanged, explicit reason in test).

## Proposal-control spine

All preserved proposal-control pytest targets pass on this branch. Fresh-main smoke ladder (`scripts/repl/orion_fresh_main_smoke.sh`) was **not run from the git worktree** (worktree `.git` is a pointer file; script requires a directory). Beta gate + proposal/Hub test matrix above covers the same rails.

## Remaining risks

- Live golden probes (`--live`) not exercised in this session.
- Other repo fixtures still use `sys.modules.pop` patterns outside context-exec; unrelated but similar class of pollution.

## Review checklist

- [x] No new proposal types
- [x] No real mutation enabled
- [x] Fallback visible in `runtime_debug`
- [x] Repo-tool failures not masked as AlexZhang noise
- [x] Deterministic fake quality failures not broadly skipped
