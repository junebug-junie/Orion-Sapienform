# PR report: preserve intentional local .env overrides in sync script

Follow-up from the graphiti_core backend activation work (PR #993, #995). Flagged as a High-severity risk in both PR reports: running `python scripts/sync_local_env_from_example.py` mid-session silently reverted `services/orion-graphiti-adapter/.env`'s live-activated `GRAPHITI_BACKEND=graphiti_core` back to the checked-in `.env_example` default `orion_postgres`, along with disabling FalkorDB and clearing the embed URL. Caught and manually corrected in that session; this PR fixes the tool.

## Summary

- `scripts/sync_local_env_from_example.py`'s `sync_file()` treated every local `.env` value that differs from `.env_example` as drift to silently correct — no distinction between genuine drift and an intentional deployment-specific override.
- Default behavior changed: missing keys still auto-add (nothing to clobber, safe). Keys that exist locally with a differing value are now left untouched and reported under a new "Diverged" section instead of being silently rewritten.
- New `--force` flag restores the old overwrite-everything behavior, for when a key's meaning genuinely changed upstream and every host's local value should follow.
- `sync_file()` now returns a `SyncResult(updated, diverged)` dataclass instead of a flat list.
- `NEVER_SYNC_KEYS` (the existing, stricter "exclude entirely" mechanism for e.g. `ORION_BUS_URL`) is unchanged — this is a separate, softer default that doesn't replace it.

## Outcome moved

The exact incident from this session (silent revert of a live backend flag) can't recur without an explicit `--force`. The script goes from "always trust the example over reality" to "trust reality unless told otherwise."

## Current architecture (before this patch)

`sync_file()` compared each `.env_example` key (filtered by `SYNC_PREFIXES`/`SYNC_EXACT`/`NEVER_SYNC_KEYS`) against local `.env`, and for any existing key with a different value, silently rewrote it to the example's value — no reporting, no opt-out short of adding the key to `NEVER_SYNC_KEYS` (a permanent, per-key exclusion, too blunt for keys that sometimes need to sync and sometimes don't).

## Architecture touched

Single script, `scripts/sync_local_env_from_example.py`, and its test coverage. No service, schema, bus, or Docker surface — this is a dev-time tool, not a running service.

## Files changed

- `scripts/sync_local_env_from_example.py`: `SyncResult` dataclass, `sync_file(..., force: bool = False)`, `--force` CLI flag, `main()` updated to print both `Updated:`/`Would update:` and `Diverged:` sections, module docstring documents the new default behavior
- `tests/scripts/test_sync_local_env_from_example.py`: extended with diverged/force/missing-key coverage; one pre-existing test updated for the new default (no-longer-silent-overwrite) semantics

## Schema / bus / API changes

None — dev tooling only.

## Env/config changes

None — this PR changes how env *syncing* behaves, not any env key itself.

## Tests run

```text
$ source venv/bin/activate && python -m pytest tests/scripts/test_sync_local_env_from_example.py -v
6 passed in 0.09s
```
Independently re-run by the orchestrator, not taken from the implementing agent's report alone.

Manual scratch-directory smoke (not run against real `services/*/.env` files, to avoid colliding with the parallel graphiti-hardening task): built a scratch `.env`/`.env_example` pair reproducing the exact incident (`GRAPHITI_BACKEND=graphiti_core` locally vs `orion_postgres` in the example) — confirmed dry-run and real run both report it under "Diverged" and leave the file unchanged; confirmed `--force` on the same scratch data does overwrite it. Full output in the implementing agent's report.

## Evals run

No eval harness applies to a standalone dev script.

## Docker/build/smoke checks

Not applicable — no Docker/service surface. Confirmed via `grep` that the only other call sites (`services/orion-signals/scripts/up.sh`/`down.sh` — informational text only, don't invoke the script; `scripts/repl/orion_fresh_main_smoke.sh` — invokes it via `run_step`, gated on exit code only) are unaffected: exit code remains `0` for a diverged-only report, matching prior behavior for a "changes needed" run.

## Review findings fixed

Implementing agent ran a self-review given the diff's small, purely additive scope: traced all `sync_file()` call sites (only `main()` and the test file), confirmed no other script imports the function directly, verified no regression to the one existing external consumer. No material findings. Orchestrator independently re-read the full diff and re-ran tests rather than trusting the report alone.

## Restart required

```text
No restart required.
```
Dev-time script, not a running service.

## Risks / concerns

- Severity: Low — `--force` still exists and can reproduce the old silent-overwrite behavior if someone reaches for it out of habit. Documented clearly in the module docstring and the flag's `--help` text; no further mitigation attempted (the flag needs to exist for the legitimate "key's meaning changed upstream" case).

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with prior PRs this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/env-sync-preserve-local-overrides?expand=1`
