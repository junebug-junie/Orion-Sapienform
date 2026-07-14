# PR report: drive_state_divergence_audit.py — warn on silent dev-default store path fallback

## Summary

- Investigated a real-looking alarm: `scripts/drive_state_divergence_audit.py` reported `drive_state.v1` 24h+ stale, all-zero pressures across every drive.
- Root cause: **false alarm, not a production issue.** The script silently fell back to `DEFAULT_CONCEPT_STORE_PATH` (a host-local dev-scratch path, `/tmp/concept-induction-state.json`) when neither `--store-path` nor `$CONCEPT_STORE_PATH` was set. An unrelated stale file happened to be sitting at that exact path from an earlier local run and was misread as the live signal.
- Live container (`orion-athena-spark-concept-induction`) was never stale — confirmed via live logs (`drive_engine_pressure_probe` firing every ~2-4s with real, changing, non-zero pressures) and the real bind-mounted store file (`/mnt/graphdb/orion/concepts/concept-induction-state.json`, 35MB, updated seconds before this check, vs. the 3.2KB decoy file the script had actually read).
- Fixed the script to track *why* `store_path` has its value and print a loud, actionable warning whenever it silently lands on the dev fallback — this exact confusion should not be possible to hit silently again.

## Outcome moved

The divergence audit tool from yesterday's round (`chore/drive-state-divergence-audit`, merged) had a real footgun: its own default behavior could produce a false "drives are stale/dead" alarm with no indication that the path itself was the problem, not the signal. Closed that gap.

## Current architecture (before this patch)

`scripts/drive_state_divergence_audit.py --store-path` defaulted to `os.getenv("CONCEPT_STORE_PATH", DEFAULT_CONCEPT_STORE_PATH)` with no distinction in the output between "read a real, intentionally-configured path" and "silently landed on the dev fallback because nothing else was set." The module docstring's own usage example set `CONCEPT_STORE_PATH=/tmp/concept-induction-state.json` as if it were a legitimate live path, reinforcing the trap.

## Architecture touched

`scripts/` only (the audit script itself and its tests) — no runtime/production code touched, since the live service was confirmed healthy throughout.

## Files changed

- `scripts/drive_state_divergence_audit.py`: `--store-path` now tracks provenance (`cli:--store-path` / `env:CONCEPT_STORE_PATH` / `default_fallback`). When it lands on `default_fallback`, prints a `WARNING:` block (prose mode) or sets `store_path_source`/`store_path_warning` fields (`--json` mode) explaining the fallback is almost certainly not the live container's real store, with the exact `docker inspect` command to find the real one and where to cross-reference the compose file's bind mount. Fixed the misleading usage example in the docstring.
- `tests/test_drive_state_divergence_audit.py`: 6 new tests covering all three provenance paths in both prose and JSON output.

## Schema / bus / API changes

None.

## Env/config changes

None. No new keys — this is argument/output-provenance tracking within the existing script.

## Tests run

```text
$ python -m pytest tests/test_drive_state_divergence_audit.py -q
21 passed
```
Independently re-run by the orchestrator (21/21, matching the implementing agent's count of 15 pre-existing + 6 new).

## Evals run

Not applicable — deterministic diagnostic tooling fix.

## Docker/build/smoke checks

Live-verified by both the implementing agent and independently by the orchestrator:

```text
$ docker exec orion-athena-spark-concept-induction env | grep CONCEPT_STORE_PATH
CONCEPT_STORE_PATH=/data/concept-induction-state.json

$ ls -la /mnt/graphdb/orion/concepts/concept-induction-state.json
-rw-r--r-- 1 root root 35774321 Jul 14 00:04 ...   # 35MB, seconds old

$ docker logs orion-athena-spark-concept-induction --since 30s | grep drive_engine_pressure_probe
drive_engine_pressure_probe subject=orion pressures={'coherence': 0.7748, ...}  # firing every few seconds, real changing values

# Script re-run with no CONCEPT_STORE_PATH set: reproduces the original false alarm
# exactly, now WITH the warning block explaining why.
# Script re-run with the real path: no warning, updated_at seconds old, matches live logs.
```

## Review findings fixed

N/A — this patch is itself the fix for a finding (a false-alarm-producing footgun in yesterday's merged script), not a patch that went through a separate review pass. Orchestrator independently re-verified both the diff scope and the live claims before pushing rather than trusting the investigating agent's self-report alone.

## Restart required

```text
No restart required.
```
No runtime service touched — the live container was confirmed healthy and untouched throughout this investigation.

## Risks / concerns

None identified. Diagnostic-tooling-only fix, fully additive (new warning, new provenance tracking), no change to existing default behavior for callers who already set `--store-path`/`$CONCEPT_STORE_PATH` correctly.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...investigate/drive-state-staleness?expand=1`
