# PR report: fix Jena cache path self-move collision in Fuseki compact script

## Summary

- Production failure (2026-07-22): `fuseki_tdb_compact.sh` (the scheduled TDB2 compaction job) failed with `mv: cannot move '/tmp/apache-jena-5.1.0' to a subdirectory of itself, '/tmp/apache-jena-5.1.0/apache-jena-5.1.0'`.
- Root cause: `JENA_CACHE`'s default (`/tmp/apache-jena-${JENA_VERSION}`) was the exact same path the downloaded Jena tarball's top-level directory extracts to. Whenever the cache was empty (e.g. after a host reboot cleared `/tmp` -- confirmed via the script's own run log showing several prior successful runs that took the Docker fallback path instead, meaning `_ensure_jena` had been a no-op on those days) and a fresh download+extract actually had to run, the final `mv` tried to move that directory onto itself.
- Fixed by nesting `JENA_CACHE`'s default under a distinct parent directory so it can never collide with the (fixed, non-configurable) tar extraction target.
- Review caught a real follow-on gap: the old buggy code's `rm -rf "${JENA_CACHE}"` had been accidentally also clearing the tar extraction scratch directory (same path). Fixed in the same commit with an explicit second `rm -rf`.

## Outcome moved

The weekly Fuseki TDB compaction job no longer fails when its Jena cache needs a fresh download+extract (e.g. after `/tmp` is cleared by a reboot). This was flagged as part of the same-day Fuseki-decommission audit -- the compaction job matters regardless of whether Fuseki itself eventually goes away, since it stays load-bearing maintenance for however long Fuseki remains up.

## Current architecture

Before this patch: `JENA_CACHE="${JENA_CACHE:-/tmp/apache-jena-${JENA_VERSION}}"`, identical to the tarball's own extraction target path, causing a guaranteed `mv` failure on any cache-cold run.

## Architecture touched

- `services/orion-rdf-store/scripts/fuseki_tdb_compact.sh`, `scripts/test_ensure_jena_cache_path.sh` (new)

## Files changed

- `services/orion-rdf-store/scripts/fuseki_tdb_compact.sh`: `JENA_CACHE`'s default changed to `/tmp/orion-jena-cache/apache-jena-${JENA_VERSION}` (nested, distinct from the extraction target), with a comment citing the exact production error. `_ensure_jena()` gained an explicit `rm -rf "/tmp/apache-jena-${JENA_VERSION}"` (the extraction scratch path) immediately before `tar -xzf`, restoring the cleanup the old code got by accident when the two paths were the same.
- `services/orion-rdf-store/scripts/test_ensure_jena_cache_path.sh` (new): a standalone bash test, same style as the existing sibling `test_compact_lock_coordination.sh` (no Docker/Fuseki, no real Jena download). Builds a small fake tarball with the same internal directory-name shape as the real release archive. Four sub-tests: (1) reproduces the exact old-style collision to prove the test bites on the real bug, not a vacuous check; (2) proves the fixed path shape never collides; (3) greps the real script to confirm its shipped default matches what was proven safe in isolation; (4) review-driven -- proves stale content in the extraction scratch directory doesn't survive a fresh extraction into the new cache.

## Schema / bus / API changes

None.

## Env/config changes

None (`JENA_CACHE` remains operator-overridable via env var, default changed; no new/removed keys).

## Tests run

```text
bash services/orion-rdf-store/scripts/test_ensure_jena_cache_path.sh
-> PASS x4, "All Jena cache path collision tests passed."

bash services/orion-rdf-store/scripts/test_compact_lock_coordination.sh
-> PASS x2 (sibling test, confirmed unaffected by this change)

bash -n services/orion-rdf-store/scripts/fuseki_tdb_compact.sh
-> syntax OK
```

## Evals run

No eval harness applies to a maintenance shell script; the new regression test plus the reused sibling test cover the changed behavior.

## Docker/build/smoke checks

Not applicable -- this fix doesn't change container config or runtime behavior of any service, only a host-side maintenance script. Verified the actual host's leftover `/tmp/apache-jena-5.1.0` from the real incident is a complete, valid, non-corrupted Jena extraction (not a broken self-nested mess) -- harmless either way, since the fix means the next compact run clears and re-extracts it fresh regardless.

## Review findings fixed

- Finding (Medium): the fix removed an accidental protection -- the old code's `rm -rf "${JENA_CACHE}"` happened to also clear the tar extraction scratch directory since they were the same path pre-fix. Post-fix, without an explicit second cleanup, stale content there (including the actual incident's own leftover state) could silently survive a future `tar -xzf` (tar doesn't delete pre-existing entries absent from the archive) and get carried into the new cache by `mv`.
  - Fix: added an explicit `rm -rf "/tmp/apache-jena-${JENA_VERSION}"` immediately before the `tar -xzf` line.
  - Evidence: new `test_ensure_jena_cache_path.sh` Test 4, which seeds stale content in the scratch directory and asserts it does not survive into the final cache.
- Informational (judgment call, not fixed): the fix only changes `JENA_CACHE`'s *default* -- an operator-supplied override that still collides with the tar extraction target would reintroduce the same bug. Reviewer's judgment (concurred): reasonable to leave as operator responsibility for a maintenance script that already trusts several other env-derived paths (`SOURCE`, `COMPACT_LOCK`, `FUSEKI_DATA_DIR`) without validation -- a much narrower, deliberate-misconfiguration failure mode than the one that actually shipped.
- Informational (no fix needed, verified true): grepped the whole repo for any other hardcoded reference to the old `JENA_CACHE` default -- none found. `fuseki_recover.sh` (the sibling script) doesn't reference `JENA_CACHE` at all.
- Informational (no fix needed, verified live): the actual host's leftover `/tmp/apache-jena-5.1.0` from the 2026-07-22 incident is a complete, valid extraction, not corrupted -- no manual cleanup required; the fix's own scratch-directory `rm -rf` will clear it on the next real compact run regardless.

## Restart required

No restart required -- this is a maintenance script invoked by cron/manually, not a running service. No action needed until the next scheduled compact run, which will now use the fixed path automatically.

## Risks / concerns

- Severity: Low.
- Concern: none blocking. The operator-override edge case (informational finding above) is a real but narrow, deliberate-misconfiguration-only gap, not fixed in this PR by design.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1307
