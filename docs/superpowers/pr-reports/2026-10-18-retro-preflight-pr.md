## Summary

- Start implementation of design PR #2751 with an offline P0 libretro probe.
- Fingerprint operator core/ROM inputs before native loading; capture callback and timing evidence.
- Compare uninterrupted and restored input replay, including retained duplicate-frame video.
- Add compiled C ABI tests, failure-verdict regressions, CI and an operator runbook.
- Full Retro Play implementation is **BLOCKED at P0**, not complete.

## Outcome moved

There is now an executable measurement seam for the first go/no-go gate.
No real NES/SNES compatibility, Athena cadence, shared play or cognitive behavior
is claimed. The C fixture is authored test code, not an emulator or game.

## Current architecture

No existing retro emulator service, configuration, bus events, tests or evals.
The design's Hub, admission and memory candidates do not yet implement Retro.

## Architecture touched

Offline `orion/retro` package only. No running service or production data touched.

## Files changed

- `orion/retro/__init__.py`: explicit offline package boundary.
- `orion/retro/probe.py`: libretro ABI callbacks, fingerprints and in-memory state.
- `orion/retro/preflight.py`: pacing, replay, report and CLI gates.
- `tests/retro/`: failure regression cases and compiled native fixture.
- `.github/workflows/retro-preflight.yml`: Python 3.11 native boundary CI.
- `docs/runbooks/retro-preflight.md`: execution, quality gate, owners and blockers.

## Schema / bus / API changes

- Added: local JSON `retro.preflight.v1` report; CLI `python -m orion.retro.preflight`.
- Removed: none.
- Renamed: none.
- Behavior changed: none in running Orion.
- Compatibility notes: no bus payload/schema registration; report is offline evidence.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable; no templates changed.
- skipped keys requiring operator action: none.

## Tests run

```text
/tmp/orion-retro-venv/bin/python -m pytest tests/retro -q
14 passed
python3 -m orion.retro.preflight --help
passed
git diff --check
passed
```

## Evals run

```text
Actual core/ROM timing and deterministic replay eval: NOT RUN / UNVERIFIED.
Operator ROM paths and validated pinned native core are unavailable.
Synthetic timing and native ABI tests are tests, not gameplay eval results.
The executable real-content eval and exact command are in the runbook.
```

## Docker/build/smoke checks

```text
C fixture compiled with cc -shared -fPIC and executed through ctypes: passed.
No Docker service/build/deploy introduced. No deployment performed.
Probe is NOT a sandbox; isolated native execution remains a P0 prerequisite.
```

## Review findings fixed

- Finding: two equally wrong restores could yield a false replay PASS.
  - Fix: compare uninterrupted continuation with restored continuation.
  - Evidence: consistently-zero restore regression fails the gate.
- Finding: duplicate video after restore could falsely fail a valid replay.
  - Fix: restore the checkpoint's frontend video digest along with native state.
  - Evidence: alternating fresh/duplicate callback regression passes.
- Independent review used the requesting-code-review skill in a read-only subagent.

## Restart required

No restart required.

## Risks / concerns

- Severity: blocker.
- Concern: #2751 is a multi-stage proposal; this delivers only the P0 tool, not P0
  acceptance or P1–P6. Need operator ROMs, validated trusted core/build provenance,
  isolated execution and the live load matrix. No ROMs were downloaded.
- Mitigation: draft PR, explicit unverified report fields, default failure on short
  runs/missing A/V/replay mismatch. No live cognition or control wiring.
- Severity: important.
- Concern: native core executes with caller privileges; unsupported environment
  callbacks may prevent compatibility; PCM counts do not verify audible output.
- Mitigation: runbook requires disposable restricted execution; P1 sandbox and
  P2 audio/browser acceptance remain separate gates.

## PR link

Implementation PR link is returned in the task handoff; source design:
https://github.com/junebug-junie/Orion-Sapienform/pull/2751
