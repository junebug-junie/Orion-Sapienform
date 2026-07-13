# PR report: `make check-activation-saturation` target

Implements item 3 of `docs/superpowers/specs/2026-07-13-recall-followups-loop-retirement-saturation-gate-spec.md`.

## Summary

- `scripts/check_activation_saturation.py` (the reinforcement/decay wiring spec's acceptance-check-1 standing gate) existed but was on-demand-only, no recurring cadence or discoverable entry point.
- Added `make check-activation-saturation`, mirroring the exact standalone-target pattern already used by `check-inner-state-registry`/`check-single-consumer-channels` (both documented in the Makefile's own comment as "the real, working pieces" of a promised-but-never-built `agent-check` chain).
- `FAIL_ABOVE=<fraction>` passes through to the script's `--fail-above`, for the "compare against a prior run" regression check the spec's acceptance check calls for.

## Outcome moved

The saturation gate goes from "a script that exists if you know to look for it" to a discoverable `make` target matching this repo's existing convention.

## Current architecture (before this patch)

Two standalone check-* Makefile targets already existed as the real (non-fictional) piece of CLAUDE.md's described `agent-check` chain. The saturation script had no Makefile entry at all.

## Architecture touched

`Makefile` only.

## Files changed

- `Makefile`: `.PHONY` list + new `check-activation-saturation` target.

## Schema / bus / API changes

None.

## Env/config changes

None — reuses `POSTGRES_URI`, already documented in `services/orion-hub/.env`.

## Tests run

Not applicable (Makefile target, no test suite). Functional verification below is the real check.

## Evals run

Not applicable.

## Docker/build/smoke checks

```text
$ POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney make check-activation-saturation
activation_saturation: 55/102 active crystallizations at or above activation=0.99 (53.9% ceiling-pinned)
Compare this fraction against a prior run from before real usage of recall_boost()+decay() --
an INCREASE is a fail, revert the patch. (Pass --fail-above <prior-fraction> to make this
check exit 1 automatically.)
$ echo "exit=$?"
exit=0

$ POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney make check-activation-saturation FAIL_ABOVE=0.1
check_activation_saturation FAILED: 53.9% exceeds --fail-above 10.0%
make: *** [Makefile:46: check-activation-saturation] Error 1
```

Both against real live Postgres data, re-verified after rebasing onto current `origin/main` (which moved twice during this task — PR #1010 then #1011 from concurrent sessions).

## Review findings fixed

None — two-line addition, low risk, matches an established pattern exactly.

## Restart required

```text
No restart required.
```
Dev-time tooling only.

## Risks / concerns

- Severity: Low — the target uses bare `python` (not `python3`), matching the two existing sibling targets' convention exactly. This requires a venv with `python` on `PATH` (confirmed working via `source venv/bin/activate` in this repo) — a pre-existing repo-wide convention, not something this patch introduces or should silently diverge from.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...chore/check-activation-saturation-make-target?expand=1`
