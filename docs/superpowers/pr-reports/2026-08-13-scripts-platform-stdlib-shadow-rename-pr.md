## Summary

- Renames `scripts/platform/` (a real, tracked "platform audits" toolkit — `audit_spine.py`, `audit_antipatterns.py`, etc.) to `scripts/platform_audits/`, because that path silently shadowed Python's stdlib `platform` module whenever `scripts/` landed on `sys.path` — which Python does automatically for any `python3 scripts/<name>.py` invocation, the dominant way scripts in this repo actually get invoked (git hooks, cron jobs, manual runs).
- Adds a deterministic gate (`scripts/check_scripts_dir_no_stdlib_shadow.py` + `make check-scripts-dir-no-stdlib-shadow`) so this exact collision class can never silently recur for any future addition to `scripts/`.
- Adds a blast-radius eval (`scripts/run_scripts_platform_shadow_blast_radius_eval.py`) that (1) really runs the renamed audit package end-to-end and (2) safely probes the ~28 other files that had already independently discovered and locally worked around this exact bug before it was traced to its root cause — proving none of them regressed.
- Leaves those ~28 pre-existing local workarounds untouched (deliberate, disclosed scope decision — see below).

This is the direct follow-up to PR #1598 (structural_mass durability + self-study-enrichment trigger loop), which found and fixed this bug for one file (`scripts/self_study_enrichment_hook.py`) but left the root cause — the colliding directory itself — in place.

## Outcome moved

- The actual root cause of a bug class that had already independently cost ~28 other files their own local workarounds is now gone, repo-wide, for any *future* script.
- A deterministic gate makes a recurrence of this exact class of bug (any future `scripts/<stdlib-name>/` or `scripts/<stdlib-name>.py`) fail fast and legibly instead of surfacing as a confusing `AttributeError` deep in an unrelated import chain.
- `scripts/platform_audits/run_all_audits.sh` — while actually exercising it end-to-end to build the eval — was found to have its own separate, pre-existing portability bug (hardcoded `python`, not guaranteed on PATH) and got fixed as a direct byproduct.

## Current architecture

- `scripts/`: ~150 flat and nested Python scripts and shell wrappers, mostly invoked directly as `python3 scripts/<name>.py` from git hooks, cron, or by hand. No prior structural guard against a script/package name colliding with the Python stdlib.
- `scripts/platform/`: an audit toolkit invoked via `bash scripts/platform/run_all_audits.sh <RUN_ID>`, which internally ran `python -m scripts.platform.audit_*`.

## Architecture touched

- `scripts/platform/` → `scripts/platform_audits/` (git-tracked rename, all internal imports and the `Usage:` docstrings in each `audit_*.py` updated, `run_all_audits.sh`'s `python -m scripts.platform.*` calls updated).
- New: `scripts/check_scripts_dir_no_stdlib_shadow.py` (deterministic gate), `scripts/run_scripts_platform_shadow_blast_radius_eval.py` (eval), `Makefile`'s `check-scripts-dir-no-stdlib-shadow` target.
- `scripts/self_study_enrichment_hook.py`'s comments updated to note the rename (its functional fix already shipped in PR #1598; no behavior change here).

## Files changed

- `scripts/platform/*` → `scripts/platform_audits/*` (renamed): `__init__.py`, `_common.py`, `audit_antipatterns.py`, `audit_channels.py`, `audit_config_lineage.py`, `audit_schemas.py`, `audit_spine.py` (import path updated in each), `run_all_audits.sh` (module paths + `python`→`python3` portability fix), `README.md` (new "Renamed from scripts/platform/" section documenting the incident and pointing at the gate/eval).
- `scripts/check_scripts_dir_no_stdlib_shadow.py`: new deterministic gate — scans `scripts/`'s own top-level entries (files and directories) against `sys.stdlib_module_names`, fails if any collide.
- `scripts/test_check_scripts_dir_no_stdlib_shadow.py`: 8 new tests for the gate (clean dir, colliding `.py` file, colliding directory — the exact real bug's shape, dotfile/dunder exclusion, dedup, the real repo's own `scripts/` post-rename, subprocess exit-code behavior).
- `scripts/run_scripts_platform_shadow_blast_radius_eval.py`: new eval — real end-to-end smoke of the renamed audit package, plus a safe regression sweep of the 28 known-previously-affected files.
- `Makefile`: `check-scripts-dir-no-stdlib-shadow` target + `.PHONY` entry.
- `scripts/self_study_enrichment_hook.py`: comment updates only (rename trail).

## Deliberately NOT touched (disclosed scope decision)

`grep -rl "shadows stdlib\|sys.path.pop(0)" scripts/ orion/` finds **28 files** that already carried their own local `sys.path`-based workaround for this exact collision, independently discovered before it was traced to its root cause here. Two distinct shapes exist (`if sys.path and sys.path[0] == _SCRIPT_DIR: sys.path.pop(0)` in most; a more elaborate `sys.modules`-pop + temporary-`sys.path`-filter dance scoped to `uuid` specifically in `scripts/agent_board_lib.py`). All 28 are now harmless no-ops — they still run, they just never have anything left to guard against. Removing them is a real, separate, much larger cleanup (28 files vs. this patch's rename) and was deliberately left out of scope here rather than silently folded in. The blast-radius eval's `KNOWN_AFFECTED_FILES` list is exactly this same 28-file set, hardcoded so a future removal of any of these workarounds is a deliberate edit to the eval, not a silent coverage loss.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest scripts/test_check_scripts_dir_no_stdlib_shadow.py -q
8 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest scripts/ -q \
  --ignore=scripts/test_recall_harness.py --ignore=scripts/test_session_stop_agent_board.py \
  --ignore=scripts/test_concept_induction_publish.py --ignore=scripts/vector_smoke_test.py
374 passed
```

(The 4 ignored files are pre-existing, unrelated to this patch — 3 need live infra this sandbox doesn't have, 1 needs a `chromadb` package not installed in this venv.)

## Evals run

```text
$ python3 scripts/run_scripts_platform_shadow_blast_radius_eval.py
=== Blast-radius eval: scripts/platform -> scripts/platform_audits rename ===

[1/2] Renamed package end-to-end smoke (run_all_audits.sh)...
  OK

[2/2] Known-affected files (28) regression sweep...

  28/28 clean, 0 regression(s), 0 timeout(s), 0 stale entries

=== PASSED ===
```

The renamed audit package produces real, substantial (12KB–58KB) JSON output artifacts when actually run — not empty-shell output. All 28 previously-affected files' module-level code (imports, and their own now-redundant local guards) executes cleanly with the exact real `sys.path[0]` shadow condition manually reproduced in the probe (confirmed empirically that `runpy.run_path()` alone does NOT replicate this — had to insert the target's directory manually), while each file's real side-effecting `main()` logic is never triggered (`run_name != "__main__"`) — deliberately avoiding running ~28 arbitrary scripts' real side effects (several are named `backfill_*`/`smoke_*`).

Confirmed the gate's detection logic is real, not tautological: `probe_file()` against a synthetic shadow scenario (a temp dir with its own fake `platform.py` + a file importing `uuid`) correctly returns `shadow_regression`.

## Docker/build/smoke checks

Not applicable — this is host-side tooling (git hooks, cron scripts, dev utilities), not container runtime code. No rebuild/redeploy needed.

## Review findings fixed

- Finding: the eval's two `subprocess.run(..., timeout=N)` calls had no `try/except subprocess.TimeoutExpired` — an uncaught timeout would crash the whole eval with a raw traceback, directly contradicting its own documented "any other failure is reported as informational, not a hard failure" contract. Plausible in practice: several `KNOWN_AFFECTED_FILES` entries attempt live Redis/Postgres connections at import time.
  - Fix: both call sites now catch `TimeoutExpired` and report a new `'timeout'` verdict (informational, not a hard failure — distinct from `shadow_regression`).
  - Evidence: re-ran the eval post-fix, still 28/28 clean with the new "0 timeout(s)" line in output.
- Finding: the new `make check-scripts-dir-no-stdlib-shadow` target used bare `python`, which this exact patch already demonstrated isn't guaranteed on PATH (found live building `run_all_audits.sh`'s own fix).
  - Fix: changed to `python3` for this new target specifically (the other 9 pre-existing `check-*` targets all use bare `python` too — that's a separate, repo-wide, out-of-scope cleanup, not touched here).
  - Evidence: `make check-scripts-dir-no-stdlib-shadow` now runs clean in this sandbox.
- Finding: new target missing from `.PHONY`, unlike all 9 sibling `check-*` targets.
  - Fix: added.
- Finding: `find_stdlib_shadows()`'s scope was slightly broader than its own docstring claimed — any non-`.py` entry (not just directories) got flagged by bare name, which would have been a false positive for a hypothetical extensionless non-directory file.
  - Fix: tightened to only the two shapes Python's import system actually resolves (`.py` file or directory).
  - Evidence: re-ran all 8 gate tests, still passing.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: 28 files' worth of now-redundant local `sys.path` workarounds remain in the codebase.
- Mitigation: harmless (verified via the blast-radius eval, all 28 clean); documented as an explicit, disclosed follow-up rather than silently left unmentioned. A future patch can remove them file-by-file once someone chooses to.

- Severity: low
- Concern: the eval's `_SHADOW_SIGNATURES` string-matches specific `AttributeError` text, which could theoretically stop matching if a future CPython version changes that exact wording.
- Mitigation: acknowledged, inherent limitation of stderr string-matching for a periodic eval (not a pytest gate blocking merges); the deterministic `check_scripts_dir_no_stdlib_shadow.py` gate is the primary, structural defense against a *recurrence* of this bug class — the eval is a one-time-per-run confirmation, not the main safety net.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1601
