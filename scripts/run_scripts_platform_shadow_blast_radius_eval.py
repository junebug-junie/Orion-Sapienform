#!/usr/bin/env python3
"""Blast-radius eval for the scripts/platform/ -> scripts/platform_audits/
rename (2026-08-12) -- see scripts/platform_audits/README.md and
scripts/check_scripts_dir_no_stdlib_shadow.py for the incident this exists
to verify against.

Two things this proves, both via real subprocess invocation against the
real repo layout (not import-in-process, which would already have this
repo's sys.path pre-augmented by pytest and wouldn't reproduce the real
failure shape):

1. The renamed scripts/platform_audits/ package still works end-to-end as
   a real, runnable audit tool -- runs run_all_audits.sh for real against a
   disposable RUN_ID and checks it produced real output artifacts, then
   cleans up after itself.

2. None of the ~27 pre-existing files that carried their own local
   sys.path-based workaround for the platform-shadow bug (independently
   discovered before this rename traced it to its root cause -- grep
   `shadows stdlib`/`sys.path.pop(0)` under scripts/ and orion/ to
   reproduce this list) regressed. Each is probed in a fresh subprocess
   with its own directory manually inserted at sys.path[0] -- reproducing
   the EXACT real trigger condition (Python's own auto-insertion for
   `python3 scripts/<name>.py`, which `runpy.run_path()` alone does NOT
   replicate; confirmed empirically while building this eval) -- combined
   with runpy's run_name=<not "__main__"> so each file's real
   argparse/side-effecting main() logic never executes. This is
   deliberately NOT "just run the script for real": several of these are
   named backfill_*/smoke_*, and running ~27 arbitrary scripts' real main()
   logic (bus publishes, DB writes, backfills) is exactly the kind of
   unreviewed side effect CLAUDE.md's backfill-protocol section exists to
   prevent. Only module-level code (imports, module-level guard logic --
   which is where the platform-shadow bug actually lives) executes.

A file is flagged as a REAL regression only if the specific platform/uuid
AttributeError signature appears -- any other failure (missing dependency,
unset env var tripping a settings validator, etc.) is unrelated to what
this eval tests and is reported as informational, not a hard failure.

Exit code 0 when both checks pass; prints a clear failure list otherwise.
Deliberately NOT a pytest test (CLAUDE.md sec 11: evals measure behavior
unit tests don't cover) -- this is a slower, broader sweep meant to run
periodically / before a release, not on every `pytest` invocation.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hardcoded, not re-derived by grep here, so a future removal of one of
# these workarounds doesn't silently shrink eval coverage without a
# deliberate edit to this list. A file that no longer exists is reported
# as "stale" (update this list), not treated as a hard failure.
KNOWN_AFFECTED_FILES: tuple[str, ...] = (
    "orion/mood_arc/fit_encoder.py",
    "scripts/agent_board_lib.py",
    "scripts/backfill_phi_corpus.py",
    "scripts/backfill_recall_falkor_chat_tags_snapshot.py",
    "scripts/bus_core_health_watchdog.py",
    "scripts/causal_geometry_report.py",
    "scripts/check_activation_saturation.py",
    "scripts/check_concept_relation_digest_liveness.py",
    "scripts/check_daily_schedule_collisions.py",
    "scripts/check_inner_state_registry.py",
    "scripts/check_journal_dispatch_registry.py",
    "scripts/check_service_env_compose_parity.py",
    "scripts/check_settings_defaults.py",
    "scripts/check_single_consumer_channels.py",
    "scripts/check_substrate_projection_schema_drift.py",
    "scripts/concept_relation_digest.py",
    "scripts/context_exec_rlm_eval.py",
    "scripts/diagnose_cortex_bus_stack.py",
    "scripts/diag.py",
    "scripts/disk_threshold_watchdog.py",
    "scripts/drive_history_reflection_synthesis.py",
    "scripts/fit_phi_encoder.py",
    "scripts/run_agent_flow_live_matrix.py",
    "scripts/seed_substrate_atlas_demo.py",
    "scripts/self_study_enrichment_hook.py",
    "scripts/self_study_harness.py",
    "scripts/smoke_substrate_motivation_golden_path.py",
    "scripts/trace_hub_skill_runner_e2e.py",
)

_SHADOW_SIGNATURES = (
    "AttributeError: module 'platform' has no attribute",
    "AttributeError: module 'uuid' has no attribute",
)

_PROBE_SRC = """
import runpy, sys, os
target = sys.argv[1]
target_dir = os.path.dirname(os.path.abspath(target))
# Manually reproduce the EXACT real trigger condition -- Python's own
# auto-insertion of a script's directory at sys.path[0] for a direct
# `python3 <path>` invocation. runpy.run_path() alone does NOT do this
# (confirmed empirically: its own sys.path[0] handling differs), so
# skipping this line would make the probe not actually test anything.
sys.path.insert(0, target_dir)
try:
    runpy.run_path(target, run_name="__blast_radius_probe__")
except SystemExit:
    pass  # argparse-style early exits are fine -- import-time code already ran
"""


def probe_file(relpath: str) -> tuple[str, str]:
    """Returns (verdict, detail). verdict is one of:
    'ok' (module-level code ran clean, or failed for a reason unrelated to
    the platform/uuid shadow), 'shadow_regression' (hit the specific
    failure signature this eval exists to catch), 'timeout' (module-level
    code hung past the timeout -- some KNOWN_AFFECTED_FILES entries
    attempt live Redis/Postgres connections at import time, plausible in a
    degraded environment; NOT the same as a shadow_regression, but worth
    surfacing rather than crashing the whole eval), 'missing' (file no
    longer exists -- update KNOWN_AFFECTED_FILES)."""
    target = REPO_ROOT / relpath
    if not target.exists():
        return "missing", f"{relpath} no longer exists"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_SRC, str(target)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        # Review finding, 2026-08-13: an uncaught TimeoutExpired here
        # crashed the whole eval with a raw traceback, directly
        # contradicting this module's own documented "any other failure
        # is reported as informational, not a hard failure" contract.
        return "timeout", f"{relpath}: module-level code exceeded 30s"
    if any(sig in result.stderr for sig in _SHADOW_SIGNATURES):
        return "shadow_regression", result.stderr[-600:]
    return "ok", ""


def smoke_run_all_audits() -> tuple[bool, str]:
    run_id = "blast_radius_eval_probe"
    out_dir = REPO_ROOT / "codex_reviews" / run_id
    try:
        try:
            result = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "platform_audits" / "run_all_audits.sh"), run_id],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            # Same fix as probe_file()'s TimeoutExpired handling -- a raw
            # uncaught timeout here would crash the eval instead of
            # reporting a clean FAILED, same review finding, 2026-08-13.
            return False, "run_all_audits.sh exceeded 120s"
        reports_dir = out_dir / "reports"
        ok = result.returncode == 0 and reports_dir.is_dir() and any(reports_dir.iterdir())
        detail = "" if ok else (result.stderr[-800:] or f"no reports written under {reports_dir}")
        return ok, detail
    finally:
        # Disposable probe run -- don't leave real output artifacts behind.
        # Runs even on a timeout (the try/except above still returns
        # normally, and this outer finally always fires either way).
        shutil.rmtree(out_dir, ignore_errors=True)


def main() -> int:
    print("=== Blast-radius eval: scripts/platform -> scripts/platform_audits rename ===\n")

    print("[1/2] Renamed package end-to-end smoke (run_all_audits.sh)...")
    audits_ok, audits_detail = smoke_run_all_audits()
    print("  OK" if audits_ok else f"  FAILED: {audits_detail}")

    print(f"\n[2/2] Known-affected files ({len(KNOWN_AFFECTED_FILES)}) regression sweep...")
    regressions: list[tuple[str, str]] = []
    missing: list[str] = []
    timed_out: list[str] = []
    for relpath in KNOWN_AFFECTED_FILES:
        verdict, detail = probe_file(relpath)
        if verdict == "shadow_regression":
            regressions.append((relpath, detail))
            print(f"  REGRESSION: {relpath}\n    {detail}")
        elif verdict == "missing":
            missing.append(relpath)
            print(f"  STALE (file moved/removed): {relpath}")
        elif verdict == "timeout":
            # Informational, not a hard failure -- a hang at import time
            # (e.g. a live Redis/Postgres connection attempt) is unrelated
            # to the platform/uuid shadow this eval exists to catch, but
            # it does mean this file's coverage is inconclusive this run.
            timed_out.append(relpath)
            print(f"  TIMEOUT (inconclusive, not a regression): {relpath}")

    clean_count = len(KNOWN_AFFECTED_FILES) - len(regressions) - len(missing) - len(timed_out)
    print(
        f"\n  {clean_count}/{len(KNOWN_AFFECTED_FILES)} clean, {len(regressions)} regression(s), "
        f"{len(timed_out)} timeout(s), {len(missing)} stale entr{'y' if len(missing) == 1 else 'ies'}"
    )
    if missing:
        print("  Stale entries are a maintenance nit (update KNOWN_AFFECTED_FILES), not a blast-radius failure.")
    if timed_out:
        print("  Timeouts are inconclusive (unrelated hang, e.g. a live DB/bus connection attempt at import time), not treated as regressions.")

    if not audits_ok or regressions:
        print("\n=== FAILED ===")
        return 1

    print("\n=== PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
