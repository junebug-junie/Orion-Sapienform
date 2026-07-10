# chore(spark): remove orphaned SparkEngine facade, fix arousal-source inconsistency, rewrite README to match reality

**Status:** IMPLEMENTED, tested, reviewed. Direct follow-on to today's arousal fix
(`fix/tissue-viz-arousal-hardware-evidence`, merged as PR #936) — Juniper flagged
`orion/spark/` as "a hot mess... not knowing what the hell the metrics mean or
where they come from or if they are theater shit." This patch is the audit +
cleanup that followed.

## Summary

- Read `orion/spark/README.md` against the actual code. ~200 of its 283 lines
  (Spark verbs `spark.introspect`/`spark.debug`/`spark.theme_weaver`, an
  `OrionState` JSON blob, "knobs & policies") describe a system that was never
  built — confirmed via repo-wide grep, zero hits outside the README itself.
- Found **three parallel, inconsistent implementations** of the same four
  felt-state metrics (valence/energy/coherence/novelty):
  1. `_phi_from_self_state()` in `orion-spark-introspector/worker.py` —
     canonical, self-state-derived. What today's earlier novelty/arousal
     fixes (PRs #934-ish, #936) target.
  2. `OrionTissue.phi()` (`orion/spark/orion_tissue.py`) — a real, live-fed
     16×16×8 tensor with decay+diffusion physics, but its own `.phi()`
     output is only read as a fallback when self-state is unavailable
     (`_get_phi_stats()` prefers self-state whenever present — essentially
     always in steady state).
  3. `orion/spark/spark_engine.py` + `integration.py` + `strategies.py` — a
     third, fully independent facade. `spark_engine.py`'s own docstring
     calls itself the API "Hub, Brain, Cortex, Dream Engine, etc." should
     call. Repo-wide grep on `main`: **zero production consumers.** Two
     abandoned worktrees (`feat+substrate-signal-bridge-v1`,
     `agent-a49e9e2330944fa62`) show an unmerged, incomplete attempt to wire
     it into `orion-llm-gateway`.
- Juniper asked to trace whether a "SQL-backed heartbeat" replaced
  `OrionTissue` before deciding its fate. Traced it: `orion-equilibrium-service`
  is real and does read `substrate_self_state`/`substrate_execution_trajectory_projection`
  directly from Postgres to score "eventfulness" and publish a periodic
  `equilibrium_heartbeat` trace event — but that's a **trigger/cadence**
  mechanism for *when* to recompute phi, not a phi **data source**. It's
  orthogonal to `OrionTissue`, which remains the only fallback source and
  was not replaced by anything. Confirmed `OrionTissue` should stay.

## Changes

- **Deleted** `orion/spark/spark_engine.py`, `orion/spark/integration.py`,
  `orion/spark/strategies.py` (936 lines) and their only consumers:
  `scripts/spark_metrics_v2_scenario.py` (fully `SparkEngine`-dependent demo
  script), and the one `SparkEngine`-specific test in
  `tests/test_spark_metrics_v2.py` (kept the two tests that exercise
  `OrionTissue`, which is still live code).
- **Fixed a real inconsistency**: `worker.py::handle_semantic_upsert`'s
  `tissue.update` broadcast computed its own bespoke arousal proxy
  (`0.15 + 1.2*novelty`) instead of `phi_stats["energy"]`, the value every
  *other* broadcast site (`handle_self_state`, `handle_trace`) uses. This
  predates today's arousal fix — before that fix, `phi_stats["energy"]` was
  permanently 0, so this was likely a local workaround for that; now that
  it's a real value, the workaround was just an inconsistency. Fixed to use
  `phi_stats["energy"]` directly, matching every other site.
- **Rewrote `orion/spark/README.md`** to describe the actual architecture:
  canonical phi vs. fallback phi, the real equilibrium-service heartbeat
  mechanism, what was removed and why, and what the README no longer claims.
- **Fixed `docs/spark_metrics_v2.md`**: its "How to verify" step 2 pointed at
  the now-deleted scenario script (a pre-existing dangling reference, caught
  by review). Folded its coverage into the unit-test description (the
  surviving `test_novelty_baseline_and_spike` test already covers the same
  scenario) and renumbered the following steps.

## Found and fixed along the way: live-file test isolation gap

While trimming `tests/test_spark_metrics_v2.py`, hit a real pre-existing bug:
`OrionTissue()` with no explicit `snapshot_path` defaults to loading/writing
`/mnt/graphdb/orion/spark/tissue-brain.npz` — the **live production
snapshot** of the actually-running `orion-athena-spark-introspector`
container (confirmed via `docker ps` and matching file-mtime churn). Neither
this test file nor the introspector's own test suite isolated this before.

Fixed both:
- `tests/test_spark_metrics_v2.py`: explicit `snapshot_path` via a
  `unittest.TestCase.setUp()` + `tempfile.TemporaryDirectory()`.
- `services/orion-spark-introspector/tests/conftest.py`: sets
  `ORION_TISSUE_SNAPSHOT_PATH` to a `/tmp` path before `app.worker` (which
  constructs a module-level `OrionTissue()` singleton at import time) is
  ever imported by any test in that suite — including the new arousal
  regression test added in this branch and the pre-existing novelty test in
  `test_tissue_viz_novelty.py`, both of which call `handle_semantic_upsert`
  and therefore `TISSUE.propagate()`/`TISSUE.snapshot()`.

**Review caught this fix was initially wrong**: first pass used
`os.environ.setdefault(...)`, which silently no-ops if the var is already
in the ambient environment — and `services/orion-spark-introspector/.env_example`
(and `.env`) already set this key to the production path. Under the exact
"source `.env` before running pytest" workflow this repo's own CLAUDE.md
encourages, `setdefault` would have been defeated and the new regression
test would have written into the live production file — reproducing the
exact bug it was meant to fix. Changed to a forced `os.environ[...] = ...`.

No lasting production impact from the pre-fix window: the live file's mtime
churns every few seconds from the actually-running service's own real
traffic, so any transient one-off contamination from an earlier test run
(before the conftest fix existed) would already be superseded by the live
service's own next snapshot.

## Review findings fixed

Ran the code-review skill in a subagent (medium/high effort). 2 findings, both CONFIRMED:

- **Finding:** `conftest.py`'s `os.environ.setdefault("ORION_TISSUE_SNAPSHOT_PATH", ...)` is a silent no-op when the var is already in the ambient environment — which `.env`/`.env_example` guarantee under a sourced-`.env` workflow. The reviewer reproduced this by replicating conftest.py's exact env/import sequence in isolation.
  - **Fix:** changed to forced `os.environ[...] = ...`, documented why `setdefault` was wrong.
  - **Evidence:** re-ran full introspector suite post-fix — 120 passed, same 1 pre-existing unrelated failure.
- **Finding:** `docs/spark_metrics_v2.md`'s "How to verify" step 2 (`python scripts/spark_metrics_v2_scenario.py`) pointed at a file this branch deletes, left dangling with no explanation.
  - **Fix:** folded the scenario's coverage description into the unit-test step (already covered by the surviving `test_novelty_baseline_and_spike`), renumbered subsequent steps.
  - **Evidence:** re-read the file, confirmed no remaining reference to the deleted script.

Reviewer also confirmed clean: no remaining references to any deleted module anywhere in `.py`/`.yaml`/`.yml`/`Dockerfile`/`requirements*.txt` (outside disclosed stale worktrees); the arousal-consistency change matches the exact pattern already used at every other broadcast site; `tissue_viz.js` (the frontend consumer) has no hardcoded assumptions about the old arousal range, so no frontend breakage.

## Files changed

- `orion/spark/spark_engine.py`, `orion/spark/integration.py`, `orion/spark/strategies.py` — deleted, zero production consumers.
- `scripts/spark_metrics_v2_scenario.py` — deleted, only consumer of the above.
- `tests/test_spark_metrics_v2.py` — removed `SparkEngine`-only test + its now-broken imports; added explicit throwaway `snapshot_path` to the two surviving `OrionTissue` tests.
- `services/orion-spark-introspector/app/worker.py` — `handle_semantic_upsert`'s arousal now sources `phi_stats["energy"]` instead of a bespoke novelty-derived proxy.
- `services/orion-spark-introspector/tests/conftest.py` — forces `ORION_TISSUE_SNAPSHOT_PATH` to a `/tmp` path for the whole test suite.
- `services/orion-spark-introspector/tests/test_tissue_viz_arousal.py` — new regression test confirming `handle_semantic_upsert`'s broadcast arousal matches `phi_stats["energy"]`.
- `orion/spark/README.md` — rewritten to match actual architecture.
- `docs/spark_metrics_v2.md` — fixed dangling reference to the deleted scenario script.

## Tests run

```text
/tmp/orion-test-venv/bin/pytest tests/test_spark_metrics_v2.py services/orion-spark-introspector/tests -q
  → 122 passed, 1 pre-existing unrelated failure (test_phi_reward_emitted_when_encoder_ok,
    same failure present on main before this branch), 1 skipped

/tmp/orion-test-venv/bin/python -c "import orion.spark.orion_tissue, orion.spark.signal_mapper, \
    orion.spark.surface_encoding, orion.spark.introspection_metadata" → OK
/tmp/orion-test-venv/bin/python -c "import orion.spark.spark_engine" → ModuleNotFoundError (expected)
```

Repo-wide grep confirmed zero remaining references to `spark_engine`,
`SparkEngine`, `orion.spark.integration`, `orion.spark.strategies`, or
`spark_metrics_v2_scenario` outside stale `.claude/worktrees/*` (disclosed,
unrelated) after the fix.

## Evals run

No dedicated eval harness for `orion-spark-introspector`'s tissue-viz stats
(same disclosed gap as the two prior fixes in this area today).

## Docker/build/smoke checks

Not run against live containers. Confirmed `orion-athena-spark-introspector`
is actually running on this host (`docker ps`) and independently verified
its live snapshot file (`/mnt/graphdb/orion/spark/tissue-brain.npz`) was not
touched by this branch's test runs post-fix (`/tmp/orion-spark-introspector-test-tissue.npz`
exists instead, confirming correct redirection).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```
After restart, `handle_semantic_upsert`'s `tissue.update` broadcasts (fired
on real chat embeddings) will report `arousal` consistent with every other
broadcast site instead of the old bespoke novelty-derived proxy.

## Risks / concerns

- Severity: low. Dead-code removal has zero production consumers (verified,
  not assumed); the arousal fix makes one broadcast site consistent with
  three others that already work this way; the README rewrite is
  documentation-only.
- `OrionTissue` remains only lightly exercised in the live read path (still
  a fallback, not canonical) — this patch doesn't change that, just
  documents it honestly and fixes the test-isolation risk around it.
- `coherence`, `policy_pressure`, and other `SelfStateV1` dimensions
  documented as still-theater in the new README remain unfixed — out of
  scope here, tracked in `docs/superpowers/specs/2026-07-10-cognition-metric-lineage-registry-design.md`.

## PR link

Branch pushed: `chore/spark-dead-code-and-canonical-phi`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...chore/spark-dead-code-and-canonical-phi

`gh` CLI unauthenticated in this environment (consistent with the rest of
this session) — PR not opened via API. To open:

```bash
gh pr create --title "chore(spark): remove orphaned SparkEngine facade, fix arousal-source inconsistency, rewrite README" \
  --base main --head chore/spark-dead-code-and-canonical-phi \
  --body-file docs/superpowers/pr-reports/2026-07-10-spark-dead-code-canonical-phi-pr.md
```
