# Autonomy Origination Measurement Gate (Step 0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer two empirical questions with **read-only** measurement before any cognition code is written, because two downstream specs are cathedrals if the questions fail:
- **(a) Does `SelfStateV1` actually drift during exogenous silence?** — gates the endogenous-origination spec (Step 1). If self-state is flat when no receipts/turns arrive, spontaneous origination has no signal to fire on.
- **(b) How often do ≥2 drives co-activate, and does `resource_pressure` actually rise?** — gates the internal-economy spec (Step 4). If drives rarely co-activate or resource pressure sits at zero, the scarcity allocator never binds and is ornamental (§0A no-cathedral rule).

**This is Step 0 of the four-area "move the origin of wanting inside" arc.** It changes no cognition, publishes no events, writes no substrate rows. It reads durable history and emits one report with two PASS/FAIL verdicts.

**Architecture:** A single read-only analysis worker queries durable Postgres history (`substrate_self_state`, `substrate_field_state`) and replays the drive-audit bus stream (or its projection), buckets time into windows, classifies each window as *silent* vs *busy* from receipt/turn activity, and computes drift/co-activation/resource-pressure statistics per class. Output is `/tmp/autonomy-gate/report.md` plus `before_after.csv`, following the §14 read-only-analysis + monitoring protocol (no data snapshot needed; progress log required).

**Tech Stack:** Python 3.12, psycopg2 (read-only), `OrionBusAsync` (stream replay, read-only `XRANGE`), pandas or stdlib statistics. No new dependencies if stdlib `statistics` suffices.

**Related specs (gated by this plan):**
- `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md` (Step 1 — gated by verdict a)
- `docs/superpowers/specs/2026-07-07-internal-economy-scarcity-allocation-design.md` (Step 4 — gated by verdict b)
- Siblings not gated by this plan: `…-phi-intrinsic-reward-value-learning-design.md` (Step 3), `…-voluntary-attention-override-design.md` (Step 2).

**Branch / worktree:**
```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
git worktree add ../Orion-Sapienform-autonomy-gate -b chore/autonomy-origination-measurement-gate
cd ../Orion-Sapienform-autonomy-gate
```

---

## Ground truth (verified data sources)

- **`substrate_self_state`** — durable `SelfStateV1` rows (`services/orion-self-state-runtime/app/store.py:54`, pruned by `self_state_id`). Carries `dimensions{…}.score`, `dimension_trajectory`, `trajectory_condition`, `overall_surprise`, `generated_at`, and `source_field_tick_id`.
- **`substrate_field_state`** — field ticks (`services/orion-field-digester/app/store.py`). Since the heartbeat pacemaker, ticks mint even in silence, so tick presence ≠ activity; use **reduction-receipt counts** (`fetch_new_receipts`) as the true "was there exogenous input" signal.
- **Drive activity** — bus channels `orion:memory:drives:state` (`DriveStateV1.activations: Dict[str,bool]`) and `orion:memory:drives:audit` (`DriveAuditV1.active_drives: list[str]`, `dominant_drive`). `active_drives` length is the direct co-activation metric.
- **`resource_pressure`** — `substrate_self_state.dimensions['resource_pressure'].score`, computed real in `orion/self_state/builder.py:194-203` (not stubbed; defaults 0.0).
- **DSN** — `POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney`. **Read-only**: connection opened with `SET default_transaction_read_only = on`.

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/analysis/measure_autonomy_gate.py` | **Create** | Read-only measurement worker; emits report + csv + progress log |
| `scripts/analysis/tests/test_measure_autonomy_gate.py` | **Create** | Deterministic tests of window classification + statistics on fixtures |
| `scripts/analysis/README.md` | **Create/Modify** | How to run, what the verdicts mean |

No `.env_example`, schema, channel, or service changes — this touches nothing runtime.

---

## Window classification (deterministic)

Bucket the analysis window (default last 7 days) into fixed `WINDOW_SEC` (default 300s) buckets. For each bucket:
- `receipt_count` = new reduction receipts landed in bucket.
- `turn_count` = chat turns landed in bucket (from the turn/chat store, or 0 if unavailable — documented).
- **silent** iff `receipt_count == 0 AND turn_count == 0`; else **busy**.

This makes "exogenous silence" a measured fact, not an assumption.

## Metrics

**Q(a) self-state drift**, per bucket class (silent vs busy):
- `mean_abs_trajectory` = mean over rows of mean(|`dimension_trajectory`.values|).
- `dim_score_variance` = mean per-dimension variance of `score` across consecutive rows in the bucket.
- `nonstable_frac` = fraction of rows with `trajectory_condition != "stable"`.
- `mean_surprise` = mean `overall_surprise`.

**Q(b) co-activation + resource pressure**, over the whole window:
- `coactivation_frac` = fraction of drive-audit events with `len(active_drives) ≥ 2`.
- `concurrent_active_hist` = histogram of `len(active_drives)`.
- `resource_pressure`: median, p90, `frac_gt_0_3` (fraction of self-state rows with `resource_pressure ≥ 0.3`).

## Verdict rules (deterministic, stated up front)

- **Verdict (a) — Step 1 GO** iff, in **silent** buckets, `median(mean_abs_trajectory) ≥ 0.03` **AND** `dim_score_variance` is materially > 0 (silent drift ≥ 25% of busy drift). Otherwise **NO-GO**: self-state is input-following/flat, and the endogenous spec must switch its dynamics source (unresolved-pressure persistence) before build.
- **Verdict (b) — Step 4 GO** iff `coactivation_frac ≥ 0.10` **AND** `frac_gt_0_3(resource_pressure) ≥ 0.05` (so `B = B_MAX·(1−resource_pressure)` actually varies and scarcity can bind). Otherwise **NO-GO**: the economy allocator is a cathedral at current activation/pressure levels — do not build.

Thresholds are the seed decision boundary; the report prints the raw numbers so Juniper can override with judgment.

---

## Task 1: Read-only measurement worker

**Files:** `scripts/analysis/measure_autonomy_gate.py`

- [ ] **Step 1:** Open a **read-only** psycopg2 connection (`SET default_transaction_read_only = on`) to `POSTGRES_URI`. Fail loudly if the session is not read-only.
- [ ] **Step 2:** Load `substrate_self_state` rows for the window ordered by `generated_at`; parse `dimensions`, `dimension_trajectory`, `trajectory_condition`, `overall_surprise`, `resource_pressure`.
- [ ] **Step 3:** Load reduction-receipt timestamps and (if available) turn timestamps for the window; build the silent/busy bucket map (`WINDOW_SEC`).
- [ ] **Step 4:** Replay `orion:memory:drives:audit` via read-only `XRANGE` over the window; extract `len(active_drives)` per event. If the stream is trimmed shorter than the window, fall back to whatever SQL projection exists and record the actual coverage in the report.
- [ ] **Step 5:** Compute all Q(a) and Q(b) metrics; apply the verdict rules.
- [ ] **Step 6:** Write `/tmp/autonomy-gate/report.md` (verdicts + numbers + coverage caveats), `/tmp/autonomy-gate/before_after.csv` (silent-vs-busy per-metric rows), and stream progress to `/tmp/autonomy-gate/progress.log` (§14: event title, % done, rows processed/total, rate, anomalies).

**Verify:** `python scripts/analysis/measure_autonomy_gate.py --window-days 7` runs read-only (a write attempt raises), produces the three `/tmp/autonomy-gate/` files, and prints two verdict lines.

## Task 2: Deterministic tests on fixtures

**Files:** `scripts/analysis/tests/test_measure_autonomy_gate.py`

- [ ] **Step 1:** Fixture A — synthetic self-state rows that are **flat in silence** → assert Verdict (a) = NO-GO.
- [ ] **Step 2:** Fixture B — self-state rows with real trajectory movement in silent buckets → assert Verdict (a) = GO.
- [ ] **Step 3:** Fixture C — drive-audit events mostly single-active + `resource_pressure ≈ 0` → assert Verdict (b) = NO-GO.
- [ ] **Step 4:** Fixture D — frequent ≥2 co-activation + resource_pressure spread → assert Verdict (b) = GO.
- [ ] **Step 5:** Window-classification unit tests: bucket with a receipt = busy; empty bucket = silent; boundary timestamps land in the correct bucket.

**Verify:** `pytest scripts/analysis/tests/test_measure_autonomy_gate.py -q` green; verdict logic proven on both GO and NO-GO fixtures.

## Task 3: Run against live history + record verdicts

**Files:** none (execution + report)

- [ ] **Step 1:** Run the worker against live durable history for the default window.
- [ ] **Step 2:** Read `/tmp/autonomy-gate/report.md`; transcribe the two verdicts and headline numbers into the PR description.
- [ ] **Step 3:** For each downstream spec, record the gate outcome:
  - Verdict (a) GO → Step 1 (endogenous origination) cleared to build as specced. NO-GO → Step 1 blocked pending mechanism change.
  - Verdict (b) GO → Step 4 (internal economy) cleared. NO-GO → Step 4 shelved; note the measured co-activation/pressure levels that would need to change.

**Verify:** PR description carries both verdicts with evidence; each gated spec has an explicit GO / BLOCKED disposition.

---

## Monitoring (§14 read-only analysis)

No data snapshot required (read-only). Progress log is required:
```bash
tail -f /tmp/autonomy-gate/progress.log
```
Each line: event title, percent done, rows processed / total, rate, anomaly count. On completion, `report.md` + `before_after.csv` are written under `/tmp/autonomy-gate/`.

## Acceptance checks

- [ ] Worker refuses to run if the DB session is not read-only.
- [ ] Silent-vs-busy classification is measured from receipt/turn activity, not assumed.
- [ ] Report states Verdict (a) and Verdict (b) with the raw numbers behind each and any stream-coverage caveats.
- [ ] Tests prove both GO and NO-GO paths for each verdict on fixtures.
- [ ] No runtime service, schema, channel, env, or substrate row is modified or written.

## Non-goals

- No cognition change, no event emission, no substrate writes — measurement only.
- Does not gate Step 2 (voluntary attention) or Step 3 (φ reward); those have their own in-spec prerequisites.
- Not a permanent dashboard; a one-shot (re-runnable) gate. If a standing metric is wanted later, that is a separate observability task.
- Does not tune the downstream spec thresholds — it reports the numbers that inform them.
