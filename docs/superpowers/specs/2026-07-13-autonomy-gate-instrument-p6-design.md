# Autonomy gate instrument fix (P6 of the motor-nerve spec) — design

**Date:** 2026-07-13
**Status:** Implementation, this session
**Mode:** Thin patch. Implements P6 of `docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md` — with its own diagnosis re-verified against live data first, since part of the original P6 scope turned out to already be fixed by unrelated work.

## Arsonist summary

P6's parent-spec framing (written earlier today) says the Fuseki `DriveAudit` graph "has been empty ≥7 days" and needs repointing. That's now **stale** — a separate commit series (`de0ad072` through `8d20cc36`, already on `main`) rewrote `fetch_drive_stats`/`build_drive_coactivation_histogram_sparql` to read the real `orion:hasDriveAssessment`/`orion:driveActive` shape via a server-side histogram query. Live-queried it directly: **444,943 `DriveAudit` rows** exist in the graph right now. Verdict (b)'s data source is not dead. Re-verified per CLAUDE.md's "runtime truth beats config truth" rather than trusting the hours-old spec text.

What's actually still missing, re-derived from reading the live script (`scripts/analysis/measure_autonomy_gate.py`) rather than the old diagnosis:

1. **No `--window-hours` flag.** Only `--window-days` (int) exists. The 1-hour run that produced today's earlier GO verdict for (a) needed an ad-hoc wrapper — there's no first-class way to ask for a sub-day window.
2. **No `UNMEASURABLE` distinction.** Both `verdict_drift` and `verdict_economy` compute their GO/NO-GO purely from the numeric thresholds. If a data source degrades to zero rows (Postgres down, Fuseki empty — the exact failure this file's own `open_readonly_connection`/`fetch_drive_stats` are built to degrade gracefully into), the metrics all resolve to `0.0`, which reads as a real behavioral "NO-GO" from the threshold comparisons — indistinguishable from "measured and it's genuinely flat." This is a real, live risk *regardless* of today's data being healthy — the whole point of a gate is to still be trustworthy the next time a source goes dark.
3. **No receipt/self-state retention-bound caveat.** Live-checked: `substrate_reduction_receipts` spans 2026-07-03 → 2026-07-13 (10 days), but `substrate_self_state` only spans 2026-07-10 → 2026-07-13 (~3 days). A `--window-days 7` run today would silently under-cover verdict (a)'s self-state metrics for the older ~4 days of the window with no indication in the report that this happened — the exact "busy/silent classification validity bound" gap P6 named, just discovered against different numbers than the original spec guessed.
4. **Today's re-verified numbers were never committed.** The origination design doc still shows only the 2026-07-08 NO-GO; the 2026-07-13 re-run (GO on verdict a) exists only in a prior session's `/tmp` output, never appended to the repo.

## Current architecture

`scripts/analysis/measure_autonomy_gate.py`: a pure/IO-split, read-only, single-file measurement CLI. Pure layer (bucket classification, drift/co-activation math, verdict rules) has 15 existing unit tests (`scripts/analysis/tests/test_measure_autonomy_gate.py`) and zero I/O. IO layer (`open_readonly_connection`, `fetch_self_state_records`, `fetch_receipt_timestamps`, `fetch_drive_stats`) degrades to empty/`None` on any failure, never raises — already follows the "never raise on read" convention this repo uses elsewhere (`services/orion-feedback-runtime/app/store.py` etc.). `run(window_days: int)` orchestrates: connect → fetch three sources → pure compute → write `report.md`/`before_after.csv`/`progress.log` to `/tmp/autonomy-gate/`.

## Proposed schema / API changes

**`build_arg_parser()`**: add `--window-hours` (float), mutually exclusive with `--window-days` (`argparse` mutually-exclusive group). `run()` gains a `window: timedelta` parameter (replacing the `window_days: int` parameter) so both flags funnel through one code path instead of forcing hours into a days-shaped API.

**Verdict functions**: `verdict_drift`/`verdict_economy` gain a `"UNMEASURABLE"` return value, returned when the relevant input has zero real rows — distinct from `"GO"`/`"NO-GO"`. Specifically:
- `verdict_drift`: `UNMEASURABLE` when both `silent.row_count == 0` and `busy.row_count == 0` (no self-state data at all in the window — can't distinguish "measured zero drift" from "measured nothing").
- `verdict_economy`: `UNMEASURABLE` when `drive.record_count == 0` (matches P6's original ask precisely) — resource-pressure zero-rows already gets its own caveat via `pressure.row_count`, but doesn't independently block the verdict since it shares the self-state source with verdict (a) and that's already covered there.

**`run()`**: exits with a distinct nonzero code (`2`) when either verdict is `UNMEASURABLE`, separate from the normal `0` exit for a completed (GO or NO-GO) measurement — so a caller (cron, CI, a human running this by hand) can immediately tell "the instrument didn't work" from "the instrument worked and said no."

**Retention-bound caveat**: after fetching self-state and receipt rows, compare `window_start` against the oldest row actually returned by each source (or the oldest row in the table generally, via a cheap `MIN(generated_at)`/`MIN(created_at)` query when the fetch itself returned rows but potentially fewer than the full window implies) and append a caveat naming which source's retention is the binding constraint and by how much, when the window requested exceeds what's actually available.

**Origination design doc**: append today's re-verified numbers (this patch's own fresh run, not the stale numbers from earlier in the day) to `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`.

## Files likely to touch

- `scripts/analysis/measure_autonomy_gate.py` — `--window-hours` flag, `UNMEASURABLE` verdict value + exit code, retention-bound caveat.
- `scripts/analysis/tests/test_measure_autonomy_gate.py` — new pure-layer tests for the `UNMEASURABLE` branches (zero-row inputs) and the mutually-exclusive-flag parsing.
- `scripts/analysis/README.md` — document `--window-hours`, `UNMEASURABLE`, exit codes.
- `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md` — append this session's fresh re-run.

## Non-goals

- Not re-deriving or re-tuning the GO/NO-GO thresholds themselves (`DRIFT_MIN_MEDIAN_ABS_TRAJECTORY`, `COACTIVATION_MIN_FRAC`, etc.) — out of scope, a separate empirical question.
- Not touching the already-fixed Fuseki co-activation query (`build_drive_coactivation_histogram_sparql`, `fetch_drive_stats`) — confirmed live and correct against 444,943 real rows; re-verifying it further is not this patch's job.
- Not building a turn-timestamp source (`run()`'s existing `# no cheap turn store located` comment stands — still true, still not invented here).
- Not flipping `ORION_ENDOGENOUS_ORIGINATION_ENABLED` — that's P7, gated on its own separate burn-in condition regardless of what this instrument reports.

## Acceptance checks

1. `pytest scripts/analysis/tests/test_measure_autonomy_gate.py -q` green, including new `UNMEASURABLE`/flag-parsing tests.
2. A live run with `--window-hours 1` succeeds and produces the same report shape as a `--window-days` run.
3. A synthetic zero-row scenario (mocked/empty inputs) returns `UNMEASURABLE` from both verdict functions, not a numeric NO-GO.
4. A live run (real Postgres + Fuseki) this session, with its actual verdicts and row counts appended to the origination design doc — not copied from an earlier session's stale numbers.

## Recommended next patch

None within this series — P6 is the last independent item; P7 (origination enable decision) remains gated on 2 weeks of clean P1–P3 burn-in per the parent spec, unaffected by this instrument fix either way.
