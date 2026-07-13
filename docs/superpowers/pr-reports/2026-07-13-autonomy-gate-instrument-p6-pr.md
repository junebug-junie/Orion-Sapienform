# fix(autonomy-gate): instrument fix -- UNMEASURABLE verdict, --window-hours, retention caveat (P6)

## Summary

- `scripts/analysis/measure_autonomy_gate.py` (the read-only gate that decides whether Orion's endogenous-drive-origination cognition feature is safe to build) had a real bug: when a data source returned zero rows -- unreachable, or a write pipeline silently stopped -- every metric degraded to `0.0`, which read as a genuine "measured and flat, NO-GO" verdict instead of "we measured nothing."
- Added a third verdict value, `UNMEASURABLE`, returned instead of a numeric NO-GO when an input is empty. `run()`/`main()` now exit `2` when either verdict is UNMEASURABLE, distinct from `0` for a completed GO/NO-GO measurement.
- Added `--window-hours` (mutually exclusive with `--window-days`) -- the 1-hour run used to produce today's earlier verdict-(a) GO needed an ad-hoc wrapper before this.
- Added a retention-bound caveat: if the requested window reaches further back than a source's actual live retention, the report names which source is the binding constraint and by how many hours, instead of silently under-covering older buckets.
- Live-diagnosed a real, previously-uncommitted finding while re-running the fixed instrument: the Fuseki `DriveAudit` graph has received zero writes since 2026-06-19T07:06:29Z (traced to commit `e9b233e9`, which disabled RDF materialization that day). The 2026-07-08 verdict-(b) NO-GO on record was computed against this same frozen snapshot the whole time -- corrected to UNMEASURABLE, with the root cause committed so it stops being re-measured as a false NO-GO.

## Outcome moved

The gate this feature (and the internal-economy spec) is blocked on now tells the truth about whether it measured anything. Verdict (a), re-run live today, is **GO** on both 1h and 7d windows -- Step 1 of the origination arc is no longer gated. Verdict (b) is now honestly **UNMEASURABLE**, not a fabricated NO-GO -- Step 4 stays blocked, but on the right basis.

## Current architecture

The original P6 scope (from `docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md`) assumed the Fuseki co-activation query itself needed repointing. Re-verified against live data before starting: that part was already fixed by an unrelated commit series (`de0ad072`..`8d20cc36`, already on `main`) -- confirmed via a direct SPARQL `COUNT` (444,943 rows exist). P6's real remaining scope was re-derived from reading the live script instead of trusting the (by-then-stale) spec text.

## Architecture touched

One script (`scripts/analysis/measure_autonomy_gate.py`), its test file, its README, and the origination design doc it gates (`docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`). No services, no runtime config, no `.env` files -- this is a standalone, read-only analysis CLI.

## Files changed

- `scripts/analysis/measure_autonomy_gate.py` -- `UNMEASURABLE` constant + guard in both verdict functions (symmetric, after a review fix -- see below), `retention_caveat()` + two new IO adapters (`fetch_earliest_self_state_ts`/`fetch_earliest_receipt_ts`), `--window-hours` flag + `resolve_window()`, exit code 2 on UNMEASURABLE.
- `scripts/analysis/tests/test_measure_autonomy_gate.py` -- 10 new pure-layer tests (no DB/network); one existing test's assertion flipped from `NO-GO` to `UNMEASURABLE` (it had been asserting the bug's own behavior as correct).
- `scripts/analysis/README.md` -- documents `--window-hours`, `UNMEASURABLE`, retention caveats, exit codes.
- `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md` -- appended today's live re-run with the frozen-graph root cause.
- `docs/superpowers/specs/2026-07-13-autonomy-gate-instrument-p6-design.md` (new) -- design spec.

## Schema / bus / API changes

None -- standalone script, no bus/schema/service surface.

## Env/config changes

None.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-autonomy-gate-instrument-fix
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest scripts/analysis/tests/test_measure_autonomy_gate.py -q

26 passed
```
`git diff --check` clean.

**Live verification (not a unit test, a real run against production Postgres/Fuseki, read-only):**

```text
POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \
AUTONOMY_GRAPH_QUERY_URL=http://localhost:3030/orion/query \
python scripts/analysis/measure_autonomy_gate.py --window-hours 1
# exit code: 2 (verdict b UNMEASURABLE)
# (a) GO -- silent median_abs_trajectory 0.0442 >= 0.03
# (b) UNMEASURABLE -- 0 DriveAudit rows in the last hour

python scripts/analysis/measure_autonomy_gate.py --window-days 7
# exit code: 2
# (a) GO -- silent median_abs_trajectory 0.0408 >= 0.03
# (b) UNMEASURABLE -- 0 DriveAudit rows in the last 7 days
# caveat fired: substrate_self_state retention only covers back to
#   2026-07-10T18:32:24Z, 95.9h short of the requested window start
```

## Evals run

None -- this is a measurement instrument, not a service; its own output (GO/NO-GO/UNMEASURABLE against real data) is the eval.

## Docker/build/smoke checks

Not applicable -- no container, no runtime service. The "smoke test" for this patch is the two live runs above, both against real production Postgres and Fuseki, read-only (`open_readonly_connection` refuses to proceed if the session isn't verified read-only).

## Review findings fixed

- Finding: `verdict_economy`'s `UNMEASURABLE` guard only checked `drive.record_count == 0`, not `pressure.row_count == 0`, even though the GO/NO-GO rule reads both. A dead Postgres source with a live Fuseki source would silently produce a real-looking `"NO-GO"` string instead of `UNMEASURABLE` -- the exact failure mode this patch exists to prevent, left open on one of its two inputs.
  - Fix: guard changed to `if drive.record_count == 0 or pressure.row_count == 0: return UNMEASURABLE`.
  - Evidence: `tests/test_measure_autonomy_gate.py::test_verdict_b_unmeasurable_when_pressure_empty_even_with_live_drive_data` -- live coactivation data (50%, well above threshold) + empty pressure now correctly returns UNMEASURABLE, not NO-GO.
- Finding: one new test (`test_unmeasurable_constant_is_distinct_from_go_and_no_go`) only compared string literals and exercised no function under test -- zero regression value.
  - Fix: replaced with the asymmetric-guard regression test above.
  - Evidence: `git diff` on the test file.
- Checked and confirmed correct (no fix needed): exit-code precedence (report/CSV/progress always written before the exit-code decision), argparse mutual-exclusion and default-preservation, `retention_caveat`'s comparison direction and hour math, the new IO adapters' degrade-on-failure convention, and the `e9b233e9` commit citation in the origination doc (verified via `git show e9b233e9` -- real commit, ~2.5 minutes after the doc's claimed last-Fuseki-write timestamp, tight and plausible match).

## Restart required

```text
No restart required.
```
Standalone script; no service reads or depends on it at runtime.

## Risks / concerns

- Severity: low
- Concern: verdict (b) (internal economy) remains genuinely unmeasured -- re-enabling Fuseki RDF materialization for `DriveAudit` (reversing `e9b233e9`) is a real infra decision (that commit disabled it to relieve Fuseki load) not made or recommended here; repointing the gate's adapter at a different live source is named as a follow-up, not attempted.
- Mitigation: none needed for this patch specifically -- the point of P6 was making the instrument honest about not knowing, not making Step 4 measurable. That's a separate, larger decision for whoever owns the internal-economy spec next.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/autonomy-gate-instrument-p6
