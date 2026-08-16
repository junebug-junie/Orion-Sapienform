# PR report — rebuild credit-integrity detection on write evidence, not shape

Branch: `fix/credit-integrity-vacuous-truth` (two commits: the vacuous-truth fix, then this rebuild)

## Summary

- Post-merge review of #1686 (`421b69936`) found the feedback-credit watch's core method measures the wrong thing: over 6,000 live ticks / 3 credited dimensions, **0 of 454 flagged `silent_producer` windows contained a single actual 0.92 decay step**. The classifier's premise ("any real write breaks monotonicity") is false for these three channels since `decay.py`'s 2026-07-17 rewrite made decay hold-then-decay-on-staleness, not unconditional.
- Concretely: `reliability_pressure` going 0.9 → 0.0 because a real problem got fixed — the single most important true positive this module should credit — read as "only the decay loop touched this." Separately, 75% of `resource_pressure`'s findings had the merge WINNER changing source mid-window, which a shape classifier cannot distinguish from an outage.
- Replaced the shape classifier entirely with two write-evidence signals, routed by how each tick's winning channel actually won: real node-write timestamps (reusing R2's `_refresh_from_timestamps` verbatim) for node-vector-sourced channels, and per-tick diffusion-contribution freshness (grounded in reading `apply_diffusion`'s actual source) for capability-routed channels.
- Moved `winning_write_time()` out of Hub-only code into `orion/field/pressure.py` (shared) so this is a second consumer of R2's mechanism, not a second implementation.
- A second review of this rebuild found one real structural bug (the per-channel mechanism choice was a majority vote over the WHOLE batch, so a real outage in a channel's numerically-minority sourcing type was invisible to both signals) and fixed it: both signals now run independently whenever they have evidence, never a single vote.

## Outcome moved

Live, most recent 6,000 ticks (`scripts/check_merge_domination.py --gate --json`, real Postgres): **0 findings**, down from 3 findings on effectively every one of 29 consecutive hourly cron runs since #1686 merged — all of which the first review showed were false. Verified against a second, independent 6,000-tick window (offset 15,000 ticks earlier, spanning 2026-08-13 15:28–18:52): also clean, same 100%/0% node-vs-capability split per channel, no missed outages, no false positives.

One genuine finding was produced and hand-verified during development: a real 31-second gap in `reliability_pressure`'s `node_vector_updated_at` (`node:rpc_timeout`, value flat at 0.5, last write frozen at `01:12:49` for the full stretch) — confirmed directly against raw Postgres rows before trusting the detector's own output.

## Current architecture

`orion/field/credit_integrity.py` (R5 of the phase-5 roadmap, PR #1622) is a report-only watch, read hourly by `scripts/check_merge_domination.py`'s cron entry, over `config/feedback/feedback_policy.v1.yaml`'s `positive_delta_channels` (`execution_pressure`, `reliability_pressure`, `resource_pressure`). No gating of the feedback loop itself — proposal mode (CLAUDE.md 0A) applies to that, and this module is the measurement a real guard would need first.

## Architecture touched

- `orion/field/credit_integrity.py`: full rebuild of the detection mechanism (see module docstring for the complete rationale, including the two post-merge reviews' findings).
- `orion/field/pressure.py`: `winning_write_time()` added (moved from Hub).
- `services/orion-hub/scripts/field_channel_glossary_routes.py`: imports the shared function instead of a private copy.
- `scripts/check_merge_domination.py`: report block updated to the new shape; dropped the hardcoded 20.1/9.1/3.7% figures the first review found the shipped module could not actually reproduce.
- `tests/test_credit_integrity.py`: rewritten.

## Files changed

- `orion/field/credit_integrity.py`: shape classifier removed; two write-evidence signals (`_find_silent_window_timestamp`, `_find_silent_run_contribution`) added, routed independently per channel via `_classify_tick`.
- `orion/field/pressure.py`: `winning_write_time()`.
- `services/orion-hub/scripts/field_channel_glossary_routes.py`: import updated, private copy deleted.
- `scripts/check_merge_domination.py`: `feedback_credit` JSON block and human-readable report updated.
- `tests/test_credit_integrity.py`: 20 tests, rewritten around the new mechanism.

## Schema / bus / API changes

None. Report-only; no new channel, schema, or API surface.

## Env/config changes

None.

## Tests run

```
/tmp/r4venv/bin/python -m pytest tests/test_credit_integrity.py -q
    20 passed

/tmp/r4venv/bin/python -m pytest services/orion-hub/tests/test_field_channel_glossary_routes.py -q
    37 passed, 1 pre-existing failure (test_channels_endpoint_returns_35_entries,
    35 vs 38 -- parked by Juniper, unrelated to this diff)
```

## Evals run

Mutation testing, two rounds:

- **My own sweep** (4 targeted mutants: dominance comparison, node-vs-capability detection, contribution-run span off-by-one, capability_fresh short-circuit): 4/4 killed.
- **Second review's broader sweep** (10 mutants): 6/10 killed, 4 survived — the dominance tie-break (moot once the real bug was fixed and the vote removed), the signal-1 verdict check (`== SILENT` vs `!= "producer_written"`, which would conflate "unknown" with "silent"), a likely-equivalent capability-freshness truthiness mutant, and `_rolling_windows`' exact-boundary case. Closed with 4 new tests (`test_low_stamp_coverage_reads_unknown_not_silent`, `test_outage_exactly_at_the_window_boundary_still_fires`, `test_minority_type_outage_is_not_dropped_by_majority_vote`, and the rewritten mixed-sourcing test) and re-ran the same 3 real mutants (the 4th was equivalent — `capability_fresh` is never `None` within `cap_samples` by construction) against them: all 3 now kill.

## Docker/build/smoke checks

```
POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney \
  /tmp/r4venv/bin/python scripts/check_merge_domination.py --gate
    merge domination gate: PASS
    feedback-credit watch: 0 finding(s) (policy window 30s)
```

Second, independent 6,000-tick window (offset 15,000): same clean result, hand-verified via direct Postgres queries during development (see module docstring's live-verification claims).

`--gate --json` together do not emit parseable JSON on this branch — confirmed **pre-existing** at `421b69936^` (before either of this arc's fixes), not introduced here. Out of scope for this PR.

## Review findings fixed

Two review passes, both subagent-run.

**First pass** (post-merge review of #1686, the version this PR replaces) found the module's entire detection method was wrong (F1/F2/F3/F4 in that review) — see the module docstring's "REBUILT" section for the full findings and how each was addressed. That review also confirmed `_rolling_windows` itself was correct (verified via brute-force enumeration) and the merge-domination gate wiring was unaffected.

**Second pass** (review of this rebuild):

- **CONFIRMED, most severe — the per-channel mechanism choice was a global majority vote, not per-window.** `analyse_credit_integrity` picked whichever mechanism (node timestamp vs capability contribution) had more ticks across the WHOLE analysed batch and only ever scanned that one. Reproduced concretely: 25 healthy capability ticks + 21 node ticks containing a genuine, real node write-outage → 0 findings under the majority-vote version, because the node stretch (minority) was never examined by either signal. Same defect class the rebuild set out to fix, reintroduced one level up.
  - **Fix:** both signals now run independently whenever they have any evidence (`if node_samples: ...` / `if cap_samples: ...`), never a vote. `test_minority_type_outage_is_not_dropped_by_majority_vote` regresses the exact reviewer repro.
  - **Evidence:** live re-check after the fix on the same 6,000-tick window and a second independent window, both still clean (0 findings) — confirming the fix doesn't introduce new false positives on real data, only closes the blind spot.
- **CONFIRMED — untested "unknown" vs "silent" boundary.** Mutating `verdict == SILENT` to `verdict != "producer_written"` (which also fires on `_refresh_from_timestamps`'s `"unknown"` return) survived all 17 original tests.
  - **Fix:** `test_low_stamp_coverage_reads_unknown_not_silent`. Kills the mutant.
- **CONFIRMED — untested exact-boundary window case.** Mutating `_rolling_windows`' `< window_seconds` to `<=` survived all original tests.
  - **Fix:** `test_outage_exactly_at_the_window_boundary_still_fires`. Kills the mutant.
- **CONFIRMED, minor — `ChannelSample.value` was dead.** Stored on every sample, read nowhere.
  - **Fix:** field removed.
- **Confirmed clean by review, not taken on my word:** `apply_diffusion`'s memorylessness (traced live: `worker.py` calls `reconcile_field_state_with_lattice` every tick with the same fixed lattice instance, so `possible_targets` is rebuilt from an identical declared edge set every tick — the "channel is a target on some ticks but not others" scenario the review specifically worried about is not reachable in the live worker loop), and the node-vs-capability classification split (traced against `config/field/orion_field_topology.v1.yaml`: every diffusion edge targets a `capability:*` id, never a raw node, so `_classify_tick`'s split is sound given this topology).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Hub-only (the `winning_write_time()` move touches its route file). `orion/field/credit_integrity.py` and `scripts/check_merge_domination.py` run via the existing cron entry, not a long-running service — no restart needed for those.

## Risks / concerns

- **Severity: low, stated honestly.** An outage that straddles the exact tick where a channel's winner type switches (node ↔ capability), with neither half alone reaching `window_seconds`, is still invisible to both signals. Same shape as `_rolling_windows`'s own documented resolution floor (an outage confined to the final `< window_seconds` of a series) — a real limit, not silently hidden. Not reachable in either live window checked; would need a genuine winner-type switch mid-outage to manifest.
- **Severity: low.** Report-only, as before — still no gate on the feedback loop itself. That remains explicit future proposal-mode work per CLAUDE.md 0A, and this rebuild is what makes the module's own findings trustworthy enough to eventually design that guard against.

## PR link

<pending>
