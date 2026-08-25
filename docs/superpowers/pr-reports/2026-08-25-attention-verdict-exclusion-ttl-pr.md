# PR report: let resolved/dismissed loop exclusion lapse after 48h

## Summary

- Root-caused (live code execution against the real DB, not inference) the 2026-08-19 68h reverie dead window: `orion.substrate.attention.verdicts.load_terminal_verdict_loop_ids()` excludes a resolved/dismissed loop from `build_open_loops()` **permanently**, with no way for a genuinely-worsening loop to compete again.
- First fix idea (reuse `orion.field.regime.channel_regime()`/`orion.field.significance.sustained_load_pressure()`, the "calm vs busy" rank-based machinery from PR #1622/#1633) was a **category error**, caught before implementation: that machinery reads `field_json.node_vectors` (infra/biometrics telemetry), a completely different store from the substrate belief-graph node metadata (`dynamic_pressure`/`prediction_error`) reverie's open-loop signals actually come from. No producer bridges the two — confirmed by a repo-wide grep.
- `attention_loop_outcome.salience_at_close` looked like the right existing column to compare against instead (real, non-degenerate, already persisted per-verdict), but live data ruled that out too: the only two loop_ids ever re-verdicted (`open-loop-5038aeb46982`, `open-loop-64730f9cfeda`) had **identical** `salience_at_close` across both of their verdicts, and both re-closes landed in ~20-second clusters shared with several unrelated loop_ids — a bulk triage sweep, not organic per-loop reassessment. A salience-delta threshold would have been calibrated against noise.
- What the same two re-verdicted loops DO show cleanly: real gaps of ~37h22m and ~35h34m between first and second verdict. `VERDICT_EXCLUSION_TTL_HOURS = 48.0` rounds that up — n=2, disclosed as thin, but real data rather than a guess.
- `min_salience` (already enforced in `substrate_pressure_signals()`) keeps a loop from re-entering the candidate pool at all unless it is independently salient again *today* — the TTL only governs how long a closed verdict blocks re-entry, no extra gate layered on top.

## Outcome moved

A loop that goes dark after a human resolve/dismiss can now surface again after 48h if it's still independently salient, instead of being suppressed forever. Directly closes the mechanism behind the 2026-08-19 incident (a real loop, real evidence, silent for 68h).

## Current architecture

`build_substrate_attention_frame()` → `substrate_pressure_signals()` (reads substrate graph node `dynamic_pressure`/`prediction_error` metadata) → `merge_signals()` → `build_open_loops(verdict_lookup=load_terminal_verdict_loop_ids)`. `load_terminal_verdict_loop_ids()` queried `attention_loop_outcome` for the most recent verdict per loop_id and excluded any loop whose most recent verdict was `resolved`/`dismissed`, unconditionally, forever.

## Architecture touched

- `orion/substrate/attention/verdicts.py` — TTL added to the exclusion query and its docstring (full live-data derivation, including the ruled-out approaches).
- `orion/substrate/attention_broadcast.py` — threads `now=resolved_now` into the `verdict_lookup` call so the TTL check uses the same tick timestamp as the rest of the frame, not a separately-read wall clock.
- `orion/substrate/tests/test_attention_verdict_exclusion.py` — new TTL-specific coverage; two pre-existing e2e tests updated for the new `verdict_lookup` call signature (`now=` kwarg).

## Files changed

- `orion/substrate/attention/verdicts.py`: `load_terminal_verdict_loop_ids(loop_ids, *, now=None)` now also selects `created_at`, computes `resolved_now = now or datetime.now(timezone.utc)`, and only keeps a terminal-verdict loop_id excluded if `resolved_now - created_at <= timedelta(hours=VERDICT_EXCLUSION_TTL_HOURS)`. A row with no `created_at` fails closed (stays excluded) rather than silently re-arming on malformed data. A naive (tz-unaware) `created_at` is treated as UTC.
- `orion/substrate/attention_broadcast.py`: `verdict_lookup=lambda ids: load_terminal_verdict_loop_ids(ids, now=resolved_now)` instead of passing the bare function reference.
- `orion/substrate/tests/test_attention_verdict_exclusion.py`: 5 new tests (within-TTL still excluded, past-TTL lapses — the exact regression this patch fixes, naive-datetime handling, missing-`created_at` fail-closed, plus a `_fake_engine` helper to de-duplicate the mock-engine setup); 2 existing e2e tests' monkeypatched callables updated to accept the new `now=` kwarg.

## Schema / bus / API changes

- Added: none (no schema/table change — `created_at` already exists on `attention_loop_outcome`, just wasn't selected before).
- Removed: none.
- Renamed: none.
- Behavior changed: `load_terminal_verdict_loop_ids()` gained a keyword-only `now` parameter (backward compatible — defaults to the real wall clock, so any other caller keeps working unchanged). A terminal verdict's exclusion now lapses after 48h instead of holding forever.
- Compatibility notes: only one real caller (`attention_broadcast.py`) exists; updated in the same patch.

## Env/config changes

- Added keys: `ORION_ATTENTION_VERDICT_EXCLUSION_TTL_HOURS` (default `48.0`) — override for `VERDICT_EXCLUSION_TTL_HOURS`, added during the review-fix pass so the thin n=2 calibration can be corrected without a code change/redeploy.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: `services/orion-substrate-runtime/.env_example`.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes — `orion-substrate-runtime: +ORION_ATTENTION_VERDICT_EXCLUSION_TTL_HOURS='48.0'`.
- skipped keys requiring operator action: none. (The sync run also reported several pre-existing, unrelated "Diverged" keys across other services — not touched by this patch.)

## Tests run

```text
docker exec orion-athena-substrate-runtime python3 -m pytest orion/substrate/tests/test_attention_verdict_exclusion.py -q
21 passed in 1.09s

docker exec orion-athena-substrate-runtime python3 -m pytest orion/substrate/ -q \
  --ignore=orion/substrate/experiments/hyperbolic_gpt/smoke_test.py \
  --ignore=orion/substrate/tests/test_causal_geometry_fdr_correction.py \
  --ignore=orion/substrate/tests/test_causal_geometry_producer.py
688 passed, 2 warnings in 16.85s
```

(The three ignored files fail to import for a pre-existing, unrelated reason — `torch`/`numpy` aren't installed in this production runtime image. Not touched by this patch; confirmed by diff.) No local pytest/pip was available outside the container (`ModuleNotFoundError: No module named 'pytest'`/`'pip'`), so `pytest` was installed inside `orion-athena-substrate-runtime` and the edited files `docker cp`'d in for a real dependency run rather than skipped.

## Evals run

No eval harness exists for `orion/substrate/attention/` — this is a scoring/exclusion-logic fix covered by the unit/e2e tests above, not a quality-judgment behavior evals would measure differently. Not adding one here to keep the patch a thin fix, per CLAUDE.md's "report the gap" option.

## Docker/build/smoke checks

Not run — no compose/runtime/dependency change, pure Python logic + one new keyword arg on an existing internal function. Ran the real code inside the live `orion-athena-substrate-runtime` container instead (see Tests run) as the closest thing to a runtime smoke for this change.

## Review findings fixed

`/code-review` (medium, forked, fanned out to 3 review angles + verification) surfaced 10 candidates; the reuse-finder's "duplicate cooldown constant" candidate was checked against `attention_loops_store.py::suppress_loop`'s pre-existing 24h cooldown docstring, found to answer a genuinely different question, and folded into finding 3 below rather than reported standalone.

- Finding: naive `now` raises inside the guarded `try/except`, silently caught by the same handler that guards real DB failures, re-arming an *entire batch* instead of failing one comparison.
  - Fix: `resolved_now` now coerced to UTC-aware the same way `created_at` already was, before entering the try block.
  - Evidence: new test `test_load_terminal_verdict_loop_ids_naive_now_does_not_poison_whole_batch` — passes.
- Finding: `VERDICT_EXCLUSION_TTL_HOURS` was a hardcoded constant, unlike sibling thresholds in `attention_frame.py` that use an `_env_*` override convention — worse reversibility for a value explicitly disclosed as thin (n=2).
  - Fix: added `_ttl_hours()` reading `ORION_ATTENTION_VERDICT_EXCLUSION_TTL_HOURS`, wired into the exclusion query; documented in `.env_example`.
  - Evidence: `test_ttl_hours_env_override`, `test_ttl_hours_invalid_env_falls_back_to_default` — pass.
- Finding: CLAUDE.md's metric quality gate items 2 (independence) and 6 (reversibility) weren't recorded in the docstring alongside the other 4 items.
  - Fix: added explicit "Independence check" and "Reversibility" paragraphs.
  - Evidence: `orion/substrate/attention/verdicts.py` module docstring.
  - Also folds in: existing-mechanism cross-reference gap (should have evaluated `attention_loops_store.py::load_pending_loops`'s evidence-based staleness pattern and `dynamics.py`'s `dormancy_updated_at` as candidates before choosing a wall-clock TTL) — addressed as a disclosed "Known limitation, not addressed here" paragraph rather than a rewrite, to keep this patch a thin, single-mechanism fix; named as the natural next step if the TTL proves too blunt.
- Finding: no test for a mixed batch (one malformed row alongside valid rows) — the exact shape where the naive-`now` batch-poisoning bug would have surfaced.
  - Fix: added `test_load_terminal_verdict_loop_ids_mixed_batch_bad_row_does_not_poison_others`.
  - Evidence: passes; asserts only the malformed row's own loop_id is affected.
- Finding: the exact 48h boundary (`<=`) was untested — only 47h/49h were covered.
  - Fix: added `exactly_48h_ago_boundary` parametrize case, asserting inclusive (still excluded at exactly 48h).
  - Evidence: `test_load_terminal_verdict_loop_ids_ttl_cases[exactly_48h_ago_boundary-...]` — passes.
- Finding: 4 near-duplicate single-row test functions differing only in `created_at`/expected result.
  - Fix: consolidated into one `@pytest.mark.parametrize`-driven test.
  - Evidence: `test_load_terminal_verdict_loop_ids_ttl_cases` — 4 parametrized cases, all pass.
- Finding (not changed): a row with `created_at IS NULL` fails closed (stays excluded forever) — reproduces the pre-patch permanent-exclusion bug for that one row.
  - Disposition: verified gated out today by `attention_loop_outcome.created_at timestamptz not null` (`services/orion-sql-db/manual_migration_attention_loop_outcome.sql:16`) — unreachable via normal writes. Kept the defensive fail-closed behavior and its existing test rather than changing behavior for an unreachable case.
- Finding (not changed): incident-narrative docstring content (rejected fix attempts, specific loop_ids/timestamps) "belongs in the PR description, not permanently in source."
  - Disposition: declined — this is the established, consistent house style across every neighboring module read this session (`salience.py`, `regime.py`, `significance.py`, `tension_outreach_trigger.py` all carry equivalent incident/derivation narratives in their own docstrings), and CLAUDE.md's metric gate explicitly wants provenance tied to the code, not passed verbally.
- Finding (not changed): the naive-datetime coercion one-liner duplicates a pattern seen elsewhere (`episodic_consolidation.py::_utc()`, `receipts/retention.py::_utc()`).
  - Disposition: declined — those are module-private (leading-underscore) single-use helpers in unrelated modules, not a shared public utility; extracting a shared helper for a 2-line coercion would add an abstraction layer without removing real duplication risk.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml \
  up -d --build orion-substrate-runtime
```

Only `orion-substrate-runtime` calls `build_substrate_attention_frame()` (the `_attention_broadcast_tick` in `services/orion-substrate-runtime/app/worker.py`) — no other service needs a restart for this to take effect.

## Risks / concerns

- Severity: low
- Concern: `VERDICT_EXCLUSION_TTL_HOURS = 48.0` is calibrated from n=2 real data points — thin, disclosed as such in the docstring and this report.
- Mitigation: 48h is a conservative round-up above both observed gaps (~37h22m, ~35h34m), not a split between them; `min_salience` independently gates re-entry so a mis-calibrated TTL produces at most an occasional stale-loop resurface, not a flood. Easy to revisit with more data once more verdicts accumulate — the constant is named and documented for exactly that.

## PR link

(added after `gh pr create`)
