# Combine `sustained_load_pressure` into the tension outreach trigger

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1723
Branch: `feat/significance-aware-outreach`

## Summary

- Wires `orion.field.significance.sustained_load_pressure` (PR #1718, shipped sensing-only) into `services/orion-hub/scripts/tension_outreach_trigger.py` — the first real consumer.
- `TensionTriggerReason` now carries the latest tick's `sustained_load_pressure` alongside the existing `peak_deviation_pressure`, read straight off the same `substrate_field_state` row this trigger already queries — no recomputation in Hub.
- `endogenous_outreach.py::build_outreach_prompt` states it as a second, separate real fact when nonzero — no scripted "worried"/"concerned" language; generation draws its own conclusion from the two real numbers.
- `status()` reports both numbers on `last_tension_reason` for operator inspection.
- Updated `services/orion-hub/README.md` §4.1 and added a missing section to `services/orion-field-digester/README.md` documenting `orion.field.significance`/`sustained_load_pressure` (never had one since PR #1718 shipped).
- Updated both design docs (`2026-08-16-tension-driven-outreach-design.md`, `2026-08-16-level-aware-significance-design.md`) marking the "combine honestly" deferred item as done.

## Outcome moved

Hub's endogenous-outreach trigger can now honestly reference a genuinely level-aware signal (not just change-detection) when deciding what to say — the combination both design docs explicitly named as real, separate follow-up work is now shipped.

## Current architecture

`tension_outreach_trigger.py::current_run()` read only `tension_borda_winner_target_id`/`tension_deviation_pressure` off `substrate_field_state.field_json`, tracking a consecutive-run persistence bar. `orion.field.significance.sustained_load_pressure` existed (PR #1718) but had zero consumers.

## Architecture touched

- `services/orion-hub/scripts/tension_outreach_trigger.py`: `_fetch_recent_winners` query + coercion, `TensionTriggerReason` dataclass, `current_run()`.
- `services/orion-hub/scripts/endogenous_outreach.py`: `build_outreach_prompt`, `status()`.
- No schema/bus/env changes — pure reader of an already-persisted column.

## Files changed

- `services/orion-hub/scripts/tension_outreach_trigger.py`: reads + carries `sustained_load_pressure`; reuses `orion.substrate.falkor_codec._safe_float` instead of a 3rd hand-rolled coercion copy; keeps SQL NULL distinct from genuine 0.0 through `_fetch_recent_winners`.
- `services/orion-hub/scripts/endogenous_outreach.py`: second prompt fact + `status()` field.
- `services/orion-hub/tests/test_tension_outreach_trigger.py`, `test_endogenous_outreach.py`: new coverage.
- `services/orion-hub/README.md`, `services/orion-field-digester/README.md`: doc updates.
- `docs/superpowers/specs/2026-08-16-tension-driven-outreach-design.md`, `2026-08-16-level-aware-significance-design.md`: combined-signal + metric-quality-gate write-up.

## Schema / bus / API changes

- Added: nothing new — `sustained_load_pressure` already exists on `FieldStateV1` (PR #1718); this patch only reads it from a new consumer.
- Removed: nothing.
- Renamed: nothing.
- Behavior changed: `TensionTriggerReason` gained a `sustained_load_pressure: float = 0.0` field (backward-compatible default); `status()`'s `last_tension_reason` dict gained the same key.
- Compatibility notes: none needed — additive, defaulted field.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no (nothing to update — `FIELD_SIGNIFICANCE_WINDOW_SECONDS`/`FIELD_SIGNIFICANCE_CHECK_INTERVAL_SEC` were already present from PR #1718, verified during this patch).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a, no env changes.
- skipped keys requiring operator action: none.

## Tests run

```text
rtk proxy /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  services/orion-hub/tests/test_tension_outreach_trigger.py \
  services/orion-hub/tests/test_endogenous_outreach.py -q
98 passed

rtk proxy /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests -q
32 failed, 1323 passed, 5 skipped -- all 32 failures verified pre-existing and identical on
an untouched main checkout (test_llm_route_selector.py, test_substrate_effect_endpoint.py,
test_memory_consolidation_draft_routes.py, test_recall_strategy_profiles_runtime.py, etc.),
none touch tension_outreach_trigger.py or endogenous_outreach.py.
```

## Evals run

No dedicated eval harness for this trigger beyond the unit-test coverage above (same as PR #1707/#1715 — this is a small, deterministic-logic patch, not a generation-quality change).

## Docker/build/smoke checks

Not run — no runtime/config/dependency change, pure application-code read of an already-shipped, already-live column. `git diff --check` clean.

## Review findings fixed

Two review passes ran (`/code-review medium`). The first accidentally scored the main checkout's unrelated, pre-existing dirty `graphify-out/` state instead of this branch's diff — re-run scoped to `feat/significance-aware-outreach` explicitly, which ran all 8 finder angles + verification.

- **Finding: `_fetch_recent_winners` collapses SQL NULL (a pre-#1718 `field_json` row lacking the key) and a genuine `0.0` `sustained_load_pressure` reading to the identical value, but the field's own comment claimed 0.0 always means "a real reading, not a missing one" — exactly the missing-looks-like-calm failure shape CLAUDE.md's metric-quality-gate names by incident (`node:substrate.route`, 2026-07-26).**
  - Fix: kept `float | None` through `_fetch_recent_winners` (`_safe_float(sustained_load, default=None)`); collapsed to `0.0` only at `TensionTriggerReason` construction, since both cases mean "nothing to add to the prompt" to every real caller. Corrected the docstring to disclose the narrow, self-healing (`LOOKBACK_MINUTES=10.0`-bounded — a pre-migration row ages out of the lookback window within 10 minutes) rolling-deploy-ordering ambiguity instead of overclaiming perfect distinction.
  - Evidence: new tests `test_fetch_recent_winners_keeps_missing_sustained_load_distinct_from_zero`, `test_missing_sustained_load_column_collapses_to_zero_in_the_reason`.
- **Finding: `sustained_load_pressure` (from PR #1718) is wired into a new consumer/cognition loop here (the outreach-message trigger) without a recorded metric-quality-gate write-up for THIS specific wiring — CLAUDE.md requires re-running the gate every time a metric enters a new pipeline, not just once at its original producer-side introduction.**
  - Fix: added a 6-point gate write-up (provenance/independence/theory anchor/live-data sanity/existing-mechanism/reversibility) scoped to this consumption context to the design doc.
  - Evidence: `docs/superpowers/specs/2026-08-16-tension-driven-outreach-design.md`'s new "Metric-quality-gate for THIS wiring" section.
- **Finding: the new `sustained_load` coercion block near-duplicates the pre-existing `deviation` coercion block in the same function, and the repo already has a shared tolerant-float-coercion helper (`orion.substrate.falkor_codec._safe_float`) matching this exact contract.**
  - Fix: imported and reused `_safe_float` for both coercions instead of adding a third hand-rolled try/except copy (the module's own docstring already names two prior copies: `attention_broadcast.py`'s `_f()`, `dynamics.py`). Not promoted to a public/shared-module name in this patch — that would touch `falkor_codec.py` for no other reason this patch has, same scope call this design doc's own "3 near-duplicate hand-rolled SQL fetches" finding already made.
  - Evidence: `tension_outreach_trigger.py`'s import line + both `_safe_float(...)` call sites; test suite still 98/98 green after the refactor.
- **Checked and refuted, not acted on: a "sustained_load_pressure could render as a misleading tiny nonzero value under `:.2f`" candidate.** Verified directly against `orion/field/regime.py`: `sustained_load_pressure` is a hard step — entry into `loaded_steady` requires `pressure_level >= LOADED_LEVEL` (0.70), so any nonzero return is bounded below by 0.70. It is always exactly `0.0` or `>= 0.70`, never a tiny in-between value.
- **Checked and refuted, not acted on: a "README not updated" candidate from the first (mis-scoped) review pass.** `services/orion-hub/README.md` was in fact updated with a dedicated "Level-aware, not just change-aware" subsection in this same diff — the reviewer had been looking at the wrong repo state.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build orion-hub
```

## Risks / concerns

- Severity: low
- Concern: `sustained_load_pressure` is honestly scoped GLOBAL (`max()` over every `loaded_steady` channel/node in the significance window), not per-target — a nonzero reading does not necessarily describe the same node the deviation run's `target_id` names.
- Mitigation: disclosed in code (module docstring, field comment), README, and design docs; the prompt itself says so explicitly ("This may or may not be the same thing as the change above."); no behavior depends on the two signals actually correlating.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1723
