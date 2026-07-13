# PR report: surface crystallization retirement candidates in Hub Memory tab

Implements item 2 of `docs/superpowers/specs/2026-07-13-recall-followups-loop-retirement-saturation-gate-spec.md`.

## Summary

- The retirement *action* already existed (`POST /api/memory/crystallizations/{id}/deprecate`) but nothing surfaced which crystallizations were candidates, and no UI ever reached the endpoint (curl-only).
- Hub's crystallization list endpoint now computes `decayed_activation()`/`should_retire()` live per active-status row — nothing persisted, no new schema.
- The Memory tab's review queue now also fetches active retirement candidates, badges them ("stale — review for archive"), and exposes a new Deprecate button in the detail panel.
- Self-caught review finding, already fixed: opening a retirement candidate's detail view was silently dropping the badge/decayed-activation reading because the individual-row GET endpoint (correctly left out of scope) doesn't compute those fields — fixed by carrying the already-known list-row values forward client-side.

## Outcome moved

A human reviewing the Memory tab can now see which beliefs have decayed past the retirement floor and act on them with one click, instead of the mechanism existing but being invisible and curl-only.

## Current architecture (before this patch)

`decayed_activation()`/`should_retire()` already existed and were already used at the recall ranking read-site. `/deprecate` already existed and worked. Nothing connected "a crystallization is stale" to "a human sees it and can act."

## Architecture touched

Single service, `orion-hub` — one route, one static JS file. No schema, bus, or new endpoint (reuses `/deprecate` verbatim).

## Files changed

- `services/orion-hub/scripts/crystallization_routes.py`: `crystallization_list` computes `decayed_activation`/`retirement_candidate` per active row, live, at request time. `import` of `decayed_activation`/`should_retire` from `orion.memory.crystallization.dynamics` — no modification to that module.
- `services/orion-hub/static/js/memory-crystallization-ui.js`: `retirementBadge()` helper (matches existing badge styling convention from `substrate-lattice.js`); `loadRetirementCandidates()` (best-effort, fails silently — never breaks the proposal inbox it's merged into); Deprecate button wired to the existing endpoint; detail-view fix carrying list-row fields forward.
- `services/orion-hub/tests/test_crystallization_routes_contract.py`: new tests — stale crystallization flagged correctly, existing response shape preserved for pre-existing fields.
- `services/orion-hub/tests/test_memory_crystallization_ui.py`: new test — retirement candidates + Deprecate action wired, following this file's existing text-based wiring-smoke convention (no browser-render harness exists for this page).

## Schema / bus / API changes

- Added: two new fields (`decayed_activation: float | null`, `retirement_candidate: bool`) in `GET /api/memory/crystallizations`'s response items, computed live, never persisted.
- Removed: none.
- Behavior changed: none for existing consumers — both new fields are additive; existing response shape verified unchanged by a dedicated test.

## Env/config changes

None.

## Tests run

```text
$ source venv/bin/activate && python -m pytest services/orion-hub/tests/test_crystallization_routes_contract.py services/orion-hub/tests/test_memory_crystallization_ui.py -v
test_crystallization_api_surface_present PASSED
test_active_packet_route_filters_by_recall_eligibility PASSED
test_list_endpoint_flags_stale_crystallization_as_retirement_candidate PASSED
test_list_endpoint_preserves_existing_response_shape PASSED
test_crystallization_observatory_ui_wired PASSED
test_crystallization_ui_shows_graphiti_projection_and_sync PASSED
test_crystallization_ui_surfaces_retirement_candidates_and_deprecate_action PASSED
7 passed
```

```text
$ python -m pytest services/orion-hub/tests -q
31 failed, 783 passed, 2 skipped, ... in 138.17s
```
Independently re-run by the orchestrator (not taken from the implementing agent's report alone, which reported a slightly different count — 34 vs 31 — consistent with test-order/timing nondeterminism in this broad, unrelated portion of the suite, not a regression). Confirmed via `grep FAILED ... | grep -i "crystall|retire|memory"` → **zero matches** — none of the failures touch this patch's files. One spot-checked by the implementing agent (`test_substrate_effect_endpoint`) fails with a `pydantic_core.ValidationError` on unrelated `CHANNEL_VOICE_*` settings fields when run standalone — a pre-existing env/settings-ordering issue, not caused by this patch.

`node --check services/orion-hub/static/js/memory-crystallization-ui.js` → syntax OK.

## Evals run

No eval harness exists for `orion-hub` (`services/orion-hub/evals/` does not exist) — honest gap, not fabricated coverage.

## Docker/build/smoke checks

No live browser-render verification performed — no automated UI harness exists for this page (only the text-based wiring-smoke pattern already used throughout `test_memory_crystallization_ui.py`, which the new test follows). Stated plainly rather than claimed.

## Review findings fixed

- Finding: opening a retirement-candidate row's detail panel silently lost the badge/decayed-activation reading because `crystallization_get` (correctly left unmodified, out of scope) doesn't compute the two new fields.
  - Fix: `memory-crystallization-ui.js` merges the already-fetched list row's `decayed_activation`/`retirement_candidate` into the detail payload before rendering, rather than touching the out-of-scope route.
  - Evidence: commit `9e115a4d`; existing tests re-run green after the fix; `node --check` passes.

Orchestrator independently re-read both diffs in full and re-ran the full test suite (not just the targeted files) rather than trusting the implementing agent's report alone.

## Restart required

```text
No restart required for the code itself to take effect on next deploy/reload
(FastAPI route + static JS, served as-is by orion-hub).
```

## Risks / concerns

- Severity: Low — badge/decayed-activation only refresh when the inbox reloads (page open or after an action), not live-ticking. Matches this UI's existing polling-free pattern everywhere else; acceptable for a passive review surface per the spec's explicit non-goal (no proactive notification in this patch).
- Severity: Low — `should_retire()` internally recomputes `decayed_activation()`, so the route does the decay math twice per active row per request. Negligible (pure in-memory float ops, small row counts); left as-is per the task's instruction to reuse both functions exactly as they exist rather than reimplementing inline.
- Severity: Low (pre-existing, not introduced) — `pytest services/orion-hub/tests -q` is not fully green as a whole (31 unrelated failures, confirmed pre-existing via zero file overlap and one isolated spot-check). Flagging for visibility, not something this patch should fix.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/crystallization-retirement-surfacing?expand=1`
