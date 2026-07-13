# PR report: wire recall_boost() and decay-at-read into live recall

Implements `docs/superpowers/specs/2026-07-13-memory-recall-reinforcement-decay-wiring-spec.md` in full — both items ship together per the spec's non-negotiable requirement (reinforcement without decay would only grow the ceiling-pinned fraction this codebase has already been burned by once).

## Summary

- `dynamics.py::recall_boost()` and `decayed_activation()` existed, fully unit-tested, with **zero live call sites**. Live data showed 55/100 active crystallizations already at the activation ceiling with no recall-time reinforcement ever having fired — a live precursor to the exact saturation bug already shipped once in this codebase (homeostatic drives pinned flat).
- `active_packet.py`'s ranking key now reads `decayed_activation(c, now=...)` instead of the raw stored value — decay is live everywhere ranking happens, no cron job, no new service.
- `retriever.py::retrieve_active_packet()` gains an optional `pool` param. After a packet is finalized, every crystallization that actually appears in `crystallization_refs` (not every candidate considered) gets `recall_boost()` applied and persisted.
- Hard invariant enforced by a dedicated test: `confidence`/`salience` are provably unchanged across any ordering of `recall_boost()`/`decay()` calls.
- New standing gate script, `scripts/check_activation_saturation.py` — the acceptance-check-1 ship/no-ship criterion from the spec, runnable by anyone, anytime.
- Live-verified end to end against real Postgres: a real row's activation moved by exactly the `recall_boost()` formula, confidence/salience genuinely unchanged.

## Outcome moved

The one dynamics signal with real variance (`activation`) now actually responds to real usage instead of being a write-once value from formation. Decay is live in ranking. The saturation gate that would have caught this codebase's last flat-pin bug earlier now exists as a standing, re-runnable check.

## Current architecture (before this patch)

`dynamics.py` had `recall_boost()` (boost=0.08) and `decay()`/`decayed_activation()` fully written and unit-tested in isolation — reachable from nowhere in the live path. `active_packet.py:55` read `c.dynamics.activation` directly.

## Architecture touched

`orion/memory/crystallization/{active_packet.py,retriever.py}`, two one-line pass-through additions in `orion-hub`/`orion-recall`'s existing `retrieve_active_packet()` callers, one new top-level script. No schema, bus, or `.env` changes — every field and function this patch wires already existed.

## Files changed

- `orion/memory/crystallization/active_packet.py`: ranking key uses `decayed_activation(c, now=ranking_time)`; `build_active_packet()` gained an optional `now` param (defaults to wall-clock).
- `orion/memory/crystallization/retriever.py`: `retrieve_active_packet()` gained an optional `pool` param; new `_apply_recall_boost()`/`_persist_recall_boost()` helpers, called after packet finalization.
- `services/orion-hub/scripts/crystallization_routes.py`, `services/orion-recall/app/collectors/active_packet.py`: pass their already-in-scope `pool` through to `retrieve_active_packet()` — one line each.
- `scripts/check_activation_saturation.py` (new): standing gate, `POSTGRES_URI`-driven, `--fail-above` for CI-style regression checks.
- `tests/test_memory_crystallization_dynamics.py`: +260 lines — `TestRecallBoostWiring` (5 tests), `TestDecayAtRankingReadSite` (3 tests), `TestRecallBoostDecayInvariant` (1 test), covering all 4 spec acceptance checks.

## Design decisions worth flagging

**Refetch-before-persist.** `_persist_recall_boost()` refetches the freshest row via `get_crystallization()` immediately before boosting and persisting, rather than reusing the snapshot collected at the start of retrieval. `update_crystallization()` does a full-row UPDATE (the existing pattern, matching `reinforce()`'s call sites) — persisting a stale snapshot risks clobbering a concurrent governance write. Refetching narrows the staleness window from "however long the whole retrieval took" (embed/chroma/graphiti round trips included) down to "between refetch and write." It does not eliminate the race (no transaction/row-lock) — that would need changing `repository.py`'s persistence contract, explicitly out of scope for this call-site-only patch per the spec. Documented as a known, accepted limitation, not silently left unaddressed.

**Concurrent persistence.** Multiple refs in one packet are boosted+persisted via `asyncio.gather` — each row's update is independent, no reason to serialize.

**Graceful degradation.** `pool=None` (offline/test callers) is a no-op, not an error. A persistence failure for one crystallization logs a warning and does not raise or affect the returned packet.

## Schema / bus / API changes

None. `retrieve_active_packet()`'s new `pool` param is optional and backward-compatible — every existing caller not updated in this patch continues to work unchanged (degrades to no persistence).

## Env/config changes

None.

## Tests run

```text
$ source venv/bin/activate && python -m pytest tests/test_memory_crystallization_dynamics.py -v
22 passed in 0.47s
```
All 4 spec acceptance checks covered: head-to-head recall competition (`test_recall_boost_wins_contested_bucket_slot`), decay reduces rank for disused items (`test_disused_item_loses_contested_slot_to_recently_recalled_competitor`), the confidence/salience invariant (`test_confidence_and_salience_unchanged_across_all_orderings`), no regression to `reinforce()`'s Phase 2 call sites (`TestReinforce` unchanged, still passing).

```text
$ python -m pytest tests/test_memory_crystallization_dynamics.py tests/test_memory_crystallization.py \
    tests/test_memory_crystallization_concept_relation.py tests/test_encode_reinforce_not_duplicate.py \
    services/orion-recall/tests/ services/orion-hub/tests/test_crystallization_routes_contract.py \
    orion/memory/crystallization/tests/ -q
4 failed, 173 passed, 7 warnings in 8.30s
```
All 4 failures independently confirmed pre-existing on clean `origin/main` earlier this same session (same test names, same failures, unrelated to this patch — `TestMemoryCardBackwardCompat` registry-gap issue and 3 `orion-recall` vector/gating tests).

All output above independently re-run by the orchestrator, not taken from the implementing agent's report alone.

## Evals run

No eval harness applies. The live end-to-end verification below is the standard this session holds every recall-quality claim to.

## Docker/build/smoke checks

```text
# Standing saturation gate, independently re-run by orchestrator:
$ POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney python scripts/check_activation_saturation.py
activation_saturation: 55/102 active crystallizations at or above activation=0.99 (53.9% ceiling-pinned)
```
Ceiling count held flat at 55 (matches spec's original 55/100 baseline); fraction actually decreased slightly (53.9% vs 55.0%) as total active count grew — no regression.

```text
# Live end-to-end smoke against real Postgres, independently written and run by the orchestrator
# (not the implementing agent's script):
BEFORE id=32c50000-... activation=0.3612 reinforcement_count=0 confidence=likely salience=0.764
crystallization_refs=['32c50000-...']
AFTER  id=32c50000-... activation=0.4123 reinforcement_count=0 confidence=likely salience=0.764
expected_activation=0.4123 actual=0.4123 match=True
confidence_unchanged=True salience_unchanged=True
```
`0.3612 + (1 - 0.3612) * 0.08 = 0.4123` — exact match to `recall_boost()`'s formula. A real row moved, confirmed by independent direct query, not by trusting the function's return value.

## Review findings fixed

Implementing agent ran its own code-review subagent pass (2 parallel finder angles) before returning:
- **Fixed**: `update_crystallization`'s full-row UPDATE now fires on every recall (not just rare dedup) against a snapshot that could be stale after embed/chroma/graphiti round trips — added the refetch-before-persist mitigation described above, with a dedicated regression test.
- **Fixed**: sequential per-item persistence changed to `asyncio.gather`.
- **Fixed**: `check_activation_saturation.py` initially had no way to fail a CI-style check — added `--fail-above`.
- **Not fixed, by design**: decay is read-time-only and never persists a decrease — this is exactly the spec's stated scope (retirement/persisted-decay is an explicitly deferred, heavier follow-on needing governor review), not an oversight. See Risks below.

**Orchestrator-caught concurrency incident (not a code defect):** mid-implementation, `git stash`/`pop` (run to isolate a pre-existing test failure) raced with a concurrent agent's stash operation in a *different* worktree — `refs/stash` is shared across all worktrees in a repository by git design, not scoped per-worktree. The agent's own work briefly vanished from its working tree; root-caused via `git fsck --unreachable` (both agents' stash entries were preserved as dangling commits, not yet garbage-collected) and fully recovered — verified byte-identical to the pre-incident diff. Orchestrator independently confirmed the sibling agent's dangling stash (`96aa0ad5...`) was a fully-redundant duplicate of already-verified, already-pushed work on `feat/crystallization-confidence-assignment` — no data was lost on either side. Flagging as a real, reproducible operational hazard for future concurrent multi-worktree sessions on this host, not something this PR's diff needed to change.

## Restart required

```text
No restart required for the code change itself (pure Python, no schema/env/config changes).
```
`orion-hub` and `orion-recall` pick this up on their normal next deploy/restart cycle.

## Risks / concerns

- Severity: Medium — stored `dynamics.activation` is now monotonically non-decreasing per row (`recall_boost`/`reinforce` only push up; decay is read-time-only, never persists a decrease). The saturation gate's *stored* metric can therefore only stay flat or grow under this patch alone — it does not by itself prove ranking correctness. Ranking correctness is separately proven by the decay-at-read tests (a stale item's *effective* rank at query time is correctly demoted even though its *stored* value hasn't moved). Retirement — the mechanism that would let stored activation actually come back down — is explicitly deferred pending governor review, matching the spec's own phasing.
- Severity: Low — the refetch-before-persist mitigation narrows but does not eliminate the read-modify-write race on `update_crystallization`'s full-row UPDATE. A full fix needs either a partial-column UPDATE or optimistic concurrency in `repository.py` — explicitly out of scope for this call-site-only patch.
- Severity: Low — this PR and the already-pushed `feat/orion-recall-retrieval-event-logging` PR both touch `services/orion-recall/app/collectors/active_packet.py`, on adjacent lines within the same function (this PR adds `pool=pool,` inside the `retrieve_active_packet(...)` call; the other PR adds a new block immediately after it). Both are independently valid and mergeable against current `main` on their own — whichever merges second will likely need a small rebase to resolve proximity, not a real logical conflict (the two changes don't touch the same lines).

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/recall-reinforcement-decay-wiring?expand=1`
