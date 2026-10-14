## Summary
- Dream replay draws candidates from closed Temporal Self arcs (patch 7): arcs that closed since the previous sleep began become `arc` replay items with `ref_id = arc:<id>`.
- Read directly from Postgres `temporal_self_arc` (dream already holds the engine): narrow columns, closed only, sleep excluded, one per (kind, subject), ≤20, 2 s statement timeout, 5 s wait, 5 s connect timeout; any failure → `[]` (sleep runs exactly as before).
- Weight `0.75 * (dwell + returns)/2`, dwell = log scale saturating at 1 h, returns = min(1, n/3); ceiling 0.75 keeps critical metacog and 0.8 crystallizations ahead.
- Arcs capped at k//3 of the replay via optional `source_caps` on `select_replay`; existing sources keep ≥2/3. Control arm sees the same ≤k//3 arcs.
- Sleep pressure, sleep reasons and the rest drive are unchanged (pinned by test). Arcs never start a sleep alone.
- `orion/schemas/dream_cycle.py`: `"arc"` added to `ReplaySourceKind` (readers checked: free-text column, generic rendering).

## Env/config changes
- Added `DREAM_REPLAY_ARC_CANDIDATES_ENABLED=true` (orion-dream). Local `.env` synced (`--all-keys orion-dream`, key outside default prefixes).

## Tests run
```text
services/orion-dream/tests + evals: 253 passed, 2 skipped (new test_replay_arc_candidates.py: 20)
tests/test_dream_hypotheses.py: 16 passed; metric lineage / env parity / definition drift: PASS
```
Live preview (window of sleep dc-3a5b86cbe76d, 10-12 04:04–13:45 UTC): 92 closed arcs → 20 candidates; replay mix vs that sleep's actual items: metacog 6, arc 4, compaction_request 2.

## Review findings fixed
Arc displacement wording + order test; control pool confound; connect timeout; weight tie vs crystallization (0.75 ceiling); guarded int parse; exact-id dedupe; tighter timeout test.

## Restart required
```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-dream up -d --build
```

## Risks / concerns
- Interoception `attention_returns` count sensor re-entries (up to 53/day), so they can outrank attention/curiosity arcs. Used as defined.
- Near (non-overlapping) edits to PR #2654 in dream cycle/main/settings.
- Candidates carry `arc_id` as the citation handle, not evidence id lists.
