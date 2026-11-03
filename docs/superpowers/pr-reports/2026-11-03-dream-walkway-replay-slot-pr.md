## Summary

- Orion's walkway-camera predictions now reach the dream replay. When at least one graded walkway prediction exists, one replay slot is held for the best one.
- "Best" means the heaviest item. A missed prediction (0.5) beats a met one (0.3). Ties keep the producer's order, which is newest first.
- `select_replay` gains a `source_floors` argument. A floor is applied before the weight ranking. A floor never goes above the source's ceiling and never goes above k.
- Walkway weights are unchanged. The lightest non-walkway item gives up its slot.
- The floor applies only when the replay has 6 or more slots. Live, the replay has 12 slots. A smaller replay does not hand a slot to the street ahead of a critical metacog item.

## Outcome moved

A read-only audit on 2026-11-03 found a failure. `vision_percept_expectation` holds 102 met and 64 missed walkway rows. 26 dream cycles since 10-29 logged `walkway candidates>=1`. But `dream_replay_item` has 0 rows with `source_kind='walkway'`, ever. Each replay filled all 12 slots with items at weight 0.5 or higher: metacog 0.6-1.0, arc 0.52-0.71, resonance 1.0, crystallization 0.66, compaction 0.5. `walkway_replay_slots` was only a ceiling. With this patch, a cycle with any walkway candidate replays exactly one walkway item, and up to 2 when they outweigh other items.

## Current architecture

`app/cycle.py` reads arc and walkway candidates after the pressure read. It then calls `select_replay(candidates + arcs + walkway, 12, source_caps={"arc": 4, "walkway": 2})`. `select_replay` took the top k by weight, with a per-source cap. Nothing guaranteed a low-weight source any slot. The weights came from the 10-15 walkway replay patch (commit 667a32d5c). They were declared heuristics, set below critical metacog, high-salience crystallization and arcs, with no floor.

## Architecture touched

orion-dream only: replay selection (`app/replay.py`) and the cycle call site (`app/cycle.py`). No change to schema, bus, API or env.

## Files changed

- `services/orion-dream/app/replay.py`: `select_replay(..., source_floors=None)` and `walkway_replay_floor(k)`. The walkway section has a comment on the floor's provenance.
- `services/orion-dream/app/cycle.py`: passes `source_floors={"walkway": walkway_replay_floor(DREAM_REPLAY_MAX)}`.
- `services/orion-dream/README.md`: the walkway replay section documents the floor.
- `services/orion-dream/tests/test_replay_walkway_floor.py`: new regression tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the replay now includes 1 walkway item when walkway candidates exist and `DREAM_REPLAY_MAX >= 6`. With no walkway candidates, the output is identical to before.
- Compatibility notes: `source_floors` is an optional keyword. Existing callers do not change.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, no template change
- skipped keys requiring operator action: none

The existing off switch still disables all walkway replay: `DREAM_REPLAY_WALKWAY_ENABLED=false`.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-dream/tests/test_replay_walkway_floor.py -q
  on main's app code: 7 failed (ImportError, plus the cycle test: assert 0 == 1 walkway items)
  on this branch: 8 passed
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-dream/tests -q
  338 passed, 9 skipped
```

New tests:
- 12 strong non-walkway items + 1 missed walkway item: exactly 1 walkway item is replayed, and the lightest other item drops out.
- A missed item beats a met item. Tied items keep the producer's order.
- No walkway candidates: the replay is identical with and without floors.
- The ceiling still holds: 8 heavy walkway items give 2. A floor above the cap is clipped. A floor is clipped to k.
- Small replays (k<6) reserve no floor.
- A full end-to-end cycle (`run_cycle_once`, all 12 replayed items at weight >= 0.5) replays 1 walkway item.

## Evals run

```text
None. orion-dream has no eval harness for replay selection. The selection is deterministic, so the gate tests above cover it. The real quality question is whether walkway items enter dream pairs and produce useful hypotheses. That is a follow-up check after deploy (see Risks).
```

## Docker/build/smoke checks

```text
Not run. Pure selection-logic change, no dependency, boot config or compose change. Not deployed (per task).
```

How to verify live after deploy. After the next completed dream cycle, run:

```sql
SELECT c.cycle_id, c.started_at, r.ref_id, r.weight, r.reason
FROM dream_replay_item r JOIN dream_cycle c USING (cycle_id)
WHERE r.source_kind = 'walkway'
ORDER BY c.started_at DESC LIMIT 5;
```

The check passes when a row appears for a cycle whose note records `walkway candidates>=1`. Also check whether walkway items reach dream pairs. Look at the cycle's hypotheses and confirm one cites a `walkway:` ref_id (column names depend on the hypothesis table).

## Review findings fixed

- Finding: when k is small, the floor takes over the replay. At k=1 the replay was only the walkway item, and a critical metacog item dropped out. At k=2..5 walkway took more than a sixth.
  - Fix: `walkway_replay_floor(k) = min(1, k // 6)`.
  - Evidence: `test_small_replay_reserves_no_walkway_floor`.
- Finding: the test name `..._regardless_of_input_order` claimed more than the test checks. Floor ties follow the producer's order by design.
  - Fix: renamed it to `test_floor_result_independent_of_non_walkway_order`.
  - Evidence: test file.
- Finding: floor ties trust the caller's order. A future source with unordered output would give an unstable pick.
  - Fix: the docstring now says to give a floor only to a source that orders its own output deterministically. `walkway_candidates` sorts by weight, then newest, then ref_id.
  - Evidence: `select_replay` docstring.
- Finding (noted, no code change): a replay slot does not guarantee a dream pair. `dream_pairs` takes 3 pairs ranked by `wa*wb*distance + bonus`, so a 0.5 walkway item can still lose to 1.0 metacog pairs.
  - Fix: listed under Risks as the next limit to check live.
- Finding (noted): when walkway items would have won on weight anyway, the tie-break between two tied walkway items is now newest-first, not lowest ref_id. Output with no walkway candidates is unchanged.
  - Fix: documented here.

## Restart required

Not done by this PR. After merge, the normal post-pull rebuild from the clean primary main checkout picks up the change:

```bash
scripts/safe_docker_build.sh orion-dream up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: the walkway item may get a replay row but never enter a dream pair, so "Orion dreamed about the street" stays UNVERIFIED.
  - Mitigation: check after deploy with the SQL above, plus a look at the hypotheses. If walkway never pairs, the next patch is pairing, not selection.
- Severity: low
  - Concern: one existing item (the lightest, often a 0.5 compaction request) leaves the replay when walkway candidates exist.
  - Mitigation: this is intended. It costs at most 1 slot in 12, and the old 2-slot ceiling still applies.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
