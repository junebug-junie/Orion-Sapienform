# fix(substrate-runtime): walkway day links only the people the rollup saw

## Summary

- Orion's 10-20 day letter said "the rollup wiring joins on name string, so both Esme twins and both Adeline twins got identical bindings." That was true.
- The "seen on <day>" loop added every person Juniper had typed the same name for, whether or not the camera saw them that day.
- Now it links only the people the day's rollup actually lists. A copied name links to the typed people of the same name in its "same person" group (the group the writer copied the name along), never to everyone with that name.
- Regression tests: two people with one name keep their own days; a copy reached through a two-hop same-person group links, a second typed "Sam" outside the group does not (real Postgres).

## Outcome moved

Wrong "seen on" links stop being written. Live graph today holds 4 wrong links out of 22 (see below), including one that is impossible: an Esme cluster linked to 10-15, a day before that cluster first existed.

## Current architecture

`services/orion-substrate-runtime/app/walkway_seen_loop.py` reads complete walkway day rollups (`vision_rollup`, grain=day) every 30 min. For each name in `rollup.named` it unioned the sighted ids with `SELECT individual_id FROM vision_individual WHERE lower(label)=lower(:l) AND NOT label_copied`, then wrote one `observed_in` edge per id with an entity node.

## Architecture touched

- `walkway_seen_loop.py`: `TYPED_SQL` (name join) replaced by `COPIED_SQL` (copied row -> typed rows of the same name in its recursive `vision_individual_pair_verdict` 'same' group, the CTE `_propagate_label` uses). One query per day row instead of one per name.
- No schema, bus, or env change.

## Files changed

- `services/orion-substrate-runtime/app/walkway_seen_loop.py`: the fix.
- `services/orion-substrate-runtime/tests/test_walkway_seen_loop.py`: twins regression test; fake DB refuses a name-only lookup; Postgres test adds a same-person group and a second typed "Sam".

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: a day links only individuals the rollup saw (copied names resolve through their same-person group).
- Compatibility notes: none. Edge ids are unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-substrate-runtime && PYTHONPATH=<worktree> pytest tests/test_walkway_seen_loop.py -q
  7 passed, 2 skipped
  with ORION_TEST_PG_URI (throwaway postgres:16 container): 8 passed, 1 skipped
  old walkway_seen_loop.py against the new tests: 5 failed
  (the 1 skip is the real-FalkorDB lane, ORION_TEST_FALKOR_URI unset)
```

## Evals run

```text
No eval harness for walkway_seen. Live read-only audit instead: every walkway "seen on" edge in
orion_substrate compared with the rollup's sighted ids -> 18 KEEP, 4 WRONG.
```

## Docker/build/smoke checks

```text
Not built. Pure-Python change in an existing loop. UNVERIFIED live until deploy; check the log line
"walkway_seen linked day=... people=..." after restart.
```

## Live wrong links (to remove, needs Juniper approval)

| day | name | cluster | edge_id |
|---|---|---|---|
| 2026-10-21 | Maeve | 62cb5fcc | sub-edge-walkway-seen-16d62f3be564c849169bfd4d |
| 2026-10-19 | Juniper (was Teddy) | f5242d40 | sub-edge-walkway-seen-7e2773010f74047fa8631464 |
| 2026-10-19 | Adeline | f5694d84 | sub-edge-walkway-seen-edd5842eec0d9f661e60749c |
| 2026-10-15 | Esme | 4f88265e | sub-edge-walkway-seen-83f3ce6fec9dba9289657b2f |

The substrate store has no edge-delete API, so removal is one Cypher statement on Falkor before the deploy restart (the restart re-hydrates from Falkor).

## Review findings fixed

- Finding (HIGH): a "same person?" yes with no typed name copies the label with the verdict's ask id, so the first version's `label_ask_id` join dropped that copy's link (`vision_individuals.py:2057`); a copy of a copy needs two hops.
  - Fix: resolve copies through the recursive same-person group instead of the ask id.
  - Evidence: Postgres test with copy -> mid -> typed (two hops, no shared ask) links `ind-sam` only; 8 passed.
- Finding (MEDIUM): after a split or rename, an old copy can match no typed source and lose its days.
  - Fix: none; fails closed (missing link, never a wrong one). 0 copied rows live.
- Finding (LOW): one group with two typed rows of the same name links both.
  - Fix: none; Juniper said they are the same person, so both are correct.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a copied name whose group no longer holds a typed row of that name (split, rename) no longer links. It fails closed: a missing link, not a wrong one.
- Mitigation: none needed today; 0 copied rows exist live.

## PR link

(pending)
