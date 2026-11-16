## Summary

- A tendency built from "I turned down every option" now says which lane it came from. "no move, 7 of 7" becomes "Prime waits instead of acting, 7 of 7". "none of these, 3 of 3" becomes "Turns down every link in a reading, 3 of 3".
- A decline only joins a live tendency whose choices all came from the same lane. A Prime decline and a reading decline now found two tendencies.
- Only matching is lane-bound. A decline still counts against an opposite tendency in another lane. A tendency Juniper said No to still absorbs declines from every lane it holds.
- The relabel planner uses the same label rule. It flags any node that holds declines from more than one lane (`decline_lanes:<a>+<b>`) and does not split it.

## Outcome moved

Decline tendencies had labels with no place: the bare "none" phrase plus a count. They now name the lane. New declines from different lanes no longer fold together just because "no move" and "none of these" embed close together.

## Current architecture

`orion/personality/graph.py` folds each choice into a direction (chosen minus mean of declined). For a decline, the chosen side is the lane's "none" phrase (`DECLINE_TEXT`). The label was `pair_label(chosen, declined)`, which gave "no move" for 3+ declined options or "say nothing over X" for 1-2. Matching was direction-only across all lanes.

## Architecture touched

- `orion/personality/graph.py`: `DECLINE_LABEL`, `decline_label`, `choice_excerpt`; decline-only lane filter on matching in `_ingest`.
- `orion/personality/relabel.py`: excerpts remade with `choice_excerpt`; `decline_lanes` anomaly.
- `services/orion-durable-runs/app/personality_driver.py`: evidence excerpt uses `choice_excerpt`.

## Files changed

- `orion/personality/graph.py`: decline label table, `choice_excerpt`, lane-bound matching for declines.
- `orion/personality/relabel.py`: same label rule in the replay; mixed-lane decline anomaly.
- `services/orion-durable-runs/app/personality_driver.py`: uses `choice_excerpt`.
- `orion/personality/tests/test_graph.py`, `orion/personality/tests/test_relabel.py`, `services/orion-durable-runs/tests/test_personality_loop.py`: tests.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: label text of decline tendencies. Decline matching is lane-bound. A decline with 1-2 alternatives no longer reads "say nothing over X"; it reads the lane phrase alone.
- Compatibility notes: no schema or column change. The direction (embedding) of a decline is unchanged.

## Env/config changes

- None. `.env_example` not touched; no sync needed.

## Tests run

```text
personality CI set (orion/personality/tests + scripts/analysis personality tests + tests/test_personality_*) + orion/prime/tests + orion/physiology/tests: 830 passed, 1 skipped
services/orion-durable-runs tests -k "personality or relabel": 76 passed, 16 skipped
Mutation: removing the lane filter fails the 2 new decline graph tests; reverting the review fixes fails the 2 new opposition / Juniper-No tests.
```

## Evals run

```text
Live dry run (read-only, no writes): scripts/personality_relabel_choice_nodes.py
  155 non-retired choice nodes, 2 labels change, 0 anomalies
  disp-3ed41b8561920c6c: 'no move, 7 of 7'        -> 'Prime waits instead of acting, 7 of 7'
  disp-4ab000eed60b89cf: 'none of these, 3 of 3'  -> 'Turns down every link in a reading, 3 of 3'
  Files: /tmp/personality-relabel/20261116T053132Z-dry/
Lane mix (read-only SQL over every node holding any declined_all choice, all statuses):
  disp-3ed41b8561920c6c: prime declines 7, nothing else
  disp-4ab000eed60b89cf: reading declines 3, nothing else
  No existing decline node mixes lanes.
```

## Docker/build/smoke checks

```text
Not run. No dependency, compose, or env change. Not deployed (per task).
```

## Review findings fixed

- Finding: the lane filter also removed other-lane nodes from opposition, so a decline stopped counting against them.
  - Fix: filter only the match candidates; opposition uses all nodes.
  - Evidence: `test_a_decline_still_opposes_a_tendency_in_another_lane`.
- Finding: a mixed-lane tendency Juniper said No to could no longer absorb declines, so a rejected tendency could be re-founded.
  - Fix: a vetoed node absorbs a decline from any lane it holds.
  - Evidence: `test_a_juniper_no_tendency_still_absorbs_declines_from_any_lane_it_holds`.
- Finding: relabel gives a mixed-lane decline node one lane's phrase.
  - Fix: `decline_lanes:<a>+<b>` anomaly in the plan. Live: 0 such nodes.
  - Evidence: `test_a_node_holding_declines_from_two_lanes_is_flagged_not_split`.
- Not fixed (minor, by design): a non-decline choice from another lane can still join a decline tendency by direction. After that, later declines of that lane found a new tendency with the same label. Orion's MERGE verdict also ignores lanes. Both are noted in the module docstring.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Deploy order: merge, rebuild orion-durable-runs from clean main, then a fresh dry run of `scripts/personality_relabel_choice_nodes.py`, then `--apply --from <that dry run>` with Juniper's approval (production write).

## Risks / concerns

- Severity: low. Concern: a duplicate same-label decline tendency can appear after a cross-lane non-decline joins a decline node. Mitigation: documented; live has none today.
- Severity: low. Concern: live labels unchanged until the relabel is applied. UNVERIFIED live.
- Issues opened: none. Issues closed: none.

## PR link

See PR.
