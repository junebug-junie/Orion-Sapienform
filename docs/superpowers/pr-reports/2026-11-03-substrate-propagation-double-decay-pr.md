# fix(substrate): decay a propagated activation once, not again by the receiver's age

## Summary

- When activation spread along an edge, the dynamics tick decayed it twice: once on the neighbour's clock (the neighbour's own value is already decayed), then again by the receiving node's full age. An old node reached by a fresh neighbour was muted. A walkway day node (observed at the start of the day, 6 h half-life) took a 0.61 sighting cluster as ~0.04 at 17:22.
- New propagation decay mode `source` (default): a signal from a node that decays is passed on as it stands this tick and is not decayed again. The day node now reads about cluster x 0.6 x 0.8 (~0.29) right after a sighting, then falls back to rest with the cluster.
- Two deliberate exceptions keep the old rule (decay once by the receiver's age):
  - A signal from a node with **no** half-life carries no decay. Found by the live replay: without this, never-decaying topic-foundry evidence (~0.5) pinned 724 old concepts at ~0.3 forever.
  - A receiver that decays more slowly than the signal (a 6 h hub feeding a 30-day concept). Found by code review: the tick saves the echo into the receiver's stored value, which then fades on the receiver's slow clock, so it would hold ~0.6 for weeks after the hub went quiet. The live graph has 0 such edges today; the rule guards future ones.
- Rollback: `SUBSTRATE_DYNAMICS_PROPAGATION_DECAY_MODE=receiver_age`. Ignored when `SUBSTRATE_DYNAMICS_DECAY_MODE=legacy` (legacy is untouched).
- Read-only live replay tool `scripts/replay_substrate_propagation_decay.py`, fixture eval `orion.substrate.evals.run_propagation_decay_eval`, and a mode flag on the person sighting replay eval.

## Proposal-mode fields (Juniper asked to implement directly)

- **Capability change:** an old node now hears a fresh neighbour. Spreading activation reflects how active the neighbour is now, not how old the receiver is. Concretely: a walkway day node echoes a sighting; an older topic concept co-occurring with a recent one is lifted by it.
- **Data touched:** only `signals.activation.activation` (and its `activation_decayed_at` stamp / dormancy flag) on cognitive substrate nodes, written by the existing dynamics tick through the existing compare-and-set write. No schema, bus, or table change. Fenced nodes (assertions, memory referents, person nodes) stay out of the cognitive view, as before.
- **Privacy boundary:** unchanged. The tick only reads and writes the substrate graph it already owned.
- **Trace that proves it worked:** a walkway day node's `activation` rises after a sighting of a linked cluster and falls back below 0.08 within ~a day (post-deploy check below). The tick's existing log line `substrate_dynamics_tick_completed activation_updates=...` keeps counting.
- **Dangerous failure mode:** flooding -- a hot hub or a never-decaying node holding many neighbours above dormancy forever, which would starve attention/curiosity budget. Measured below: 173 of 10,790 nodes rise (1.6%), none above 0.5, all on a decaying clock. The first version without the no-half-life exception failed this (724 pinned for good) and was changed before shipping.
- **Disable / rollback:** set `SUBSTRATE_DYNAMICS_PROPAGATION_DECAY_MODE=receiver_age` in `services/orion-substrate-runtime/.env` and restart orion-substrate-runtime. Values already lifted then decay on their own half-life; nothing needs to be unwound.

## Outcome moved

- Day node echo of a 0.61 cluster: **0.04-0.08 -> 0.29** (fixture), live replay `evidence:walkway_day:walkway:2026-11-01` **0.003 -> 0.226**, `2026-11-02` **0.056 -> 0.166**.
- Returns to rest: the day node reads 0.019 (< 0.08 dormancy) 24 h after the sighting; 300 hub spokes all < 0.08 24 h after the hub goes quiet.

## Metric gate (CLAUDE.md 0A)

1. **Provenance:** `orion/substrate/dynamics.py` `_compute_activations_from_source` -> `tick()` since_last branch (`new_activation = combined_activation`). Previously `_compute_activations` + `decay_activation(fresh, elapsed_seconds=age_seconds)` at the receiver.
2. **Independence:** no new metric; this changes how an existing value (activation) is computed. The propagated term is a monotone transform of the source's activation, which is the point of propagation.
3. **Theory anchor:** a half-life decay applied to a quantity must be applied once per unit of real time on one clock. The source's value already carries its decay up to now; applying the receiver's clock over the receiver's whole life is a second decay over a time span that has nothing to do with the signal. Same principle as the 2026-10-06 since_last fix (PR for `9d5541075`).
4. **Live sanity / rest check:** replay below, both horizons. Every rising node decays on a real clock; with no new input the day nodes and hub spokes return to < 0.08.
5. **Existing mechanism:** `decay_mode` / `normalize_decay_mode` (2026-10-06) -- this follows the same pattern (`propagation_decay_mode` / `normalize_propagation_decay_mode`, unknown value raises).
6. **Reversibility:** one env flag; no persisted schema.

## Live replay (read-only, FalkorDB `GRAPH.RO_QUERY`, snapshot 2026-11-03T22:05Z)

`python scripts/replay_substrate_propagation_decay.py --uri redis://127.0.0.1:6380 --save <pickle>` then `--load` for repeat runs. Graph: 11,456 nodes; cognitive view 10,790 nodes / 51,720 edges. No new producer writes during the replay, so this is "what changes right after deploy" and "does it rest on its own".

Final rule (after the review fix), same snapshot. The short horizon matches the pre-review run exactly, because the live graph has no fast-source-to-slow-receiver edges. The 48 h column is from the pre-review run of the same rule on the same snapshot.

| | short: 60 ticks x 30 s | 48 ticks x 1 h | 30 days: 30 ticks x 24 h |
|---|---|---|---|
| nodes changed (> 0.001) | 175 (173 concept, 2 evidence), all rises | 173 concept, all rises | 170 concept, all rises |
| p50 / p90 / p99 old | 0.4612 / 0.5695 / 0.6204 | 0.4606 / 0.5446 / 0.5934 | 0.4499 / 0.4706 / 0.4935 |
| p50 / p90 / p99 new | identical | identical | identical |
| count > 0.5 old -> new | 2228 -> 2228 | 1968 -> 1968 | 45 -> 45 |
| count > 0.2 old -> new | 9995 -> 10009 | 9984 -> 9993 | 9596 -> 9596 |
| count > 0.08 old -> new | 10398 -> 10468 | 10359 -> 10445 | 10003 -> 10055 |
| max | 0.9991 both | 0.9554 both | 0.6088 both |

Rest check at 30 days: the raised concepts fall from 0.2586 to 0.1324. That is one half-life, on the 30-day clock of the young sibling that lifts them. None are pinned, and none rise above 0.2 at 30 days that were not already above it in rollback mode. The walkway day nodes are back to the rollback values well before 30 days.

Top rises (short): `evidence:walkway_day:walkway:2026-11-01` 0.003 -> 0.214; topic-foundry concepts ~0.11 -> 0.259 ("Support and readiness", "Daydream and memory", "Conversation Continuity", "Orion chat context", ...) and ~0.08 -> 0.196; `evidence:walkway_day:walkway:2026-11-02` 0.053 -> 0.157. Traced example: "Orion chat context" (observed 2026-09-29, 30-day half-life) is lifted by a `co_occurs_with` sibling observed 2026-10-29 at 0.431 -> 0.431 x 0.6 = 0.259. Over 48 h it falls 0.2586 -> 0.2473, exactly the sibling's 30-day clock. Its never-decaying `supports` evidence (0.52) still reaches it only through the old age-decayed rule.

Dominance: the top 10 by activation are identical in both modes (seed concepts Orion / Juniper / Claude at 0.9996, `node:substrate.*` prediction-error nodes ~0.92). `node:athena` is not in the substrate graph. No new node above 0.5.

**Rejected first version (pure "never re-decay"), same snapshot:** 700 nodes rose after 30 min and **724 were still raised after 48 h with no input** (count > 0.2: 9984 -> 10584). Cause: 6,551 of 10,790 cognitive nodes have no half-life; topic-foundry evidence at ~0.5 never decays, so it pinned every supported concept at ~0.3 for good. That fails the rest check, so the no-half-life exception was added.

**Per-node propagated cap:** not added. Measured bound is already tight: max-not-sum means a node reads at most source x 0.6 x confidence; nothing new crossed 0.5; every raised node is on a decaying clock.

## Person sighting replay eval (14 days, live Postgres, read-only)

```text
--propagation-decay-mode source       : persons 11, clusters 60, active person-hours 845, top Amanda 23.43% (limit 25%), flood 0, fenced 0, not resting 0 -> PASS
--propagation-decay-mode receiver_age : identical -> PASS
synthetic (both modes)                : top share 20.59% -> PASS
```

The eval's graph has only clusters, fenced person nodes and a fenced referent (no cluster-to-cluster or day-node edges), so propagation mode cannot move it; identical results are expected. The #3075 figure was 22.7% on an earlier 14-day window.

## Current architecture

`SubstrateDynamicsEngine.tick()` (sole activation-decay writer since 2026-10-06) computes each node's value as `max(decay(fresh, age), decay(stored, since_last))`, where `fresh = max(seed, propagated)` and `propagated` comes from the neighbour's undecayed `max(seed, stored)`. So the propagated term was decayed by the receiver's age, on the receiver's half-life. Constructed in one production place: `services/orion-substrate-runtime/app/worker.py` `_dynamics_tick`.

## Architecture touched

- `orion/substrate` dynamics engine (no contract change).
- orion-substrate-runtime config: one new env key.

## Files changed

- `orion/substrate/activation.py`: `PROPAGATION_DECAY_MODE_*`, `normalize_propagation_decay_mode`.
- `orion/substrate/dynamics.py`: `propagation_decay_mode` ctor arg, `_own_activation_now`, `_compute_activations_from_source`, since_last branch uses it.
- `orion/substrate/tests/test_dynamics_propagation_decay.py`: new unit tests (day node, rest, no-neighbour identity, one hop from never-decaying source identity, legacy unchanged, hub bound, loop, bad mode).
- `orion/substrate/evals/run_propagation_decay_eval.py`: new fixture eval.
- `tests/test_cognitive_substrate_phase4_dynamics.py`: fresh decaying drive keeps an old neighbour awake; rollback mode mutes it.
- `services/orion-substrate-runtime/tests/test_person_sighting_refresh.py`: day-node pin updated to the new response (~cluster x 0.48) + a rollback pin.
- `services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py`: worker passes the mode; rollback passes through.
- `services/orion-substrate-runtime/{app/settings.py,app/worker.py,.env_example,docker-compose.yml,README.md}`: the new key.
- `services/orion-substrate-runtime/evals/run_person_sighting_replay_eval.py`: `--propagation-decay-mode`.
- `scripts/replay_substrate_propagation_decay.py`: read-only live replay.
- `tests/test_replay_substrate_propagation_decay.py`: smoke for the replay script on a pickled in-memory snapshot.
- `scripts/sync_local_env_from_example.py`: `SUBSTRATE_DYNAMICS_` prefix (the new key matched no prefix).
- `.github/workflows/substrate-neighborhood.yml`: new tests + eval.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: activation propagated from a decaying node is not decayed again by the receiver's age.
- Compatibility notes: no persisted format change; rollback is the env flag.

## Env/config changes

- Added keys: `SUBSTRATE_DYNAMICS_PROPAGATION_DECAY_MODE=source` (orion-substrate-runtime).
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`+SUBSTRATE_DYNAMICS_PROPAGATION_DECAY_MODE='source'`, after adding the `SUBSTRATE_DYNAMICS_` prefix).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/substrate/tests/test_dynamics_propagation_decay.py                     13 passed
tests/test_replay_substrate_propagation_decay.py                             1 passed
orion/substrate tests + tests/test_cognitive_substrate_phase{4,5}*.py        same failure set as origin/main (pre-existing:
  test_felt_state_lived_answers x1, test_felt_state_self_definition_lane x3, phase4 drive_goal_pressure + dynamic_pressure_reason)
services/orion-substrate-runtime/tests (+evals tests), PYTHONPATH=.:services/orion-substrate-runtime
                                                                             same 23-failure set as origin/main; none touch dynamics
services/orion-substrate-runtime/tests/test_person_sighting_refresh.py       22 passed, 1 skipped (real Falkor)
services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py          6 passed
services/orion-hub/tests/test_hub_does_not_decay_substrate_activation.py + test_world_pulse_read_assertion_links.py  28 passed
python scripts/check_env_template_parity.py                                  PASS (94 services)
```

## Evals run

```text
python -m orion.substrate.evals.run_propagation_decay_eval   passed (day 0.2922 vs rollback 0.0786; day 24 h 0.0186; spokes <= 0.5988 first, 0.0382 at 24 h; 30-day spokes of a 6 h hub identical in both modes over 14 days (0.378 -> 0.2735); never-decaying neighbour identical in both modes)
python -m orion.substrate.evals.run_decay_since_last_eval     passed
python -m orion.substrate.evals.run_dynamics_lost_update_eval passed
python -m orion.substrate.evals.run_complete_hydration_eval   passed
services/orion-substrate-runtime/evals/run_pe_writer_tick_lost_update_eval.py  passed
services/orion-substrate-runtime/evals/run_person_sighting_replay_eval.py --live --days 14  PASS (23.43%), both modes
scripts/replay_substrate_propagation_decay.py (live, read-only)  numbers above
```

## Docker/build/smoke checks

```text
Not run: no dependency, Dockerfile, or boot-path change beyond one env key read by settings.py. Not deployed (per task).
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on `6d0458443..2f655060f`.

- Finding (material): one short hub pulse could leave old slow-decaying neighbours held high for weeks. The tick persists the propagated value with a fresh stamp, and it then decays on the receiver's clock (measured: 6 h hub, 30-day spoke 60 days old -> spoke 0.60 at day 0, 0.434 at day 14, vs 0.15 / 0.109 in rollback mode). The docstring and README claimed it rests with the hub. The hub test only checked the 0.6 cap, not rest.
  - Fix: a decayed signal is taken as is only when the receiver's half-life is no slower than the signal's clock; otherwise the old rule applies to the signal's undecayed value. The frontier now carries the signal's clock and its undecayed value. The docstring and README are corrected.
  - Evidence: `test_a_fast_hub_cannot_latch_a_slow_neighbour_above_the_old_level` (source == receiver_age over 15 days, 30-day spokes); `test_a_busy_hub_lifts_same_clock_neighbours_..._then_they_rest` (spokes exactly hub x 0.6, then all < 0.08 at 24 h); the eval's new slow-hub fixture (source == receiver_age over 14 days). Live snapshot: 0 edges from a shorter-half-life source to a longer-half-life receiver.
- Finding (minor): the replay script had no smoke test and was not in the CI path filter.
  - Fix: `tests/test_replay_substrate_propagation_decay.py` (pickled in-memory snapshot -> `main()`); the script and its test were added to the workflow paths and test list.
- Finding (minor): `_stale_node_with_drive(drive_half_life)` was only ever called with 3600.
  - Fix: `test_a_never_decaying_drive_is_still_decayed_once_by_the_neighbours_age`.
- Finding (minor): `test_no_loop_sustains_itself` passes in either mode.
  - Fix: labelled as a bound check.
- Not changed (minor, recorded):
  - `source` ignores edges to targets missing from the cognitive view. `receiver_age` added such "ghost" targets; `cognitive_view` already drops those edges, so the case only arises for a dangling edge.
  - Tests that stub `_compute_activations` (`test_dynamics.py`, `test_dynamics_lost_update.py`) now exercise only the write/CAS path, not decay.
  - A node with no half-life that receives a signal keeps it forever. This is not new, and the received values are now no larger than before.
  - A naive-datetime `observed_at` in the recency computation is a pre-existing issue in both modes.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Only orion-substrate-runtime runs the dynamics tick, so it is the one restart that changes behaviour. The post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh --list-only`) also lists every service that bakes `orion/substrate` into its image: orion-attention-runtime, orion-cortex-exec, orion-cortex-orch, orion-durable-runs, orion-equilibrium-service, orion-execution-dispatch-runtime, orion-feedback-runtime, orion-field-digester, orion-graph-compression, orion-harness-governor, orion-hub, orion-memory-consolidation, orion-policy-runtime, orion-recall, orion-spark-concept-induction, orion-substrate-runtime, orion-thought, orion-world-pulse.

## Post-deploy live check (UNVERIFIED until run)

1. Pick today's walkway day node and a linked cluster, and read both after a sighting:
   `docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate "MATCH (c:SubstrateNode)-[e]->(d:SubstrateNode) WHERE d.node_id STARTS WITH 'evidence:walkway_day:walkway:' AND type(e)='observed_in' RETURN d.node_id, d.activation, c.node_id, c.activation ORDER BY d.node_id DESC LIMIT 10"`
2. Expect: within one tick (30 s) of a cluster's sighting refresh, its day node reads about cluster x 0.48 (was ~0.04-0.08), and below 0.08 within ~a day with no new sighting.
3. Tick log keeps reporting: `docker logs orion-athena-substrate-runtime --since 10m | grep substrate_dynamics_tick_completed`.

## Risks / concerns

- Severity: low. Concern: a person seen today also lifts their *past* day nodes (11-01 rose to 0.21 in the replay) because every day node they were in keeps its `observed_in` edge. Mitigation: bounded by the cluster's 6 h half-life; rests within ~a day. If unwanted, the walkway adapter could stop propagating to past day nodes (separate patch).
- Severity: low. Concern: walkway day nodes from before #3075 (10-15 .. 10-31) have no half-life and sit frozen at 0.3; this patch does not touch them. Mitigation: follow-up in the walkway day adapter / a one-off backfill.
- Severity: low. Concern: 6,551 of 10,790 cognitive nodes have no half-life, so the graph is already saturated (p50 0.46, 96% above dormancy). This patch keeps that out of propagation but does not fix it. Mitigation: separate follow-up on half-life defaults for topic-foundry evidence.
- Severity: low. Concern: multi-hop chains starting at a never-decaying node now decay at the first decaying hop and travel on decayed, where `receiver_age` decayed by each receiver's age. Mitigation: covered by tests; one-hop results are bit-for-bit identical.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3102

🤖 Generated with [Claude Code](https://claude.com/claude-code)
