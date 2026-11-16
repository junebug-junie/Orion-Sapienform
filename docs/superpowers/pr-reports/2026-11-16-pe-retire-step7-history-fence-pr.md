# PE suite rebuild, retire step 7: fence or purge history for every retired node

## Summary

- One live leak found and closed. Orion's "what my body signal tends to lead to" rollup (orion-equilibrium-service) still computed results for the four retired prediction-error nodes, every hour. It read their old hour rows, which are never pruned. Since the retirements it wrote 1,164 rows for them. It also counted them in the multiple-comparison correction, so every live signal needed stronger evidence than it should (25 comparisons instead of 10).
- The same rollup kept the walkway camera, which became a world source on 2026-10-21. It is now skipped too.
- The episode gate (the metacog trigger that opens and closes "my body signal is unusual" episodes) now closes a retired node's episode on the next poll, with `close_reason=retired`. Before, it waited one hour and then closed it as `went_silent`. It also never opens one for a retired node.
- `node:substrate.harness_closure` joins the shared retired set. Its producer was removed on 11-11, but it was only fenced indirectly, by "no glossary entry = not a measurement".
- Live inventory of every store, below. Nothing else is still read as a current reading. Everything else is durable history or ages out on its own within about 4 days.

## Outcome moved

- Rows written for retired nodes per hour: 12 (4 nodes x 3 outcomes), plus 3 for the walkway. After deploy: 0.
- Bonferroni family per hourly computation: 25 before, 10 after. This is replayed on all 957 persisted `self_signal_hour` rows with the new reducer: 4 signals, 12 associations, `n_comparisons=10`, and no verdict turns into a clause (so nothing new reaches Orion's prompt today).
- A tracked episode on a retired node: before, it closed after 3600 s as `went_silent` (worded "went silent, not recovered"). After, it closes on the next 30 s poll as `retired`.

## Current architecture

Shared retired set: `RETIRED_PREDICTION_ERROR_NODE_IDS` in `orion/substrate/prediction_error_freshness.py`. The 7-day self-reading cache (`orion/attention/pe_history_cache.NodePeHistoryCache`) drops retired rows. It is used by attention-runtime, orion-thought self-readings, and the equilibrium episode gate and rollup. The rollup's `body_signal_ids` skipped retired nodes for **new** hours only. `associate_all` recomputed over **every persisted hour** with no filter.

## Live inventory (2026-11-16, read-only)

Retired ids: bus_synaptic, route, chat, codebase, harness_closure.

| Place | Rows naming a retired node | Live reader? | Decision |
|---|---|---|---|
| `substrate_node_prediction_error_history` | 3,295 (bus_synaptic 1,936, chat 724, codebase 428, harness_closure 110, route 97); newest 11-13 08:24 | Every generic reader goes through `NodePeHistoryCache`, which skips the retired set (harness_closure added here). The strain-news reader is per-node (strain only). | **Fenced.** Ages out through the 7-day prune by about 11-20. No purge. |
| `substrate_node_prediction_error_baseline` | 0 | n/a | Nothing to do |
| `self_signal_outcome_association` | 9,288 (2,322 per node, 4 nodes); **still written hourly, last 04:10 today**; 1,164 written after each node's last real reading | Producer: `associate_all` (**not fenced, now fixed**). Readers: the orion-thought self-readings cue and the episode gate's `renderable`, both through `held_latest_by_pair` (now refuses retired rows; before, only indirectly fenced because the cache had no retired node to look up). | **Fixed + fenced.** Rows computed before retirement: keep (durable audit). The 1,164 written after retirement are artifacts of the leak: **purge, snapshot first** (commands below). |
| `self_signal_hour` | 889 of 957 rows carry a retired signal entry; newest 11-13 07:00 | Only `associate_all` (now fenced) and the truth script | **Keep** (durable history, never pruned by design) |
| Redis `equilibrium:self_signal_episodes:v1` | 0 tracked episodes (`{"episodes": {}}`) | Equilibrium gate | **Nothing open.** No state fix needed. Code added so a future retirement closes explicitly. |
| `metacog_trigger` (self_signal_episode rows) | bus_synaptic 138 open / 140 close, chat 33/33, codebase 27/28, route 26/26; **0 opens without a close** | History only | **Keep** (audit) |
| `substrate_attention_self_model` | 332 rows (in `broadcast_attended_node_ids`); newest 11-13 08:27 | Hub/equilibrium read the latest row or recent windows; the latest is clean | **Keep.** True record of what was attended. 168 h retention prunes it about 11-20. |
| `substrate_endogenous_curiosity_candidates` | 4,639 rows; newest 11-13 08:32 | Readers take the latest row or marker-matched rows; the latest is clean | **Keep** (retention prunes) |
| `substrate_attention_broadcast_log` | 2,971 rows; newest 11-13 08:29 | The act ledger looks rows up by time or id for acts that happened then | **Keep** (audit) |
| `substrate_field_state` | 7,296 rows, all before 11-13 08:26; latest tick clean | The field digester prunes `RETIRED_PSEUDO_NODES` on reconcile (harness_closure added) | **Fenced.** The 3-day retention clears it. |
| `substrate_prediction_raw_reading` (chat domain) | 1,879; no writer since 11-03 | **No code reader at all** (dead table) | Keep for now; issue #3505 |
| FalkorDB PE nodes (`orion_substrate`, `orion_substrate_self`, archive) | 0 nodes with `node_id` = a retired id | n/a | Nothing to do |
| FalkorDB topic-foundry `Entity` nodes whose *label text* is a retired id | 3 (`node:substrate.codebase` in orion_substrate; `substrate.bus_synaptic` in self + archive) | Not PE state (no `prediction_error`; node_id `sub-entity-topicfoundry-*`) | **Keep** (they are about the word) |
| `want_ledger`, `act_downstream_use`, `grammar_acts`, `substrate_self_monitor_history`, `substrate_signal_control_cells`, `hub_attention_minute`, `attention_loop_outcome`, `temporal_self_*` | 0 | n/a | Nothing to do |

## Architecture touched

- orion-equilibrium-service: self_signal_outcome rollup (through the shared reducer) and the self_signal_episode gate.
- Shared: `orion/attention/self_signal_outcomes.py`, `orion/metacog/self_signal_episode.py`, `orion/substrate/prediction_error_freshness.py`.
- orion-field-digester: `RETIRED_PSEUDO_NODES` (+harness_closure).

## Files changed

- `tests/test_field_channel_glossary.py`, `.github/workflows/self-readings-tests.yml`: glossary count 60 -> 59 (codebase retirement) and a CI step so it cannot rot again (Fixes #3506).

- `orion/attention/self_signal_outcomes.py`: `associate_all` skips retired and non-body signals; `held_latest_by_pair` refuses retired rows.
- `orion/metacog/self_signal_episode.py`: `close_retired()` and the `retired` close reason.
- `services/orion-equilibrium-service/app/self_signal_episode_gate.py`: routes a retired node's observation to `close_retired`, never `step_signal`.
- `orion/substrate/prediction_error_freshness.py`: adds `node:substrate.harness_closure`.
- `services/orion-field-digester/app/tensor/channels.py`: adds harness_closure to `RETIRED_PSEUDO_NODES` (the existing agreement test requires it).
- `services/orion-equilibrium-service/README.md`: documents both behaviours.
- Tests: `tests/test_self_signal_outcomes.py` (+3), `services/orion-equilibrium-service/tests/test_self_signal_episode_gate.py` (+3), `tests/test_retired_pe_step5_readers.py` (exact set), `services/orion-equilibrium-service/evals/test_self_signal_episode_replay_eval.py` (placeholder check replays harness_closure explicitly).

## Schema / bus / API changes

- Added: none. `close_reason` is a free-form string in `MetacogTriggerV1.upstream`; `retired` is a new value.
- Behaviour changed: there are fewer association rows per hour (no retired nodes or world sources), and `n_comparisons` drops from 25 to 10.
- Compatibility: none needed.

## Env/config changes

- None. No `.env_example` changed.

## Tests run

```text
tests/test_self_signal_outcomes.py tests/test_retired_pe_step5_readers.py tests/test_attention_self_readings.py orion/metacog/tests/test_self_signal_episode.py tests/test_pe_alarm_refile.py -> 130+ passed
tests/test_metric_* tests/test_trace_unified_turn.py tests/test_attention_world_first.py tests/test_attention_candidate_society_of_mind.py orion/substrate/tests/test_endogenous_curiosity.py orion/attention -> passed
tests/test_field_channel_glossary.py -> 18 passed (was failing on main: 59 != 60; fixed here, Fixes #3506)
services/orion-equilibrium-service tests+evals -> 288 passed
services/orion-field-digester tests (excl. test_heartbeat_chassis.py, env FileNotFoundError on main too) -> 297 passed, 6 skipped
services/orion-substrate-runtime test_retired_pe_step5/turn_referent/post_turn_closure -> 15 passed (cursor/quarantine tests fail identically on main)
services/orion-thought reverie/stance tests + evals -> 116 passed, 3 skipped
services/orion-attention-runtime world_first + goal_provenance -> 31 passed
```

## Evals run

```text
services/orion-equilibrium-service/evals -> passed (episode replay, outcome replay)
services/orion-thought/evals -> 28 passed
Live replay (read-only): new associate_all over 957 persisted self_signal_hour rows -> 4 signals, 12 associations, n_comparisons=10, no clause
```

## Docker/build/smoke checks

```text
Not run. Python-only change, with no dependency, port or compose change.
```

## Review findings fixed

The review ran in a subagent on commit 4b837d364. It found one "should" and three nits, and all four are fixed.

- Finding (should): the metacog evidence line rendered a `retired` close as "settled (retired): readings back within its usual range", which claims a recovery that never happened.
  - Fix: `orion/metacog/evidence_map.py` now has an explicit `retired` branch: "its signal was retired; no further readings, NOT a recovery". The severity stays nominal.
  - Evidence: `test_retired_close_is_not_rendered_as_a_recovery`.
- Finding (nit): `scripts/analysis/replay_self_signal_episodes.py --from-db` still replayed retired nodes' lingering rows, so it did not match the live gate.
  - Fix: `build_windows` drops retired ids, like `pe_history_cache`. The eval's hand filter is removed, so the replay is the only fence.
  - Evidence: `test_retired_nodes_are_not_replayed_at_all`, which runs on the unfiltered fixture.
- Finding (nit): there was no direct unit test for `close_retired` (None, pending, silent, confirmed).
  - Fix: added `test_close_retired_writes_a_row_only_for_a_confirmed_non_silent_episode`.
- Finding (nit): a confirmed episode whose open row was never published, on a node that then retires, was not tested.
  - Fix: added `test_confirmed_but_unpublished_episode_on_a_retired_node_drops_without_a_row`.

Found by me after the review: the walkway camera (a world source since 10-21) had the same hourly association leak. It is now skipped too, with a test and the live replay above.

## Restart required

Rebuild through the normal post-pull path (`scripts/rebuild_services_from_git_diff.sh` from clean primary main). The services that matter:

```bash
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Post-deploy commands (production writes: Juniper runs these, after equilibrium is redeployed)

1. Verify that the leak stopped. This is read-only. Expected: only biometrics, cabinet, execution and strain, with `n_comparisons=10`.

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select signal_id, max(n_comparisons), count(*) from self_signal_outcome_association where computed_at > now() - interval '2 hours' group by 1 order by 1"
```

2. Snapshot the 1,164 post-retirement association rows (CSV, read-only):

```bash
mkdir -p /tmp/pe-retire-step7 && docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "\copy (select a.* from self_signal_outcome_association a join (values ('node:substrate.bus_synaptic','2026-11-11 17:45:05+00'::timestamptz),('node:substrate.route','2026-11-11 15:44:54+00'),('node:substrate.chat','2026-11-13 03:54:23+00'),('node:substrate.codebase','2026-11-13 08:24:38+00')) l(sid,last_at) on a.signal_id=l.sid and a.computed_at > l.last_at) to stdout csv header" > /tmp/pe-retire-step7/assoc_post_retirement.csv && wc -l /tmp/pe-retire-step7/assoc_post_retirement.csv
```

3. Purge them. Run only after step 2 shows 1,165 lines (header + 1,164) and step 1 shows the leak stopped:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "delete from self_signal_outcome_association a using (values ('node:substrate.bus_synaptic','2026-11-11 17:45:05+00'::timestamptz),('node:substrate.route','2026-11-11 15:44:54+00'),('node:substrate.chat','2026-11-13 03:54:23+00'),('node:substrate.codebase','2026-11-13 08:24:38+00')) l(sid,last_at) where a.signal_id=l.sid and a.computed_at > l.last_at"
```

Expected: `DELETE 1164`, plus any rows written between now and the deploy. The cutoffs are each node's last real reading in `substrate_node_prediction_error_history`. Rows before them are kept as audit history.

No Redis state fix is needed: `equilibrium:self_signal_episodes:v1` tracks 0 episodes, and every retired-node open row has its close row.

## Risks / concerns

- Severity: low. Concern: live signals' association intervals get narrower (family 25 to 10), so a clause can now render with less evidence than before. Mitigation: 10 is the correct family size. The live replay shows no new clause today, and the 6-hour hold rule still applies.
- Severity: low. Concern: harness_closure history rows (110, newest 11-11) now disappear from the cache. Mitigation: they were already judged "unknown" placeholders and never opened or seeded anything.
- Issues opened: #3505 (dead `substrate_prediction_raw_reading` table), #3506 (glossary count test failing on main), #3507 (verify-live + needs-juniper: leak stopped after deploy, then run the purge).
- Issues closed: Fixes #3506. The glossary has 59 entries because the `node:substrate.codebase` prediction_error entry left with the node (2005c2b39, #3314); `node:substrate.chat` was swapped 1:1 for `node:prediction.chat_turn` the same day. The test now expects 59 and asserts that codebase is absent. It also now runs in `.github/workflows/self-readings-tests.yml`, which already triggers on the glossary YAML, plus paths for the test and `orion/field/channel_glossary.py`. Before this, no workflow ran it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3508
