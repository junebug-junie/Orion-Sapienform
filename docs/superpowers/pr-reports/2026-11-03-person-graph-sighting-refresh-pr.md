## Summary

When a named person is seen on the walkway, their camera cluster in Orion's graph now lights up and then fades over about a day. Before, every named cluster held the value it got when Juniper named it (0.625), forever.

- The person-graph pass (every 10 min) reads each linked cluster's `vision_individual.last_seen_at` and moves the cluster node's `observed_at` forward to it. The existing substrate tick does the rest. This pass never writes activation.
- Camera-cluster nodes and new walkway day nodes get a 6-hour decay half-life.
- The rename guard now compares against a separate `label_set_at` stamp (when the name was set), so renames still work after `observed_at` starts following sightings. The first refresh copies the old `observed_at` into it, once.
- `label_set_at` is a native FalkorDB property on entity nodes. Falkor keeps no free-form metadata on entity nodes (checked live: a cluster node has no `individual_id`/`label_ask_id` keys), so the spec's `metadata["label_set_at"]` alone would have vanished on the first restart.
- Kill switch `PERSON_SIGHTING_REFRESH_ENABLED=true` (shipped ON). New 14-day replay eval, run read-only against live data: PASS, top person 22.7% of active person-hours (limit 25%).

Design: `docs/superpowers/specs/2026-11-03-person-graph-sighting-refresh-design.md` (PR #3072, separate).

## Outcome moved

- Before: 48 of 128 cluster nodes read exactly 0.625 and none had a half-life. The value meant "was named", not "was seen". Day nodes read a flat 0.3.
- After (tested on the real dynamics engine): a sighting reads >= 0.5 ten minutes later, < 0.35 six hours later, <= 0.08 (dormant) twenty hours later. The old shape is asserted to fail that check.
- Failure mode closed: renames would have been refused forever once `observed_at` followed sightings (the old guard was `observed_at > labeled_at`). A test and a mutation check cover it.

## Current architecture

- `app/person_graph_loop.py` (10 min) wrote person nodes and `instance_of` claims, built missing cluster nodes and renamed stale ones. It never touched a cluster's `observed_at`.
- Cluster nodes came from `map_vision_individual_label_to_substrate` with no half-life, so `decay_activation` returned `max(floor, current)` unchanged: a ratchet.
- `orion/substrate/dynamics.py` `tick()` is the only activation owner (seed = 0.45 x recency(1 h from `observed_at`) + 0.35 x salience + 0.2 x pressure).

## Architecture touched

- Service: `orion-substrate-runtime` (person-graph loop, settings, env, compose, README, tests, eval).
- Shared: `orion/substrate/adapters/vision_individual.py` (half-life, `sighting_refreshed_entity`, `label_set_at`), `orion/substrate/falkor_codec.py` + `falkor_store.py` (native `label_set_at` entity property).
- No bus channel, schema registry, node kind or predicate change.

## Files changed

- `orion/substrate/adapters/vision_individual.py`: `SIGHTING_HALF_LIFE_SECONDS`, `LABEL_SET_AT_KEY`, `label_set_at()`, `sighting_refreshed_entity()`; half-life on new cluster and day nodes; naming and rename write `label_set_at`; a rename no longer moves `observed_at`.
- `services/orion-substrate-runtime/app/ask_answered_listener.py`: the rename write skips activation state.
- `orion/substrate/falkor_codec.py`: encode/decode `label_set_at` on entity nodes; omitted from SET when absent, so a writer that does not carry it never clears it.
- `orion/substrate/falkor_store.py`: `label_set_at` in `NATIVE_NODE_RETURN_FIELDS` (hydration and direct reads).
- `services/orion-substrate-runtime/app/person_graph_loop.py`: `LINKS_SQL` reads `v.last_seen_at`; `LinkRow.last_seen_at`; `_refresh_sightings` step; `NAMED_CLUSTERS_SQL` (half-life only for named clusters outside the live links); rename writes skip activation state; relabel guard on `label_set_at`; a missed ask row falls back to Postgres `labeled_at` before "now" (so a rebuild does not fake a sighting); trace `sightings_refreshed=N`; flag passthrough.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `PERSON_SIGHTING_REFRESH_ENABLED`.
- `scripts/sync_local_env_from_example.py`: `PERSON_GRAPH_` and `PERSON_SIGHTING_` prefixes. Without them the default sync visited the `.env` and silently added nothing.
- `config/field/field_channel_glossary.v1.yaml`: note that `node:substrate.perception_walkway` keeps 0 edges on purpose.
- `services/orion-substrate-runtime/tests/test_person_sighting_refresh.py` (new): unit, lifecycle, idempotence, rename, flag, real-Falkor round trip.
- `services/orion-substrate-runtime/evals/run_person_sighting_replay_eval.py` + `test_person_sighting_replay_eval.py` (new): 14-day domination replay.
- `.github/workflows/substrate-neighborhood.yml`: runs the person-graph and sighting tests and requires the real-Falkor test (none of the person-graph tests ran in CI before).

## Schema / bus / API changes

- Added: native FalkorDB property `label_set_at` (ISO string) on entity nodes. Only named camera clusters carry it.
- Removed: none.
- Renamed: none.
- Behavior changed: cluster node `observed_at` means "last seen" (was "named"). Clusters and new day nodes decay with a 6 h half-life. Existing day nodes the walkway loop re-applies within its 3-day lookback (after a restart) also pick up the half-life through `merge_node`'s existing-or-incoming rule; older day nodes are untouched.
- Compatibility notes: additive. Readers that do not know `label_set_at` ignore it. Nodes without it fall back to `observed_at` (the old guard).

## Env/config changes

- Added keys: `PERSON_SIGHTING_REFRESH_ENABLED=true` (orion-substrate-runtime).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, `orion-substrate-runtime: +PERSON_SIGHTING_REFRESH_ENABLED='true'` (after adding the prefix; the first run added nothing).
- skipped keys requiring operator action: none.

## Metric quality gate (cluster activation driven by `last_seen_at`)

1. Provenance: `vision_individual.last_seen_at` is set by orion-sql-writer `app/vision_individuals.py:587` (`max(last_seen_at, crop.observed_at)` per matched crop). Path: `_refresh_sightings` -> `observed_at` -> `dynamics.recency_score` -> `seed_activation`. All existing code except the refresh step.
2. Independence: separate from `perception_walkway` prediction error (identity match + Juniper's name vs whole-frame embedding distance). The surprise node keeps 0 edges. Day-node activation is a 1-hop copy of the cluster value, not separate evidence.
3. Theory anchor: recency-of-exposure accessibility, ACT-R base-level activation (Anderson & Schooler 1991), recency term only; frequency is left out on purpose.
4. Live sanity incl. rest: replay of the last 14 days of live walkway hour rollups through the real engine: every cluster with no sighting in the last 24 h is at rest (<= 0.08); a sighting raises a cluster for about 9 h above 0.2. Floor 0. The old 0.625 can never reach rest (asserted in tests).
5. Existing mechanism: recency seed, half-life decay, decay anchor and `last_seen_at` all existed. `StreetEventV1 known_person_arrived` (#2906) stays declined.
6. Reversibility: flag false + restart; values fade within a day. `label_set_at` is an additive property; no schema, manifest or training default.

Domination number vs the spec: the spec estimated ~13% from "hours with a sighting". The replay measures "hours above 0.2", and one sighting keeps a person above 0.2 for ~9 h, so the real number is higher: **22.7%** for the top person (person A in the spec's naming; 195 of 859 active person-hours), then 174 and 149. It passes the 25% limit, but with little margin. If a frequent visitor pushes it over, the lever is the half-life constant.

## Tests run

```text
PYTHONPATH=$PWD .venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_person_sighting_refresh.py \
  services/orion-substrate-runtime/evals/test_person_sighting_replay_eval.py \
  services/orion-substrate-runtime/tests/test_person_graph_loop.py \
  services/orion-substrate-runtime/tests/test_walkway_seen_loop.py \
  services/orion-substrate-runtime/tests/test_ask_answered_listener.py -q
-> 70 passed, 2 skipped (real-Falkor/PG tests skip without env)

ORION_TEST_FALKOR_URI=<throwaway falkordb:v4.18.11 container> pytest .../test_person_sighting_refresh.py -q -rs
-> 18 passed (incl. the real-Falkor round trip)

Full service suite (tests + evals, --ignore test_grammar_consumer_integration.py which fails collection on main too):
  branch:   17 failed, 575 passed, 4 skipped
  baseline (origin/main detached worktree): 17 failed, 556 passed, 4 skipped
  same 17 failures on both (diff of failure ids empty)

orion/substrate/tests + tests/test_field_channel_glossary.py + metric lineage/drift:
  5 failed, 1276 passed -- all 5 fail identically on origin/main
  (test_field_channel_glossary 48-channels, test_felt_state_lived_answers, 3x test_felt_state_self_definition_lane)

Mutation checks (each reverted):
  old observed_at guard            -> rename test fails
  no label_set_at backfill         -> 5 tests fail
  label_set_at not a return field  -> 2 tests fail (incl. real Falkor)

python scripts/check_env_template_parity.py -> PASS (94 services)
git diff --check -> clean
```

## Evals run

```text
python services/orion-substrate-runtime/evals/run_person_sighting_replay_eval.py            # synthetic
-> PASS, top_share 0.2059, flood_writes 0, fenced_changed [], not_resting []

POSTGRES_URI=... run_person_sighting_replay_eval.py --live --days 14   # READ ONLY transaction
-> PASS: 2026-10-20T09:00Z .. 2026-11-03T09:33Z, 2020 passes, 11 persons, 60 clusters,
   337 hours, 859 active person-hours, top_share 0.227 (limit 0.25),
   flood_writes 0, fenced person nodes unchanged, no cluster left above rest
```

## Docker/build/smoke checks

```text
No image build run. Change is Python + compose env only; no dependency or Dockerfile change.
Throwaway FalkorDB container (falkordb/falkordb:v4.18.11, 127.0.0.1:16399) used for the real-Falkor test; removed after.
Live FalkorDB and Postgres were read only (GRAPH.RO_QUERY, READ ONLY transaction).
```

## Post-deploy live checks (UNVERIFIED until run)

1. Within ~10 min of a real sighting, a cluster node's `observed_at` equals its `last_seen_at`:
   ```bash
   docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT individual_id, last_seen_at FROM vision_individual WHERE label IS NOT NULL ORDER BY last_seen_at DESC LIMIT 3"
   docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate "MATCH (n:SubstrateNode) WHERE n.node_id STARTS WITH 'entity:vision_individual:' RETURN n.label, n.observed_at, n.label_set_at, n.decay_half_life_seconds, n.activation ORDER BY n.observed_at DESC LIMIT 5"
   ```
2. Activation rises then decays: the freshest cluster reads > 0.5 within an hour of its sighting; a cluster unseen for 48 h reads < 0.08.
3. The count of cluster nodes at exactly 0.625 falls from 48 to 0 within one day:
   ```bash
   docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate "MATCH (n:SubstrateNode) WHERE n.node_id STARTS WITH 'entity:vision_individual:' RETURN sum(CASE WHEN n.activation = 0.625 THEN 1 ELSE 0 END), sum(CASE WHEN n.decay_half_life_seconds IS NULL THEN 1 ELSE 0 END)"
   ```
4. Log: `docker logs orion-athena-substrate-runtime 2>&1 | grep "person_graph synced" | tail -3` shows `sightings_refreshed=` and `half_lives_set=` (large on the first pass, then `half_lives_set=0` and a small `sightings_refreshed`).

## Review findings fixed

Code review ran in a subagent (no blockers, 4 should-fix, several nits). All should-fix items are fixed.

- Finding: walkway day nodes do not rise with a sighting (the spec's "capability change" says they do). The tick decays propagated input by the TARGET's own age, and a day node's `observed_at` is the day's start: a 17:22 sighting gives cluster 0.61, day node 0.039.
  - Fix: documented honestly in the README and here, and pinned by `test_day_node_no_longer_holds_a_frozen_value`. "Propagated input decays by the target's age" is filed below as found-not-fixed. The spec (#3072) still needs its capability-change sentence corrected.
  - Evidence: the test asserts the day node reads < 0.2 at the sighting and <= 0.08 at T+20 h.
- Finding: acceptance 3c (48 -> 0 nodes at 0.625) could not pass. 14 of 128 live cluster nodes are outside `LINKS_SQL` (10 linked as `vision_belonging`, 3 to `not_a_person`, 1 unlinked); 6 of them read 0.625 and had no way to rest.
  - Fix: new `NAMED_CLUSTERS_SQL` (every typed named `vision_individual`, 131 live rows). Those outside the live links get the half-life and `label_set_at` only, never an `observed_at` move. New trace field `half_lives_set`. `LINKS_SQL`, claims and `_ensure_cluster_nodes` are unchanged.
  - Evidence: `test_named_clusters_outside_the_live_links_get_the_half_life_only` (half-life set, observed_at kept, idempotent, rests).
- Finding: a rename still counted as a sighting (`relabelled_individual_entity` moved `observed_at` to the answer time), so a bulk rename would wake every renamed cluster.
  - Fix: renames keep `observed_at`; `label_set_at` carries the rename time. The loop's rename stamp is `max(ask time, Postgres labeled_at)`.
  - Evidence: `test_a_rename_is_not_a_sighting` (a bumped `observed_at` would seed ~0.55 and fail it).
- Finding: two checks could pass for the wrong reason. (a) The eval's fence check had no edges into fenced nodes. (b) The "refresh never writes activation" checks would pass with `ACTIVATION_STATE_KEYS` dropped from the skip set.
  - Fix: (a) the replay now adds an `instance_of` `semantic_projection` edge to each person node and a propagating `observed_in` edge to a `referent-*` node, and snapshots before the first tick. The replay also starts from today's legacy node shape (no half-life, no stamp). (b) `test_the_refresh_names_activation_state_as_not_owned` checks the skip set on the write itself.
  - Evidence: mutations. Unfencing `memory.referents` makes the eval fail on the referent; unfencing `person.record` makes it fail on the person nodes; dropping `ACTIVATION_STATE_KEYS` from the refresh write fails (b).
- Nit fixed: the loop relabel and the listener relabel now also skip `ACTIVATION_STATE_KEYS` (the tick is the only activation owner).
- Nits noted, not changed: day nodes older than the walkway loop's 3-day lookback keep no half-life (spec question 2, left open); the spec file lives on PR #3072, which should merge first or alongside.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- Severity: low. Concern: the domination margin is thin (22.7% vs 25%). Mitigation: rerun the eval weekly; the half-life is one constant.
- Severity: low. Concern: walkway day nodes stop reading a flat 0.3 but do not light up "today" (their propagated echo is ~0.04, because the tick decays it by the day node's own age). Mitigation: the cluster carries the signal. Found, not fixed: propagated input decays by the target's age (`orion/substrate/dynamics.py` since_last branch), which mutes any old node reached by a fresh neighbor.
- Severity: low. Concern: day nodes older than the 3-day walkway lookback keep no half-life and stay at 0.3 (spec missing-question 2). Mitigation: a one-off backfill under the section 14 protocol, if Juniper wants it.
- Severity: low. Concern: on the first tick after deploy, a frozen cluster whose decay stamp dates from its birth tick drops straight to rest (its stamp is days old once it has a half-life). That is the intended end state, reached in one tick instead of over a day.
- Severity: info. Concern: the spec lives on PR #3072; this branch cites it. Mitigation: merge #3072 first or together.
- Severity: low. Concern: a person with many clusters (one has 26) can wake many nodes at once. The eval caps by person, not by cluster. Mitigation: domination eval.
- Found, not fixed (spec non-goal): ~6,681 other entity/evidence nodes with activation > 0 and no half-life (the graph-wide ratchet).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
