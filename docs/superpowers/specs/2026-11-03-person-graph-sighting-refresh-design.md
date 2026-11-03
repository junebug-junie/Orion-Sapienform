# Person graph: a sighting refreshes the camera cluster (design, 2026-11-03)

Status: proposal. No code in this patch. Implementation waits for Juniper's OK
(CLAUDE.md proposal mode: this touches what Orion's graph treats as "recently
active").

## Arsonist summary

Orion's graph knows who the walkway people are, but it cannot tell who was seen
recently. Nothing that writes these nodes was ever built to refresh them on a
sighting. The writers are alive. The activation numbers you see are leftovers
from the moment each node was created.

- The 17 `person-*` nodes read activation 0.0 because they never had any. The
  dynamics tick skips them on purpose (they are identity-fenced). They did not
  "decay since 10-25".
- The 128 camera cluster nodes (`entity:vision_individual:*`) hold the value
  they got on their first tick after Juniper named them: 0.45 recency + 0.35 x
  0.5 salience = **0.625**. They have no decay half-life, so the tick can only
  raise that value, never lower it. It is a ratchet, not a reading.
- The walkway day nodes read a flat **0.3** = 0.625 x 0.6 attenuation x 0.8
  edge confidence. The cluster pushes this value into them through
  `observed_in`. The day nodes also have no half-life, so they are frozen too.

The fix is one small change. The person-graph loop already reads each linked
cluster's Postgres row every 10 min. That row already has
`vision_individual.last_seen_at`. The loop moves the cluster node's
`observed_at` to that time and gives the node a half-life. The existing tick
then does the rest: the cluster goes up when the person is seen and drops back
to rest within about a day. No new channel, schema, node kind or edge.

## Current architecture

- **Service:** `orion-substrate-runtime`. Three loops touch people:
  - `app/ask_answered_listener.py`: Juniper names a cluster, and the listener
    writes `entity:vision_individual:<sha>`
    (`orion/substrate/adapters/vision_individual.py`
    `map_vision_individual_label_to_substrate`). `observed_at` is the time of
    the answer. Salience is 0.5. No half-life.
  - `app/walkway_seen_loop.py` (30 min, `WALKWAY_SEEN_ENABLED`): reads
    *complete* `vision_rollup` day rows and writes one
    `evidence:walkway_day:<stream>:<date>` node plus `observed_in` edges from
    each named cluster to it. The docstring says "Entity nodes are never
    created here". The loop never touches the cluster node itself.
  - `app/person_graph_loop.py` (10 min, `PERSON_GRAPH_ENABLED`): writes
    `person-*` nodes (producer `person.record`) and `instance_of` claims
    through the journal. It rewrites a person node only when its label or
    state changes. It builds or renames missing or stale clusters.
- **Activation:** `orion/substrate/dynamics.py` `tick()` is the only owner of
  activation. Seed = 0.45 x recency (1 h horizon from `observed_at`) + 0.35 x
  salience + 0.2 x pressure. Propagation is up to 2 hops over
  `allowed_predicates`, which include `observed_in`. Decay needs
  `decay_half_life_seconds`. When it is `None`, `decay_activation` returns
  `max(floor, current)` unchanged (`orion/core/activation_decay.py`).
- **Eligibility:** `orion/substrate/eligibility.py`. Fenced producers
  (`memory.referents`, `person.record`) and assertions are not cognitive.
  `semantic_projection` edges (`instance_of`, 124 live) and
  `assertion_structure` edges (`assertion_object`, 110 live) do not propagate.
  So even a live cluster could not move a person node today.
- **Bus channels / schema registry:** no change needed.
- **Tests:** `services/orion-substrate-runtime/tests/test_person_graph_loop.py`
  and `test_walkway_seen_loop.py`.
- **Evals:** `services/orion-substrate-runtime/evals/` has no people or
  sighting eval.

### Live evidence (read-only, 2026-11-03 ~09:00 UTC)

FalkorDB `orion_substrate` (`GRAPH.RO_QUERY`):

| what | value |
|---|---|
| `person-*` nodes | 17 (13 provisional, 4 deprecated). All have `activation=0`, `recency_score=0`, and `activation_decayed_at` = their creation time (10-23..11-02). `observed_at` moved on 10-30 for some of them (label/state rewrites), and activation still did not move. |
| cluster nodes | 128. 48 read exactly 0.625, 8 read 0.175, and the rest read 0.613..0.6247 (first tick a few minutes after naming). None has `decay_half_life_seconds`. `observed_at` = naming day (10-15..11-02), not the sighting day. |
| `observed_in` | 449 total. 148 come from 71 clusters into 17 day nodes, all `legacy_unreviewed`, producer `orion-substrate-runtime.walkway_seen`. |
| day nodes | Every `evidence:walkway_day:walkway:*` node reads 0.3. |
| `instance_of` / `assertion_object` | 124 `semantic_projection` / 110 `assertion_structure`. |
| `node:substrate.perception_walkway` | concept, activation 0.92, 0 edges. |
| graph-wide | 119 of the 452 nodes with activation >= 0.6 are frozen clusters. 2,589 entity and 4,092 evidence nodes have activation > 0 and no half-life. |

Container `orion-athena-substrate-runtime` (started 11-03 04:02Z). Logs:
`walkway_seen linked day=2026-11-02 ... people=Bethany,Mail carrier edges=3
edges_created=3`, and `person_graph synced persons=19 ... nodes_written=0`
every pass. Both writers are alive.

Postgres (`conjourney`, read-only transaction): `vision_individual` has
`last_seen_at` and `sighting_count`. orion-sql-writer updates them on every
matched crop (`services/orion-sql-writer/app/vision_individuals.py:587`
`ind.last_seen_at = max(ind.last_seen_at, crop.observed_at)`). The freshest
walkway value was 11-03 05:43Z. 6 clusters were seen in the last 6 h.

**Verdict: never built, not a dead writer.** The two loops that run are
correct for what they were built to do. Nobody was ever assigned to write
"seen at" onto the node. The flat numbers come from a missing half-life, which
turns the one-time birth seed into a permanent value.

## Missing questions

1. Half-life. The proposal is 6 h. A sighting reads about 0.55..0.62. It
   crosses the dormancy threshold 0.08 about 18 h later. That means "seen
   today" fades by tomorrow. Is a day the right memory for "recently around"?
   (Small and reversible. It is a constant.)
2. Should the same pass also give the 17 existing day nodes a half-life (a
   one-off write to 17 nodes, under the section 14 backfill protocol)? Or
   should only new day nodes get one? Recommendation: do it in the same pass.
   It is idempotent.
3. The graph-wide ratchet (6,681 entity/evidence nodes with no half-life) is
   real and out of scope. File it as found-not-fixed. Do not fix it here.

## Proposed schema / API changes

None to schemas, channels or the registry. Behavior only:

- `orion/substrate/adapters/vision_individual.py`:
  - add `SIGHTING_HALF_LIFE_SECONDS = 6 * 3600`;
  - `map_vision_individual_label_to_substrate` and
    `map_walkway_day_seen_to_substrate` set
    `signals.activation.decay_half_life_seconds` to that value;
  - new pure `sighting_refreshed_entity(existing, last_seen_at)`. It returns
    `existing` with `temporal.observed_at = max(observed_at, last_seen_at)` and
    the half-life set. It returns `None` when nothing would change. It also
    sets `metadata["label_set_at"]` (see below) when that key is missing.
- `app/person_graph_loop.py`:
  - `LINKS_SQL` adds `v.last_seen_at`, and `LinkRow` gets `last_seen_at`;
  - new step `_refresh_sightings` runs after `_ensure_cluster_nodes`. For each
    active, non-copied link of a live person it calls
    `sighting_refreshed_entity` and `store.upsert_node(...,
    skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS | ACTIVATION_STATE_KEYS)`.
    The writer never sets activation. The tick stays the only owner.
  - **Relabel guard fix (required).** Today the guard is
    `existing.temporal.observed_at > link.labeled_at`, which means "the node
    is newer than Postgres, skip". Once `observed_at` follows sightings, that
    guard would refuse every later rename. It must compare against
    `metadata["label_set_at"]` instead. The listener and relabel write this key.
    It falls back to `observed_at` only when the key is missing. The refresh
    copies the old `observed_at` into `label_set_at` before it moves
    `observed_at` for the first time.
  - The trace line adds `sightings_refreshed=<n>`.
- Kill switch: `PERSON_SIGHTING_REFRESH_ENABLED=true` in `settings.py`,
  `.env_example`, `docker-compose.yml` and the README (shipped ON). When it is
  false, the step does not run. Then run
  `python scripts/sync_local_env_from_example.py`.

Person nodes are **not** written. They stay fenced and untouched by the tick.
A consumer that wants "when was person P last seen" reads the max over P's
`instance_of` clusters. The neighborhood read already walks
`semantic_projection`. Letting `instance_of` carry activation is the separate,
measured decision that `eligibility.py` reserves. This doc does not make it.

### The surprise node gets no edges

`node:substrate.perception_walkway` should stay at 0 edges. Its value is
whole-frame embedding distance (`services/orion-substrate-runtime/app/worker.py`
`_perception_walkway`). It does not know who was in the frame. A car or a
change in the light can raise it just as much as a person can. With an
outgoing edge, `_compute_pressures` would push its prediction-error pressure,
and `_compute_activations` would push its 0.92 activation, into every named
cluster or day node on each surprising frame. That is the flood this design
must avoid. Two consumers also read it as a fenced world source
(`endogenous_curiosity`, `world_first`). Edges would change what those
consumers see, and no gate has checked that. Write this decision into its
`config/field/field_channel_glossary.v1.yaml` entry, so a later patch does not
"fix" the 0.

## Metric quality gate (cluster activation driven by `last_seen_at`)

1. **Provenance.** `vision_individual.last_seen_at` comes from orion-sql-writer
   `vision_individuals.py:587`. It is the max `crop.observed_at` of crops
   matched to the cluster. The path to activation:
   `_refresh_sightings` sets `observed_at`, then `dynamics.recency_score`,
   then `seed_activation`. That path is all existing code.
2. **Independence.** It is independent of `perception_walkway`
   prediction_error. Both start from walkway frames, but this signal goes
   through detection, crop, identity match and Juniper's name. The surprise
   signal is the embedding distance of the whole frame. The two are not
   combined anywhere. The 0-edge rule above keeps them apart. The day-node
   `observed_in` edges already copy the cluster value 1 hop down (x 0.48).
   That copy is the same signal, not a second one. Do not count day-node
   activation as separate evidence.
3. **Theory anchor.** Recency-of-exposure accessibility: ACT-R base-level
   activation (Anderson & Schooler 1991). The chance that a memory item is
   needed falls off with the time since it was last encountered. The base-level
   term also includes frequency. This design leaves frequency out on purpose
   (see 4). The node value means "seen recently", not "seen often".
4. **Live sanity, including rest.** Last 14 days of walkway hour rollups (335
   hours): 47 of the last ~59 hours had no named person. So the signal is at
   rest most of the time, and with the half-life it can return to rest: 0.625
   reaches 0.08 in about 3 half-lives (~18 h). Its floor is 0. Today's frozen
   0.625 can never reach rest. The domination check compares two counts.
   Raw sighting counts: person A (most sightings) has 220 of ~731 linked sightings (30%). Hours
   with any sighting: the top person (person B) has 44 of 335 hours (13%), then
   person C 35 and person A 32. A sighting only resets the recency clock and never
   adds to it. So the most-seen person can at most stay "recently seen" during
   their own hours. Their activation cannot pile up. The live check after
   deploy is UNVERIFIED until it runs (acceptance 3).
5. **Existing mechanism.** The parts already exist: the
   `observed_at`-to-recency seed, half-life decay, `activation_decayed_at`
   anchoring (`activation_decay_anchor` takes the max of the stamp and
   `observed_at`, so a sighting restarts decay from the sighting), and the
   `last_seen_at` column. Nothing new is added. One idea was rejected:
   `StreetEventV1 known_person_arrived` (#2906). It is about attention and was
   declined. See Non-goals.
6. **Reversibility.** Cheap. Set the flag to false and the nodes decay to rest
   within a day. `observed_at` moves forward, and `label_set_at` keeps the
   naming time. No schema, manifest or training default changes.

## Proposal-mode fields

- **Capability change.** When a named person is seen on the walkway, their
  camera clusters (and that day's walkway node, through the existing edge)
  read active for a few hours and then fade. Orion's graph-ranked readers can
  tell "B was here this afternoon" from "B was named two weeks ago".
  Those readers are brain-frame samples, the concept-region recall and the
  Hub workbench.
- **Data touched.** `observed_at`, `decay_half_life_seconds` and
  `metadata.label_set_at` on at most 128 `entity:vision_individual:*` nodes,
  plus the half-life on new (and, if approved, the 17 existing)
  `evidence:walkway_day:*` nodes. The writer reads Postgres only. It writes no
  Postgres and no person node.
- **Privacy boundary.** None between Juniper and Orion (standing decision).
  What leaves the graph: nothing new. The same nodes go to the same readers.
  Only the timing of their activation changes. Neighbors' names already live
  in these nodes.
- **Trace that proves it.** The `person_graph synced ... sightings_refreshed=N`
  log line, plus a Cypher read of one cluster before and after a known
  sighting (acceptance 2).
- **Dangerous failure modes.**
  - *Activation flooding:* each pass sets `observed_at` to the current time
    instead of to `last_seen_at`. Then every cluster reads 0.625 forever (the
    same bug as today, just renewed every pass). Acceptance 4 catches this.
  - *node:athena-style domination:* the most-seen person's clusters hold the
    top activation slots. Leaving frequency out bounds this to hours-seen. But
    person A has 26 clusters, so one sighting can wake many clusters at once if
    the identity match fires on all of them. Acceptance 5 caps this by person,
    not by cluster.
  - *Juniper's own clusters:* she has 13 named walkway clusters, linked to a
    `referent-*` (chat) node. They stay non-cognitive edges. No path may lift
    her referent node through this seam.
  - *Rename lockout:* this happens if the relabel guard is not changed in the
    same patch. A test guards it.
- **Disable / roll back.** Set `PERSON_SIGHTING_REFRESH_ENABLED=false` and
  restart `orion-substrate-runtime`. To undo the code, revert the patch. Values
  fade on their own. No data cleanup is needed.

## Files likely to touch

- `orion/substrate/adapters/vision_individual.py`: half-life constant,
  `sighting_refreshed_entity`, `label_set_at`
- `services/orion-substrate-runtime/app/person_graph_loop.py`: `LINKS_SQL`,
  `_refresh_sightings`, relabel guard, trace
- `services/orion-substrate-runtime/app/ask_answered_listener.py`: write
  `label_set_at` on name/rename
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`,
  `docker-compose.yml`, `README.md`: `PERSON_SIGHTING_REFRESH_ENABLED`
- `services/orion-substrate-runtime/tests/test_person_graph_loop.py`,
  `test_walkway_seen_loop.py`
- `services/orion-substrate-runtime/evals/run_person_sighting_replay_eval.py`
  (new)
- `config/field/field_channel_glossary.v1.yaml`: the perception_walkway
  0-edge note

## Non-goals

- No attention change. A known person's arrival does not win attention or
  reach world-first contests. Juniper declined that (#2906, closed 10-29).
  This patch changes graph activation only. The attention fence and
  `ATTENTION_MIN_BAND_BY_NODE_ID` are untouched.
- No person-node activation and no activation over `instance_of`.
- No frequency or sighting-count signal. No new node kind, predicate, channel
  or taxonomy.
- No edges on `node:substrate.perception_walkway`.
- No fix to the graph-wide no-half-life ratchet. It is filed separately.
- No top-down personality or tone use of these values. The personality arc
  stays bottom-up.

## Acceptance checks

1. Unit: `sighting_refreshed_entity` moves `observed_at` only forward. It sets
   the half-life and `label_set_at`, and returns `None` when nothing would
   change.
2. Lifecycle test with the real `SubstrateDynamicsEngine` (not just the math):
   a named cluster seen at T reads >= 0.5 at T+10 min, < 0.35 at T+6 h, and
   <= 0.08 (dormant) at T+20 h. A cluster never seen again reaches rest.
   Today's frozen-0.625 shape must fail this test.
3. Live read-only check after deploy (until then: UNVERIFIED). First, a cluster
   with `last_seen_at` in the last hour reads > 0.5. Second, a cluster not seen
   for 48 h reads < 0.08. Third, the count of cluster nodes at exactly 0.625
   falls from 48 to 0 within one day.
4. Flood guard: two passes with no new sightings leave `observed_at` byte-equal
   (`sightings_refreshed=0` on the second pass).
5. Domination eval (replay of 14 days of hour rollups through the engine): no
   single person holds > 25% of the person-hours above 0.2. Today's data
   predicts about 13%.
6. Rename regression: a cluster refreshed by a sighting can still be renamed
   by a newer `labeled_at`.
7. Fenced nodes: `person-*` and `referent-*` nodes stay byte-equal in
   activation across a full replay.

## Recommended next patch

`feat/person-sighting-refresh` makes one change in `orion-substrate-runtime`
plus the adapter. The person-graph loop's existing 10-minute pass already
reads each linked cluster's Postgres row. It adds that row's `last_seen_at`.
It moves the cluster node's `observed_at` forward to it, gives the cluster
(and new walkway day nodes) a 6-hour half-life, and never writes activation
itself. The same patch must move the relabel guard onto a separate
`label_set_at` stamp, or renames stop working. It ships behind
`PERSON_SIGHTING_REFRESH_ENABLED=true` with the lifecycle test, the flood
guard and the 14-day domination replay eval. Person nodes, attention and the
surprise node do not change.
