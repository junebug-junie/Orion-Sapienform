# Body atlas

Orion's body, converged from parts to organs. Rungs 1–3 of the self-now ladder in
`docs/superpowers/specs/2026-10-23-repo-chronicle-design.md` (PR #2888). Since 2026-10-29 the
atlas sees the whole body, not only the bus: the machines, cameras, stores and processes too.

1. **Parts.** Each part has a kind:
   - `service` / `endpoint`: declared services (`orion/bus/channels.yaml`) plus every source id
     seen on the live bus.
   - `host` (`host:athena`): a physical machine. From `config/biometrics/node_catalog.yaml`
     `nodes:` (biometrics watches them; live = biometrics reported in the last 24 h) and
     `unmonitored_nodes:` (carbon-x1, janus, pan-1: real machines biometrics does not watch).
     `hardware:` is Juniper's words; an unknown model stays "not recorded".
   - `sensor` (`camera:walkway`): a camera stream from `config/vision_frame_router.yaml`
     `streams:` (its stale `cameras:` section is ignored). Live activity from `vision_events`.
   - `store`: `pg:<table>`, `chroma:<collection>`, `falkor:<graph>`. A Postgres table is a part
     only when something links to it (a known writer); Chroma collections and FalkorDB graphs
     are the ones that exist live (test/scratch graphs dropped by rule). A store has no
     last-write time: `live` means it holds rows, `dormant` means it is empty.
   - `process`: `run:<workflow>` (the `DurableWorkflowV1` Literal, plus any workflow seen in
     `substrate_durable_run_state`) and `reducer:<cursor>` (`substrate_reduction_cursor`).

   Status is `live` / `dormant` / `declared_only` / `transient` (smokes and probes, kept out
   of organs). Each part carries a few `facts` (hardware, rows, readers, runner, last run).
2. **Circulation.** Directed flows. Each flow has a `relation` and says where it came from:

   | relation | from → to | evidence |
   |---|---|---|
   | `signals` | service → service | `declared` (catalog), `observed_publish`, `observed_causal` |
   | `runs_on` | service or camera → host | `observed_placement` (this host's `docker ps`), `declared_placement` (`mesh-utilities/common/include_services_<host>.txt`, `runs_services`, `cameras`) |
   | `sees_through` | camera → service | `declared_config`: the compose mounts the frame-router policy, or a `*STREAM_ID` setting names the stream |
   | `writes` | producer → table | `declared_route`: orion-sql-writer's `DEFAULT_ROUTE_MAP` (kind → table) joined with that kind's channel producers |
   | `writes` / `reads` / `uses` | service → Chroma/FalkorDB store | `declared_env`: its `.env_example` names the store (`uses`: the env does not say which way). `declared_code`: its code names the store (`store_code.py`, below); `writes` / `reads` only when the call says which |
   | `runs` | runner → process | `declared_runner` (producer of `durable.run.state.v1`), `assumed_runner` (reducers → orion-substrate-runtime) |
   | `does` | service → durable run | `observed_trace` (a `grammar_events` / `cognition_traces` / `cockpit_turn_sighting` row with the run's `correlation_id` names the service), `observed_turn` (same, keyed by the turn the run started, `detail.turn_correlation_id`), `declared_turn_writer` (a `harness_turn_trace` row exists for the run; that table's declared writers) |
   | `spawns` | run → run | `observed_child_run` (`detail.child_run_ids`, e.g. dream.carry → reverie.visual) |
   | `feeds` | service → reducer | `declared_reducer_source` (orion-substrate-runtime's `REDUCER_SPECS`: the one service whose grammar events the reducer reads) |

   A channel is one hyperedge: it contributes one unit, split over the pairs it joins, so a
   broadcast heartbeat cannot glue the body together. The new kinds follow the same rule: one
   camera, store or process adds at most one unit, split over its edges. With the body, a
   channel orion-sql-writer persists no longer also links its producer to orion-sql-writer:
   that fact now lives on the `writes` flow to the real table (counted once, not twice).

   A Chroma collection's or FalkorDB graph's one unit is split by how specific each source is:
   `log(1 + N / df)`, `df` = how many of the N stores that source (an env file, a code module, a
   persisted channel) names. A module naming one store keeps most of it; a registry naming ten
   keeps a little of each. One edge per service: its strongest relation (writes > reads > uses),
   every evidence kept. A store no service's code, env or route names is printed as
   `UNATTRIBUTED`, with its row count, never guessed.

   **Code evidence** (`store_code.py`, by rule, no store → service list). A module names a store
   when its AST holds a string constant exactly equal to the name (never a substring, a docstring
   sentence or an f-string), or reads an env key that service's `.env_example` / compose sets to
   the name. `services/<svc>/` is `<svc>`. A shared `orion/` module is split into its top-level
   symbols, and a service gets the stores of the symbols it imports or reads off the module (one
   hop; a shared module no service imports is reported). Direction is claimed only from three
   plain shapes, else `uses`: a method on a handle bound from the store (`col.upsert` writes,
   `col.query` reads, a `query`'s first argument decides when it is upper-case Cypher); a plain
   function taking the store directly and named for its act (`_publish_semantic_upsert(...)`);
   `read_only=True`. A handle is the store object or its name (`get_collection(NAME)`,
   `os.getenv(KEY)`, a constructor), never the result of an act. A method of any other object
   (`seen.add(NAME)`, `parser.add_argument`) and a script run by hand (`if __name__ ==
   "__main__":`) never set a direction. `tests/`, `evals/` and the atlas itself are skipped.

   A process's one unit is split over its runner and its doers by
   `log(1 + runs) × log(1 + N / df)`: how many of its runs the part showed up in, times how rare
   the part is across the N processes (`df` = processes it appears in). The scheduler
   (orion-durable-runs) and the GPU lease appear in nearly every process, so they keep a small
   share without any hand-typed "plumbing" list. A process with only its runner keeps the whole
   unit on it. Each process's `work_split` fact shows its top three shares.
3. **Organs.** Louvain communities over Newman modularity. `stats.modularity` (Q) is the
   convergence test: about 0.3 and up is real structure; near 0 is mush. **Hosts never join
   organs**: athena runs ~130 containers, so placement edges would glue the body into one
   blob, and where a part runs is a different axis from what it does. Each organ shows its
   placement counts instead (`on athena 11 circe 1`). Organ `name` is `None` until Orion names
   it. `namespaces` is a code-derived hint, not a name.

```bash
# live (inside orion-harness-governor, where chat and curiosity run Bash):
python -P -m orion.dig body                 # the live body, capped printout
python -P -m orion.dig body <organ_id>      # every member of one organ
FALKORDB_URI=redis://orion-athena-falkordb:6379 python -P -m orion.body_atlas --out /tmp/atlas.json
python -P -m orion.body_atlas --organ <organ_id>
# declared only, e.g. a past checkout:
python -P -m orion.body_atlas --root /path/to/checkout --no-live --out /tmp/atlas.json
# the v1 bus-only atlas:
python -P -m orion.body_atlas --bus-only
```

The printout is capped at 40 lines and 4,000 chars; the full atlas goes to `--out` (dig writes
it under `$ORION_DIG_HOME/dig/`). A failed live bus read is an error. Every other source is
optional but loud: one that fails is recorded in `sources.reads` as `unread` with its error,
and the printout lists it under `UNREAD`. A declared file a past checkout lacks is `absent`.

Postgres is read through `orion/evals/model_replay/sandbox.py`'s `ReadOnlyPsql`: `docker exec
orion-athena-sql-db psql` as `orion_readonly`, inside a READ ONLY transaction. Nothing here
writes to any DB, graph or bus.

Gates:

- `pytest orion/body_atlas/tests`
- `python orion/body_atlas/evals/run_organ_structure_eval.py`: four frozen fixtures, the real bus
  graph from 2026-10-23 and the whole body from 2026-10-29 and 2026-10-30 (with process doers,
  then with stores named in code). Each must have Q ≥ 0.3, 5–25 organs, no organ above 35% of
  parts, every non-host part in exactly one organ, no host in any, no store with a writer or
  reader left alone, and obvious pairs together.
- `python orion/body_atlas/evals/run_store_direction_eval.py`: the `writes` / `reads` labels on
  this checkout against hand-checked facts (true ones, and false ones an earlier version
  claimed). Precision must be 1.0 and no false claim may come back. `--live` runs it on the live whole body; `--export F`
  refreshes a fixture.

Live results:

- 2026-10-23, bus only: 137 parts (33 transient), 7 organs, Q = 0.33.
- 2026-10-29, whole body: 254 parts (7 hosts, 4 cameras, 91 stores, 13 processes, 35
  transient), 19 organs (8 singletons), Q = 0.350, largest organ 32%. Bus only, same day:
  Q = 0.334, largest 31%.
- 2026-10-30, process doers: the largest organ went from 75 parts (34%) to 68 (31%), Q 0.350 →
  0.349, 18 organs both ways. 7 of 8 reducers left it for the organ of the service they read.
  The 12 durable workflows stay: their readable doers are the harness and the scheduler. The rest of that organ is real: orion-hub,
  orion-gpu-pool, orion-harness-governor, orion-durable-runs and orion-substrate-runtime signal
  each other on the bus, and most durable runs are harness turns scheduled by orion-durable-runs.

- 2026-10-30, stores named in code: singletons 8 → 5, organs 18 → 15, Q 0.3464 → 0.3469,
  largest organ 31% → 31%. `chroma:orion_social_room` joined orion-vector-host (memory organ;
  it writes it), `chroma:orion_atlas_concepts` and `chroma:orion_day_letter_parts` joined
  orion-hub (substrate organ). The rarity split also moved orion-bus-mirror and
  `falkor:orion_bus_synapse` from substrate to guardian: bus-mirror's `.env_example` names only
  that graph, so it now holds most of it, next to orion-mesh-guardian, which reads it. Names
  carried (substrate 0.94, memory 0.96, guardian 0.75). The 5 left alone are FalkorDB graphs
  with 0 nodes that no code, env or compose file in the repo names (`orion_bus_mirror`,
  `orion_episodic`, `orion_express`, `orion_kg`, `orion_turns`): leftovers, not hidden organs.
  Direction is claimed for 5 links, all hand-checked (`run_store_direction_eval.py`); every
  other code link is `uses`.

Known gaps (also in every whole-body atlas as `known_gaps`):

- **Grants.** On 2026-10-30 `orion_readonly` cannot read `grammar_events`, `cognition_traces` or
  `cockpit_turn_sighting`, so the `observed_trace`/`observed_turn` doers are missing (each is
  UNREAD with the exact GRANT it needs). The run-state, cursor and camera tables are readable now.
  Those two evidence kinds have never run live: after the grant, check the `grammar_events` join
  (1.4 M rows, no `correlation_id` index; bounded to 31 days) with `EXPLAIN` and a live run.
- **Run products.** A run's detail carries ids it made (`entry_id`, `dream_id`, `episode_id`), but
  no declared source says which table holds each id, so processes have no `writes` flow yet.
- **Other hosts' docker.** Only this host's containers are observed; circe placement is
  declared (its allowlist), hecate has none.
- **No machine-readable source yet:** the mood v4 autoencoder, the personality program as a
  process, AI Town's bus identity (traces only), graphify (host-only storage), and who reads
  a Postgres table.
- **Old checkouts.** Before 2026-01 there is no producer catalog, so an older checkout gives no
  atlas.
- **Wildcard channels are not joined.** Observed channels are matched to catalog names exactly,
  so wildcard catalog entries add no `observed_publish` flows.
- **Replicas stay separate.** Replicas (`atlas-worker-1`, `-2`) are separate parts that land in
  one organ.
