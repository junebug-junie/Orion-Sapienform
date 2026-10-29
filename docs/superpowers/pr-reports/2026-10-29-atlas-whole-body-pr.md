## Summary

Orion's body atlas used to see only the services that talk on the bus. Juniper's feedback on v1: "quite reductive and misses things". Now it also sees the machines, the cameras, the memory stores and the ongoing processes, each from a file or a live read that already exists (no hand-typed list of parts in atlas code).

- **Machines ("bodies").** Each machine shows its role, its hardware and whether it is alive. Machines never join organs, because where a part runs is a different question from what it does. Instead each organ says where its members run ("on athena 11, carbon-x1 1, circe 1").
- **Cameras.** The four camera streams the frame router knows. Each camera links to the services that read it, found by rule (the service mounts the router policy, or a `*STREAM_ID` setting names the stream).
- **Stores.** Postgres tables that have a known writer, Chroma collections, and FalkorDB graphs (test and scratch graphs are dropped by a rule).
- **Processes.** The durable workflows (curiosity, reading, dream carry, ...) and the substrate reducers.
- **A readable printout.** Each organ shows member counts by kind and its strongest members. `python -P -m orion.dig body <organ_id>` lists every member of one organ.
- **Loud gaps.** A source that cannot be read shows up as `UNREAD` with its error, and the printout lists what the atlas cannot see yet.

## Outcome moved

The live body went from 104 bus parts in 6 organs to 212 parts in 19 organs (11 with more than one member), plus 7 machines. The new parts are 4 cameras, 91 stores and 13 processes. Organ quality held. Modularity (Q, a score where about 0.3 and up means the organs are real groups) is 0.350 for the whole body and 0.334 for the bus alone on the same day. The largest organ holds 32% of parts; the gate is 35%. Organ ids were the same on two consecutive runs.

## Current architecture

`orion/body_atlas` built parts and flows from the channel catalog (`orion/bus/channels.yaml`) and the bus mirror's live graph in FalkorDB, then ran Louvain to group parts into organs. Every part was a service or an endpoint. `python -P -m orion.dig body` printed up to 4 member names per organ and hid the rest behind "+28".

## Architecture touched

- `orion/body_atlas/sources.py` (new) gathers the rest of the body. Declared sources are read from the checkout, so a past export works too. Live sources: `docker info`/`docker ps` on this host, Postgres through the model-replay sandbox's `ReadOnlyPsql` (`docker exec orion-athena-sql-db psql`, role `orion_readonly`, READ ONLY transaction), the Chroma HTTP API, and FalkorDB `GRAPH.LIST` + `GRAPH.RO_QUERY`.
- `build.py` adds the new parts and flows. Each camera, store or process adds at most one unit of flow weight, split over its edges, so none of them can glue organs together. When the body is read, a channel that orion-sql-writer stores no longer also links its producer to orion-sql-writer. That fact now lives once, on the producer → table `writes` flow. Without this, the largest organ held 36% of parts and failed the gate.
- `dig body <organ_id>` is new. A date is `YYYY-MM-DD` and an organ id is 10 hex characters, so the two cannot be confused.
- `config/biometrics/node_catalog.yaml` gains `hardware:` text and a new `unmonitored_nodes:` section (carbon-x1, janus, pan-1). These go in a separate section because orion-biometrics reports every `nodes:` entry it has not heard from as absent in each cluster message. Adding them under `nodes:` would have changed biometrics output. `NodeCatalog` reads only `nodes`, `retired_nodes` and `defaults`, and it loads the same 4 profiles before and after (checked).

## Files changed

- `orion/body_atlas/sources.py`: new. Declared parsers, live readers and the gather step.
- `orion/body_atlas/model.py`: new part kinds (`host`, `sensor`, `store`, `process`), `facts`, flow `relation`, `SourceReadV1`, organ `kinds`/`placement`, atlas `known_gaps`/`unmatched`/`missing_declared_stores`. All new fields have defaults.
- `orion/body_atlas/build.py`: body parts and flows, `match_part` name rules, hosts kept out of Louvain, the persisted-channel dedupe.
- `orion/body_atlas/__main__.py`: new summary (bodies, UNREAD, kind counts), `organ_view`, `--organ`, `--bus-only`.
- `orion/body_atlas/evals/run_organ_structure_eval.py` + `fixtures/whole_body_2026-10-29.json`: a second frozen fixture (the whole body), a host check, and `--live` on the whole body.
- `orion/body_atlas/tests/test_sources.py`, `test_whole_body.py`, `test_eval_gate.py`; `orion/dig/tests/test_body_whole.py`: new tests.
- `orion/dig/__main__.py`, `orion/dig/README.md`, `orion/body_atlas/README.md`: whole-body `body`, `body <organ_id>`, HELP text.
- `config/biometrics/node_catalog.yaml`: hardware text and `unmonitored_nodes`.
- `scripts/service_rebuild_paths.yaml`: `evals: [orion-harness-governor]`, because the atlas now imports `orion/evals/model_replay/sandbox.py`. Checked with `rebuild_affected_services.py --list-only`: before this line, a sandbox change rebuilt nothing.
- `.github/workflows/body-atlas-tests.yml`: also triggers on the catalog, the frame-router config and the sandbox.

## Schema / bus / API changes

- Added: atlas file fields only (`BodyAtlasV1` is a file artifact, not a bus payload). New part kinds, `PartV1.facts`, `FlowV1.relation`, `AtlasSourcesV1.reads`, `OrganV1.kinds`/`placement`, `AtlasStatsV1.hosts`, `BodyAtlasV1.known_gaps`/`unmatched`/`missing_declared_stores`.
- Removed: none. Renamed: none.
- Behavior changed: `dig body` is the whole body by default. `python -m orion.body_atlas --bus-only` gives the v1 atlas. `stats.transient_parts` now counts transient parts directly (before, it was total minus in-organs, which hosts would have broken).
- Compatibility notes: organ ids change because organs gain members. That is expected. The names ledger carries a name over by member overlap; `test_a_v1_name_carries_onto_the_whole_body_and_new_parts_can_be_cited` proves a v1 name lands on the grown organ. The live ledger is empty today, so no name moved.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. python -m pytest -q -p no:cacheprovider orion/body_atlas/tests orion/dig/tests orion/evals/model_replay/tests
196 passed
PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_static_body_descriptions.py tests/test_organ_names_partition.py
5 passed
services/orion-biometrics tests: 183 passed, 1 failed
  test_circe_node_availability_reflects_expected_offline also fails on origin/main: it expects circe
  to be expected-offline, but circe has been expected_online: true since 2026-10-14. The circe entry
  is unchanged here. Not caused by this patch.
orion/substrate/tests -k "catalog or pressure or biometric": 65 passed, 1 skipped
```

## Evals run

```text
python orion/body_atlas/evals/run_organ_structure_eval.py
organ structure (bus_graph_2026-10-23.json): PASS   Q=0.3335, largest organ 26%
organ structure (whole_body_2026-10-29.json): PASS  Q=0.3503, largest organ 32%
run_organ_structure_eval.py --live (inside orion-athena-harness-governor, worktree code): PASS
```

## Docker/build/smoke checks

Live smoke inside `orion-athena-harness-governor`, using this branch's code without rebuilding the image. I tarred the worktree's `orion/`, `config/`, `mesh-utilities/common/`, `services/orion-sql-writer/app/` and every service's `.env_example` and `docker-compose*.yml` into `/tmp/atlas-wt` in the container. Then I ran this command twice:

```text
env PYTHONPATH=/tmp/atlas-wt ORION_DIG_NOW_ROOT=/tmp/atlas-wt ORION_DIG_HOME=/tmp/atlas-dig-home python -P -m orion.dig body
```

The organ ids were identical on both runs, and the same again after the review fixes. The command took 3.0 s and printed 29 lines / 3,414 chars. Nothing was written to any DB, graph or bus. The dig output went to `/tmp/atlas-dig-home` in the container, not to Orion's real dig home.

```text
parts 254: 100 svc, 4 endpoint, 4 camera, 91 store, 13 process, 7 hosts, 35 transient · organs 19 (8 singletons)
flows 828 signals, 98 writes, 79 runs_on, 36 uses, 13 runs, 11 sees_through · Q=0.3503 · largest organ 32%
bodies (machines; not in organs):
  athena: orchestration, HP DL380 G10 (orchestration + vision/audio node); live, telemetry biometrics; services 71 running
  circe: primary_cognition_gpu, Gigabyte (model not recorded; daily inference node); live, telemetry biometrics; services 6 declared
  carbon-x1: juniper_laptop, Lenovo ThinkPad X1 Carbon (Juniper's laptop); declared_only, telemetry none
  hecate: gpu_server, Inspur NF5288M5, 8-GPU (deep inference and training node); dormant last seen 2026-10-15T06:31
  janus / pan-1 / prometheus: declared_only
UNREAD 3 source(s): substrate_durable_run_state, substrate_reduction_cursor, vision_events (permission denied)
 4. organ:f4c1053caf [19: 13 svc, 4 camera, 2 store; 14 live] on athena 11 carbon-x1 1 circe 1: orion-vision-council, ...
 5. organ:868715ebc1 [17: 2 svc, 15 store; 17 live] on athena 2: orion-world-pulse, orion-attention-runtime, pg:world_pulse_article, ...
```

## Review findings fixed

A review subagent read the branch diff (`origin/main...HEAD`). It found nothing that writes and no secrets in the fixture. Its findings and fixes:

- Finding (HIGH): old organ names would mostly stop carrying over. A name carries when the member sets overlap by at least half. The new cameras, stores and processes diluted that overlap: on the real fixture, 4 of 6 bus-only organs fell to 0.14-0.32.
  - Fix: `names._overlap` compares only the kinds the named organ could have held. A name given to a bus-only organ is compared against the bus members only.
  - Evidence: `test_old_names_carry_onto_the_real_whole_body`. On the real fixture the vision organ's overlap is 1.0 (it was 0.68 before the fix), and no organ scores worse than before. Three of the six organs still fall below one half, because the organs really regrouped (cortex now shares an organ with sql-writer). That is the carry rule doing its job. The live names ledger is empty today, so no name was lost.
- Finding (MEDIUM): a stored channel's bus flow could be dropped with nothing put in its place. This happens when its table is missing from the live database.
  - Fix: the flow moves only when the table became a part and has a `writes` flow for that channel. This is checked in `sources.py` and again in `build.py`.
  - Evidence: `test_a_persisted_channel_whose_table_is_not_in_the_db_keeps_its_bus_flow`.
- Finding (MEDIUM): a compose file that did not parse was skipped silently.
  - Fix: a new read status, `partial`, printed under UNREAD. The live run then exposed a real case: `orion-cortex-exec`'s compose uses `ports: !override []`, which `safe_load` refuses. Compose-only tags now parse as plain values.
  - Evidence: `test_a_compose_file_that_does_not_parse_makes_the_read_partial`, `test_compose_only_tags_parse`. The live printout no longer shows the partial read.
- Finding (MEDIUM): two status labels claimed more than was known.
  - Fix: Postgres row counts are labeled `rows_est`, because `n_live_tup` is an estimate. When a camera or process table is read and holds no rows, the part becomes `dormant` with 0. It no longer reads `declared_only`, which looked the same as "not read".
  - Evidence: `test_read_with_no_rows_is_zero_not_unknown`.
- Finding (LOW): a psql NOTICE or WARNING line could parse as a data row.
  - Fix: those lines are filtered before CSV parsing.
  - Evidence: `test_psql_notices_are_not_rows`.
- Finding (LOW): the scratch-graph rule was too broad (`testimony`, `smoker`, `foo_t_bar`), and it dropped graphs without naming them.
  - Fix: tighter tokens, and the dropped names are listed in the read detail.
  - Evidence: `test_scratch_rule_keeps_real_words`, `test_dropped_scratch_graphs_are_named_in_the_read`.
- Finding (LOW): a compose `${VAR:-walkway}` default was not read as a stream.
  - Fix: the default value is read.
  - Evidence: `test_compose_defaults_name_a_stream`.
- Finding (LOW): `dig body <word>` gave a date error for anything that was not a 10-hex organ id.
  - Fix: anything that is not `YYYY-MM-DD` is looked up as an organ.
  - Evidence: `test_body_with_a_non_date_word_is_an_organ_lookup`.

## Restart required

No restart is needed for config: the harness reads the declared files from the mounted primary checkout when a command runs. The atlas code is baked into the harness image, so the new code goes live after the normal post-pull rebuild of orion-harness-governor (mapped in `scripts/service_rebuild_paths.yaml`). To do it by hand, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

The `node_catalog.yaml` change also triggers the post-pull rebuild of orion-biometrics and orion-substrate-runtime. Their behavior does not change, because `NodeCatalog` ignores the new keys.

## Risks / concerns

- Severity: medium. Concern: `orion_readonly` cannot read `substrate_durable_run_state`, `substrate_reduction_cursor` or `vision_events`. It can read 24 of 351 tables. So processes and cameras show as `declared_only`, and reducers are missing. Mitigation: the printout says UNREAD every time. The fix needs Juniper to approve a grant (not applied here, and a `GRANT` migration does not fit the post-pull auto-apply allowlist): `GRANT SELECT ON substrate_durable_run_state, substrate_reduction_cursor, vision_events TO orion_readonly;`
- Severity: low. Concern: only athena's docker is observed. Placement on circe is declared (its allowlist), and hecate has none. 60 of 133 athena containers matched no part (zelda dojo workers, lightdash, otel, the DB containers), plus 2 declared circe services (orion-ai-town, orion-kev: not on the bus). They are listed in the atlas `unmatched` field.
- Severity: low. Concern: four Chroma collections that services name in `.env_example` do not exist (`orion_chat_gpt`, `orion_chat_gpt_turns`, `orion_latent_store`, `orion_metacog_store`). They are reported in `missing_declared_stores` and never made into parts.
- Severity: low (not caused by this patch). Concern: CI "Hub — full tests folder" hung and was cancelled at its 30-minute limit twice at about 39%, then passed on a rerun. Mitigation: I reproduced the hang locally on a clean `origin/main` checkout. It stalls in `services/orion-hub/tests/test_endogenous_outreach.py`'s concurrency tests (`test_concurrent_ticks_cannot_both_send`), but only when the whole suite runs; the file alone passes. The same suite passed on this branch's first commit. This patch touches no hub code or hub import. It is a racy test on main and needs its own fix.
- Severity: low. Concern: no machine-readable source exists yet for the mood v4 autoencoder, the personality program as a process, AI Town's bus identity, graphify, or Postgres readers. They are listed in `known_gaps` and in the README. They are not faked.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2979
