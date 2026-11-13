## Summary

- The body atlas now links a Chroma collection or FalkorDB graph to the services whose **code** names it, not only to services whose `.env_example` names it. Before, 8 stores sat alone as one-part organs, and Orion read that as "orphaned".
- New `orion/body_atlas/store_code.py` finds the evidence by rule (no store → service list): an exact string constant, an env key whose value is the store, or a shared `orion/` module symbol that a service imports. It also says writer vs reader when the call makes it clear.
- A store's one unit of weight is split by how specific each source is (`log(1 + N / df)`), so a module that names every collection cannot glue organs together.
- Stores that no code, env or route names are printed as `UNATTRIBUTED`, with their row count.
- Writer vs reader is claimed only from three plain call shapes. Everything else is `uses`. A new eval checks every claimed direction against hand-checked facts from the live repo.
- Eval: a new live fixture, and a new check that a store with a link is never left alone.

## Outcome moved

Live, 2026-10-30, same moment, main code vs this branch (inside `orion-athena-harness-governor`):

| | before | after |
|---|---|---|
| lone organs | 8 | 5 |
| organs | 18 | 15 |
| Q (modularity) | 0.3464 | 0.3469 |
| largest organ | 31% | 31% |

Where each store went:

- `chroma:orion_social_room` → memory organ, with orion-vector-host (it writes the collection: `_publish_semantic_upsert(collection_name=SOCIAL_ROOM_COLLECTION)`).
- `chroma:orion_atlas_concepts` → substrate organ, with orion-hub (hub imports `ATLAS_CONCEPT_COLLECTION` and builds its concept search with it).
- `chroma:orion_day_letter_parts` → substrate organ, with orion-hub (`orion_day_introspect_listener.py` names it).
- `falkor:orion_bus_mirror`, `orion_episodic`, `orion_express`, `orion_kg`, `orion_turns` stay alone. Each has **0 nodes** and no code, env or compose file in the repo names it. These five really are leftovers. Orion's "orphaned" was right for them and wrong for the other three.

- One more move comes from the rarity split, not from code evidence. orion-bus-mirror and `falkor:orion_bus_synapse` moved from substrate to guardian. Bus-mirror's `.env_example` names only that graph, so bus-mirror now holds most of its weight, next to orion-mesh-guardian, which reads it. I checked this against a fresh main-code run in the same minute, so it is not live drift.

No other part changed organ. On the frozen fixture, adding code evidence moves exactly the 3 stores.

Organ names. The cortex organ keeps its id (`organ:2ad84a72c8`). Three organs get new ids, and each one keeps its name (the carry-over needs 0.5 overlap):

- substrate: `organ:8626d4fdb8` → `organ:b039ea861e`, overlap 0.94
- memory: `organ:995cb31c70` → `organ:28fdd9fc38`, overlap 0.96
- guardian: `organ:e791979b19` → `organ:6395916951`, overlap 0.75

All 10 of Orion's names survive. The ledger was read from a copy and never written.

## Current architecture

Chroma and FalkorDB stores came from live reads. Their only links came from `env_store_refs`: a service `.env_example` line whose value is exactly the store name gave a `uses` flow. A store named only in code had no flow, so it became a lone organ.

## Architecture touched

`orion/body_atlas` only, which is a file artifact and not a bus payload. There are no bus, schema-registry or env changes.

## Files changed

- `orion/body_atlas/store_code.py` (new): the code-evidence scanner.
- `orion/body_atlas/sources.py`: runs the scanner as a declared read (reported in `reads`). Env links now carry their env file as `via`. `StoreIn.named_in_unimported` is new.
- `orion/body_atlas/build.py`: one edge per service per store (strongest relation, all evidence kept), the rarity split (`_store_source_idf`), and `written_by` / `read_by` / `used_by` / `evidence` facts. A `via` is used as a bus channel only for `declared_route`.
- `orion/body_atlas/model.py`: relation `reads` and evidence `declared_code` are new.
- `orion/body_atlas/__main__.py`: a one-line `UNATTRIBUTED` list in the printout.
- `orion/body_atlas/evals/run_organ_structure_eval.py`: a fourth fixture and the "linked store left alone" check.
- `orion/body_atlas/evals/run_store_direction_eval.py` (new): the ground truth for `writes` / `reads`.
- `orion/body_atlas/evals/fixtures/whole_body_2026-10-30_stores.json`: a live export made with this branch.
- `orion/body_atlas/tests/test_store_code.py` (new), `test_eval_gate.py`, `test_sources.py`.
- `orion/body_atlas/README.md`: the flows table, the weighting and code-evidence rules, and live results.

## Schema / bus / API changes

- Added: `FlowRelation` `reads`; `FlowEvidence` `declared_code`; `StoreIn.named_in_unimported` (default empty).
- Removed / renamed: none.
- Behavior changed: an env link's `via` is now its `.env_example` path, not `""`.
- Compatibility notes: old fixtures load unchanged and give the same Q and organs, because equal sources still give the plain even split.

## Env/config changes

- None. No `.env_example` changed, so no `.env` sync was needed.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/body_atlas orion/dig -q
156 passed
pytest tests/test_organ_names_partition.py -q   1 passed
```

## Evals run

```text
python orion/body_atlas/evals/run_organ_structure_eval.py
bus_graph_2026-10-23.json        PASS  Q=0.3335 largest 26%
whole_body_2026-10-29.json       PASS  Q=0.3503 largest 32%
whole_body_2026-10-30.json       PASS  Q=0.349  largest 31%
whole_body_2026-10-30_stores.json (final export) PASS Q=0.3469 largest 31%
run_store_direction_eval.py: directional claims 5, judged 5, precision 1.00 (PASS)
run_organ_structure_eval.py --live (in orion-athena-harness-governor, branch code): PASS
```

## Docker/build/smoke checks

```text
The worktree is not mounted in orion-athena-harness-governor. I copied the branch's orion/,
config/, mesh-utilities/common/ and services/**/{*.py,.env_example,docker-compose*} into
/tmp/atlas-stores/root in the container. Then I ran:
  before: PYTHONPATH=/mnt/scripts/Orion-Sapienform ORION_DIG_HOME=/tmp/atlas-stores/home-main python -P -m orion.dig body
  after:  PYTHONPATH=/tmp/atlas-stores/root ORION_DIG_HOME=/tmp/atlas-stores/home ORION_DIG_NOW_ROOT=/tmp/atlas-stores/root python -P -m orion.dig body
Each ORION_DIG_HOME held a copy of /mnt/orion-fcc/self/organ_names.jsonl, so nothing was
written to /mnt/orion-fcc. The live code scan adds about 3 s to `dig body` (222 modules parsed). A full `dig body` took 10 s.
```

## Review findings fixed

A review subagent read the diff (target: `fix/atlas-store-writers` vs origin/main) and ran a tracer on the real repo.

- Finding (HIGH): any call holding the store anywhere in its arguments was classed by its own name. On the real repo this gave 6 false directions, for example hub "writes" orion_curiosity through `asyncio.create_task`, and recall "writes" crystallizations through a Postgres insert.
  - Fix: direction is claimed only for a method called on a handle, a plain function taking the store directly and named for its act, or `read_only=True`. `get_collection` and `select_graph` bind a handle; they are not acts.
  - Evidence: `test_calls_of_other_objects_are_not_the_stores_act` (11 real patterns, all `uses`).
- Finding (HIGH): tagging spread through any assignment that mentioned a store, and attribute labels dropped their receiver (`p.name` matched `name`).
  - Fix: a name becomes a handle only when bound from the store object or its name (a handle binder, a constructor, `os.getenv`, `str`). The result of an act is not a handle. Bound handles are keyed by their full spelling (`self._col`).
  - Evidence: the same test, including `result = backend.ingest_episode(graph_name=G); result.get(...)`.
- Finding (HIGH): nothing measured whether `writes` / `reads` is true.
  - Fix: `run_store_direction_eval.py` checks 5 hand-checked true facts and 9 known false claims. Precision must be 1.0. It runs in pytest.
  - Evidence: precision 1.00, no false claim returns.
- Finding (MEDIUM): one-off scripts counted as the running service.
  - Fix: a module with a top-level `if __name__ == "__main__":` only names a store; it never sets a direction.
- Finding (MEDIUM): the Cypher check matched English words in any argument and could not see a query held in a constant.
  - Fix: only the first argument is checked, with upper-case keywords. A module-level constant resolves.
- Finding (MEDIUM): `if __name__` blocks counted as import-time code, and a bare module import gave every symbol.
  - Fix: the main guard is dropped. Top-level `if` / `try` are opened. A bare import gives only import-time code.
- Finding (MEDIUM): a store whose links matched no part was printed as named by nobody.
  - Fix: it now says "named by X, which matched no part" and is not listed as UNATTRIBUTED.
- Finding (LOW): rows text ("not read rows"), scan speed, docstring precision, eval wording.
  - Fix: rows fall back cleanly. `os.walk` now prunes skipped directories, and the C YAML loader is used. Each shared symbol is scanned once. The scan time went from 4.3 s to 3.0 s.

## Restart required

```text
No restart required. The harness reads the repo from the mounted primary checkout, so the next `dig body` after merge + pull uses it.
```

## Risks / concerns

- Severity: medium. Concern: direction recall is low. Only 5 links get `writes` / `reads`; every other code link is `uses`. For example, orion-chat-memory really writes `orion_chat`, but the scan says `uses`. The 5/5 recall in the eval is over facts chosen from the claims, so it proves precision, not coverage. Mitigation: organ structure does not depend on direction. Add a fact to the eval before widening a rule.
- Severity: low. Concern: shared-module attribution is one hop. A service that uses a store through a function that calls another function in the same shared module is missed. Mitigation: the store then shows as `UNATTRIBUTED` or keeps its other links. It is never guessed.
- Severity: low. Concern: `services/<svc>/scripts/` counts as the service, because the hub app runs from `scripts/`. Mitigation: a hand-run script (it has a main guard) still links the service, but it never sets a direction.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3002

🤖 Generated with [Claude Code](https://claude.com/claude-code)
