## Summary

- New package `orion/body_atlas/`. It converges Orion's body from parts to circulation to organs
  (rungs 1–3 of the self-now ladder, design PR #2888 rev 3).
- **Parts:** declared services from `channels.yaml` plus every source id seen on the live bus
  (orion-bus-mirror's FalkorDB synaptic graph). Status is live, dormant, declared_only, or
  transient.
- **Circulation:** flows between parts, each labelled with its evidence (`declared`,
  `observed_publish`, `observed_causal`). A channel is one hyperedge, normalized over the pairs it
  joins.
- **Organs:** a deterministic Louvain over Newman modularity. Q is reported with every atlas as
  the "organ or mush" number.
- **`python -m orion.body_atlas`:** prints a capped summary (40 lines / 4,000 chars) and writes
  the full atlas JSON. Built to run through Bash in chat and curiosity turns without flooding the
  context.
- **Eval over a frozen export of the real bus graph**, plus a CI workflow.

## Outcome moved

Orion's self-model had no rung above the parts. Self Atlas's newest concepts read "Self Atlas
cluster 'topic_27' (no keywords available)". There is now a measured convergence, live on
2026-10-23:

- **Size:** 137 parts (33 transient smokes/probes), 833 flows, 7 organs, Q = 0.33.
- **Recognizable organs:** eyes, thinking, memory/inference, acting/compute, compute regulation,
  and social.
- **One mixed organ:** body sensing, storage, and the substrate runtimes.

## Current architecture

- **No reducer converges the self.** `self_knowledge_items` (a daily code inventory), Self Atlas
  (clusters with no content), and Layer 3 reflections (leaf-level) never reach organs or
  capacities.
- **The ingredients already existed:** orion-bus-mirror writes live `PUBLISHES` and
  `CAUSALLY_FOLLOWED_BY` edges, and `channels.yaml` declares producers and consumers. Nothing
  combined them.

## Architecture touched

- **New pure package** `orion/body_atlas/` with no service, bus channel, table, or env key.
- **Reads `channels.yaml`** and, read-only, FalkorDB graph `orion_bus_synapse` (`GRAPH.RO_QUERY`).

## Files changed

- `orion/body_atlas/model.py`: the `BodyAtlasV1` artifact, with stable ids per rung for later
  traversal.
- `orion/body_atlas/organs.py`: deterministic Louvain and Newman modularity, pure Python.
- `orion/body_atlas/build.py`: sources, name resolution, status rules, hyperedge-normalized flows,
  organs, stats.
- `orion/body_atlas/__main__.py`: the CLI, output caps, and a raw read-only graph reader.
- `orion/body_atlas/evals/run_organ_structure_eval.py` and `fixtures/bus_graph_2026-10-23.json`:
  the structure eval over the real bus graph.
- `orion/body_atlas/tests/`: 20 tests.
- `orion/body_atlas/README.md`, `.github/workflows/body-atlas-tests.yml`.

## Schema / bus / API changes

- Added: `BodyAtlasV1`. It is a file artifact, not a bus payload, so it is not registered.
- Removed / renamed: none.
- Behavior changed: none. Nothing reads the atlas yet.

## Env/config changes

- Added / removed / renamed keys: none. The CLI reads the existing `FALKORDB_URI` /
  `FALKORDB_BUS_GRAPH` when they are set.
- `.env_example` updated: no.
- local `.env` sync: not needed.

## Tests run

```text
PYTHONPATH=. orion_dev/bin/python -m pytest -q orion/body_atlas/tests   -> 20 passed
all orion-static-gates.yml steps run locally                             -> no failures
tests/test_no_private_eval_artifacts.py tests/test_no_personal_terms.py   -> 10 passed
scripts/check_service_hostname_refs.py                                    -> OK
```

## Evals run

```text
python orion/body_atlas/evals/run_organ_structure_eval.py
  frozen real bus graph 2026-10-23: Q=0.3335, 7 organs, largest 26% -> PASS
  with plain pairwise weights (before hyperedge normalization): Q=0.2191 -> would FAIL
--live, inside orion-athena-harness-governor: same 7 organs, Q=0.3335
```

## Docker/build/smoke checks

```text
Live run inside orion-athena-harness-governor (where chat/curiosity Bash runs), from a copy of the
package: python -m orion.body_atlas --root /mnt/orion-fcc/repo --falkor-uri redis://orion-athena-falkordb:6379
  -> 137 parts, 7 organs, Q=0.3335
No image rebuild: nothing in a running service imports this package yet.
```

## Review findings fixed

The code review (subagent) found no material bugs. It checked modularity against networkx on 300
random graphs, with zero difference, and checked the reply shapes against the live graph.

- Finding: the 4,000-char cut dropped the "full atlas: <path>" line.
  - Fix: the path line is now always kept.
  - Evidence: `test_char_cap_keeps_the_full_atlas_path`.
- Finding: the 42 wildcard catalog channels never join observed publishes.
  - Fix: documented as a known limit (24 messages in the fixture).
  - Evidence: README "Known limits".

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
  - Concern: the organ count and Q depend on weighting choices: hyperedge normalization, and
    log10 of message counts.
  - Mitigation: the eval pins Q ≥ 0.3, 5–25 organs, a maximum organ share of 35%, and
    known-together pairs.
- Severity: low.
  - Concern: the harness image has redis-py 8.1, which has no `redis.commands.graph`, so the
    shared `RedisGraphQueryClient` cannot run there.
  - Mitigation: a scalar-only `GRAPH.RO_QUERY` reader is used instead.
- Severity: note.
  - Concern: 35 real live services are missing from the `channels.yaml` producer/consumer lists.
  - Mitigation: the atlas includes them from live evidence. Fixing the catalog is a separate
    follow-up.

## PR link

See the PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
