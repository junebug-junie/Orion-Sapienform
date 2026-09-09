# Kill the dead `self_study` belief producer and the stale hub identity card

Branch: `chore/kill-dead-self-study-producer`

## Summary

- Orion's chat stance used to ask a "self_study" producer for beliefs about
  themself on every turn. That producer ran a SPARQL query against an RDF
  named graph. The graph name (`SELF_STUDY_NAMED_GRAPH`) was empty in every
  env, and RDF/Fuseki is retired repo-wide, so it returned nothing every time
  while still taking a thread in the cold-pull fan-out. It is now gone from
  both registries that listed it (`chat_stance.py` and
  `projection_builder.py`).
- The adapter module behind it, its three package re-exports, its test
  class, and its env key are deleted -- nothing live imported it once the
  registry entries were gone. "Kill means kill": no stub, no fallback.
- The stale identity card `services/orion-hub/scripts/memory/identity.yaml`
  (still listing Ollama, an RDF Memory Writer and "Collapse Mirror" as
  Orion's services; zero code readers) is deleted. The live card,
  `orion/cognition/personality/orion_identity.yaml`, is untouched.
- A regression test refuses any of it coming back. Two stale count/list
  assertions were corrected, one of which had been failing on `main` since
  `self_state`/`orionmem` were removed.
- Its replacement, the `self_definition` producer (PR #2158), is still
  registered and still tested.

## Outcome moved

- One fewer dead producer thread on every chat turn's cold pull
  (`CognitiveUnificationLayer.beliefs_for_stance`), in both cortex-exec's
  chat stance and cortex-orch's Mind cold projection.
- One fewer place where Orion's self-description could be read from a stale,
  wrong source.
- `orion/cognition/tests/test_projection_builder.py::test_projection_registry_matches_expected_chat_stance_producers`
  goes red -> green (it was already failing on `main`, see Tests).

## Current architecture

Before this patch, two functions built the list of "belief producers" that
feed Orion's pre-LLM cognitive projection:

- `services/orion-cortex-exec/app/chat_stance.py::_build_unification_registry`
  (chat turns), and
- `orion/cognition/projection_builder.py::build_projection_unification_registry`
  (shared spine; called live by
  `services/orion-cortex-orch/app/mind_runtime.py::_build_cold_cognitive_projection_facet`
  and, via `unified_beliefs_for_chat_stance`, by
  `services/orion-cortex-exec/app/chat_stance_shared_spine.py`).

Both listed `producer_id="self_study"` with
`adapter_fn=map_self_study_to_substrate`
(`orion/substrate/relational/adapters/self_study.py`). That adapter read
`SELF_STUDY_NAMED_GRAPH` and `SELF_STUDY_GRAPHDB_TIMEOUT_SEC`, resolved a
SPARQL endpoint, and returned `None` unless both an endpoint and a graph
were configured. Neither ever was. Confirmed in the design doc's
"What the durable self-stores hold" table
(`docs/superpowers/specs/2026-09-08-orion-sense-of-self-design.md`, branch
`docs/orion-self-sense-design`): "Zombie producer."

`services/orion-hub/scripts/memory/identity.yaml` sat next to a sibling
`narrative.yaml`; `rg` across `services/`, `orion/`, `scripts/`, `tests/`
finds no reader of either.

## Architecture touched

- cortex-exec chat stance producer registry (one entry removed).
- Shared cognitive-projection producer registry (one entry removed) -- this
  is a live path in cortex-orch too, so cortex-orch's Mind cold build also
  loses the dead producer.
- `orion.substrate` / `orion.substrate.relational` /
  `orion.substrate.relational.adapters` public exports (one symbol removed).
- cortex-exec env contract (one key removed).
- No bus channel, schema, or HTTP API changes.

### Blast-radius decision on the adapter module

The task said: delete the adapter only if nothing live imports it. Importers
found by `rg` before editing:

| importer | what it did | decision |
|---|---|---|
| `services/orion-cortex-exec/app/chat_stance.py` | registered the producer | entry + import removed |
| `orion/cognition/projection_builder.py` | registered the producer (live via cortex-orch `mind_runtime.py:288` and cortex-exec `chat_stance_shared_spine.py`) | entry + import removed -- the live path only *listed* the producer; it never depended on its output because the output was always `None` |
| `orion/substrate/__init__.py`, `orion/substrate/relational/__init__.py`, `orion/substrate/relational/adapters/__init__.py` | re-export only | export removed |
| `orion/substrate/relational/tests/test_adapters.py::TestSelfStudyAdapter` | tested that it returns `None` | test class removed |
| `services/orion-cortex-exec/tests/test_skill_verbs.py`, `.../test_docker_compose_service_bringup.py` | reference `verb_adapters.self_study_module` -- that is the **live** self-study verbs module `services/orion-cortex-exec/app/self_study.py`, not this adapter | untouched |

After the registry entries were removed, no live code imported the adapter,
so it was deleted.

## Files changed

- `services/orion-cortex-exec/app/chat_stance.py`: drop the `self_study`
  `ProducerEntryV1` and the adapter import.
- `orion/cognition/projection_builder.py`: same, in the shared registry.
- `orion/substrate/relational/adapters/self_study.py`: deleted.
- `orion/substrate/__init__.py`, `orion/substrate/relational/__init__.py`,
  `orion/substrate/relational/adapters/__init__.py`: drop the re-export.
- `orion/substrate/relational/tests/test_adapters.py`: drop
  `TestSelfStudyAdapter`.
- `orion/substrate/relational/tests/test_reducer_lane_adapters.py`: producer
  count 13 -> 12, assert `self_study` absent.
- `orion/cognition/tests/test_projection_builder.py`: expected producer list
  now matches the real registry (was stale on `main`: still listed
  `self_state` and `orionmem`, missing `attention`/`episodes`/`curiosity`).
- `services/orion-cortex-exec/tests/test_chat_stance_no_self_study_producer.py`:
  new regression test -- no `self_study` in either registry,
  `self_definition` still present, adapter module not importable, no
  re-export, no env key.
- `services/orion-cortex-exec/.env_example`: remove `SELF_STUDY_NAMED_GRAPH`
  and its comment line.
- `services/orion-hub/scripts/memory/identity.yaml`: deleted.
- `docs/superpowers/pr-reports/2026-09-09-kill-dead-self-study-producer-pr.md`:
  this report.

## Schema / bus / API changes

- Added: none
- Removed: public symbol `map_self_study_to_substrate` from
  `orion.substrate`, `orion.substrate.relational`,
  `orion.substrate.relational.adapters`.
- Renamed: none
- Behavior changed: the belief-set fan-out no longer spawns a producer that
  always returned `None`. Belief output is byte-identical because that
  producer never contributed a node.
- Compatibility notes: any out-of-tree import of the removed symbol would
  now fail at import time. `rg` finds none in this repo outside `docs/`.

## Env/config changes

- Added keys: none
- Removed keys: `SELF_STUDY_NAMED_GRAPH` (`services/orion-cortex-exec/.env_example`).
  `SELF_STUDY_GRAPHDB_TIMEOUT_SEC` was read by the adapter but was never
  declared in any `.env_example`, `.env`, `settings.py`, or compose file,
  so there was nothing to remove.
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  ran it. **It does not remove keys** -- by design it only adds missing keys
  and reports diverged values. The operator should delete this one line by
  hand (harmless if left; nothing reads it any more):
  - `services/orion-cortex-exec/.env` line 248: `SELF_STUDY_NAMED_GRAPH=`
- skipped keys requiring operator action: none beyond the line above.
- The live `SELF_STUDY_*` keys used by the self-study verbs
  (`SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR`,
  `SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH`, `SELF_STUDY_REFLECT_LLM_ROUTE`,
  `SELF_STUDY_REFLECT_TIMEOUT_SEC`) are untouched.

## Tests run

Interpreter: `/mnt/scripts/Orion-Sapienform/.venv/bin/python` (pytest 8.3.4).

Baseline on `main` HEAD `9572e8eb7`, before any edit, same file set:

```text
FAILED services/orion-cortex-exec/tests/test_chat_stance_brief.py::test_build_chat_stance_inputs_falls_back_when_identity_missing
FAILED orion/cognition/tests/test_projection_builder.py::test_projection_registry_matches_expected_chat_stance_producers
2 failed, 106 passed
```

After the patch:

```text
pytest -q services/orion-cortex-exec/tests/test_chat_stance_no_self_study_producer.py \
  services/orion-cortex-exec/tests/test_chat_stance_brief.py \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  services/orion-cortex-exec/tests/test_chat_stance_self_definition.py \
  orion/cognition/tests/test_projection_builder.py \
  orion/cognition/tests/test_projection_starvation_diagnostics.py \
  orion/cognition/tests/test_recall_prefetch.py \
  orion/substrate/relational/tests \
  services/orion-cortex-orch/tests/test_mind_projection_resolver.py \
  services/orion-cortex-orch/tests/test_mind_orch.py \
  services/orion-cortex-exec/tests/test_skill_verbs.py
1 failed, 244 passed
  FAILED ...test_chat_stance_brief.py::test_build_chat_stance_inputs_falls_back_when_identity_missing  (pre-existing on main, unrelated)

pytest -q services/orion-cortex-exec/tests/test_docker_compose_service_bringup.py
15 passed   (run alone; collected together with test_skill_verbs it hits a pre-existing `app` module-name collision)

pytest -q services/orion-hub/tests -k "identity or memory_dir or scripts_dir"
18 passed
```

Not attributable to this patch: `orion/cognition/tests/test_packs.py`,
`test_planner.py`, `test_reflect_recall_integration.py` fail collection on
`main` with `No module named 'orion_cognition'`.

Gates:

```text
python scripts/check_env_template_parity.py   -> env template parity: PASS (85 service(s) compared)
git diff --check                               -> clean
graphify prs --conflicts                       -> No community overlap between open PRs
```

## Evals run

```text
None. There is no eval harness for the producer registry; the behavior
removed here produced no output to measure (the adapter returned None on
every call), so a before/after eval would show two identical belief sets.
The regression test above is the gate.
```

## Docker/build/smoke checks

```text
Not run. The change removes a producer that never emitted a node and one
empty env key; no dependency, port, health check, or compose wiring
changed. Runtime proof of "the producer is gone" will be the absence of
the `self_study` producer_id in the belief-set lineage on the next
cortex-exec / cortex-orch turn after redeploy -- UNVERIFIED until then.
```

## Review findings fixed

Review ran in a subagent against commit `5dce52672`. Verdict: no must-fix.

- Finding (should): `orion/substrate/relational/registry.py:39-41` -- the
  producer contract's own comment still listed `self_study` (and the
  already-dead `orionmem`) as live network-based adapters.
  - Fix: comment rewritten to name only `autonomy` as network-based and to
    record that `self_study`/`orionmem` are gone.
  - Evidence: `rg -n "self_study" orion/substrate/relational/registry.py`
    now hits only the "both gone" line.
- Finding (nit): local `services/orion-cortex-exec/.env:248` still carries
  `SELF_STUDY_NAMED_GRAPH=`; the sync script never prunes.
  - Fix: reported explicitly under Env/config changes as an operator
    hand-delete; nothing reads it.
  - Evidence: `rg SELF_STUDY_NAMED_GRAPH` finds no reader outside the new
    negative test.
- Finding (nit): `services/orion-hub/scripts/memory/narrative.yaml` is now
  a one-file orphan directory with the same zero-consumer status.
  - Fix: not deleted here -- the task scoped the kill to the two artifacts
    confirmed dead in the design doc; flagged under Risks as a follow-up.
  - Evidence: `rg narrative.yaml` across `services/ orion/ scripts/ tests/`
    returns nothing.
- Reviewer note, not a defect: no CI workflow runs
  `services/orion-cortex-exec/tests` or `orion/cognition/tests`, so the new
  regression test is a local gate only. Pre-existing gap.

## Restart required

cortex-exec and cortex-orch import the changed modules at boot; orion-hub
only loses a file nothing read, but its image bakes `scripts/` so a rebuild
keeps the container honest.

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: `projection_builder.py`'s registry is a live path in cortex-orch
    (Mind cold projection). Removing an entry there changes the producer
    list Mind sees.
  - Mitigation: the removed producer never returned a node, so the belief
    set is unchanged; `test_reducer_lane_adapters.py` and
    `test_projection_builder.py` pin the new list; cortex-orch's Mind tests
    pass.
- Severity: low
  - Concern: `services/orion-hub/scripts/memory/narrative.yaml` is the
    sibling of the deleted card and is equally unread.
  - Mitigation: out of scope here; left in place. Flagging so it can go in
    a follow-up rather than silently widening this PR.
- Severity: nit
  - Concern: `graphify-out/graph.json` was not refreshed in this PR.
  - Mitigation: the graph sits at GitHub's 100MB cliff (see
    `2026-09-08-graph-json-lfs-migration-pr.md`); a refresh here would
    block the push. Left for the LFS migration.

## PR link

(filled in after push)
