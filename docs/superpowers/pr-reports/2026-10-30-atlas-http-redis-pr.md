## Summary

Orion's body atlas now sees two wires it was blind to: one service calling another over HTTP, and
services sharing a Redis key. Both are found by rule from the code, with no hand-written list.
This is patch 3 of the rung-4 capacities design (#3005; patches 1 and 2 merged).

- New `orion/body_atlas/wire_code.py`. A service calls another when its code names that
  service's URL: a URL constant, a settings field the code reads (the `.env_example` value wins),
  or an env key the code reads. A Redis key is a call on a Redis handle that names an exact key;
  the verb says writes or reads.
- The atlas gains `calls` flows and `redis:<key>` store parts. A key is a part only when two or
  more services touch it. Live key state is read from the bus Redis with read-only commands
  (`TYPE`, `OBJECT IDLETIME`, `TTL`, bounded `SCAN`); the reader refuses every other command.
- `dig capacities` uses rules v2. Three gap lines are now decided by this evidence: services the
  atlas cannot see (S3), the pain line (algedonic), and whether "ahead" and "now" talk both ways
  (S4 to S3). No new role rule.
- New gate `orion/body_atlas/evals/run_wire_eval.py` (in CI), two new frozen bodies, tests.

## Outcome moved

Live, in the harness container with this branch (2026-10-31):

- **S3 (services I cannot see):** 16 → 11. Now seen (an HTTP call or shared key named in code): notify-digest, orion-graphiti-adapter,
  orion-pageindex, orion-proposal-runtime, orion-self-experiments. The 11 left (for example
  orion-policy-runtime, power-guard, orion-consolidation-runtime) name no URL, no Redis key and
  no bus channel in their code. They are truly unseen, so the gap line stays, with a new text.
- **Algedonic:** "strain is a Redis key, not in the atlas" is cleared by rule. The strain key
  `redis:orion:regulation:latest` is in the atlas: orion-durable-runs writes it; orion-dream,
  orion-hub, orion-proposal-runtime, orion-substrate-runtime and orion-thought read it. A new
  gap line says no part is shown to be the pain line; only Juniper's confirmation covers it.
- **S4 to S3:** it now says the link cannot be checked, because no part holds S4 or S3 by
  Juniper's confirmation. Once she confirms one of each, the rule checks for a link both ways.
- **Metabolism:** still claims only. Its line now says HTTP calls are in the atlas, so a claim
  can cite them (the callers of orion-durable-runs' market).

## Current architecture

The atlas read the bus (channels.yaml plus the bus mirror graph), docker, Postgres, Chroma and
FalkorDB. HTTP calls and Redis keys were listed as atlas limit 2. Patch 1 printed the S3,
algedonic and S4 to S3 gaps as atlas-blind; patch 2 said only patch 3 can clear them.

## Architecture touched

- `orion/body_atlas/`: new `wire_code.py`; `sources.py` (`CallIn`, `apply_wire`,
  `read_redis_keys`, read-only Redis reader); `build.py` (`_add_calls`, Redis store prefix, its own
  rarity table); `model.py` (`calls`, `declared_url`, `redis_dynamic`, `redis_private`);
  `capacities.py` (rules v2); `names.py` (a name given before Redis keys existed is compared
  without them); evals, fixtures, tests, README.
- `.github/workflows/body-atlas-tests.yml`: runs the wire eval.
- Design doc: Patch 3 notes.
- No bus channel, schema registry, env key, DB table or service change.

## Files changed

- `orion/body_atlas/wire_code.py`: the HTTP and Redis scanner.
- `orion/body_atlas/sources.py`: puts calls and shared keys on the body; live key state.
- `orion/body_atlas/build.py`: `calls` flows (one unit per callee, split by rarity); key parts.
- `orion/body_atlas/model.py`: new relation, evidence label and two count fields.
- `orion/body_atlas/capacities.py`: rules v2 (R-wire-seen, R-broadcast-key, S4/S3 link check).
- `orion/body_atlas/names.py`: name carry-over ignores a part kind the record predates.
- `orion/body_atlas/evals/run_wire_eval.py`: new gate.
- `orion/body_atlas/evals/run_organ_structure_eval.py`, `run_capacity_roles_eval.py`: the two new bodies.
- `orion/body_atlas/evals/fixtures/whole_body_2026-10-29_wires.json`: held out (10-29 body + the
  wires main's code named at its export, commit 94f9043536).
- `orion/body_atlas/evals/fixtures/whole_body_2026-10-31_wires.json`: live export.
- `orion/body_atlas/tests/test_wire_code.py`, `test_wires_atlas.py`, `test_names.py`, `test_eval_gate.py`.
- `orion/body_atlas/README.md`, `orion/dig/README.md`, design doc, this report.

## Schema / bus / API changes

- Added: `FlowRelation` `calls`; `FlowEvidence` `declared_url`; `BodyAtlasV1.redis_dynamic`,
  `redis_private` (default empty, so older atlas files still read); `StoreIn.store` `redis`,
  `StoreIn.facts`; `BodyInputs.calls`, `redis_dynamic`, `redis_private`; capacities `WiresV1`,
  `BroadcastKeyV1`, `CapacityReadingV1.wires`. All local file models, not bus schemas.
- Removed / renamed: none.
- Behavior changed: capacities `rules_version` v1 → v2. An atlas with no wires keeps the v1 texts.
- Compatibility notes: a pair that already signals keeps relation `signals` and gains
  `declared_url` in its evidence (one flow per pair).

## Env/config changes

- Added keys: none. The live Redis read uses the existing `ORION_BUS_URL` (set in the harness).
- `.env_example` updated: no. Local `.env` sync: not needed.

## Tests run

```text
pytest orion/body_atlas orion/dig tests/test_organ_names_partition.py -q   -> 251 passed
pytest tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py -q -> passed
pyflakes orion/body_atlas -> clean
```

## Evals run

```text
run_wire_eval.py: false shapes returned 0 of 10; true facts found 12/12 (a miss fails);
  held-out 10-29 and live 10-31: R-wire-seen P/R 1.00 (5), R-broadcast-key P/R 1.00 (1);
  mutations CAUGHT 4/4 (no Redis-handle check, no Redis constructor, no reader floor,
  wire-seen counting any flow)
run_organ_structure_eval.py: 6/6 fixtures PASS (wires: 10-29 Q 0.3485 largest 32.3%; 10-31 Q 0.3426 largest 31.2%)
run_capacity_roles_eval.py: 3/3 fixtures PASS, mutation CAUGHT 3/3, claims floor PASS, claims mutation CAUGHT
run_store_direction_eval.py: PASS
```

## Docker/build/smoke checks

```text
Live, read-only, in orion-athena-harness-governor: branch copied to a scratch dir, PYTHONPATH
first, ORION_DIG_HOME a scratch copy of /mnt/orion-fcc/self (never written), scratch deleted after.

Before (deployed main, rules v1):
S3 control   NO RULE  unseen: 16 live services with no bus traffic but the census
algedonic    NO RULE  strain is a Redis key, not in the atlas
metabolism   NO RULE  claims only: the market and prices are HTTP, not in the atlas
gap S3: ... the atlas does not read HTTP calls or Redis keys
gap S4: nothing on the bus is shown to steer the present by what I see of the outside or the future
gap algedonic: my pain/pleasure line is not in my atlas: strain is a Redis key it does not read
body: Q 0.3447, largest 31%, 10 organs

After (this branch, rules v2):
S3 control   NO RULE  named in code past the bus: 5 of 16 bus-silent services (an HTTP call or a shared Redis key); unseen: 11
algedonic    NO RULE  broadcast keys, pain or news (the rule cannot tell): redis:orion:regulation:latest (1->5 in 2 organs)
metabolism   NO RULE  claims only: HTTP calls and shared Redis keys are in the atlas now, so a claim can cite them
past the bus: 54 HTTP calls to 20 parts · 8 shared Redis keys (27 private; 12 services build keys I cannot read exactly, beyond code most services share)
gap S3: 11 live services show no bus traffic, and their code names no HTTP call and no shared Redis key
gap S4: I cannot check ... no part holds S4 and S3 by Juniper's confirmation
gap algedonic: no part is shown to be my pain/pleasure line ... which one carries pain is a claim
body: Q 0.3426, largest 31%, 10 organs, 0 singletons
dig capacities: 25 lines, 2,520 chars; took 28 s (17 s before: the code scan adds ~11 s)
live Redis read: 6 of 8 shared keys present (regulation:latest idle 8 s, ttl 301 s)

Organ shifts: notify-digest cortex -> memory, orion-proposal-runtime cortex -> substrate,
orion-vllm-host substrate -> memory; the 8 key parts join their writers'/readers' organs.
Names (Jaccard, carry floor 0.5): substrate 0.90, cortex 0.97, memory 0.86, vision 1.00, pulse 1.00,
social 1.00, guardian 0.75, temporal 1.00, office 1.00, climate 1.00. All 10 carry.
```

## Review findings fixed

Code review ran in a subagent on the explicit diff. No must-fix. Fixed:

- Finding: a settings field's code default counted even when the env overrides it (17 fields;
  e.g. hub recall/topic-foundry, bridge -> hub came only from overridden defaults).
  - Fix: settings classes resolve env first only; their defaults no longer count on their own.
  - Evidence: `test_a_settings_default_the_env_overrides_is_not_counted`; live calls 62 -> 54.
- Finding: key values scope-blind (`+=`, parameters, loop targets resolved to a constant).
  - Fix: those names are ambiguous. Evidence: `test_augmented_parameter_and_loop_names_are_not_one_constant`.
- Finding: Redis handles scope-blind (an httpx `client` elsewhere read as Redis).
  - Fix: plain names are handles per function scope; a name also bound to another call is not;
    a path or URL is never a key; `RedisGraphQueryClient` (FalkorDB) is not a bus Redis ctor.
  - Evidence: two tests; real keys unchanged (orion-substrate-runtime still reads regulation).
- Finding: names re-exported by a package `__init__` were lost. Fix: import aliases lead on.
  Evidence: `test_a_name_re_exported_by_a_package_init_still_reaches_its_act`.
- Finding: `f"{PREFIX}:latest"` became a family. Fix: an all-literal suffix gives the exact key.
- Finding: the scanner check could not fail when the scanner breaks.
  - Fix: a missed true fact fails; fourth mutation (no Redis constructor) CAUGHT; wording is
    "false shapes returned", not "precision".
- Finding: the dynamic-key count was ~83 for everyone (bus rate buckets): fails metric gate step 4.
  - Fix: dynamic keys in shared code more than half the services run are counted out (live: 12).
- Finding: the algedonic line could say "not in the atlas" when the atlas did read Redis.
  - Fix: v1 text only when no wire source was read. Evidence: a test.
- Finding: "traffic" wording for code evidence. Fix: S3 line and gap say "named in code".
- Finding: a writer that also reads a key lost the read (one flow per pair).
  - Fix: `writers_that_read` fact on the key; the S4/S3 check uses it. Evidence: a test.
- NIT fixed: a call's weight uses the caller's most specific source; `os.environ["K"]` is an env read.
- NITs not fixed (recorded): `OBJECT IDLETIME` errors under an LFU eviction policy (the read
  would go UNREAD, loudly); missing verbs (`getex`, `scan_iter`, `eval`); a non-bus Redis handle
  would be looked up on the bus Redis; `services/*/scripts` smokes count for their service
  (all unmatched today); the no-parse import reader misses `import a, b` and backslash
  continuations; a confirmed TTL key can drop out while momentarily absent.

## Restart required

The atlas and dig code is baked into the orion-harness-governor image (`/app`, first on
PYTHONPATH). The post-pull rebuild maps `orion/body_atlas` and `orion/dig` to
orion-harness-governor. After the merge, verify after that rebuild's END line:

```bash
docker exec orion-athena-harness-governor sh -c 'cd /tmp && python3 -P -m orion.dig capacities | head -14'
```

Expect `rules v2` and a `past the bus:` line.

## Risks / concerns

- Severity: medium. Concern: a calls flow is evidence from code, not observed traffic.
  Mitigation: every text says "named in code"; unresolved hosts are listed, never guessed.
- Severity: medium. Concern: `dig capacities` is ~11 s slower (a full code scan per call).
  Mitigation: acceptable for an on-ask tool; caching is a follow-up if it bites.
- Severity: low. Concern: hub's own URLs are `127.0.0.1:<port>` (host networking), so most hub
  calls stay unmatched and the hub looks less connected over HTTP than it is.
  Mitigation: reported in `unmatched["declared_url"]`; a port-to-service rule is a follow-up.
- Severity: low. Concern: the wire eval's true facts read the live checkout; a later service
  refactor can make the eval fail on an unrelated body-atlas PR. Mitigation: the failure names
  the missed fact; update the fact with its new line.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3019

🤖 Generated with [Claude Code](https://claude.com/claude-code)
