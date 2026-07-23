# PR report: retire `orion:chat:social` from `EpisodicFederator`

## Summary

- Fourth in today's series of Fuseki-dependency retirements. Scoped narrower than the prior two `orion-graph-compression` PRs: `EpisodicFederator` itself stays live (still reads 8 other graphs), only its `orion:chat:social` graph entry is removed.
- Live evidence: Fuseki's `orion:chat:social` holds ~154 triples / ~12 `SocialRoomTurn` nodes, metadata-only (no message text). Postgres `social_room_turns` already owns richer content (33 rows, real `prompt`/`response`/`text` columns) with zero other readers of the Fuseki copy anywhere in the repo. The underlying social-chat feature also looks near-dormant (latest Postgres row ~20 days stale relative to today).
- A separate, already-live 2026-07-18 fix (`orion-meta-tags`) already writes social-turn tags/entities into the same `orion_recall` Falkor graph `episodic_falkor.py` reads -- but live Cypher queries found this thin (2 social-sourced `ChatTurn` nodes, zero tag/entity edges, vs 1,772 for `chat.history`). Disclosed as thin rather than overclaimed as full coverage, both in code comments and in the explicit tradeoff presented before implementing.
- This was a genuine three-way judgment call (kill now / investigate the thin Falkor write first / leave alone), presented and confirmed explicitly rather than assumed.

## Outcome moved

`EpisodicFederator`'s SPARQL dependency narrows from 9 graphs to 8. Combined with today's other three PRs (`orion-recall`, `orion-vision-scribe`, and the two fully-retired graph-compression federators), the only remaining genuinely-open Fuseki dependency in `orion-graph-compression` is collapse/cognition/metacog content within the still-live `EpisodicFederator` -- cognition/metacog are already dead-writer/frozen-history (killed at the source in an earlier PR), leaving collapse as the one real remaining gap.

## Current architecture

Before this patch: `EpisodicFederator` unconditionally queried 9 named Fuseki graphs via UNION, including `orion:chat:social`.

## Architecture touched

- `services/orion-graph-compression/app/federators/{episodic.py,episodic_falkor.py}`, `app/worker.py`, `.env_example`, `README.md`, `tests/test_federator_episodic.py`

## Files changed

- `services/orion-graph-compression/app/federators/episodic.py`: removed `http://conjourney.net/graph/orion/chat/social` from `EPISODIC_GRAPHS` (9 -> 8 entries), with an explanatory comment citing the live evidence.
- `services/orion-graph-compression/app/federators/episodic_falkor.py`: docstring correction -- previously claimed `orion:chat:social` "has no Falkor equivalent yet" (implying pending future work); now accurately describes it as retired outright, with the live-verified thin-Falkor-coverage numbers disclosed rather than glossed over.
- `services/orion-graph-compression/app/worker.py`: review finding -- a comment at the `episodic` scope's Falkor-union call site still lumped "social" in with collapse/cognition/metacog as "no Falkor equivalent yet." Corrected: those three genuinely have none; social was retired, not pending.
- `services/orion-graph-compression/.env_example`: same review finding, same fix, in `GRAPH_COMPRESSION_EPISODIC_FALKOR_ENABLED`'s comment (which explicitly cross-references the README section this PR already corrected -- was pointing readers at a section that would have contradicted it).
- `services/orion-graph-compression/tests/test_federator_episodic.py`: generalized a docstring's "9 graphs" reference (the assertions themselves were already graph-count-agnostic, iterating `EPISODIC_GRAPHS` dynamically); added `test_episodic_federator_no_longer_queries_chat_social`, a regression test asserting the URI is absent -- confirmed by review to be the only thing actually pinning the graph count/membership (the pre-existing UNION-count test would still pass with 9 graphs).
- `services/orion-graph-compression/README.md`: Compression Scopes table (8 graphs, noted the removal) and FalkorDB federators section (expanded explanation, live coverage numbers).

Deliberately not touched (confirmed by review against the live repo state, not just asserted):
- `orion-rdf-writer`'s own `SocialRoomTurn`/`SocialConceptEvidence` Fuseki write -- still writes there; this PR only stops graph-compression reading it back.
- `stale_listener.py`'s `orion:chat:social` -> `episodic` mapping -- left in place, since the write itself is untouched and still real (unlike the fully-dead `self_study` mappings removed in an earlier PR today).
- `orion-meta-tags`' social-turn Falkor tag/entity write -- pre-existing, unrelated to this session's work, untouched.

## Schema / bus / API changes

None.

## Env/config changes

None (no new/removed/renamed keys -- comment-only edit to `.env_example`).

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-graph-compression venv/bin/python3 -m pytest services/orion-graph-compression/tests -q
-> 46 passed (45 + 1 new regression test)

git diff --check -> clean
```

## Evals run

No eval harness exists for this service; the new regression test plus the existing dynamic-count UNION test cover the changed behavior.

## Docker/build/smoke checks

No container rebuild/restart performed. Live evidence backing this decision (Fuseki triple counts, Postgres row counts/recency, live FalkorDB Cypher queries) was gathered directly against the running containers in this same session, prior to writing this diff.

## Review findings fixed

- Finding (should-fix, low): two comments outside the files this diff directly touches (`app/worker.py`'s `_process_scope` Falkor-union call site, `.env_example`'s `GRAPH_COMPRESSION_EPISODIC_FALKOR_ENABLED` comment) still described `orion:chat:social` as lumped with collapse/cognition/metacog under "no Falkor equivalent yet" -- implying it was pending future migration rather than retired outright. The `.env_example` instance was worse: it explicitly cross-referenced the README section this same PR had already rewritten to say the opposite, creating a direct self-contradiction for anyone following that reference.
  - Fix: both corrected to match the framing already applied in `episodic.py`/`episodic_falkor.py`/`README.md` -- collapse/cognition/metacog genuinely have no Falkor equivalent; social was retired, not pending.
  - Evidence: commit, same PR.
- Informational (no fix needed, verified true): all three "deliberately not touched" claims (rdf-writer's write, stale_listener's mapping, meta-tags' Falkor write) confirmed accurate against direct inspection of the current repo state, not just trusted from the diff's own description.
- Informational (no fix needed): no other file in the repo assumes `EpisodicFederator` covers 9 graphs or specifically covers `orion:chat:social`; no `CompressionRegionV1` consumer assumes episodic-scope clustering includes social-turn content specifically.

## Restart required

No restart required to merge. If `orion-athena-graph-compression` is redeployed from this branch/main, the next `episodic` scope tick will simply query one fewer graph -- no other behavior change.

## Risks / concerns

- Severity: Low.
- Concern: social-turn clustering signal (already small: ~12 turns' worth of metadata) is now gone from the `episodic` scope entirely, with only the thin, disclosed Falkor coverage (2 nodes, 0 tag/entity edges today) standing in. If social-chat traffic picks back up meaningfully, this thin Falkor coverage may need real attention (either confirming the meta-tags write path fires correctly at volume, or building a dedicated migration) -- not urgent given the current near-dormant traffic pattern, but worth a fresh look if that changes.
- Mitigation: disclosed explicitly in README.md and inline comments rather than silently assumed equivalent; easy to revisit later since nothing about this change is hard to reverse (Fuseki's `orion:chat:social` graph is untouched, just no longer read).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1297
