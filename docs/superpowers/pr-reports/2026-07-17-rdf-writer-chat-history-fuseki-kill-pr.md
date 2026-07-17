# PR report: stop writing chat.history / chat.history.message.v1 to Fuseki

## Summary

- Live investigation traced `orion:chat` Fuseki graph contents directly against Postgres: 299 `ChatTurn` nodes / 23 sessions vs 2,579 `chat_message` rows / 1,699 `chat_history_log` rows — roughly **11-18% coverage** of real chat volume. Almost none of the richer fields the builder supported (`model`, `intent`, `topic`, token counts) were ever populated as real triples; `timestamp` appeared on just 1 of 299 turns.
- `orion-recall` does issue live SPARQL queries against this graph (keyword/neighbor search in `rdf_adapter.py`) — a real consumer, unlike the metacog/cognition-trace case (PR #1155). But a consumer querying a graph missing ~85% of its source data isn't evidence the data is valuable; it's evidence recall's graph-based retrieval has likely been quietly degraded by this same gap. `rdf_adapter.py`'s dense `try/except` coverage (repo-standard fail-open pattern) means removing the writes makes results sparser over time, not a crash.
- Both kinds are already fully durable via `orion-sql-writer` (`chat_message`, `chat_history_log` tables).
- Removed both channels from `orion-rdf-writer`'s subscribe list, dispatch, and handler functions. `channels.yaml` needed no edit — `orion-rdf-writer` was never declared as a consumer there to begin with (verified, not assumed).
- Recorded the unexplained coverage gap as an open question in the split-design spec and filed an agent-board finding — the same upstream mechanism, if it's a producer-side drop rather than an rdf-writer-side one, could be starving other real consumers (`orion-vector-writer`, `orion-vector-host`, `orion-spark-concept-induction`) of the same missing ~85%.
- Code review caught and fixed a real doc-staleness issue: `README.md` still called chat "an acceptance canary" and pointed operators at a smoke test that will now always report `FAIL`.

## Outcome moved

`orion-rdf-writer` no longer writes a chronically under-populated, ~85%-incomplete copy of chat history to Fuseki. This is the third concrete "wrong-tool precedent" / redundancy cleanup in the same session's arc (drive-audit RDF kill precedent, metacog/cognition-trace kill in PR #1155, this one) — but unlike the first two, this one surfaces a genuine open question (why the coverage gap exists) rather than pure redundancy with zero consequence.

## Current architecture

Before this patch, `orion-rdf-writer` subscribed to `orion:chat:history:turn` (`chat.history`) and `orion:chat:history:log` (`chat.history.message.v1`), building RDF triples via `_handle_chat_turn`/`_handle_chat_message` into the Fuseki `orion:chat` graph. `orion-sql-writer` was independently, fully subscribed to both with real, complete SQLAlchemy models. `orion-recall`'s `rdf_adapter.py` issues live SPARQL queries against the same graph for chat memory retrieval — the only channel in this session's Fuseki-reduction arc with a confirmed real reader, not just a fail-open compression federator.

## Architecture touched

- `services/orion-rdf-writer/app/settings.py`: subscribe list, removed fields.
- `services/orion-rdf-writer/app/rdf_builder.py`: dispatch + handler removal.
- `services/orion-rdf-writer/.env_example` / `docker-compose.yml`: dangling key cleanup (done proactively this time, matching the review-finding pattern from PR #1155).
- `services/orion-rdf-writer/README.md`: contract doc accuracy (review finding).
- `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`: new open question.

## Files changed

- `services/orion-rdf-writer/app/settings.py`: removed `CHANNEL_CHAT_HISTORY_TURN`/`CHANNEL_CHAT_HISTORY_LOG` from `get_all_subscribe_channels()` and the fields themselves; added explanatory comment mirroring the drives-audit precedent.
- `services/orion-rdf-writer/app/rdf_builder.py`: removed the `chat.history`/`chat.history.message.v1` dispatch branches and the `_handle_chat_turn`/`_handle_chat_message` functions.
- `services/orion-rdf-writer/.env_example`, `docker-compose.yml`: removed the now-dangling keys.
- `services/orion-rdf-writer/README.md`: removed stale channel-table rows, corrected the "chat is an acceptance canary" claim, flagged the now-always-`FAIL` smoke test as expected/stale (review finding).
- `services/orion-rdf-writer/tests/test_autonomy_materialization.py`: two new channel-not-subscribed regression tests and two new quiet-no-op dispatch tests, mirroring the existing drives-audit precedent.
- `services/orion-rdf-writer/tests/test_service_rdf_store_integration.py`: removed both kinds from the parametrized "writes for kind" test (mocks the builder entirely — cosmetic, confirmed by review).
- `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`: new open question #3 recording the coverage-gap TODO.

## Schema / bus / API changes

- Added: none.
- Removed: `orion-rdf-writer` as a (never-declared) consumer of `orion:chat:history:turn`/`orion:chat:history:log` — no `channels.yaml` edit needed, confirmed it was never listed.
- Renamed: none.
- Behavior changed: `orion-rdf-writer` no longer writes the `orion:chat` Fuseki graph. `orion-recall`'s SPARQL-based chat retrieval will see progressively sparser (not growing) results from this graph going forward — a continuation of an existing, unexplained degradation, not a new one introduced by this patch.
- Compatibility notes: `orion:chat:social` (a separate graph, `social.turn.stored.v1` kind) is untouched — confirmed by review, byte-for-byte unchanged.

## Env/config changes

- Added keys: none.
- Removed keys: `CHANNEL_CHAT_HISTORY_TURN`, `CHANNEL_CHAT_HISTORY_LOG` from `services/orion-rdf-writer/.env_example`, `docker-compose.yml`, and the local `.env` (synced by hand in the primary checkout).
- Renamed keys: none.
- `.env_example` updated: yes, with an explanatory comment.
- local `.env` synced: yes, directly edited in the primary checkout.
- skipped keys requiring operator action: none.

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-rdf-writer \
  /tmp/orion-test-venv/bin/python3 -m pytest services/orion-rdf-writer/tests -q
→ 41 passed

git diff --check → clean
scripts/check_service_env_compose_parity.py orion-rdf-writer → same pre-existing
  17-key gap (unrelated: PROJECT, NET, Fuseki JVM tuning), no new gap
```

## Evals run

```text
No eval harness exists for rdf-writer's channel dispatch; focused deterministic
tests cover subscription list and dispatch behavior.
```

## Docker/build/smoke checks

```text
No Docker rebuild/restart performed for this PR. Two root-level smoke scripts
(scripts/smoke_chat_to_rdf.py, scripts/smoke_chat_to_rdf_store.py) will now
always report FAIL for their stated chat-history-round-trip purpose after this
service is redeployed -- flagged in the README as expected/stale, left as a
follow-up (shared with orion-rdf-store, a different service, so retiring/
repointing them is a separate decision, not a one-line fix for this patch).
```

## Review findings fixed

- Finding (should-fix): `README.md` left stale — still called chat "an acceptance canary," still documented both removed channels in its consumed-channels/env-var tables, and pointed operators at a smoke test that will now always `FAIL`.
  - Fix: removed stale table rows, corrected the canary claim, added explicit "stale as of 2026-07-17, will always FAIL, that's expected" notes.
  - Evidence: commit `bd7422d2`.
- Informational (not fixed, follow-up recommended): `scripts/smoke_chat_to_rdf.py`/`smoke_chat_to_rdf_store.py` and their mentions in `services/orion-rdf-store/README.md`/`scripts/README.md` are now dead for their stated purpose — shared with a different service, so retiring/repointing them is a real decision outside this patch's scope.

Reviewer also confirmed: no dangling references to removed symbols, `channels.yaml` needed no edit (verified `orion-rdf-writer` was never a declared consumer), new tests are real/unmocked, the parametrize-list removal is cosmetic (confirmed the fixture mocks the builder entirely), `uuid` import still needed elsewhere, `_handle_social_room_turn`/`orion:chat:social` untouched, and env parity holds.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-rdf-writer/.env \
  -f services/orion-rdf-writer/docker-compose.yml up -d --build
```

Not required for correctness before merge, but needed to actually stop the live redundant Fuseki writes.

## Risks / concerns

- Severity: Medium
- Concern: `orion-recall`'s chat retrieval via `orion:chat` graph will not gain new data going forward. Given the ~85% coverage gap was already present before this patch, this is a continuation of an existing degradation, not a new regression — but it's worth root-causing (open question #3 in the split-design spec, agent-board finding filed) since the same upstream mechanism may also be starving `orion-vector-writer`/`orion-vector-host`/`orion-spark-concept-induction`.
- Severity: Low
- Concern: Two root-level smoke scripts will now always fail their documented purpose. Flagged in README, not fixed (shared with a different service).

## PR link

<link — to be filled in after `gh pr create`>
