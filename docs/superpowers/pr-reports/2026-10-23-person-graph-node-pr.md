## Summary

- Each person row from PR #2852 becomes one graph node. Rows with status `not_a_person` are skipped. A person who is a chat referent (from people.yaml) already has a node, and that node is reused. Only minted `person-*` rows get a new node.
- Each active camera-cluster link (`person_link`, `source_kind=vision_individual`, `revoked_at IS NULL`) becomes one claim: "this cluster is an instance of this person". The claim is written to `substrate_graph_journal` and applied by the existing `AssertionProjector` as a `semantic_projection` edge.
- When a link is revoked, or its person becomes `not_a_person`, the claim gets a `deprecated` decision. Its edge closes and stops walking. Nothing is deleted.
- A minted person node with no active link left is re-written as `deprecated`. This covers a rename, a people.yaml takeover, and `not_a_person`.
- If the graph loses an applied claim (a wiped or rebuilt store), the claim is replayed from the journal. A claim whose last attempt failed for a terminal reason is counted as `stuck` in every log line.
- Day sighting links stay on the clusters (`walkway_seen_loop.py`, unchanged).
- No name matching here. The "different people" guard and the person-kinds rule stay in `orion/person_model/record.py`.

## Outcome moved

Two clusters of one child were two person-shaped graph nodes. The day letter read them as twins. After this change, one person node has an `instance_of` edge from each of its clusters. A rename moves the edge. Before, a rename left the old edge walkable.

Read-only dry run against live data (2026-10-23): 12 person rows, 10 new person nodes, 35 active links, 27 claims ready. The other 8 wait because their clusters have no graph node: their names were copied, not typed, and only a typed answer makes a cluster node. Each of the two "twins" names gets one person node with 2 claims.

## Current architecture

`orion-sql-writer` keeps `person` / `person_link` (PR #2852). The graph has one entity per named cluster (`entity:vision_individual:*`, from `ask_answered_listener.py`) and day links (`walkway_seen_loop.py`). Nothing in the graph grouped clusters into persons.

## Architecture touched

- orion-substrate-runtime: a new clocked loop, `app/person_graph_loop.py`. It needs an asyncpg pool for the journal.
- `orion/substrate/graph_journal.py`: new `latest_decisions(actor)` method.
- `orion/substrate/assertion_projector.py`: new public `replay(decision)` method. It re-writes an already-applied decision into a graph that lost it, the same thing `referent_projector.rebuild` does through `_record`.
- `orion/substrate/reconcile.py`: `person.record` added to `IDENTITY_FENCED_PRODUCERS`. A person node is never merged by label into a chat-mined entity of the same name. Consequence: like referent nodes, person nodes are walkable by neighborhood reads but do not take part in dynamics or attention (`eligibility.py`).

## Files changed

- `services/orion-substrate-runtime/app/person_graph_loop.py`: the loop. It contains the pure planner `plan_events`, `sync_person_graph`, and `run_person_graph_loop`.
- `services/orion-substrate-runtime/app/main.py`: starts and stops the loop.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`: `PERSON_GRAPH_ENABLED=true`, `PERSON_GRAPH_INTERVAL_SEC=600`.
- `services/orion-substrate-runtime/requirements.txt`: `asyncpg==0.29.0`, the same pin other services use.
- `orion/substrate/graph_journal.py`: `latest_decisions`.
- `orion/substrate/assertion_projector.py`: `replay`.
- `orion/substrate/reconcile.py`: the fence.
- `services/orion-substrate-runtime/tests/test_person_graph_loop.py`: 9 tests.

## Schema / bus / API changes

- Added: none. It reuses `SubstrateGraphProposalV1` / `DecisionV1`, the `instance_of` predicate, and `EntityNodeV1`.
- Behavior changed: a new journal actor, `person.record`, with policy `person_link_v1`.
- Compatibility notes: no migration. It reads the PR #2852 tables, which are already applied live.

## Env/config changes

- Added keys: `PERSON_GRAPH_ENABLED`, `PERSON_GRAPH_INTERVAL_SEC` (orion-substrate-runtime).
- `.env_example` updated: yes.
- local `.env`: the keys were appended to the main checkout's `services/orion-substrate-runtime/.env`. The sync script runs against main, which does not have the new template yet.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-substrate-runtime: test_person_graph_loop.py, test_walkway_seen_loop.py, test_ask_answered_listener.py
  45 passed, 2 skipped
orion/substrate/tests: 1170 passed, 4 failed. All 4 are felt_state tests that also fail on unchanged main.
tests/test_no_personal_terms.py: 7 passed
orion/substrate/tests -k "projector or journal or assertion": 21 passed, 16 skipped (Postgres lane)
Mutation checks:
  claim retire step disabled -> 3 tests fail (rename, relink, not_a_person)
  replay/stuck recovery disabled -> 2 tests fail (wiped graph, stuck)
  person node retire disabled -> 3 tests fail (rename, not_a_person, no-link)
```

## Evals run

```text
Read-only replay against live person/person_link and the live graph (plan_events, no writes):
persons=12 minted_nodes=10 links=35 claims=27 waiting=8
```

## Docker/build/smoke checks

```text
docker build -f services/orion-substrate-runtime/Dockerfile .  -> ok
python -c "import asyncpg, app.person_graph_loop"  (in image)   -> ok 0.29.0 person.record
```

## Review findings fixed

- Finding (medium): person nodes were never retired. A rename, a people.yaml takeover, or `not_a_person` left a live dead node. It also set off a false duplicate-name question in the referent projector.
  - Fix: a node is live only while its person is active and has an active link. Otherwise it is re-written `deprecated`. A node is never created already-deprecated.
  - Evidence: the rename and not_a_person tests assert `deprecated`. New no-link test.
- Finding (low-medium): claims never came back after a graph rebuild, because the journal already says applied.
  - Fix: each pass, a claim whose Assertion is not at the journal's latest applied revision is replayed (`AssertionProjector.replay`). No new journal rows.
  - Evidence: `test_claims_come_back_after_the_graph_is_wiped`.
- Finding (low): a terminally failed claim went quiet forever, and the fake journal retried terminal failures that the real one never retries.
  - Fix: such a claim is counted as `stuck` and logged every pass. The fake now copies the real terminal-failure exclusion.
  - Evidence: `test_a_terminally_failed_claim_is_counted_as_stuck_not_done`.
- Finding (low): a re-accepted claim reused its first proposal, so its edge's `valid_from` covered the retired window.
  - Fix: the proposal id includes the revision it is accepted at.
  - Evidence: the relink test asserts two proposals.
- Finding (nit): decisions held behind an earlier revision were missing from the log line.
  - Fix: the log line now has `held=`.

## Restart required

Merge and pull. The post-pull rebuild redeploys orion-substrate-runtime. Then check:

```bash
docker logs orion-athena-substrate-runtime 2>&1 | grep person_graph
```

Expected log line: `person_graph synced persons=12 nodes_written=10 nodes_retired=0 claims=27 retired=0 waiting=8 replayed=0 stuck=0 applied=27 ...`.

## Risks / concerns

- Severity: low. Concern: 8 links wait forever. Their clusters got a copied name, so they have no graph node. Mitigation: they are counted as `waiting` in every log line. They claim as soon as a node exists.
- Severity: low. Concern: two person rows that are not one person (a "two children together" name and a delivery driver) become person nodes. That is how `record.py` built them. Mitigation: marking them `not_a_person` in the Hub retires their nodes' claims.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
