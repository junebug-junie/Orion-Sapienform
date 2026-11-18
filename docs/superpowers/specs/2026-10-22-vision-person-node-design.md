# One graph node per person, camera clusters kept for counting

Status: PROPOSAL (identity change; CLAUDE.md proposal mode). Not implemented.
Date: 2026-10-22. Follows PR #2842 (walkway day links no longer join on name).

## Arsonist summary

The camera splits one person into several appearance clusters. That is fine for
counting, and the day rollups already report it honestly as a range. It is wrong
for the graph: today every cluster Juniper names gets its own person-shaped
entity node, so one child can be two nodes, and Orion reads two nodes with one
name as "twins". Live 2026-10-22: two names each have two entity nodes for the
same child (Juniper confirmed: "just different pictures of them").

The fix is one person node per person, with each named cluster linked to it by
an assertion that Juniper's answer made. Clusters stay in Postgres and stay in
the graph as what the camera saw. A rename closes the old link instead of
leaving it walkable. No new identity bridge to chat memory: the existing
person-model linkage (`orion/person_model/linkage.py`, merge log with exact
unmerge) is the only place a camera person meets a chat-memory person.

## Current architecture

- **Clusters (Postgres).** `vision_individual` (orion-sql-writer) = one
  appearance cluster. `label` is the name Juniper typed (`label_copied=false`)
  or a name copied along a "same person" group (`label_copied=true`,
  `vision_individual_pair_verdict`). The writer's split
  (`app/vision_split.py`) moves crops to a new cluster on "someone else"
  picture verdicts.
- **Albums (Postgres).** `vision_person_exemplar` / `_bar` / `_attribution`
  (`app/vision_person_model.py`) already treat a person as `person_label`, the
  name string, across clusters and streams. So the camera side already has a
  person concept; the graph does not.
- **Graph.** `orion-substrate-runtime/app/ask_answered_listener.py` writes one
  `EntityNodeV1` per answered cluster (`entity:vision_individual:<sha(id)>`,
  `subject_ref=vision_individual:<id>`, adapter
  `orion/substrate/adapters/vision_individual.py`). Deliberately per cluster,
  never per label, so two different people with one name never merge.
  `walkway_seen_loop.py` adds `observed_in` edges from each sighted cluster to a
  day evidence node.
- **Edges cannot be retired today.** The store has no delete, and a
  `legacy_unreviewed` edge walks even with `valid_to` set
  (`orion/substrate/neighborhood.py:walkable_edge`). Only a
  `semantic_projection` edge stops walking when its Assertion is
  rejected/deprecated (`orion/substrate/assertion_projector.py`).
- **Person model (spec PR #2646, open).** The canonical person node is the
  chat-memory referent (`memory.referents`, `entity_type=person`), identity
  fenced: only a recorded merge decision joins it to another node.

### What goes wrong live

| Case | Today |
|---|---|
| One child, two clusters, each named separately | two entity nodes, same label |
| One cluster holding two people (an adult and a child) | named for the child, then renamed for the adult; the child's identity is gone from the graph |
| "Do you know whose vehicle?" answered "no" | a vehicle entity named "no" |

## Missing questions

1. Is a typed name enough to say "same person" across clusters, or does it need
   a "same person" card? **Recommendation: the name is enough, unless a
   `different` pair verdict exists between two clusters with that name.** The
   albums already make this call (`person_label`), and Juniper does not answer
   a card per cluster pair. Two real people with one name is the rare case. It
   is caught by a `different` verdict, or by Juniper later typing a
   distinguishing name ("<name> next door").
2. Do camera persons cross streams (walkway and doorbell)? **Recommendation:
   yes.** It's one household and one name space, as the albums already do.
3. Should day links move from clusters to the person? **Recommendation: no.**
   Clusters keep their `observed_in` day edges (what the camera saw, for
   accounting). A reader that asks "when did I see this person" walks
   person <- instance_of - cluster -> day. No duplicate edges, and no
   re-derivation when a cluster moves.

## Proposed schema / API changes

No new node kind, predicate, bus channel or table.

1. **Person node.** `EntityNodeV1`, `node_id = entity:vision_person:<sha256(fold(name))[:24]>`,
   `entity_type="person"`, `anchor_scope="world"`, `label` = the most recent
   typed spelling, `subject_ref = vision_person:<fold(name)>`. Created only for
   clusters whose detector kind maps to `person`
   (`entity_type_for_kind`), so a vehicle named "no" never becomes a person.
2. **Cluster -> person link as an Assertion.** For every typed or copied
   person cluster: one `SubstrateGraphProposalV1(proposal_kind="relationship_assertion")`
   "cluster instance_of person", plus a `SubstrateGraphDecisionV1` to
   `provisional`, `actor="juniper"` (authority `user_asserted`), evidence = the
   `orion_ask` that set the label. These are written to the existing
   `substrate_graph_journal` and projected by the existing `AssertionProjector`,
   which writes the `instance_of` edge with `edge_role=semantic_projection`.
   - **Rename / split.** The cluster's label changed, so a decision moves the
     old assertion to `deprecated`. Its edge stops walking (existing
     `walkable_edge`), and a new assertion links to the new person.
   - **Guard.** Two clusters with the same name and a `different` verdict
     between them: no link for the newer one. Log
     `vision_person_conflict name=<hash> a=<id> b=<id>` and keep both clusters
     unlinked until Juniper answers.
3. **Producer.** A small reducer pass in orion-substrate-runtime, next to
   `walkway_seen_loop.py` and on the same clock. It reads `vision_individual`
   (person kinds, `label IS NOT NULL`), `vision_individual_pair_verdict`, and
   the journal's current assertions for `vision_person:*`. It writes only the
   proposals and decisions whose state differs. It is idempotent: ids are uuid5
   of (cluster, person, label_ask_id).
4. **Bridge to chat memory (not here).** `entity:vision_person:*` is a normal
   entity candidate for `orion/person_model/linkage.py` once PR #2646's apply
   step exists. The fence and the merge log decide whether the walkway child is
   the child Juniper talks about. This patch does not join them.

## Privacy boundary

Names of household members and children, in the graph only. That is the same
data the cluster nodes already hold, no new source and no new reader. No crops
or embeddings enter the graph. Repo docs, tests and fixtures use placeholders
(`tests/test_no_personal_terms.py`).

## Files likely to touch

- `orion/substrate/adapters/vision_individual.py`: person node id/builder,
  proposal/decision builders.
- `services/orion-substrate-runtime/app/vision_person_loop.py` (new), wired in
  `app/main.py` beside the walkway_seen loop; `settings.py`, `.env_example`
  (`VISION_PERSON_LINK_ENABLED=true`, ships on).
- `services/orion-substrate-runtime/tests/test_vision_person_loop.py`.
- `services/orion-substrate-runtime/README.md`, `orion/substrate/README.md`
  (producer row).

## Non-goals

- Merging cluster rows, crops or albums in Postgres.
- Joining camera persons to chat-memory referents (person-model linkage owns
  that).
- Fixing "no" / "I do not know" answers on vision asks. The Hub card should
  offer an explicit "I don't know" button; that is a separate patch. This design
  only stops those answers from becoming persons.
- Splitting the mixed adult/child cluster. Juniper does that with "someone
  else" picture verdicts (existing `vision_split.py`). This design makes the
  result land correctly.

## Dangerous failure modes

- **Two real people merged under one name.** Mitigated by the `different`
  verdict guard. Reversible: a decision to `deprecated` un-walks the link, and
  no node is deleted.
- **Stale links after a rename read as current.** Mitigated by semantic
  projection (closed, unwalkable). A test pins it.
- **Projector not ready.** `AssertionProjector.readiness()` gates every write.
  Nothing is written until the readers can read the new edge roles.

## Rollback

`VISION_PERSON_LINK_ENABLED=false` stops new proposals. Deprecating every
`vision_person:*` assertion un-walks every link. Person nodes left behind have
no walkable edges and are inert.

## Acceptance checks

- **Unit:** two clusters, one name -> one person node, two `instance_of` edges.
  A rename -> the old edge is unwalkable through `walkable_edge` and the new one
  walks. A `different` verdict -> no second link plus the conflict log. A
  vehicle named "no" -> no person node.
- **Replay:** run the reducer read-only against live `vision_individual`. It
  should report person count = distinct folded typed person names, and 0 links
  for vehicles.
- **Live (after deploy):** each of today's double-node names has one person node
  with two walkable `instance_of` edges. The renamed cluster links to the
  adult's person node, and the child has none until split. Log line
  `vision_person linked people=N links=M deprecated=K`.
- **Orion-facing:** the next day letter no longer calls one child "twins". This
  is UNVERIFIED until a letter is written after deploy.

## Recommended next patch

Implement items 1-3 behind `VISION_PERSON_LINK_ENABLED` (ships on), with the
unit and replay checks above. Then ask Juniper for the one split: mark the
child's pictures "someone else" on the next card that shows them.
