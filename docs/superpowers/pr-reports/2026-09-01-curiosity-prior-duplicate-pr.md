# A prior_id is an identity, so forming one must MERGE

Status: **DONE_WITH_CONCERNS** — code complete and verified against live FalkorDB;
the two forked nodes already in `orion_worldview` are NOT repaired, because Hub
never writes to that graph and the repair is a production write.

## Summary

- The kickoff prompt told Orion to form a prior with `CREATE (:Prior {...})`.
  Run `ed05344f8a39` meant to refute a claim it already held, emitted that
  CREATE, and forked the node. `concept_induction_overload_rate` now exists
  twice, reading `tested 1x` and `tested 6x`.
- The template is now `MERGE (p:Prior {prior_id: "..."}) ON CREATE SET ...`,
  on the id **alone** — any other property inside the pattern is part of the
  match key and would fork the node again on a changed `formed_from`.
- A duplicate census now runs in `read_snapshot` across the live **and** closed
  reads, and the settled list is deduped.
- `COUNTS_CYPHER` counts distinct claims instead of nodes.
- 24 new tests; 14 mutations verified red.

## Outcome moved

One claim is one claim. Before: a fork silently split a claim's history and its
evidence, and nothing anywhere said so.

## Current architecture

Orion writes its own Cypher into `orion_worldview` (FalkorDB `127.0.0.1:6380`).
`kickoff_prompt.py` builds the instructions; `worldview.py` reads the graph back
and assembles the menu the next run sees. Hub never writes to that graph — which
is why containment and repair are different problems here.

## The defect, end to end

`prior_id` is an identity. The prompt tests a claim with:

```cypher
MATCH (p:Prior {prior_id: "..."}) SET p.times_tested = p.times_tested + 1
```

That binds to **every** node carrying the id. So a duplicate does not merely show
a claim twice — the increment hits each copy from its own base and the claim's
history forks permanently.

Evidenced, not inferred. Run `ed05344f8a39`'s own journal says *"I've recorded
these findings in the graph and refuted the overload prior"* and its footprint
line reads `Wrote to its own graph: -> SUPPORTS 2, Finding 1, Prior 1` — it
formed a node for a claim it already held. The next run `8e3ff4c7b4b2` then ran
the MATCH+SET, which is why **both** copies now carry `last_run_id =
8e3ff4c7b4b2` and `status = refuted` from different `times_tested` bases.

The evidence forked too. Live:

| copy | run_id | times_tested | inbound edges |
|---|---|---|---|
| A | `ba4003577bb5` | 1 | 3 SUPPORTS |
| B | `ed05344f8a39` | 6 | 2 SUPPORTS |

Asking what supports that claim returns 3 findings or 2, depending on which node
binds. Never 5.

## Review findings fixed

- **Finding (S1, HIGH): the containment did not cover the incident it cites.**
  Both copies are `refuted`, so `_LIVE_WHERE` excludes them, `select_priors`
  never sees them, and the warning that justified shipping without a repair
  could not fire. The fork reached Orion only via `RECENT_SETTLED_CYPHER`,
  which had no dedup — one claim filling two of eight settled slots.
  - Fix: census moved to `read_snapshot` across both reads; `dedupe_settled_rows`.
  - Evidence: `test_a_fork_hiding_entirely_in_the_closed_half_is_still_reported`;
    mutations "census over live rows only" and "skip the settled dedup" both red.
  - I had found this half myself before the review returned; the review found
    the rest.

- **Finding (S2, HIGH): a cross-status fork is invisible to any per-list check.**
  Right after a run refutes one copy, each query returns exactly one, so no list
  contains a duplicate. That is the fork that renders one `prior_id` under both
  "still unsure of" and "recently settled" — and the settled block invites Orion
  to reopen it by an id that binds both copies.
  - Fix: the census is over `prior_rows + settled_rows`, so the two halves meet.
  - Evidence: `test_a_fork_split_across_live_and_closed_is_reported`.

- **Finding (S3, MED-HIGH): the prompt tests were vacuous.** `ON CREATE SET` →
  `SET` left the suite green. That mutation is **worse than the bug being
  fixed**: a bare SET re-runs `p.times_tested = 0, p.confidence = 0.55` on every
  re-statement, so the count resets, the prior never reaches `stale_after`, it is
  never retirable, and the recorded confidence is wiped to the template default.
  `ON MATCH SET` and `CREATE (p:Prior {` under a variable were also green.
  - Fix: assert `ON CREATE SET` present, `ON MATCH SET` absent, every property
    written, and a regex ban on `CREATE (<var>:Prior {`.
  - Evidence: all three mutations now red.

- **Finding (S4, MED): the silent no-op has no symptom.** A bound MERGE writes
  nothing, so `run_footprint_cypher` reports the run wrote no prior at all —
  indistinguishable from a run that chose not to form one.
  - Fix: the prompt now says so, and says a genuinely new claim needs a new id
    while a claim already held is the MATCH below.
  - Evidence: `test_prompt_warns_that_a_bound_merge_writes_nothing`.

- **Finding (S5, MED): the documented tie rule was false.** It claimed ties break
  on `prior_id` — which the copies share by definition. Ties are the *common*
  case, since the template writes `times_tested = 0`, and the real tiebreak was
  FalkorDB's row order from an unordered `MATCH`.
  - Fix: `(times_tested, last_tested_at, claim)` — content-based and
    order-independent. Docstring rewritten to state what is actually guaranteed.
  - Evidence: `test_collapse_is_not_decided_by_row_order_on_a_full_tie`.

- **Finding (S6, MED): "most-tested wins" is not "the real one wins".** The
  winner is picked whole, so a newer copy holding a revised belief at
  `times_tested = 0` loses to a stale copy at 6.
  - Fix: documented and pinned rather than papered over. Merging field-by-field
    would synthesise a claim no run ever wrote, which is worse. The real remedy
    is that forks stop being created and existing ones get repaired.
  - Evidence: `test_collapse_can_show_a_stale_claim_when_the_copies_disagree`.

- **Finding (S7, LOW-MED): the prompt taught two opposite MERGE semantics.**
  "A MERGE binds by exact string or it binds to nothing, silently" is true for
  the edge MERGE and false for the new node MERGE, where a mistyped id forms a
  second claim rather than failing loudly.
  - Fix: reworded to distinguish the two, and "Each CREATE is independent" →
    "Each write below is independent".

- **Finding (S8, LOW): collapsing before the sort can drop a claim.** The
  surviving copy's confidence alone sets the sort position, so a fork can fall
  past `sample` and vanish rather than appear once.
  - Fix: pinned as a decision — `test_collapsing_can_drop_a_forked_claim_out_of_the_sample`.

- **Finding: `COUNTS_CYPHER` counted nodes, not claims.** The prompt renders
  "N live priors, M closed", so a fork told Orion it held a belief it does not.
  - Evidence: live query returns 5 closed **nodes** and 4 closed **claims**.

- **Review item left UNVERIFIED by the reviewer, closed by me:** the rendered
  `MERGE … ON CREATE SET` with inline `//` comments mid-SET-clause had not been
  parse-checked. Run against live FalkorDB on a scratch graph: parses, creates
  1 node / 9 properties, and the second run is a genuine no-op.

## Files changed

- `orion/curiosity/kickoff_prompt.py`: CREATE → MERGE ON CREATE SET; the no-op
  warning; corrected MERGE semantics.
- `orion/curiosity/worldview.py`: `collapse_duplicate_priors`, `prior_id_forks`,
  `dedupe_settled_rows`, census in `read_snapshot`, distinct-claim counts.
- `tests/test_curiosity_worldview.py`: 24 tests.

## Deliberately NOT changed

`orion/curiosity/atlas.py` lists a forked prior twice and counts it twice in
`priors_created`. That surface is a record of **what each run wrote**, and both
runs really did create a node — collapsing there would erase a real write from
the history. Fixing it needs a decision about run credit, not a dedup.

## Tests run

```text
tests/test_curiosity_worldview.py + atlas + atlas_template + study_material
  + acl_and_credentials .......... 236 passed
services/orion-hub/tests/test_curiosity_investigation.py ..... 96 passed
14 mutations verified red (each anchor asserted to match exactly once)
10/10 CI static gates PASS
```

## Live verification

```text
MERGE ... ON CREATE SET, run twice -> Nodes created: 1, then no-op. count = 1.
MATCH ... SET times_tested + 1     -> 0 -> 1, keys intact.
count(DISTINCT ...) on live graph  -> live 5, closed 4 (vs 5 closed nodes).
```

## Env/config changes

None. No new keys, no `.env_example` change.

## Restart required

```text
No restart required for the code. The prompt is read per-run.
```

## Production repair — NEEDS JUNIPER'S APPROVAL

The two forked nodes are still in `orion_worldview`. This is a production graph
write and I have not run it. Validated on a scratch graph that mirrored the live
shape (edges split 3/2): after the repair one node survives with all 5 findings
re-pointed onto it.

```bash
# 1. Re-point the loser's edges onto the most-tested copy (same rule the code uses)
redis-cli -h 127.0.0.1 -p 6380 GRAPH.QUERY orion_worldview '
MATCH (keep:Prior {prior_id:"concept_induction_overload_rate"})
WITH keep ORDER BY keep.times_tested DESC LIMIT 1
MATCH (loser:Prior {prior_id:"concept_induction_overload_rate"})
WHERE id(loser) <> id(keep)
MATCH (f)-[r:SUPPORTS]->(loser)
MERGE (f)-[:SUPPORTS]->(keep)
DELETE r'

# 2. Delete the emptied copy
redis-cli -h 127.0.0.1 -p 6380 GRAPH.QUERY orion_worldview '
MATCH (keep:Prior {prior_id:"concept_induction_overload_rate"})
WITH keep ORDER BY keep.times_tested DESC LIMIT 1
MATCH (loser:Prior {prior_id:"concept_induction_overload_rate"})
WHERE id(loser) <> id(keep) DELETE loser'

# 3. Verify: expect one node, 5 inbound SUPPORTS
redis-cli -h 127.0.0.1 -p 6380 GRAPH.RO_QUERY orion_worldview '
MATCH (p:Prior {prior_id:"concept_induction_overload_rate"})
OPTIONAL MATCH (p)<-[r]-() RETURN count(DISTINCT p), count(r)'
```

Until this runs, the census logs a warning on every run — which is intended.

## Risks / concerns

- Severity: medium. Concern: `times_tested` is wrong on both copies (1 and 6;
  the claim was genuinely tested ~3 times) and the repair preserves 6 rather
  than correcting it. Mitigation: the prior is already `refuted`, so
  `stale_after` never reads it. Correcting it would be inventing a number.
- Severity: low. Concern: the MERGE trades a detectable permanent corruption for
  an undetectable transient loss (S4) if Orion reuses an id for a genuinely new
  claim. Mitigation: the prompt now warns; the fork it replaces was permanent.
- Severity: low. Concern: run `a52115e68603` has two `:TurnOutcome` nodes with
  different continue-notes. Both consumers already handle it deterministically
  (`ORDER BY written_at DESC LIMIT 1`, and `build_recent_runs` groups by run),
  so this is an observation, not a defect. Not touched.
- Observed, not this patch's business: that claim has **zero** `:PriorRevision`
  nodes, so its move from open to refuted has no record — the run did not follow
  the "record what it was, in the same breath" instruction.

## PR link

<pending>
