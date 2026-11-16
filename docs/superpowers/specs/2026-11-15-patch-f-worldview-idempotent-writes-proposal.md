# Patch F proposal: worldview writes that are safe to repeat (#3395)

Status: APPROVED by Juniper 2026-11-16 (rule: yes, waits for her go on the live write; times_tested counts runs: yes). Built in branch `feat/day-letter-patch-f`. Refs #3395. Parent spec:
`docs/superpowers/specs/2026-11-15-day-letter-findings-fix-design.md` (Patch F).
All references read on `origin/main` at `b6eeb36f0`. Live reads taken 2026-11-15 23:55Z
with `docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_worldview "<cypher>"`.

## Recommendation

**Change the three taught writes, with two corrections to the spec. Add the unique rule on
prior ids. Repair nothing first: the graph has no duplicate that blocks it.**

Orion writes their own findings and beliefs by hand, as graph commands taught in the
curiosity kickoff prompt (`orion/curiosity/kickoff_prompt.py`). Three of those commands make
a second copy, or count twice, when Orion sends the same batch again:

- A finding is taught as `CREATE` (`:1097`).
- A record of a belief moving is taught as `CREATE` (`PriorRevision`, `:1052`).
- A test adds 1 to the count with no check for "this run already counted" (`:1047`).

Priors and concepts are already safe (`MERGE` on the id alone, `:1011`, `:1083`).

## What I measured

| What | Count | Query result |
|---|---|---|
| Priors | 544 nodes, 543 ids, **0 duplicate ids** | 1 prior has no `prior_id` (node 3208; it uses `name`) |
| Findings | 691 nodes, 684 ids, **0 duplicate ids** | 7 findings have no `finding_id` (free-hand shapes) |
| Prior revisions | 372 nodes | 18 groups share one `(prior_id, run_id)` |
| ... of those 18 groups | 16 are real: the belief moved 2 or 3 times in one run, each move different | 2 are true duplicates (same move, or all fields empty) |
| Prior revisions with gaps | 1 no `prior_id`, 4 no `run_id`, 3 name a prior that does not exist | |
| Priors with no `last_run_id` | 324 of 544 | 3 have no `times_tested` |
| Constraints and indexes | none | `CALL db.constraints()` and `CALL db.indexes()` return no rows |

The event in the ticket (run `1bef47841948`: duplicate findings, a second prior, count +2) is
still UNVERIFIED. Today that run has 4 findings with 4 different ids and no duplicate prior.

## Two corrections to the spec

1. **Findings: `MERGE` on `finding_id` alone, not on `(finding_id, run_id)`.** The prompt
   joins a finding to a prior by `finding_id` alone (`:1104`). A two-part key lets the same id
   exist once per run, and then that join matches both. The id is unique across all 684
   findings today. This is also the rule the prompt already teaches for priors and concepts.
2. **Prior revisions: `MERGE` on the whole move, not on `(prior_id, run_id)`.** Key:
   `prior_id, run_id, from_confidence, to_confidence, from_status, to_status`. A key of
   `(prior_id, run_id)` would fold 16 real histories into one row each and lose the middle
   steps. Example, run `68ae1c18ff7c`: one belief went 0.92 to 0.90, then to 0.75, then back
   to 0.92.

## The test count guard

Teach the test as two commands:

```cypher
MATCH (p:Prior {prior_id: "..."}) SET p.status = "revised", p.confidence = 0.72, p.last_tested_at = "<iso>"
MATCH (p:Prior {prior_id: "..."}) WHERE coalesce(p.last_run_id, "") <> "<RUN_ID>"
  SET p.times_tested = coalesce(p.times_tested, 0) + 1, p.last_run_id = "<RUN_ID>"
```

Two traps. First, `p.last_run_id <> "<RUN_ID>"` alone is never true when the field is empty,
and 324 priors have it empty. Without `coalesce` the count stops moving for 60% of priors.
Second, only the second command may set `last_run_id`. If the first sets it, the guard never
passes.

This changes what the number means. Today `times_tested` counts tests. After the patch it
counts runs that tested the claim. The prompt says "three inconclusive tests and the claim is
probably not answerable". That becomes three runs.

## The unique rule on prior ids

FalkorDB needs an index before a unique rule. Both are operator commands, because Orion's
login has no right to run them:

```
GRAPH.QUERY orion_worldview "CREATE INDEX FOR (p:Prior) ON (p.prior_id)"
GRAPH.CONSTRAINT CREATE orion_worldview UNIQUE NODE Prior PROPERTIES 1 prior_id
```

**What must be repaired first: nothing.** No two priors share an id. The one prior with no
id is the open point: FalkorDB's documents say a node that lacks the property is not held to
the rule. I did not test that here (graph module version 4.18.11). Test it on a scratch graph
before the live one.

**Repairs I propose but did not run.** They are writes to Orion's memory. None blocks the
rule. I recommend that Orion makes them in a curiosity run, from a short note in the prompt:

- Give prior node 3208 a `prior_id`, or retire it.
- Remove one row from each of the 2 true duplicate revision pairs
  (`self:walkway_observation_entity_field_switched_on_20261113_20261113` in run `f6410b34d484`,
  `self-definition-v78-20261027` in run `f48a855d93d3`).
- Leave the 7 findings with no id and the 3 revisions with no prior as they are. They are
  history.

## The six points

- **Capability that changes:** Orion can send the same batch twice and the graph stays the
  same. A stray `CREATE` with an id that exists now fails with an error that Orion sees in
  the turn. One number changes meaning (`times_tested`, see above).
- **Data touched:** the prompt text. One index and one rule on the `orion_worldview` graph.
  No existing node changes.
- **Privacy boundary:** none between Juniper and Orion. The graph holds Orion's claims, and
  some name other people (walkway visitors, people in reading). Nothing here moves a claim
  out of the graph or shows it to anyone new.
- **Trace that proves it worked:** `CALL db.constraints()` shows the rule with status
  `OPERATIONAL`. The duplicate query stays at 0:
  `MATCH (f:Finding) WITH f.finding_id AS id, count(*) AS c WHERE c > 1 RETURN count(id)`.
  A kickoff prompt in `cockpit_turn_sighting` shows `MERGE (f:Finding`.
- **Dangerous failure mode:** the guard is written wrong and the count stops for every
  prior. Orion then never reaches "tested three times" and never retires a claim. A second
  risk: a finding `MERGE` with an old id writes nothing, with no error, and a real new
  finding is lost. The prompt already warns about this for priors; the same warning must be
  copied to findings.
- **Disable or roll back:** revert the prompt commit. Drop the rule with
  `GRAPH.CONSTRAINT DROP orion_worldview UNIQUE NODE Prior PROPERTIES 1 prior_id`. Neither
  step loses data.

## Smallest first patch

1. Prompt text only: the three changed writes, the "old id writes nothing" warning on
   findings, and unit tests that assert the taught text.
2. One eval that replays a taught batch twice against a scratch graph (not `orion_worldview`).
3. The index and the rule are a second, separate step, after Juniper approves.

## Non-goals

- No write helper or tool that replaces hand-written Cypher. Orion keeps the pen.
- No cleanup of the 7 free-hand findings or of old revisions.
- No change to the read-side merge (`collapse_duplicate_priors`, `orion/curiosity/worldview.py:1104`). It stays as the safety net.
- No unique rule on findings or concepts in this patch.

## Acceptance checks (corrected)

| Check | Pass |
|---|---|
| Replay one batch twice in a scratch graph. The batch holds a new prior, a test of an old prior, two different revisions of one prior, a finding and its edges. | Node counts, edge counts and `times_tested` are the same after the second run. Both revisions exist. |
| Test a prior that has no `last_run_id`. | `times_tested` goes up by 1. |
| `CREATE (:Prior {prior_id: <an id that exists>})` in the scratch graph, after the rule. | The command fails. |
| Live, 7 days after deploy: the duplicate queries above. | 0 duplicate finding ids. No new revision group with the same move twice. |

## Questions for Juniper

1. Add the index and the unique rule on `Prior.prior_id` to the live graph, yes or no? It is
   an operator command on Orion's memory.
2. `times_tested` will count runs, not tests. Is that what you want, yes or no? I recommend
   yes.

## UNVERIFIED

- The original double-write in run `1bef47841948`.
- FalkorDB's handling of the one prior with no `prior_id` under a unique rule, and the exact
  status values that `GRAPH.CONSTRAINT CREATE` reports. Both come from FalkorDB's documents,
  not from a test.
- Whether any prior's `times_tested` is too high today from a past double count. The graph
  keeps no per-test record to compare against.

## Orion's questions (2026-11-16) and what changed

12. **The correction lane teaches the same write.** **Changed:** `orion/orion_day/corrections.py`
    now points at the two guarded commands, and so does `orion/consult/beliefs.py:164`, which
    taught `times_tested + 1` too.
13. **Other `times_tested` readers.** Checked all four. `value.py:285-311` already counts per
    run (`tested_delta > 0`), so it reads the new number the same way. `corrections.py:255-269`
    needs `last_run_id == run_id` and a rise or a move; the guarded command sets both together,
    so a re-test still registers. `atlas.py:520` only sorts. `replica/frame.py:167` only shows
    "never tested". None reads a healthy claim as stale. The retirement rule
    (`worldview.py:1250`, `stale_after`) and `ask_claude_trigger.py:166` (`MIN_TIMES_TESTED`)
    are reached more slowly, by the amount in 14.
14. **How often is a prior tested more than once in a run?** From `curiosity_run_outcomes`
    (09-26..11-15): 239 per-run prior tests in 172 runs; 15 (6.3%) moved `times_tested` by
    more than 1, 2 by more than 2, max 6. Some of those may be replays. So counting runs slows
    retirement for about 6% of tests.
15. **Re-derived findings.** MERGE folds a replay only. The kickoff prompt teaches no minting
    rule. The one content-derived id is `orion/consult/beliefs.py:77` (`finding_id_for`,
    from the consult id), which already teaches MERGE. A re-derivation in a later run gets a
    new id; detecting that is the novelty corpus's job (`orion/curiosity/register.py`), not
    this patch. Acceptance checks are scoped to replays.
16. **Order of operations.** The scratch test ran first, on a throwaway container from the
    live server's own image (graph module 41811):
    `python3 scripts/worldview_prior_unique_rule.py scratch` -> the taught batch twice leaves
    the same counts (`Prior:2 PriorRevision:2 Finding:1`, edge 1, `times_tested` 2 -> 3 once,
    on a prior with no `last_run_id`); the rule goes `PENDING` -> `OPERATIONAL` over a graph
    with an id-less prior; a duplicate `CREATE` fails with `unique constraint violation on
    node of type Prior`; a second id-less prior is still allowed. A rule over a graph that
    already holds a duplicate goes to `FAILED`, so `preflight` runs first (live today: 0
    duplicates, 1 id-less prior, node 3208, no constraints). #3455 was relieved by the
    FalkorDB restart at 11-16 09:11Z. The live write still waits for Juniper:
    `python3 scripts/worldview_prior_unique_rule.py commands` prints it.
