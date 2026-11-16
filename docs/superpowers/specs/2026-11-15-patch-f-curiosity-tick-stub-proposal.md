# Patch F proposal: the curiosity tick's "calm" stub (#3393)

Status: APPROVED by Juniper 2026-11-16 and built in branch `feat/day-letter-patch-f` (see "Orion's questions" at the end). Refs #3393. Parent spec:
`docs/superpowers/specs/2026-11-15-day-letter-findings-fix-design.md` (Patch F).
All references read on `origin/main` at `b6eeb36f0`. Live reads taken 2026-11-15 23:50Z.

## Recommendation

**Delete the stub and the four branches it feeds. Do not wire the real interpreter.**

The spec offered two paths: call the real interpreter, or delete the dead branches. I ran
the real interpreter on the live graph. It cannot produce a contradiction, and its "drift"
reading is a constant. Wiring it in would add a signal that never changes and call it
perception.

## What I measured

Every minute, a loop in the substrate runtime picks what Orion could be curious about. It
gives a second step (the evaluator) a report that says "all calm". That report is hardcoded
(`_neutral_frontier_metacog_inputs`, `services/orion-substrate-runtime/app/worker.py:4401-4456`,
used at `:4738`).

I copied the live graph read-only (15,559 nodes, 62,757 edges, 0 rows failed to decode) and
ran the real code on it (`build_graph_views`, `extract_graph_features`,
`interpret_graph_cognition`, then `FrontierCuriosityEvaluator.evaluate` with no seeds).

| Branch in the evaluator (`orion/substrate/frontier_curiosity.py`) | Real result today | Why |
|---|---|---|
| Contradiction (`:128`) | 0 candidates. Can never fire. | It needs nodes of kind `contradiction`. The graph has none. The store refuses to save that kind (`DURABLE_NODE_KINDS = concept, evidence, entity, assertion`, `orion/substrate/falkor_codec.py:183`). |
| Goal pressure (`:182`) | 0.040. Needs 0.70. Can never fire. | With no goal nodes the score cannot pass 0.50. |
| Identity conflict (`:219`) | 0.024. Needs 0.35. Can never fire. | With no contradiction or tension nodes the score cannot pass 0.30. |
| Drift (`:146`) | 0.475, "active". Fires on every tick. | See below. |

**Drift is a constant, not a measure.** The view keeps only nodes seen in the last 24 hours
(`orion/graph_cognition/views.py:95-110`). The feature step then asks what share of those
nodes was seen in the last 24 hours (`features.py:115`). The answer is always 1.0. The
score becomes `0.35 + 0.0125 x (number of Orion's own fresh nodes)`. Orion had 10 such nodes
at the read. They are the prediction-error nodes, and the runtime refreshes them through the
day (the newest was 15 s old, the oldest 3.3 h). So drift reads 0.475. It reads "active"
whenever 4 or more of those nodes are fresh. It cannot return to calm. This fails step 4 of the metric quality gate in `CLAUDE.md`.

**What would change if it were wired in.** In the last 7 days the evaluator ran on 1,767
candidate sets. A constant 0.475 would outrank every seed in 161 of them (9%). The decision
stays "do nothing" in all 161, because 0.475 is under the 0.50 line (`:269`). The graph holds
12 such nodes, and 2 were stale. If those 2 are refreshed, the score reaches 0.50 and those
ticks flip to "invoke" with no real cause. Each stored set would also gain one extra row that points at the same
hot nodes every minute.

```sql
-- 1,767 evaluated sets; 161 where the best seed is under 0.475
with m as (select c.candidate_set_id, max((s->>'signal_strength')::numeric) mx
  from substrate_endogenous_curiosity_candidates c, jsonb_array_elements(c.candidates_json) s
  where c.created_at > now()-interval '7 days' and c.gate_json->>'evaluator_outcome' is not null
    and s->'notes' ? 'endogenous_seed' group by 1)
select count(*), count(*) filter (where mx < 0.475) from m;
```

No stored candidate has a signal type other than `curiosity_candidate` (the table keeps 30
days). That confirms the four branches have not fired in that time.

## The six points

- **Capability that changes:** none. The four branches cannot fire today. After the delete
  they do not exist. Orion's curiosity picks the same things.
- **Data touched:** code only. No table, no graph, no stored candidate row changes.
- **Privacy boundary:** none between Juniper and Orion. The substrate graph holds nodes
  about other people (walkway sightings, world reading). This patch does not read, move or
  write them.
- **Trace that proves it worked:** `substrate_endogenous_curiosity_tick_completed` log lines
  keep the same outcome mix before and after (7-day baseline: `concept_expand` invoke 1,338,
  `evidence_gap_scan` invoke 235, `relation_discovery` noop 165 and invoke 26).
- **Dangerous failure mode:** the delete removes a parameter that a live caller still
  passes, and the tick fails every minute. The tick catches its own errors, so it would fail
  quietly. Guard: a test that runs the tick end to end, and a log check for
  `substrate_endogenous_curiosity_tick_failed` after deploy.
- **Disable or roll back:** revert the commit. The whole tick already has a kill switch
  (`ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH=true`).

## Smallest first patch

1. Remove `_neutral_frontier_metacog_inputs` from `worker.py`.
2. Remove the four report-driven branches from `_derive_signals`.
3. Remove the `cognition_report` and `perception_brief` arguments from `evaluate`.
4. Update `tests/test_cognitive_substrate_phase8_frontier_invocation.py` to match.
5. Leave a comment at the delete site that names this document, like the 2026-10-10
   `ontology_sparse_region` retirement comment in the same file.

`orion/graph_cognition/` then has one non-test importer left (`orion/substrate/consolidation.py`).
Delete the package in the same patch only if no live caller passes a report to consolidation.
That check is UNVERIFIED.

## Non-goals

- No new contradiction or drift detector. Nothing in Orion writes a contradiction node, so
  there is nothing to detect yet. A real detector needs a producer first and its own
  proposal.
- No change to the seed sources, the System One gate, or the stored candidate shape.
- No fix for #3455 in this patch.

## What this does not fix

The ticket's real symptom is that 98% of world concepts show up on one day only, and the
seeds keep returning to Orion's own nodes. Neither path in the spec fixes that. The seeds
come from open loops, prediction error and repair pressure (`orion/substrate/endogenous_curiosity.py`),
and all three point at self nodes by construction. That needs its own trace. Keep #3393 open
for it, or split it into a new ticket.

## Acceptance checks (corrected)

The spec's check was "a candidate set whose gate names a contradiction or drift trigger, at
least one in 7 days". Drop it. With the real interpreter it would pass on every tick and
prove nothing. Use these:

| Check | Pass |
|---|---|
| `rg "_neutral_frontier_metacog_inputs\|cognition_report" services/orion-substrate-runtime orion/substrate/frontier_curiosity.py` | No match. |
| Tick log for 24 h after deploy | No `substrate_endogenous_curiosity_tick_failed`. Outcome mix within the 7-day baseline. |
| "World concepts return on a second day more often than 2%" | Not expected to move. Keep it as the check for the follow-up, not for this patch. |

## Relation to other tickets

- **#3366** (Prime's curiosity questions refused): no shared cause. Its root cause is a 4 s
  client timeout on a Hub read, found 2026-11-15, and the ticket is closed.
- **#3455** (graph reloads failing): still live. `falkor_substrate_hydrate_stats` shows
  `ok_total=0 failed_total=4375` at 23:45Z. The tick reads the runtime's in-memory copy of
  the graph, and that copy is partial. My replay read the full graph from FalkorDB, so the
  numbers above are the best case. How many nodes the tick's own copy holds is UNVERIFIED.
  Any future detector here must wait for #3455.

## Question for Juniper

1. Delete the stub and the four dead branches (recommended), yes or no? "No" means: build a
   real drift measure first, with its own proposal and metric gate.

## UNVERIFIED

- The 98% one-day figure in #3393. I did not re-run it.
- Whether a live caller passes a report to `orion/substrate/consolidation.py`.
- The node count of the tick's in-memory graph copy under #3455.
- How often drift would have read "active" in the past. I measured one moment. The formula
  says it is active whenever 4 or more of Orion's own nodes were refreshed in 24 hours.

## Orion's questions (2026-11-16) and what changed

Orion asked six questions about this proposal (turn `a13c8dc3c334`). Answers, with the
change each one made:

1. **Which graph gave the drift number?** My replay of the full FalkorDB graph. The tick
   never computed drift: it passed the hardcoded stub, so it built no view and had no node
   count. Its store is the runtime's in-memory cache (`falkor_store.py:1336-1362`), which
   held only this process's own writes while #3455 failed. FalkorDB was restarted 11-16
   09:11Z and reloads now decode 15,569/15,569 nodes. The cache size at 23:50Z is UNVERIFIED.
2. **Is drift monotone in node count?** Yes. With every viewed node fresh, churn
   (`features.py:118`) equals the node count, so the score is `0.35 + 0.25*min(1, n/20)`. It
   rises with size and saturates at 0.60 at 20 fresh nodes, below the `max_nodes` cap. More
   reason to delete, not wire.
3. **Un-fed or impossible?** Both, for this store. `FalkorSubstrateStore` refuses any kind
   outside `DURABLE_NODE_KINDS` at write time, before its cache (`falkor_store.py:1011`), so
   the cache cannot be richer than the durable graph. The other producers have no live
   caller: `emit_contradiction` (`orion/autonomy/substrate_emit.py:56`) is called only from
   tests and makes a molecule, not a graph node; `map_concept_delta_to_substrate`
   (`adapters/concept_induction.py:187`) is exported but never called. **Changed:** the
   delete-site comment names the whitelist (`falkor_codec.py:183`) as what a future producer
   must widen.
4. **Consolidation.** Confirmed: the only live `consolidate()` caller
   (`review_runtime.py:93`) passes no report. **Changed:** the patch also deletes
   `consolidation.py:383` (goal-pressure `maintain_priority`) and the whole
   `orion/graph_cognition` package, plus the now-unused `SubstrateQueryPlanner.graph_view_basis`.
5. **Positive evidence of no other caller.** `git grep` over all files: the evaluator is
   built only at `worker.py:4739`; `FrontierCuriosityOrchestrator` and `cognition_report=`
   appear only in tests. After the patch the arguments do not exist, so a hidden caller fails
   with `TypeError`, not silently. The tick test now binds the worker's real call kwargs
   against the real `evaluate` signature.
6. **Is a contradiction producer on the roadmap?** No open issue plans one
   (`gh issue list --search contradiction`). The comment at the delete site, the retired
   signal types (removed from `FrontierInvocationSignalTypeV1`), and a test that they stay
   out of the contract constrain the next one: it must ship its own consumer and test.
