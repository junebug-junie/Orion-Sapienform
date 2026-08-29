# Did the finding get joined to anything

## Summary

- `wrote=` proves an edge was drawn somewhere in a run. It cannot prove the
  **findings** were what got joined — and that is the claim the kickoff
  prompt's edge instruction actually makes.
- Added `finding_connectivity_cypher` / `read_finding_connectivity` to
  `orion/curiosity/worldview.py`, returning `FindingConnectivity(total, connected)`.
- Wired it into the loop as a fourth slot on `_read_turn_result` and an
  `evidence=` field on the `curiosity_investigation_journaled` log line.
- Kept it **out** of the journal. That artifact is Orion's own written result;
  an orphan-rate statistic there would be a health check wearing Orion's voice,
  and nothing reads a journal entry back into a later prompt anyway.
- First live reading is a negative, and it is reported as one: run
  `d05ef10b303a` wrote 2 findings and joined neither.

## Outcome moved

The edge instruction shipped in PR #1941 and deployed 2026-08-29 20:46 UTC.
Before this patch, nothing in the system could say whether it took. The
footprint that would have been used to judge it is structurally incapable of
answering the question — see the independence check below.

## Current architecture

`run_footprint_cypher` counts nodes by label for one run.
`run_edge_footprint_cypher` counts edges by type. `read_run_footprint`
concatenates both into one dict, which the tick renders as `wrote=` and the
journal renders as "Wrote to its own graph: …".

## Architecture touched

One new read on the existing `WorldviewReader`, one new field on an existing
log line. No new service, channel, schema, env key, or container.

## Files changed

- `orion/curiosity/worldview.py`: the dataclass, the Cypher, the reader.
- `services/orion-hub/scripts/curiosity_investigation.py`: `format_evidence`,
  the fourth tuple slot, the log field.
- `services/orion-hub/README.md`: a "Did the finding get joined to anything"
  subsection, including the four readings and what each means.
- `tests/test_curiosity_worldview.py`: 7 tests.
- `services/orion-hub/tests/test_curiosity_investigation.py`: 3 tests.

## Metric quality gate

Run in full, as CLAUDE.md requires, and recorded here rather than passed
verbally.

1. **Provenance.** The value comes from a Cypher aggregation over
   `(f:Finding {run_id})` degree in FalkorDB `orion_worldview`. The producer is
   Orion's own `CREATE`/`MERGE` inside the turn, instructed at
   `orion/curiosity/kickoff_prompt.py:592-596`.
2. **Independence.** Not a transform of anything already collected. `Finding 3`
   plus three edges in the footprint is **identical** whether one finding
   carries all three edges (2 orphans) or each carries one (0 orphans). Only a
   per-finding degree separates those cases. This is asserted as a test —
   `test_connectivity_is_not_derivable_from_the_footprint_counts` — so the
   justification cannot rot silently.
3. **Theory anchor.** Named, specific, and not a vibe: it detects the exact
   pre-deploy defect the edge instruction was written to fix — a finding that
   points at nothing. It is not a proxy for anything.
4. **Live-data sanity.** Verified against a live FalkorDB *before* wiring in,
   on a throwaway graph, deleted after: three findings with edges on two reads
   `connected=2`, and **stays 2** when a second edge is added to one of them —
   the check that separates this from a second edge count. It is not
   floor-bound: `total == connected` is a reachable, meaningful rest state
   ("everything joined"), and `total=0` is distinguished from both zero-joined
   and unreadable.
5. **Existing mechanism.** Searched. `PriorRevision` was the near-miss and is
   genuinely different: it carries `prior_id`, `from_confidence`,
   `to_confidence`, `from_status`, `to_status` — *that* a prior moved, never
   *on what evidence*. Confirmed against live node properties. No degree or
   connectivity read existed.
6. **Reversibility.** One dataclass, one Cypher, one reader, one log field.
   No schema, manifest, or training default. Cheap to remove.

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- No `.env_example` touched, so `python scripts/sync_local_env_from_example.py`
  was not required and was not run.

## Tests run

```text
pytest tests/test_curiosity_worldview.py tests/test_curiosity_atlas.py \
       tests/test_curiosity_study_material.py -q     -> 163 passed
pytest services/orion-hub/tests/test_curiosity_investigation.py -q -> 95 passed
```

Run separately, not in one invocation: the two suites collide on the `scripts`
package name, a pre-existing collection error documented in the hub test file's
own import comment.

Seven CI static gates, all OK:

```text
check_metric_lineage.py --gate        check_definition_drift.py --gate
check_inner_state_registry.py         check_scripts_dir_no_stdlib_shadow.py
check_service_hostname_refs.py        check_journal_dispatch_registry.py
check_daily_schedule_collisions.py
```

## Evals run

No eval harness exists for `orion-hub`'s curiosity seam; the tests above are
the gate lane. Not claiming eval coverage. The live acceptance check for the
instruction this metric watches is a longitudinal reading across runs, and it
has one sample so far.

## Mutation testing, before review

8 mutations, each asserted to have actually landed in the file before the run,
file restored after. All 8 RED:

```text
worldview.py
  unreadable graph reads as zero, not None                RED
  drop the per-finding collapse (2nd edge count)          RED
  make the match directed                                 RED
  clamp connected to total (hides a broken instrument)    RED
  skip the run_id guard                                   RED
curiosity_investigation.py
  None renders as a healthy-looking 0/0                   RED
  connectivity never read (always None)                   RED
  stale 3-tuple on the no-reader path                     RED
```

## Live evidence

```text
run d05ef10b303a, 2026-08-29 21:16:30 UTC (first run after the edge instruction deployed)
  footprint : Finding 2, Hop 1, PriorRevision 2, TurnOutcome 1
  evidence  : 0/2 joined
  graph     : MATCH ()-[r]->() RETURN type(r), count(r)  -> empty
```

That run was not a bad run. 55 harness steps, grounded, refuted two priors,
wrote a policy note. It wrote findings plainly bearing on
`gate_bias_manual_review_7736d5271d97` and connected none of them. The
footprint alone reads as productive, because it was.

## Review findings fixed

Six findings, all fixed. The reviewer probed the live FalkorDB rather than
reading the code, which is what turned the first one up.

- **Finding: an unparseable reply rendered as a healthy run.** `if not rows:
  return FindingConnectivity(0, 0)` logged the benign `no findings`. Verified
  live: a run with no findings returns a real `(0, 0)` ROW, so `rows == []` is
  reachable only when `rows_from_reply` cannot parse the reply — a driver or
  protocol change. Every run would have logged a healthy string while the
  metric was silently dead.
  - Fix: return `None` and log `curiosity_finding_connectivity_unparseable`.
  - Evidence: `GRAPH.RO_QUERY orion_worldview` on a run_id with no findings
    returns `total=0, connected=0`. Mutation reverting the fix → RED.

- **Finding: the test locked the bug in.** The no-findings case used the
  default `_FakeReader()` (returns `[]`) — a reply the real graph never sends
  — so it pinned the broken-reply path to a healthy reading.
  - Fix: the fake now returns a real `(0, 0)` row; the empty-reply case got
    its own test asserting `None`.
  - Evidence: 106 pass; mutation → RED.

- **Finding: `orphaned` laundered a broken instrument.** The reader refuses to
  clamp `connected` to `total`, but `orphaned` clamped with `max(0, ...)` — so
  `(total=2, connected=5)` printed an honest `5/2 joined` while the one derived
  number an alert would read said `0 orphans`. My own test asserted that `0`,
  cementing the contradiction into the test named for refusing it.
  - Fix: `orphaned` is unclamped. Negative is nonsense on its face, which is
    the point. The test now asserts `-3`.
  - Evidence: mutation restoring the clamp → RED.

- **Finding: the README overclaimed what the query measures.** `3/3 joined` was
  documented as "attached to a claim". The query counts any edge, undirected,
  any type, any neighbour — so two `ABOUT -> :Concept` edges read `2/2 joined`
  while no finding touches a `Prior`.
  - Fix: reworded to "joined to something", plus a paragraph stating the limit
    outright and why narrowing to priors would answer something narrower than
    the instruction being watched. The query is unchanged — it should match
    what the prompt teaches, and the prompt teaches `ABOUT`.

- **Finding: "Three readings" above a four-row table, and the `unreadable` row
  contrasted against `0/0`, a string the code cannot emit.**
  - Fix: five rows, header says five, counted against the rendered table.

- **Finding: a graph that was never configured read as an outage.**
  `HUB_CURIOSITY_GRAPH_ORION_PASSWORD` ships blank in `.env_example`, so a
  default install has no reader and every run logged `evidence=unreadable` —
  sending an operator hunting a FalkorDB outage that was not happening. This
  field exists to keep "did not answer" apart from "answered zero" and had
  inherited that same conflation one level out.
  - Fix: `format_evidence(..., graph_configured=)` renders `no graph`.
  - Evidence: mutation reverting it → RED.

## Mutation testing, after the fixes

11 mutations, each asserted to have landed before the run, both files restored
after. All 11 RED — including one per review fix, so reverting any of them
fails a test.

## Restart required

Hub bakes `orion/` into its image, so this needs a rebuild, not a restart:

```bash
scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-hub up -d
```

## Risks / concerns

- **Severity: low.** The metric is read-only and additive; a failure renders
  `unreadable` and changes nothing else in the tick.
- **Severity: medium — open question, not a defect in this patch.** The edge
  instruction did not take on its first run. n=1 is a sample, not a verdict,
  which is exactly why this instrument ships before any change to the prompt.
  Do not tune the prompt on one reading.

## Adjacent defect found, not fixed here

One `PriorRevision` (run `a394430e781e`) carries
`prior_id: "curation confidence=0.75"` — a fragment of the prompt's preview
line written into an id field, so the revision points at a prior that does not
exist. Pre-existing, unrelated to this patch, and worth its own look.
