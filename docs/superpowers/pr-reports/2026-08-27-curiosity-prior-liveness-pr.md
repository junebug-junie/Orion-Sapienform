# A prior is live until Orion closes it, not until it is tested

## Summary

- `OPEN_PRIORS_CYPHER` asked FalkorDB for `p.status = 'open'`, so a prior left
  Orion's world view the first time Orion tested it. On 2026-08-27 that took the
  accumulation loop to zero: three priors in the graph, `priors=0/0` offered.
- Liveness is now the **complement of two closing statuses** (`refuted`,
  `retired_unresolvable`) rather than equality with one open status. `supported`
  and `revised` record what a test returned and keep the claim in play.
- Snapshot fields renamed `open_priors`/`open_total`/`resolved_total` →
  `live_priors`/`live_total`/`closed_total`, because a field named for one
  status while holding several is how this bug comes back.
- The prompt now names which two statuses close a prior, and says confidence
  going **down** on a second look is a real result.
- New `curiosity_worldview_pool_dead` warning: every prior closed is legal, but
  it is also what this outage looked like from outside, and the only symptom was
  a run quietly starting from nothing.
- Live-verified against `orion_worldview`: the new query returns all 3 priors
  where the old one returned 0.

## Outcome moved

Orion's curiosity loop can test a claim more than once. Before this, the second
test was unreachable by construction: confidence could move on a prior's first
test and never again, which made the loop's headline acceptance check ("does
confidence ever go DOWN?") not merely unmet but impossible.

## Current architecture

Hub reads Orion's own FalkorDB graph (`orion_worldview`) once per run and
renders what it read into the kickoff prompt. Hub never writes; Orion writes its
own nodes in hand-authored Cypher inside the turn. Property names are therefore
a contract between a prompt and a reader, and drift shows up as an empty pool
rather than an error.

`read_snapshot` issued four queries: open priors, counts, concept count, and
recently-settled. `select_priors` then split the result into an offered list
(sorted most-uncertain-first) and a stale bucket (`times_tested >= stale_after`),
both capped at `HUB_CURIOSITY_PRIOR_SAMPLE`.

## The failure

Three runs on 2026-08-27, from the live graph and Hub logs:

```text
10:03:05  run 7736d5271d97  offered priors=1/1  hops=5  -> tested its inherited
                            prior, revised it, wrote its own new prior
14:06:31  run 0a14e9531089  offered priors=0/0  hops=1
```

At 14:06 the graph held three priors — `revised`, `supported`, `supported` —
and the reader returned none of them. Two independent paths into the same hole:

1. **Testing a prior closed it.** Run 5 revised run 3's prior, which took it out
   of `status = 'open'` permanently. `revised` means the claim itself just
   changed; that is the most live a prior can be.
2. **A prior could be born closed.** Run 6 wrote its new prior as `supported` in
   the same breath it formed it, having confirmed it that run. It was never
   `open` for even one run, so it was deleted from its own future at creation.

The continuation-note channel is independent and did carry run 6 forward, which
is why the loop looked like it was working. The *prior* channel was dead.

## Architecture touched

- `orion/curiosity/worldview.py` — status set, three Cypher constants, snapshot
  field names, one new warning.
- `orion/curiosity/kickoff_prompt.py` — prior section wording and the status
  legend Orion writes against.
- `services/orion-hub/scripts/curiosity_investigation.py` — the `priors=%s/%s`
  log line, the surface where this was visible all along.

No bus channel, schema registry entry, env key, or Docker change.

## Files changed

- `orion/curiosity/worldview.py`: `CLOSED_STATUSES` and the reasoning for
  defining liveness as its complement; `OPEN_PRIORS_CYPHER` →
  `LIVE_PRIORS_CYPHER`; counts split live/closed; `RECENT_SETTLED_CYPHER`
  narrowed to closed only; `curiosity_worldview_pool_dead`.
- `orion/curiosity/kickoff_prompt.py`: `live_*` fields; "ONLY TWO STATUSES CLOSE
  A PRIOR"; the inline status legend; "confidence is allowed to move DOWN".
- `services/orion-hub/scripts/curiosity_investigation.py`: field rename.
- `tests/test_curiosity_worldview.py`: `_StatusAwareReader`, incident replay,
  status parametrisation, pool-dead tests.
- `services/orion-hub/tests/test_curiosity_investigation.py`: loop-level test
  now uses a `revised` prior, which is the status that actually went missing.
- `orion/curiosity/README.md`: the lifecycle paragraph that stated the bug in
  prose, plus an operator query for a dead pool.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: `WorldviewSnapshot.open_priors` → `live_priors`, `.open_total` →
  `.live_total`, `.resolved_total` → `.closed_total`; `OPEN_PRIORS_CYPHER` →
  `LIVE_PRIORS_CYPHER`. Internal to `orion.curiosity`; both consumers updated.
- Behaviour changed: the set of priors offered to a run. No node property, node
  label, or value Orion writes changed, so **existing graph data needs no
  migration** — the three priors already in `orion_worldview` become visible
  again on the next read.
- Compatibility: a prior with an unrecognised or missing status now reads as
  live rather than being silently dropped.

## Env/config changes

None. No `.env_example` touched, so no sync required. `HUB_CURIOSITY_PRIOR_SAMPLE`
and `HUB_CURIOSITY_STALE_PRIOR_TESTS` keep their meanings and values.

## Design note: why the complement, not a list

The obvious shape is `WHERE p.status IN ['open','supported','revised']`. It was
rejected: Orion hand-writes this Cypher inside a turn, and a typo'd status under
that rule makes a belief vanish with no error anywhere. Under the complement
rule, an unrecognised status reads as live — the failure is one extra claim in a
list that is already capped at 8 and sorted most-uncertain-first, versus a
silently lost belief.

Re-litigation, the risk this widening actually creates, already has a bound:
`stale_after` (3) moves a repeatedly-tested prior into its own bucket with
retiring named as a real outcome, and the uncertainty sort puts confident priors
last so they fall off the sample naturally once real competitors exist.

`p.status IS NULL OR NOT p.status IN [...]` rather than `NOT p.status IN [...]`:
in Cypher `NOT null IN [...]` evaluates to null, and a null `WHERE` filters the
row out — so the null arm is load-bearing, not defensive noise.

## Tests run

```text
$ .venv/bin/python -m pytest services/orion-hub/tests/test_curiosity_investigation.py \
    tests/test_curiosity_worldview.py tests/test_curiosity_acl_and_credentials.py \
    tests/test_curiosity_study_material.py -q
186 passed, 18 warnings in 3.40s
```

Mutation test — the fix reverted to `WHERE p.status = '{STATUS_OPEN}'`:

```text
FAILED test_live_and_settled_reads_are_actually_different_queries
FAILED test_the_two_closing_statuses_are_the_only_ones_the_query_excludes[refuted]
FAILED test_the_two_closing_statuses_are_the_only_ones_the_query_excludes[retired_unresolvable]
FAILED test_a_prior_with_no_status_survives_the_cypher_null_trap
FAILED test_the_2026_08_27_graph_state_is_not_read_back_as_an_empty_worldview
FAILED test_a_closed_prior_does_not_come_back_through_the_live_read
6 failed, 73 passed
```

The first mutation run is worth recording: it produced **4** failures, and the
incident replay was not among them. `_FakeReader` answers by query *shape* and
ignores the `WHERE` clause — which is the exact clause the bug lived in — so the
replay passed against the broken code it was written to catch.
`_StatusAwareReader` now applies the status predicate read off the real query
text, and the replay fails as it should.

## Evals run

```text
No eval harness exists for orion/curiosity. The loop's own evaluation is the
long-run acceptance check recorded in orion/curiosity/README.md section 13
("does confidence ever go DOWN"), which needs ~20 runs of live data and is not
a harness this patch can add honestly.
```

## Docker/build/smoke checks

Live verification against the real graph, before and after, same FalkorDB:

```text
$ redis-cli -u redis://127.0.0.1:6380 GRAPH.RO_QUERY orion_worldview \
    "MATCH (p:Prior) WHERE p.status = 'open' RETURN p.prior_id"
(0 rows)

$ redis-cli -u redis://127.0.0.1:6380 GRAPH.RO_QUERY orion_worldview \
    "MATCH (p:Prior) WHERE (p.status IS NULL OR NOT p.status IN
     ['refuted', 'retired_unresolvable']) RETURN p.prior_id, p.status"
editorial_bias_concrete_over_atmospheric_32b42392f495   revised
gate_bias_manual_review_7736d5271d97                    supported
automated_intake_gate                                   supported

$ ... COUNTS_CYPHER
live_total 3    closed_total 0
```

FalkorDB accepts the generated syntax and returns the three priors that were
invisible to the deployed code.

## Restart required

Hub bakes `orion/` into its image (`COPY orion ./orion`), so a restart is not
enough:

```bash
scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-hub up -d
```

## Risks / concerns

- Severity: low. Concern: the offered list now includes `supported` priors, so
  Orion may re-test a claim it considers settled. Mitigation: uncertainty sort
  deprioritises confident priors, `stale_after=3` caps repeats, and re-testing a
  once-tested claim is the intended behaviour, not a side effect.
- Severity: low. Concern: `_cypher_admits_status` in the tests is a small
  hand-rolled reader of the `WHERE` clause and could drift from FalkorDB's real
  semantics. Mitigation: it covers only the two predicate forms this module
  emits, and the live check above is the ground truth.
- Severity: informational. `HUB_CURIOSITY_OUTREACH_ENABLED=false` meant run 6's
  `reach_out=true` — Orion's first-ever decision to interrupt Juniper — was
  logged and dropped. Out of scope here; it is a Juniper decision, not a bug.

## PR link

<pending>
