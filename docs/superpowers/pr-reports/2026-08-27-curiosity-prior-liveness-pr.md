# A prior is live until Orion closes it — plus the surface that would have caught it

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

## The operator surface

The outage was visible in one log line for four hours because nothing put the
loop on one screen. `/curiosity` is that screen: pool tiles, confidence small
multiples, per-run graph growth, a run ledger, and every prior. Read-only by
design and asserted so — Hub never writes to `orion_worldview`, and a route that
could edit a belief Orion formed needs an auth story this does not have.

**"Before and after" turned out to be two questions, and only one was
answerable from what exists.** Which run created or last touched each node is
real history: every node carries `run_id`, a tested prior carries `last_run_id`.
The projection reproduces exactly what the loop logged at the time
(`7736d5271d97` → 8 nodes / 5 hops, `0a14e9531089` → 3 nodes / 1 hop).

The confidence a prior held *before* a test was not recoverable — `SET
p.confidence` overwrites it. So Orion now writes a `:PriorRevision` in the same
statement, **Orion and not Hub**, because Hub never writes to this graph and
that invariant is worth more than a backfilled chart. The cost is stated on the
page rather than hidden: an empty trajectory reads "not recorded yet", never
"confidence never moved".

The page also surfaces the run it cannot see. A turn killed before writing
carries no `run_id` on anything, so it appears in no panel — but Redis counted
it, and the gap between the counter and the graph is its only evidence. That is
a banner, not a silence, and it fires on today's real data: 3 counted, 2 wrote.

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
192 passed, 18 warnings in 3.39s
```

Note: running `tests/` before `services/orion-hub/tests/` gives a collection
error. Reproduced on unmodified `origin/main`; pre-existing, not this patch.

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

## Review findings fixed

Ran at `high`. Six findings, all real; a seventh ("no PR report committed") was
against `HEAD~1` and the report landed in the following commit.

Two of them are defects this patch *introduced* by making the pool accumulate,
which is worth saying plainly — the fix removed the mechanism that had been
hiding them.

- Finding: `LIVE_PRIORS_CYPHER` kept `LIMIT 200` with no ordering, but the live
  set no longer drains, so rows past the limit would never be shown, never
  accumulate `times_tested`, and never become retirable.
  - Fix: `LIVE_PRIORS_LIMIT = 2000` plus `curiosity_worldview_priors_truncated`
    when a read reaches it. **Not** the reviewer's suggested server-side
    `ORDER BY abs(p.confidence - 0.5)`: FalkorDB rejects the whole query on the
    first string-typed confidence, which costs Orion its entire world view
    rather than mis-ordering it, and Orion writes this Cypher by hand.
  - Evidence: reproduced live — `UNWIND [0.72, '0.72', null] AS c RETURN
    abs(coalesce(c, 0.5) - 0.5)` → `Type mismatch: expected Datetime, Date,
    Time, Duration, Integer, Float, or Null but was String`. `_as_float`
    tolerates the same value in Python, which is where the sort belongs.
    Pinned by `test_the_live_read_never_orders_on_a_value_orion_might_quote`.

- Finding: the sort key `(uncertainty, times_tested, prior_id)` is fixed across
  runs. Every prior formed at the prompt's template `confidence: 0.55` ties on
  the first two terms, so the same 8 lexicographically-lowest ids would be shown
  every run forever once the pool exceeds `sample`, and the rest never tested or
  retired.
  - Fix: the last term is now `_rotation_key(prior_id, run_id)` — a per-run
    hash. One run is reproducible; the window moves between runs. Rotation is
    strictly the last term, so it never reorders real signal.
  - Evidence: `test_least_tested_still_wins_over_the_rotation` and
    `test_the_most_uncertain_prior_leads_whatever_the_seed_is` sweep five seeds;
    `test_one_run_is_reproducible_but_the_window_moves_between_runs` asserts all
    three properties over a 40-prior pool.

- Finding: the `_FakeReader` needle `"p.status IN ['refuted'"` is a substring of
  `LIVE_PRIORS_CYPHER`'s own `NOT p.status IN [...]` and of `COUNTS_CYPHER`, so
  the settled rows were answering all three queries and `build_prior` was
  dropping them — the same fake-collision class `_StatusAwareReader` was written
  to eliminate, reintroduced two tests later.
  - Fix: anchored on `"ORDER BY p.last_tested_at"`, unique to that query.
  - Evidence: `test_the_settled_read_does_not_also_answer_the_live_read` now
    asserts `curiosity_worldview_unreadable_priors` does **not** appear in the
    log, which is the symptom that had been firing silently.

- Finding: the hub loop test's needle `"NOT p.status IN"` also matched
  `COUNTS_CYPHER`, so `live_total` fell back to `len(offered)` and the counts
  path was never exercised.
  - Fix: anchored on `"RETURN p.prior_id AS prior_id"`.
  - Evidence: 192 tests pass; the loop test now answers only the priors read.

- Finding: `RESOLVED_STATUSES` was dead repo-wide and asserted that `supported`
  and `revised` are resolved — the exact belief that caused the outage, sitting
  twenty lines above the comment warning against it.
  - Fix: deleted.
  - Evidence: `grep -rn RESOLVED_STATUSES` returns nothing.

- Finding: the `priors=%s/%s` log counted only `live_priors`, omitting
  `stale_priors`, which the prompt does show. A healthy run whose whole pool is
  stale would log `priors=0/3` — the outage signature this PR exists to
  prevent — while `curiosity_worldview_pool_dead` stayed quiet.
  - Fix: the log now counts live + stale, since this line is the operator-facing
    recurrence signal.
  - Evidence: `services/orion-hub/scripts/curiosity_investigation.py:844`.

## Review findings fixed — round two (the operator surface)

Eight findings, all valid. The first is the one worth reading:

- Finding: the schedule read built the daily-counter key from
  `datetime.now().astimezone()` — the *process's* local date. orion-hub sets no
  `TZ`, so that is UTC, while the loop keys on `HUB_ENDOGENOUS_OUTREACH_TZ`
  (`America/Denver`). Between 18:00 and 23:59 MDT the page would read tomorrow's
  key, find nothing, and show "Runs today 0 of 3" while the loop was at cap.
  - Fix: derive the zone from the same setting the loop uses.
  - Evidence: `docker exec orion-athena-hub` reports UTC. At 20:00 MDT the old
    expression gives `2026-08-28`, the new one `2026-08-27`. **The docstring
    directly above the bug claimed to avoid it** — same shape as
    `feedback_a_gate_can_be_green_through_the_incident_it_cites`, and it passed
    my live check only because today UTC and MDT share a date.

- Finding: the growth panel filtered segments to four hardcoded kinds while
  `total_added` summed every kind, so a run writing `:PriorRevision` (added by
  this very branch) rendered a full-width bar labelled "1" beside a total of
  "6", and the table's Total did not equal its own row.
  - Fix: kinds come from the data; slots 5–6 validated on the adjacent pairlist
    in both modes; a seventh folds into one "Other" segment.
  - Evidence: `test_every_node_kind_gets_a_segment_even_one_nobody_hardcoded`,
    `test_the_segments_always_sum_to_the_total_column`.

- Finding: the "left no trace" banner counted a run killed *mid*-write as having
  written nothing, contradicting its own body text and the ledger pill that
  already reported it correctly.
  - Fix: a run that wrote nodes but has no timestamp is accounted for; only a
    run that wrote nothing at all is reported missing.
  - Evidence: `test_a_run_killed_mid_write_is_not_reported_as_having_written_nothing`.

- Finding: `to_payload` ran on the event loop — only the query was threaded. It
  walks every prior's trajectory and deep-copies every dataclass, on Hub's
  single uvicorn worker, every 60s.
  - Fix: the whole projection moved into the thread, and revisions indexed by
    `prior_id` once instead of rescanned per prior (was O(priors × revisions),
    10M iterations at the query limits).

- Finding: an untimestamped `:PriorRevision` sorted *first* and drew that
  prior's chart backwards — Orion writes these by hand and can omit
  `timestamp()`.
  - Fix: unknown is not oldest, the same rule `assemble_runs` already applied.

- Finding: the appended trajectory endpoint is never a revision, yet was tagged
  `recorded: True` whenever any other revision existed — inverting the one
  distinction the flag carries.
  - Fix: always `False`, with the reason in the code.

- Finding: the page's pool counts come from rows that survived the read limit,
  while the loop counts server-side, so `pool_is_dead` and
  `curiosity_worldview_pool_dead` could disagree with no warning.
  - Fix: added the same truncation warning the loop's read has.

- Finding: `README.md` still said outreach was off by default after this branch
  flipped it, and `worldview.py` still described the tiebreak as stable across
  runs after `_rotation_key` made it deliberately not.
  - Fix: both corrected.

**Two of these were template-logic bugs with zero test coverage**, so the render
functions now execute under node against fixture payloads
(`tests/test_curiosity_atlas_template.py`, 11 tests). Both mutation-tested:
reverting either fix turns them red.

The new tests also exposed a collision worth recording — the repo root has its
own `scripts` package, so `from scripts.curiosity_routes import ...` was green
alone and `ModuleNotFoundError` once the orion-hub suite ran first. Loaded by
file path now.

## Risks / concerns

- Severity: low. Concern: the offered list now includes `supported` priors, so
  Orion may re-test a claim it considers settled. Mitigation: uncertainty sort
  deprioritises confident priors, `stale_after=3` caps repeats, and re-testing a
  once-tested claim is the intended behaviour, not a side effect.
- Severity: low. Concern: `_cypher_admits_status` in the tests is a small
  hand-rolled reader of the `WHERE` clause and could drift from FalkorDB's real
  semantics. Mitigation: it covers only the two predicate forms this module
  emits, and the live check above is the ground truth.
- Severity: informational. `HUB_CURIOSITY_OUTREACH_ENABLED` is now `true`, by
  Juniper's decision on 2026-08-27, after run `0a14e9531089` set
  `reach_out=true` for the first time and the message was logged as
  `curiosity_outreach_disabled` and dropped. No code changed; delivery already
  inherits every endogenous-outreach gate and still spends a second unified turn
  with its own stance gate.
- Severity: MEDIUM, and the reason this is `DONE_WITH_CONCERNS`. **The route's
  happy path through real Hub settings is UNVERIFIED.** Hub's `Settings`
  requires channel keys that only exist inside the container, so
  `_build_reader()` and `_read_schedule()` could not be exercised outside it.
  What *is* verified: `read_atlas` against the live FalkorDB with real data, the
  router importing and exposing nothing but GET, the render functions against a
  real payload, and graceful degradation when settings are absent. One
  `curl http://localhost:<hub-port>/curiosity/api/atlas` after deploy closes it.
- Severity: LOW. The page has not been seen in a browser — no browser on this
  host. Layout, wrapping and paint are unverified by anything but reading.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1915
