# The crystallization grammar-ref resolver was a string check in a costume

## Summary

- `resolve_grammar_event_ref` had been dead since it was written. It queried
  `grammar_traces WHERE trace_id = $1 OR event_id = $1` — but `grammar_traces` has **no
  `event_id` column**, so the statement raised on every call into a bare
  `except Exception: pass`. It then fell through to `substrate_grammar_events`, a table that
  does not exist here, which raised too. Its real behaviour was its last line and nothing
  else: `return str(event_id).startswith("gev_")`.
- It sits in the path that decides whether a crystallization proposal gets **quarantined**.
- Now queries `grammar_events`, which really does have `event_id` and `trace_id`, and accepts
  either.
- **An absent grammar ref is now reported and never fatal**, because `grammar_events` is the
  only retention-bounded source store and no timestamp can distinguish "aged out" from "never
  existed" (see below — I tried, and review proved it wrong on live data).
- Both carriers of grammar ids are walked as one deduplicated set, so a ref cannot be excused
  by one loop and condemned by the next.
- Surfaced in the Hub UI, guarded the route against a new 500.

## Outcome moved

A validator that could not fail now actually resolves references. Live 2026-08-20: 1,260
crystallizations, 124 carrying grammar refs, 1,167 distinct ids — **all** `gev_`-prefixed, so
the old code returned `True` for every one, including the 999 whose events no longer exist.

## Current architecture

`resolve_crystallization_sources` validates three source families and feeds
`crystallization_routes.py`'s validate endpoint, which quarantines on any `unresolved` entry
and persists that status.

## Files changed

- `orion/memory/crystallization/sources.py`: real lookup; `_grammar_ref_ids` union;
  `absent_grammar_refs` / `unverified_grammar_refs`; retention-horizon inference **removed**.
- `services/orion-hub/scripts/crystallization_routes.py`: 503 guard around resolution;
  surfaces both new fields.
- `services/orion-hub/static/js/memory-crystallization-ui.js`: renders them.
- `orion/memory/crystallization/tests/test_grammar_source_resolution.py`: new, 14 tests.

## The inference I removed, and why it is not worth retrying

My first version classified an absent ref as benign ("pruned") when the crystallization's own
`created_at` predated the live retention horizon `MIN(created_at) FROM grammar_events`, and
fatal otherwise. It looked well-evidenced: on live data it split 876 aged-out refs from 14
"genuinely missing" ones.

Review killed it, with live data, on three independent grounds:

1. **Crystallizations copy refs forward.** `65b0662d` inherited seven ids verbatim from
   `4b4bd619`, minted 25 hours earlier. The *same seven ids* got opposite verdicts from the two
   carriers — proof the rule measured the carrier, not the reference.
2. **Refs are not contemporaneous with their carrier.** Ref age relative to the carrying
   crystallization: p50 495s, **p95 18.3 hours, max 43.6 hours** — a large fraction of the
   3-day window.
3. **All 14 "genuinely missing" refs were in fact aged out.** One crystallization, seven
   distinct ids, each listed twice. A **100% false-positive rate on its own error bucket**, and
   "14" overstated the evidence by 14×.

I then checked whether the ref row's own `memory_crystallization_sources.created_at` would
work instead. It does not: it defaults to `now()` at INSERT, so a copied ref carries the
*copying* proposal's timestamp, not the original event's.

So there is no timestamp on either side that distinguishes the two cases, and a validator
claiming to make that distinction is confabulating. The honest contract is: look it up for
real, report what was found, and never invalidate a proposal because the substrate did its own
housekeeping. If the distinction is ever genuinely needed, the sound fix is to persist the
resolution outcome when the ref is first recorded — not to re-derive it from clocks afterwards.

## Review findings fixed

- **HIGH — the feature was inert in production.** `pruned` was computed from
  `source_grammar_event_ids`, then the evidence loop walked
  `evidence[kind=grammar_event]` — the *same ids* — with no pruned logic, appending them to
  `unresolved` and `errors`. Live: of the 61 crystallizations the rule would excuse, **61
  also carried the identical id as evidence**, so all 61 still quarantined. The headline claim
  "does not invalidate" was false for every row it claimed to protect.
  - Fix: `_grammar_ref_ids()` returns the deduplicated union of both carriers, walked once.
  - Evidence: `test_the_evidence_loop_cannot_re_flag_what_the_grammar_loop_excused`.

- **HIGH — the horizon rule was wrong.** See above. Removed entirely, and
  `test_the_retention_horizon_is_never_consulted` pins that it cannot come back.

- **MEDIUM-HIGH — the new `raise` escaped into an unhandled 500.**
  `resolve_evidence_ref` → `resolve_grammar_event_ref` had no `try/except`, and the route did
  not wrap `resolve_crystallization_sources`. Before the patch that path returned
  `startswith("gev_")` and never raised, so this was strictly worse.
  - Fix: probe failure is caught and recorded as `unverified_grammar_refs`; the route also
    wraps resolution in the 503 pattern already used for `get_crystallization`.

- **MEDIUM — the raise bought nothing.** The docstring said "let the caller decide", but the
  caller appended to `unresolved` + `errors`, which quarantines — byte-identical to returning
  `False`, the outcome the comment claimed to prevent. Compounding it,
  `grammar_retention_horizon` *swallowed* its own failure while the ref probe raised: opposite
  policies for the same underlying failure.
  - Fix: unverified refs never invalidate; the horizon function is gone.

- **MEDIUM — a transient window flipped `active` proposals to `quarantined`, permanently.**
  Drift is monotone, so a proposal's lifecycle was `valid` → **`quarantined`** → `valid`, with
  the middle window as wide as the ref lag (up to 43.6h). 38 of the 62 affected rows are
  `status='active'`, and nothing walks a demotion back.
  - Fix: dissolved — grammar-ref absence no longer invalidates at all.

- **MEDIUM — the tests structurally could not catch any of this.** `_FakeCrystallization`
  hardcoded `self.evidence = []`, so the evidence loop never ran once across all 12 tests.
  That is precisely where the feature was defeated and where the 500 lived. My 5-mutation
  exercise could not have caught it either: mutations to unreached code are invisible.
  - Fix: the fixture now mirrors ids into `evidence` **by default**, matching live data (2,586
    `grammar_event` rows in `memory_crystallization_sources`; every affected crystallization
    carried the same ids in both carriers).

- **MEDIUM — `pruned_sources` was write-only.** Repo-wide it occurred exactly once: the line
  that wrote it. The only client discarded the response body entirely.
  - Fix: the UI now reads both fields and appends them to the validate status line. (I first
    wrote this with an early `return` that skipped the inbox reload — caught and fixed before
    commit.)

- **LOW — "14 genuinely missing" was misleading** even setting aside that none were genuine:
  it was 7 distinct ids in 1 crystallization, each listed twice. Corrected here.

- **LOW — duplicate array entries inflated the counts.** Fixed by the dedup above.

Review verified and I did not change: query plans (BitmapOr, 8 buffers / 0.32ms; horizon was
an Index Only Scan), asyncpg fake fidelity (`row is not None`, not truthiness, so `Record` vs
dict is irrelevant), backward compat (one keyword-only construction site, no positional
unpacking or serialisation anywhere), and timezone safety.

## Schema / bus / API changes

- Added to the validate response: `absent_grammar_refs`, `unverified_grammar_refs`.
- Removed: `pruned_sources` (never merged; introduced and removed within this branch).
- Behaviour changed: a proposal is **no longer quarantined** for grammar refs that do not
  resolve. Non-grammar evidence still invalidates exactly as before.

## Env/config changes

None.

## Tests run

```text
$ PYTHONPATH=. pytest orion/memory/crystallization/tests \
    tests/test_crystallization_hub_provenance_ui.py \
    services/orion-hub/tests/test_crystallization_review_queue_ux.py -q
47 passed
```

`tests/test_memory_crystallization.py::TestMemoryCardBackwardCompat::test_memory_card_v1_unchanged_in_registry_gap`
fails on `main` too and is unrelated.

Earlier mutation run against the original implementation: restoring
`return str(event_id).startswith("gev_")` fails 8 of 12 tests; 5 targeted mutations all caught.
Noting honestly that this exercise did **not** catch the inert-feature bug, because the code it
would have mutated was never reached by any test — which is the lesson, not the reassurance.

## Evals run

No eval harness for `orion/memory/crystallization/`. The live-data determination that drove
this design (ref-lag percentiles, the copy-forward chain, 61/61 still quarantining) came from
the review's read-only probe against Postgres rather than a committed eval. Worth a follow-up.

## Docker/build/smoke checks

Not deployed. This touches `orion/` (imported by Hub) and Hub route/JS; no service was rebuilt
as part of this branch. **Flagged under concerns.**

## Restart required

```bash
# to pick up the Hub route + static JS change
./scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: **not deployed or smoke-tested.** The route and JS changes are
  unexercised outside unit tests. Mitigation: restart command above; verify one validate call
  returns the new fields.
- Severity: medium. Concern: proposals can no longer be quarantined for unresolvable grammar
  evidence at all. That is deliberate and argued above, but it does remove a signal. Mitigation:
  the absence is reported in the API and rendered in the UI rather than silently dropped. The
  sound long-term fix is persisting resolution outcome at record time.
- Severity: low. Concern: 38 crystallizations are currently `status='active'` and 24
  `rejected` among the 62 affected; any that were quarantined by the *old* behaviour are not
  walked back by this patch. Mitigation: re-validating them now returns `valid`.
- Severity: informational. Concern: reviewed by subagent, but the two earlier review attempts
  died to session limits and this branch has had one full pass, not two.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1783
