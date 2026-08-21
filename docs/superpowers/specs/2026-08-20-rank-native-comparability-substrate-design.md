# Rank-native comparability substrate: stop patching the same bug in five costumes

2026-08-20. Design mode. Supersedes the habituation-only design
(`2026-08-20-habituated-society-of-mind-design.md`) and the first cut of
`concern_state.py` — both were real progress but were still patches at the wrong layer.
This is the level-up: the architecture that closes the *class* of bug those patches were
each fixing one instance of.

## Arsonist summary

Five separate incidents, all in this repo, all the same disease:

1. **The killed drives system**: `dominant_drive=relational` in 96% of ticks — one
   hand-tuned linear formula's raw output compared directly against every other drive's
   raw output, no shared scale.
2. **`compute_salience()`**: `pressure*0.45 + novelty*0.20 + urgency*0.25 +
   confidence*0.10` — hand-typed weights blending four differently-scaled signals into
   one cardinal number, zero citation, zero calibration.
3. **`node:substrate.route`** (PR #1774, today): a domain with genuinely near-zero
   organic variance got compared, via `1/variance`, against domains with real organic
   variance 1,270x-19,300x larger — same absolute floor applied to incomparable scales.
4. **`node:substrate.chat`**: multi-thousand-tick mega-streaks (up to 16,036), found
   during today's item-4 calibration run, structurally the same shape, never fully
   root-caused.
5. **`concern_state.py`'s first cut** (today, caught by code review before commit):
   Candidate A's min-max-normalized node scores and Candidate B's raw, un-normalized
   host novelty scores were classified against the same cardinal thresholds as if they
   were on the same scale. They were not.

Every one of these is the same root cause: **something compared raw cardinal numbers
across domains that were never guaranteed to share a scale.** Fixing each instance
individually — a variance floor here, a population split there — is patching turds. The
fifth instance happened in code written *today*, in the same session that fixed the
third, by the same author who diagnosed the pattern. That is direct proof patch-by-patch
fixing does not stop this from recurring; only removing the code path that makes the
mistake expressible does.

**The fix already half-exists in this repo and has been sitting unused for a month.**
`orion.attention.rank_aggregation` (Borda count, de Borda 1770, extracted 2026-07-31 from
Candidate B) compares targets by *rank*, never by raw magnitude. Rank is scale-invariant
by construction — a scorer's raw numbers can be anything, on any scale, and the
aggregation is unaffected, because only ordinal position ever crosses the domain
boundary. This single property closes incidents 3, 4, and 5 simultaneously, not because
someone remembers to apply the right patch each time, but because the comparison that
caused all three becomes structurally inexpressible.

## Current architecture

- **`orion.attention.rank_aggregation`**: `aggregate_borda(scorer_scores: dict[str,
  dict[str, float]], universe=None) -> BordaResult` — already generic, already accepts
  any number of scorers, already has no cross-scorer weight to calibrate (`totals`,
  `ranking`, `winner`, `per_scorer_top1`, `disagreement`). Two real consumers already:
  Candidate B field attention and `orion.substrate.attention.salience` (chat-level
  coalition strength). Neither uses it as the *only* comparison path in its subsystem —
  both still have direct-cardinal-comparison code paths elsewhere (Candidate A's argmax,
  `select_actions()`'s hand-set thresholds).
- **Candidate A** (`candidate_precision_weighted.py`): cardinal, `1/variance`-based,
  today's `cross_domain_variance_floor` fix is a real, working *patch* on this cardinal
  path — it does not remove the cardinal-comparison code path itself, it makes one
  specific failure mode of it less likely.
- **Candidate B** (`candidate_society_of_mind.py`): rank-native by design, but its own
  docstring discloses the full three-scorer combination has never been validated against
  real data — only `magnitude_scorer()` has (`measure_society_of_mind_magnitude_probe.py`).
  A month-old, still-open gap.
- **`select_actions()`** (`orion/substrate/attention/policy.py`): cardinal thresholds
  (0.48/0.35), the function's own comment discloses these were tuned against an older
  score shape and never revalidated against its current one.
- **`concern_state.py`** (today, uncommitted, not merged): multi-item classification is
  the right *shape* — proven live-data-supported even after removing the scale-mixing
  bug (Candidate A alone, properly isolated: 65.28% of real ticks have 2+ genuinely
  comparable concerns active at once) — but it classifies by a borrowed cardinal
  threshold, the same category of mistake this document exists to stop.

## Missing questions

1. Does replacing Candidate A's cardinal argmax with rank-aggregation lose real
   information the cardinal score carries (e.g., *how much* more surprising one target is
   than another, not just that it ranks higher)? Real trade-off, not free — Borda
   deliberately discards magnitude in exchange for scale-safety. Named, not resolved here:
   the acceptance check below measures whether this costs anything real.
2. How many scorers does a target need real votes from before its rank is trustworthy?
   A target scored by only one scorer still gets a rank, but with less real evidence
   behind it than one scored by three agreeing scorers. `per_scorer_top1`/`disagreement`
   already carry this information — not yet surfaced to any consumer.
3. Can tier boundaries (active/watch/quiet) be derived from each tick's own rank/points
   distribution instead of any hand-set or borrowed constant? Proposed below: yes,
   via the tick's own percentile distribution — self-calibrating per tick, no global
   constant, same precedent `cross_domain_variance_floor` already established.

## Verified correction: naive union-universe Borda does not actually fix the bug

**The first version of this document asserted "make `aggregate_borda()` the mandatory
substrate" without checking whether calling it on the union of two non-overlapping
target universes actually produces a fair comparison. It does not — checked with real
code, not asserted, after the claim was challenged.**

Reproduced the exact real shape from today's data — Candidate A real-scores 5 targets
(`node:substrate.*`), Candidate B real-scores 3 (`node:athena`/`atlas`/`circe`), and
`candidate_society_of_mind.py`'s own docstring already discloses these universes never
overlap in live data:

```python
from orion.attention.rank_aggregation import aggregate_borda
scorer_a = {"route": 0.1, "chat": 0.3, "biometrics": 0.5, "execution": 0.7, "bus_synaptic": 0.9}
scorer_b = {"athena": 0.2, "atlas": 0.5, "circe": 0.8}
result = aggregate_borda({"magnitude": scorer_a, "novelty": scorer_b})
```

Result: Candidate A's best total (`bus_synaptic`, 9.0) beats Candidate B's best
(`circe`, 8.0); Candidate A's worst (`route`, 5.0) is *below* Candidate B's worst
(`athena`, 6.0). Candidate A structurally gets both the higher ceiling and the lower
floor, purely because it has 5 targets to Candidate B's 3 — more room to spread across
Borda's `0..N-1` point range on its own ballot, nothing to do with real cross-domain
surprise. Naive union-universe Borda reproduces a version of the exact bug it was
proposed to fix, just keyed on universe size instead of raw magnitude.

**Real fix, verified, not asserted:** normalize each scorer's ballot to its own
`rank_position / (own_universe_size - 1)` — a percentile in `[0, 1]` computed
independently within each scorer's real universe — before combining across scorers.
Re-run on the same data:

```python
def percentile_ballot(scores: dict) -> dict:
    n = len(scores)
    if n <= 1:
        return {k: 1.0 for k in scores}
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    return {tid: i / (n - 1) for i, (tid, _) in enumerate(ordered)}
```

Result: best-of-Candidate-A (`bus_synaptic`, 1.000) exactly ties best-of-Candidate-B
(`circe`, 1.000); worst-of-Candidate-A (`route`, 0.000) exactly ties worst-of-Candidate-B
(`athena`, 0.000) — the universe-size bias is gone, verified by direct computation, not
assumed from the theory.

**This fix has its own real, disclosed cost, also checked, not glossed over:** percentile
normalization guarantees each domain's own top target always hits exactly 1.0, regardless
of true magnitude — the same "always-ceiling" artifact `normalize_across_targets()`'s own
"single/all-equal edge case" already has (see that function's docstring), just now
happening independently per domain instead of once globally. A domain with only one real
competitor this tick will *always* report a 1.0-percentile target, real surprise or not.
This means "how many domains have an active concern" is a more honest question for this
mechanism to answer than "how surprising is the most active concern" — the latter is
still not comparable across domains, even after this fix.

## Proposed schema / API changes

No new schema, bus channel, or registry entry in this patch. New pure functions only,
built on the VERIFIED percentile mechanism above, not the disproven naive-union approach:

```python
# orion/attention/rank_aggregation.py, new function alongside aggregate_borda() --
# percentile normalization happens BEFORE cross-scorer combination, not after, since
# aggregate_borda()'s own point range depends on total universe size in a way percentiles
# must correct for first (see verified correction above).

def percentile_ballot(scores: dict[str, float]) -> dict[str, float]:
    """Rank position / (own real universe size - 1), computed independently within one
    scorer's own real universe. [0,1], comparable across scorers with different universe
    sizes -- verified 2026-08-20 to remove the universe-size bias naive aggregate_borda()
    union has (see this module's design doc). Single-target universe -> 1.0 (matches
    normalize_across_targets()'s own single-target-edge-case precedent: no real basis to
    call a sole real competitor anything but fully salient)."""
```

```python
# orion/attention/field_attention/concern_state.py, REPLACING the cardinal-threshold
# version built today (which is not being kept -- same mistake this doc exists to stop)

def classify_concern_state_from_percentiles(percentile_ballots: dict[str, dict[str, float]]) -> ConcernSetV1:
    """Combine each scorer's percentile_ballot() (already comparable across domains) via
    aggregate_borda() -- now safe, since inputs are already [0,1]-normalized per-scorer
    before combination, not raw magnitudes or raw universe-dependent point counts. Tiers
    derived from THIS TICK's own combined-percentile distribution mean/std -- self-
    calibrating, no hand-set or borrowed constant (the exact mistake in
    `select_actions()`'s disclosed-stale 0.48/0.35 and in this module's own first cut).

    active:  combined_percentile[target] >= mean + 0.5 * std
    watch:   combined_percentile[target] >= mean
    quiet:   below mean

    Degenerate case (std == 0): falls back to watch-only classification for anything at
    or above the single mean value -- matches normalize_across_targets()'s own
    "all-equal -> no arbitrary floor" precedent.
    """
```

```python
# orion/attention/field_attention/selectors.py or a new module -- wiring Candidate A's
# raw precision-weighted output as ONE Borda scorer (not the sole cardinal decision-maker)

def precision_rank_scorer(node_target_results: dict[str, "PrecisionWeightedSalienceResult"]) -> dict[str, float]:
    """Candidate A's raw `.salience` values, submitted as ballots to Borda -- NOT
    compared to each other directly (that comparison is exactly what
    `cross_domain_variance_floor` was patching around). Candidate A's own internal
    ranking is preserved (a real, still-useful signal -- "route is more surprising than
    it usually is" is real information); only the CROSS-domain step changes from
    cardinal to ordinal.
    """
```

## Files likely to touch

- `orion/attention/rank_aggregation.py` — new `percentile_ballot()` function (verified
  above; naive `aggregate_borda()` on a raw union of non-overlapping universes is
  confirmed biased by universe size and must not be used directly for cross-Candidate
  comparison without this normalization step first).
- `orion/attention/field_attention/concern_state.py` — replace cardinal-threshold
  classification with percentile/Borda-distribution classification (function signature
  changes; this is a rewrite, not an addition, since the cardinal version is the exact
  thing being retired).
- `scripts/analysis/measure_society_of_mind_full_replay.py` (new) — finally closes
  Candidate B's month-old validation gap: real `novelty_scorer`/`dwell_scorer` data,
  full three-scorer `aggregate_borda()`, replayed against real history. This is the
  FIRST real acceptance check both for Candidate B itself and for this architecture,
  not a separate task.
- `scripts/analysis/measure_concern_state_multiplicity.py` — rebuilt to compute Borda
  totals (magnitude scorer at minimum; novelty/dwell once Phase 1 validates them) instead
  of reading pre-mixed cardinal scores from `substrate_attention_frames`.
- `tests/test_attention_concern_state.py` — rewritten against the Borda-based
  classifier; the cardinal-threshold tests built today are retired, not extended.
- `orion/sentience_striving_program/README.md` §9a item 2 — record as superseding both
  the habituation-only design and today's first `concern_state.py` cut, with the honest
  reason why (caught a real instance of the exact bug this design exists to prevent, in
  code written the same day).
- No `.env_example`, docker-compose, bus channel, or schema registry files — pure
  functions and read-only replay only, per "measure before minting."

## Non-goals

- Not wiring anything to a live consumer — shadow/read-only until real replay data
  exists, same discipline as every other instrument in this program.
- Not deprecating Candidate A's cardinal score as a per-domain internal signal — a
  domain's own raw magnitude is still real, useful information *within* that domain
  (e.g., for the domain's own EWMA/variance tracking). What changes is that raw magnitude
  never crosses a domain boundary to be compared against another domain's raw magnitude
  again, anywhere.
- Not solving `select_actions()`'s own disclosed threshold-staleness in this patch —
  named as a real, separate follow-up this architecture makes easier (once rank-native
  comparison is the norm, `select_actions()` becomes a candidate for the same rewrite,
  not forced into it here).
- Not reintroducing a named-drive taxonomy — this is a comparison mechanism, not a
  category system.
- Not claiming this makes Orion sentient, or asserting any subjective state exists.
  This closes a structural precondition (multiple real concerns become representable
  without a recurring scale-mixing bug); what, if anything, is ever built on top of that
  is a separate, later, evidence-gated decision.

## Acceptance checks

1. **Candidate B's own month-old gap, finally closed**: real three-scorer Borda
   replay against historical `substrate_attention_frames`/`substrate_coalition_dwell_log`
   data. Report whether the three scorers ever disagree, and whether that disagreement is
   informative — the literal, never-met bar from the original design doc.
2. **Does rank-native classification actually avoid the mixed-scale bug**: replay the
   SAME real Candidate-A + Candidate-B population that produced today's false 92% figure,
   through `classify_concern_state_from_percentiles()` instead of the retired cardinal
   classifier, and confirm the tier distribution no longer depends on which raw scale a
   target's scorer happened to use. Concretely checkable, same as the verified check
   above: shuffle/rescale one scorer's raw output (multiply by 1000, or divide by 1000)
   and confirm the resulting tiers are IDENTICAL — real, mechanical proof, not an
   assertion. **Partially pre-verified above with synthetic data** (universe-size bias
   confirmed real, percentile fix confirmed to remove it) — this check is the same
   proof re-run against real historical ticks instead of a 5-vs-3 synthetic example.
3. **Does self-calibrated tier banding produce non-degenerate tiers on real data**: report
   the real distribution of active/watch/quiet counts per tick, same shape as today's
   (now-retired) multiplicity replay, so the "no hand-picked constant" claim is checked
   against real numbers, not just the formula's structure.
4. **Cost check on the magnitude information Borda discards** (Missing Question 1): compare,
   on the same real data, how often Candidate A's cardinal argmax winner and the
   rank-native active-tier's top rank disagree, and whether the cases they disagree on
   look like real, meaningful differences or noise.

## Recommended next patch

Phase 1: build `measure_society_of_mind_full_replay.py`, closing Candidate B's real
three-scorer validation gap — the actual foundation this whole architecture rests on.
Phase 2, only after Phase 1's real numbers exist: rewrite `concern_state.py` on Borda
totals instead of cardinal thresholds, and re-run the multiplicity replay through it,
reporting the scale-invariance proof from acceptance check 2 explicitly. Both phases
shadow/read-only, replayed against real historical data, nothing wired to a live
consumer, per this program's own standing rule.
