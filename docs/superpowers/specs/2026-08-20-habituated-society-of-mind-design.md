# Habituation as a fourth Society-of-Mind voter: closing the argmax-domination trap

2026-08-20. Design mode. Follows directly from the Sentience Striving Program §6 item 7
decision ("integration not warranted by current data") — this is the concrete answer to
"then let's build the version that doesn't have that problem," not a reversal of that
decision. Item 7 said no to fusing/integrating the *current* instruments as-is; this
proposes what a non-myopic instrument would actually need to look like before any future
integration question is worth re-asking.

## Arsonist summary

Every scoring formula in this repo that has ever picked a single attention/salience
"winner" — the killed drives system, `compute_salience()`'s hand-tuned linear blend, and
Candidate A's precision-weighted competition (`orion/attention/field_attention/
candidate_precision_weighted.py`) — shares one structural property nobody has named
directly: **argmax with no memory.** Pick the single highest score, every tick, judged
fresh each time as if no tick before it happened. `node:substrate.route` winning by a
real math bug (fixed today, PR #1774) and `node:substrate.bus_synaptic` winning 36.31% of
104,778 real ticks by honest organic signal (measured today, item 7) are the *same
failure mode wearing two different masks* — one was a broken number, the other is a
broken shape. Fixing the number did nothing to the shape.

The one real attempt at a structurally different shape — Candidate B, Society-of-Mind
Borda rank-aggregation (`candidate_society_of_mind.py`, built 2026-07-21/22) — has been
sitting shadow-only for a month. Nobody killed it, nobody finished validating it either:
its own module docstring discloses that only ONE of its three scorers
(`magnitude_scorer`) has ever been run against real data; `novelty_scorer`,
`dwell_scorer`, and the full three-way Borda combination have never been replayed once.
That is the real gap this design should close first, before adding anything new to it —
building a fourth mechanism on top of an unvalidated three-scorer base would repeat
exactly the "build without validating" mistake this whole program exists to correct.

## Current architecture

- **Candidate A** (`candidate_precision_weighted.py`): precision-weighted argmax,
  per-tick, no cross-tick state except each target's own EWMA baseline (which tracks the
  target's *own* statistics, not how long it has been winning). Today's fix
  (`cross_domain_variance_floor`, PR #1774) makes cross-target confidence comparisons
  honest but does not touch the winner-take-all *shape* — whichever target has the
  highest corrected score still wins every single tick, with nothing discounting a target
  that has already won many ticks in a row.
- **Candidate B** (`candidate_society_of_mind.py` + `orion.attention.rank_aggregation`):
  three independent scorers (magnitude, novelty, dwell) combined via Borda count
  (de Borda 1770) — no cross-scorer weight, by design. `aggregate_borda(scorer_scores:
  dict[str, dict[str, float]], universe=None) -> BordaResult` is already generic — it
  accepts any number of scorers, not hard-coded to three. Shadow-only, never wired to any
  live consumer, and its own docstring's "Live-data sanity check status" section (still
  accurate as of this design) discloses the three-scorer combination has never been
  replayed against real data. A real, disclosed, unresolved gap: the magnitude scorer's
  target universe (`node:substrate.*`, five domains) and the novelty scorer's target
  universe (`node:athena`/`node:atlas`/`node:circe`/`capability:*`/
  `field:recent_perturbations`) did not overlap in live data as of 2026-07-31 — genuine
  three-way competition on the same target is rare.
- **`DominanceStreak`** (`orion/attention/field_attention/goal_provenance.py`): already
  tracks, per real tick, how many consecutive ticks the *current* node-target-subset
  winner has held the top-1 slot (`target_id: str | None`, `count: int`) — scoped to
  `orion-attention-runtime`'s goal-provenance producer only, not threaded into the raw
  per-tick salience score itself, and not covering system/capability targets. This is
  the closest existing real ingredient to what habituation needs, and today's calibration
  run (item 4) already used its distribution for a different question
  (`ORION_GOAL_PROVENANCE_MIN_STREAK`'s debounce value).
- **Item 7's measurement, today**: full-history top-1 concentration is 36.31%
  (`bus_synaptic`), comparable to the killed drives system's own post-fix concentration
  (31.65%) — the number this design's acceptance check must beat.

## Missing questions

1. Has Candidate B's own three-scorer Borda combination ever actually been validated
   against real data? **No — confirmed via the module's own current docstring.** This
   design proposes closing that gap as Phase 1, before Phase 2 (adding a fourth scorer).
   Skipping straight to Phase 2 would mean nobody has ever checked whether the base
   mechanism this proposal extends even works.
2. Does the magnitude/novelty target-universe-overlap gap need solving before a real
   three-way (or four-way) competition is meaningful? Real open question, not assumed —
   Phase 1's replay will show directly whether real competition happens often enough to
   evaluate, or whether this needs its own follow-up (e.g. teaching `orion-field-digester`
   to ingest prediction-error domain nodes into `FieldStateV1.node_vectors`, named as a
   real follow-up in Candidate B's own docstring already).
3. What decay rate/floor should habituation use? Answered below with a self-calibrating
   approach (no hand-picked constant), consistent with `cross_domain_variance_floor`'s own
   precedent from today.
4. Should habituation live inside Candidate A (a multiplier on the existing cardinal
   score) or as a fourth Candidate B voter (a rank)? **Recommend the latter** — Candidate
   B's entire design philosophy is "no cross-scorer weight is ever guessed or
   calibrated"; a multiplicative discount bolted onto Candidate A's cardinal score would
   itself be exactly the kind of hand-tuned weight this program exists to avoid. A fourth
   *rank* (how fresh is this target, ordinally) fits Borda's existing weight-free
   combination mechanism directly. This does knowingly revise Candidate B's own current
   "does not invent a fourth scorer" non-goal — a deliberate, disclosed change, not an
   oversight.
5. Does real historical data show habituation would actually change the winner
   distribution, or would `bus_synaptic` just keep winning because its raw scores are
   *that much* higher even after a freshness penalty? Not knowable without running it —
   named explicitly as Phase 2's acceptance check, not assumed to work.

## Proposed schema / API changes

No new schema, bus channel, or registry entry. This stays shadow-only/read-only through
both phases — the whole point is to validate before minting, per §7's own rule.

**Phase 1** (validate the existing candidate): no code changes to `candidate_society_of_mind.py`
itself. New read-only replay script, `scripts/analysis/measure_society_of_mind_full_replay.py`,
extending the existing `measure_society_of_mind_magnitude_probe.py` pattern to also fetch
real `novelty_scorer`/`dwell_scorer` inputs (`substrate_attention_frames`,
`substrate_coalition_dwell_log`) and run the full three-scorer `aggregate_borda()` against
real historical data, reporting: how often the three scorers actually disagree (the plan
doc's own named acceptance check, never run), and what the resulting Borda winner
distribution/concentration looks like compared to Candidate A's argmax.

**Phase 2** (habituation as a fourth voter), only after Phase 1's real numbers are in
hand:

```python
# orion/attention/field_attention/candidate_society_of_mind.py, new function

def freshness_scorer(
    target_ids: list[str],
    streak_state: dict[str, "HabituationState"],
) -> dict[str, float]:
    """Fourth independent Society-of-Mind voter: how long has this target been
    the *current* winner, inverted -- a target on a long win streak scores low,
    a target that just started winning (or has never won) scores at its ceiling.
    Real theory anchor: Groves & Thompson 1970, "Habituation: A dual-process
    theory" (Psychological Review) -- repeated/sustained stimulation reduces a
    stimulus-response pathway's output over time (habituation), but a stimulus
    that changes or intensifies relative to its own recent baseline overrides
    that suppression (dishabituation) via a separate state/arousal process.
    This is why the discount is keyed on *sustained sameness*, not sustained
    presence -- see HabituationState.
    """
```

```python
@dataclass(frozen=True)
class HabituationState:
    """Per-target habituation state, same explicit-threading shape as
    `PrecisionEwmaBaseline`/`DominanceStreak`. consecutive_win_ticks tracks how
    long this target has held the top-1 slot in whatever competition feeds it
    (initially: Candidate B's own Borda winner, or Candidate A's argmax winner
    for a direct concentration comparison). baseline_value/baseline_variance
    are the SAME EWMA machinery already used everywhere in this repo
    (orion.bus.ewma.compute_ewma_update) tracking this target's own recent raw
    reading -- not a new mechanism, reuse. dishabituation fires when the
    current reading deviates from baseline_value by more than a z-score
    threshold, resetting consecutive_win_ticks toward zero -- a genuinely
    escalating signal is never suppressed by having "already been noticed."
    """
    target_id: str
    consecutive_win_ticks: int = 0
    baseline_value: float = 0.0
    baseline_variance: float = 0.0
```

Decay shape: `freshness_score = floor + (1 - floor) * exp(-k * consecutive_win_ticks)`.
`floor` bounds the discount so a channel is *never* fully silenced by habituation alone
(matches CLAUDE.md's "no empty-shell cognition" — a real ongoing concern must never read
as literal zero just because it has already been flagged once). `k` is not hand-picked:
calibrate it the same self-calibrating way `cross_domain_variance_floor` avoided a
borrowed constant — derive it from the real win-streak-length distribution
`measure_goal_provenance_streak_distribution.py` already pulled today (median streak
length, 5,516 real runs), so the habituation curve's half-life is anchored to this
system's own real behavior, not a guessed number.

## Files likely to touch

- `scripts/analysis/measure_society_of_mind_full_replay.py` (new, Phase 1) — real replay
  closing Candidate B's own disclosed validation gap.
- `orion/attention/field_attention/candidate_society_of_mind.py` (Phase 2) —
  `freshness_scorer()`, `HabituationState`, explicitly revising the module's current
  "does not invent a fourth scorer" non-goal (disclosed above, not silent).
- `scripts/analysis/measure_habituated_concentration_probe.py` (new, Phase 2) — replays
  habituation-adjusted Borda winners against the same real historical window item 7 used,
  reporting concentration side-by-side against both Candidate A's argmax (36.31%) and
  Candidate B's un-habituated Borda (from Phase 1).
- `tests/test_attention_candidate_society_of_mind.py` — new tests for both functions,
  including a synthetic case proving a genuinely escalating signal is NOT suppressed
  (the dishabituation guarantee), matching this session's "hand-compute test fixtures"
  discipline.
- `orion/sentience_striving_program/README.md` — record both phases as dated entries
  under §9a item 2 (Society-of-Mind), graduating it from an un-sequenced blue-sky option
  to an active, phased thread.
- No `.env_example`, docker-compose, bus channel, or schema registry files — pure
  functions and read-only analysis scripts only, both phases.

## Non-goals

- Not wiring anything to a live consumer in either phase — stays shadow/read-only, per
  §7's "measure before minting."
- Not reintroducing named drives or a taxonomy — `freshness_scorer` is a scoring
  mechanism with a real theory citation, not a named motivational category.
- Not touching `cross_domain_variance_floor` or Candidate A at all — orthogonal, already
  shipped and correct for the problem it solves (cross-domain confidence comparability,
  not temporal monopolization).
- Not solving item 7's O3/narrative-legibility gap — separate, already-disclosed problem.
- Not closing the magnitude/novelty target-universe-overlap gap in this same patch unless
  Phase 1's real numbers show it's blocking a meaningful replay — named as a possible
  follow-up, not assumed necessary.
- Not deciding, in this document, whether Candidate B (habituated or not) should ever
  replace Candidate A live — that is a real integration decision for a future Objective
  7-shaped re-evaluation, once both phases produce real data, not this document's call to
  make.

## Acceptance checks

**Phase 1** — Candidate B's own long-standing, never-met acceptance bar
(`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-
plan.md`'s Candidate B section): do the three scorers ever disagree on real data, and is
that disagreement informative — reported with real examples, not a summary claim. Plus:
what is Candidate B's own top-1 concentration on the same real historical window, compared
directly against Candidate A's 36.31%.

**Phase 2**: (a) habituation-adjusted concentration must be measurably lower than both
Candidate A's argmax (36.31%) and Candidate B's un-habituated Borda from Phase 1 — if it
isn't, the mechanism doesn't work and that must be reported as a real negative result, not
hidden. (b) A synthetic test proves a target whose raw reading is genuinely escalating
(not just persisting) is not suppressed by its own prior win streak — the dishabituation
guarantee, checked directly, not assumed from the formula's shape. (c) No target's
freshness score ever hits exactly zero (the floor is respected) — checked against real
replay data, not just the unit-level formula.

## Recommended next patch

**Phase 1 only, first**: build `measure_society_of_mind_full_replay.py`, run it against
real `substrate_attention_frames`/`substrate_coalition_dwell_log`/
`substrate_reduction_receipts` history, and report Candidate B's actual real-data
behavior for the first time since it was built a month ago. This is the honest
prerequisite — do not start Phase 2 (habituation) until Phase 1's numbers exist, per the
same "measure before minting" discipline this whole program is built on. If Phase 1 shows
Candidate B's existing three-scorer mechanism is itself already meaningfully less
concentrated than Candidate A's argmax, that changes how much Phase 2 needs to add.
