# Objective 7 — re-evaluate integration (design-mode pass)

Status: design mode, per root `CLAUDE.md` §1 and `orion/sentience_striving_program/README.md`
§2 ("every phase below still requires explicit sign-off before implementation"). No code
changed by this document. Written by re-reading the full charter
(`orion/sentience_striving_program/README.md`, all ~1250 lines as of this pass) plus the two
most load-bearing referenced specs for items 2/3/5/6
(`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`,
`docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-observability-
design.md`), not recalled from memory.

## Arsonist summary

Objective 7 (`README.md:703`) asks whether Orion's parallel consciousness-theory instruments
should be further integrated, and gates that decision explicitly on items 4 and 5 "producing
real, comparable data — not before." Re-reading the actual recorded data against that gate,
the honest answer is: **items 4 and 5 have not yet produced what the gate asks for, so there
is nothing to decide integrate-vs-stay-separate from yet.** Three concrete gaps, not vibes:

1. **Item 5's emergent-clustering probe is stale relative to fixes that postdate it.** Its
   only real run (2026-07-21, `README.md:589-638`) already read AMBIGUOUS on its own literal
   acceptance check — one surviving stable pair (`capability:llm_inference` <->
   `node:atlas`, r≈0.92 in both windows) out of 21 checkable pairs, and NOT MET on the
   separate monoculture-differentiation check (99.98% single-target dominance). Two real
   salience-formula fixes have landed since: the 2026-07-28 `recent_perturbations` EWMA
   z-score fix (`README.md:639-652`, re-measured 2026-07-29,
   `README.md:653-667`) and the 2026-08-20 `cross_domain_variance_floor()` fix for
   `node:substrate.route` (PR #1774, `README.md:512-582`). The 2026-07-29 follow-up
   re-checked only the monoculture *percentage* against the first fix — it never re-ran the
   actual correlation-clustering measurement, and it predates the second fix entirely. The
   data item 5 is supposed to hand item 7 is real, but it is not current.
2. **Item 4 has never produced its own acceptance evidence.** Its one-line status
   (`README.md:583-586`) reads as satisfied purely by pointing at items 2 and 3's existing
   live instrumentation ("RPT/Lamme and predictive processing are already live... IIT
   continues independently... not gated by this program") — there is no dated status entry
   for item 4 itself anywhere in the document, unlike every other numbered item (2, 3, 5, 6),
   which all carry real, dated `Status:` blocks with concrete measurements. "Already covered
   by other items" may be a legitimate answer, but it has never been stated as a decision —
   it is an absence, not a checked box.
3. **O3 has never been run, for any instrument, ever.** §5's own falsifiable outcomes
   (`README.md:113-125`) name O3 as "a blind rater, given only the instrument's output on
   real historical data (not shown which theory produced it), can distinguish it from noise
   and describe what it appears to track" (`README.md:119-122`) — the literal bar this whole
   consciousness-theory-instrumentation effort exists to clear. A full-text search of the
   charter for O1-O4 shows O3 defined once and never referenced again as something measured,
   attempted, or scheduled. This is the cheapest of the four outcomes to check (read-only,
   already-collected real data, no service change) and it is the one most directly relevant
   to an integration decision — if a blind rater cannot even tell what a single instrument is
   tracking, integrating multiple instruments together is premature by construction.

Given this, the design-mode answer is not "integrate" and it is not "decide to stay separate
forever" either — both are decisions this document is not entitled to make without the data
§7's own process rule requires (`README.md:724-726`: "Integration is decided from data,
later, not from a Design Mode debate now"). The answer is: **spend one more thin,
read-only measurement patch actually producing that data, then let item 7 be re-opened with
real evidence in hand** — the same "measure before minting" discipline (`README.md:707-710`)
this program has applied to every other decision so far.

## Current architecture

- **Item 2 (AST/HOT reducer)** — Phase 1 status: **MET** on reducer-correctness scope, per
  the 2026-08-20 metric-quality-gate correction (`README.md:320-417`). `reduce_attention_
  self_model()` (`orion/substrate/attention_self_model.py`) is unified across
  `AttentionBroadcastProjectionV1` (GWT-dispatch/Lamme lane) and `FieldAttentionFrameV1` +
  `SelfStateV1`-derived active-inference confidence, replayed against 98,785-99,668 real
  ticks with real correctness cross-checks (7,330/7,330 checkable agreement) and a real
  live-fire round-trip drill (PASS). Whether a real `top_down_override` ever fires in
  production live is a **separate, still-open** question (reframed as "B", tracked under
  item 4, `README.md:342-346`) — `node:substrate.route`'s pinned `salience=1.000`/climbing
  streak was the live symptom that led to the 2026-08-20 variance-floor investigation below.
- **Item 3 (route producers onto `FieldStateV1`, retire bucket-vote)** — all five original
  active-inference domains (execution, transport, biometrics, chat, route) plus two later
  ones (bus_synaptic, codebase mass) are shadow-measured and writing real, comparably-shaped
  scalar signals via the shared `_write_prediction_error_node()` writer
  (`README.md:427-460`, `README.md:242-277`). The `goal.drive_origin` replacement is built
  and wired live (`FieldGoalProvenanceV1`, 2026-07-30, `README.md:493-511`). The
  root-cause fix for `node:substrate.route`'s structural salience-monoculture
  (`cross_domain_variance_floor()`, PR #1774, `README.md:512-582`) landed 2026-08-20, same
  day as this pass. **Important sequencing note**: §8 (`README.md:735-768`) records that
  `DriveEngine`/`tensions.py` were fully **deleted**, not just halted, on 2026-07-30 — by
  direct instruction, *ahead of* item 3's own planned validation sequencing ("retire the
  bucket-vote layer only once every producer has moved and the item-2 reducer is proven,"
  `README.md:424-426`). The deletion happened; the validation gate it was originally
  supposed to wait for did not complete first. This means item 3's "comparable data" is real
  and live, but item 3's own internal acceptance bar for "prove the reframing before
  retiring" was bypassed by events, not satisfied.
- **Item 4 (stand up read-only measurement for remaining instruments)** — text-only pointer
  to items 2/3 plus a note that IIT (mood-arc encoder, `orion/mood_arc/`) runs independently,
  not gated by this program. No dated status entry of its own.
- **Item 5 (emergent-clustering probe)** — built and run 2026-07-21
  (`scripts/analysis/measure_emergent_clustering_probe.py`, `README.md:589-638`): AMBIGUOUS
  on the baseline design's literal "not identical across windows" bar (one real, stable,
  cross-window pair found — `capability:llm_inference` <-> `node:atlas` — out of 21
  checkable pairs of a 9-target universe), NOT MET on the separate monoculture check
  (`field:recent_perturbations` won top-1 in 99.98% of 127,936 ticks). The monoculture cause
  was fixed 2026-07-28 (EWMA z-score reframing of `select_system_targets`) and spot-verified
  post-deploy 2026-07-29 (`README.md:653-667`) — but only the monoculture *percentage* was
  re-checked (11.13% recent_perturbations / 88.87% `node:athena`), not the underlying
  correlation-clustering measurement itself. `node:athena`'s 88.87% share is separately
  confirmed architectural (host node, structurally busiest by design), not an open question
  — closed in a parallel same-session doc pass (branch `docs/athena-dominance-doc-
  correction`, referenced from `README.md:663-667`'s original open-question framing).
- **Item 6 (capability_policy ↔ salience coupling)** — built and wired live 2026-07-31
  (`README.md:673-702`): `evaluate_capability()` reads the real field-native active goal
  instead of a synthetic per-call stub. This is the one piece of "integration" this program
  has already shipped — a real consumer reading a real theory-adjacent signal — and is a
  useful precedent for how thin a real integration seam can be (one field read, no fused
  schema) if item 7 ever does conclude "integrate."
- **§7 process rules directly bearing on item 7**: "measure before minting" (`README.md:707-
  710`), "multi-theory, not single-theory... Integration is decided from data, later, not
  from a Design Mode debate now" (`README.md:724-726`), "no keyword cathedrals" (`README.md:
  730-731`).
- **Existing "compare, don't fuse" precedent**:
  `scripts/analysis/measure_candidate_a_vs_b_head_to_head.py` — a real, already-built,
  read-only script that asks "do two real candidate scoring theories ever disagree on the
  same real history," reports disagreements with evidence, and explicitly does **not**
  average or fuse the two candidates together. This is the right shape to reuse for any
  future cross-instrument comparison, including the blind-rater harness this document
  recommends — an existing-mechanism check (metric-quality-gate step 5) that this exact
  problem shape has already been solved once in this codebase.
- **Constraint 1 (settled, not to re-litigate)**: `node:athena` dominating node-target
  salience/attention data is architectural — it is the host node Orion's substrate runs on,
  structurally the busiest/most-instrumented node by design — not a bug. Confirmed directly
  by Juniper, closed in the parallel `docs/athena-dominance-doc-correction` pass this same
  session.
- **Constraint 2 (a real failure mode that already hit this program once)**: PR #1774
  (2026-08-20, same day as this pass) replaced a flat global variance floor with a
  cross-domain-derived one (`cross_domain_variance_floor()`,
  `orion/attention/field_attention/candidate_precision_weighted.py:484`) after finding
  `node:substrate.route`'s genuinely-near-constant real signal was structurally guaranteed
  to win salience competition because one domain's organic variance floor was ~1,270x-
  19,300x smaller than its real competitors' — a "one domain's organic scale dwarfs
  another's" pathology. Any design that proposes weighting or fusing salience-like signals
  *across* theory-instruments inherits this exact risk, at a wider scope (theories, not just
  domains within one theory) — worth naming explicitly before anyone reaches for it.

## Missing questions

1. Has item 5's clustering probe been re-run against current live data since **both** the
   2026-07-28 z-score fix and the 2026-08-20 `cross_domain_variance_floor()` fix (PR #1774)?
   No — confirmed by reading every dated entry under §6 item 5 in the charter. This is the
   single most concrete, cheapest-to-close gap named in this document.
2. Has O3's blind-rater test ever been run, for any instrument, in this program's history?
   No — confirmed via full-text search of the charter for O1-O4 references. Is this an
   oversight, or a deliberate deferral nobody wrote down? Not resolvable from the document
   alone; worth asking Juniper directly if this design doc's recommendation isn't obviously
   the answer.
3. Is "item 4 is satisfied by pointing at items 2/3, plus IIT running independently" a real,
   considered decision, or just an artifact of item 4 never having been revisited with its
   own dated status entry the way items 2/3/5/6 all were? The charter reads as the latter —
   worth an explicit one-line decision either way, not left ambiguous.
4. Item 3's own phased plan explicitly gated bucket-vote retirement on "every producer has
   moved and the item-2 reducer is proven a real legibility replacement" (`README.md:424-
   426`), but §8 records the deletion happened by direct instruction ahead of that gate
   completing (`README.md:763-768`). Does that change what "comparable data" from item 3
   means for item 7's own gate — i.e., is the *data* item 3 produced still trustworthy as
   "comparable" even though the validation sequence that was supposed to certify it as such
   never finished as designed? Not decided here; flagged as a real gap in the evidence
   chain, not assumed away.
5. If/when items 1-2 above are answered with real numbers and Objective 7 is genuinely
   ripe, what would "integration" concretely mean here — a shared consumer reading multiple
   instruments' output, a wider `capability_policy.py` gate beyond item 6's single field
   read, or just a single legibility UI panel showing several instruments side by side
   (which would arguably not be "integration" of the *substrate* at all, just a display
   concern)? Deliberately not answered in this document — it depends on what the
   comparative data actually shows, which does not exist yet.

## Proposed schema / API changes

None. This is the substantive answer of this design pass, not an omission: no new schema,
bus channel, registry, or fused consumer is proposed. Any future integration schema (e.g., a
shared theory-instrument comparison contract, or a cross-theory legibility projection) should
wait until Missing Questions 1-3 are answered with real data — inventing that schema's shape
now, before the comparative data that would justify it exists, is exactly the "formalize
before validating" failure mode this whole program was built to stop repeating (the charter's
own re-sequencing note, `README.md:131-137`, and its still-standing "no keyword cathedrals"
rule, `README.md:730-731`).

## Files likely to touch

For the recommended next patch (measurement only, not integration):

- `scripts/analysis/measure_emergent_clustering_probe.py` — re-run against current live data;
  code changes only if the two 2026-07-28/2026-08-20 salience fixes changed the shape of data
  this script reads (check first, don't assume).
- A new, small, read-only script — e.g.
  `scripts/analysis/measure_theory_instrument_blind_legibility.py` — modeled directly on
  `measure_candidate_a_vs_b_head_to_head.py`'s existing shape: pull real historical output
  from one instrument (the AST/HOT reducer's `field_salience_only`/narrative fields are the
  natural first candidate, since it is already MET on reducer-correctness per the 2026-08-20
  entry), present samples to a rater blind to which theory/instrument produced them alongside
  real noise/baseline samples, and record real verdicts — not a self-graded summary.
- `orion/sentience_striving_program/README.md` — record whatever these two runs find as a
  new dated entry under §6 item 7 (or items 4/5, wherever the finding most precisely
  belongs), following this document's own convention.
- No service code, docker-compose, `.env_example`, bus channel, or schema registry files —
  this patch is entirely read-only analysis against already-collected real data.

## Non-goals

- Not building any fused/shared consumer combining AST/HOT, prediction-error domains, and
  emergent-clustering output in this document or its recommended next patch.
- Not expanding `capability_policy.py`'s coupling beyond what item 6 already shipped
  (2026-07-31) — that is item 6's own closed scope, not this objective's.
- Not re-litigating `node:athena`'s dominance as a bug — settled, per Constraint 1 above.
- Not proposing any cross-theory salience-fusion mechanism — the exact "one domain's organic
  scale dwarfs another's" failure class that `cross_domain_variance_floor()` (PR #1774)
  already had to fix once *within* a single theory's competition; fusing *across* theories
  multiplies that exposure, not reduces it, with no demonstrated need yet.
- Not deciding, in this document, whether the eventual answer to "integrate or stay
  separate" is yes or no — per §7's own rule (`README.md:724-726`), that is a data decision
  for later, not a design-mode debate now. This document's only claim is that the data to
  make that decision does not exist yet.
- Not re-litigating the O1-O4/O2/O3 six-drive signal-integrity series, the taxonomy
  grounding work, or any already-shipped item (2, 3, 6) — all cited, none redone here.

## Acceptance checks

- The recommended next patch is done when: (a) `measure_emergent_clustering_probe.py` has a
  real, dated re-run recorded in the charter, run after both the 2026-07-28 and 2026-08-20
  fixes, either confirming or revising the 2026-07-21 AMBIGUOUS/NOT MET findings; (b) a real
  O3 blind-rater trial has been run against at least one instrument's real historical output,
  with an actual rater's verdict recorded (not a self-assessment by whoever built the
  instrument), and the result — pass, fail, or genuinely ambiguous — written up honestly; (c)
  item 4 gets its own one-line dated status entry in the charter, either naming a real
  measurement or explicitly deciding "no dedicated check needed, here is why," so it is no
  longer the only numbered item with zero dated status history.
- Objective 7 itself is not "answered" (integrate, or deliberately stay separate) until all
  three of the above exist and are read together against O1-O4. This document's own
  recommendation is **"not yet,"** not **"no"** — and that distinction matters: "not yet"
  needs an actual next patch to become "yes" or "no" later, or it risks calcifying into
  permanent inaction the same way item 4's status already has.

## Recommended next patch

Two read-only measurement scripts, zero schema/service/consumer changes:

1. Re-run `measure_emergent_clustering_probe.py --window-hours 24 --gap-hours 12` against
   current live data, now that both the 2026-07-28 `recent_perturbations` EWMA z-score fix
   and the 2026-08-20 `cross_domain_variance_floor()` fix (PR #1774) are live — record a
   dated update under §6 item 5.
2. Build one small blind-rater harness satisfying O3's literal definition (`README.md:119-
   122`), modeled on the existing `measure_candidate_a_vs_b_head_to_head.py` "compare, don't
   fuse" precedent, and run it against the AST/HOT reducer's real narrative output — the
   cheapest instrument to test first, since it is already MET on reducer-correctness
   (2026-08-20). Record the result as a dated update under §6 item 4, giving that item, for
   the first time, real acceptance evidence of its own instead of a pointer to other items.

Only after both of those land with real numbers does Objective 7's actual
integrate-vs-stay-separate question become answerable from data — which is exactly what
item 7's own text (`README.md:703`) already says to wait for.

## Source material

- `orion/sentience_striving_program/README.md` — full charter, all sections read in full for
  this pass (§2-§9, §12-§14).
- `docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`
  — the phased Objective 2/3 roadmap that named item 2's AST/HOT reducer as a precondition
  for item 3's routing math, and Phase 6's "revisit Objective 2 with a real foundation" as
  the closest existing precedent for a data-gated re-evaluation like this one.
- `docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-observability-
  design.md` — the field-native goal-provenance producer design underlying item 3's
  2026-07-30 status entry.
- `scripts/analysis/measure_candidate_a_vs_b_head_to_head.py` — the existing "compare, don't
  fuse" pattern this document recommends reusing for the blind-rater harness.
- `orion/attention/field_attention/candidate_precision_weighted.py:484`
  (`cross_domain_variance_floor()`) — PR #1774, the same-day fix motivating Constraint 2.
