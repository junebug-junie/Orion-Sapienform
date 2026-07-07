# Computed/Learned Salience + Actionable Pending Attention — Design

- Date: 2026-07-07
- Status: Design (approved for implementation planning)
- Owner: Juniper / Orion
- Related review: choke-point review of reverie salience + rumination lock (this chat)
- Depends on: recommendation #1 (salience discrimination eval) — built as part of this spec

## Arsonist summary

Orion's attention salience is a lie. Reverie salience has been pinned at exactly
`0.72` for hours because `derive_salience` takes `max()` of an `OpenLoopV1` whose
score fields are set by a hand-tuned regex/keyword ladder in `build_open_loops`
(`orion/substrate/attention/scoring.py`), where `predictive = 0.72 if
target_type in {"plan","future_event","anomaly"}`. The same constant ladder feeds
`score_loop` → `select_actions` → the substrate coalition competition
(`orion/substrate/attention_broadcast.py`), which is what keeps re-electing the
same open loop and drives the observed rumination lock
(`loop:open-loop-2eb998452183`, resonance violations 32→69, refractory only
pausing not resolving).

This spec replaces the constant ladder with a single **evidence-derived salience**
module used everywhere a loop is scored, adds a **habituation/decay** feature so a
repeatedly-attended loop loses the coalition on its own, and gives Juniper an
**operator-grade Pending Attention surface** in the Hub to Resolve/Dismiss loops.
Each human close emits a feedback label; those labels are the seam that later
upgrades the deterministic combiner into a learned one (hybrid → learned) without
re-plumbing.

This violates nothing new and fixes three standing rule violations: no-regex-swamp,
no-keyword-cathedrals, and the §4 miscategorization of a latent judgment (salience)
hard-coded as deterministic bookkeeping.

## Decisions (locked during brainstorming)

1. **Approach: Hybrid.** Deterministic computed salience now, with a learning seam
   (feature + outcome telemetry) so a learned combiner can drop in later.
2. **Scope: one shared salience module** consumed by coalition selection AND reverie.
   Single source of salience truth; addresses the coupling.
3. **All-in-one spec:** shared salience + habituation/decay + interactive human
   close-the-loop + per-loop outcome/label events.
4. **Closer: human-only.** Juniper Resolves/Dismisses from the Hub. Orion never
   self-closes this round. Decay self-breaks rumination when Juniper is away.
5. **Labels:** `Resolve` (positive), `Dismiss` (negative), implicit
   `decayed_unattended` (weak negative).
6. **Operator-actionable rows:** plain-language context + metadata, never raw
   trace ids. Hard UX requirement.
7. **Combiner form: Approach 2** — feature vector + tiny linear combiner object
   with hand-seeded weights, refit-able later (same code path becomes learned).
8. Keep the 7 legacy `OpenLoopV1` score fields for one deprecation release.
9. Reuse the existing Hub Pending Attention panel; tag cognitive-loop rows as a
   distinct source from ops notify items.
10. **Shadow-first rollout**, flag-gated, default-off.

## Current architecture

- Signal detection is a real pipeline: `orion/substrate/attention/detectors/`
  (`current_turn`, `autonomy`, `concept_induction`, `situation`, `legacy_regex`)
  each emit `AttentionSignalV1` carrying `salience`, `confidence`, `evidence_refs`,
  `provenance`. Some carry real upstream salience (e.g. autonomy attention_items),
  some emit their own constants.
- `orion/substrate/attention/scoring.py`
  - `build_open_loops()` converts merged signals → `OpenLoopV1`, **overwriting**
    signal evidence with a regex/keyword constant ladder
    (`novelty/continuity/relational/predictive/concept/emotional/askability`).
  - `score_loop()` = fixed weighted sum of those constant fields.
- `orion/substrate/attention/policy.py` `select_actions()` ranks loops by
  `score_loop` and picks one winner.
- `orion/substrate/attention_frame.py` `build_attention_frame()` — chat-turn scope
  (consumed by `services/orion-cortex-exec/app/chat_stance.py`).
- `orion/substrate/attention_broadcast.py`
  `build_substrate_attention_frame()` + `broadcast_projection_from_frame()` —
  substrate-wide competition producing `AttentionBroadcastProjectionV1`
  (`selected_open_loop_id`, `dwell_ticks`, `coalition_stability_score`). This is
  the reverie's input and the rumination producer.
- `services/orion-thought/app/reverie.py` `derive_salience()` = `max()` of the
  selected loop's constant fields, else `coalition_stability_score`.
- Hub Pending Attention panel exists: `services/orion-hub/templates/index.html`
  (`#attentionList`, `#attentionCount`, line ~809), currently rendering
  `orion-notify` `/attention` operational items.

Score consumers of the constant ladder (three call sites, one module to replace):
`chat_stance.py` (chat turn), `attention_broadcast.py` (substrate broadcast),
`reverie.py` (`derive_salience`).

Out of scope subsystem: `orion/attention/field_attention/builder.py` (self-state
attention) is a different pipeline — not touched here.

## Missing questions (resolved)

- Learning label source: no clean per-loop label exists today (only turn-level
  `turn_outcome.surprise_resolved`). Resolved by making human Resolve/Dismiss the
  label source; sparse but clean, accumulates for later fitting.
- Where habituation lives: in the shared salience feature vector (graded), with
  refractory kept as a coarse backstop.

## Proposed architecture

### A. Shared salience module — `orion/substrate/attention/salience.py`

Pure, import-light (no heavy `requests`/graph deps) so `orion-thought` may import it,
matching the existing thin-import discipline in reverie/store.

- `SalienceFeaturesV1` (pydantic, registered) — evidence feature vector:
  - `evidence_strength: float` — max(`signal.salience * signal.confidence`) over
    the signals backing the loop.
  - `evidence_breadth: float` — normalized count of distinct detectors /
    `evidence_refs` backing the loop.
  - `recurrence: float` — normalized count of recent frames/chains where this
    loop id / theme key appeared.
  - `recency: float` — freshness decay since first observation.
  - `novelty_vs_known: float` — inverse of `already_known` / `known_blob` overlap.
  - `dwell: float` — normalized `dwell_ticks` from the broadcast.
  - `habituation: float` — penalty term from repeated recent attendance +
    resonance-alert history for the theme (higher = more habituated).
- `LinearSalienceCombiner`:
  - `weights: dict[str, float]` + `weights_version: str`.
  - `score(features) -> float` = bounded weighted sum with `habituation` applied
    as a subtractive/penalty term.
  - Hand-seeded defaults (documented rationale per weight). No training now.
- `compute_salience(*, loop, signals, history, now) -> tuple[float, SalienceFeaturesV1]`.
- `default_combiner()` factory reading optional weights override from config.

### B. Wire the combiner into the three consumers

- `scoring.py`:
  - `build_open_loops()` stops inventing target-type constants; instead computes
    `SalienceFeaturesV1` per loop and stores it in `OpenLoopV1.salience_features`.
    Legacy 7 fields retained one release, populated from evidence (best-effort),
    marked deprecated in docstring/schema comment.
  - `score_loop(loop)` → `default_combiner().score(loop.salience_features)`.
- `attention_broadcast.py`: unchanged selection *mechanism* (still `select_actions`
  with `max_asks=0`), but now selecting on real salience; habituation flows through
  `score_loop`, so a high-dwell loop is demoted over ticks.
- `reverie.py` `derive_salience(broadcast)` → build features for the selected loop
  and call the same combiner (delete the `max()` logic). Same salience for the same
  coalition still holds (deterministic), but now it discriminates.

`OpenLoopV1` schema change: add `salience: float = 0.0` and
`salience_features: dict = {}`. Update `orion/schemas/registry.py`.

### C. Habituation / breaking the rumination lock

`habituation` rises with `dwell_ticks`, recent re-selection count of the theme
(`substrate_reverie_chain` theme events / a lightweight attention-visit ledger),
and resonance-alert presence for the theme. As it rises, combiner salience for the
stuck loop falls until a competitor wins `select_actions`. Inhibition-of-return.
Refractory (`substrate_reverie_refractory`) stays as a coarse backstop.
Flag: `ORION_ATTENTION_HABITUATION_ENABLED`.

### D. Operator-actionable Pending Attention

- `PendingAttentionCardV1` (registered) — human-readable card per surfaced loop:
  - `loop_id`, `theme_key`
  - `title` — plain phrase (never a bare id)
  - `why_it_matters` — plain language
  - `what_triggered` — evidence in words (detectors/signals + count)
  - `age_seconds`, `recurrence_count`, `salience`, `weights_version`
  - `top_contributing_features: list[str]` — features rendered in words
  - `narrative` — reuse reverie `interpretation` for the theme when present;
    deterministic template fallback otherwise. Never id-only.
  - `status` — `pending` / `resolved` / `dismissed`
- Surfacing policy: only loops worth a human's time (resonance-flagged OR
  high-salience-unresolved beyond an age threshold), to keep the panel quiet.
- Hub render: existing Pending Attention panel; cognitive-loop rows tagged with a
  distinct source badge vs ops notify items. Resolve / Dismiss buttons per row.
- Privacy: cards carry only plain summaries; no raw private trace material,
  journal text, or blocked material. Summaries preserve privacy boundaries.
- Flag: `ORION_ATTENTION_PENDING_CARDS_ENABLED`.

### E. Learning seam (hybrid → learned later)

Event-substrate path (`event → schema → trace → reducer → projection → eval → UI`):

- `AttentionSalienceTraceV1` on `orion:attention:salience:trace` — every scored
  loop logs feature vector + score + `weights_version` → reducer →
  `attention_salience_trace` table.
- `AttentionLoopOutcomeV1` on `orion:attention:loop_outcome` — verdict
  `resolved` / `dismissed` / `decayed_unattended`, actor, `salience_at_close`,
  feature snapshot → reducer → `attention_loop_outcome` table (the labels).
- Closing a loop also writes resolved/suppressed state so it exits the coalition
  and won't re-ignite (resolves, not just pauses).
- `scripts/refit_salience_weights.py` — documented, **not run now**. Joins traces +
  labels, emits candidate weights + `weights_version`. Ships as a stub with a unit
  test proving it consumes the label table; production weights stay seeded.

## Data flow

```text
detectors → AttentionSignalV1
  → build_open_loops (compute SalienceFeaturesV1, no constants)
  → score_loop = LinearSalienceCombiner.score(features)   [habituation applied]
  → select_actions → coalition winner
      ├─ chat: chat_stance.py
      ├─ broadcast: attention_broadcast.py → AttentionBroadcastProjectionV1
      │     → reverie derive_salience (same combiner)
      └─ surfacing: PendingAttentionCardV1 → Hub Pending Attention panel
  → telemetry: AttentionSalienceTraceV1
  → human Resolve/Dismiss → AttentionLoopOutcomeV1 (label) + suppress loop
  → reducers → attention_salience_trace / attention_loop_outcome tables
  → (later) refit_salience_weights.py → new weights_version
```

## Schema / bus / API changes

- Added schemas: `SalienceFeaturesV1`, `PendingAttentionCardV1`,
  `AttentionLoopOutcomeV1`, `AttentionSalienceTraceV1`.
- Changed schema: `OpenLoopV1` + `salience`, `salience_features` (additive,
  back-compatible; legacy 7 fields deprecated for one release).
- Registry: register the four new schemas (`orion/schemas/registry.py`).
- Channels (`orion/bus/channels.yaml`): `orion:attention:salience:trace`,
  `orion:attention:loop_outcome`. Pending cards exposed via Hub API projection
  (no new consumed bus channel required for rendering; publish trace/outcome only).
- Hub API (`services/orion-hub`): `GET /api/attention/loops`,
  `POST /api/attention/loops/{id}/resolve`, `POST /api/attention/loops/{id}/dismiss`.
- Tables: `attention_salience_trace`, `attention_loop_outcome` (+ loop
  resolved/suppressed state; may reuse `substrate_reverie_refractory` for suppress).

## Env / config changes

- `ORION_ATTENTION_SALIENCE_V2_ENABLED` (default false) — shadow vs live switch.
- `ORION_ATTENTION_HABITUATION_ENABLED` (default false).
- `ORION_ATTENTION_PENDING_CARDS_ENABLED` (default false).
- Optional `ORION_ATTENTION_SALIENCE_WEIGHTS` (JSON override; empty = seeded).
- Update every touched service `.env_example` and run
  `python scripts/sync_local_env_from_example.py`.

## Rollout (shadow-first)

1. Land module + schemas + telemetry with `SALIENCE_V2` off: compute new salience
   in shadow, log `AttentionSalienceTraceV1` for both old and new, do not change
   selection.
2. Run the #1 discrimination eval + inspect shadow traces for parity/discrimination.
3. Flip `SALIENCE_V2` on for selection; enable habituation; watch the rumination
   replay + live resonance counts drop.
4. Enable pending cards + closure.

Rollback = flip flags off → old constant path. Each stage independently reversible.

## Non-goals

- Training/fitting the model now (weights stay seeded; refit documented, not run).
- Orion autonomous closure (human-only this round).
- Approach-3 embedding/similarity salience.
- Rewriting the detectors' internal constants (separate cleanup).
- `orion/attention/field_attention/` self-state pipeline.

## Tests & evals

Gate tests (fast, deterministic):
- `test_salience_discrimination_eval` (**recommendation #1**, built here first):
  distinct coalitions → distinct salience; fails on today's constants.
- Combiner unit tests: per-feature monotonicity; habituation strictly lowers
  salience across repeated visits; bounded [0,1]; deterministic for same input.
- Rumination replay test: same high-pressure node over N ticks → a competitor wins
  once habituation engages (lock breaks). Uses injected clock/history.
- Card legibility test: a `PendingAttentionCardV1` never renders an id-only string;
  `why_it_matters` and `title` always plain text.
- Closure e2e test: resolve/dismiss emits `AttentionLoopOutcomeV1` + suppresses
  the loop (won't re-select).
- Learning-seam test: outcomes land in `attention_loop_outcome`; refit stub reads
  them and returns candidate weights.
- Contract tests: new channels registered; schemas in registry; producer/consumer
  round-trip for trace + outcome.

Evals:
- Salience discrimination eval (above) as the acceptance gate for the flip.
- Reverie quality unaffected/improved (existing reverie evals still pass).

## Acceptance checks

- #1 discrimination eval passes.
- Rumination replay proves the lock breaks with habituation on.
- Shadow parity/discrimination reviewed before the live flip.
- Hub shows legible, actionable rows; Resolve/Dismiss works end-to-end and writes
  labels to `attention_loop_outcome`.
- No raw private trace material in cards.
- All flags default-off; old path intact when off.

## Risks / concerns

- Severity medium: changing coalition selection is system-wide (chat + broadcast).
  Mitigation: shadow-first + independent flags + replay test.
- Severity medium: seeded weights are still human-chosen; "computed" not "learned"
  until labels accumulate. Mitigation: explicit hybrid framing; telemetry from day
  one; refit path documented.
- Severity low: sparse human labels. Mitigation: implicit `decayed_unattended`
  weak-negative + traces give feature distributions even without labels.
- Severity low: surfacing noise. Mitigation: strict surfacing policy (resonance /
  high-salience-unresolved + age threshold).

## Recommended next patch

Phase 1 (contract + module, shadow): `salience.py` + `SalienceFeaturesV1` +
`OpenLoopV1` fields + registry + the #1 discrimination eval + combiner unit tests,
all behind `SALIENCE_V2` off. Smallest slice that creates the seam and proves the
stub with a failing-then-passing eval.
