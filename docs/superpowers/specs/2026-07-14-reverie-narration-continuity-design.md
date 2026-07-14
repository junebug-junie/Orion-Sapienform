# Reverie narration continuity — design spec

Status: DESIGN, not implemented. Ground truth verified live 2026-07-14 against
`main` (post `37e97b92`, the open-loop recency fix).

Scope decided by Juniper: build #1, #2, #3, #6, #7 below. Pin #4 (cross-chain
theme memory) and #5 (deterministic narrative-stability trace) for later.
Confirmed: reverie is live in prod. Confirmed: the intended shape is **a
conscious stream when chained, episodic when standalone** — i.e. continuity is
scoped to the chain, not injected into every lone tick.

## Arsonist summary

Reverie's per-tick discipline (grounding, `is_hollow()`, deterministic
salience) is solid and unchanged by this spec. The actual gap: every tick,
chained or not, rebuilds its prompt from nothing but the live broadcast — no
tick knows what a prior tick said, whether the thing it's narrating was
already resolved by a human, or how strong the evidence for its own claim
actually is. This patch closes those five gaps without touching the
deterministic substrate selection path (`attention_broadcast.py`,
`scoring.py`) and without letting the LLM steer its own future attention —
both of which stay exactly as they are today.

## Current architecture

- `services/orion-thought/app/reverie.py:322-463` (`run_reverie_once`) — one
  tick: read live broadcast → `build_reverie_context()` → LLM narrates →
  `parse_reverie_payload()` stamps hollow/salience deterministically →
  publish + `persist_reverie_thought()`.
- `services/orion-thought/app/chain.py:186-300` (`run_reverie_chain`) — runs
  up to `max_steps` (default 4, `ORION_REVERIE_CHAIN_MAX_STEPS`) calls to
  `run_reverie_once` **in one process invocation**, each stamped with
  `chain_context=(chain_id, index)`. `chain_context` only labels the
  *output* — it is never read back into the *input* of the next step.
  `ema_summary` (`chain.py:260`) is a fixed bookkeeping string, not narrative
  content.
- `services/orion-thought/app/store.py` — persistence is write-heavy,
  read-thin. The only reads that exist today are theme/timestamp pairs for
  the resonance detector (`load_recent_chain_theme_events`,
  `load_recent_resonance_alerts`) — no function anywhere reads a prior
  thought's `interpretation` text back.
- `orion/schemas/reverie.py:72-73` — `SpontaneousThoughtV1.next_focus` /
  `.drift` exist on the schema, are never requested by the prompt
  (`reverie_narrate.j2` only requires `interpretation` + `evidence_refs`),
  and are never read anywhere else. `settings.py:85`
  (`ORION_REVERIE_DRIFT_TEMP`) is a matching dead config knob — declared,
  never read.
- `attention_loop_outcome` (verdicts: resolved/dismissed/etc, written by
  `services/orion-hub/scripts/attention_loops_store.py`) is consumed today
  only by `scripts/refit_salience_weights.py` (offline) and Hub display
  routes — never by live selection, never by reverie.
- `OpenLoopV1` (`orion/schemas/attention_frame.py:61-89`) already carries
  `salience_features: dict`, `salience: float`, `dwell_ticks` (on the
  broadcast) — real quantitative signal the prompt template currently
  either omits or passes raw for the LLM to eyeball.
- `services/orion-hub/scripts/substrate_observability_routes.py:112-140`
  (`_reverie_section`) already loads the full `thought_json` per row but
  explicitly whitelists only `salience`, `interpretation`,
  `attended_node_ids`, `selected_open_loop_id` — `next_focus`/`drift` are
  fetched and silently dropped.

## Missing questions

Resolved by Juniper this session:
- Stream vs episodic → **both**: stream inside a chain, episodic for
  standalone ticks. Standalone (`chain_context=None`) ticks get none of the
  new continuity behavior — this is load-bearing, see Acceptance checks.
- Live in prod → yes. Ship #1 (correctness fix) independently of the rest;
  don't gate it behind the larger continuity patch.

Still open, low-stakes enough not to block a first patch, flagged for
implementation-time judgment:
- Does `_reverie_section` need `next_focus`/`drift` surfaced for this patch
  to satisfy §0A's "producer needs a consumer" bar, or is the eval/test
  suite itself sufficient consumer? Recommend doing the one-line Hub addition
  anyway — it's cheap and it's real UI/debug surface, not busywork.
- `ORION_REVERIE_DRIFT_TEMP`: keep as a still-dead knob, wire it to actually
  vary generation temperature per chain step, or delete it? Recommend
  **delete** — it's orthogonal to narration continuity and reintroducing
  temperature scheduling later is its own thin patch if ever wanted.

## Proposed schema / API changes

No breaking changes to `SpontaneousThoughtV1` — `next_focus`/`drift` already
exist; this patch makes them real by giving them a producer (prompt asks for
them, chain-mode only) and a consumer (next chain step's prompt + Hub panel).

New functions:

```python
# services/orion-thought/app/store.py
def load_recent_loop_outcomes(loop_ids: list[str]) -> dict[str, dict]:
    """Most recent attention_loop_outcome verdict per loop_id. Best-effort,
    never raises, returns {} on any miss — mirrors every other reader in this
    module. Direct DSN query (no orion.substrate import), same pattern as
    persist_reverie_thought."""
```

New pure functions (deterministic, unit-testable, no I/O — §4 split):

```python
# services/orion-thought/app/reverie.py (or orion/reverie/framing.py if it
# grows; start colocated, split out only if it earns it)
def salience_margin(selected: float, runner_up: float | None) -> float: ...
def dwell_framing(dwell_ticks: int, stability_score: float) -> str: ...
```

`build_reverie_context()` signature grows additive-only kwargs:

```python
def build_reverie_context(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    concern_cards: list[ConcernCardV1] | None = None,
    loop_outcomes: dict[str, dict] | None = None,      # #1
    prior_thoughts: list[str] | None = None,            # #2, chain-mode only
    next_focus_hint: str | None = None,                 # #3, chain-mode only
) -> dict[str, Any]:
```

All four new kwargs default to `None`/absent — a standalone tick
(`chain_context=None`) never populates `prior_thoughts` or
`next_focus_hint`, and `loop_outcomes` is populated for *every* tick
(chained or not — verdict-awareness is a correctness fix, not a continuity
feature, so it applies everywhere).

Prompt template additions (`orion/cognition/prompts/reverie_narrate.j2`,
non-`concern_cards` branch only — `#1/#6/#7` also apply to the semantic-lift
branch for outcomes, see note below):

- Per-loop outcome line when `loop_outcomes` has an entry for that id:
  `"- id: {{ loop.id }} | ALREADY VERDICTED: {{ loop.outcome.verdict }}
  ({{ loop.outcome.note }}, {{ loop.outcome.age_days }}d ago) — narrate as
  settled, not as an open struggle."`
- New `CONTINUITY` block, only rendered when `prior_thoughts` is non-empty:
  the last 2 verbatim interpretations, framed as "what you already said —
  do not repeat; extend, narrow, or reconsider."
- New line when `next_focus_hint` is set: `"Last step you said you wanted to
  focus next on: {{ next_focus_hint }}."` — framed as a hint, not a
  directive; the coalition data below is still the only thing you may
  narrate.
- `salience_margin` and `dwell_framing` rendered as plain pre-computed
  sentences in the `SOURCE` section (not raw numbers) — e.g. `"This loop
  edged out the next-best candidate by a narrow margin (0.04) — hedge your
  certainty accordingly."` / `"This has been the winning focus for 6
  consecutive ticks."`
- `REQUIRED JSON FIELDS`, chain-mode only (i.e. `chain_context is not None`
  and `index < max_steps - 1`): add optional `next_focus` (1 sentence, what
  to turn to next) and `drift` (1 phrase, how this step's focus differs from
  the last). Not required on the final step of a chain or on standalone
  ticks — there is no "next" to point to.

Note on `concern_cards` branch: `#1` (outcomes) and `#6`/`#7` (margin/dwell
framing) are orthogonal to whether semantic lift is active and should apply
there too if `open_loops`/`coalition_projection` data is available in that
branch's context. `#2`/`#3` (chain continuity) stay reverie-branch-only for
this patch — concern-card chains are not in scope; revisit if that combo
ever ships.

`run_reverie_once()` grows two optional kwargs threaded straight into
`build_reverie_context`: `prior_thoughts` and `next_focus_hint`. Nothing
else in its signature changes.

`chain.py::run_reverie_chain()` — no new persisted state, no new schema.
The loop already accumulates `thought_ids`/`ema` per step; add two
in-memory accumulators scoped to the single `run_reverie_chain` call:

```python
prior_texts: list[str] = []
last_next_focus: str | None = None

for index in range(max(1, max_steps)):
    thought = await step_fn(chain_id, index, prior_texts[-2:], last_next_focus)
    ...
    if thought is not None:
        prior_texts.append(thought.interpretation)
        last_next_focus = thought.next_focus
```

`StepFn` type widens to `Callable[[str, int, list[str], str | None],
Awaitable[SpontaneousThoughtV1 | None]]`; `chain.py::_step` passes those
straight through to `run_reverie_once`. **In-memory only** — no DB read
needed for chain-local continuity, since a chain fully completes inside one
`run_reverie_chain` call today. (This is the thing that makes #2 cheaper
than originally scoped in the brainstorm — no `load_chain_thoughts` store
function needed.)

`services/orion-hub/scripts/substrate_observability_routes.py:130-138`
(`_reverie_section`) — add `next_focus`/`drift` to the per-row dict already
being built from `payload` (both already present in `thought_json`, this is
a two-line addition, no new query).

## Files likely to touch

- `services/orion-thought/app/reverie.py` — `build_reverie_context`,
  `_open_loops_for_prompt`, `run_reverie_once`, new `salience_margin`/
  `dwell_framing` pure functions.
- `services/orion-thought/app/chain.py` — `run_reverie_chain` (in-memory
  accumulators), `_step`, `StepFn` type.
- `services/orion-thought/app/store.py` — new `load_recent_loop_outcomes`.
- `orion/cognition/prompts/reverie_narrate.j2` — outcome lines, CONTINUITY
  block, next_focus hint line, margin/dwell framing sentences, conditional
  `next_focus`/`drift` in REQUIRED JSON FIELDS.
- `orion/schemas/reverie.py` — doc-comment update only (next_focus/drift
  are no longer aspirational).
- `services/orion-thought/app/settings.py` — remove
  `ORION_REVERIE_DRIFT_TEMP` (dead knob; see Missing questions).
- `services/orion-hub/scripts/substrate_observability_routes.py` —
  surface `next_focus`/`drift` in `_reverie_section`.
- Tests: `services/orion-thought/tests/test_reverie_spontaneous_thought.py`,
  `test_reverie_chain.py`, `services/orion-thought/evals/
  test_reverie_hollow_guard_eval.py`, `orion/reverie/evals/
  test_reverie_semantic_quality_eval.py` — new cases per Acceptance checks
  below. New unit test file for `salience_margin`/`dwell_framing` (pure,
  no fixtures needed).
- `.env_example` — remove `ORION_REVERIE_DRIFT_TEMP` if deleted; no new env
  keys (outcome lookup uses the existing `POSTGRES_URI` pattern already in
  `store.py`).

## Non-goals

- No change to `attention_broadcast.py` / `scoring.py` / live coalition
  selection. Verdict-awareness (#1) is narration-only — a human-closed loop
  can still win selection; it just won't be narrated as urgent. Suppressing
  it from selection entirely is a separate, bigger substrate-contract patch,
  not in scope here.
- No cross-chain or cross-tick memory (#4, pinned). A new chain on a
  recurring theme has no idea it's happened before. Left for later.
- No deterministic narrative-stability/repetition trace (#5, pinned). #2's
  anti-repetition guarantee is prompt-instruction only in this patch, not
  measured. This is the biggest accepted risk in this spec — see Tensions.
- No LLM-driven override of *which* loop/coalition gets selected. The hard
  version of #3 (next_focus steering selection) needs §0A proposal mode and
  is explicitly out of scope.
- No change to standalone (non-chain) tick behavior beyond #1 and #6/#7 —
  episodic ticks stay memoryless of prior ticks, by design (confirmed).
- No concern-card-branch chain continuity (see note above).

## Acceptance checks

1. Given a broadcast whose selected loop has a persisted
   `attention_loop_outcome` row, `build_reverie_context()`'s rendered prompt
   contains that verdict text; a thought generated against a mocked
   "resolved" verdict does not describe the loop as an open struggle in a
   golden-prompt test.
2. Given a 2+ step chain, step index 1's context contains step 0's
   `interpretation` text verbatim (capped at last 2); step 0's context
   contains no `CONTINUITY` block (nothing prior).
3. Given step 0 emits `next_focus="X"`, step 1's prompt contains the literal
   hint line referencing `X`.
4. `salience_margin`/`dwell_framing` are pure functions with unit tests
   covering: near-tie margin, landslide margin, `runner_up=None` (single
   loop), zero dwell, high dwell.
5. **Regression guarantee**: standalone `run_reverie_once(bus)` calls
   (`chain_context=None`, no `prior_thoughts`/`next_focus_hint` passed)
   produce **byte-identical** `build_reverie_context()` output to today,
   except for the new `loop_outcomes` framing (#1 applies everywhere) and
   `salience_margin`/`dwell_framing` (#6/#7 apply everywhere). No
   `CONTINUITY` block, no `next_focus` hint line, ever, outside a chain.
6. `is_hollow()` / grounding discipline unchanged: a thought whose only
   "evidence" is continuity text or a next_focus hint (not a real coalition
   id) is still correctly marked hollow — continuity content must never
   become a grounding backdoor. New explicit regression test for this.
7. Chain-mode `REQUIRED JSON FIELDS` correctly omit `next_focus`/`drift` on
   the final step of a chain (nothing to point to).
8. Full existing suites green:
   `pytest services/orion-thought/tests -q`,
   `pytest services/orion-thought/evals -q`,
   `pytest orion/reverie/tests orion/reverie/evals -q`.

## Recommended next patch

Two changesets, not one — they don't share a dependency and #1 is the
correctness fix that's been live-broken the longest:

1. **`fix/reverie-verdict-aware-narration`** — #1 alone. Smallest possible
   diff: `load_recent_loop_outcomes`, wire into `build_reverie_context` +
   `_open_loops_for_prompt`, one prompt block. Ships independent of
   everything else below.
2. **`feat/reverie-chain-continuity`** — #2 + #3 + #6 + #7 together, since
   they land in the same call sites (`build_reverie_context`, `chain.py`'s
   step loop, the prompt template) and are easier to review and eval as one
   coherent "chain narration" changeset than three separate diffs touching
   the same three files. Ship after #1 is merged and verified live (so a
   confabulation regression in the bigger patch isn't confused with the
   verdict-awareness fix).

Both changesets are thin per §0A: no new service, no new abstraction layer,
additive-only kwargs, in-memory (not DB-backed) chain continuity, and every
new schema field this patch touches (`next_focus`/`drift`) gets both a real
producer and a real consumer in the same changeset.
