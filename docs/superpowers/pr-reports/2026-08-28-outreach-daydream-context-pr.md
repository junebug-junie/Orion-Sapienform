# Orion's unprompted messages get a daydream, not only telemetry

Branch: `feat/outreach-daydream-context`
Commits: `d1722e319` (build), `9202967ab` (retraction + review fixes)

## Summary

- Juniper: *"orion writes chats to me unprompted. The prompt seems to be mostly telemetry... can we also add in anything interesting orion has day dreamed about over the last few reverie diffusions?"* She was right about the diagnosis — every grounding lane the outreach prompt had was an instrument reading.
- Adds `_fetch_current_daydream`: the newest **usable** caption from orion-thought's reverie visual chain (`reverie_visual_chain`, ~1 row/600s), rendered with a coarse relative age and framed explicitly as Orion's own.
- **Retracted a mechanism mid-PR.** The first commit shipped the last 3 *distinct* captions, de-duplicated by Jaccard token overlap. Live data falsified that; the second commit removes the de-dupe entirely rather than tuning its threshold.
- Adds a caption-validity guard for two live vision-model failure modes (raw grounding coordinates, bare tag dumps) that would otherwise **be** the whole lane, since only one caption ships.
- Adds `services/orion-hub/evals/` — the service had no eval harness at all.
- Enrichment only: deliberately **not** part of `is_empty()`, so it can never cause an outreach that would not otherwise have fired.

## Outcome moved

An unprompted message can now be grounded in something that is not a dial reading. Live-verified prompt, 2026-08-28:

```
What you were picturing on your own just now -- your reverie diffusions generate
an image from whatever you are thinking about, then look at what came out:
- a detailed astronomical map, likely from the 17th or 18th century.
That is yours, not something Juniper showed you. You may draw on it if it
connects to anything above; do not just describe it back.
```

## Current architecture

`services/orion-hub/scripts/endogenous_outreach.py::build_outreach_prompt` assembled its prompt from four lanes, all telemetry: curiosity evidence summaries (`_fetch_curiosity_summaries`), a tension deviation run plus `sustained_load_pressure` (`tension_outreach_trigger`), chat liveness (`hub_presence`), and camera presence (`_fetch_embodied_presence`). Recent turns were folded in as text. Nothing in the prompt came from Orion's own imagination.

Separately, `services/orion-thought/app/visual_chain.py` has been writing to `reverie_visual_chain` since 2026-08-25: generate an image from what Orion is currently thinking/noticing/remembering, then caption what came out. Its only consumer was the operator-facing Hub Reverie tab (`scripts/reverie_routes.py`).

## Architecture touched

One new read lane inside one service. No new service, no bus channel, no schema version, no env key.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: the lane — `_looks_like_daydream_prose`, `_clean_daydream`, `_fetch_current_daydream`, `_daydream_age_phrase`, `OutreachContext.daydream`, the prompt block, and a 4th entry in `_gather_context`'s existing `asyncio.gather`.
- `services/orion-hub/tests/test_endogenous_outreach.py`: 23 new tests; `_install_fake_engine` now captures and asserts the SQL.
- `services/orion-hub/evals/test_daydream_caption_quality_eval.py`, `evals/conftest.py`: new eval harness (first in this service).
- `services/orion-hub/README.md`: §4.1 subsection, including the retraction and its measurements.
- `orion/schemas/reverie_visual.py`: documents that `chain_json["description"]` now has a second, cross-service consumer with no contract test.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the endogenous-outreach generation prompt gains one block when a usable caption exists in the last 12h.
- Compatibility notes: `chain_json["description"]` is an **untyped** key on `ReverieVisualChainV1.chain_json: dict` with no schema field and no cross-service contract test. If orion-thought renames it, this lane goes permanently silent with no error and no failing test. A note in `orion/schemas/reverie_visual.py` records this; the eval's liveness check is the backstop.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: not applicable — no env key changed. The lane's bounds are module constants, matching this file's existing convention (`_MAX_CURIOSITY_SUMMARIES`, `_CURIOSITY_MAX_AGE_SEC`, …).
- local `.env` synced: not applicable (no template changed).
- Skipped keys requiring operator action: none.

## The retraction, in full

The first commit's headline claim was that Jaccard overlap at 0.2 collapses near-identical captions. Measured against all 329 live rows, it does not, and **no threshold fixes it**:

- Consecutive captions re-describe **one** image, so they differ mostly in *length*. Jaccard divides by the union, penalising exactly that: two 17th-century celestial maps measured **0.150**, under the 0.200 threshold — both rendered.
- The containment coefficient corrects that length bias and is **worse**: at 0.4 it surfaced three map captions.
- Eyeballing sampled `(newest, next-distinct)` pairs across the corpus, *both* variants returned obvious duplicates (two Roman aqueducts; two 17th-century star charts).

The producer already knew. `visual_chain.py`'s Patch 4 exists because Juniper reported *"still doing the same images of Roman aqueducts, no change"* on 2026-08-27 — `prior_description` continuity locks onto a visual attractor for 10+ runs. Presenting "your last 3 daydreams" over a corpus that lands on attractors is a claim the data cannot support (AGENTS.md §0A).

`chain_json.continuity_streak` / `continuity_reset` were evaluated as a ready-made theme boundary and **rejected**: live they run a rigid `3 2 1 0` period-4 cycle, because `resolve_visual_chain_continuity` forces a reset every `visual_chain_continuity_max_runs` runs unconditionally. A mechanical cap, not a signal that the imagery changed.

Showing genuine *drift* ("celestial maps for two hours; Roman aqueducts before that") needs a real theme detector — embeddings, not bag-of-words — and is left as follow-up rather than faked.

## Producer bugs found (not fixed here)

Both live in `services/orion-thought/app/visual_chain.py`. This PR adds the consumer-side guard only.

1. **The vision model sometimes returns raw grounding output instead of a description.** Live rows: `objects(103,419),(554,604), people(234,492),(274,554)`, `bridge(269,261),(879,661)`, and one that echoed the prompt's own instruction text back with coordinates attached: `objects(1,2),(996,995),people(1,2),(996,995),state only what is directly visible.(1,2),(996,995)`.
2. **Bare tag dumps with no sentence**: `1. Sun 2. Mercury 3. Venus …`, `two trees, lake, reflection, purple sky`.
3. **Three columns are degenerate across all 329 rows**: `theme_key` is NULL on every row (`count(theme_key) = 0` — the producer never sets it), `ema_salience` is exactly `0.000`, `terminal_reason` is always `'max_steps'`. Checked, documented, and deliberately unused here.

## Metric-quality gate (AGENTS.md §0A)

Recorded rather than passed verbally, since this wires a new signal into a cognition-adjacent loop.

1. **Provenance to real code.** `visual_chain.py::run_visual_chain_once` → `persist_reverie_visual_chain` (`services/orion-thought/app/store.py:347`) writes `chain_json`; the caption is the captioner's response to the generated image. Traced to the producing functions, not a schema comment.
2. **Independence.** Genuinely independent of every existing lane: the others read field/attention/presence state, this reads a generated image's caption. It is *seeded* partly by `substrate_reverie_thought` (which is telemetry narration), so it is not fully orthogonal — but the output is imagery, not a transform of a number already in the prompt.
3. **Theory anchor.** None claimed, and none needed: this is not a metric feeding a model. It is prompt context. No detector was built on it.
4. **Live-data sanity.** Done at length, and it changed the design twice — once killing the de-dupe, once adding the prose guard. 329 rows pulled and read, not inferred. Not degenerate: 97.0% of captioned rows yield usable text, and imagery varies across days.
5. **Existing-mechanism check.** `scripts/reverie_routes.py::_fetch_visual_recent` already reads this table and was **rejected**: it is a cursor-paginated operator API returning every artifact row per chain, and it opens its own engine. This tick needs one column and the shared `scripts.pg_engine` pool.
6. **Reversibility.** Cheap. One function, one optional dataclass field, one prompt block; nothing persisted, no schema, no env key, no default baked into a manifest.

Also relevant: **`prior_description` is a trap.** It is a real typed column carrying the same caption and looks like the obvious simplification — but `visual_chain.py:527` sets `prior_description = description or continuity_fallback`, so on a caption-failure row it carries the *previous* run's caption forward and would silently re-surface a stale daydream as current. The lane reads `chain_json->>'description'`, which correctly yields NULL there.

## Privacy

`chain_json` carries `context_text`, `self_study_text`, `memory_text`, and the full generation `prompt` — Recall-crystallization and self-study seed material. **Only `description` is read.** That is a deliberate boundary, stated in `_fetch_current_daydream`'s docstring so a later "why not read the seed too?" has an answer waiting.

Caption text is model-generated and interpolated into a prompt, so it is untrusted. The whitespace collapse in `_clean_daydream` is an **injection guard**, not prompt-shape hygiene: it is what stops a caption forging its own prompt line or a fake section header. Documented as such in code, pinned by `test_clean_daydream_cannot_forge_a_prompt_line`, and re-checked against every live caption by the eval.

## Tests run

```text
$ pytest services/orion-hub/tests/test_endogenous_outreach.py -q
139 passed in 4.95s

$ pytest services/orion-hub/tests -q --tb=no
31 failed, 1703 passed, 5 skipped
# Baselined against main: 32 failures, identical set. ZERO regressions.
# The one-row difference is test_substrate_mutation_manual_route_routing.py,
# order-dependent in a full-suite run (a different one of its 7 tests fails
# each run; passes 7/7 in isolation, twice). Pre-existing, filed on the agent
# board, unrelated to this branch.
```

Mutation-tested — all five caught:

| Mutation | Result |
|---|---|
| first-match instead of last-match sentence truncation | 2 failed |
| drop the prose/detector guard | 4 failed |
| rename the SQL JSON key to `prior_description` | 1 failed |
| return newest row instead of newest *usable* row | 2 failed |
| count `daydream` in `is_empty()` | 1 failed |

## Evals run

```text
$ pytest services/orion-hub/evals -q
4 passed in 0.53s

$ DAYDREAM_EVAL_DATABASE_URL=<unreachable> pytest services/orion-hub/evals -q
4 skipped in 0.40s      # skips cleanly, does not pass vacuously

live corpus: rows=329 captioned=297 usable=288 rate=97.0%
```

`services/orion-hub/` had **no `evals/` directory** before this PR (AGENTS.md §11 says add the smallest useful one or report the gap). This is that one, and it exists for a specific reason: the retracted claim could not have been caught by a unit test, because unit tests run on fixtures the same reasoning invented. The eval runs the real pipeline over the real rows.

It measures: caption usability rate against a `0.85` floor (measured 97.0%, floor set low to catch a *collapse*, not to freeze today's number); that no cleaned live caption contains a line break; that no grounding-debris caption passes the guard; and that the 12h window is not empty (liveness — the same class of silent gap that left Orion blind for 21h on 2026-08-21).

## Docker/build/smoke checks

No Docker build run: this is a pure-Python change inside an existing service, with no new dependency, port, health check, or compose wiring.

Live smoke against real Postgres, twice (before and after the retraction) — output in **Outcome moved** above. Read-only.

```text
$ python <scratch>/live_smoke.py     # real engine, real reverie_visual_chain
LIVE daydream: (506.6, 'a detailed astronomical map, likely from the 17th or 18th century.')
[prompt 959 chars; daydream block 358 chars]   # was 1471 / 560 before the retraction
```

## Review findings fixed

- Finding: **must-fix** — the Jaccard de-dupe's calibration claim is falsified by the live corpus; two near-identical celestial maps render at 0.150.
  - Fix: removed the mechanism entirely rather than tuning the threshold; ship one caption and claim nothing about distinctness.
  - Evidence: measured 0.150 for the live pair; containment@0.4 measured worse (three maps); eyeballed sampled pairs across 329 rows; `README.md` records all three.
- Finding: `_fetch_recent_daydreams`'s docstring said "Never raises" — it does; the *caller* swallows.
  - Fix: reworded to name where the failure is actually absorbed.
  - Evidence: reviewer triggered a real `OperationalError` out of `engine.connect()`.
- Finding: `theme_key` claimed as `''` on all rows; it is NULL.
  - Fix: corrected in code comment and README.
  - Evidence: `count(theme_key) = 0` over 329 rows; `grep theme_key visual_chain.py` returns nothing — the producer never sets it.
- Finding: `_install_fake_engine` discarded its arguments, so the SQL was 100% untested.
  - Fix: it now captures statement and params; `test_fetch_current_daydream_sql_names_the_real_table_and_key` asserts table, JSON key, ORDER BY, and both binds.
  - Evidence: mutation M3 (rename the JSON key) now fails the suite.
- Finding: the daydream block was 38% of the prompt — the largest block, for the least load-bearing lane; `_MAX_DAYDREAM_CHARS` (260) inverted `_MAX_CURIOSITY_SUMMARY_CHARS` (160).
  - Fix: one caption at a 200-char budget.
  - Evidence: live prompt 1471 → 959 chars; block 560 → 358 (23%).
- Finding: the ellipsis fallback produced `_MAX_DAYDREAM_CHARS + 1` chars — a soft cap here, a hard cap in the sibling function.
  - Fix: `text[: _MAX_DAYDREAM_CHARS - 1] + "…"`, matching `_fetch_curiosity_summaries`.
  - Evidence: `test_clean_daydream_ellipsis_fallback_respects_the_hard_cap`.
- Finding: stale comment — "the SUM of all **three** round trips" after the gather went to four.
  - Fix: corrected.
- Finding: the caption-boilerplate comment sat above the wrong constant.
  - Fix: moved.
- Finding: `chain_json["description"]` is an untyped cross-service dependency with no contract test.
  - Fix: documented at the field in `orion/schemas/reverie_visual.py`, naming the consumer and why `prior_description` is wrong.
- Finding: a live caption reads *"The graph you provided is a phase diagram…"* — second-person, contradicting the ownership framing.
  - Fix: partially addressed. That row is not currently the newest and only one caption ships, so the exposure is far smaller than with a 3-item list; the framing sentence remains the mitigation. **Not fully solved — see Concerns.**
- Finding: `services/orion-hub/` has no `evals/` directory (§11).
  - Fix: added, with the eval that would have caught the retracted claim.
  - Evidence: `4 passed` against live data; `4 skipped` with no DB.

## Restart required

```bash
# orion-hub, once merged. Not run by me (sudo).
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

No restart needed for anything else; no other service's behavior changes.

## Risks / concerns

- Severity: **medium**. Concern: second-person captions ("The graph you provided…") can still reach the prompt beneath a line asserting the image is Orion's own — 1 of 297 live captions has this shape, because the *captioner* was addressed conversationally. Mitigation: the ownership sentence, plus one-caption exposure. Proper fix is upstream (the captioner's prompt) or a second-person detector here; neither is in this PR.
- Severity: **low**. Concern: the lane changes what generation weighs, so it will move the PASS rate for unprompted outreach, and nothing measures that. Mitigation: the block is 23% of the prompt and explicitly subordinate ("if it connects to anything above"). A PASS-rate eval is the honest follow-up; `endogenous_outreach_decisions` already logs the data it would need.
- Severity: **low**. Concern: `chain_json["description"]` is an untyped cross-service key. Mitigation: documented at the schema; the eval's liveness test fails if the window empties.
- Severity: **low**. Concern: three degenerate columns and two captioner failure modes are real producer bugs left unfixed in orion-thought. Mitigation: documented here and in the README; the consumer-side guard holds regardless.

## PR link

<to be filled after `gh pr create`>
