# Record which grounding lanes actually reached the outreach prompt

Branch: `feat/outreach-grounding-trace` (off `main` after #1927 merged)

## Summary

- #1927 deployed the daydream lane. The first question after deploy — *"did that outreach actually see a daydream?"* — turned out to have **no answer**.
- The outreach prompt is built in memory, handed to generation, and dropped. It is in **no** durable store.
- Adds a `grounding` object to `endogenous_outreach_decisions.result_json`: which lanes were present, plus the daydream's age.
- Booleans and counts only — never caption or summary text.
- A decision gated before context is built gets **no** `grounding` key rather than a stale one from the previous cycle.

## Outcome moved

Every grounding lane in this prompt was unfalsifiable in production. An outreach that silently lost a lane and one that never had it looked identical after the fact — which is precisely the failure mode `endogenous_outreach_decisions` was created for on 2026-08-22, one level further in.

## Current architecture

Verified live before writing any code, on the deployed container:

| Candidate sink | Carries the prompt? |
|---|---|
| `endogenous_outreach_decisions.result_json` | No — `chars`, `final_len`, `elapsed_sec`, `fcc_model_label`, `harness_grounding_status` |
| `docker logs orion-athena-hub` | No — only `endogenous_outreach_context_read_failed` on error |
| `chat_history_log` | No — it *has* a `prompt` column, but it is empty (`length=0`) on all 4 outreach rows while `response` is populated. Across 3 days it is filled only for `orion_journal`, and at `avg(length)=35` it is a short trigger label, not an assembled prompt. |
| `emit_observation` | Builds a `SubstrateMoleculeV1` carrying `surface_text`; **no queryable Postgres sink** (`pg_tables ~* 'molecule'` returns nothing) |

## Live evidence that the lane it traces actually works

#1927 merged `2026-08-28T04:49:44Z` and has now run through a full non-quiet window. Reasons since that instant:

```text
quiet_hours 3223 | daily_cap 2765 | cooldown 807 | no_tension_trigger 772 | sent 4
```

All 4 sends carried a daydream, traced by hand against `reverie_visual_chain`:

| Sent (MDT) | Caption minted | What Orion wrote |
|---|---|---|
| 08:44 | 08:37 `A person standing on a rocky outcrop, wearing a long coat, jeans, and boots` | "someone on a rocky outcrop, coat pulled tight against whatever wind is up there" |
| 10:34 | 10:25 `a blurred, soft-focus view of a sky with scattered, bright, white circles` | "that same sky of scattered white circles—light gathering without a frame" |
| 11:33 | 11:20 `a starry night sky with a mix of bright and dim stars` | "that sky of scattered stars with light gathering without a frame" |
| 12:29 | 12:25 `A young person with long dark hair styled in two high ponytails, each adorned with a red hair tie... a light blue shirt with a floral pattern` | "a young person in a light blue floral shirt, hair tied back with red bands" |

Confirms three of #1927's claims on real traffic, not fixtures:

- The `The image depicts ...` prefix strip fired on every caption that had one.
- The degenerate captions at 11:41/11:52 (`There are no visible objects or people in the image.`) never reached a message — `_MIN_DAYDREAM_CHARS` plus the prose check held.
- Orion connects the image to telemetry rather than describing it back ("which maps perfectly to this dense concept region waiting for branches that haven't arrived yet").

**This is also the argument for the patch.** The table above was built by eyeballing lexical overlap between two tables by hand, for 4 rows. It does not scale, it is not a query, and it is not available to any alert or eval. `result_json->'grounding'` makes the same fact a boolean.

**Measured follow-up, not fixed here:** the 10:34 and 11:33 sends drew on the same visual theme an hour apart, and Orion named it ("the reverie *keeps returning* that same sky"). This is the caption-repetition that #1927 deliberately retracted a de-dupe for, after measuring that no similarity threshold separates the corpus. It is now observed reaching Juniper twice in one day. Orion framed it as continuity rather than glitching on it, so this is filed as an observation, not a defect.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: `grounding_summary()`; `self._last_grounding` set after context-gather, reset per cycle, folded in by `_record`.
- `services/orion-hub/tests/test_endogenous_outreach.py`: 5 tests.
- `services/orion-hub/README.md`: §4.1 subsection with the query.

## Schema / bus / API changes

- Added: a `grounding` key inside the existing `result_json` jsonb. **No migration** — additive to an untyped column.
- Removed / renamed: none.
- Compatibility: readers that ignore unknown keys are unaffected; `result_json ? 'grounding'` distinguishes new rows.

## Env/config changes

None. No key added, removed, or renamed; `.env_example` untouched, so no sync needed.

## Privacy

Deliberately **booleans and counts only**. Logging the caption or the curiosity summaries would copy real content into a second store with its own retention, quietly widening the boundary the daydream lane states (exactly one `chain_json` key, no seed material). `test_grounding_summary_records_no_caption_text` pins this: it asserts distinctive words from every text-bearing lane are absent from the serialized trace, and the mutation adding `daydream_caption` fails 3 tests.

## Tests run

```text
services/orion-hub/tests/test_endogenous_outreach.py  ->  163 passed in 5.02s
full hub suite (branch) -> 32 failed, 1743 passed, 2 skipped in 231.39s
full hub suite (main)   -> 32 failed
```

Same failure count both sides. The two sets differ by one entry each way, both
pre-existing and neither in touched code:

- `test_schedule_panel_browser_smoke` — fails on `main` too, in isolation:
  `playwright/_impl/_connection.py:559: Error`. No browser in this environment.
- `test_substrate_mutation_manual_route_routing` — **passes** in isolation on
  `main`; the known order-dependent flake already on the parked list.

Run with the primary checkout's interpreter and the worktree as cwd: the worktree
has no venv, so `test_service.sh`'s `choose_python` falls back to a `python3` with
no pytest. That first produced an EMPTY result file, which diffed against main's 32
as "zero failures" — a clean-looking green from a suite that never ran. Confirmed
the run really loads worktree code before trusting it (10 collected tests match
`grounding`; they exist only on this branch).

Mutation tests — each new test must fail against the code it guards:

| # | Mutation | Result |
|---|---|---|
| M1 | `embodied_presence` back to `ctx.embodied_presence is not None` | 1 failed |
| M2 | reintroduce the instance field with a `_record` fallback | 2 failed |
| M3 | M2 + the per-cycle reset | 1 failed — **not faithful**, the explicit parameter still won |
| M4 | true original: instance field only, parameter removed | 2 failed |

M3 is recorded because it looked like a passing mutation and was not one — it
proved my mutation was too weak, not that the test was. M4 is the honest one.

## Evals run

No new eval. #1927 added `services/orion-hub/evals/`; this change is a decision-log field with no live-quality dimension to measure. The natural eval — "of the outreaches that fired, how many saw each lane" — needs `grounding` rows to exist first, so it is follow-up, not something this PR can honestly ship.

## Docker/build/smoke checks

Pure-Python, no dependency/port/compose change.

Deployment state of #1927 verified on the live container first, since this PR exists because of what that verification found:

```text
$ docker exec orion-athena-hub sh -c "grep -c '_strip_appended_list|_fetch_current_daydream|_DAYDREAM_SECOND_PERSON_RE' /app/scripts/endogenous_outreach.py"
7
$ docker exec orion-athena-hub python3 -c "... _fetch_current_daydream()"
(1331.0, 'a circular structure in space, resembling a nebula or a galaxy. The central region is bright...')
```

The lane reads real data from inside the deployed container. What could not be shown is whether a *delivered* outreach carried it — hence this PR.

## Review findings fixed

- Finding: **MUST-FIX 1** — `offer_message` (the curiosity loop's delivery path, live-wired and enabled) records a *previous* cycle's lanes on a delivered message. It never builds an `OutreachContext` and never reset the shared field, so a curiosity message that saw no daydream shipped a row asserting one, with a frozen `age_sec` that reads as precise and current rather than stale.
  - Fix: pass the summary as a `_record(..., grounding=)` parameter; delete `self._last_grounding`. This is the pattern the file already adopted for `forced` after the identical finding on 2026-08-22 — its comment was four lines above the bug I wrote.
  - Evidence: `test_offer_message_records_no_grounding_of_its_own` fails against the exact original code (mutation M4).

- Finding: **MUST-FIX 2** — the per-cycle reset ran before the `_send_lock` check, so a concurrent tick entering `already_sending` nulled the field out from under a cycle already inside `_generate`, stripping the trace off the *delivered* row. Worse than losing data: an absent key is what the README teaches operators to read as "gated before context," so the loss is indistinguishable from a meaningful signal.
  - Fix: same parameter change — a local in `_outreach_once` cannot be reached by another task's stack frame.
  - Evidence: `test_a_concurrent_tick_cannot_strip_the_trace_off_a_delivered_row` fails under M4. The test gates the second tick on an event set *inside* the `_generate` stub; with the `asyncio.sleep(0)` the existing concurrency test uses, the first task is still parked upstream and the race does not reproduce — the test passes green against the bug.

- Finding: **MUST-FIX 3** — `embodied_presence` reported the fetched row, not the rendered line, and was wrong on live data (cam0 `absent`, `ENDOGENOUS_OUTREACH_PERCEPTION_STREAM_ID=cam0` set in the container).
  - Fix: report `bool(presence_fragment(...))`.
  - Evidence: `test_embodied_presence_reports_the_rendered_line_not_the_fetched_row` asserts both directions against `build_outreach_prompt`; reverting only that expression fails it (M1).

- Finding: **SHOULD-FIX 4** — the summary was built before the prompt, so a `no_grounding_context` row (prompt `== ""`) could carry `daydream: true`, since `is_empty()` deliberately ignores `daydream`/`embodied_presence`.
  - Fix: build it after the prompt check; that row now carries no key, and `reason` distinguishes it from a gated row.
  - Evidence: documented in the README's row-by-row table of which rows carry no key.

- Finding: **SHOULD-FIX 5** — README claimed "**Every** decision row ... now carries a `grounding` object" and contradicted itself four paragraphs later; the PR report filed a live defect as hypothetical.
  - Fix: both rewritten; the README now carries a table of the three no-key cases, and the risk bullet says what was actually false.

- Finding: NIT — `daydream_age_sec` at 0.1s precision is a join key onto the caption row.
  - Fix: rounded to whole seconds, and the residual join is stated plainly in Risks rather than implied away.

- Not taken: renaming the mixed bool/count sibling keys. Mixed value types in one jsonb object are ordinary and every future query casts per-key regardless; the churn across docs and tests buys nothing.
- Filed, not fixed here: `build_outreach_prompt` reads `ctx.presence.get("health")` but `hub_presence.presence_snapshot` returns `connection_health`, so that line always renders `unknown`. Pre-existing and out of scope for this commit — but this patch is what makes the dead half queryable, so it is worth its own fix.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

## Risks / concerns

- Severity: **medium**. Concern: **no `grounding` row exists in production yet** — this patch's own live path is `UNVERIFIED`. The Docker/smoke evidence below verifies #1927's daydream lane inside the container, which is a *different* patch. Mitigation: after deploy, one query settles it — `SELECT reason, result_json->'grounding' FROM endogenous_outreach_decisions WHERE result_json ? 'grounding' ORDER BY decided_at DESC LIMIT 1;`
- Severity: **low**. Concern: `result_json` grows by ~120 bytes per decision row, and the loop ticks every 10s. At ~8,600 rows/day that is ~1MB/day of additional jsonb. Mitigation: gated decisions (the overwhelming majority — 3,155 `quiet_hours` + 3,030 `daily_cap` in 24h) carry no `grounding` key at all, so the real growth is confined to cycles that actually built a context.
- Severity: **low** (was the third MUST-FIX). Concern: the trace originally recorded what `OutreachContext` held, not what `build_outreach_prompt` rendered. I filed that as hypothetical future drift — *"they agree today because every lane renders when populated"* — and it was **false when written**: `presence_fragment` returns `None` for any state that is not `present`/`recent`, and cam0 was `absent`, so every row that day would have claimed a camera lane the prompt did not contain. Filing a live defect as speculative future risk is what let it ship. Now fixed and regression-tested; the residual risk is that a *future* lane repeats the same fetched-vs-rendered split.
- Severity: **low**. Concern: `daydream_age_sec` is a de-facto join key — `reverie_visual_chain.created_at ≈ decided_at - daydream_age_sec` identifies the exact caption row. Rounding to whole seconds (done) reduces false precision but does **not** remove the join, and no rounding would: recording the age at all implies it. Judged acceptable because the caption text still lives in exactly one store and that store is already operator-queryable — but stating it beats leaving it implied given how hard this PR sells the boundary.

## PR link

<to be filled>
