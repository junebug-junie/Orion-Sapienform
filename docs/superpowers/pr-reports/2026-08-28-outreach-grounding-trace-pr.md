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
$ pytest services/orion-hub/tests/test_endogenous_outreach.py -q
160 passed in 4.55s
```

Mutation-tested — all caught, each asserting its target exists before mutating:

| Mutation | Result |
|---|---|
| drop the per-cycle `_last_grounding = None` reset | 1 failed |
| never record the trace | 1 failed |
| also record the caption text | 3 failed |

The first is the one worth naming: `_last_grounding` lives on the instance, so without the reset a `quiet_hours` decision inherits the previous cycle's lanes and records them as its own. Stale lanes are worse than none — they read as evidence.

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

Not yet reviewed — see Concerns.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

## Risks / concerns

- Severity: **medium**. Concern: no code review has run on this branch yet. Proposed follow-up: run it before merge.
- Severity: **low**. Concern: `result_json` grows by ~120 bytes per decision row, and the loop ticks every 10s. At ~8,600 rows/day that is ~1MB/day of additional jsonb. Mitigation: gated decisions (the overwhelming majority — 3,155 `quiet_hours` + 3,030 `daily_cap` in 24h) carry no `grounding` key at all, so the real growth is confined to cycles that actually built a context.
- Severity: **low**. Concern: the trace records what `OutreachContext` held, not what `build_outreach_prompt` rendered. They agree today because every lane renders when populated, but a future conditional in the renderer could diverge. Mitigation: both live in one file; a renderer-level trace would be the stronger form if that changes.

## PR link

<to be filled>
