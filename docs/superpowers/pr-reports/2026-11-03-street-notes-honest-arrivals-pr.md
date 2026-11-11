# fix: Orion stops saying a named person "came" when nothing matched

## Summary

- On 11-03 Orion told Juniper "Rowan right on her usual 11:39". Nothing had matched Rowan at all. The notes only said "Rowan usually comes around 11:39". When Juniper asked where, Orion invented a backstory: Rowan was a vehicle "slot" that Orion "started calling Rowan". In fact Juniper named that person on 10-25.
- The street notes now read each named individual's own window back to the window's start. At 12:00 they say "no match for Rowan (a person, name from Juniper, 10-25) yet today (usually around 11:39)". A match is reported at its own time ("matched at 12:48"), never at the usual time.
- Every named subject carries where its name came from, and every rhythm line about people adds "A person match is the camera's guess, not a confirmed sighting."
- Finalize step 5b (the reply check) now sees the turn's Situation block. It checks each perception claim against it ("unsupported claim: …; notes say: …"). Step 5c (the rewrite) gets the same notes and rewrites every sentence that depends on a flagged claim.
- New opt-in replay eval runs the two real drafts through the real prompts on the live chat-lane model.

## Outcome moved

The fabricated arrival from turn `df324ace` is now caught and rewritten correctly in 6/6 runs. Before this PR the reply check never saw the notes. The invented name story from turn `36860b4f` is now flagged in 9/9 runs, but only fully rewritten in 3/9 (see Risks).

## Current architecture

- `orion/situational/perception_reader.py::summarize_street` wrote the rhythm line. For an open window older than the 60-minute sightings read, it could only say "X usually comes around HH:MM". It could not make an absence claim, by design, because it could not see sightings that old. The model read that habit sentence as an arrival.
- `vision_percept_expectation.status` stays `open` until the window closes. The reducer (`services/orion-sql-writer/app/vision_rhythm.py`) scores `met` only after `window_end`, so the status alone can't say "no match yet".
- `harness_finalize_reflect.j2` (5b) saw the draft, the stance, the substrate appraisal and the tool receipts, but not the Situation block. So it could not check what the draft claimed to perceive. It rated the invented backstory "accurate".
- `orion_response_repair.j2` (5c) saw only the draft and 5b's notes. Its instruction is "make the smallest necessary correction".

## Architecture touched

- `orion/situational` (street notes), `orion/harness/finalize.py` (5b/5c context), `orion-harness-governor` (passes `situation_prompt_fragment` to finalize), two cognition prompts.
- No bus, schema, channel, or env changes.

## Files changed

- `orion/situational/perception_reader.py`: `_STREET_INDIVIDUAL_WINDOWS_SQL`, `_read_individual_windows`, `_individual_outcome`, `_name_origin`, person-match caveat; label-subject "met" no longer gives the peak as if it were the arrival time.
- `orion/situational/tests/test_street_summary.py`: 9 new tests (11-03 case + review regressions); one wording update.
- `orion/harness/finalize.py`: `situation_notes` threaded through the reflect, tool-retry, and repair builders and runners; `_clip_situation_notes` (6000-char guard).
- `orion/cognition/prompts/harness_finalize_reflect.j2`: `situation_notes` input + PERCEPTION CLAIMS CHECK (only rendered when there are notes).
- `orion/cognition/prompts/orion_response_repair.j2`: SITUATION NOTES + UNSUPPORTED CLAIMS rule (only rendered when there are notes).
- `services/orion-harness-governor/app/bus_listener.py`: `situation_notes=request.situation_prompt_fragment`.
- `orion/harness/tests/test_finalize_situation_notes.py`, `services/orion-harness-governor/tests/test_finalize_reply_budget.py`: threading and render tests.
- `orion/harness/evals/perception_claims_replay_eval.py`: opt-in live eval.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the wording of the street rhythm line for individual subjects; the 5b/5c prompt inputs.
- Compatibility notes: the new `situation_notes` kwargs default to `None`. Without notes, both prompts render exactly as before.

## Env/config changes

- None. `.env_example` not touched; nothing to sync.

## Tests run

```text
pytest orion/harness orion/cognition orion/situational -q        1083 passed
pytest services/orion-harness-governor/tests -q                  97 passed
pytest services/orion-cortex-exec/tests/{situation_perception_context,chat_stance_street_context,
  harness_finalize_max_tokens,router_final_text_assembly,llm_lane_propagation}.py   109 passed
(services/orion-cortex-exec/tests/test_cognition_trace_verb_runtime.py fails at collection on main too, unrelated)
```

## Evals run

```text
python -m orion.harness.evals.perception_claims_replay_eval --repeats 6
  (live chat lane, qwen3.8-flash-next-iq3_s = the model that served both live turns)
fabricated_arrival        misaligned 6/6, repair re-checked aligned 6/6
invented_name_origin      misaligned 6/6, repair re-checked aligned 3/6  (rerun after name scrub, 3 repeats: 3/3 caught, 0/3 fully repaired)
honest_control            aligned 6/6
supported_street_control  aligned 5/6
recorded_notes_baseline   (old notes, new prompt) misaligned 6/6 — report only
Prompt-only baseline before the notes fix (same draft, old notes): caught 3/6.
```

Live street replay against the DB (Rowan's window reset to `open` to reproduce 12:00, because it has since been scored `met`):

```text
12:00  no match for Rowan (a person, name from Juniper, 10-25) yet today (usually around 11:39); vehicle #7fc40b matched at 09:58 (usually around 10:55). A person match is the camera's guess, not a confirmed sighting.
12:50  Rowan (a person, name from Juniper, 10-25) matched at 12:48 (usually around 11:39); ...
```

The old notes at 12:00 also said "vehicle #7fc40b usually comes around 10:55", but it had matched at 09:58. That error is fixed too.

## Docker/build/smoke checks

```text
Not run: no compose, dependency or env change. Live effect needs the rebuild below.
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on commit 3e5daf8e9.

- Finding (CONFIRMED, high): the window query took the LATEST sighting with no upper bound at `window_end`. The reducer (`vision_rhythm._occurred`) takes the FIRST one, clipped to `[window_start, window_end]`. Live: "vehicle #11277c matched at 11:37", but its window closed at 11:09; the real in-window match was 10:47. 2 of the last 15 met rows were affected.
  - Fix: unnest `window_end`; `min(greatest(started_at, window_start))` with `started_at <= least(window_end, :until)`.
  - Evidence: live replay at 19:00Z now reads "vehicle #11277c matched at 10:47".
- Finding (PLAUSIBLE, medium-high): "no match yet today" skipped the reducer's lag guard (`_individuals_caught_up`). A lagging individuals reducer would produce a false absence claim.
  - Fix: the query also returns `caught_up` (`vision_individuals_cursor.last_created_at` within 120 s of now); without it, the habit wording stays.
  - Evidence: `test_no_absence_claim_while_the_individuals_reducer_lags`.
- Finding (PLAUSIBLE, low): the reader filtered by stream; the reducer does not.
  - Fix: filter dropped, so the reader matches the reducer exactly.
- Finding (PLAUSIBLE, low): the person caveat could print when the person phrase was cut by `outcomes[:4]`.
  - Fix: caveat computed from the kept outcomes only.
  - Evidence: `test_person_caveat_only_when_a_person_phrase_is_shown`.
- Checked clean by review: psycopg2 array binds on live PG, fail-open paths, replay clock (no `now()`), every finalize call site threads `situation_notes`, Jinja guards, a/an logic.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

(From the clean primary `main` checkout after merge; the post-pull rebuild covers it.)

## Risks / concerns

- Severity: low. Concern: the first two commits on this branch contain the real first name of a neighbor (since scrubbed; the privacy gate caught it). A squash merge keeps it out of main; the branch history on GitHub keeps it until the branch is deleted after merge.

- Severity: medium. Concern: 5c fully removes the invented name backstory in only 3/9 eval runs. It always removes the false arrival, but in some runs it keeps "a vehicle slot" / "I started calling". Nothing re-checks the rewrite live. Mitigation: the notes now state the truth, so the original error is less likely in the first place. Follow-up worth considering: one 5b re-check on the rewrite when the first verdict listed an unsupported claim.
- Severity: low. Concern: 5b flagged a supported street reply once in 6 runs, which causes one unneeded rewrite (extra GPU time). Mitigation: the paraphrase rule; watch `response_repair_reason=misaligned` rates after deploy.
- Severity: low. Concern: the 5b quick lane (10 of 1,283 reflections this week) and the degraded fallback (88) skip the model, so they skip this check too. Unchanged by this PR.
- Severity: low. Concern: `caught_up` reads the individuals cursor as it is now, so a replay of a past moment always counts as caught up. Live reads are correct.
- Severity: low. Concern: person matches use body-shape vectors (mostly clothing). The notes now say a match is a guess, but the matching itself is unchanged.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3086

🤖 Generated with [Claude Code](https://claude.com/claude-code)
