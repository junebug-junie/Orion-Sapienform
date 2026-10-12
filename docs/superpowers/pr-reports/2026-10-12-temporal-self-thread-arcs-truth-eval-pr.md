# Temporal Self: open threads carry their arc id, and a truth eval for day citations

## Summary

- Each open conversation thread in Orion's day record now carries the id of the record it came from (`OpenThreadV1.arc_id`). The chat cue cites every thread it mentions instead of "2 open threads from conversation, 1 with its record".
- The closed-day block in Orion's Day letter already cited every thread with a row; it now also names a thread open at close whose row is missing, rather than dropping it.
- New read-only truth eval (`make eval-temporal-self-truth`): for stances whose day citations passed the gate, it checks whether each sentence citing an arc actually describes that arc (kind, subject, "N min ago", clock times, counts). Fixed rules only, no model judge. Anything the rules can't read is counted as undecidable. Output is per sentence plus a weekly truth rate.
- The eval found a real issue in the letter's closed-day block: two attention lines state a group total ("11 returns" over 11 arcs) but cite only the longest arc, which has 3. Pinned by a test, not fixed here (it changes letter wording; see Risks).

## Outcome moved

Before: the live cue at 2026-10-12 01:16 UTC said
`2 open threads from conversation, 1 with its record [arc:fd63c7ab0ad3ecbe].`
The other thread's concern arc (684d45c2010540ed, carried in at local midnight) had dropped off the frame's 64-arc `arcs_today` cap, and the thread had no id of its own. So Orion could not cite that thread, and the gate would have stripped the id if Orion had cited it.

After, rendered by this branch's code from the same live frame (thread ids looked up in `temporal_self_arc`, since the deployed producer doesn't emit them yet):

```text
Dusk, Sun 19:16 local (day record as of 19:15). No attention or conversation arc is open right now.
Before that: attention on Codebase prediction error, ended 31 min ago [arc:381981f663496a47]. Juniper
last spoke 2 h ago [arc:571bce17764b0a9e]. 2 open threads from conversation [arc:684d45c2010540ed]
[arc:fd63c7ab0ad3ecbe]. Last slept 16 h 10 min ago [arc:460c1b680514c3e8].
```

375 characters (cap 400). The gate's shown set is the cue's `arc_ids`, and it now includes `684d45c2010540ed`.

Before: nothing checked whether a cited sentence was true to its arc (spec acceptance check 6, never built). After: a periodic eval does, and reports what it cannot decide.

## Current architecture

- `build_frame` lists open threads from open concern arcs. `OpenThreadV1` had no arc id.
- `project_stance_cue` matched threads to arcs by `subject_ref` through `frame.arcs_today`, which is capped at 64 (non-reverie, most recent first). It cited at most 3 threads.
- The citation gate (`citations.py`, `gate_day_citations`) checks that cited ids exist and were shown. Results go in `thought_decision.day_citations`.

## Architecture touched

- Schema: `OpenThreadV1.arc_id: str | None = None` (`orion/schemas/temporal_self.py`). It was already registered; no new models.
- Producer: `orion/temporal_self/frame.py` (runs in orion-durable-runs `chronicle`).
- Readers: `orion/temporal_self/stance_cue.py` (orion-thought), `orion/temporal_self/day_block.py` (orion-hub letter gather). `orion/orion_day/` is untouched.
- New: `orion/temporal_self/citation_truth.py` (pure), `orion/temporal_self/evals/run_day_citation_truth_eval.py` (read-only SQL), `Makefile` target `eval-temporal-self-truth`.

## Files changed

- `orion/schemas/temporal_self.py`: `OpenThreadV1.arc_id` (optional, with a consumer-first note).
- `orion/temporal_self/frame.py`: fills `arc_id` from the concern arc.
- `orion/temporal_self/stance_cue.py`: `_threads_clause` cites every thread, most recently returned first, up to 5. Past 5 it reads "N open threads from conversation, the 5 most recent [...]". When the 400-char cap is hit, the sleep line goes first, then the thread refs shrink (the count stays true, down to "the most recent"), and the thread clause is dropped last. Old frames keep the old lookup and wording.
- `orion/temporal_self/day_block.py`: a thread open at close whose arc row is missing renders as "<ref>, open at close, arc <id> not in the day record [day:<date>]".
- `orion/temporal_self/citation_truth.py`: the deterministic checker.
- `orion/temporal_self/evals/run_day_citation_truth_eval.py`: the live read-only entry point.
- `Makefile`: `eval-temporal-self-truth`.
- `orion/temporal_self/tests/test_thread_arcs.py`, `test_citation_truth.py`, `test_arcs.py`: tests.
- `orion/temporal_self/README.md`: docs.

## Schema / bus / API changes

- Added: `OpenThreadV1.arc_id` (optional). Shows up in `GET /temporal-self/frame`, `GET /temporal-self/threads`, `temporal_self_projection` and `temporal_self_day.day_json`.
- Removed / renamed: none.
- Behavior changed: the cue's thread clause. The day block only changes in the missing-row case.
- Compatibility: rows stored before this still validate (`arc_id` defaults to None; tested on the real 10-11 frames and the 10-09 closed day). `OpenThreadV1` is `extra="forbid"`, so a reader on old code would reject a frame that has `arc_id`. That means the **readers deploy first**: orion-thought and orion-hub, then orion-durable-runs. If durable-runs goes first, orion-thought shows no cue (`skip_reason=invalid_frame`, fail-open) until it is rebuilt, and the Hub's letter gather would fail its `temporal_day` source on a day closed by the new code. No bus channel change. No migration.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.

## Tests run

```text
pytest orion/temporal_self/tests orion/thought/tests                 -> 219 passed
services/orion-durable-runs: PYTHONPATH=.:../.. pytest tests         -> 388 passed, 85 skipped
services/orion-thought: pytest tests/test_stance_temporal_self_cue.py -> 12 passed
services/orion-hub: pytest tests/test_orion_day_temporal_day.py       -> 6 passed
scripts/check_env_template_parity.py                                 -> PASS
```

New tests cover:
- schema round trip, old and new: an old thread, frame and closed day still load; new frame and day survive JSON
- exact cue string with every thread cited
- old-frame wording kept
- over-cap wording
- the gate accepts a thread arc that only the thread carried, and rejects it against the old cue
- the day block cites every open-at-close thread, names a missing-row thread, and leaves the 10-09 render byte-identical
- truth-eval fixtures from live arc rows: match; wrong kind; wrong time; right time; wrong subject; undecidable (no readable fact); missing row; row moved after the sentence (undecidable, not a mismatch); counts; evidence_refs attribution; weekly summary; the entry point on DB-shaped rows

## Evals run

Live, read-only, 2026-10-12 01:20 UTC (`make eval-temporal-self-truth ARGS="--labels verified,UNVERIFIED,no_day_claims"`):

```text
Day-citation truth eval: decisions since 2026-10-05 01:20Z, labels ['verified', 'UNVERIFIED', 'no_day_claims'], 1 row(s). Deterministic checks only; no LLM judge.

2026-10-12T00:59:41.540555+00:00 fbac3775-02be-44ae-a92f-c6f67ac2fa29 label=verified cited=4 reply=none stored
  note: no reply text stored
  note: no sentence to check: no inline arc citation and no reply sentence stating a checkable day fact

Weekly summary (truth_rate = match / (match + mismatch)):
  2026-W42: rows 1 (0 with sentences), sentences 0: match 0, mismatch 0, undecidable 0; truth_rate n/a (nothing decided)
```

What that means: the only cited row so far is the one from 00:59. It cited all 4 shown arcs, and only in `evidence_refs`. Its imperative and tone carry no inline `[arc:]`, and no reply for that correlation id exists in `chat_history_log` or `harness_turn_trace`. The orion-thought log shows the stance itself took 211 s (`[rpc] reply received ... elapsed_ms=210881.3`). Whether the turn timed out upstream is **UNVERIFIED**. So "verified" on that row means "the ids are real and were shown". Whether any sentence about them was true has not been tested on live data yet. The eval says so instead of reporting a rate.

Calibration on real rows (tests, in CI). The checker is run over the renderers' own output:
- the stance cue from the real 10-11 frames, both at render time and with the frame about 10 min old: every sentence `match`;
- the 10-09 closed-day block: 14 match, 1 undecidable (the instrument-warnings line), 2 mismatch, both real (see Summary and Risks).

Calibrating against the renderers caught two checker bugs before the live run:
- an attention label containing parentheses was cut short;
- "I slept 3 h ago" matched a conversation arc's start time.

Times and counts are now scoped to the kind the sentence names.

## Docker/build/smoke checks

```text
Not built or deployed (Juniper: do not deploy). Live reads only: GET /temporal-self/frame (host :8124),
temporal_self_arc / thought_decision / chat_history_log / harness_turn_trace via a read-only session.
```

## Review findings fixed

The code review ran in a subagent against `origin/main...feat/temporal-self-thread-arcs-truth-eval`. Most of the material findings were the truth checker calling a true sentence a mismatch. The rule is now: a sentence is a **mismatch** only when its citation is one-to-one (inline, and every cited arc is of a kind the sentence names) and the row could not have moved since the frame. Everything else that misses is undecidable.

- Finding (must): if the frame was up to 900 s behind the row, a true return count or clock was called a mismatch.
  - Fix: "row moved" is now measured from `at - 900 s` for times, clocks and counts.
  - Evidence: `test_a_lagging_frame_does_not_turn_a_true_count_into_a_mismatch`; calibration also re-run with the frame about 10 min old.
- Finding (must): ordinary words like "talking with Juniper" next to a concern citation were called a wrong kind.
  - Fix: a kind mismatch needs a single named kind plus a timed or counted claim ("Last slept 2 h ago [arc:<attention>]"). Kind phrases inside a subject label are no longer read as kinds.
  - Evidence: `test_kind_words_without_a_timed_claim_are_not_decisive`, `test_a_label_holding_a_separator_or_a_kind_phrase_is_read_whole`, `test_wrong_kind_is_a_mismatch`.
- Finding (must): a paraphrased subject was called a mismatch.
  - Fix: a subject is a mismatch only when it is exactly the label of another, uncited arc; otherwise undecidable. Cited labels are matched literally first.
  - Evidence: `test_a_paraphrased_subject_is_undecidable`, `test_wrong_subject_is_a_mismatch`.
- Finding (must): hedged times ("over 2 h ago", "about", "under N min") were checked as exact.
  - Fix: a hedge word before the number makes the claim a lower bound, an upper bound or an estimate; a miss on a hedged time is undecidable.
  - Evidence: `test_hedged_times_are_bounds_not_values`.
- Finding (should): "for N" outside the template was read as the arc's start.
  - Fix: only "attention on X for N" / "open for N" mean start; other uses match any of the arc's times and are never decisive.
  - Evidence: `test_for_outside_the_template_is_not_a_start_time`.
- Finding (should): kind scoping dropped a cited arc and then reported a mismatch.
  - Fix: if a cited arc is of a kind the sentence doesn't name, misses are undecidable.
  - Evidence: `test_an_uncited_kind_makes_a_time_miss_undecidable`.
- Finding (should): a label containing a separator (", ", " for ") was cut short. Fix: literal label match first. Evidence: as above.
- Finding (should): a clock giving the current time was called a mismatch. Fix: a clock within the frame-lag window of the turn is undecidable. Evidence: `test_a_clock_naming_the_time_of_writing_is_undecidable`.
- Finding (should): the open-thread count could mismatch in evidence_refs mode. Fix: inline and one-to-one only. Evidence: `test_open_thread_count_in_a_reply_never_mismatches`.
- Finding (should): raising the thread cap to 5 could push the whole thread clause out of the 400-char cue.
  - Fix: refs shrink before the clause drops.
  - Evidence: `test_many_threads_with_long_labels_shrink_the_refs_not_the_clause`, `test_two_threads_keep_the_sleep_line`.
- Nits fixed:
  - uppercase arc ids are now fetched (`re.I`, lowercased);
  - label containment needs at least 6 characters;
  - reasons quote without trailing punctuation (`test_reasons_quote_without_trailing_punctuation`).
- Finding (rollout): once durable-runs ships, `temporal_self_day.day_json` stores `arc_id`, so a rollback of orion-hub to a pre-patch image cannot load days closed after that. Recorded under Risks.

## Restart required

Deploy order matters (readers first). From the primary checkout on main, after merge:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

No migration.

## Risks / concerns

- Severity: medium. Concern: deploying durable-runs before orion-thought/orion-hub makes the cue disappear (`invalid_frame`) and fails the letter's `temporal_day` source on the next close. Mitigation: the order above. Both readers fail open (no cue, a named gap in the letter), not into a crash.
- Severity: medium. Concern: rollback. After durable-runs ships, every frame and closed day it stores carries `arc_id`. Rolling orion-thought or orion-hub back to a pre-patch image would make them reject those rows: no cue, and the letter's `temporal_day` source fails for days closed after the deploy. Mitigation: roll back durable-runs first (new rows stop carrying the field; already-closed days still do), or roll the readers forward again.
- Severity: low. Concern: the closed-day block's attention lines cite one arc for a group total ("11 returns [arc:<longest>]", where that arc has 3). The truth eval flags it, and a test pins it. Mitigation: a follow-up wording change, such as citing the day for the group total or stating the cited arc's own numbers. It changes the letter's text, so it is left for Juniper.
- Severity: low. Concern: arc rows are mutable. When a cited row moved after the sentence and no longer matches, the eval says undecidable, never mismatch. That under-counts mismatches on long-running arcs. Mitigation: counted and visible in `decided_share`.
- Severity: low. Concern: the eval reads only the stance path. Letter citations (`orion_day_letter.sources.arc_citations`) are not covered yet; no letter with citations exists as of 10-12. Follow-up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2623

🤖 Generated with [Claude Code](https://claude.com/claude-code)
