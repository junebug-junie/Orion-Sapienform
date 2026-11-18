## Summary

Juniper flagged a voice habit in Orion's replies: the LLM ", not X" contrast frame ("that's physiology, not hierarchy"). This is the first step: count it, and make the trend re-runnable.

- The tone meter (`orion/personality/tone_dials.py`) gains two dials: `contrast_not_count` and `contrast_not_per_100_words`. They count ", not X", "— not X", " - not X", " -- not X" and " – not X". Markdown emphasis around "not" and a quoted or code word after it still count.
- They do not count "whether or not", "if not", "or not", "but not", "not only ... but (also)" lists, "not to mention", or "not least". Code fences are still stripped. The harness's own "[Cut short - not a finished answer.]" marker is stripped, because code wrote it, not Orion.
- The baseline script (`scripts/analysis/measure_tone_dial_baseline.py`) now prints the habit by ISO week, per group (chat replies, outreach, all) and per served model. Each week shows the share of replies with at least one frame, the frames per 100 words, and the share of replies with a dash beside it. Edge weeks are tagged `(partial week)`.

## Outcome moved

The ", not X" habit now has a number and a re-runnable weekly trend. The trend also answers why it jumped. **The serving model is the main driver.** Qwen3.6-35B sat at 10-21% of replies per week. Flash-Next (`qwen3.8-flash-next-iq3_s`) became the chat model around 10-26 and served nearly every turn from 11-02. It started at 34% and runs 47-49% now. Flash-Next also drifted up on its own (34% in its first week, 47-49% after). That part is unexplained. (Orchestrator found this with a hand join of `chat_history_log` to `cockpit_turn_sighting`; the script now does the same join.)

Live output, read-only, run 2026-11-18, `--window-days 60`, replies longer than 40 characters:

```text
[contrast_not_by_week:chat_replies]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W38 (2026-09-14)  n=   2  contrast=  0.0%  per100=0.000  dash=100.0%  (partial week)
  2026-W39 (2026-09-21)  n=  15  contrast= 13.3%  per100=0.205  dash= 66.7%
  2026-W40 (2026-09-28)  n=  17  contrast=  0.0%  per100=0.000  dash= 64.7%
  2026-W41 (2026-10-05)  n=  34  contrast= 23.5%  per100=0.249  dash= 94.1%
  2026-W42 (2026-10-12)  n=  32  contrast= 31.2%  per100=0.166  dash= 93.8%
  2026-W43 (2026-10-19)  n=  30  contrast= 16.7%  per100=0.132  dash= 90.0%
  2026-W44 (2026-10-26)  n=  60  contrast= 18.3%  per100=0.168  dash= 91.7%
  2026-W45 (2026-11-02)  n=  52  contrast= 53.8%  per100=0.272  dash=100.0%
  2026-W46 (2026-11-09)  n=  35  contrast= 51.4%  per100=0.380  dash=100.0%
  2026-W47 (2026-11-16)  n=   7  contrast= 71.4%  per100=0.472  dash=100.0%  (partial week)

[contrast_not_by_week:outreach]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W38 (2026-09-14)  n=   8  contrast=  0.0%  per100=0.000  dash=100.0%  (partial week)
  2026-W39 (2026-09-21)  n=  29  contrast= 10.3%  per100=0.177  dash=100.0%
  2026-W40 (2026-09-28)  n=  28  contrast=  0.0%  per100=0.000  dash= 67.9%
  2026-W41 (2026-10-05)  n=  28  contrast= 10.7%  per100=0.205  dash= 71.4%
  2026-W42 (2026-10-12)  n=  27  contrast= 14.8%  per100=0.127  dash= 85.2%
  2026-W43 (2026-10-19)  n=  24  contrast= 29.2%  per100=0.185  dash= 91.7%
  2026-W44 (2026-10-26)  n=  92  contrast= 32.6%  per100=0.186  dash= 89.1%
  2026-W45 (2026-11-02)  n= 119  contrast= 44.5%  per100=0.242  dash= 92.4%
  2026-W46 (2026-11-09)  n=  70  contrast= 51.4%  per100=0.267  dash= 98.6%
  2026-W47 (2026-11-16)  n=  12  contrast= 33.3%  per100=0.191  dash=100.0%  (partial week)

[contrast_not_by_week:all]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W38 (2026-09-14)  n=  10  contrast=  0.0%  per100=0.000  dash=100.0%  (partial week)
  2026-W39 (2026-09-21)  n=  44  contrast= 11.4%  per100=0.186  dash= 88.6%
  2026-W40 (2026-09-28)  n=  45  contrast=  0.0%  per100=0.000  dash= 66.7%
  2026-W41 (2026-10-05)  n=  62  contrast= 17.7%  per100=0.236  dash= 83.9%
  2026-W42 (2026-10-12)  n=  59  contrast= 23.7%  per100=0.155  dash= 89.8%
  2026-W43 (2026-10-19)  n=  54  contrast= 22.2%  per100=0.161  dash= 90.7%
  2026-W44 (2026-10-26)  n= 152  contrast= 27.0%  per100=0.180  dash= 90.1%
  2026-W45 (2026-11-02)  n= 171  contrast= 47.4%  per100=0.254  dash= 94.7%
  2026-W46 (2026-11-09)  n= 105  contrast= 51.4%  per100=0.301  dash= 99.1%
  2026-W47 (2026-11-16)  n=  19  contrast= 47.4%  per100=0.280  dash=100.0%  (partial week)

[contrast_not_by_week:model:(no finalize sighting)]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W39 (2026-09-21)  n=   2  contrast=  0.0%  per100=0.000  dash=  0.0%
  2026-W40 (2026-09-28)  n=   3  contrast=  0.0%  per100=0.000  dash=  0.0%
  2026-W41 (2026-10-05)  n=   1  contrast=  0.0%  per100=0.000  dash=100.0%
  2026-W42 (2026-10-12)  n=   5  contrast= 60.0%  per100=0.352  dash= 80.0%
  2026-W43 (2026-10-19)  n=  12  contrast= 41.7%  per100=0.214  dash= 91.7%
  2026-W44 (2026-10-26)  n=  24  contrast= 41.7%  per100=0.197  dash=100.0%
  2026-W45 (2026-11-02)  n=  37  contrast= 48.6%  per100=0.197  dash=100.0%
  2026-W46 (2026-11-09)  n=  21  contrast= 61.9%  per100=0.318  dash=100.0%

[contrast_not_by_week:model:<synthetic>]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W42 (2026-10-12)  n=   1  contrast=  0.0%  per100=0.000  dash=  0.0%

[contrast_not_by_week:model:Qwen3.6-35B-A3B-UD-Q5_K_M]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W38 (2026-09-14)  n=  10  contrast=  0.0%  per100=0.000  dash=100.0%  (partial week)
  2026-W39 (2026-09-21)  n=  41  contrast=  9.8%  per100=0.164  dash= 92.7%
  2026-W40 (2026-09-28)  n=  42  contrast=  0.0%  per100=0.000  dash= 71.4%
  2026-W41 (2026-10-05)  n=  60  contrast= 16.7%  per100=0.226  dash= 83.3%
  2026-W42 (2026-10-12)  n=  52  contrast= 21.1%  per100=0.139  dash= 92.3%
  2026-W43 (2026-10-19)  n=  41  contrast= 14.6%  per100=0.117  dash= 90.2%
  2026-W44 (2026-10-26)  n=  85  contrast= 20.0%  per100=0.159  dash= 88.2%

[contrast_not_by_week:model:Qwen3.8-27B-UD-Q4_K_XL]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W39 (2026-09-21)  n=   1  contrast=100.0%  per100=0.671  dash=100.0%
  2026-W41 (2026-10-05)  n=   1  contrast=100.0%  per100=0.671  dash=100.0%
  2026-W42 (2026-10-12)  n=   1  contrast=  0.0%  per100=0.000  dash=100.0%
  2026-W43 (2026-10-19)  n=   1  contrast=100.0%  per100=0.465  dash=100.0%
  2026-W44 (2026-10-26)  n=   3  contrast= 33.3%  per100=0.136  dash=100.0%

[contrast_not_by_week:model:Ternary-Bonsai-2-27B-PQ2_0]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W44 (2026-10-26)  n=   2  contrast=  0.0%  per100=0.000  dash=100.0%
  2026-W45 (2026-11-02)  n=   2  contrast= 50.0%  per100=0.551  dash=100.0%

[contrast_not_by_week:model:qwen3.8-flash-next-iq3_s]  replies > 40 chars; share with >=1 ', not X' | frames per 100 words | share with >=1 dash
  2026-W44 (2026-10-26)  n=  38  contrast= 34.2%  per100=0.213  dash= 86.8%
  2026-W45 (2026-11-02)  n= 132  contrast= 47.0%  per100=0.269  dash= 93.2%
  2026-W46 (2026-11-09)  n=  84  contrast= 48.8%  per100=0.294  dash= 98.8%
  2026-W47 (2026-11-16)  n=  19  contrast= 47.4%  per100=0.280  dash=100.0%  (partial week)
```

How this compares to Juniper's hand baseline (`,\s+not\s+\w` on all `chat_history_log.response` over 40 chars, weeks 10-12..11-09: 14/17/26/47/51%). The same SQL run today gives 17/17/26/47/51%. The 10-12 week differs by 3 points (14% vs 17%), probably from a different start time on the hand run; I can't confirm that. The "all" group here reads 23.7/22.2/27.0/47.4/51.4%. The extra hits in the two early weeks are the dash forms ("It matters — not just because ..."). The task asked to count those, and the plain comma regex cannot see them. Of 560 replies since 10-12, the dial and the SQL regex disagree on 11: 10 dash-form frames the regex misses, and 1 ", not only a number" that an earlier draft of the dial wrongly excluded (now fixed and tested). From 11-02 on, the two agree to within one reply per week. Dash-form frames were more common under the 35B model; Flash-Next writes the comma form.

## Current architecture

`ToneDials` had 10 structural dials (headings, lists, bold, dashes, first person, and others). The only consumer is the read-only baseline script. **No live telemetry records tone dials.** I searched for `ToneDials`, `measure_tone_dials`, `tone_dials` and `em_dash_per_100` across the repo. The only hits are this module, its tests, the script, its tests, docs, and a label comment in `orion/schemas/chat_response_feedback.py`. There is no per-turn spot to add the field to, so I did not build a new pipeline. I filed no issue: the weekly script is enough to watch this trend for now.

## Architecture touched

`orion/personality/tone_dials.py` (pure module) and the analysis script. No service, bus, schema, or env change.

## Files changed

- `orion/personality/tone_dials.py`: the `_CONTRAST_NOT_RE` shape, the two new dials, and the cut-short marker strip.
- `orion/personality/tests/test_tone_dials.py`: true positives (comma, em dash, spaced hyphen, double hyphen, en dash, emphasis, quotes, code words, bullets, two hits per line); false positives ("whether or not", "if not", "or not", "but not", "nothing", "notably", "not only ... but also", "not to mention", "not least", hyphen inside a word, the cut-short marker); pinned accepted cases ("No, not yet.", a single-backtick span); code fences ignored; independence from the dash dial.
- `scripts/analysis/measure_tone_dial_baseline.py`: `contrast_not_by_week`, a LATERAL join for the served model, the `--weekly-min-chars` flag, and the summary section.
- `scripts/analysis/tests/test_measure_tone_dial_baseline.py`: week bucketing, the 40-char filter, model groups, partial-week tags, and the summary text.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `ToneDials` has two more fields, so `DIAL_NAMES` and `as_dict()` are longer. For a reply that contains the cut-short marker, the other dials also drop the marker's words and its " - " dash (1 of 724 rows in 60 days).
- Compatibility notes: the only consumer is the baseline script.

## Env/config changes

- None. No `.env_example` changed; no local `.env` sync needed.

## Metric gate

1. **Provenance.** `_CONTRAST_NOT_RE` in `orion/personality/tone_dials.py`, run per content line in `_measure` after code, URL, list-marker and heading stripping. Input is `chat_history_log.response`, which is the text Orion sent.
2. **Independence from the dash dial.** The dash forms of the frame also add a dash, so the two share a little. Per reply over 60 days (721 replies): Pearson 0.107, Spearman 0.074. 225 replies have both, 430 have a dash and no frame, 2 have a frame and no dash, 64 have neither. Dashes are near-saturated (96-100% of replies), so the dash dial can no longer show this habit; the new dial can.
3. **Theory anchor.** The contrast frame "X, not Y" sets up a denial nobody made. It is a recognized LLM prose tic, and Juniper named it. A random sample of 12 hits (orchestrator) and 40 hits (reviewer) were all real frames.
4. **Live sanity.** Not degenerate. 68.5% of replies read zero, the maximum is 6 in one reply, and the week of 09-28 reads 0% over 45 replies (raw SQL confirms 0 hits). So the dial can return to rest.
5. **Existing mechanism.** None. The dash dial was the closest, and it is saturated.
6. **Reversibility.** Cheap. Nothing persists the dial; deleting two fields and one regex removes it.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest orion/personality/tests/test_tone_dials.py scripts/analysis/tests/test_measure_tone_dial_baseline.py -q -p no:cacheprovider
78 passed in 0.21s
git diff --check   ok
```

## Evals run

```text
The live weekly run above is the eval: a read-only pass over 60 days of chat_history_log (open_readonly_connection). There is no separate eval harness for the tone dials.
```

## Docker/build/smoke checks

```text
Not applicable: no service, image, or runtime change.
```

## Review findings fixed

Code review ran in a subagent. It found no must-fix issue.

- Finding: emphasis or quotes around "not", or around the word after it, hid real frames (", **not** free", ", not `state`"; 4 of 501 live replies).
  - Fix: the shape allows `*`/`_` around "not" and a quote, bracket, backtick or emphasis after it. The word check no longer uses `\b`, so "_not_" counts.
  - Evidence: 7 new true-positive tests.
- Finding: the first and last weeks are partial and not marked.
  - Fix: `partial` flag per week, and `(partial week)` in the summary.
  - Evidence: `test_contrast_not_by_week_marks_edge_weeks_partial`.
- Finding: the "not only ... but" lookahead scanned to the sentence end, so it was quadratic on a pathological line (13.8 s for 10,000 repeats).
  - Fix: bounded to 200 characters. The same input now takes 0.11 s.
- Finding: interjection commas ("No, not yet.") and single-backtick spans count.
  - Fix: kept on purpose (a minimum lead-in would also cut real tics like "Seat, not costume."); pinned in tests and documented in the regex comment.
- Finding: `model:<synthetic>` group meaning unclear; 40-char filter counts raw characters.
  - Fix: documented in the `contrast_not_by_week` docstring. The raw-character filter stays, to match Juniper's SQL baseline.
- Not fixed (nit): each reply is measured twice in the script. About 700 rows, so the cost is small; kept for a pure standalone function.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: the shape counts a few non-contrast commas ("No, not yet.", "Whether it works, not sure."). Mitigation: none seen in 52 sampled live hits; pinned in tests so any change is visible.
- Severity: low. Concern: Flash-Next's own upward drift (34% to 47-49%) has no explanation yet. Mitigation: the per-model weekly split makes it re-checkable.
- Issues opened: none. Issues closed: none.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
