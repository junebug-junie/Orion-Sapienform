Fixes #3554
Fixes #3541

## Summary

- The private-memory check at Orion's edge is stricter. Before, it refused only a sentence copied word for word from a core page or a stored memory. Now it also refuses text that shares a run of 8 or more words in a row with such a sentence. So a copied sentence with one end word changed, a clause cut off, or "a" swapped for "the" is caught. (`orion/memory/mesh_edge.py`)
- The run length is a setting, `ORION_MESH_EDGE_SHINGLE_WORDS`. It ships on at 8; 0 turns it off. A bad value keeps 8, so a typo cannot switch the check off.
- Prompts that Orion sends to Claude in a consult now pass the same check. Juniper decided on 2026-11-18 that consult briefs are outside the mesh edge (#3541). A sentence that trips the check is replaced with `[withheld: private memory]` before Claude sees it, and Hub chat says how many passages were withheld. (`services/orion-hub/scripts/consult_loop.py`)
- Fail-closed holds on every seam. If the check cannot load the guarded sentences, a GitHub write or email is refused, as before, and a consult send is held and retried on the next tick. Hub chat says so once.
- A new eval measures the threshold on made-up paraphrases and, read-only, on real public PR and issue text. (`orion/memory/evals/run_mesh_edge_shingle_eval.py`)

## Outcome moved

Near-copies of private sentences can no longer reach GitHub, non-Juniper email, or Claude. Before this patch, a one-word edit got through all three seams, and consult prompts were not checked at all.

Measured:

| run length | made-up near-copies refused | made-up rewordings refused | made-up unrelated refused | real public docs refused (of 866) |
|---|---|---|---|---|
| verbatim only | 1/14 | 0/7 | 0/14 | 1 |
| 5 | 14/14 | 0/7 | 1/14 | 29 |
| 6 | 14/14 | 0/7 | 0/14 | 6 |
| 7 | 14/14 | 0/7 | 0/14 | 1 |
| **8 (shipped)** | **13/14** | 0/7 | 0/14 | **1** |
| 9-12 | 12-13/14 | 0/7 | 0/14 | 1 |

- The 866 real docs are the last 300 PRs and 300 issues with their comments, fetched read-only with `gh`.
- At 8, the shingle check refuses no real public doc that the verbatim check does not already refuse. That gives a false-positive rate of 0 of 866.
- The one doc that is refused at every setting is an older PR body that already quotes a stored stance and a person fact. This is a real past leak, not a false positive. Its id is withheld from this public report and was given to Juniper directly.
- The one near-copy missed at 8 shares only 7 words in a row. 8 was chosen over 7 to keep one word of margin above the 6-word level, where real refusals start.
- Real rewordings are not caught at any setting. That is the stated limit; the follow-up is #3699.

Speed, on the live guard (1,348 guarded rows, 1,808 sentences, 13,588 eight-word runs):

- Loading the rows from Postgres takes about 65-75 ms.
- Building the index takes about 80-150 ms, compared with about 40-80 ms for verbatim only.
- One check takes 1.7 ms at the median, 18 ms at p95 and 33 ms at the maximum. The largest doc has 6,432 words. Adding the shingle check made a check about 0.1 ms slower at the median.
- Hub and notify cache the built index for 300 s, so it is not rebuilt per call. The FCC hook is a new process for each Bash call, so it loads and builds the index once per GitHub write, about 0.2 s. A non-write call never loads the index.

## Current architecture

`orion/memory/mesh_edge.py` loaded every page:juniper and page:anchor version, plus all episode_memory, stance and person_fact text, live or retired, through `ORION_MESH_EDGE_PG_DSN` (or the fallbacks). It matched only whole normalized sentences of 6 or more words. Three writers called it: the FCC `gh`/`curl` PreToolUse hook, Hub's PR commenter, and notify email to anyone but Juniper. Consult prompts went to Claude through `ConsultLoop._send_claude` -> `room_claude_relay.invite` and were not checked.

## Architecture touched

- `orion/memory/mesh_edge.py`:
  - shingle index (`MeshEdgeGuard.shingles`, `shingle_n`) and `shingle_words()` config;
  - light function-word folding: a/an/the and which/that each count as one word, with no stemming;
  - URL, path and code tokens in a guarded sentence split its runs. A single slash does not split, so "she/her" still counts;
  - invisible characters are stripped;
  - `redact()` blanks each whole outgoing sentence that holds a match.
- `orion/fcc/mesh_edge_hook.py`: the refusal message also names near-copy runs.
- `services/orion-hub/scripts/consult_loop.py`: the check runs in `_send_claude`. That covers every Claude-bound send: the first send, the budget-hold resend, the lost-reply resend and the gate-retry correction. New hold state `mesh_edge_hold:`.

**Why redact rather than refuse for consults:** by the time a consult prompt is sent, Orion's turn has already run as a durable GPU run, and no author is waiting to rewrite it. In the FCC hook, the refusal goes back to Orion, who can rewrite. Here a refusal would only kill the consult. The rest of the prompt is the plumbing question that Claude needs. Withholding the whole sentence instead of only the matched words keeps near-miss neighbours from going out in clear. If any match survives redaction, the consult fails and nothing is sent. Orion's stored turn keeps the original text, because Hub is inside the mesh.

## Files changed

- `orion/memory/mesh_edge.py`: the shingle check, folding, pointer-token split, `redact()`, and the config key.
- `orion/fcc/mesh_edge_hook.py`: the refusal message.
- `services/orion-hub/scripts/consult_loop.py`: the check on Claude-bound prompts; hold, refuse and redact states; one-time Hub notices.
- `orion/memory/evals/run_mesh_edge_shingle_eval.py`, `orion/memory/evals/mesh_edge_paraphrase_set.py`: the eval and the made-up test set.
- `orion/memory/tests/test_mesh_edge.py`, `orion/fcc/tests/test_mesh_edge_hook.py`, `services/orion-hub/tests/test_consult_loop.py`: tests.
- `services/orion-hub/.env_example`, `services/orion-notify/.env_example`, `services/orion-harness-governor/.env_example`: `ORION_MESH_EDGE_SHINGLE_WORDS=8`.
- `.github/workflows/lived-memory-recall-tests.yml`: runs the consult tests and the fixture eval.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: the mesh-boundary sections now say what shipped.
- `orion/temporal_self/tests/test_now_privacy_boundary.py`: docstring now points at the runtime check.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Mesh-edge refusals now include near-copies.
  - Consult prompts to Claude are checked and redacted.
  - A consult can hold in `mesh_edge_hold:unavailable` or fail with `mesh_edge_refused`.
- Compatibility notes: `LeakHit` gained `kind` (default `"verbatim"`). Existing callers that use `find()` and `check()` are unchanged.

## Env/config changes

- Added keys: `ORION_MESH_EDGE_SHINGLE_WORDS=8` in the orion-hub, orion-notify and orion-harness-governor `.env_example` files.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced: yes. I ran `python scripts/sync_local_env_from_example.py --all-keys orion-hub orion-notify orion-harness-governor`, and the key is in all three live `.env` files. The sync also added the missing `HEARTBEAT_INTERVAL_SEC=10` to orion-notify.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/memory/tests/test_mesh_edge.py orion/fcc/tests/test_mesh_edge_hook.py orion/temporal_self/tests/test_now_privacy_boundary.py  -> 91 passed
(services/orion-hub) pytest tests/test_consult_loop.py tests/test_session_roundup_loop.py                                       -> 65 passed, 24 skipped
(services/orion-notify) pytest tests                                                                                            -> 72 passed
pytest tests/test_no_personal_terms.py                                                                                         -> passed
python scripts/check_env_key_single_source.py                                                                                  -> OK
git diff --check                                                                                                               -> clean
```

## Evals run

```text
python orion/memory/evals/run_mesh_edge_shingle_eval.py                      -> fixture pass at n=8 (table above)
python orion/memory/evals/run_mesh_edge_shingle_eval.py --live --limit 300   -> read-only Postgres + gh; numbers above (ids only)
```

## Docker/build/smoke checks

```text
Not run. Not deployed (out of scope by instruction). The change is library code plus one Hub module, read at boot.
Live check after deploy: UNVERIFIED. Two checks are needed:
- In an FCC turn, a gh write with a one-word edit of a guarded sentence is blocked with "repeats a run of 8+ words".
- Hub logs consult_mesh_edge_redacted (kinds only) when a consult prompt holds a match.
```

## Review findings fixed

- Finding: a near-miss left the guarded words next to the match in clear in a redacted consult prompt.
  - Fix: `redact()` now withholds the whole outgoing sentence.
  - Evidence: `test_redact_does_not_leave_near_miss_neighbours_in_clear`.
- Finding: a held consult had no notice, and it logged on every tick.
  - Fix: one Hub chat notice and one log line on the first hold.
  - Evidence: `test_any_guard_error_holds_the_send_with_one_hub_notice`.
- Finding: only `MeshEdgeUnavailable` was caught. Any other guard error could leave the old prompt stored on the gate-retry path.
  - Fix: any guard error holds the send with the right prompt stored.
  - Evidence: same test (the loader raises RuntimeError).
- Finding: no test covered the lost-reply resend or the budget-hold resend against a guard hit.
  - Fix: added both tests.
  - Evidence: `test_the_lost_reply_resend_re_checks_against_a_newer_guard`, `test_the_budget_hold_resend_is_redacted`.
- Finding: the pointer-token rule split prose like "she/her" and "24/7".
  - Fix: the rule is narrowed to URLs, 2+ slashes, backslashes, backticks, snake_case and dotted names.
  - Evidence: `test_a_single_slash_in_a_guarded_sentence_does_not_split_its_runs`.
- Finding: the verbatim match did not fold articles.
  - Fix: phrases are compared in folded form.
  - Evidence: `test_articles_fold_in_the_verbatim_match_too`.
- Finding: punctuation-only tokens split a withheld run into two markers.
  - Fix: whole-sentence redaction keeps one marker.
  - Evidence: `test_redact_keeps_one_marker_across_punctuation_tokens`.
- Finding: zero-width characters split words.
  - Fix: they are stripped in `normalize`.
  - Evidence: `test_zero_width_characters_do_not_split_words`.
- Finding: the PR quoted no build-time or index-size numbers.
  - Fix: they are in "Outcome moved" above.
- Finding: one word changed mid-sentence in a guarded sentence under about 15 words beats the check. This is a design limit, not fixed.
  - Fix: a test states the current behaviour (`test_known_gap_a_mid_sentence_swap_in_a_short_sentence_passes`), and the limit is ticketed as #3699.
- CI finding: the fixture used a real first name, which the personal-terms gate refused.
  - Fix: renamed it to a made-up name.

## Restart required

Hub, notify and the harness governor read the code at boot. FCC turns read it per call from the image.

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-notify up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Run these only from the clean primary `main` checkout after merge. The post-pull rebuild normally does it.

## Risks / concerns

- Severity: medium.
  - Concern: the check still passes real rewording, and a one-word mid-sentence edit of a short guarded sentence.
  - Mitigation: stated, tested, and ticketed as #3699.
- Severity: low.
  - Concern: a consult held because the guard is down blocks the one-at-a-time consult queue until the guard loads.
  - Mitigation: this is fail-closed by design. Hub chat says so once.
- Severity: low.
  - Concern: whole-sentence redaction can remove a useful clause next to a match.
  - Mitigation: Hub chat shows the count. The original stays in Orion's stored turn.
- Severity: info.
  - Concern: one older public PR body already holds guarded text, so it is a past leak.
  - Mitigation: reported to Juniper by id, outside this public report.
- Issues: opened #3699. Closes #3554 and #3541.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3698

🤖 Generated with [Claude Code](https://claude.com/claude-code)
