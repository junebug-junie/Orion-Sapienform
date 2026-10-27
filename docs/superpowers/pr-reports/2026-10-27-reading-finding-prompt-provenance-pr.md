## Summary

- The prompt Orion writes "a read changed me" messages from no longer tells Orion "Nobody asked you to read it" or "it changed you". It also no longer asks "why that change matters to you".
- The prompt now shows how the read started, built from the reading queue's own rows. The chain is listed oldest first: who or what queued the first read, then each link Orion chose to follow, ending at this read.
- A request made during a chat turn is Orion's own tool call, so the prompt says "you queued". Only an operator request says "Juniper asked you". The reason text is labelled as Orion's note or as Juniper's.
- A sizing line ("one link is one link") keeps a small change from being made bigger than the record.
- If the chain lookup fails, the message still goes out, with "How this read started was not recorded."

## Outcome moved

On 2026-10-26 Orion sent Juniper a message about a read that said the link formed "not because a turn asked me to, not because a process scheduled it". The live queue shows the real chain:
1. On 10-19, the daily world-pulse digest put the RTX Spark N1X article in the queue.
2. On 10-24, Orion chose a link from it to the GB10 Hot Chips article.
3. On 10-25, Orion chose a link from that to the DGX Spark article.

The prompt said "Nobody asked you to read it" and gave Orion no record to check against, so Orion's claim was false. Now Orion gets that chain as numbered lines. The change itself was one link between "GB10" and "GB10 (Blackwell Arm)", and the old prompt gave no permission to call a change small.

## Current architecture

`orion/world_pulse_read/finding_prompt.py` builds the prompt from a `ReadingFindingBundleV1`: url, title, Orion's summary, the article summary, and the findings. Nothing about where the read came from reached the prompt, but the opening sentence stated it anyway.

Where reads come from:
- Every row in `world_pulse_read_seed` carries `request_json.parent_request_id` when Stage 2 asked for it. Stage 2 asks through `need_stage1_urls`, and `_hop_allowlist` checks each URL.
- A chain's first read is one of four things: a digest item or finding, a chat-turn tool call, an operator request, or a curiosity request.

## Architecture touched

- Hub only. No bus, channel, or registry change, because the bundle is Hub-internal by its own docstring.
- The new read-only recursive SELECT `ORIGIN_SQL` walks `parent_request_id` to `request_id`. It is capped at depth 12.

## Files changed

- `orion/world_pulse_read/origin.py`: new. `ORIGIN_SQL`, `origins_from_rows` (pure), and `read_origins`.
- `orion/schemas/reading_finding.py`: adds `ReadingOriginStepV1` and the `ReadingFindingBundleV1.origin` field (additive, defaults to empty).
- `orion/world_pulse_read/finding_prompt.py`: adds `origin_lines` and rewrites the opening and instruction wording.
- `services/orion-hub/scripts/reading_finding_outreach.py`: looks up origins only for reads that have a message to send. A failed lookup logs and degrades.
- `services/orion-hub/evals/run_reading_finding_replay.py`: the replay attaches and prints the chain.
- Tests: `orion/world_pulse_read/tests/test_finding_prompt_origin.py` (new), `services/orion-hub/tests/test_reading_finding_outreach.py`, `services/orion-hub/tests/test_reading_postgres.py` (real-SQL chain), and `services/orion-hub/evals/test_reading_finding_eval.py`.

- `services/orion-sql-writer/tests/test_vision_osnet_shadow.py`: an unrelated time bomb that blocked this PR's CI. Its rows were stamped from a fixed 2026-10-16 date, but the live eval looks back 8 days from the real clock. Since 10-24 the two Postgres tests found 0 pairs. Rows are now stamped from the real clock. Verified: 16 passed on a throwaway Postgres 16 with CI's migrations.

## Schema / bus / API changes

- Added: `ReadingOriginStepV1` and `ReadingFindingBundleV1.origin` (default `[]`).
- Removed: none.
- Renamed: none.
- Behavior changed: the prompt text.
- Compatibility notes: the field is additive, and producer and consumer are the same process. Bundle dedupe (`bundle_key`) does not include origin, so no message is sent twice.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/world_pulse_read/tests services/orion-hub/evals/test_reading_finding_eval.py   -> 187 passed
RUN_READING_POSTGRES=1 pytest services/orion-hub/tests/test_reading_finding_outreach.py \
  tests/test_reading_postgres.py tests/test_endogenous_outreach.py                          -> 303 passed
orion-reading-tests.yml step 1 subset (ingress, world_pulse_read_*, introspect, outreach)  -> 1088 passed, 86 skipped (before the eval fake was fixed: 1 fail, then fixed)
```

## Evals run

```text
run_reading_finding_replay.py --days 3 (live Postgres, read-only):
reads=74 findings={'link_made': 16} messages=11; every message carries its chain. The DGX Spark read renders
  1. Your daily world-pulse news run put .../nvidia-rtx-spark-n1x-launch-overview... in your reading queue.
  2. After that read, you chose to read .../nvidia-outlines-gb10-soc-architecture-at-hot-chips-2025/ next (...)
  3. After that read, you chose to read .../the-nvidia-dgx-spark-is-a-tiny-128gb... next (...). This is the read below.
which matches the chain traced by hand from world_pulse_read_seed.
```

Orion's reply to the new prompt is UNVERIFIED. The model is not run here, so whether Orion now PASSes on a one-link change, or describes it at its real size, is seen only after deploy.

## Docker/build/smoke checks

```text
No image, dependency or compose change. Not built.
```

## Review findings fixed

- Finding: `requested_by="juniper"` is hard-coded for every chat-turn `recommend_reading` call, so "Juniper asked you in chat" put Orion's own choice and note in Juniper's mouth.
  - Fix: chat-turn requests now say "During a chat with Juniper, you queued X (your note: ...)". "Juniper asked you" is used for operator requests only.
  - Evidence: `test_starts_name_who_queued_it_and_whose_note_it_is`. In the live replay, the PR #2751 chain shows Orion's own note.
- Finding: an origin query failure would fail the whole tick, so no reading messages would go out.
  - Fix: the lookup is wrapped in try/except. It logs `reading_finding_origin_unavailable` and falls back to "not recorded".
  - Evidence: `test_a_failed_origin_lookup_still_sends_the_message`.
- Finding: "followed its link" overstated the source. The hop allowlist also admits pages that Stage 2 fetched.
  - Fix: the line now reads "you chose to read X next (a link you came across while working on it)".
  - Evidence: updated chain tests.
- Finding: a chain with a missing parent (2 live orphans) rendered a hop as a start, with requester fields inherited from the parent.
  - Fix: the first step's `parent_request_id` is carried. A missing start renders as "This continues a chain from an earlier read whose record is missing."
  - Evidence: `test_a_chain_whose_start_is_missing_is_not_rendered_as_a_request`.
- Finding (info): `why_now` can be up to 4000 characters.
  - Fix: clipped to 200 characters.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run this from the clean primary main checkout after merge. The post-pull rebuild covers it.

## Risks / concerns

- Severity: low.
  - Concern: "Your daily world-pulse news run" describes the observed daily 12:00 UTC cadence. `WORLD_PULSE_SCHEDULE_ENABLED=false` in the service, so the trigger lives somewhere else.
  - Mitigation: the wording names no mechanism.
- Severity: low.
  - Concern: a read that Orion also tried to reach by a refused `already_queued` hop shows only its original start.
  - Mitigation: the chain is still true, just incomplete.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2913

🤖 Generated with [Claude Code](https://claude.com/claude-code)
