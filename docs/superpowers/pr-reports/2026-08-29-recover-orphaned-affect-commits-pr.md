# Recover three commits that were never delivered to main

## Summary

- Three commits were pushed to `feat/affect-read-vision-backend` **after** its PR
  (#1903) had already merged on 2026-08-26. No PR ever covered them, so 860 lines
  across 21 files sat on the branch and never reached `main`.
- The code is not stale scratch work, but the "live on circe for two days" framing
  in this report's first draft was too broad and is corrected below: **only the
  circe half is live.** The athena half (`orion-whisper-tts`, `orion-hub`) has never
  run, and deploying half of it caused a live regression -- see "Live regression".
- This was found while trying to move circe's checkout onto `main` to deploy the
  power-intent loop. Doing so would have silently reverted a live fix.
- This PR merges current `origin/main` into the branch (clean, zero conflicts) and
  recovers the work. No new code is written here.
- Verified against a real baseline: the branch introduces **zero** new test
  failures.

## Outcome moved

`main` regains a fix it had silently lost, circe becomes safe to move onto `main`
(which unblocks the power-intent deploy), and a live regression caused by the split
deploy gets closed.

Concretely, the fix being recovered stops **fabricated speech being treated as
Juniper's words**. On 2026-08-26 a clip measured at peak=114 / rms=8.68 (-49 dBFS)
passed the amplitude-only gate; Whisper returned a fully-formed sentence about
Egyptians plus a repetition-looped "Tired, tired, tired" on a turn where Juniper had
actually said "I'm feeling really tired." AffectGPT then anchored its affect read on
the invented sentence. Amplitude alone cannot catch this — 0.15% of full scale is
still "loud enough" numerically. The recovered gate uses Whisper's own per-segment
`no_speech_prob` instead.

## Current architecture

`feat/affect-read-vision-backend` merged as PR #1903 at 2026-08-26 18:16 MDT.
Three more commits landed on the same branch afterwards:

```
da994913c  2026-08-26 22:03 MDT  feat(affect): one recording, not two -- merge the mic press into the read
f977dd9a3  2026-08-26 22:03 MDT  docs: PR report for the recording merge
e9b5d3ce8  2026-08-26 23:32 MDT  fix(stt): block fabricated transcripts when there is no speech
```

Nothing picked them up. Evidence that they were genuinely absent from `main`:

```text
$ git merge-base --is-ancestor e9b5d3ce8 origin/main   -> NO (all three)
$ git show origin/main:services/orion-affectgpt-worker/app/settings.py \
    | grep -c MAX_NO_SPEECH_PROB
0
$ gh pr list --state open        # only #1940 and #1212 -- nothing for this branch
$ git ls-remote --heads origin | grep affect
e9b5d3ce84d0bc81702f89cae541275302dfcf8c refs/heads/feat/affect-read-vision-backend
```

And evidence it is live on circe right now:

```text
$ ssh circe -- docker exec orion-circe-affectgpt-worker printenv | grep NO_SPEECH
AFFECTGPT_TRANSCRIBE_MAX_NO_SPEECH_PROB=0.6
$ ssh circe -- docker exec orion-circe-affectgpt-worker \
    grep -c MAX_NO_SPEECH_PROB /app/app/settings.py
1
```

This is the same failure mode recorded previously as "a commit isn't delivered until
a PR covers it" — pushing to an already-merged branch. It recurred, undetected, for
three days.

## Architecture touched

None. This is a recovery merge; the only new content is this report.

## Files changed

- `docs/superpowers/pr-reports/2026-08-29-recover-orphaned-affect-commits-pr.md`:
  this report.
- Everything else in the diff is the three pre-existing commits, unmodified.

## Schema / bus / API changes

- Added: `orion/schemas/vision.py` gains fields from the original commits.
- Removed: none.
- Renamed: none.
- Behavior changed: none introduced by this PR.
- Compatibility notes: the branch merges into `origin/main` with zero conflicts
  (`git merge-tree --write-tree` reported none before the merge was performed).

## Env/config changes

- Added keys (from the recovered commits, not new here):
  - `AFFECTGPT_TRANSCRIBE_MAX_NO_SPEECH_PROB=0.6` (orion-affectgpt-worker)
  - keys in `services/orion-whisper-tts/.env_example`
- `.env_example` updated: yes, in the original commits.
- local `.env` synced: `python3 scripts/sync_local_env_from_example.py` reports only
  pre-existing "Diverged" keys, no additions or changes required on athena.
- Skipped keys requiring operator action: none.
- Note: circe already carries `AFFECTGPT_TRANSCRIBE_MAX_NO_SPEECH_PROB=0.6` in its
  live `.env` — it was hand-added there when the code was deployed.

## Tests run

Each service's suite runs from its own directory; they share an `app` package name
and collide if collected together.

```text
services/orion-affectgpt-worker        40 passed
services/orion-juniper-affective-state 92 passed
services/orion-whisper-tts             2 failed, 52 passed
services/orion-hub                     64 failed, 1742 passed, 4 skipped
```

The failures are **pre-existing on main**, not introduced here. Baseline from the
same suites on `origin/main` @ 47c8db0c4:

```text
services/orion-whisper-tts             2 failed, 48 passed
services/orion-hub                     64 failed, 1739 passed, 4 skipped
```

Counts alone are not evidence, so the failure *sets* were diffed by name:

```text
hub: failures only on branch  -> 1  (see flakiness note)
hub: failures only on main    -> 0
tts: failures only on branch  -> 0
```

The single set difference was
`test_substrate_mutation_manual_route_routing.py::test_routing_apply_succeeds_for_auto_promote_and_can_rollback`.
It is flaky, and flaky **on main**, not a branch regression — the file was run three
times on each tree:

```text
MAIN   run1: 7 passed   run2: 1 failed, 6 passed   run3: 7 passed
BRANCH run1: 7 passed   run2: 7 passed             run3: 7 passed
```

Main's own two full-suite runs also disagreed with each other (63 vs 64 failures,
different routing test each time). The branch therefore adds no failure and adds 3
passing hub tests and 4 passing whisper-tts tests.

The pre-existing failures are largely environment-dependent (no local Postgres on
:5432 — `hub_presence_write_failed ... Connection refused`). Out of scope here.

## Evals run

```text
None. No eval harness exists for the touched services; the recovered commits ship
their own unit tests (test_transcribe.py +80, test_stt_engine.py +53,
test_capture_and_assess.py +103, test_vision_retina_clip_rpc.py +85).
```

## Docker/build/smoke checks

```text
Not rebuilt by this PR. The recovered code is already running on circe and has been
for two days, which is the strongest available runtime evidence for the circe-side
services (affectgpt-worker, juniper-affective-state, vision-retina).
```

## Live regression this closes

Verified live 2026-08-29, and it is the reason this cannot just sit on a branch.

Circe runs the branch (`want_audio=not use_vision`, `app/main.py:372`) with
`AFFECT_BACKEND=vision`, so **no audio is recorded** on the vision path. The caller
subtitle that was designed to replace it is produced by Hub -- which runs `main` and
does not send one. Half the change is deployed, so the affect read currently gets no
speech context at all.

```text
$ docker exec orion-athena-hub grep -c 'subtitle=transcript' /app/scripts/websocket_handler.py
0
$ docker exec orion-athena-whisper-tts grep -c no_speech_prob /app/app/stt.py
0
$ docker exec orion-athena-whisper-tts printenv | grep NO_SPEECH
STT_MAX_NO_SPEECH_PROB=0.6          # config truth without runtime truth
```

```sql
select subtitle_source, count(*), max(created_at) from juniper_multimodal_affect_log group by 1;
 none        | 6 | 2026-08-27 05:15:24+00   -- after the merge, backend=vision
 transcribed | 4 | 2026-08-26 23:58:17+00   -- before
 (null)      | 1 | 2026-08-26 22:33:15+00
```

`subtitle_source = 'caller'` has **never** occurred in the table's entire history:
the transcript-threading half has never once executed. The state since 2026-08-26
evening is worse than either the before or the after -- previously the read got a
transcript (sometimes a fabricated one); now it gets nothing.

Note on verification method: the two services name the filter differently
(`keep_only_speech_segments` in orion-affectgpt-worker, `_keep_only_speech_segments`
in orion-whisper-tts). Grepping the underscore-prefixed name against circe returns a
false zero. Circe's gate was confirmed present by name and by threshold:

```text
$ ssh circe -- docker exec orion-circe-affectgpt-worker \
    grep -nE 'def .*speech|no_speech_prob' /app/app/transcribe.py
112:def keep_only_speech_segments(
141:            prob = float(seg.get("no_speech_prob") or 0.0)
```

## Review findings fixed

Review ran on the recovered diff (these commits never passed a review gate, which is
the underlying defect). Findings that changed code:

- Finding: a non-empty `segments` list whose entries are not dicts falls through the
  loop, collects no probs, and returns `""` -- silently discarding a real transcript
  while `meta` reports `no_speech_filter: "applied", segments_total: 0`. The gate
  would be the last thing suspected, because its own telemetry says it ran fine.
  Realistic trigger: swapping to faster-whisper, whose segments are objects, not
  dicts. Every voice turn would transcribe to empty.
  - Fix: both copies of the filter now return `raw_text` with
    `{"no_speech_filter": "unavailable", "reason": "no_parseable_segments"}` when no
    segment parsed -- the same contract the no-segments branch already had.
  - Evidence: `test_segments_present_but_all_unparseable_keeps_the_raw_text` in both
    services. Mutation-checked: reverting the fix fails exactly 1 test in each.

- Finding: `chat_turn_affect.fire` forwards `subtitle` unconditionally, but with
  `AFFECT_CHAT_TURN_SCOPE=all` it also fires on typed turns, where `transcript` is
  the typed message body. The VL prompt renders it as *"the person said this around
  the time these frames were captured"* -- asserting she spoke words she silently
  typed, to a model reading her face for affect.
  - Fix: `fire()` drops the subtitle on a non-voice turn. Done in `fire()` rather
    than at the call site so it covers every caller, present and future.
  - Evidence: `test_typed_turn_never_forwards_the_message_as_a_spoken_subtitle`,
    plus `test_voice_turn_still_forwards_the_subtitle` to stop the fix over-reaching.
    Mutation-checked: reverting fails exactly 1 test.

Findings deliberately **not** actioned here, because they change behavior and are
Juniper's call -- see Risks:

- Unknown `no_speech_prob` is treated as `0.0` ("definitely speech"), i.e. the gate
  fails open. Correct for orion-whisper-tts (dropping her real speech is worse than
  a hallucination she can see and correct); arguably wrong for orion-affectgpt-worker,
  where a fabricated transcript becomes the grounding for a read she never sees.
- `want_audio=not use_vision` means the affectgpt rollback path still records,
  uploads, and cross-host-fetches her voice, then never transcribes it (because
  `resolve_subtitle` returns early on a caller subtitle). Dead I/O plus a real
  privacy surface.

Verified clean by review: env reachability for both new keys, schema registry needs
no change, `clip_capture.py`'s `want_audio=False` branch loses no data on the vision
path, and there is no call site that bypasses the filter.

## Restart required

Merging this PR alone changes nothing at runtime. Deploying it does:

```bash
# circe (already running this code -- rebuild only to move it onto main)
#   affectgpt-worker, juniper-affective-state, vision-retina

# athena -- NOTE: these services have NOT run this code before
scripts/safe_docker_build.sh orion-whisper-tts up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
  Concern: circe-side services have two days of live runtime on this code, but
  **athena's `orion-whisper-tts` and `orion-hub` have never run it**. Merging plus
  deploying introduces genuinely new code to athena, whereas for circe it is a
  no-op. These two deployments deserve to be treated as a normal change, not as
  "restoring what was already running."
  Mitigation: deploy circe first (no-op there), then athena separately, and watch
  STT output for over-rejection — the new gate can only make the system reject more
  audio, never less. Athena's whisper-tts already has `STT_MAX_NO_SPEECH_PROB=0.6`
  set in its live `.env` with no code to read it, so anyone checking env parity today
  gets a false "the gate is on".

- Severity: medium
  Concern: two review findings change behavior and were left for Juniper — whether
  the affectgpt path should fail *closed* on an unknown `no_speech_prob`, and whether
  `want_audio` should also be gated on the presence of a caller subtitle so her voice
  stops being recorded and shipped cross-host for no consumer.
  Mitigation: both are small, isolated follow-ups; neither blocks this recovery.

- Severity: medium
  Concern: deploy order is load-bearing. `RetinaClipCaptureRequestPayload` sets
  `extra="forbid"` and validates before the camera-identity check, so a retina that
  has not been updated answers `invalid_request` to every affect capture. Carbon's
  retina must go before or with circe's affective-state; rollback has the same hazard
  reversed.

- Severity: low
  Concern: the gate is a fixed threshold (0.6) on Whisper's `no_speech_prob`. It is
  a knob, and a knob is not a finding. No calibration data is presented for 0.6.
  Mitigation: it is a single env key and trivially reversible. Worth revisiting with
  real data on how often it now rejects, which nothing currently records.

- Severity: low
  Concern: the process failure that caused this is unfixed. Nothing detects a commit
  pushed to an already-merged branch, so this can recur silently — it already has,
  and it took three days and an unrelated deploy attempt to notice.
  Mitigation: a deterministic gate is the right fix per CLAUDE.md ("if Juniper has
  to repeat a rule twice, turn it into a script"). Not built here; filed as
  follow-up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1945
