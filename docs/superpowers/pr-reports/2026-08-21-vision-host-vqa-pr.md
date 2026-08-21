# PR report: vision-host VQA (`kind=vlm`) — real vision-language Q&A

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1791
Branch: `feat/vision-host-vqa`

## Summary

- P1's "look() as a verb" / active-vision primitive from `docs/superpowers/specs/2026-08-12-perception-frontier-design.md` was structurally inert: `config/vision_profiles.yaml`'s `vlm_vqa` profile (`task_type=vqa` already routed to it) hit `runner.py`'s generic "kind not implemented yet" fallback — contract-only, zero real inference, same "config exists is not proof" gap this repo's own P0 perception work found for `retina_track`.
- Implements `_run_vlm_vqa()`: same VLM prompt→generate→decode mechanics `_run_caption_frame` already uses (BLIP's "conditional generation" *is* text-conditioned image description — VQA and captioning are the same model call with a different prompt, not a separate architecture), but with a caller-supplied `request.question` instead of the fixed `CAPTION_PROMPT`.
- Enabled `vlm_vqa` (`enabled: true`, `warm_on_start: false` — lazy-load only) after live-checking real VRAM headroom (not assumed): ~4.2GB free on the P4 serving this host at the time.
- Three real bugs found and fixed via live testing against the actual model on real camera frames (not assumed, not just unit-tested against mocks) — see "Outcome moved" below.
- Code review found one additional gap (a second, separate kind-allowlist gate in `warm_profiles()` that would have silently kept the warm path dead even if `warm_on_start` were flipped later) — fixed in a follow-up commit with its own regression test.
- Also discovered and explicitly NOT fixed (separate, pre-existing, unrelated issues — see "Risks / concerns"): `orion-vision-frame-router` has been stopped for ~7h, and the scheduler's VRAM reserve/floor settings currently block even the pre-existing `caption_frame` task via the HTTP endpoint on this specific P4.

## Outcome moved

Orion can now ask a real, specific question about the current camera frame and get a real answer from a real vision-language model — not just a fixed, passive caption. This is the first real "eyes as something invoked, not just a servant that describes things" capability from the design doc's near-term `look()` unlock.

Three real defects found and fixed via live testing (real BLIP-base model, real frames, real GPU — not mocks):

1. **Case-sensitive prefix strip silently failed on a lowercased echo.** `generated_text.replace(prompt, "")` never caught it when BLIP lowercased its whole output regardless of input casing — confirmed live: question `"What color is the door?"` (capital W) came back prefixed with `"what color is the door?..."` (lowercase), sailing straight through the old exact-match replace. New `strip_echoed_prompt_prefix()` helper does this case-insensitively; fixed in both `_run_caption_frame` and `_run_vlm_vqa` (same bug, same file).
2. **A punctuation-only response (`"?"`) passed `sanitize_answer` as a valid answer.** The token filter checked `t.strip()` (whitespace only) instead of stripping punctuation too, so a token that becomes empty after stripping punctuation still counted as real content — a 1-token, 0%-stoplist "answer" survived every check.
3. **Obvious repetition-loop garbage** (`"| by person | cci | cci | cci | cci | cci | cci |"`) wasn't caught by the stoplist (none of those tokens are topic-specific slop words) — added a shared repetition-dominance check (single token >40% of a >=4-token response) used by both sanitizers.

## Current architecture

`VisionRunner._run_profile()` dispatches by `ProfileDef.kind` to real inference methods for `embedding`/`detect_open_vocab`/`caption_frame`; everything else fell through to a "contract-only, kind not implemented yet" placeholder. `task_routing: {vqa: vlm_vqa}` in `config/vision_profiles.yaml` already existed and correctly resolved `task_type=vqa` to the `vlm_vqa` profile — the profile itself (`kind: vlm`) just had no execution behind it, and was `enabled: false`.

## Architecture touched

- `services/orion-vision-host/app/runner.py`: new `_run_vlm_vqa()`, wired into `_run_profile`'s `kind == "vlm"` dispatch and `_warm_profile_backend`'s (now correctly reachable) `kind == "vlm"` branch. `_run_caption_frame` also touched (shared echo-strip fix).
- `orion/vision/caption_echo.py`: `is_caption_prompt_echo()` gained an optional `prompt` kwarg (default `CAPTION_PROMPT`, every pre-existing caller unaffected — confirmed via the one other real caller, `orion-vision-window/app/projection.py`, which calls it positionally); new `strip_echoed_prompt_prefix()`.
- `services/orion-vision-host/app/caption_sanitize.py`: shared `_tokenize()`/`_degenerate_reason()` helpers, new `sanitize_answer()`, repetition-dominance check added to both sanitizers.
- `config/vision_profiles.yaml`: `vlm_vqa.enabled` false → true.
- `services/orion-vision-host/.env_example` / `.env`: `vlm_vqa` added to `VISION_ENABLED_PROFILES` — the actual live gate for direct (non-pipeline) task dispatch, distinct from the YAML `enabled:` flag (which only gates pipeline-step filtering and the warm loop).

## Files changed

- `services/orion-vision-host/app/runner.py`: `_run_vlm_vqa()`, `kind=="vlm"` dispatch + warm branch, shared echo-strip fix in `_run_caption_frame`, `warm_profiles()`'s kind-allowlist tuple gains `"vlm"`.
- `orion/vision/caption_echo.py`: `is_caption_prompt_echo(prompt=...)`, `strip_echoed_prompt_prefix()`.
- `services/orion-vision-host/app/caption_sanitize.py`: `sanitize_answer()`, shared `_tokenize`/`_degenerate_reason`, repetition check.
- `config/vision_profiles.yaml`: `vlm_vqa.enabled: true` + rationale comment (live VRAM check cited).
- `services/orion-vision-host/.env_example`: `vlm_vqa` in `VISION_ENABLED_PROFILES`.
- Tests: `orion/vision/tests/test_caption_echo.py`, `services/orion-vision-host/tests/test_caption_sanitize.py`, `services/orion-vision-host/tests/test_caption_profile_routing.py` (updated), new `services/orion-vision-host/tests/test_run_vlm_vqa.py`.

## Schema / bus / API changes

- Added: none published to the bus. `task_type=vqa` was already a valid HTTP/bus request shape — this PR makes it actually execute instead of erroring.
- Removed: none.
- Renamed: none.
- Behavior changed: `vlm_vqa` profile flips from `profile_disabled` to real execution for any caller sending `task_type=vqa`.
- Compatibility notes: `is_caption_prompt_echo`'s signature change is additive/backward-compatible.

## Env/config changes

- Added keys: none new — `VISION_ENABLED_PROFILES` (existing key) gains `vlm_vqa` in its CSV value.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes, hand-edited (copied from the primary checkout's live `.env` into this fresh worktree, then applied the same `vlm_vqa` addition — confirmed `git check-ignore`'d).
- skipped keys requiring operator action: none.

## Tests run

```text
Venv (no torch/transformers -- pure-logic tests):
  /mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest \
    orion/vision/tests/test_caption_echo.py \
    services/orion-vision-host/tests/ \
    --ignore=services/orion-vision-host/tests/test_heartbeat_chassis.py \
    --ignore=services/orion-vision-host/tests/test_run_vlm_vqa.py
  -> 66 passed

  (test_heartbeat_chassis.py's collection failure is pre-existing on `main`,
  confirmed byte-identical via `git stash` -- unrelated to this patch.)

Real environment (torch/transformers, inside orion-athena-vision-host):
  test_run_vlm_vqa.py's assertions validated via standalone scripts run
  against the real deployed container with the real config path (this
  file's parents[N]-relative config lookup, shared by every sibling test
  in this suite, only resolves correctly from the real repo checkout
  depth -- not a bug specific to this file).
  -> All assertions pass, including the warm_profiles() allowlist fix.

Live, real-model, real-frame verification (the actual point of this
patch -- proof, not assumption):
  Called runner._run_vlm_vqa() directly against the real BLIP-base model
  and real frames from /mnt/telemetry/vision/frames/ inside the running
  container, iterating three times as each bug above was found and fixed.
  Final clean pass:
    Q: "How many monitors are visible on the desk?"  A: '' (rejected: empty)
    Q: "Is there a person visible in this image?"    A: '' (rejected: empty)
    Q: "What color is the door?"
    A: 'a black and white photo of a room with a desk and a computer'
       (accepted -- clean, no echoed-question prefix, not repetition
       garbage; weak/generic answer quality is an honest, expected limit
       of BLIP-base being a captioner, not an instruction-tuned VQA model)
```

## Evals run

None — no eval harness exists for `orion-vision-host` (same gap PR #1679's own report already flagged for this service).

## Docker/build/smoke checks

```text
bash scripts/safe_docker_build.sh orion-vision-host up -d --build
  -> built, deployed twice (once per commit in this PR), each time
     confirmed via docker exec md5sum match against the worktree source
     for every touched file before trusting it -- per this session's own
     earlier-discovered cross-worktree Docker deploy collision risk (see
     the perception-noise-floor PR, #1776, for the full mechanism).
  -> clean startup both times: [WARM] warmed=['retina_detect_open_vocab',
     'embed_image', 'vlm_caption'] -- vlm_vqa correctly stays cold
     (warm_on_start: false), no VRAM cost until a real VQA request lands.
  -> nvidia-smi confirmed VRAM usage unchanged pre/post-deploy (3489MB
     used, same as before this patch) -- the new profile adds zero
     standing cost.
```

## Review findings fixed

- Finding: `_warm_profile_backend`'s new `kind == "vlm"` branch was dead code for a reason its own comment didn't identify — `warm_profiles()`'s own loop has a second, separate kind-allowlist tuple that filtered `"vlm"` out before `_warm_profile_backend` was ever called, so flipping `warm_on_start` alone (as the original comment implied would be sufficient) would have silently still not warmed it.
  - Fix: added `"vlm"` to the allowlist tuple (not just corrected the comment).
  - Evidence: new `test_warm_profiles_kind_allowlist_includes_vlm` exercises the real `warm_profiles()` loop end-to-end, confirming it now reaches `_warm_profile_backend` for a `kind=="vlm"` profile once both gates agree; verified live against the real container via a standalone script.
- All other review findings: none — reviewer confirmed error-code classification, return-shape parity with `_run_caption_frame`, `is_caption_prompt_echo` backward-compatibility, `sanitize_caption`'s refactor preserving prior behavior, the repetition threshold's reasoning, the `VISION_ENABLED_PROFILES` gate claim, and env parity all check out. Full test suite (minus one pre-existing unrelated collection failure) re-confirmed: 59 passed at review time.

## Restart required

```bash
bash scripts/safe_docker_build.sh orion-vision-host up -d --build
```

Already deployed and live-verified during this session — this is the exact command to re-run after merge if the running container needs to pick up `main`.

## Risks / concerns

- Severity: should-know, not blocking — Concern: answer quality from the current model (`Salesforce/blip-image-captioning-base`, a captioner, not an instruction-tuned VQA model) is honestly weak/generic, and easy questions can come back empty (correctly rejected as degenerate) rather than genuinely answered. Real, expected limitation of shipping an honest v1 on the model that's actually loaded today, not a defect in this patch's logic — `model_id: "REPLACE_ME/qwen2-vl_or_llava_next"` stays pointed at the real target model for whenever an operator wants to swap it in via `VISION_VLM_MODEL_ID`. Mitigation: none needed; documented in code comments.
- Severity: should-fix, tracked, not blocking (pre-existing, unrelated) — Concern: `orion-vision-frame-router` has been stopped (clean exit, not a crash) for ~7 hours as of this session — nothing is currently converting captured frames into `orion:exec:request:VisionHostService` requests, so the whole passive vision pipeline is currently silent. Discovered while live-testing this patch, not caused by it. Mitigation: none attempted — restarting a stopped service without understanding why it exited is out of scope and potentially unsafe; flagging for investigation.
- Severity: should-fix, tracked, not blocking (pre-existing, unrelated) — Concern: the scheduler's `VISION_VRAM_RESERVE_MB` (3500) + `VISION_VRAM_HARD_FLOOR_MB` (1400) = 4900MB fixed overhead leaves almost no scheduling headroom on this P4's actual 7.68GB capacity once the 3 warm profiles are loaded — confirmed live this blocks even the pre-existing `caption_frame` task via the HTTP endpoint right now, independent of this patch. `.env_example`'s own comment names the captioner default as "P100-safe" — these thresholds were very plausibly calibrated for a larger GPU and never re-tuned for this specific P4. Practical consequence: `vlm_vqa` requests will likely hit this identical scheduler-level block today via the live HTTP/bus path, even though the runner-level code is proven correct via direct calls that bypass the scheduler. Mitigation: none attempted — retuning a global VRAM budget that gates every profile on this host is a larger, more consequential change than "add VQA" and deserves its own dedicated verification.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1791
