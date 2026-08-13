# Muse Glimmer 30B model card + DFlash speculative decoding — Circe 1x V100 32GB agent lane

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1627
Branch: `feat/muse-glimmer-30b-profile`

This report supersedes the version committed earlier in this branch (`d1cfeb7e7`), which
described an intermediate state (profile-only, DFlash intentionally not wired) that a later
commit (`3949d5c72`) changed. That version is stale; this one reflects what the branch
actually ships as of `HEAD`.

## Summary

- Adds `muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision` to `config/llm_profiles.yaml`: `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL text weights + Q8_0 mmproj, tools + vision + **DFlash speculative decoding** enabled.
- Sampling matches the model card exactly: `temperature=1.0`, `top_p=0.95`, `top_k=64`.
- Sized for 1x V100 32GB, wired for `ATLAS_AGENT_PROFILE_NAME` (Circe's Agent compute GPU lane, per `services/orion-llamacpp-host/README.md`'s worker table).
- **Service code change**: `services/orion-llamacpp-host/app/profiles.py` gains `LlamaCppConfig.spec_type`/`spec_draft_n_max`; `app/main.py`'s `build_llama_server_cmd_and_env` gains the CLI-emission logic for llama-server's `--spec-type`/`--spec-draft-n-max` (three distinct speculative-decoding mechanisms: classic small-LM draft, block-drafting types incl. DFlash, and n-gram types).
- Model's "thinking" control documented but **not wired** via `chat_template_kwargs` — this model uses a different, system-prompt-based convention (see below).
- **Requires a llama-server build past `ggml-org/llama.cpp#26841` (merged 2026-08-10)** — not just past the DFlash merge. This gates the whole model (text + vision + draft), not only speculative decoding. See "Research findings" below.
- Adds/rewrites regression tests in `services/orion-llamacpp-host/tests/test_profile_forwarding.py` covering: the profile's flags on a supporting build, fail-closed behavior on an old build, and two CLI-builder correctness fixes surfaced by review (see "Review findings fixed").

## Outcome moved

New GPU lane (Circe, agent/tool-use + vision + speculative decoding) has a schema-checked, CLI-builder-tested model card. **Not yet deploy-ready as-is**: it requires bumping this worker's `LLAMACPP_IMAGE_TAG` past a 3-day-old upstream merge, which has not been live-smoke-tested from this environment (no Docker/GPU access here).

## Current architecture

`orion-llamacpp-host` is a profile-driven llama.cpp wrapper: `config/llm_profiles.yaml` is the source of truth for model path/download spec/runtime knobs, `LLM_PROFILE_NAME` selects the active profile per container, and `docker-compose.atlas-workers.yml` defines fixed worker slots (`chat`/`metacog`/`fast`/`agent`) each bound to an env-supplied profile name, all reading `LLAMACPP_IMAGE_TAG` from one shared build-arg default. The `agent` slot (`ATLAS_AGENT_PROFILE_NAME`, port 8014) is the "Agent compute GPU lane" — already the physical target for tool-use/agentic workers (see `qwen3-coder-next-*-agent-*` profiles). `build_llama_server_cmd_and_env` in `app/main.py` translates a validated `LLMProfile` into a `llama-server` argv, gating each flag behind a live `--help` probe (`_get_supported_llama_server_flags`) so an older pinned binary degrades gracefully instead of getting an unrecognized flag.

## Architecture touched

- `config/llm_profiles.yaml`: new profile.
- `services/orion-llamacpp-host/app/profiles.py`: new `LlamaCppConfig` fields (`spec_type`, `spec_draft_n_max`).
- `services/orion-llamacpp-host/app/main.py`: new CLI-emission logic in `build_llama_server_cmd_and_env` — this function runs at every container boot for this service, so this is a runtime behavior change for every profile that sets `draft_filename` and/or `spec_type`, not just the new one (existing profiles leave both unset, so their emitted argv is unchanged — covered by the pre-existing `test_draft_fields_emit_speculative_decoding_flags` and the two new draft-simple/ngram tests below).
- `services/orion-llamacpp-host/tests/test_profile_forwarding.py`: new/updated tests.

## Files changed

- `config/llm_profiles.yaml`: new `muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision` profile (chat/vision/tools/sampling + wired DFlash draft config).
- `services/orion-llamacpp-host/app/profiles.py`: adds `LlamaCppConfig.spec_type` (`--spec-type`) and `spec_draft_n_max` (`--spec-draft-n-max`).
- `services/orion-llamacpp-host/app/main.py`: adds `_BLOCK_DRAFT_SPEC_TYPES`/`_NGRAM_SPEC_TYPES` constants, `_flag_confirmed_supported()` helper, and rewrites the speculative-decoding block in `build_llama_server_cmd_and_env` to handle three spec_type families correctly (classic/block-drafting/n-gram).
- `services/orion-llamacpp-host/tests/test_profile_forwarding.py`: replaces the earlier "draft intentionally unset" test with a supporting-build test and a fail-closed-old-build test for the new profile, plus two standalone regression tests for CLI-builder bugs review caught (`draft-simple` explicit spec_type, `ngram-*` without `draft_filename`).

## Schema / bus / API changes

- Added: `LlamaCppConfig.spec_type: Optional[Literal[...]]`, `LlamaCppConfig.spec_draft_n_max: Optional[int]` in `services/orion-llamacpp-host/app/profiles.py`. Both optional, default `None` — no change to any existing profile's parsed shape.
- Removed: none.
- Renamed: none.
- Behavior changed: `build_llama_server_cmd_and_env`'s speculative-decoding branch is restructured (see "Architecture touched"). Behaviorally identical for any profile with `spec_type` unset (every profile in the registry except the new one).
- Compatibility notes: **`services/orion-llm-gateway/app/profiles.py`** has its own, independent `LLMProfile`/`GPUConfig` that doesn't model the `llamacpp:` block at all — pydantic's default `extra="ignore"` means it silently drops the whole block including the new fields. Pre-existing drift (not introduced by this PR), harmless for that service's actual use (routing/capability metadata only, no CLI construction). **`services/orion-llamacpp-neural-host/app/profiles.py`** has its own, independent `LlamaCppConfig` (no `mmproj_filename`/`image_min_tokens`/`image_max_tokens`/`spec_type`/`spec_draft_n_max`) and its `settings.py:load_profile_registry()` eagerly parses *every* profile in the shared `config/llm_profiles.yaml` — including this one — at every startup via `extra="ignore"`, silently dropping those fields. Confirmed via direct read of that file; not fixed here (separate service, separate schema, out of scope for this PR) but flagged because review found the earlier version of this report didn't disclose it. If `LLM_PROFILE_NAME` is ever pointed at `muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision` for `orion-llamacpp-neural-host`, it would boot with none of the vision/DFlash config and no error.

## Env/config changes

- Added keys: none (profile *values*, not new env keys — assign via existing `ATLAS_AGENT_PROFILE_NAME` on the Circe worker's `.env`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable (no new key; `LLAMACPP_IMAGE_TAG` already exists as a key, only its per-worker *value* needs setting — see "Restart required").
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable, no `.env_example` change.
- skipped keys requiring operator action: `LLAMACPP_IMAGE_TAG` on the Circe worker's own `.env` — see "Restart required".

## Research findings

- **Your question, answered**: DFlash is confirmed real and native to this model, not a hand-picked example. HF README, verbatim: *"Muse Glimmer ships with a lightweight 'drafter' model based on DFlash, a small companion network that proposes entire blocks of tokens at once."* The mechanism itself is real and merged upstream ([ggml-org/llama.cpp#22105](https://github.com/ggml-org/llama.cpp/pull/22105), 2026-06-28), needing `--spec-type draft-dflash` + `--spec-draft-n-max` (`docs/speculative.md`), which is what this PR now wires.
- **Bigger finding, corrects the earlier version of this report**: the whole model — not just DFlash — needed its own dedicated llama.cpp architecture PR: [ggml-org/llama.cpp#26841 "model: Muse Glimmer Support"](https://github.com/ggml-org/llama.cpp/pull/26841), merged **2026-08-10** (3 days before this PR). It's a new `LLM_ARCH` enum plus the full C++ graph-building code for the target model, its vision encoder, and its DFlash drafter — llama.cpp cannot recognize a Muse Glimmer GGUF at all before that merge. This stack's current default pin, `LLAMACPP_IMAGE_TAG=server-cuda-b8740` (~April 2026), predates it by months. The earlier version of this report claimed "chat/tools/vision already work fine on the current pin" — that was wrong; it verified the generic mmproj *mechanism* against other models, never architecture recognition for this one.
- **Required tag**: `server-cuda-b10398` (confirmed to exist on `ghcr.io/ggml-org/llama.cpp`, published within the last day as of 2026-08-13) is safely past both merges. **Do not bump the repo-wide `LLAMACPP_IMAGE_TAG` default** — `config/biometrics/node_catalog.yaml` confirms Circe already runs `orion-atlas-llamacpp-chat` continuously, and if that worker shares a compose invocation/`.env` with the new agent worker on Circe (not confirmed from here), a default-fallback bump would silently rebuild that already-stable lane onto 3-day-old master with no bench pass. Set the tag explicitly as an override on whichever `.env` governs the Circe agent worker.
- **Known open upstream risk**: [ggml-org/llama.cpp#25116](https://github.com/ggml-org/llama.cpp/issues/25116) (status: `bug-unconfirmed`, open as of 2026-08-13) reports a GGUF architecture-name/nested-key-scheme mismatch breaking DFlash draft loading for a different model (Qwen3.6-27B) on the same DFlash mechanism. Not confirmed to affect Muse Glimmer's own `dflash-kquant.gguf` — but the failure mode is a hard load error (`"DFlash model requires 'target_layers' in GGUF metadata"`), not silent degradation, so first boot will surface it immediately if it applies.
- **Thinking control is different for this model.** No `chat_template_kwargs.enable_thinking`/`--reasoning-budget` equivalent — the card controls reasoning depth via literal `"Reasoning strength: <low|medium|high|xhigh>"` text in the system prompt. Left `llamacpp.reasoning*`/`chat_template_kwargs` unset rather than wiring a flag that would silently no-op for this model.
- **VRAM budget**: UD-Q4_K_XL weights (15.9GB) + Q8_0 mmproj (~1.9GB) ≈ 17.8GB, leaving ~14GB of the 90%-of-32GB budget for 32k KV + batch/ubatch overhead — same order as `qwen36-35b-a3b-udq4km-v100-32gb-balanced`, already proven to fit this card class. The draft model + its own KV overhead is additional and has not been separately benched.

## Tests run

```text
$ /mnt/scripts/Orion-Sapienform/venv/bin/python3 -m pytest services/orion-llamacpp-host/tests -q
25 passed, 1 failed (pre-existing, unrelated), 19 warnings

FAILED tests/test_profile_forwarding.py::test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k
  -> pydantic ValidationError: llm_profile_name Field required (Settings env-parse
     ordering issue). Reproduces identically against unmodified origin/main -- confirmed
     pre-existing, not caused by this change.
```

New/changed tests, all passing:
- `test_muse_glimmer_30b_agent_vision_profile_forwards_flags_on_supporting_build` — full argv on a build advertising `--spec-type`/`--spec-draft-n-max`/`--model-draft`.
- `test_muse_glimmer_30b_agent_vision_profile_skips_draft_entirely_on_old_build` — old build advertises `--model-draft` but not `--spec-type`; asserts the draft is skipped *entirely* (no flags at all), not loaded under the wrong path, and that an error is logged.
- `test_explicit_draft_simple_spec_type_still_falls_back_on_old_build` — regression test for a review-caught bug: `spec_type="draft-simple"` on an old build must still fall back to classic `--model-draft`/`--draft-min`/`--draft-max`, not be treated as a hard prerequisite the way block-drafting types are.
- `test_ngram_spec_type_emits_flag_without_draft_filename` — regression test for a review-caught bug: `ngram-*` spec types need no `draft_filename` and must still emit `--spec-type`, which the first version of this logic couldn't do (the whole emission lived inside the `draft_filename` branch).

Also manually validated the new profile parses cleanly against the live `LLMProfile`/`LlamaCppConfig` pydantic schema in `app/profiles.py` (separately from the full-registry parse, which fails on an unrelated pre-existing `llama3-1-cola` backend-literal mismatch).

## Evals run

No eval harness exists for `orion-llamacpp-host`; this is a config-card + CLI-builder addition, not a behavior change to an existing running model. No live-inference eval is possible pre-deploy (model not yet downloaded to Circe, and the required build tag has not been smoke-tested).

## Docker/build/smoke checks

Not run — no Docker/GPU access from this environment. **This is the main open gap**: the profile and CLI-builder logic are schema- and unit-tested, but no live `llama-server` boot has been attempted against `server-cuda-b10398` with this model, its mmproj, or its DFlash drafter. Required before trusting this in production — see "Restart required".

## Review findings fixed

`/code-review high` (background subagent) against `feat/muse-glimmer-30b-profile`, second pass after wiring DFlash — 8 findings, 6 confirmed as real issues to fix, 2 lower-priority cleanup findings:

- **Finding**: `spec_type is not None` branch treated explicit `spec_type="draft-simple"` identically to block-drafting types (`draft-dflash`/`draft-dspark`/`draft-mtp`) — silently dropped `draft_min`/`draft_max` and hard-failed the whole draft on an old build, even though `draft-simple` has a working classic-flag fallback that `draft-dflash` etc. do not.
  - Fix: split spec_type handling into `_BLOCK_DRAFT_SPEC_TYPES` (hard `--spec-type` prerequisite, no classic tuning) vs. everything else (classic tuning applies, `--spec-type` emitted opportunistically but never required).
  - Evidence: `test_explicit_draft_simple_spec_type_still_falls_back_on_old_build` passes.
- **Finding**: `ngram-*` spec types (schema-valid) could never actually emit `--spec-type`, since the whole branch lived inside `if cfg.draft_filename:` and n-gram drafting needs no draft file.
  - Fix: moved n-gram handling to its own top-level block independent of `draft_filename`.
  - Evidence: `test_ngram_spec_type_emits_flag_without_draft_filename` passes.
- **Finding**: the hard-prerequisite check for block-drafting types failed *open* (treated `--spec-type` as supported) when the `--help` probe itself returned `None`, contradicting the "hard prerequisite" design intent — risking a load of an incompatible draft GGUF architecture on unconfirmed support.
  - Fix: added `_flag_confirmed_supported()`, which fails *closed* (`supported_flags is not None and flag in supported_flags`), and used it specifically for the block-drafting hard-prerequisite check — left the rest of the file's existing fail-open convention alone (unrelated, wider-scoped change; those paths degrade safely with a no-op rather than a crash risk).
  - Evidence: code inspection; the two branches now use different helpers intentionally, documented inline.
- **Finding**: the committed PR report was stale against the actual shipped diff (written before DFlash was wired) — claimed "not wired," "config only, no service code changes," "ready to deploy today," named a test that no longer exists.
  - Fix: this document replaces it.
- **Finding**: `services/orion-llamacpp-neural-host`'s independent, unsynced schema would silently drop the new profile's fields if ever pointed at it, and this wasn't disclosed anywhere (unlike the `orion-llm-gateway` drift, which was).
  - Fix: disclosed above under "Schema / bus / API changes." Not fixed in that service's own schema — separate service, separate contract, out of scope for this PR.
- **Finding** (lower priority, fixed): `main.py` re-implemented the `supported_flags is None or flag in supported_flags` shape twice with no shared helper.
  - Fix: extracted `_flag_confirmed_supported()` (used for the fail-closed case; the fail-open case is unchanged, matching the rest of the file).
- **Finding** (lower priority, skipped): `tests/test_profile_forwarding.py` has 6+ pre-existing near-identical profile-loading boilerplate blocks the new tests could have joined into one shared fixture.
  - Outcome: skipped — refactoring the whole file's existing pattern is a separate, unrelated cleanup, out of scope for this PR. The one new helper added (`_load_muse_glimmer_profile`) is scoped to the tests this PR adds.

## Restart required

```bash
# On whichever host/compose invocation runs Circe's atlas-agent worker, set an
# explicit override (do NOT touch the repo-wide LLAMACPP_IMAGE_TAG default):
#   ATLAS_AGENT_PROFILE_NAME=muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision
#   LLAMACPP_IMAGE_TAG=server-cuda-b10398
# Confirm first whether Circe's chat lane shares this same .env/compose invocation
# (config/biometrics/node_catalog.yaml confirms orion-atlas-llamacpp-chat already
# runs there continuously) -- if so, this tag change also rebuilds that lane, and
# that needs its own explicit sign-off, not a silent side effect.

scripts/safe_docker_build.sh orion-llamacpp-host up -d --build atlas-agent
curl -fsS http://localhost:8014/health
docker compose -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml logs --tail=200 atlas-agent
# Check specifically for a "DFlash model requires 'target_layers'" error (upstream
# issue #25116's signature) if the container fails to become healthy.
```

## Risks / concerns

- Severity: medium
  - Concern: no live boot has been attempted against `server-cuda-b10398` with this model. Schema and CLI-argv construction are tested; actual model load (target arch, vision encoder, DFlash drafter, and whether upstream issue #25116 applies to this specific GGUF) is unverified.
  - Mitigation: exact smoke-test commands are in "Restart required"; profile `notes` include a hard-load-error signature to check logs for first.
- Severity: low
  - Concern: bumping `LLAMACPP_IMAGE_TAG` for Circe's agent worker may also affect its already-running chat worker if they share a compose env — topology not confirmed from this environment.
  - Mitigation: flagged explicitly in the profile header, notes, and "Restart required"; needs a human check of Circe's actual `.env`/compose layout before deploy.
- Severity: low
  - Concern: first boot will pull ~17.8GB of GGUF weights (Q4_K_XL text + Q8_0 mmproj) plus the DFlash drafter from HuggingFace; VRAM headroom with the draft model loaded hasn't been benched.
  - Mitigation: `notes` field documents an OOM fallback ladder, including dropping `spec_type`/`hf_draft_*` entirely to fall back to plain non-speculative decoding without touching anything else.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1627
