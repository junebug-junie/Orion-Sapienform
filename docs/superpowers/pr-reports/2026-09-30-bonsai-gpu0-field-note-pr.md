## Summary

- Bonsai image now carries the current wrapper, profiles and `orion/` itself. The circe boot failed with `KeyError` because the old base image's baked `config/` predated the profile.
- Default card moves gpu2 -> gpu0, chat's card. The README gives the stop-chat / start-chat swap.
- Profile `flash_attn` off -> **on**, from a measured A/B on circe.
- The compose file gets a `build:` section, so the post-merge `up -d --build` really rebuilds.
- Field note `docs/2026-09-30-ternary-bonsai2-27b-1xv100-circe.md` plus raw data and scripts in `docs/bench/2026-09-30-bonsai2-circe/`.
- **Flash attention on for every stock-image Volta profile** (`config/llm_profiles.yaml`), from a same-card A/B of Bonsai / Qwen3.8-27B Q4 / Qwen3.6-35B: ~2x faster at 61K depth, never slower short. Live lanes already ran it (wrapper read b10398 as build 0 and dropped "off"), so this is no change today and blocks a slowdown at the next rebuild. Report: `docs/2026-09-30-flash-attention-volta-circe.md`.
- Design proposal (no code): `docs/superpowers/design/2026-09-30-agent-lanes-to-bonsai-design.md`, moving the gpu1/gpu2 agent lanes to Bonsai via the existing llamacpp-host workers.
- Reasoning at depth (hard, 20 items): Q4 27B 20/20, Bonsai 19/20, 35B 19/20. At 4 concurrent sessions Bonsai is ~45% faster than Q4 27B, and fits 2 × 131K per card against 1.

## Outcome moved

The bake-off ran live on circe gpu0.
- **Speed:** 4 concurrent runs at ~27 tok/s each (~99 total). At 61K tokens of context, decode is 32 tok/s with flash attention on and 17.6 with it off. The 35B chat model holds ~70 tok/s at every depth but has one slot.
- **Recall:** held to the 65K slot limit on an exact-string test.
- **Caching:** 73% of prompt tokens reused. 23.5% of the re-processed tokens were the model re-reading its own prior reply, which is Prism's known template issue.

## Files changed

- `services/orion-llamacpp-bonsai-host/Dockerfile`: final stage copies `app/`, `config/`, `orion/`.
- `services/orion-llamacpp-bonsai-host/docker-compose.yml`, `.env_example`: `build:` section, gpu0 default.
- `services/orion-llamacpp-bonsai-host/README.md`: gpu0 swap, what stopping chat also takes down (agent/metacog/fast fallback), don't lend gpu0, measured VRAM, how to turn thinking off.
- `services/orion-llamacpp-bonsai-host/tests/test_bonsai_contract.py`: COPY overlay, `build:` present, `--flash-attn on`.
- `config/llm_profiles.yaml`: `flash_attn: on`, `device_ids: [0]`, measured VRAM.
- `docs/2026-09-30-ternary-bonsai2-27b-1xv100-circe.md`, `docs/bench/2026-09-30-bonsai2-circe/*`.

## Schema / bus / API changes

None.

## Env/config changes

- `BONSAI_CUDA_VISIBLE_DEVICES` default changes 2 -> 0 (meaning changed, key unchanged).
- Local `.env` updated on athena (primary and worktree) and on circe.

## Tests run

```text
pytest services/orion-llamacpp-bonsai-host/tests -q              -> 4 passed
pytest services/orion-llamacpp-host/tests -q (LLM_PROFILE_NAME=ci) -> 40 passed, 1 failed (same failure on main)
check_env_template_parity.py orion-llamacpp-bonsai-host          -> PASS
docker compose ... config -q                                     -> ok
```

## Evals run

```text
Live bake-off on circe gpu0 (the field note). Bench scripts committed under docs/bench/.
```

## Docker/build/smoke checks

```text
circe: build-bonsai-volta.sh -> ok; image contains the profile; _parse_llama_build -> 10750
circe gpu0: /health 200; argv includes --flash-attn off (soak) and later --flash-attn on (A/B run)
```

## Review findings fixed

- Finding: `up --build` was a no-op with no `build:` section.
  - Fix: `build:` section plus a test.
- Finding: the field note said the prompt cache was fully reused. The raw data shows the prior assistant reply re-processed every step.
  - Fix: corrected, quantified (8,607 / 36,597 = 23.5%).
- Finding: the FA-on VRAM figures had no raw samples.
  - Fix: labelled as two hand readings.
- Finding: the 35B wall times show live chat queueing.
  - Fix: the note now says so.
- Finding: the README understated what stopping chat takes down.
  - Fix: added the agent/metacog/fast fallback loss and "don't lend gpu0".
- Finding: stale Dockerfile comments; minor ranges.
  - Fix: corrected.
- Finding: circe's auto-rebuild would start it on gpu0 over chat.
  - Fix (Juniper: manual only): removed from `include_services_circe.txt`; a test fails if any include list names it; CI triggers on those lists.

## Restart required

```bash
# circe, to run the bake-off again:
docker stop orion-circe-atlas-llamacpp-chat
scripts/safe_docker_build.sh orion-llamacpp-bonsai-host up -d --build
# after:
scripts/safe_docker_build.sh orion-llamacpp-bonsai-host down
docker start orion-circe-atlas-llamacpp-chat
```

## Risks / concerns

- Severity: medium. Exactly two concurrent runs are slower with flash attention on (23 vs 41 tok/s each). Three are unmeasured.
- Severity: medium. It burned an 8K output budget thinking on one math prompt.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2434

🤖 Generated with [Claude Code](https://claude.com/claude-code)
