## Summary

- Give AI Town's NPC-to-NPC opener a user turn so Circe's Qwen3.5 fast worker can render its chat template.
- Make the AI Town LLM smoke use the actual `quick_background` route and an opener-shaped request.
- Keep the Qwen3.5 fast worker in place; this patch does not change metacog, GPU pool, or gateway policy.

## Outcome moved

The pinned AI Town patch chain now produces a `startConversationMessage` request with system instructions followed by `Greet ${otherPlayer.name} now.` as a user turn. The proposed shape rendered on the live Qwen3.5 worker (`/apply-template`: 200). End-to-end NPC dialogue is `UNVERIFIED` until the Convex functions deploy and the pool admits `quick_background` again.

## Current architecture

AI Town's Convex functions live in the pinned `upstream/` clone, with tracked patches applied by `scripts/pre_rebuild.sh`. `scripts/post_rebuild.sh` pushes the functions after the frontend compose build. `convex/util/llm.ts` forwards chat messages over OpenAI HTTP to gateway route `LLM_MODEL`, normally `quick_background` (GPU pool class `fast`, Circe port 8013). The opener sent only a system message; continue and leave already add user turns through `replyCue`. AI Town has no Python `requirements.txt`, `settings.py`, or `evals/` directory; its patch-chain tests and smoke script are the checks.

## Architecture touched

One tracked patch changes the NPC opener's message shape. The patch-chain runner, its applied-source test, the live smoke, and service documentation change with it. No gateway, model, or pool routing code changes.

## Files changed

- `services/orion-ai-town/patches/orion-qwen35-user-turn.patch`: add one user greeting request after the opener's system prompt.
- `services/orion-ai-town/scripts/apply_upstream_patches.sh`: apply the new patch after speaker-role handling.
- `services/orion-ai-town/tests/test_live_turn_roles_patch.py`: assert the fully applied pinned upstream has system then user in the opener.
- `services/orion-ai-town/scripts/smoke_llm_rail.sh`: probe `quick_background` with the opener message shape and require nonempty final content without reasoning.
- `services/orion-ai-town/README.md`: incident and rollout checks.

## Schema / bus / API changes

No fields, bus channels, schemas, or HTTP routes change. The AI Town request body now includes a user turn in the NPC opener. Human-first conversations and continue/leave roles stay as they are.

## Env/config changes

No template keys or config values change. Ignored local `.env` copies were created for the local build gate; they are not committed. `check_env_template_parity.py orion-ai-town`: PASS.

## Tests run

```text
AITOWN_UPSTREAM_GIT=<local pinned clone> python3 -m pytest -q services/orion-ai-town/tests/test_live_turn_roles_patch.py services/orion-ai-town/tests/test_pre_rebuild.py services/orion-ai-town/tests/test_sim_core.py
17 passed, 18 skipped locally (Node 18; CI provides Node 24)
bash -n services/orion-ai-town/scripts/smoke_llm_rail.sh: PASS
git diff --check: PASS
```

## Evals run

- Live Circe Qwen3.5 `/apply-template` accepted the proposed system-plus-user shape: HTTP 200, nonempty rendered prompt.
- The pooled gateway smoke got HTTP 503 `shed:orion_self_shed` before reaching the worker. Full completion and NPC dialogue remain `UNVERIFIED`; the smoke must be rerun after admission resumes and after deployment.

## Docker/build/smoke checks

`scripts/safe_docker_build.sh orion-ai-town build frontend`: PASS after refreshing the ignored clone's generated Convex API type from Circe. The initial local attempt failed because the pin's `_generated/api.d.ts` predates the existing `sim` module; production `pre_rebuild.sh` runs Convex codegen before the build.

## Review findings fixed

Independent review found no functional patch defect. It confirmed AI Town attribution: the 488 failed `http:openai` / `quick_background` rows used three correlation IDs, all matching `aitown_session_opened` entries in Athena embodiment logs. Six additional `quick_background` 5xx rows were `cortex-exec` `journal.compose` and remain a separate issue.

## Restart required

After merge, deploy from Circe's clean primary `main` synchronized with `origin/main` using AI Town's normal `pre_rebuild.sh` → safe frontend `up -d --build` → `post_rebuild.sh` sequence. The post hook pushes the Convex functions; a frontend restart alone does not apply this fix. Run `smoke_llm_rail.sh` after pool admission resumes and inspect new AI Town dialogue plus `llm_call_usage` for `quick_background` 5xx. Production deployment requires Juniper's confirmation under `AGENTS.md` section 13.

## Risks / concerns

- `quick_background` is currently shed with `shed:orion_self_shed` despite an idle healthy fast worker. The model-compatible prompt cannot produce dialogue until the pool admits it again.
- The six `journal.compose` failures are outside AI Town and are not fixed here.
- The gateway's general HTTP ledger holder is `http:openai`; the session correlation IDs provide attribution for this incident but not a permanent per-caller field.

## PR link

Pending.
