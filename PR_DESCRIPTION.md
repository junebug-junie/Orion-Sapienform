# AI Town: Orion unified-turn speech + facing guarantee + Juniper human + fresh 8-NPC cast

Branch: `feat/aitown-orion-unified-cast` → `main`
Open PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/aitown-orion-unified-cast

## Summary

- **Orion town speech → full unified turn.** `orion-embodiment` now generates Orion's AI-Town utterances via the hub's unified-turn saga (`POST /api/chat` `mode=orion`) instead of the lightweight `chat_quick` cortex rail, with automatic **fallback to the quick lane** on timeout/error/`turn_error`/`turn_deferred`/empty. NPCs are unchanged (still on the `chat` lane via Convex `LLM_MODEL`).
- **Facing guarantee + inspectability.** `WorldPerceptionV1` now exposes Orion's own `facing`/`pathfinding`; perception computes `facing_partner`; the worker issues a single zero-length stop when Orion is `participating` with residual pathfinding, so the engine's `Conversation.tick` orients Orion toward its partner.
- **Juniper Feld is the human player** (`DEFAULT_NAME` + join description) via a tracked upstream patch.
- **Fresh 8-NPC town cast** (Mara Vale, Nico Sable, Dr. Elian Cross, Juno Park, Tessa Quinn, Vale Moreno, Sofia Bell, Cam Lin) via a tracked upstream patch; Orion joins externally and Juniper is the human (neither is in `Descriptions`).

## Env/config changes

- Added (orion-embodiment): `EMBODIMENT_SPEECH_UNIFIED_ENABLED=true`, `EMBODIMENT_HUB_CHAT_URL=http://100.92.216.81:8080/api/chat`, `EMBODIMENT_UNIFIED_TIMEOUT_SEC=120`, `EMBODIMENT_UNIFIED_SESSION_PREFIX=aitown`.
- `.env_example` updated; `python scripts/sync_local_env_from_example.py` ran → `skip orion-embodiment: no .env` (no local env on this box).

## Tests

```
.venv/bin/python -m pytest services/orion-embodiment/tests orion/embodiment/tests -q  →  122 passed
```
Deterministic AI Town checks: upstream patches reverse-check clean; `apply_upstream_patches.sh` idempotent (exit 0); `tsc --noEmit` on `data/characters.ts` exit 0.

## Review findings fixed

- Task 1 spec: double fallback log → single line.
- Task 1 quality: fallback reason now discriminates `turn_error`/`turn_deferred`/`non_final:<t>`/`empty` (+corr id, +test) — anti silent-failure.
- Task 2 quality: stop now targets exact current position (zero-length) instead of tile-center micro-move.
- Final review: `EMBODIMENT_HUB_CHAT_URL` default fixed to host-reachable Tailscale IP (hub is `network_mode: host`, Docker DNS name would not resolve → silent quick-only); dead `import math` removed; README facing/log wording corrected.

## Restart / operator commands

On the node running orion-embodiment (only when `ORION_EMBODIMENT_ENABLED=true`):
```bash
python scripts/sync_local_env_from_example.py
# set EMBODIMENT_HUB_CHAT_URL to this node's host-reachable hub IP if not the default
docker compose --env-file .env --env-file services/orion-embodiment/.env \
  -f services/orion-embodiment/docker-compose.yml up -d --build
```
Fresh game (destructive; node with `upstream/` cloned):
```bash
cd services/orion-ai-town && bash scripts/apply_upstream_patches.sh
cd upstream && npx convex dev --once
npx convex run testing:stop
npx convex run testing:wipeAllTables
npx convex run init
npx convex run testing:resume
cd ../../.. && python services/orion-embodiment/scripts/bootstrap_orion_agent.py --write
```

## Risks / concerns

- Medium — **UNVERIFIED at runtime**: live town + hub run on a mesh node; unified town speech and Orion's visual facing not observed end-to-end. 122 unit tests + deterministic checks pass. Operator should confirm (1) a town utterance with hub correlation and no fallback spam, (2) Orion visibly faces its partner.
- Low — hub URL default assumes hub at athena Tailscale IP (`100.92.216.81:8080`); override per node. Requires hub `ORION_UNIFIED_TURN_ENABLED=true` + `ORION_HARNESS_GOVERNOR_ENABLED=true`, else always falls back to quick.
- Low — unified town speech publishes into Orion's main cognition rail; `aitown:` session prefix keeps it separable from Juniper↔Orion continuity.
- Note — AI Town conversations are hard 2-party; no group-chat facing case.
