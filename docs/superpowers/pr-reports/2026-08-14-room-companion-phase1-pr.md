# PR report: Claude as a third social-room participant (phase 1)

Branch: `feat/hub-social-room-claude-companion` · PR #1668

## Summary

- Widened `RoomPlatform` from `Literal["callsyne"]` to include `"hub"` and `"aitown"` — a contract/reality fix, since both were already being emitted by live code and live rows.
- Added `orion/schemas/room_claude.py`: `RoomClaudeRequestV1` / `RoomClaudeUtteranceV1` on two catalogued channels, plus `ExternalRoomResponderV1` — the missing half of speaker identity, since every stored room turn today names who spoke *to* Orion and assumes Orion answered.
- New `services/orion-room-companion/`: owns the Claude credential and the `claude -p` subprocess, cloned from `orion-self-study-enrichment`'s hardened credential pattern.
- One room = one durable Claude session (`--session-id` then `--resume`), so a turn sends only the new message.
- Two live smokes: an in-image subprocess smoke and a bus round-trip smoke.
- Taught `scripts/check_service_env_compose_parity.py` about compose-interpolation-only keys, fixing a permanently-red gate for two services.

**No UI.** Hub wiring and the invite button are phase 2. Phase 1 stops at "a real Claude utterance with a real cost, proved end to end."

## Outcome moved

Claude now speaks in an Orion room, on Juniper's subscription, with per-turn cost metered from the CLI's own `total_cost_usd`. That number is the input the v2 autonomy budget needs and did not previously exist.

## Current architecture

The social room was already real and already multi-participant — this was the main correction during design:

- `policy.py:1110 _thread_routing` is an N-party arbiter (`_rank_threads`, `_participant_is_orion`, peer-targeted suppression at `:1186`); ≥7 tests run 3–4 participants.
- Live proof: `aitown-town`, 1235 turns, roster `["Nico Sable","Tessa Quinn","Sofia Bell","Juno Park"]` in `social_room_continuity`.
- Speaker identity already rides in `client_meta.external_participant`, so **no new turn schema was needed**.
- `orion-social-room-bridge` runs with `SOCIAL_BRIDGE_ENABLED=false` and has zero live rows; its policy engine is the asset, its CallSyne transport is not on this path.

The existing Hub `agent-claude` lane was not a starter: `fcc_claude_bridge.py:258-264` sets `ANTHROPIC_BASE_URL` to the local FCC gateway, so it is Claude Code the harness driving local models.

## Architecture touched

Shared schemas + bus catalog; one new minimal-privilege service. Hub is untouched in this phase.

## Files changed

- `orion/schemas/social_bridge.py`: widen `RoomPlatform`
- `orion/schemas/room_claude.py` *(new)*: the two channel payloads + responder/transcript models
- `orion/schemas/registry.py`: registered in **both** `_REGISTRY` and `SCHEMA_REGISTRY` (that file warns a schema in only one is half-registered)
- `orion/bus/channels.yaml`: two channels, request declared `single_consumer`
- `services/orion-room-companion/**` *(new)*: app, tests, smokes, Dockerfile, compose, env template, README
- `scripts/check_service_env_compose_parity.py`: interpolation-only key handling
- `docs/architecture/hub-social-room-claude-companion.md`: design doc + corrections

## Schema / bus / API changes

- Added: `RoomPlatform` values `"hub"`, `"aitown"`. Additive to a Literal; no consumer relied on the narrower type (only `ExternalRoomMessageV1.platform` uses it, and the bridge treats it as an opaque label).
- Added: `orion:room:claude:request` (`room.claude.request.v1`, single_consumer) and `orion:room:claude:utterance` (`room.claude.utterance.v1`).
- Compatibility: purely additive; no migration.
- **Deferred to phase 2**: the `external_responder` field on `SocialRoomTurnV1`'s `client_meta`. `ExternalRoomResponderV1` exists and is used by the utterance payload, but nothing writes it into a stored turn yet — that lands with the Hub relay.

## Env/config changes

- Added keys (companion service only): `ROOM_COMPANION_{NODE_NAME,CLAUDE_CONFIG_DIR,CLAUDE_BIN,MODEL,EFFORT,TIMEOUT_SEC,SETTING_SOURCES,WORKSPACE,PARTICIPANT_ID,PARTICIPANT_NAME,SESSION_STATE_PATH,CLAUDE_CREDENTIALS_HOST_PATH}`, `CHANNEL_ROOM_CLAUDE_{REQUEST,UTTERANCE}`.
- `.env_example` added for the new service. No secrets — the credential key is a **host path**, `:?`-required with no default in code.
- Local `.env` synced: **by hand, deliberately.** `scripts/sync_local_env_from_example.py` reads `.env_example` files from the *primary* checkout, so it silently did not see a worktree-added service — a clean run was the failure mode, not a pass. Wrote `services/orion-room-companion/.env` (mode 600, gitignored) in both the worktree and the primary checkout so the post-merge deploy does not fail the `:?` guard.
- Skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-room-companion && PYTHONPATH=<repo>:. pytest tests -q
44 passed

# and now under the repo-standard command too (was a collection error before):
PYTHONPATH=. pytest services/orion-room-companion/tests -q
44 passed

PYTHONPATH=. python scripts/check_service_env_compose_parity.py orion-room-companion
  OK -- all 19 .env_example keys exposed
PYTHONPATH=. python scripts/check_service_env_compose_parity.py orion-self-study-enrichment
  OK  (was permanently red before this patch)
ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/check_single_consumer_channels.py
  OK 1 orion:room:claude:request   (32 channels, 0 violations)
```

Static gates were mutation-tested rather than trusted:

```text
compose + docker.sock mount        -> test_compose_does_not_mount_high_value_targets FAILS
credential bind widened to a dir   -> test_compose_credential_mount_is_a_single_file... FAILS
ROOM_COMPANION_MODEL dropped       -> parity gate FAILS (so the new exemption is not too broad)
orion-recall (untouched)           -> parity gate still reports its 2 real missing keys
```

## Evals run

```text
None. This service has no evals/ directory, matching orion-self-study-enrichment
(the service it was cloned from, which also has none).
```

Gap is deliberate for phase 1: the thing worth evaluating is room conversation quality, which needs the Hub relay and real multi-turn traffic to measure. Follow-up for phase 2.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-room-companion up -d --build     -> Started

# isolation, verified in the running container
/root/.claude/.credentials.json   present (600)
/root/.ssh  /var/run/docker.sock  /repo    all absent
env | grep -i anthropic                    -> (none)

# in-image smoke (scripts/ now ships in the image; previously hand-copied)
docker exec orion-athena-room-companion python -m scripts.smoke_room_turn
  [PASS] turn 1 produced text
  [PASS] turn 2 produced text
  [PASS] turn 2 recalled the canary (4471)
  [PASS] both turns share one claude session
  [PASS] cost was metered (> 0)
  [PASS] served by a claude model
  session ac7dcbe2-66a6-4868-b912-2ea2a6a2ff85, model claude-sonnet-5, $0.009261
  SMOKE PASSED

# bus round-trip (the path Hub will actually use)
PYTHONPATH=. python services/orion-room-companion/scripts/smoke_bus_roundtrip.py
  1 exact subscriber on a single_consumer channel
  all 8 checks PASS; correlation id preserved; model claude-sonnet-5; $0.0036896
  reply: "A tool does what you tell it; a companion has its own take and will
          tell you when it disagrees with you."
  BUS ROUNDTRIP PASSED
```

## Review findings fixed

Code review ran in a subagent and found 21 items. The material ones:

- **Finding: the room persona was silently dropped on every turn after the first.** `append_system_prompt` was passed only when minting a session, on the false assumption that `--resume` carried it forward.
  - Fix: pass it on every turn. **The review's recommended fix — moving the framing into the user-role text — was tested and is wrong**: Claude correctly refused it as a prompt-injection pattern ("embedded in a user-turn message formatted to look like a system directive"). Re-passing the flag on resume works, 3/3.
  - Evidence: marker probe in the live container. Turn 2 without the flag: `'Goodbye! Let me know if you need anything else.'` (default assistant voice). With the flag re-passed: `'Goodbye!\n\nZQ7-MARKER'`, 3/3 runs. The old test asserted the bug as correct and was rewritten to assert the effect on turn 2 specifically.
- **Finding: `scripts/` was never copied into the image**, so the documented acceptance surface only ran because it had been hand-copied into a running container.
  - Fix: `COPY services/orion-room-companion/scripts /app/scripts` + `scripts/__init__.py`.
  - Evidence: `docker exec ... ls /app/scripts` after a clean rebuild; smoke runs with no `docker cp`.
- **Finding: the credential-isolation claim was overstated.** Hub mounts `/var/run/docker.sock` rw and ships the `docker` CLI, so a Hub-resident agent can `docker exec` into the companion and read the credential.
  - Fix: corrected the claim in the design doc, README, and the test module docstring. Separation is defense-in-depth and accident-surface reduction, **not** an enforcement boundary. Flagged explicitly that the v2 budget cap must not be built on it.
  - Evidence: from inside `orion-athena-hub`, `docker exec orion-athena-room-companion head -c 40 /root/.claude/.credentials.json` returns `{"claudeAiOauth":{"accessToken":"sk-ant-`.
- **Finding: Claude's CLI session store was on the container's ephemeral layer** while the room→uuid map was on a volume, so every rebuild pointed live mappings at dead sessions.
  - Fix: named volume for `CLAUDE_CONFIG_DIR`, declared before the `:ro` credential bind so the file layers on top. Test asserts both the volume and the ordering.
- **Finding: `build_subprocess_env` was a 3-entry denylist** that missed `CLAUDE_CODE_OAUTH_TOKEN` — the one variable proven to bypass the mounted credential entirely.
  - Fix: allowlist + deny-prefix backstop; test drives 9 hostile vars including Bedrock/Vertex/AWS.
- **Finding: `filtered_summary` was dead code presenting as the privacy boundary.** README sold it, a test covered it, nothing called it.
  - Fix: called in `run_turn`. Test asserts `self_state` and `recall_debug` never reach the prompt.
- **Finding: a session id was persisted before the turn that created it succeeded**, so a failed first turn left the map pointing at a session the CLI never made.
  - Fix: `peek_or_mint_session` writes nothing; `remember_session` persists on success, recording what the CLI *reported* rather than what was requested.
- **Finding: a crash inside `run_turn` published nothing**, contradicting the design's own "failures must be audible".
  - Fix: guarded; publishes `ok=False`. Also broadened `session_store._read_all` (it caught only `FileNotFoundError`/`JSONDecodeError`, letting `UnicodeDecodeError`/`PermissionError` escape) and made corrupt state quarantine rather than silently overwrite every room's mapping.
- **Finding: tests did not collect under the repo-standard command.**
  - Fix: `tests/conftest.py`. Both invocations now pass 44.
- **Finding: the service turned an existing gate red** (`check_service_env_compose_parity`).
  - Fix: taught the gate that `${VAR}` used in `volumes:`/`ports:` is legitimately not container env. Fixes `orion-self-study-enrichment` too; `orion-recall`'s 2 real gaps still fail.

Corrected in the review itself: the `_looks_like_missing_session` precedence concern is not a bug (`and` already binds tighter), but the clause was parenthesized and given a negative test.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-room-companion/.env \
  -f services/orion-room-companion/docker-compose.yml up -d --build
```

Already applied in this worktree. No other service needs restarting — nothing else changed at runtime.

## Risks / concerns

- Severity: **medium**. Concern: the credential is reachable from Hub via the docker socket, so a v2 spend cap enforced by credential ownership would be advisory. Mitigation: documented in three places; v2 must either drop the socket from Hub, enforce budget in a bus service Orion cannot reach, or ship an advisory cap and label it.
- Severity: **medium**. Concern: the `:ro` credential mount cannot refresh, so it depends on the host rewriting `.credentials.json` in place. If refresh uses atomic rename, the bind pins to a stale inode and auth dies silently — affecting `orion-self-study-enrichment` equally. Mitigation: baseline recorded (`inode=5936136 mtime=1786697381`); re-compare after the next real refresh. On the agent board.
- Severity: **low**. Concern: turns are serialized and `stop` is not checked mid-turn, so SIGTERM can wait past Docker's 10s grace and get SIGKILLed mid-turn. No dedupe on `request_id` either, so a redelivered request bills a second turn. Mitigation: phase 2, once Hub is the only producer and the real traffic shape is known.
- Severity: **low**. Concern: `RoomPlatform` widening removes a type-level guarantee that `CallSyneClient.post_message` only ever sees CallSyne rooms. Safe today only because `SOCIAL_BRIDGE_ENABLED=false`. Mitigation: noted; that client should assert its platform if the bridge is ever re-enabled.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1668
