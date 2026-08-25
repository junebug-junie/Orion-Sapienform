## Summary

- Closes a real gap found while scoping "what's next on the affect arc":
  `orion-affectgpt-worker` (circe GPU1, real AffectGPT + Whisper inference)
  already produced a grounded facial+vocal affect read of Juniper on every
  Hub capture, published on `orion:affectgpt:assessment` — but nothing
  downstream of `orion-juniper-affective-state` ever consumed it except a
  manual debug CLI (`scripts/tap_assessments.py`). Orion's own chat turns
  never found out; only the Hub UI panel showed it to Juniper.
- Adds a single-key Redis mirror (`orion/situational/juniper_affect_state.py`,
  modeled closely on the existing `session_turn_phase.py`) so
  `orion/situational/context.py` can fold a fresh-enough affect read into
  the situation brief every "orion" mode chat turn builds, in both
  orion-hub and orion-cortex-exec.
- Juniper's explicit decisions (asked directly, not my own judgment call):
  auto-color chat turns within a TTL window rather than requiring an
  explicit ask each time; leave the older, separate
  `orion:substrate:juniper_affective_state` text-only signal alone in this
  patch.
- Privacy contract: the situation brief gets a truncated excerpt of the
  model's `raw_response` reasoning, never the verbatim `transcript` of
  Juniper's actual speech.
- Full real end-to-end loop live-verified on real running containers (no
  mocks) — see Docker/build/smoke checks below.

## Outcome moved

Orion's own chat turns can now ground on a real, recent facial+vocal
affect read of Juniper — the perception→inference pipeline built across
this session's earlier PRs (#1846 Whisper, #1857 GPU1) is no longer a dead
end that only the Hub UI could see.

## Current architecture

Hub's "Check now"/ambient toggle → `orion-juniper-affective-state` →
`orion-affectgpt-worker` (circe GPU1) → `JuniperMultimodalAffectV1`
published on `orion:affectgpt:assessment`. `orion/situational/context.py`
builds a "situation brief" injected into the unified-turn harness prompt on
every "orion" mode chat turn (`orion.hub.turn_orchestrator.run_unified_turn`
in orion-hub; `app/executor.py` in orion-cortex-exec), already following an
established provider pattern (weather/lab/perception/runtime) with a hard
staleness gate and a `_build_prompt_fragment` compact-text renderer.

## Architecture touched

`orion/situational/context.py`, `orion/situational/juniper_affect_state.py`
(new), `orion/schemas/situation.py`, `orion/schemas/registry.py`,
`services/orion-juniper-affective-state/app/main.py`,
`services/orion-hub/scripts/main.py`,
`services/orion-cortex-exec/app/main.py`, settings/`.env_example` for
orion-hub and orion-cortex-exec. No changes to `orion-affectgpt-worker` or
`orion-affectgpt-worker`'s own bus contract.

## Files changed

- `orion/situational/juniper_affect_state.py` (new): single JSON payload
  per key (`orion:juniper_affect:latest`), one SETEX, fail-open,
  ok-vs-confirmed-empty-vs-unknown state distinction — same shape
  `session_turn_phase.py` already established. Write side takes an
  explicit `bus` param (the writer already carries `self.bus`); read side
  uses the same module-level bind pattern `session_turn_phase.py` uses,
  since `context.py`'s call chain never receives a bus parameter.
- `orion/schemas/situation.py`: `AffectContextV1` (mirrors
  `PerceptionContextV1` — `available`/`summary`/`observed_at`/
  `observation_age_seconds`/`source`/`privacy_mode`, plus
  `trigger`/`subtitle_source`). `extra="forbid"`, no `transcript` or
  `raw_response` field. Added to `SituationBriefV1`.
- `orion/schemas/registry.py`: `AffectContextV1` registered in `_REGISTRY`
  alongside `PerceptionContextV1` (same precedent — neither is an
  independent bus payload, so neither needs `SCHEMA_REGISTRY`).
- `orion/situational/context.py`: `_build_affect_context` (mirrors
  `_build_perception_context`'s hard staleness gate — a read past
  `affect_max_age_seconds` is withheld entirely, `summary` is never
  carried on a stale/error/disabled result), wired into
  `build_situation_for_ctx`/`SituationBriefV1`/`source_summary`, a new
  compact-text line ("Juniper's affect (captured N min ago[, no speech
  detected]): ...") and a caution line ("a model's inference, not a
  diagnosis..."). New settings: `affect_enabled` (default **True** —
  unlike perception, capture is already an explicit Juniper action, so
  this surfaces an already-consented read, not new surveillance) and
  `affect_max_age_seconds` (default 300s, tighter than perception's 900s).
- `services/orion-juniper-affective-state/app/main.py`: `_publish_event`
  mirrors every successful capture (`ok=True`, non-empty `raw_response`)
  into the Redis key, truncated to 300 chars with an ellipsis. Runs after
  the real `orion:affectgpt:assessment` publish already succeeded —
  additive, fail-open, never blocks or breaks that publish.
- `services/orion-hub/scripts/main.py`,
  `services/orion-cortex-exec/app/main.py`: `bind_juniper_affect_state_bus`
  at startup, mirroring cortex-exec's existing
  `bind_session_turn_phase_bus` call. Both are real, live, verified paths
  for `build_situation_for_ctx` — enabled in both (unlike perception/lab,
  which stayed off in Hub's adapter pending a verified DSN/HTTP
  dependency this feature doesn't need: a plain Redis GET on a bus
  connection both services already hold).
- `services/orion-hub/app/settings.py`,
  `services/orion-cortex-exec/app/settings.py`, both services'
  `.env_example`: `ORION_SITUATION_AFFECT_ENABLED` (default true),
  `ORION_SITUATION_AFFECT_MAX_AGE_SECONDS` (default 300).
- `services/orion-juniper-affective-state/README.md`: new section
  documenting the mirror write.
- Tests: see below.

## Schema / bus / API changes

- Added: `AffectContextV1` (nested sub-schema of `SituationBriefV1`, not
  an independent bus payload — same status as `PerceptionContextV1`).
- Removed: none.
- Renamed: none.
- Behavior changed: `SituationBriefV1` gains an `affect` field
  (additive, defaults to `available=False` — an unpatched producer or
  disabled flag yields "no recent affect read", not a missing field).
- Compatibility notes: no existing bus channel, schema, or payload shape
  changed. `orion:affectgpt:assessment`'s own payload
  (`JuniperMultimodalAffectV1`) is unchanged.

## Env/config changes

- Added keys: `ORION_SITUATION_AFFECT_ENABLED` (both services, default
  `true`), `ORION_SITUATION_AFFECT_MAX_AGE_SECONDS` (both services,
  default `300`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, both `services/orion-hub/.env_example` and
  `services/orion-cortex-exec/.env_example`.
- local `.env` synced: yes, by hand — `sync_local_env_from_example.py`
  silently skipped these genuinely-new keys (known bug, confirmed again
  this session), so appended directly to `services/orion-hub/.env`,
  `services/orion-cortex-exec/.env` (both the primary checkout and this
  worktree), and `services/orion-juniper-affective-state/.env` on circe
  (no new keys needed there — the writer only needs an already-connected
  `self.bus`).
- skipped keys requiring operator action: none.

## Tests run

```text
cd orion/situational && PYTHONPATH=<repo> venv/bin/python -m pytest tests -q
  15 passed (test_juniper_affect_state.py: 8 new; test_hub_settings_adapter.py:
  7, 2 new)

cd services/orion-cortex-exec && PYTHONPATH=<repo> venv/bin/python -m pytest \
  tests/test_situation_provider.py tests/test_situation_conversation_phase.py \
  tests/test_situation_settings_env.py tests/test_session_turn_phase.py \
  tests/test_situation_affect_context.py -q
  48 + 17 = 65 passed (test_situation_affect_context.py: 17 new;
  test_situation_provider.py's prompt_max_chars fix included)

cd services/orion-hub && PYTHONPATH=<repo> venv/bin/python -m pytest \
  tests/test_situation_request_builder.py tests/test_situation_settings_env.py \
  tests/test_websocket_agent_claude_routing.py -q
  7 passed (real .env loaded, not clean-worktree defaults)

cd services/orion-juniper-affective-state && PYTHONPATH=<repo> venv/bin/python \
  -m pytest tests/ -q
  28 passed (test_affect_state_mirror.py: 5 new; 23 pre-existing unaffected)
```

`test_situation_prompt_integration.py`'s 2 failures are pre-existing on
`main` (confirmed by running it unmodified on the primary checkout —
`FileNotFoundError: orion/cognition/prompts/chat_quick.j2`, a cwd/relative-
path harness issue, not caused by this diff), unrelated to this patch.

## Evals run

No dedicated eval harness for `orion/situational/`. Live end-to-end
verification (below) is the real evidence for this feature.

## Docker/build/smoke checks

**Full real end-to-end loop, no mocks, real running containers:**

```text
# Rebuilt/redeployed orion-athena-hub and all 4 orion-athena-cortex-exec
# containers (main/chat/background/spark) from this worktree via
# scripts/safe_docker_build.sh. All started clean, no exceptions,
# exec_rpc_bus_fork_ready logged on all four cortex-exec replicas.

# Rebuilt/redeployed orion-circe-juniper-affective-state on circe (fresh
# worktree there too, not the shared checkout) with the new mirror-write
# code. Clean startup, "[READY] bus connected".

$ ssh circe@circe curl -X POST http://localhost:32799/v1/juniper/affect/trigger \
    -d '{"video_path": ".../demo/sample_00000000.mp4", "audio_path": ".../demo/sample_00000000.wav"}'
  ok=true, real GPU1 inference, subtitle_source=transcribed

# Confirmed the real write landed on the real shared bus, read from
# athena's Hub container:
$ docker exec orion-athena-hub python3 -c "... bus.redis.get('orion:juniper_affect:latest') ..."
  {"summary": "In the text, the caption reads: \"I don't know...\"…",
   "observed_at": "2026-08-25T05:58:24...", "trigger": "manual",
   "subtitle_source": "transcribed"}

# Confirmed the real read side (build_situation_for_ctx, the actual
# production function every chat turn calls) picks it up with NO manual
# write in this run -- the capture above was the only producer:
$ docker exec orion-athena-hub python3 -c "... build_situation_for_ctx(...) ..."
  affect.available=true, source=live, observation_age_seconds=14
  compact_text contains:
  "Juniper's affect (captured just now): In the text, the caption
   reads: \"I don't know, I just don't know how to explain it.\" ..."

# Same check inside orion-athena-cortex-exec with a hand-written
# subtitle_source=none payload -- confirmed the "(no speech detected)"
# qualifier renders correctly, alongside that container's own real live
# weather/perception/runtime-model lines (already working there,
# unaffected by this change).

# Both manual-write test payloads deleted immediately after confirming
# the read path, so no fabricated test data could leak into a real chat
# turn within the 300s freshness window. The real-capture payload above
# was also deleted after confirming build_situation_for_ctx picked it up,
# for the same reason.
```

## Review findings fixed

Code review dispatched via the code-review skill in a subagent (high
effort, given this touches a prompt-injection path per CLAUDE.md's
cognition-change gate). **Verdict: no material findings, approved.**
Independently verified, not just asserted: the reviewer confirmed zero
source drift between this worktree and the deployed circe copy
(`sha256sum` match on both touched files), traced the privacy contract
line-by-line (only `raw_response` ever feeds `summary`; `transcript` is
never referenced), traced the fail-open write path concretely rather than
trusting the docstring (including that `OrionBusAsync.redis`'s
`RuntimeError` on an unconnected bus is caught inside the same try), hand-
verified the `prompt_max_chars` 400→1200 test fix was a genuine boundary
interaction (not a papered-over bug) by computing the exact character
offsets before/after, and re-ran all new/touched tests independently
(46 + 5 passed).

- Finding: none material.
- Noted, not a blocker: `brief.affect.summary` is interpolated into the
  prompt fragment with no sanitization — a pre-existing pattern this PR
  extends (matches `perception_ctx.scene_summary`'s identical shape,
  already shipped un-sanitized), not something this PR originates. No
  action taken; flagging for the record.

## Restart required

Already done as part of live verification — `orion-athena-hub`, all four
`orion-athena-cortex-exec*` containers, and `orion-circe-juniper-affective-state`
are live on this branch's commit right now. No further action needed once
this merges to `main`, unless `main` diverges before another deploy.

## Risks / concerns

- Severity: low. Concern: `affect_enabled` defaults `True` in both
  orion-hub and orion-cortex-exec, a deliberate deviation from
  perception/lab's default-False precedent. Mitigation: justified in code
  comments (capture is already an explicit Juniper action; no new
  DSN/HTTP dependency); reversible via one env flag in either service if
  Juniper disagrees after living with it.
- Severity: low. Concern: the affect line adds ~50-350 chars to every
  situation brief's compact text, competing with other providers for the
  shared `prompt_max_chars` budget (production default 1200, plenty of
  headroom; only a test using an artificially tight 400-char budget hit
  this, fixed in this PR). Mitigation: none needed at the 1200-char
  production default; worth revisiting if `prompt_max_chars` is ever
  lowered.
- Severity: low. Concern: the older `orion:substrate:juniper_affective_state`
  text-only signal remains unwired into cognition, same as before this
  PR — deliberately left alone per Juniper's explicit direction, not
  fixed here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1865
