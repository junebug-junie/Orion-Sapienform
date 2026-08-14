# llm-gateway: correct agent route to Circe (was wrongly pointed at Atlas)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1640
Branch: `fix/llm-gateway-agent-route-circe`

## Summary

Corrects a mistake in #1636 (merged): that PR pointed `orion-llm-gateway`'s `agent` route at `100.121.214.30:8014` (Atlas's IP), `served_by: "atlas-worker-agent-1"`. Wrong — confirmed directly by the operator, Muse Glimmer was deployed on **Circe** from the start, matching the profile's original design (their very first request: "1x V100 32GB on Circe node on the Agent compute GPU lane").

## Root cause

I (the agent working this task) inferred physical host from the `ATLAS_AGENT_*` env var names / `atlas-agent` compose service name when writing #1636. Those are a **fixed naming convention for this worker pattern**, reused across whichever physical host actually runs it — the same reason `orion-atlas-llamacpp-chat` (this repo's own `chat` route entry, and its own `served_by: "circe-worker-1"` label) already runs on Circe hardware despite its name. This is exactly why Hub reported "Compute: Agent" as **down** right after #1636 landed: nothing was listening at the Atlas IP the gateway was pointed at. The un-aliasing itself (`agent` no longer silently reusing `chat`'s worker) was the correct fix in #1636; only the address was wrong.

Timeline of the confusion, for the record:
1. Operator's very first request explicitly named Circe as the target.
2. Operator deployed Muse Glimmer using the `ATLAS_AGENT_*`-named env vars (correct — that's just this compose pattern's fixed naming) on Circe hardware.
3. I (incorrectly) read "ATLAS_AGENT_*" env vars + `orion-atlas-llamacpp-agent` container name in the boot log as evidence the deploy was on Atlas physical hardware, and said so repeatedly across several turns.
4. #1636 pointed the gateway's `agent` route at Atlas's IP based on that wrong inference.
5. Operator tested via Hub, got "Compute: Agent is down" — because nothing is listening at Atlas's IP.
6. Operator corrected me directly (twice, with increasing frustration) that it was Circe all along.
7. This PR fixes the route; a companion fix on `docs/muse-glimmer-live-verification` (PR #1634) corrects the same wrong-host claims in the profile's own notes and its PR report.

## What changed

- `services/orion-llm-gateway/.env_example`: `agent` → `http://100.112.254.99:8014` (Circe), `served_by: "circe-worker-agent-1"` (renamed from `atlas-worker-agent-1` specifically so `served_by` states the actual host going forward, rather than inheriting the host-agnostic compose service name).
- `services/orion-llm-gateway/README.md`: same correction in the callout and the "default" route-table example, plus an explicit "do not infer physical host from `atlas-*` naming" note so this doesn't recur a third time.
- Live local `.env` in the primary checkout (`/mnt/scripts/Orion-Sapienform/services/orion-llm-gateway/.env`) hand-synced to match — not part of this PR's diff (gitignored), but required for this repo's own dev environment. Same reason as #1636: the sync script's diverged-key protection doesn't overwrite an already-differing existing value.

## Files changed

- `services/orion-llm-gateway/.env_example`: `agent`'s `url`/`served_by`.
- `services/orion-llm-gateway/README.md`: callout text + "default" example block.

## Schema / bus / API changes

- Behavior changed: `agent` logical-lane requests now resolve to Circe's dedicated agent worker instead of a dead Atlas address. `chat`/`metacog`/`quick`/`quick_background` unchanged.
- Compatibility notes: purely corrective — restores the intent of #1636 without which #1636 was actively broken (routed to a host running nothing).

## Env/config changes

- Added keys: none.
- `.env_example` updated: yes, `agent`'s url/served_by (see above).
- local `.env` synced: hand-edited directly in the primary checkout, not via `scripts/sync_local_env_from_example.py` (diverged-key protection).
- skipped keys requiring operator action: whichever host actually runs the live `orion-llm-gateway` process needs the same manual `.env` update + restart — this repo's local `.env` doesn't reach that host automatically.

## Tests run

```text
$ pytest services/orion-llm-gateway/tests/test_lane_routes.py services/orion-llm-gateway/tests/test_route_catalog.py -q
11 passed
```

Same pure routing-logic tests as #1636 — pass regardless of which URL a route points at, so they don't catch host-choice mistakes like this one. That's a real coverage gap (see "Risks / concerns").

## Evals run

Not applicable — routing config, not model behavior.

## Docker/build/smoke checks

Not run from this dev environment (no access to whichever host runs the live gateway). See "Restart required".

## Review findings fixed

- `/code-review medium` against `fix/llm-gateway-agent-route-circe`: a third, earlier README example block (the "Optional per-route upstream model alias" JSON snippet under the Anthropic-passthrough section) still showed `agent` pointed at Atlas's IP (`100.121.214.30:8011`, `served_by: "atlas-worker-1"`) — directly contradicting this PR's own new "do not infer physical host from `atlas-*` naming anywhere in this file" claim. A reader copying that snippet would reintroduce the exact bug this PR fixes.
  - Fix: updated the snippet to the real `agent` values (Circe, `100.112.254.99:8014`, `served_by: "circe-worker-agent-1"`).
  - Evidence: full-file sweep for any remaining `:8014`/`atlas-worker-agent` references after the fix — only the corrected entries remain.

## Restart required

```bash
# Wherever the live orion-llm-gateway instance actually runs:
docker compose -f services/orion-llm-gateway/docker-compose.yml up -d --build llm-gateway
curl -s http://127.0.0.1:8210/v1/models | jq
# Then in Hub: select Compute: Agent and confirm it now resolves (not down).
```

## Risks / concerns

- Severity: low
  - Concern: same VRAM/port-conflict considerations as #1636 apply, now correctly against Circe instead of Atlas — Circe already runs `chat` continuously; confirm the agent worker's GPU index and port don't collide with it.
  - Mitigation: none new beyond what #1627/#1634 already document (OOM fallback ladder, port/GPU-index operator responsibility).
- Severity: low
  - Concern: `test_lane_routes.py`/`test_route_catalog.py` test route-*resolution logic* only — they would not have caught either the original alias bug or this host mis-pointing, since both are about which real-world URL a route's value happens to contain, not the resolution logic itself. No unit test can meaningfully assert "this URL is physically correct" without a live network check.
  - Mitigation: none proposed here — flagging as a known gap rather than pretending test coverage would have caught this.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1640
