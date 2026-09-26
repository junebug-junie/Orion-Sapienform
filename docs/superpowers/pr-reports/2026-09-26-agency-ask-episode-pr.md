## Summary

An offered peer brief previously looked consumed before its investigation ran.
This patch commits an explicit expectation before a peer invocation (when the
new gate is enabled), then joins the returned brief to a later completed run
and Orion's account of the decision it made. Offered material alone remains
unverified. This is a contractor episode, not Juniper outreach or causal proof.

## Outcome moved

Patch 2 is implemented and checked on an isolated real graph. Production is
**UNVERIFIED** and has not been deployed. The commitment gate defaults off.
This branch is stacked on `feat/agency-episode-audit`, PR #2365, which must land
first. The separate Juniper ask proposal #2255 is unchanged.

## Current architecture

Hub publishes existing graph HelpRequests after investigations complete. The
peer service gates spending and persists replies; later Hub runs offer them in
the prompt. Existing worldview Hop records represent investigation steps.

## Architecture touched / files changed

- `orion/curiosity/agency_episode.py`: atomic acknowledged claim, offer/completion queries, source-checked decisions and prompt contract.
- Shared peer schemas, registry and channel docs: additive expectation and run/phase fields using existing registrations.
- Shared peer reader/persistence and kickoff/self-inquiry prompts: preserve IDs/times, match responses, compare forecast and record choice.
- Peer worker/settings/compose/template: default-off commitment gate and contextual receipt consumer.
- Hub investigation loop: run-specific offers and completed-run receipts.
- Audit reader/projection: bounded commitment and later-choice metadata, preserving missing/conflicting evidence.
- Peer tests/eval, Hub regression assertions, audit tests and focused CI workflow.
- Service READMEs, audit docs, `docs/peer-ask-episodes.md` and this report.

## Schema / bus / API changes

`HelpRequestV1.expectation` adds a validated forecast, no-ask alternative,
considered actions and response window. `PeerBriefConsumedV1` adds optional
consumer run and offered/completed phase. Existing legacy payloads still parse.
New graph records are PeerAskCommit, PeerBriefOffer and PeerBriefDecision.
No SQL migration, new channel, learner or metric is added; metric wiring gate
is not applicable. Consumer-first rollout is required because older models
forbid unfamiliar payload fields.

## Env/config changes

Added `CURIOSITY_PEER_EPISODES_ENABLED=false` in settings, compose and
`.env_example`. Ran `python scripts/sync_local_env_from_example.py`; verified
the local ignored peer `.env` contains false. No secrets or `.env` committed.
No task-specific key was skipped. Existing bus URL remains the Tailscale URL.

## Tests run

Fresh Python 3.12 virtual environment with the same dependency set as CI:

- Peer suite: 56 passed.
- Hub help-request/completion suite: 6 passed, separate process.
- Shared contracts/prompts/schema discovery plus audit: 73 passed (43 shared, 30 audit).
- Env parity for the peer service: PASS.
- `git diff --check`: clean.

Service suites must run separately because both use a top-level `app` package.
The clean-environment run identified missing CI-only dependencies; the workflow
now explicitly installs SQLAlchemy, httpx and pytest-asyncio alongside peer
requirements, pytest and psycopg2-binary.

## Evals run

- Contractor discrimination: 3/3.
- Isolated real FalkorDB lifecycle: 8/8 (pre-call acknowledgment, duplicate
  protection, matched return, brief replay, offered vs completed, decision/Hop
  join, completion replay and wrong-run rejection).
- Metadata audit replay: 10/10.

The isolated lifecycle uses deterministic provider and in-process bus fixtures;
it does not invoke Cursor/Claude or connect to production bus. Model adherence
and production learning quality remain UNVERIFIED.

## Docker/build/smoke checks

Built both affected services successfully through `scripts/safe_docker_build.sh`.
No production restart/deploy. The lifecycle eval starts and stops its own
FalkorDB container on an ephemeral loopback port without mounted volumes.
Safe graphify update passed: 77,966 to 87,621 nodes; graph artifacts preserved.

## Review findings fixed

- Finding: completion serialization omitted fields needed by envelope parsing.
  - Fix: preserve schema identity and the empty brief list; omit only legacy offered default.
  - Evidence: JSON publisher-to-consumer regression and isolated graph eval.
- Finding: refused/failed briefs lacked visible IDs for later decisions.
  - Fix: render their IDs, provider and status.
  - Evidence: prompt regression test.
- Finding: the audit still reported missing alternatives despite a valid commitment.
  - Fix: expose recorded alternatives from commitment evidence.
  - Evidence: audit regression test.

A required independent review-agent subagent re-reviewed the fixes and reported
no findings. It confirmed attributed self-report is kept separate from causal
proof and noted production quality remains unverified.

## Restart / rollout

See [rollout and rollback](../../peer-ask-episodes.md#rollout-and-rollback).
After production authorization, deploy peer consumer first, then Hub producer:

```bash
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Only enable the new flag after both versions are present, then recreate peer.
To remove commitment gating, set it false and recreate peer. To pause all peer
work, disable Hub's existing `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` switch.

## Concerns / limits

At-most-once claims can leave an uncertain ask after a crash; they do not resend
automatically. PubSub receipt loss leaves an evidence gap. Run completion and
Hop provenance validate an attributed account, not experimental causality.
No production episode, Juniper outreach, posterior update or sentience is claimed.

## PR / CI

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/2366

Base: #2365. Final-head CI status is reported in the handoff after all checks finish.

## CI-discovered queue monitor correction

The broad GPU caller job exposed a pre-existing mismatch from base PR #2364:
reading claims now use FIFO within priority, while the digester's head-of-line
age query still sorted attempts before age. Updated that query/docstring and
made its regression compare the actual monitor and claim ordering directly.
This preserves the existing signal's meaning. Focused digestion tests: 18 passed.
The service has no eval directory; this follow-up remains: add a dedicated
queue-monitor replay harness. Existing tests cover empty/negative age and
reader failure; no additional eval coverage is claimed for this correction.

Metric gate recheck for this existing instrument:

1. Producer: `FieldDigesterStore.oldest_world_pulse_seed_pending_age_sec` in
   `services/orion-field-digester/app/store.py`; reads `now()-created_at` from
   the first pending row ordered exactly as `orion/world_pulse_read/queue.py::CLAIM_SQL`.
2. Independence: age and pending count share the seed queue, so are related,
   not independent evidence. Existing durable/pool waiting readers measure
   different queues but can share capacity causes. No new input or weight is added.
3. Anchor: FIFO head-of-line waiting time in priority queueing. This measures
   delay of the next eligible item, not an inferred emotion or independent cause.
4. Live read-only sanity (2026-09-26): 65 pending rows; youngest 75,008 seconds,
   oldest 1,651,559 seconds. Old and corrected queries currently select the
   same priority-0, attempts-1 head. The regression catches divergent ordering
   when fresh arrivals coexist with retries. Empty queue returns SQL NULL,
   mapped directly to 0 by `_oldest_age_sec`; there is no decay or permanent
   positive floor. Existing regression covers empty and negative/clamped values.
5. Existing mechanism: corrected this producer, without adding a second one.
6. Reversibility: one query ordering/docstring and its regression; no schema,
   manifest, config, training default or new producer to retire.

This additionally affects the field-digester image. Its restart, only after
production authorization, is:

```bash
scripts/safe_docker_build.sh orion-field-digester up -d --build
```

Field-digester Docker build passed through the safe wrapper; no restart.
Independent review of this correction returned no findings.

## Requested configuration follow-up

Juniper requested enabling the flags in the example and local env.
`CURIOSITY_PEER_EPISODES_ENABLED=true` now matches in both. Verified
`CURIOSITY_PEER_ENABLED` and `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` are
also true in both surfaces. Ran the env sync script and peer parity check.
Earlier default-off notes above describe initial validation, before this
explicit configuration change. No containers were restarted.
