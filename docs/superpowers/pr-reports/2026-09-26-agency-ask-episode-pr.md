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

PR link and final CI status are recorded in the handoff. Base: #2365.
