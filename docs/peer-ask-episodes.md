# Contractor peer ask episodes

Patch 2 of the [agency episode plan](superpowers/specs/2026-09-26-agency-episode-plan.md).
This extends the existing contractor peer loop; Juniper outreach remains the
separate #2255 proposal. It does not add a service, bus channel, learner, reward,
or cognition metric.

## Capability and current architecture

Hub reads Orion-authored `HelpRequest` nodes after an investigation completes
and publishes registered `HelpRequestV1` envelopes. The peer worker applies its
existing budget gate, invokes the existing provider and persists `PeerBrief`
before publishing it. Hub offers returned briefs to later investigations.
Previously a pre-dispatch `consumed` bit was the only receipt; it did not prove
that a run completed or that Orion used the answer.

With `CURIOSITY_PEER_EPISODES_ENABLED=true`, the peer worker now requires an
acknowledged graph commitment before an external invocation. The later prompt
asks Orion to compare the answer with that commitment and record its actual
choice. A separate Hub completion receipt closes the offered run. The audit
joins these records and reports attributed self-report, never causal proof.

## Contracts and data

`HelpRequestV1.expectation` is optional for legacy payloads. When present it has:
`expected_reply`, `if_not_asked`, distinct `alternatives` including `hire_peer`,
`chosen_action=hire_peer`, and `within_seconds` (1–86400). Orion writes the
matching properties on its existing HelpRequest; alternatives are stored as
`alternatives_json`. The reader validates them and preserves `written_at`.

The peer-owned `PeerAskCommit` stores the validated request JSON, digest,
commit token, run/help IDs, commit time and deadline. One atomic query matches
the existing HelpRequest and claims its help ID; the acknowledged token alone
allows invocation. Changed content under that ID is rejected. A crash after
claiming leaves an uncertain ask and deliberately does not auto-resend. An
operator must inspect the evidence before consciously issuing a new help ID.
This is at-most-once claiming, not guaranteed delivery or exactly-once work.
Budget refusals occur before this claim and do not invoke a peer.

Persisting the reply links `PeerAskCommit-[:RETURNED]->PeerBrief` by help/run
identity and records first response time. Empty successful replies or wrong
request identities are converted to a matched failed brief. Brief replay does
not reset its original timestamp or consumed bit.

`PeerBriefConsumedV1` retains its registration/channel and adds optional
`consumer_run_id` and `phase=offered|completed`. Legacy payloads retain their
old consumed-bit meaning. Offered receipts create `PeerBriefOffer` per run and
brief. Completion receipts need a run ID and may have an empty brief list;
they complete that run's existing offers. Replays preserve first timestamps.
Redis PubSub receipt loss leaves missing evidence; there is no durable replay
queue in this patch and no inference that delivery or completion occurred.

Orion authors `PeerBriefDecision` with run/brief IDs, disposition `used|not_used`,
reason, actual decision, optional existing `Hop.n`, and timestamp. For `used`,
the reader requires a nonempty matching Hop within the offer-to-decision window.
Both dispositions require nonempty reasoning and ordered offer, decision and
completion times. A missing/failed/dispatched-only run cannot validate a receipt.
Conflicting rows retain source digests and cannot silently become one account.

The read-only audit additionally reports precommit alternatives, response
window status and validated later choices. An elapsed deadline means no
recorded timely reply; it does not mean rejection. A late reply remains visible.
No receipt claims an experimental counterfactual, a posterior update, or sentience.

## Privacy and failure boundaries

Records live in Orion's existing private worldview graph under its existing
ACL. The committed request is the existing contractor request plus its explicit
forecast and alternatives; no private recall source or new recipient is added.
Audit snapshots retain receipt metadata/digests, not the decision prose.

Dangerous failures are a call before commitment, duplicate spending after an
ambiguous crash, attributing a reply to the wrong ask/run, and calling an offered
brief learning. Tests and the isolated graph eval exercise these boundaries.
Model-authored decisions remain attributed accounts; provider-driven quality
and the production episode path are **UNVERIFIED**.

## Rollout and rollback

The new peer flag defaults to **false**, including the synced local `.env`.
No production deploy or restart was performed. Additive fields still fail old
extra-forbid consumers: deploy the new peer consumer first, then Hub producer.
Use the existing Tailscale `ORION_BUS_URL=redis://<tailscale-node-ip>:6379/0`;
do not substitute a Docker service hostname.

After production rollout is authorized, from this worktree:

```bash
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Verify both services before setting the peer's local
`CURIOSITY_PEER_EPISODES_ENABLED=true` and recreating the peer with the first
command. Require one actual committed ask, matched reply, completed later run,
and source-checked decision before claiming a production episode. Preserve
missing evidence rather than manufacture receipts for old asks.

To disable commitment gating, set the flag false and recreate the peer. That
restores the legacy invocation path; it is not a global pause and does not stop
Hub receipts. Pause the contractor loop using its existing enable switch when
stopping all new work. Keep the upgraded consumer while the producer can emit
new fields. Stored receipts are additive and need no destructive migration.

## Acceptance and validation

- No provider call without a durable acknowledged commitment when enabled.
- Duplicate or changed requests and graph failure cannot trigger another hire.
- Empty or mismatched replies cannot become successful evidence.
- Offered, completed, used and explicitly not-used remain distinct.
- A used claim needs its real matching Hop; timeout remains unresolved evidence.
- Legacy payloads still parse; new envelopes survive JSON transport end to end.

Run peer and Hub tests in separate processes because their packages share the
name `app`. See the PR report for commands and results. The isolated eval runs
its own disposable FalkorDB with no volumes; its provider and bus are fixtures.
It proves the real graph query/receipt lifecycle, not production model behavior.
