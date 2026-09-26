# Reading status by URL

## Problem and behavior

Asked to check `https://arxiv.org/abs/2310.19279`, the supplied FCC trace showed tool discovery followed by a demand for the request UUID. The later answer asserted queued and never started without a status result in that trace. The tool and harness previously required UUIDs.

`reading_status` now accepts exactly one `url` or `request_id`. URL lookup uses existing ingress normalization, selects the newest matching queue row (creation time, then seed ID), resolves aliases, and returns its status plus match count and normalized lookup URL. It never fetches, enqueues, or retries. A missing URL returns `not_found` with a null request ID and zero matches. Legacy rows without UUIDs remain visible through their real seed IDs. Recommendation acceptance still requires a UUID.

The harness and tool description direct Orion to use the supplied link, require an actual successful tool call before reporting status, and distinguish a queued latest request from the claim that no earlier work happened. This is guidance, not a deterministic guarantee about generated final prose.

## Current architecture

- Service: Hub owns the queue listener; governor hosts the per-turn MCP producer.
- Entry points: `orion/world_pulse_read/mcp_server.py`, `tools.py`, Hub `scripts/reading_listener.py`.
- Config: Hub `app/settings.py`, `.env_example`, and `docker-compose.yml`; no env/dependency changes.
- Bus: existing `orion:reading:tool:request` and `orion:reading:tool:result:*`.
- Registry: existing `ReadingToolRequestV1` and `ReadingToolResultV1`; no new event kinds.
- Storage: existing `world_pulse_read_seed`; no migration or data rewrite.
- Tests: reading ingress, MCP, harness prefix, queue, evidence, and disposable PostgreSQL suites.
- Evals: existing offline reading handoff and receipt-truth evals; no live model quality claim.

## Contract and rollout

The existing experimental RPC gains optional `url`, mutually exclusive with `request_id`. Existing ID calls retain their behavior. Status receipts may have a null UUID only for not-found or a found legacy row with a nonempty seed ID. The durable recommendation receipt subtype still requires a UUID.

URLs match the normalized stored URL exactly: distinct query strings, paths, and arXiv versions are not collapsed. Match count includes aliases and legacy rows. The selected status is not an aggregate of all historical outcomes.

Deploy Hub's updated listener before the governor/tool producer. An old listener rejects URL selectors. Roll back both components together; no data rollback is needed. No production restart was performed. Operator commands from the task worktree after normal env provisioning:

```sh
bash scripts/safe_docker_build.sh orion-hub up -d --build
bash scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Runtime evidence

Read-only checks on 2026-09-26 found three matching records. An older request had Stage 1 done and Stage 2 failed. The latest request aliases a pending request with one Stage 1 attempt.

The deployed UUID tool RPC returned `queued`, queue position 20 of 67, Stage 1 attempts 1, and a gateway-capacity deferral error. The new worktree queue function was separately loaded into a short-lived Python process in the Hub container and run inside a PostgreSQL read-only transaction: URL lookup returned that same latest request, `queued`, three matches, and position 20 of 67. No deployed files or queue rows were changed.

Full deployed URL MCP path: UNVERIFIED until rollout. The pasted trace alone cannot establish deliberate lying or prove that the later answer performed a tool call.

## Review findings fixed

- Finding: filtering URL matches to non-null request IDs hid supported legacy World Pulse rows and could falsely report not-found.
  - Fix: include all matching rows and share status formatting, retaining a real seed ID for legacy status.
  - Evidence: additive-migration PostgreSQL fixture verifies legacy URL lookup; receipt tests ensure legacy status cannot prove recommendation acceptance.

Independent review applied the requesting-code-review skill; no other material findings.

## Validation

Focused regression tests and existing offline evals cover URL and ID ingress, aliases, missing URLs, legacy rows, read-only SQL, no DNS/enqueue, strict receipt validation, MCP exposure, and rendered harness instructions. Final counts and CI status are reported in the PR.
