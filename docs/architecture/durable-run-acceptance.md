# Durable-run acceptance completion

This patch completes the connected Curiosity acceptance coverage after PRs
#2205, #2206 and #2209 and corrects the runner failures it exposed.
PR #2208 is part of the baseline: aligned drafts pass through unchanged;
misaligned drafts use conditional response repair.

## Current architecture and gap

The existing durable service owns LangGraph checkpoints and retries. Cortex's
`dispatch_durable_run` obtains a persisted receipt over the typed bus; the real
`DurableRunner._run_turn` sends a Curiosity turn RPC to Hub. Hub invokes its
unified turn; Thought and repair calls use Cortex Exec, while governor/FCC uses
Gateway's Anthropic path. The shared Postgres broker/permit authority fences
those inference requests. This patch does not replace these boundaries.

Existing tests verify each part separately. They did not execute this connected
receipt-to-artifact path or kill a process while its backend request remained
active. In addition, the actual runner RPC used its fixed legacy timeout even
when an admitted run declared a longer inference budget, and accepted replies
without checking the admitted run/attempt identity.

The fix raises only an admitted turn's transport ceiling to at least its declared
inference budget. `AdmissionRuntime.execute` still owns the inference timeout and
optional overall deadline. Admitted replies must have the Curiosity result kind
and matching envelope correlation, payload correlation and run ID. Attempt
correlation already changes with the lease generation. Invalid replies enter the
existing failure/retry path; legacy timeout and parsing behavior remain intact.

## Acceptance checks

1. A second study receives a committed receipt while the first run holds agent.
2. The waiting graph is interrupted with no open per-run task or RPC; its
   checkpoint remains recoverable beyond the short legacy transport budget.
3. A replacement runtime recovers the same run ID. At 20 minutes, explicit
   compatible alternatives widen; otherwise the preferred lane resumes on release.
4. Real service adapters carry the same lease/generation through primary
   inference, stance, reflection and conditional response repair. No repair call
   is made when the draft is accepted.
5. Completion produces nonempty checkpointed content, journal/attention events,
   lineage/history and terminal release. Duplicate wakeups do not replay completion.
6. Killing the runner during inference fences its old generation without freeing
   the separately owned request permit until the backend ends. Recovery completes
   once with maximum backend concurrency one.

All acceptance infrastructure is isolated: disposable Postgres schemas, typed
in-process bus delivery, and fixture model/FCC cognition. Production Redis,
model hosts and private graphs are not invoked. These checks establish execution
contracts; they do not claim a production cognition run has completed.

## Coverage of the original implementation contract

| Requirement | Implementation / executable evidence |
| --- | --- |
| Persisted prompt receipt, no waiting RPC/task, restart while waiting | Real Cortex dispatch and `AdmissionRuntime` in `tests/test_durable_acceptance.py` |
| Checkpointed resource wait, stable demand, duplicate grant/resume | `tests/test_admitted_graph.py`, `tests/test_admission_runtime_postgres.py`, persisted grant delivered through the actual bus handler in connected acceptance |
| Compatible 1200-second widening, preferred lane retained, atomic winner, hysteresis, overrides, pins, queue age | `tests/test_admission_policy.py`, `tests/test_admission_runtime_postgres.py`, both connected lane cases |
| Existing execution physiology, owning lease on primary and auxiliary inference, #2208 conditional repair | Real Hub/Thought/Governor/Exec/Gateway adapters in `tests/acceptance_turn.py`; all four connected lane/repair combinations |
| Running-worker death, fencing and capacity retained until backend drain | A killed child process and independent loopback backend in `tests/test_running_recovery_postgres.py` |
| Pause, cancellation, terminal races, deadlines, retry/backoff and stable assignment | `tests/test_admission_review_regressions.py`, `tests/test_admitted_graph.py`, `tests/test_admission_runtime_postgres.py` |
| Journal, attention, lineage/history, one terminal event and release | Connected acceptance asserts nonempty checkpointed content and actual emitted envelopes |
| Fairness and shared Gateway capacity | Separate real-Postgres evals: `evals/admission_fairness.py`, `evals/gateway_capacity.py`; Gateway transport regression suite |

Paths in the table are relative to `services/orion-durable-runs` unless a service
is named. SQL-writer persistence of the emitted artifacts is outside this fixture;
the durable history/checkpoints and lease/permit state are verified in Postgres.

## Configuration, compatibility and remaining limits

No new schema, channel, migration, metric or runtime configuration is introduced.
The existing exclusive/run-scoped contract is the permitted smallest slice;
shared/opportunistic run admission is still not implemented. Ordinary synchronous
calls remain on their existing route and participate in Gateway capacity only
when its existing flag is enabled.

The [admission ADR](durable-resource-admission.md) records repository discovery,
ownership, conservative estimated-start inputs, operator APIs and admission
flags. The [capacity ADR](durable-gateway-capacity.md) adds the required request
permit migration and consumer-first activation sequence. Both migrations are
additive and operator-applied. Widening may be enabled before an alternative is
approved; with an empty lane policy it keeps only the preferred lane eligible.
The implementation patch performed no activation or deployment. The checked-in
operator templates now select this path, but production acceptance remains
**UNVERIFIED** until migrations, ordered restarts and the live artifact smoke
have completed.

The repository's existing Graph API and Hub/governor/FCC primary execution path
are retained: replacing them with a new Functional API workflow or forcing FCC
primary inference through Cortex Exec would change Orion's physiology. Stance,
reflection and conditional repair already use the real Cortex Exec/Gateway seam.
No new self-concept cognition algorithm is invented for the representative study.

Fencing cannot undo external tool effects from an interrupted FCC attempt; that
phase remains at least once. The crash test covers runner death with Gateway
ownership still alive, not simultaneous Gateway loss or an upstream that ignores
cancellation indefinitely. Direct backend callers and unrelated DNS aliases are
outside the shared authority's guarantee. The next operational step is the
documented selected-workflow rollout and an observed production artifact chain;
it requires a separate activation decision, not more infrastructure in this patch.
