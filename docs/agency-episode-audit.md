# What can we prove about an action and its consequences?

`scripts/analysis/report_agency_episodes.py` reconstructs a bounded sample of
existing ask and motor records. It shows which links exist, which are absent
from the queried sources, and which cannot be verified. It never sends an
action, publishes a bus event, marks a brief consumed, or changes a database.

This is patch 1 of [the agency episode plan](superpowers/specs/2026-09-26-agency-episode-plan.md).
It is an audit, not a learning organ. No episode in the current report can be
declared causally closed, because the audited contracts do not supply both a
durable pre-action commitment and a later decision receipt.

## Run

Use a Python environment with the repo's existing `psycopg2`, `redis` and
`pydantic` dependencies for live reads. Offline reconstruction and its CLI
need only the standard library. No new service dependency or env key is added.

```bash
# Set the existing ORION_PG_DSN in your environment to the intended database.
python scripts/analysis/report_agency_episodes.py --limit 10 \
  --graph-host 127.0.0.1 --graph-port 6380 --graph-name orion_worldview \
  --snapshot /tmp/agency-episode-metadata.json

# Replay without any network connections:
python scripts/analysis/report_agency_episodes.py \
  --input /tmp/agency-episode-metadata.json

python -m pytest orion/autonomy/tests/test_agency_episode.py \
  orion/autonomy/tests/test_agency_episode_reader.py -q
python orion/autonomy/evals/run_agency_episode_eval.py
```

The snapshot destination must not exist and is created with mode `0600`.
Snapshots and reports retain IDs and timestamps, not questions, replies,
prompts, request envelopes or generated text. IDs themselves may contain
descriptive names; treat a snapshot as local operational data before sharing.
The checked-in replay fixture contains selected operational metadata from the
September 26 audit; it is not a private conversation dump.

Exit 0 means all selected sources were read without conflicting identities.
It does **not** mean all links exist or learning was verified. Exit 2 means
an unavailable/truncated/conflicting source, invalid input, or failed read;
inspect the report when present. Missing records remain findings on exit 0.

## Evidence boundaries

- PostgreSQL connections have `default_transaction_read_only=on` and a 5-second
  statement timeout. The collector refuses writable sessions. Queries are
  bounded; relationship reads use exact identities from the sampled records.
- Graph reads use `WorldviewReader`, which issues `GRAPH.RO_QUERY`. Graph and
  SQL reads are not one atomic snapshot; cross-store disagreement may be lag.
- No event-bus connection is made. Redis here is FalkorDB's graph endpoint,
  not an alternate `ORION_BUS_URL`.
- `--limit` is 1–50: newest asks, newest execution results, and newest field
  scores are sampled independently. Both motor samples are joined back to
  their results, dispatch frames, proposal frames, and feedback frames.
  Sampling only scored actions would hide activity on a different outcome path.
- Relationship caps are explicit: over-cap sets become `truncated`, not empty.
  Identical duplicate records collapse; contradictory identities are excluded
  and reported as conflicts, with affected negative claims remaining unverified.
- `missing` means absent from the queried source for that identity. It is not
  proof the event never happened: retention can remove historical records.
- A current timestamp is never substituted for a missing source timestamp.
  `generated_at` is not a database commit acknowledgment. Rows may be upserted;
  `created_at` alone also cannot prove when a particular payload revision existed.
- The report does not infer causality, inspect image bytes, or prove a model
  used evidence because its prompt contained it. It does not turn narrative
  journal entries into historical facts.

## Live findings on September 26, 2026

The read-only CLI succeeded against the host's PostgreSQL and worldview graph.
The versioned fixture keeps one contractor ask, one recent visual execution,
and one historical field-scored execution, with their matching metadata.
Earlier in the same audit a three-per-source sample gave the same distinctions.
This is a bounded audit, not a census of Orion's agency.

**Contractor replies have real persistence but no decision receipt.**
`hr_6b3e0ad9fdc6_n6_menu_assembly` joins to `brief-27576141f7fd` in the
graph and SQL; the graph has the `ANSWERS` edge and `consumed=true`.
`orion/curiosity/peer_briefs.py::peer_brief_consume_cypher` stores only a
boolean. Hub's `CuriosityInvestigation` kickoff paths publish that mark
after preparing the prompt and before dispatching the next run. Thus even
successful consumption marking does not prove the run executed, much less
that the answer changed its decision. The request contract lacks a recorded
alternative set and dated forecast. The Juniper adapter remains proposed
in open PR #2255; the checked code's peer enum still contains only
`cursor_auto` and `claude_room`.

**Recent images use a different outcome contract.**
`dispatch:proposal:render_scene:tick_3aeed7bbf7f9:none:execution_dispatch_policy.v1`
has a September 26 execution result with `status=success` and
`visual_outcome=produced`. It has no field score. This is **not evidence of a
scoring outage**: `orion/execution_dispatch/builder.py::build_expected_effect`
intentionally returns `None` for `render_scene` and visual-baseline actions.
The current path has `VisualRunOutcome`, `VisualExecutionReceiptV1` and an
artifact receipt in `orion/schemas/reverie_visual.py`. The audit reports the
persisted visual outcome but does not independently inspect the artifact or
its later perception. That further causal chain remains UNVERIFIED.

**Historical field predictions were saved after execution.**
`dispatch:proposal:render_scene:tick_d0db6ae013df:none:execution_dispatch_policy.v1`
joins a result, dispatch frame and scored field outcome from September 8.
Its result was inserted at `08:31:33.913681Z`; the prediction-bearing frame
was inserted at `08:31:33.996928Z`. Its `dispatched_at` was `08:30:20.793548Z`.
The current worker also calls `_send_prepared_candidates` before
`save_dispatch_frame`. The saved frame therefore is not a durable pre-send
expectation receipt. Its old proposal frame was not returned by an exact-ID
lookup; this is a retained-evidence gap, not proof alternatives never existed.

**The field learner already has a consumer.**
`orion/feedback/outcome_resolution.py::resolve_action_outcomes` scores
observations; feedback store `_write_action_outcomes` advances posteriors with
ledger deduplication. Dispatch `_load_effect_posteriors` feeds prediction
construction and allocator preview/enforcement. A new generic learner would
duplicate existing machinery. What this audit cannot join is one specific
episode update to a particular later decision. The separate visual path
needs its own provenance mapping before claiming its feedback is absent.

## Recommended next patch

Complete the ask lane with #2255 rather than fork its ledger. Persist the
actual ask decision and expectation before delivery, and record which later
run actually consumed a brief, what decision it made, and its explicit
no-update reason where applicable. A mark made before run dispatch must not
stand in for that receipt. Use explicit reply identity; timeout does not mean
rejection or resolution.

For the motor phase, preserve the visual contract and fix precommit ordering
without weakening retry/idempotency safeguards. Map visual artifact/perception
and field-posterior consumers separately. No new reward, prediction-error or
agency metric is wired by this patch; any future cognitive signal must pass
the repo's full metric gate against real data before use.

No deployment or restart is required for this audit. Disabling it means
stopping the local command; it leaves no daemon, schema migration or consumer.
