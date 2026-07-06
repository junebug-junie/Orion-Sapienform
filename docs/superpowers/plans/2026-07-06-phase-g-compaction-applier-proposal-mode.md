# Phase G — Compaction Applier (Proposal Mode)

**Status:** BUILT, HARD-GATED OFF. Awaiting explicit sign-off + live §14 verification before the gate is considered.

This is the one rung of the reverie/dream/compaction weave that **mutates memory**.
Per CLAUDE.md §0A it requires proposal mode before the gate is ever flipped. The
code exists (default-off, snapshot-first, rollback, unit-tested against fakes with
zero live writes); this document is the sign-off artifact.

## What capability changes

Today (Phases A–F): reverie reasons, chains, and asks; REM narrates a
`MemoryCompactionDeltaV1` of what sleep *would* do — and **nothing applies it**.

Phase G adds an applier that, when its hot gate is on **and** the delta's proposal
was policy-approved *for execution*, actually:

- **downscale-renormalizes** edge/weights (never deletes) — the safe subset;
- **prunes** low-salience episodics — only when `DOWNSCALE_ONLY=false`;
- **consolidates** gist cards via crystallization with `source_kind="dream"`.

## What data is touched

- edge/weight rows (downscale) — reversible renormalization;
- episodic rows (prune) — removal, gated behind `DOWNSCALE_ONLY`;
- a new crystallized gist card per consolidate entry, dream-provenance.

The applier touches memory **only through an injected `CompactionMemoryStore`**.
`services/orion-dream/app/compaction_applier.py` imports no sqlalchemy/psycopg and
binds no canonical store — a test asserts this. The real adapter is a separate,
signed-off change; until then, importing the module mutates nothing.

## Privacy boundary

Consolidated gist cards are written with `source_kind="dream"` and a
**dream-origin provenance boundary — never promoted to fact.** Dream-derived
content must not leak into fact-grade recall or convenience surfaces (§0A privacy).
Downscale/prune act on existing memory; they surface no new private material.

## What trace proves it worked

Every apply returns and logs a `CompactionApplyReceiptV1`:
`{delta_id, status, applied, downscaled, pruned, cards_written,
prune_skipped_downscale_only, snapshot_path, error}`.

Runtime-truth evidence for a real apply (§0A) must include: the receipt with
`status="applied"`, the `snapshot_before.json` under
`DREAM_COMPACTION_SNAPSHOT_DIR/<delta_id>/`, and before/after row/edge counts.

## Failure modes that would be dangerous, and the guards

| Danger | Guard |
| --- | --- |
| Auto-apply a reverie whim | Requires a decision that is `approved_for_execution` **with `policy_gate="execution_policy"`** (the human-review gate — the autonomy engine's `autonomy_policy` gate is rejected here) inside a frame with `execution_allowed=True`, **and** the approval must reference this **exact `delta_id`** (a shared `source_request_id` is not enough). Missing/malformed/mismatched frame fails closed. |
| Irreversible prune | `DOWNSCALE_ONLY=true` default skips prune entirely; downscale is renormalize-only (no delete). |
| Partial apply corrupts memory | **Snapshot precedes every write**; any mid-apply error triggers `store.restore(snapshot)`. Snapshot-write failure fails closed (no apply). |
| Rollback itself fails | Receipt returns `status="rolled_back"` with `rollback_failed` + the snapshot path logged for manual restore. |
| Dream content promoted to fact | `source_kind="dream"` provenance boundary; never fact-grade. |
| Unbounded delta | Schema caps on every op list (Phase F). |

## How to disable / roll back

- **Disable:** `ORION_DREAM_COMPACTION_APPLY_ENABLED=false` (default). No restart-time
  side effects; the applier is a typed no-op (`status="disabled"`).
- **Roll back a bad apply:** restore from `snapshot_before.json` under the snapshot dir.
- **Full backout:** the delta staging table and channel remain; only the applier
  gate flips off.

## Gates (all default-safe)

```text
ORION_DREAM_COMPACTION_APPLY_ENABLED=false   # the hot gate — mutates memory
ORION_DREAM_COMPACTION_DOWNSCALE_ONLY=true    # prune stays gated until downscale trusted
DREAM_COMPACTION_SNAPSHOT_DIR=/tmp/dream-compaction-apply
```

## Before the gate is flipped (acceptance)

1. Wire a real `CompactionMemoryStore` adapter (separate change) with its own tests.
2. Full §14 backfill protocol on a bounded, snapshotted target set.
3. Demonstrate before/after counts + a working rollback on real data.
4. Confirm the policy path can only reach `approved_for_execution` via operator review.
5. Sign-off from Juniper recorded here.

Until all five hold, status stays **UNVERIFIED / hard-off**.
