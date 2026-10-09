# Plan 2 — `urgent` GPU priority: jump the queue, pause background runs, resume in place

**Spec:** `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` Part 1.
**Branch:** `feat/gpu-pool-urgent-class` (worktree `/mnt/scripts/Orion-Sapienform-urgent-curiosity-hardware-watch`).
**Python:** `PY=/mnt/scripts/Orion-Sapienform/.venv/bin/python` (system python has no pytest).

## Goal

A durable run whose admission says `priority: urgent` gets a GPU slot ahead of every queue except
chat. If no slot is free, the pool pauses one running **background** durable-run hold (then
**system** if no background is left): recall with a 5 s grace, then abort. The paused run is put
back in line **in its original place** (same `lease_id`, same `created_at`) without spending a retry
attempt. Durable-runs sees the preemption, cancels the in-flight turn, and replays the node on the
next freed slot. At most 3 urgent leases are active at once.

## Deviations from the approved spec (decided here, reported to Juniper)

1. **Only holds are paused (victims = durable-run holds).** One-shot request leases (single
   inferences) are never recalled for urgent work. They finish in seconds, and urgent is first in
   line for the freed slot. A request can't be paused and resumed, only killed and replayed, so the
   spec's gateway replay and `gpu_pool_preempted` error code are dropped.
2. **Rule U4 (AC load shedding) moves to Plan 5**, where the hardware watcher that turns it on is
   built. Shipping the scheduler branch now would leave an input with no producer.
3. **No `curiosity_urgent` route.** Durable-run holds take their priority from
   `ResourceRequirementV1.priority` (`hold_placement`), not from the route, so the route would be an
   unused label.
4. **Added:** durable-runs drives at most 4 runs at once (`MAX_CONCURRENT_DRIVERS`). An urgent run
   must not wait behind 4 long background turns before it even asks the pool, so urgent runs are
   driven first and bypass that cap, bounded by `urgent_max_concurrent`.

## Global constraints

- Chat is untouched: urgent never recalls an `interactive` lease, never recalls another `urgent`
  lease, never recalls an operator lease, and borrows the chat role only when gpu0 is lent (existing
  `fits`).
- Victim order: `background` before `system`; within a priority, most recently granted first.
- One victim per waiting urgent lease, counting holds already `recalling` with reason
  `urgent_preempt` (so a 5 s grace doesn't recall a new victim every tick).
- `urgent_max_concurrent: 0` is the rollback switch: urgent then behaves exactly like background
  (no pause, no stacking). It takes a redeploy of both `orion-gpu-pool` and `orion-durable-runs`.
- Known limitation: a Door-A outreach hold (a finished run's hold kept for Hub outreach compose) is
  a background hold and can be paused; durable-runs then ends it, so a compose in progress loses
  its GPU hold.
- The preempted hold's abort must not consume a pool retry attempt and must not count as a failed
  durable-run attempt.
- Never commit `.env`. Never `--no-verify`. Commit per task. Work only in the worktree.

## Current architecture (grounded)

- `orion/gpu_pool/scheduler.py` `schedule()`: pure. Queue order is priority then `created_at`
  (`_order`). Recalls happen only for `owner_waiting`, `card_unlent`, `max_hold` and `draining`.
  H1: `free_for` returns 0 for a hold on a role that already has a hold. Step 1 turns an expired
  recall into `Abort(lease_id)` with the fixed reason `recall_grace_exceeded`.
- `orion/gpu_pool/lease_graph.py` `transition`: `("recalling","abort") -> retry_wait`. `abort` is in
  `_FAILURES` (attempt + 1, `not_before` delay; dead-letter after `retry.max_attempts`), and
  `retry_due` later re-queues it. `created_at` is never reset, so a re-queued lease keeps its place.
- `services/orion-gpu-pool/app/runtime.py`: `_view` doesn't pass the row's `reason`.
  `_reply_for` answers `queued` with no `reason`.
- `services/orion-durable-runs/app/admission_runtime.py`: `_beat` raises `HoldLost` when the hold
  isn't granted/recalling at our generation. `admitted_graph.py` treats `HoldLost` as a generic
  failure (attempt + 1, 30 s backoff, `release(keep_requeued=True)`). `HoldRecalled` is the existing
  "not a failed attempt" precedent.
- `reconcile()` drives at most `MAX_CONCURRENT_DRIVERS = 4` pending runs, in `list_pending()` order.
- `gpu_pool.yaml` is baked into the pool and gateway images (not mounted), so a new priority can't
  break an old image. Old consumers of `GpuPoolEventV1` / `GpuPoolStateV1` (sql-writer, hub, gateway)
  would reject `priority: "urgent"`, but only once an urgent lease exists, which Plan 3 makes happen.
  Deploy them together anyway.

---

### Task 1: `urgent` priority contract + defaults

**Files:** `orion/gpu_pool/config.py`, `orion/schemas/gpu_pool.py`, `orion/schemas/resource_admission.py`,
`config/gpu_pool.yaml`, `orion/gpu_pool/tests/test_config.py`.

1. `config.py`:
   - `PRIORITIES = ("urgent", "interactive", "system", "background")`
   - `RouteSpec.priority: Literal["urgent", "interactive", "system", "background"] = "system"`
   - `Defaults` gains:
     ```python
     urgent_preempt_grace_sec: float = Field(5, ge=0)   # a hold paused for urgent work gets this long, then is aborted + re-queued in place
     urgent_max_concurrent: int = Field(3, ge=0)        # active urgent leases at once; 0 = urgent behaves like background (rollback)
     ```
2. `schemas/gpu_pool.py`: `Priority = Literal["urgent", "interactive", "system", "background"]`.
3. `schemas/resource_admission.py`: `priority: Literal["background", "urgent"] = "background"`.
4. `gpu_pool.yaml`:
   - `priorities: [urgent, interactive, system, background]   # highest first`
   - under `defaults:`, the two keys with the comments above (5, 3).
5. Tests (test-first):
   - the shipped yaml loads with `priorities[0] == "urgent"` and the new defaults
   - `ResourceRequirementV1(priority="urgent")` validates, and `"interactive"` still doesn't
   - a yaml whose `priorities` omits `urgent` fails the permutation check
6. Run `$PY scripts/check_gpu_pool_config.py` if it runs without live services; record the result.
7. Grep for other enumerations of the three old priorities outside tests (e.g. Hub route picker,
   `pool_placement._definitional_priority`, `wait_budget_sec`, sql-writer models, Hub JS). Update
   any that validate or branch on the set. Report what you found.

Commit: `feat(gpu-pool): urgent priority contract and defaults`.

### Task 2: Scheduler rules U1 (pause), U3 (stacking + cap), rollback

**Files:** `orion/gpu_pool/scheduler.py`, new `orion/gpu_pool/tests/test_scheduler_urgent.py`.

1. Docstring: add the rules, numbered like the rest:
   ```text
   Urgent (docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md):
     U1 a waiting urgent lease that got no slot pauses one granted background (then system)
        durable-run hold on a role it could use: recall urgent_preempt with urgent_preempt_grace_sec;
        most recently granted first; never interactive, urgent, operator or request leases.
        An urgent owner reclaiming its own role pauses that role's borrower whatever its priority
        (background or system)
     U2 the paused hold's abort re-queues it in place without spending an attempt (lease_graph)
     U3 urgent holds are exempt from H1 (bounded by slots and urgent_max_concurrent) and skip a
        swap seat's after_wait_sec (guards still apply)
   ```
2. `LeaseView` gains `reason: str | None = None` (the row's last transition reason; the recall
   reason while `recalling`).
3. Module constants: `URGENT = "urgent"`, `PREEMPT = "urgent_preempt"`,
   `PREEMPTIBLE = ("background", "system")`.
4. In `schedule()`, right after `d = cfg.defaults`:
   ```python
   if d.urgent_max_concurrent <= 0:
       # Rollback switch: urgent behaves exactly like background (no pause, no stacking).
       leases = [replace(l, priority="background") if l.priority == URGENT else l for l in leases]
   ```
   (`from dataclasses import dataclass, field, replace`).
5. Step 1 abort: when the `recalling` lease's `reason == PREEMPT`, emit `Abort(lease_id, PREEMPT)`;
   otherwise keep `Abort(lease_id)`.
6. `_Ctx.free_for`: H1 applies only to non-urgent holds:
   `if lease.kind == "hold" and self.holds.get(role) and lease.priority != URGENT: return 0`.
7. Cap:
   - Count active urgent leases (`active_now`, `hold_lease_id is None`, `priority == URGENT`) into a
     one-element list `urgent_n`.
   - `grant()` increments it for urgent leases.
   - Both grant loops (owners-first and the rest) skip an urgent lease while
     `urgent_n[0] >= d.urgent_max_concurrent`.
8. Recall grace: in `recall()`, `reason == PREEMPT` uses `d.urgent_preempt_grace_sec`. Other reasons
   are unchanged.
9. U1, in step 4 after the existing recall loops:
   ```python
   room = max(0, d.urgent_max_concurrent - urgent_n[0])
   waiting_urgent = [q for q in order if q.priority == URGENT and q.lease_id not in granted
                     and q.hold_lease_id is None][:room]
   pausing = sum(1 for l in leases if l.status == "recalling" and l.reason == PREEMPT)
   victims = sorted(
       (l for l in active if l.kind == "hold" and not l.operator and l.priority in PREEMPTIBLE
        and l.lease_id not in recalled),
       key=lambda l: (-rank(l.priority), -(l.granted_at or l.created_at).timestamp()))
   for u in waiting_urgent[pausing:]:
       for v in victims:
           if v.lease_id not in recalled and ctx.placeable(u, v.role):
               recall(v, PREEMPT)
               break
   ```
   (`active` is the existing list of granted non-child leases; `recalled` dedupes within a tick.)
10. Step 6 swap seats: in `waited`, an urgent lease counts as having waited:
    `... >= cfg.swap_after_wait_sec(seat) or q.priority == URGENT`.
11. Tests, test-first, reusing `test_scheduler.py` helpers (`lease`, `run`, `grants`, `of`, `live`,
    `cards`, `T0`, `CFG`) and a local `hold()` like `test_scheduler_holds.py`. In the default `live()`,
    the agent role has 1 slot and agent-gpu2 is unloaded.
    - An urgent hold queued behind a granted background hold on `agent`: no grant; one
      `Recall(victim, T0 + 5 s, "urgent_preempt")`.
    - A background victim is picked before a system one. Among two background holds, the most
      recently granted is picked (use a live() override giving agent 2 slots, or holds on agent + a
      loaded agent-gpu2).
    - Never recalled: an interactive request, an urgent hold, an operator hold, a plain request lease
      (e.g. `agent-burst` system request) or a child lease. With only those occupying the slot, there
      are no recalls.
    - Dedupe: a victim already `recalling` with `reason="urgent_preempt"` means no second recall for
      the same waiting urgent lease. Two waiting urgent leases give two victims.
    - An expired `urgent_preempt` recall gives `Abort(id, "urgent_preempt")`. An expired recall with
      another reason still gives `Abort(id, "recall_grace_exceeded")`.
    - Once the slot is free, the urgent hold is granted before an older queued background hold and
      before an older interactive request on a role both can use.
    - U3 stacking: with agent at 2 slots and a granted background hold on it, an urgent hold is
      granted on `agent` (H1 exempt). A second non-urgent hold is still refused there.
    - Cap: with 3 active urgent leases, a 4th urgent hold is neither granted nor causes a recall.
    - Rollback: with `urgent_max_concurrent=0` (`CFG.model_copy(update={"defaults": ...})` or a helper
      building a modified config), an urgent hold behaves like background: no recall, no H1 exemption.
    - Chat: with gpu0 not lent, a queued urgent hold never gets or pauses anything on the `chat` role.
      With gpu0 lent and a background hold borrowing `chat`, that hold may be the victim.
    - Swap seat: an urgent hold that has waited 0 s justifies `SwapLoad("agent-gpu2", "demand")` when
      residents are idle and guards are clear (`guards=CLEAR`). With `thermal` failing, it's
      `SwapBlocked`.
    - The existing `test_hold_waits_for_a_busy_slot_rather_than_preempting_it` stays green
      (background never preempts).
12. Run `(cd /mnt/scripts/Orion-Sapienform-urgent-curiosity-hardware-watch && $PY -m pytest orion/gpu_pool/tests -q)`.
    All must pass.

Commit: `feat(gpu-pool): urgent leases pause background holds and stack past H1 (U1, U3)`.

### Task 3: Resume in place: lease graph + pool runtime

**Files:** `orion/gpu_pool/lease_graph.py`, `services/orion-gpu-pool/app/runtime.py`,
`orion/gpu_pool/tests/test_lease_graph.py`, `services/orion-gpu-pool/tests/test_runtime.py` (or the
closest existing runtime test file).

1. `lease_graph.transition`: after `nxt` is looked up, a retryable lease aborted for urgent work goes
   straight back to the queue:
   ```python
   preempted = kind == "abort" and event.get("reason") == "urgent_preempt" \
       and bool(state["request"].get("retryable"))
   if preempted:
       nxt = "queued"   # U2: back in line in its original place (created_at kept); not a failed attempt
   ```
   Then in the update: when `preempted`, set `queued_since=now`, `not_before=None`, `role=None`,
   `recall_by=None`, `expires_at=None`, and leave `attempt` unchanged. The `_FAILURES` block must not
   run (it only runs for `retry_wait`, so check that stays true). History records reason
   `urgent_preempt`. A non-retryable lease aborted with `urgent_preempt` follows the normal path.
2. `runtime._view`: pass `reason=row.get("reason")`.
3. `runtime._reply_for`: the `queued`/`retry_wait` reply carries `reason=row.get("reason")`. The field
   already exists on `GpuLeaseReplyV1`; this is additive.
4. Check how `_emit_row` names the recalling→queued transition. The emitted `GpuPoolEventV1` must
   carry `reason="urgent_preempt"` (proof trace). If it maps to an existing event name (`aborted` or
   `queued`), keep it; don't add a new event literal unless there's no fit.
5. Tests:
   - `transition` on a retryable recalling hold with `{"type": "abort", "reason": "urgent_preempt"}`
     gives status `queued`, `attempt` unchanged, `role` None, `created_at` unchanged, and history
     reason `urgent_preempt`.
   - The same with a plain abort still gives `retry_wait` with attempt + 1.
   - Non-retryable + urgent_preempt gives `released`.
   - Runtime: `_view` carries reason, and the queued reply carries reason. Use the existing runtime
     test fixtures (memory store / fake bus) and read them before writing.
6. Run `$PY -m pytest orion/gpu_pool/tests -q` and
   `(cd services/orion-gpu-pool && PYTHONPATH=../..:. $PY -m pytest tests -q)`.

Commit: `feat(gpu-pool): urgent-preempted hold re-queues in place without spending an attempt`.

### Task 4: Durable-runs: urgent runs driven first, preemption replays the node

**Files:** `services/orion-durable-runs/app/admitted_graph.py`, `app/admission_runtime.py`,
`app/pool_hold.py` (constant), tests in `services/orion-durable-runs/tests/` (`test_admitted_graph.py`
and the closest admission-runtime unit test; read both first).

1. `pool_hold.py`: `URGENT_PREEMPT = "urgent_preempt"`.
2. `admitted_graph.py`:
   ```python
   class HoldPreempted(RuntimeError):
       """An urgent run took this run's slot mid-node; the pool re-queued the hold in its original
       place. The node replays when it is granted again. Not a failed attempt."""
   ```
   In the work-node handler, next to `except HoldRecalled:` and before the generic `except Exception`:
   ```python
   except HoldPreempted:
       released = await admission.release(dict(state), "urgent_preempt", keep_requeued=True)
       return {**released, "status": "retrying", "retry_node": None,
               "retry_at": admission.now().isoformat()}
   ```
   (`attempt` untouched.) Record a run event naming the preemption (e.g. `run.preempted` with lease id
   and generation). Check the store accepts arbitrary event names, and match existing naming.
3. `admission_runtime._beat`: before raising `HoldLost`, if
   `reply.status in WAITING and reply.reason == URGENT_PREEMPT`, raise `HoldPreempted(...)`.
4. `admission_runtime.execute`: the victim's turn can also fail on its own first, because its next
   LLM call can't attach to an aborted hold. On any exception from `work` other than
   `HoldPreempted` / `RunControlPending` / `WorkflowDeadline` / timeout, read `holds.status(lease_id)`
   once. If it's `WAITING` with reason `urgent_preempt`, raise `HoldPreempted` from the original
   error. Also inspect how a failed Hub turn surfaces from the node: as an exception, or as a returned
   result the graph later treats as failure. If it's a result, apply the same check before returning
   it. Cover both in tests.
5. `reconcile()`:
   - Order `list_pending()` rows urgent-first, stable otherwise. Priority comes from the run's stored
     admission; find the real field path, e.g. `row["request"]["admission"]["priority"]`.
   - An urgent row doesn't count against `MAX_CONCURRENT_DRIVERS` (and the `break` must not stop
     urgent rows further down).
   - Urgent drivers are capped at `self.holds.cfg.defaults.urgent_max_concurrent`.
6. Tests:
   - Graph: `HoldPreempted` from execute gives status `retrying`, the same `attempt`,
     `release(..., keep_requeued=True)` called, and `retry_at` ≈ now.
   - `HoldLost` still consumes an attempt.
   - `_beat`: a queued reply with reason `urgent_preempt` raises `HoldPreempted`. A queued reply with
     no reason raises `HoldLost`.
   - execute: work raises an error while status says queued/urgent_preempt, giving `HoldPreempted`.
     Work raises while status says granted: the original error propagates.
   - reconcile: with 4 active background drivers and an urgent pending row, the urgent row is driven.
     Urgent rows beyond the cap aren't. Urgent rows are driven before background rows.
7. Run `(cd services/orion-durable-runs && PYTHONPATH=../..:. $PY -m pytest tests -q)`. Postgres-marked
   tests may skip; record the counts, and note any pre-existing failures by checking them on `origin/main`.

Commit: `feat(durable-runs): urgent runs driven first; urgent preemption replays the node, no failed attempt`.

### Task 5: Eval scenario + docs

**Files:** `services/orion-gpu-pool/evals/run_pool_day_eval.py`, the spec, `services/orion-gpu-pool/README.md`.

1. Read the eval. Add an urgent scenario to its replay: two background holds running on agent +
   agent-gpu2 (seat loaded), then an urgent hold arrives. Assert:
   - the urgent hold is granted within `urgent_preempt_grace_sec` + one tick
   - exactly one victim is paused (the most recently granted background hold)
   - the victim is re-granted before a background hold created after it
   - no chat/interactive lease is ever recalled

   Print these as eval lines, in the eval's existing reporting style.
2. Spec Part 1: rewrite to match what shipped. That means the deviations section above (victims are
   holds only; no gateway replay or `gpu_pool_preempted`; U4 moved to Plan 5; no `curiosity_urgent`
   route; the durable-runs driver-cap bypass). Update "Proposed schema / API changes" and acceptance
   check 2 (gateway) to say it was dropped and why.
3. `services/orion-gpu-pool/README.md`: a short "Urgent priority" section covering what it does, the
   two defaults, and the rollback (`urgent_max_concurrent: 0`).

Commit: `docs+eval(gpu-pool): urgent pause/resume scenario; spec matches shipped rules`.

### Task 6: Gates, review, PR, deploy

1. Gates:
   ```bash
   git diff --check origin/main...HEAD
   $PY -m pytest orion/gpu_pool/tests -q
   (cd services/orion-gpu-pool && PYTHONPATH=../..:. $PY -m pytest tests -q)
   (cd services/orion-durable-runs && PYTHONPATH=../..:. $PY -m pytest tests -q)
   (cd services/orion-llm-gateway && PYTHONPATH=../..:. $PY -m pytest tests -q -k "pool or route")
   (cd services/orion-gpu-pool && PYTHONPATH=../..:. $PY evals/run_pool_day_eval.py)
   python3 scripts/check_env_template_parity.py
   $PY scripts/check_gpu_pool_config.py   # if it runs offline
   ```
2. Final whole-branch review in a subagent (review-agent skill); fix findings in one batch.
3. Push and open the PR (AGENTS §18 template), then watch CI.
4. Deploy after merge, from a worktree on up-to-date main. Order: `orion-gpu-pool` first (knows
   `urgent`), then `orion-llm-gateway`, `orion-durable-runs`, `orion-sql-writer`, `orion-hub` (event /
   state consumers).
   - **circe `orion-gpu-lane-controller` goes with the pool, in one step with pulling circe's
     checkout.** It re-reads `/repo/config/gpu_pool.yaml` live from circe's checkout mount, but
     parses it with the `orion` code baked into its image, whose `Defaults` forbid unknown keys. A
     pulled checkout with the old image makes every actuation fail `config_unloadable`. Print the
     pull + rebuild commands for Juniper as one step.
   - `orion-llamacpp-host` needs nothing.
   - `orion-sql-writer` must be redeployed before Plan 3 sends urgent work: it validates
     `GpuPoolEventV1.priority` against the priority list, so an old image drops urgent events.
   - Rollback (`urgent_max_concurrent: 0`) needs a redeploy of **both** `orion-gpu-pool` and
     `orion-durable-runs`: each bakes its own yaml copy.
5. Live proof (no urgent trigger exists until Plan 3):
   - Pool health and `gpu_pool.state` show the new config digest.
   - No regression: a normal curiosity run is granted and completes.
   - **Preemption smoke is an operator action.** Acquiring a synthetic urgent hold pauses a real
     background run, so ask Juniper first. If approved: acquire an urgent hold via
     `orion.gpu_pool.client.acquire_hold` from a one-off script, observe `recalled
     reason=urgent_preempt` then the victim re-queued at its old position, release the urgent hold,
     and see the victim's `run.preempted` event and replayed node. Otherwise mark the preemption path
     live-UNVERIFIED (unit + eval covered).

## Acceptance

- Scheduler tests prove: pause order, chat/interactive/urgent/operator/request never paused, dedupe,
  cap, rollback, stacking, swap skip.
- Lease graph: urgent-preempt abort gives queued, same place, no attempt spent.
- Durable-runs: preemption gives a node replay with no attempt spent; urgent runs aren't starved by
  the driver cap.
- Eval scenario passes.
- PR merged with CI green; deploy order followed; live status stated honestly.
