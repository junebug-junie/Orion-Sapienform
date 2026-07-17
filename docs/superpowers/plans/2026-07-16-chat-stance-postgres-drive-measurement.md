# Chat stance → Postgres drive measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make chat stance read drive measurement from bus-fed Postgres `drive_audits` (same rail as Mind), and stop treating substrate graph drive snapshots as SoR.

**Architecture:** Add a bounded fail-open latest-row fetch in `orion-cortex-exec` that maps `drive_audits` into the existing `chat_drive_state` shape. Prefer that fetch in `build_chat_stance_inputs`. Stop projecting drive measurement from `snapshot_source=="drive_state"` graph nodes. Default-off (then leave off) substrate-runtime graph materialization of drive snapshots. No Hub→HTTP→sql-writer; no new table unless `activations` fidelity forces it (prefer `active_drives` → activations map).

**Tech Stack:** Python, SQLAlchemy/psycopg2 (already in cortex-exec), pytest, existing `drive_audits` schema via sql-writer bus consumer.

## Global Constraints

- Postgres is SoR for drive measurement; graph must not remain dual-SoR.
- Bus → sql-writer only for durable writes; no producer HTTP to sql-writer.
- Fail-open: missing/empty/timeout must not break chat stance.
- Reuse `drive_audits` latest-row; do not add `drive_state_latest` unless fields diverge.
- Thin Concept/Falkor work is out of scope (another agent owns induction slice).
- Env parity: any `.env_example` change → `python scripts/sync_local_env_from_example.py`.

---

### Task 1: Postgres fetch helper (stance shape)

**Files:**
- Create: `services/orion-cortex-exec/app/drive_state_postgres.py`
- Create: `services/orion-cortex-exec/tests/test_drive_state_postgres.py`
- Modify: `services/orion-cortex-exec/.env_example` (timeout + DSN comment)
- Modify: `services/orion-cortex-exec/app/settings.py` (optional timeout setting)
- Modify: `services/orion-cortex-exec/docker-compose.yml` (passthrough if needed)
- Modify: `services/orion-cortex-exec/README.md` (one-line consumer note)

**Interfaces:**
- Produces: `async def fetch_drive_state_for_chat_stance(correlation_id: str) -> tuple[dict|None, dict]`
- Stance dict keys: `pressures`, `activations`, `dominant_drive`, `summary`, `tension_kinds`
- Mapping: `drive_pressures→pressures`, `active_drives→activations` as `{d: True for d in list}`, include `tension_kinds` in SELECT
- DSN: `ORION_ACTION_OUTCOME_DB_URL` (same conjourney DB sql-writer writes); blank → fail-open None
- Timeout: `CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC` default `0.4`

- [ ] **Step 1: Write failing tests** for success mapping, empty/quiet tick, timeout, missing DSN, jsonb string coercion
- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement minimal fetch helper** (sync SQLAlchemy query + `asyncio.to_thread` + `wait_for`, SET LOCAL statement_timeout)
- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Commit**

---

### Task 2: Wire stance inputs onto Postgres; stop graph drive_state SoR read

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py` (`build_chat_stance_inputs`, `_project_autonomy_from_beliefs`)
- Modify: `services/orion-cortex-exec/tests/test_chat_stance_drive_state_projection.py`
- Modify: `orion/self_state/inner_state_registry.py` notes for `drive_state.v1` consumer path

**Interfaces:**
- Consumes: `fetch_drive_state_for_chat_stance`
- Produces: `ctx["chat_drive_state"]` and optional `inputs["drive_state"]` from Postgres when present
- `_project_autonomy_from_beliefs` no longer fills `drive_state` from `snapshot_source=="drive_state"` (pressures/activations/dominant/summary/tension_kinds). Autonomy summary from DriveNode / `autonomy` snapshots may remain.

- [ ] **Step 1: Rewrite projection tests** — graph drive_state snapshots must not populate `drive_state`; Postgres fetch (mocked) must populate `ctx["chat_drive_state"]`
- [ ] **Step 2: Run — expect FAIL**
- [ ] **Step 3: Implement wire + remove graph SoR projection**
- [ ] **Step 4: Run — expect PASS**
- [ ] **Step 5: Commit**

---

### Task 3: Gate off substrate graph drive snapshot materialization

**Files:**
- Modify: `services/orion-substrate-runtime/app/settings.py` (default `False`)
- Modify: `services/orion-substrate-runtime/.env_example` (`=false` + comment that stance reads Postgres)
- Modify: `services/orion-substrate-runtime/tests/test_drive_state_substrate_materialization.py` (assert default-off / still works when explicitly enabled for rollback)
- Sync local `.env` via `python scripts/sync_local_env_from_example.py`

**Interfaces:**
- `DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED` defaults false
- Writer code may remain behind the flag (rollback ladder); no HTTP sql-writer path introduced

- [ ] **Step 1: Update tests for default-off + enabled-still-works**
- [ ] **Step 2: Flip default + `.env_example`; sync `.env`**
- [ ] **Step 3: Run substrate-runtime + cortex-exec focused tests**
- [ ] **Step 4: Commit**

---

### Task 4: Docs, review, PR

**Files:**
- Create: `docs/superpowers/pr-reports/2026-07-16-chat-stance-postgres-drive-measurement-pr.md`
- Optionally check off design acceptance bullets in the cypher-native split design (drive items only)

- [ ] **Step 1: Run focused pytest + sync env parity checks**
- [ ] **Step 2: Code review subagent; fix material findings**
- [ ] **Step 3: `scripts/safe_graphify_update.sh`**
- [ ] **Step 4: Push + `gh pr create` + PR report**

## Acceptance checks (this slice)

- [ ] Chat stance drive projection reads Postgres `drive_audits` rail (fail-open)
- [ ] Graph drive snapshot materialization default off; no new HTTP→sql-writer
- [ ] Tests cover projection + fetch + materialization gate
- [ ] `.env_example` updated and local `.env` synced
- [ ] No Falkor/induction scope creep

## Non-goals

- substrate-runtime Falkor writer cutover
- concept induction → Falkor
- New `drive_state_latest` table
- Deleting `_materialize_drive_state_to_substrate` body in this PR (default-off is enough; delete in a follow-up if live stays green)
