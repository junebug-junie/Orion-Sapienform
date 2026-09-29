# Plan 3 — seeded urgent curiosity runs: Hub button, investigation turn, must-deliver report

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. One fresh
> subagent per task; controller review between tasks.

**Spec:** `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` Parts 2, 2b, 3.
**Branch:** `feat/curiosity-urgent-runs` (worktree `/mnt/scripts/Orion-Sapienform-curiosity-urgent-runs`).
**Python:** `PY=/mnt/scripts/Orion-Sapienform/.venv/bin/python`; run each service suite separately with
`PYTHONPATH=<worktree root>` (a combined run collides on `app` packages).

## Goal

Juniper types a question in the Hub Curiosity panel and presses "Run urgent". Hub collects an
evidence bundle (the hardware readings right now), starts a curiosity durable run whose GPU admission
says `priority: urgent` (Plan 2 makes that jump the queue), and the turn runs a focused
investigation prompt instead of Orion's self-directed one. Orion's answer includes a structured
`:IncidentReport` graph node. Whatever happens — a good report, no structured verdict, a failure, a
timeout, or no GPU at all — Juniper gets a critical Hub + email notice. Nothing ends silently.

The same path is fed by a bus request (`orion:curiosity:urgent:request`) so Plan 4's hardware
watcher can start urgent runs without touching Hub internals.

## Deviations from the approved spec (decided here, reported to Juniper)

1. **Stall limit stays at 420 s, not 180 s.** The per-step stall limit is a process-wide env value
   in the harness (`HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC`), not a per-turn field. Making it per-turn
   means a new `HarnessRunRequestV1` field and a governor change for a small gain; the turn limit
   (900 s) already bounds the run.
2. **The spec's "7200 s turn" is wrong for held turns.** Held curiosity turns get the brief's
   `timeout_sec` (today 8840 s). Urgent briefs set `timeout_sec = HUB_CURIOSITY_URGENT_TURN_TIMEOUT_SEC`
   (900), which is what actually limits the turn.
3. **Hub dedupes its own notices.** orion-notify stores `dedupe_key` but never enforces it, so Hub
   records "sent" per `(incident_id, kind)` in Redis before calling it done.
4. **Evidence for manual runs lives in the run brief** (it is carried in the seed and persisted in
   durable-runs' run row). The `hardware_watch_incident` table is Plan 4.
5. **Hub restart limitation:** the "no GPU within 120 s" and "20-minute overall" timers are
   in-process tasks. A Hub restart mid-run loses them; the completed/failed report still goes out
   because it rides the durable run-state event. Documented, not fixed here.
6. **Applying the `orion_readonly` grant is an operator step** (a production DB write). The SQL
   file ships; Juniper runs it (command in the PR).

## Global constraints

- **Do not restart, reconfigure, or touch the cooler / AC path** (`orion-zwave`,
  `athena-zwave-js-ui`, the Shelly plug) — Juniper's explicit instruction. Read-only queries of
  `home_cooling_sample` are fine.
- Ordinary curiosity is unchanged: no change to `tick()`, its gates, cooldown, daily cap, prompt, or
  reach-out when `brief.urgent` is absent.
- Urgent runs bypass `_run_lock`, cooldown, daily cap, waking window, energy hold. They do **not**
  consume the daily budget or cooldown (`_record_investigation` is not called).
- Urgent runs require durable admission (`HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED=true`; live Hub
  has it). With admission off, the urgent entry refuses with reason `durable_admission_disabled`
  — it never silently runs at background priority.
- At most one open urgent run per `incident_id`.
- Run ids stay lowercase hex (`^[0-9a-f]{6,32}$`, `orion/curiosity/worldview.py:65`); incident ids
  are `uuid4().hex`.
- New optional schema fields are omitted when `None` on the wire (both producers already dump with
  `exclude_none=True`: Hub `curiosity_investigation.py:3331`, durable-runs `runner.py:337`), so
  services can deploy in any order. Tests assert the key is absent when unset.
- No keyword lists on the question text. The question is passed through verbatim as the
  assignment; nothing branches on its words.
- Privacy: evidence is hardware telemetry and pool state only — no chat, memory, or journal content.
- Never commit `.env`. Never `--no-verify`. Commit per task. Work only in the worktree. If any
  `.env_example` changes, run `$PY scripts/sync_local_env_from_example.py` from repo root.

## Current architecture (grounded)

- **Dispatch:** `CuriosityInvestigation.tick()` (`services/orion-hub/scripts/curiosity_investigation.py:1452`)
  → gates (:1501-1616) → `_investigate` (:1643) builds `build_kickoff_prompt` (:1721) →
  `_dispatch_durable_run` (:3367) sets `ResourceRequirementV1(...)` at :3384 (priority defaults
  `background`) → `_dispatch_via_cortex` (:3300) → cortex-orch → `orion:durable:run:request`.
- **Brief:** `CuriosityRunBriefV1` (`orion/schemas/durable_run.py:108`, `extra="forbid"`), built by
  `_run_brief` (hub :3280). Consumers: cortex-orch (`durable_runs.py:48`), durable-runs.
- **Turn callback:** durable-runs `harness_turn` (`services/orion-durable-runs/app/graph.py:216`)
  sends `CuriosityTurnRequestV1` (`durable_run.py:218`, `extra="forbid"`) with only prompt, model,
  timeout, source_tag, attempt, gpu_lease — **not the brief**. Hub `_turn_result_for` (:3530) →
  `_generate` (:2839) → `execute_unified_turn` (`orion/hub/turn_orchestrator.py:849`).
- **Stance:** `ThoughtClient.react()` (turn_orchestrator :1090). `thought is None` → `turn_deferred`
  (:1170-1195); `disposition in ("defer","refuse")` → `_thought_deferred_frame` (:1196-1214).
  `ThoughtEventV1.disposition: Literal["proceed","defer","refuse"]` (`orion/schemas/thought.py:101`).
  Downstream needs the thought (`HarnessRunRequestV1.thought_event`, :1390-1419), so stance cannot
  be skipped — only its defer/refuse overridden.
- **Structured output:** the model writes `:TurnOutcome` itself via `redis-cli ... GRAPH.QUERY`
  (template `_outcome_section`, `orion/curiosity/kickoff_prompt.py:767`). Reader:
  `outcome_for_run_cypher` / `read_turn_outcome` (`orion/curiosity/worldview.py:695`, :1227).
  durable-runs `DurableRunner._read_turn_result` (`runner.py:422`) returns a dict and never raises.
- **Finish:** `finish_detail` (`graph.py:322`) → run-state event. Hub `_handle_run_state` (:3694)
  handles only `curiosity.investigate` + `completed`, then maybe `_maybe_reach_out` (:3071).
- **Retry:** `admitted_graph.py:154-160` uses service-wide `max_attempts=3`, `30 s·2^n` backoff.
- **Notify:** `orion/notify/client.py:25` `NotifyClient.send(NotificationRequest) -> NotificationAccepted`
  (never raises; `ok=False` on failure). `/notify` emails when severity is `critical` or
  `channels_requested` has `email`; in-app always. Hub has `NOTIFY_BASE_URL` / `NOTIFY_API_TOKEN`
  settings but no sender. Pattern: `services/orion-sql-writer/app/fallback_watch.py:240-290`.
- **Telemetry readers in Hub:** `cabinet_cooling_routes.query_latest_row` + `_load_latest` (:111,
  :203); `cabinet_sensors_routes.query_sensor_history_rows` (:251); `biometrics_node_client.fetch_snapshot`;
  `biometrics_preview_routes` GPU cards (:586); `gpu_pool_routes.feed.snapshot()` (:113).
- **UI:** `services/orion-hub/templates/curiosity_atlas.html` (inline JS; `arm()` at :1358;
  runs list `loadRuns` :1240). Bind-mounted from the shared checkout: live after merge + pull.

---

### Task 1: Contracts — urgent seed, request, channel

**Files:**
- Create: `orion/schemas/curiosity_urgent.py`
- Modify: `orion/schemas/durable_run.py` (`CuriosityRunBriefV1`, `CuriosityTurnRequestV1`)
- Modify: `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- Test: `tests/test_curiosity_urgent_schema.py` (new)

**Produces:**
```python
URGENT_REQUEST_CHANNEL = "orion:curiosity:urgent:request"
Trigger = Literal["manual", "heat", "cooling"]

class CuriosityUrgentSeedV1(BaseModel):      # extra="forbid"
    incident_id: str        # ^[0-9a-f]{12,32}$
    question: str           # 1..2000 chars, stripped
    trigger: Trigger
    subject: str = ""       # e.g. "athena", "circe/gpu2", "cabinet_ac"; <=120 chars
    evidence: dict[str, Any] = {}   # JSON-serialisable; serialized size <= 32_000 bytes (validator)
    requested_at: datetime
    requested_by: str = "hub"

class CuriosityUrgentRequestV1(CuriosityUrgentSeedV1):   # the bus request: same fields
    pass
```
- `CuriosityRunBriefV1.urgent: CuriosityUrgentSeedV1 | None = None`
- `CuriosityTurnRequestV1.urgent: CuriosityUrgentSeedV1 | None = None`

Steps (test-first):
1. Tests: valid seed round-trips; question empty/whitespace/2001 chars rejected; non-hex incident id
   rejected; evidence over 32 000 bytes rejected; `CuriosityRunBriefV1(...).model_dump(mode="json",
   exclude_none=True)` has no `"urgent"` key when unset and has it when set; same for
   `CuriosityTurnRequestV1`; an old-shape payload (no `urgent`) still validates.
2. Implement. Register `CuriosityUrgentRequestV1` in `orion/schemas/registry.py` following its
   existing entries. Add the channel to `orion/bus/channels.yaml` (producers `orion-hub`
   (manual) and `orion-hardware-watch` (Plan 4); consumer `orion-hub`; message kind
   `curiosity.urgent.request.v1`; one-line description pointing at the spec) following sibling entries.
3. Run: `$PY -m pytest tests/test_curiosity_urgent_schema.py -q`, the repo's schema/channel tests
   (`rg -l "channels.yaml" tests | head` — run the ones that load it), and
   `ORION_BUS_URL=redis://100.92.216.81:6379/0 $PY scripts/check_single_consumer_channels.py` if it
   runs offline-safe (record result).

Commit: `feat(curiosity): urgent seed/request contract and orion:curiosity:urgent:request channel`.

### Task 2: Urgent investigation prompt + `:IncidentReport` reader

**Files:**
- Create: `orion/curiosity/urgent_prompt.py`, `orion/curiosity/incident_report.py`
- Test: `tests/test_curiosity_urgent_prompt.py`, `tests/test_curiosity_incident_report.py` (new)

**Consumes:** `CuriosityUrgentSeedV1` (Task 1). Reuses from `orion/curiosity/kickoff_prompt.py`:
`_access_section` (tool guide), `_budget_section` (clock) — import them; do not copy.

**Produces:**
```python
def build_urgent_prompt(seed: CuriosityUrgentSeedV1, *, run_id: str,
                        own_graph: str = "orion_worldview", hub_url: str = "http://127.0.0.1:8080",
                        pool_url: str = "http://orion-athena-gpu-pool:8127",
                        graph_enabled: bool = True) -> str

@dataclass(frozen=True)
class IncidentReport:
    incident_id: str
    is_real: Literal["real", "sensor_fault", "unclear"]
    likely_cause: str
    evidence: tuple[str, ...]
    severity: Literal["low", "high", "critical"]
    operator_action: str
    confidence: float           # clamped to [0, 1]

def incident_report_for_run_cypher(run_id: str) -> str            # refuses non-hex run ids (reuse _RUN_ID_RE)
def read_incident_report(reader, run_id: str) -> tuple[IncidentReport | None, str | None]
#   -> (report, None) valid; (None, "no_structured_verdict") missing/malformed/empty evidence;
#      never raises (reader errors -> (None, "no_structured_verdict"))
```

Prompt sections, in order (no self material, priors, peer briefs, dreams, continuation):
1. Assignment header: "Juniper asked you to investigate this now" (trigger `manual`) or "A hardware
   rule fired" (other triggers) + the question verbatim + subject.
2. Evidence bundle: pretty JSON of `seed.evidence`, truncated to 24 000 chars with a marker.
3. Checklist: confirm real vs sensor fault → likely cause → severity → one concrete operator action.
   Cite readings/queries you actually ran as evidence.
4. Tool guide: `_access_section(...)` output, plus one line naming the readable tables
   `orion_biometrics_summary`, `home_cooling_sample` and the pool mirror
   `GET http://orion-athena-gpu-pool:8127/v1/pool` (verified 2026-09-28: reachable from the
   `orion-athena-harness-governor` sandbox on `app-net`; `127.0.0.1:8127` is not). Make the URL a
   keyword param `pool_url` with that default.
5. Report instruction (only when `graph_enabled`): write exactly one node with `redis-cli ...
   GRAPH.QUERY <own_graph>`:
   `CREATE (:IncidentReport {run_id:"<RUN_ID>", incident_id:"...", is_real:"real|sensor_fault|unclear",
   likely_cause:"...", evidence:["..."], severity:"low|high|critical", operator_action:"...",
   confidence:0.0, written_at: timestamp()})` — mirror `_outcome_section`'s quoting guidance.
   When `graph_enabled` is false: say the prose answer is the report.
6. Clock: `_budget_section(...)`.
7. Closing: answer in plain prose: verdict, cause, action first.

Steps (test-first):
1. Prompt tests: contains the question verbatim and a key from the evidence; contains
   `:IncidentReport` and the run id when graph enabled, not when disabled; contains **none** of the
   kickoff's self-material/priors/peer/dream section headers (import the header strings from
   `kickoff_prompt` where they are constants; otherwise assert on distinctive phrases taken from each
   section function's output for a minimal fixture); evidence over the cap is truncated with the
   marker.
2. Reader tests with a fake reader returning rows (copy the fake used by `read_turn_outcome` tests —
   `rg -n "read_turn_outcome" tests orion | head`): valid row → report; confidence 1.7 → 1.0;
   `is_real:"maybe"` → no_structured_verdict; empty evidence list → no_structured_verdict; no rows →
   no_structured_verdict; reader raises → no_structured_verdict; newest `written_at` wins; bad run id
   → cypher builder raises `ValueError`.
3. Implement; run both test files.

Commit: `feat(curiosity): urgent investigation prompt and IncidentReport reader`.

### Task 3: durable-runs — carry the seed, tighter retries, read the report

**Files:**
- Modify: `services/orion-durable-runs/app/graph.py` (`harness_turn`, `finish_detail`),
  `services/orion-durable-runs/app/runner.py` (`_read_turn_result`),
  `services/orion-durable-runs/app/admitted_graph.py` (retry budget)
- Test: `services/orion-durable-runs/tests/test_urgent_runs.py` (new)

**Consumes:** Task 1 schema fields; Task 2 `read_incident_report`.

**Produces (finish detail keys, present only when `brief.urgent` is set):**
```python
"urgent": {"incident_id", "trigger", "subject", "question", "requested_at"},
"incident_report": {<IncidentReport fields>} | None,
"report_flag": None | "no_structured_verdict",
```

Steps (test-first):
1. Tests:
   - `harness_turn` copies `brief.urgent` onto `CuriosityTurnRequestV1.urgent`; with no seed the
     dumped request has no `"urgent"` key.
   - `_read_turn_result` for an urgent brief calls `read_incident_report` and returns
     `incident_report` / `report_flag`; for a non-urgent brief it does not query it (assert the fake
     reader never saw the IncidentReport cypher).
   - `finish_detail` includes the three keys for urgent, none of them for ordinary runs.
   - Retry budget: an urgent brief gets `max_attempts = min(deps.max_attempts, 2)` and backoff
     `min(retry_max, 10 * 2**(attempt-1))`; ordinary runs unchanged (3, 30 s·2^n).
2. Implement. `_read_turn_result` must keep its never-raise contract (report read inside the same
   guarded block; on error `incident_report=None, report_flag="no_structured_verdict"`).
3. Run: `PYTHONPATH=$PWD $PY -m pytest services/orion-durable-runs/tests -q`.

Commit: `feat(durable-runs): urgent runs carry their seed, retry twice fast, and read the IncidentReport`.

### Task 4: Hub turn path — urgent seed reaches the prompt and overrides a stance deferral

**Files:**
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (`_turn_result_for`, `_generate`)
- Modify: `orion/hub/turn_orchestrator.py` (`execute_unified_turn`)
- Test: `services/orion-hub/tests/test_curiosity_urgent_turn.py` (new), plus an orchestrator test
  next to the existing `execute_unified_turn` stance tests (`rg -n "turn_deferred" tests services/orion-hub/tests | head`)

**Produces:** `execute_unified_turn(..., urgent: bool = False)`.

Behaviour when `urgent=True`:
- `react()` returns `disposition in ("defer","refuse")` → continue with
  `thought.model_copy(update={"disposition": "proceed", "disposition_reasons":
  [*thought.disposition_reasons, f"urgent_override:{original}"]})`. Log
  `urgent_stance_override correlation_id=... original=...`.
- `react()` returns `None` (timeout) → return an error final frame with reason
  `urgent_stance_unavailable` (a run failure, never a deferral).
- When `urgent=False`, every existing branch is byte-for-byte unchanged.

Hub: `_turn_result_for` passes `urgent=request.urgent is not None` into `_generate`, which passes it
to `execute_unified_turn`. The prompt is already the urgent prompt (built at dispatch, Task 5), so
no prompt work here. `MIN_HARNESS_STEPS` still applies (`require_lookup=True`).

Steps (test-first):
1. Orchestrator tests with the existing fake ThoughtClient: urgent + defer → harness request built
   with disposition `proceed` and reason `urgent_override:defer`; urgent + refuse → same with
   `refuse`; urgent + None → error frame `urgent_stance_unavailable`; non-urgent + defer → unchanged
   deferred frame.
2. Hub test: a `CuriosityTurnRequestV1` with `urgent` set reaches `_generate` with `urgent=True`;
   without it, `urgent=False`.
3. Implement; run the touched Hub and orchestrator tests plus
   `PYTHONPATH=$PWD $PY -m pytest services/orion-hub/tests -q -k "curiosity or turn"`.

Anti-slop gate note (`.cursor/rules/conversational-behavior-anti-slop.mdc`): this does not change
Orion's voice; it is an operator-assigned run overriding a stance deferral, keyed on a typed
contract field, never on message text.

Commit: `feat(hub): urgent turns carry their seed and proceed past a stance deferral`.

### Task 5: Hub urgent entry — evidence bundle, dispatch, bus consumer

**Files:**
- Create: `services/orion-hub/scripts/urgent_evidence.py`, `services/orion-hub/scripts/curiosity_urgent.py`
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (new `start_urgent`, dispatch
  priority), `services/orion-hub/scripts/main.py` (start the consumer loop where
  `curiosity_investigation` starts), `services/orion-hub/app/settings.py`,
  `services/orion-hub/.env_example`, `services/orion-hub/docker-compose.yml`, `services/orion-hub/README.md`
- Test: `services/orion-hub/tests/test_curiosity_urgent_start.py`, `test_urgent_evidence.py` (new)

**Consumes:** Tasks 1, 2 (`build_urgent_prompt`).

**Produces:**
```python
# urgent_evidence.py
async def collect_evidence(*, nodes=("athena", "circe"), per_section_timeout=5.0) -> dict[str, Any]
#   sections: "cooling" (latest + freshness via query_latest_row/_load_latest),
#   "cabinet_trend" (last 60 min cabinet_temp_c via query_sensor_history_rows),
#   "hosts" (per node biometrics snapshot: temp_c_max, fan, power),
#   "gpus" (per-GPU util/power/memory cards), "pool" (active leases from gpu_pool_routes.feed.snapshot()),
#   "collected_at". A failing/slow section becomes {"error": "<type>: <msg>"}; never raises.
#   Whole bundle trimmed to fit the 32 000-byte seed cap (drop oldest trend points first).

# curiosity_investigation.py
async def start_urgent(self, seed: CuriosityUrgentSeedV1) -> dict[str, Any]
#   -> {"ok": True, "run_id", "incident_id"} or {"ok": False, "reason"}
#   reasons: "urgent_disabled", "durable_admission_disabled", "incident_already_open", "dispatch_failed"
```

`start_urgent`:
- Refuse if `HUB_CURIOSITY_URGENT_ENABLED` is false or durable admission is off.
- `SET NX` Redis key `orion:curiosity:urgent:open:{incident_id}` = run_id, TTL
  `HUB_CURIOSITY_URGENT_TIMEOUT_SEC + 600`; if taken → `incident_already_open`.
- `run_id = uuid4().hex[:12]`; correlation id as `_investigate` makes it.
- `prompt = build_urgent_prompt(seed, run_id=run_id, graph_enabled=self._reader is not None, ...)`.
- `self._mind_appraisal_by_run_id[run_id] = seed.question` (becomes the stance user message).
- Brief via `_run_brief` plus `urgent=seed`, `timeout_sec=HUB_CURIOSITY_URGENT_TURN_TIMEOUT_SEC`.
- Dispatch through `_dispatch_durable_run(..., priority="urgent", urgent=seed)` — add those two
  keyword params (defaults keep today's behaviour) and set `ResourceRequirementV1(priority=...)`.
- No `_run_lock`, no `_record_investigation`, no gate calls.
- Record the incident in a Redis hash `orion:curiosity:urgent:incidents` (field incident_id →
  JSON `{incident_id, run_id, question, trigger, subject, requested_at, status: "dispatched"}`),
  capped to the newest 50 fields. On dispatch failure: status `dispatch_failed`, release the NX key,
  and hand to the reporter (Task 6) so the failure is still reported.
- Start the watchdog (Task 6 provides `UrgentReporter.watch(incident)`).

`curiosity_urgent.py`: `urgent_request_loop(bus, investigation)` subscribes to
`URGENT_REQUEST_CHANNEL`, validates `CuriosityUrgentRequestV1`, calls `start_urgent`; invalid
payloads are logged and dropped. Follow `_turn_request_loop` (:3439) for subscribe/decode style.

Env (Hub `.env_example` + settings + compose, then sync local `.env`):
```
HUB_CURIOSITY_URGENT_ENABLED=true
HUB_CURIOSITY_URGENT_TURN_TIMEOUT_SEC=900
HUB_CURIOSITY_URGENT_TIMEOUT_SEC=1200
HUB_CURIOSITY_URGENT_GRANT_WAIT_SEC=120
```

Steps (test-first):
1. Evidence tests: each section success shape; one section raising → `{"error": ...}` and the rest
   present; slow section → timeout error; oversized bundle trimmed under the cap.
2. `start_urgent` tests (fakes for redis, cortex dispatch, reporter): admission `priority ==
   "urgent"` on the dispatched request; brief has `urgent` and `timeout_sec == 900`; prompt is the
   urgent prompt; `_run_lock` held by another task does not block it; cooldown/daily cap untouched
   (assert `_record_investigation` not called); second call with same incident → refused; admission
   off → refused; dispatch failure → reporter told, NX key released.
3. Consumer test: a valid bus payload calls `start_urgent`; invalid payload dropped without raising.
4. Implement; `$PY scripts/sync_local_env_from_example.py`; `$PY scripts/check_env_template_parity.py`.

Commit: `feat(hub): start urgent curiosity runs with an evidence bundle, bypassing curiosity gates`.

### Task 6: Must-deliver report

**Files:**
- Create: `services/orion-hub/scripts/urgent_report.py`
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (`_handle_run_state`: urgent
  completed **and failed** → report; skip reach-out for urgent)
- Test: `services/orion-hub/tests/test_urgent_report.py` (new)

**Consumes:** finish detail keys (Task 3), incident hash (Task 5).

**Produces:**
```python
def compose_urgent_report(incident: dict, *, kind: Literal["final", "failed", "timeout", "no_gpu"],
                          detail: dict | None = None, reason: str = "") -> NotificationRequest
class UrgentReporter:
    def __init__(self, *, notify: NotifyClient, redis, settings, run_state_reader): ...
    async def deliver(self, incident: dict, request: NotificationRequest, *, kind: str) -> bool
    def watch(self, incident: dict) -> None      # schedules no_gpu + timeout checks
```

Report rules (all: `severity="critical"`, `channels_requested=["in_app","email"]`,
`source_service="orion-hub"`, `event_kind="curiosity.urgent.report"`,
`dedupe_key=f"urgent:{incident_id}:{kind}"`, `correlation_id=run_id`):
- **Body order:** flags line (INCOMPLETE / `no_structured_verdict` / "not investigated" when
  present) → verdict line (`is_real` / severity / confidence) → operator action → likely cause →
  cited evidence → Orion's prose (`finding_text`) → the question.
- `final`: from `detail.incident_report`; `report_flag` → flag shown, prose attached.
- `failed`: title "Urgent investigation failed", `investigation failed: <reason>` + the raw
  evidence bundle (pretty JSON, capped).
- `timeout` (no terminal state by `HUB_CURIOSITY_URGENT_TIMEOUT_SEC`): INCOMPLETE + evidence; the
  run keeps going and a later final/failed report still goes out.
- `no_gpu` (run not past `resource_wait` by `HUB_CURIOSITY_URGENT_GRANT_WAIT_SEC`): "not
  investigated" + fired rule/question + evidence bundle. Run keeps going.
- Title leads with the verdict when present, e.g. `URGENT: real / critical — <subject or question[:60]>`.
- **Delivery:** skip if Redis `orion:curiosity:urgent:sent:{incident_id}:{kind}` exists. Else
  `notify.send`; on `ok=False` retry with backoff 2, 4, 8, 16, 32, 60, 60, … seconds up to 30 minutes,
  logging `urgent_report_retry incident_id=... kind=... attempt=...`. On accept set the sent key
  (TTL 7 days) and update the incident hash status. After 30 minutes, log
  `urgent_report_undelivered` at ERROR and set incident status `report_undelivered`.
- `_handle_run_state`: for `curiosity.investigate` with `detail.urgent` present: `completed` → final
  report; `failed` → failed report (reason from `state.detail` error field — check what the failed
  transition carries in `admitted_graph.py` and use it). Release the NX open key. Never call
  `_maybe_reach_out` for urgent runs. Ordinary runs: behaviour unchanged (test it).
- `run_state_reader` for the watchdog: read the run's latest node/status from
  `substrate_durable_run_state` the same way `curiosity_run_store` does.

Steps (test-first):
1. Composer tests (pure): each kind's title/body order/flags; severity critical; both channels;
   dedupe key; no chat/memory fields leak (only incident + detail fields used).
2. Reporter tests with a fake NotifyClient: `ok=False` twice then `ok=True` → delivered once, sent
   key set; already-sent key → no call; 30-minute exhaustion (patch sleep/clock) → undelivered log +
   status.
3. Watchdog tests: run still in `resource_wait` at 120 s → `no_gpu` sent; run past it → none; no
   terminal status at 1200 s → `timeout` sent; terminal before → none.
4. `_handle_run_state` tests: urgent completed → final report, no reach-out; urgent failed → failed
   report; ordinary completed → reach-out path unchanged, no report.
5. Implement; run Hub curiosity tests.

Commit: `feat(hub): every urgent run ends in a critical Hub + email report`.

### Task 7: Hub API + Curiosity panel UI + grant SQL

**Files:**
- Modify: `services/orion-hub/scripts/curiosity_routes.py`, `services/orion-hub/templates/curiosity_atlas.html`
- Create: `scripts/sql/2026-09-28_grant_orion_readonly_hardware.sql`
- Test: `services/orion-hub/tests/test_curiosity_urgent_routes.py` (new), extend
  `tests/test_curiosity_atlas_template.py`

**Routes:**
- `POST /curiosity/api/urgent` body `{"question": str}` → 400 `question_required` / `question_too_long`;
  503 `loop_not_running`; builds a `CuriosityUrgentRequestV1` (trigger `manual`, subject `""`,
  `requested_by="juniper"`, evidence from `collect_evidence()`), publishes it on
  `URGENT_REQUEST_CHANNEL`, returns `{"ok": true, "incident_id": ...}`. (Publishing, not calling
  `start_urgent` directly, keeps one path for manual and Plan 4 requests.)
- `GET /curiosity/api/urgent` → newest-first list from the incident hash (max 20).

**UI (inline in `curiosity_atlas.html`, match existing styles):** a textarea (maxlength 2000) +
"Run urgent" button near the existing run-now buttons; confirm dialog; POST JSON; show
`incident <id> started` or `Refused: <reason>`; an "Urgent runs" list (question, trigger, status,
time, link to the run via existing `openRun(run_id)` when present), refreshed on `load()`.

**Grant SQL** (header comment in the style of `2026-09-18_grant_orion_readonly_self_questions.sql`:
reason, apply command, undo, verify):
```sql
BEGIN;
GRANT SELECT ON public.orion_biometrics_summary TO orion_readonly;
GRANT SELECT ON public.home_cooling_sample TO orion_readonly;
COMMIT;
```

Steps (test-first):
1. Route tests with a fake bus/investigation: valid question publishes one request with trigger
   `manual` and returns an incident id; empty → 400; 2001 chars → 400; no loop → 503; list returns
   newest first.
2. Template test: the page contains the textarea, the button id, `fetch("/curiosity/api/urgent"`,
   a JSON `Content-Type`, and the urgent list container.
3. Implement; run Hub route + template tests.

Commit: `feat(hub): Run urgent button, urgent runs list, and readonly hardware grant`.

### Task 8: Eval, docs, gates, review, PR

1. **Eval** `services/orion-hub/evals/run_urgent_report_eval.py`: replay fixtures through the real
   `read_incident_report` (fake graph rows) → `compose_urgent_report` for: valid report,
   malformed report, empty-evidence report, failed run, timeout, no GPU. Print one line per case
   (kind, flag, title) and exit 1 if any case is missing its flag, isn't critical, lacks email, or
   puts the prose before the verdict. Plain structure like `run_pool_day_eval.py`'s verdict line.
2. **Docs:** spec Parts 2/2b/3 updated to what shipped (the deviations above; 8840 s correction;
   notify dedupe note; stall stays 420 s); Hub README "Urgent curiosity runs" section (what it does,
   env keys, the grant step, rollback `HUB_CURIOSITY_URGENT_ENABLED=false`).
3. **Gates:** durable-runs tests; Hub tests `-k "curiosity or urgent or turn"`; root curiosity tests;
   cortex-orch tests (brief parsing); the eval; `check_env_template_parity.py`;
   `check_definition_drift.py --gate` (re-lock with `--update` only if it fails for a definition this
   branch changed); `git diff --check`.
4. **Review:** whole-branch review subagent; fix findings; re-review.
5. **PR** (§18 template + deviations section); push `-u`; watch CI; resolve conflicts.
6. **Deploy (after merge, Juniper confirms):** order is free (optional fields omitted when unset);
   rebuild `orion-cortex-orch`, `orion-durable-runs`, `orion-hub`; `git pull --ff-only` on the shared
   checkout for the template. Juniper applies the grant SQL.
7. **Live smoke (after deploy):** one "Run urgent" from the Hub with a harmless question → pool
   shows an `urgent` hold (this also exercises Plan 2's pause path live if the seat is full) →
   run completes → critical notice in Hub → **Juniper confirms the email arrived**. Anything not
   seen stays UNVERIFIED.

## Acceptance

- Spec 5a: seed bypasses stance defer; urgent prompt has question + evidence and no self-material
  sections; valid `:IncidentReport` → report; missing/empty-evidence → `no_structured_verdict`;
  timeout → INCOMPLETE; no grant in 120 s → "not investigated" with evidence.
- Ordinary curiosity runs: unchanged (tests on `tick`, reach-out, brief/turn-request wire shape).
- Spec 6 (live Hub → urgent hold → report in Hub + email): post-deploy smoke, UNVERIFIED until run.
