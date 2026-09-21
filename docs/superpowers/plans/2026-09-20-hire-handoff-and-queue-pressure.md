# Hire Handoff + Queue Contention Pressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Orion strongly prefer Cursor on Mind-`deep` sittings, hand off after ≥2 access refusals, resume (not re-hire) when Cursor budget is spent, and see one official digester EWMA queue-contention score (0–10) — not Hub shadow state, not raw counts.

**Architecture:** Teach + pure disclosure formatters first (curiosity package). Then a FieldState-backed digester producer (sibling to `sustained_load_pressure`) with full metric semantic-layer registration and CI static gates. Hub only reads FieldState and passes progress strings into the existing splice path.

**Tech Stack:** Python 3, pytest, `orion/curiosity/*`, `orion/schemas/field_state.py`, `orion/bus/ewma.py::compute_ewma_update`, `services/orion-field-digester`, `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`, Hub `turn_orchestrator` / curiosity investigation.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-20-hire-handoff-and-queue-pressure-design.md` (PR #2259).
- Python never MERGEs `:InvestigationRole` or `:HelpRequest`.
- No Hub Redis / Hub-scheduler EWMA for queue pressure; digester owns the meter.
- No raw-count disclosure to Orion (score + driving source only).
- Score scale locked: **0.0–10.0** on `FieldStateV1.queue_contention_score` (quiet tick = real `0.0`).
- Metric must pass `check_metric_lineage.py --gate`, `check_definition_drift.py --gate`, `check_inner_state_registry.py` in the same PR that introduces the FieldState fields + Hub consumer.
- Conversational anti-slop: no keyword detectors on Juniper chat text; denial counter is hop/tool scoped allow-list only.
- Work in a linked worktree off current `main` (not shared checkout); commit as you go.
- Do not stage `.env`.

## File map

| File | Responsibility |
| --- | --- |
| `orion/curiosity/kickoff_prompt.py` | Role teach wording (no crawl-first bias; Cursor = normal deep hands) |
| `orion/curiosity/role_teach_disclosure.py` | Deep strong line; format denial/budget/queue progress strings |
| `orion/curiosity/access_refusals.py` | Pure: count allow-listed refusal matches in hop notes |
| `orion/curiosity/queue_contention_disclosure.py` | Pure: FieldState reading → one disclosure line (no recompute) |
| `orion/field/queue_contention.py` | Pure: EWMA + `max()` score from three counts |
| `services/orion-field-digester/app/digestion/queue_contention.py` | Digester tick: read sources, update FieldState |
| `orion/schemas/field_state.py` | Additive score / driver / EWMA fields |
| `orion/inner_state_registry.py` | Register signal + cognition consumer (hire disclosure) |
| `config/metrics/metric_definitions.lock.json` | Re-lock after registry/FieldState change |
| `orion/hub/turn_orchestrator.py` | Splice accepts progress_lines; fail-open without mind shape |
| `services/orion-hub/scripts/curiosity_investigation.py` | Gather denials / budget / FieldState → progress_lines |

---

### Task 1: Deep strong nudge in disclosure + teach rewrite

**Files:**
- Modify: `orion/curiosity/role_teach_disclosure.py`
- Modify: `orion/curiosity/kickoff_prompt.py` (`_role_and_help_section`)
- Test: `tests/test_role_teach_disclosure.py`
- Test: existing kickoff teach tests under `orion/curiosity/tests/` (extend or add `test_role_teach_deep_nudge.py`)

**Interfaces:**
- Consumes: existing `format_role_teach_disclosure(mind_work_shape, progress_lines=())`
- Produces: when `expected_depth == "deep"`, lines include an explicit strong handoff sentence (not only `- expected depth: deep`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_role_teach_disclosure.py

def test_format_deep_includes_strong_hire_cursor_nudge() -> None:
    lines = format_role_teach_disclosure(
        {
            "expected_depth": "deep",
            "cross_cutting": "yes",
            "foresight_note": "Lease TTL archaeology.",
        }
    )
    text = "\n".join(lines).lower()
    assert "hire_cursor" in text or "hire cursor" in text
    assert "strongly" in text or "strong" in text
    assert "expensive" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_role_teach_disclosure.py::test_format_deep_includes_strong_hire_cursor_nudge -v`
Expected: FAIL (strong nudge sentence missing)

- [ ] **Step 3: Implement formatter nudge + teach rewrite**

In `format_role_teach_disclosure`, after work-shape bullets when `depth == "deep"` (case-insensitive), append one fixed advisory sentence:

```text
Mind reads this sitting as deep work. Strongly prefer hire_cursor for the archaeology; keep a short local look only so tried_summary is grounded. You still author priors and findings.
```

In `_role_and_help_section` (`kickoff_prompt.py`):

- Change MERGE example to show both choices without crawl-first bias, e.g. `choice: "local_crawl|hire_cursor"`.
- Keep: role ≠ enqueue; peer read-only; HelpRequest after short look for `tried_summary`.
- Do not call Cursor expensive or last-resort.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_role_teach_disclosure.py orion/curiosity/tests/ -q -k "role_teach or kickoff or InvestigationRole or help_request" `
Expected: PASS for touched tests

- [ ] **Step 5: Commit**

```bash
git add orion/curiosity/role_teach_disclosure.py orion/curiosity/kickoff_prompt.py tests/test_role_teach_disclosure.py
git commit -m "feat(curiosity): strong Mind-deep hire_cursor nudge in role teach"
```

---

### Task 2: ≥2 access-refusal progress line

**Files:**
- Create: `orion/curiosity/access_refusals.py`
- Create: `orion/curiosity/tests/test_access_refusals.py`
- Modify: `orion/curiosity/role_teach_disclosure.py` (optional helper `format_access_refusal_progress(count: int) -> list[str]`)
- Modify: `tests/test_role_teach_disclosure.py`

**Interfaces:**
- Consumes: hop note strings `Sequence[str]` (from `read_hop_notes` second tuple element)
- Produces:
  - `count_access_refusals(notes: Sequence[str]) -> int`
  - Allow-list substrings (case-insensitive): `"permission denied"`, `"permissiondenied"`, `"acl"`, `"insufficient_privilege"` — keep the list short and documented in module docstring; do not expand into a feelings/keyword cathedral

- [ ] **Step 1: Write the failing tests**

```python
# orion/curiosity/tests/test_access_refusals.py
from orion.curiosity.access_refusals import count_access_refusals, ACCESS_REFUSAL_THRESHOLD
from orion.curiosity.role_teach_disclosure import format_access_refusal_progress

def test_count_two_permission_denied() -> None:
    notes = [
        "psql: permission denied for table durable_admission_runs",
        "second hop: Permission Denied reading schema",
        "ordinary bash ok",
    ]
    assert count_access_refusals(notes) >= 2

def test_progress_line_only_at_threshold() -> None:
    assert format_access_refusal_progress(1) == []
    lines = format_access_refusal_progress(2)
    assert lines
    assert "hand off" in "\n".join(lines).lower()
    assert "cursor" in "\n".join(lines).lower()
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest orion/curiosity/tests/test_access_refusals.py -v`
Expected: FAIL import / missing symbols

- [ ] **Step 3: Minimal implementation**

```python
# orion/curiosity/access_refusals.py
ACCESS_REFUSAL_THRESHOLD = 2
_NEEDLES = ("permission denied", "permissiondenied", "insufficient_privilege")

def count_access_refusals(notes: Sequence[str]) -> int:
    n = 0
    for note in notes:
        low = str(note).lower()
        if any(needle in low for needle in _NEEDLES):
            n += 1
    return n
```

```python
# in role_teach_disclosure.py
def format_access_refusal_progress(count: int) -> list[str]:
    if count < 2:
        return []
    return [
        "Access refused at least twice this sitting. Hand off to Cursor now "
        "(write hire_cursor + HelpRequest with what you already tried)."
    ]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest orion/curiosity/tests/test_access_refusals.py tests/test_role_teach_disclosure.py -q`

- [ ] **Step 5: Commit**

```bash
git add orion/curiosity/access_refusals.py orion/curiosity/tests/test_access_refusals.py orion/curiosity/role_teach_disclosure.py tests/test_role_teach_disclosure.py
git commit -m "feat(curiosity): ≥2 access-refusal handoff progress line"
```

---

### Task 3: Budget-spent resume progress line

**Files:**
- Modify: `orion/curiosity/role_teach_disclosure.py` (add `format_budget_spent_progress(...)`)
- Modify: `orion/curiosity/peer_briefs.py` only if a tiny pure helper is cleaner than duplicating status checks — prefer keeping format in `role_teach_disclosure.py`
- Test: `tests/test_role_teach_disclosure.py` or `orion/curiosity/tests/test_budget_spent_progress.py`

**Interfaces:**
- Consumes: `status: str` (PeerBrief status) and optional `next_hop_n: int | None`
- Produces: progress lines when `status == "refused_budget"`

- [ ] **Step 1: Failing test**

```python
def test_budget_spent_progress_names_resume_not_rehire() -> None:
    lines = format_budget_spent_progress(status="refused_budget", next_hop_n=3)
    text = "\n".join(lines).lower()
    assert "budget" in text
    assert "helprequest" in text.replace(" ", "") or "help request" in text
    assert "do not" in text or "don't" in text
    assert "resume" in text or "hop" in text
    assert format_budget_spent_progress(status="ok", next_hop_n=3) == []
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
def format_budget_spent_progress(*, status: str, next_hop_n: int | None = None) -> list[str]:
    if status != "refused_budget":
        return []
    hop = f"hop {next_hop_n}" if next_hop_n is not None else "your last hop notes"
    return [
        f"Cursor budget is spent. Do not open another HelpRequest until budget is clear. "
        f"Resume from {hop} / continue local crawl from what you already wrote."
    ]
```

- [ ] **Step 4: PASS + commit**

```bash
git commit -m "feat(curiosity): budget-spent resume progress line for role teach"
```

---

### Task 4: Hub splice accepts progress_lines without requiring mind_work_shape

**Why:** Today `_maybe_splice_role_teach_disclosure` returns early if `not mind_work_shape`, so denial/budget/queue-only progress can never splice. Spec requires progress_lines to work even when Mind fails open.

**Files:**
- Modify: `orion/hub/turn_orchestrator.py` (`_maybe_splice_role_teach_disclosure`)
- Modify: `services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py`

**Interfaces:**
- Consumes: `progress_lines: Sequence[str] = ()`
- Produces: spliced prompt when origin=`orion`, flag on, and (`mind_work_shape` or `progress_lines`) yields formatter lines

- [ ] **Step 1: Failing test**

```python
def test_splice_progress_only_without_mind_work_shape():
    prompt = "intro\nASKING FOR CONTRACTOR HELP. rest"
    out = _maybe_splice_role_teach_disclosure(
        prompt,
        utterance_origin="orion",
        mind_work_shape=None,
        enabled=True,
        progress_lines=["Access refused at least twice this sitting. Hand off to Cursor now."],
    )
    assert "Hand off to Cursor" in out
    assert out.index("Hand off") < out.index("ASKING FOR CONTRACTOR HELP")
```

- [ ] **Step 2: Run — expect FAIL** (keyword arg / early return)

- [ ] **Step 3: Patch helper**

```python
def _maybe_splice_role_teach_disclosure(
    user_message: str,
    *,
    utterance_origin: str | None,
    mind_work_shape: Mapping[str, Any] | None,
    enabled: bool,
    progress_lines: Sequence[str] = (),
) -> str:
    if not enabled or utterance_origin != "orion":
        return user_message
    if not mind_work_shape and not progress_lines:
        return user_message
    from orion.curiosity.role_teach_disclosure import (
        format_role_teach_disclosure,
        splice_role_teach_disclosure,
    )
    lines = format_role_teach_disclosure(mind_work_shape, progress_lines=progress_lines)
    if not lines:
        return user_message
    return splice_role_teach_disclosure(user_message, lines)
```

Update the call site (~line 1267) to pass `progress_lines=` once Task 7 gathers them; until then pass `()` (tests can call the helper directly).

- [ ] **Step 4: PASS + commit**

```bash
git commit -m "fix(hub): role-teach splice allows progress-only disclosure"
```

---

### Task 5: Metric quality gate evidence (blocking before digester wire)

**Files:**
- Create: `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md` (short evidence appendix; or a section appended to the parent design)
- Optional analysis script under `scripts/analysis/` only if needed to pull history — prefer one-off documented `psql` + gateway snapshot results pasted into the gate doc

**Interfaces:**
- Produces: written answers for CLAUDE.md §0A items 1–6 for `queue_contention_score`, including live multi-day sanity and independence vs `gpu_pressure` / `sustained_load_pressure`

- [ ] **Step 1: Pull multi-day histories** (operator machine with DB + gateway)

For each source, record min/max/median and whether calm (~0 or near baseline) ever appears:

```bash
# seeds
docker exec -e PGPASSWORD="$PW" orion-athena-sql-db psql -U postgres -d conjourney -c "
SELECT date_trunc('hour', created_at) AS hr, count(*) FILTER (WHERE status='pending')
FROM world_pulse_read_seed GROUP BY 1 ORDER BY 1 DESC LIMIT 72;"

# durable pending (if history is only current row, note UNVERIFIED for time series and use demand created_at churn)
# gateway: curl admission snapshot over time or logs — document what is available
```

- [ ] **Step 2: Write gate doc** with explicit verdicts:

```markdown
## queue_contention_score — metric quality gate

1. Provenance: ...
2. Independence: vs gpu_pressure / sustained_load_pressure / cortex_exec_step_load — ...
3. Theory: shared agent/curiosity queue contention ...
4. Live sanity: source A can calm? Y/N; drop source X if degenerate
5. Existing mechanism: rg results ...
6. Reversibility: FieldState additive + lock/registry ...
```

If a source is degenerate, **remove it from the three-source set in Task 6** before coding (do not leave a dead `max()` limb).

- [ ] **Step 3: Commit gate doc**

```bash
git commit -m "docs(field): queue contention metric quality gate evidence"
```

**Stop rule:** Do not start Task 6 until gate item 4 is answered with real numbers (or explicit `UNVERIFIED` + Juniper go-ahead). Prefer real numbers.

---

### Task 6: Pure score math + FieldState fields + digester producer

**Files:**
- Create: `orion/field/queue_contention.py`
- Create: `orion/field/tests/test_queue_contention.py` (or `tests/test_queue_contention.py` if that is the local convention — match `orion/field/significance` test location)
- Modify: `orion/schemas/field_state.py`
- Create: `services/orion-field-digester/app/digestion/queue_contention.py`
- Modify: `services/orion-field-digester/app/tensor/update_rules.py`
- Create: `services/orion-field-digester/tests/test_queue_contention_digestion.py`

**Interfaces:**
- Consumes: `compute_ewma_update` from `orion/bus/ewma.py`; three floats `seed_pending`, `durable_pending`, `gateway_waiting`
- Produces:
  - `QueueContentionReading(score: float, driver: str | None, raw: dict[str, float], ewma: dict[str, float])`
  - `score_queue_contention(counts: Mapping[str, float], prev_ewma: Mapping[str, float], prev_n: Mapping[str, int], *, alpha: float, floor: float = 1.0) -> QueueContentionReading`
  - Digester: `update_queue_contention_pressure(state, *, counts, alpha, ...) -> FieldStateV1`
  - FieldState fields:
    - `queue_contention_score: float = 0.0`  # 0–10
    - `queue_contention_driver: str | None = None`
    - `queue_contention_ewma: dict[str, float] = Field(default_factory=dict)`
    - `queue_contention_ewma_n: dict[str, int] = Field(default_factory=dict)`
    - `queue_contention_computed_at: datetime | None = None`

Source keys (exact strings): `world_pulse_seed_pending`, `durable_demand_pending`, `gateway_waiting` (omit any dropped in Task 5).

Score formula (locked):

```text
ratio = count / max(ewma, floor)
sub = clip(10 * (ratio - 1) / 4, 0, 10)   # 1x→0, 5x→10
score = max(subs)
driver = argmax(subs)  # None if all ~0
```

Alpha: start from half-life ~24h at digester tick ~2s → `alpha = 1 - exp(-ln(2) * dt / half_life_sec)` with `dt` from tick interval settings; document the constant in settings/env if added.

- [ ] **Step 1: Failing pure tests**

```python
def test_at_baseline_scores_zero():
    reading = score_queue_contention(
        {"world_pulse_seed_pending": 100.0},
        prev_ewma={"world_pulse_seed_pending": 100.0},
        prev_n={"world_pulse_seed_pending": 50},
        alpha=0.01,
    )
    assert reading.score == 0.0

def test_five_x_baseline_scores_ten():
    reading = score_queue_contention(
        {"durable_demand_pending": 10.0},
        prev_ewma={"durable_demand_pending": 2.0},
        prev_n={"durable_demand_pending": 50},
        alpha=0.0,  # freeze ewma for this assert if API allows; else pre-seed and alpha tiny
    )
    assert reading.score == 10.0
    assert reading.driver == "durable_demand_pending"

def test_max_not_average():
    # one hot source must dominate
    ...
```

- [ ] **Step 2: FAIL then implement `orion/field/queue_contention.py`**

- [ ] **Step 3: Add FieldState fields with docstrings** (quiet `0.0` / `None` are real calm/absent-driver readings; cite independence theory vs `gpu_pressure` / `sustained_load_pressure`)

- [ ] **Step 4: Digester wrapper reads counts**

Implement count readers as injectable callables in tests; production:

- SQL: `SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'`
- SQL: `SELECT count(*) FROM durable_resource_demands WHERE status='pending'`
- HTTP: gateway `/admission` → sum `waiting` across `upstreams` (fail-open: skip gateway key if unreachable)

Wire `update_queue_contention_pressure` into `update_rules.py` **after** significance, **before** dimension precision baseline (same ordering comment pattern).

- [ ] **Step 5: Digester unit test with fake store/counts — PASS**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(field): queue contention EWMA score on FieldState"
```

---

### Task 7: Semantic layer registration + static gates green

**Files:**
- Modify: `orion/inner_state_registry.py`
- Optionally: `config/field/field_channel_glossary.v1.yaml` only if you expose a field channel; **prefer FieldState scalar + inner-state entry first** (like other FieldState scalars) unless glossary is required for Hub panel — if required, add a short glossary row
- Modify: `config/metrics/metric_definitions.lock.json` via `--update`
- Ensure Hub consumer code from Task 8 is either already landed or land Tasks 7+8 in one PR so lineage is not orphan

**Interfaces:**
- Inner-state entry for `field_state.v1#queue_contention_score` (follow existing FieldState scalar patterns in the lock)
- Cognition consumer named: hire role-teach disclosure (`orion/hub/turn_orchestrator.py` / curiosity path)

- [ ] **Step 1: Add registry entry** naming producer `orion-field-digester`, cadence per digester tick, composition, consumer paths

- [ ] **Step 2: Re-lock definitions**

```bash
python3 scripts/check_definition_drift.py --update
python3 scripts/check_definition_drift.py --gate
python3 scripts/check_inner_state_registry.py
```

Expected: gate clean after update; `_last_change` mentions the new metric in plain English

- [ ] **Step 3: Lineage gate**

```bash
python3 scripts/check_metric_lineage.py --gate
python3 scripts/check_metric_lineage.py --metric queue_contention_score
```

Expected: producer digester; consumer Hub hire disclosure visible after Task 8 lands. If Task 8 is separate commit on same PR, run gate after Task 8.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(metrics): register queue_contention_score; refresh metric lock"
```

---

### Task 8: Hub read-only disclosure wire (denials + budget + FieldState score)

**Files:**
- Create: `orion/curiosity/queue_contention_disclosure.py`
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` and/or `orion/hub/turn_orchestrator.py`
- Modify: `services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py`
- Mirror pattern: `services/orion-hub/scripts/tension_outreach_trigger.py` SQL against `substrate_field_state` for latest score fields

**Interfaces:**
- `format_queue_contention_progress(score: float | None, driver: str | None) -> list[str]`
  - omit if `score is None` or read failed
  - omit if `score <= 0` (optional: still show when score ≥ some soft threshold like 3 — default: show when `score > 0`)
  - never include raw counts
- Curiosity turn path builds `progress_lines = refusal + budget + queue` and passes into `_maybe_splice_role_teach_disclosure`

- [ ] **Step 1: Failing unit test for formatter**

```python
def test_queue_line_names_score_and_driver_not_raw_count():
    lines = format_queue_contention_progress(score=8.0, driver="durable_demand_pending")
    text = "\n".join(lines)
    assert "8" in text and "/10" in text
    assert "durable" in text.lower() or "demand" in text.lower()
    assert "pending" not in text.lower() or "121" not in text  # no raw backlog dump
```

- [ ] **Step 2: Implement formatter**

Example line:

```text
Queue pressure: 8/10 (high) — durable GPU demand is running well above its normal level. Deep Cursor digs compete with that. Prefer hire when Mind says deep.
```

- [ ] **Step 3: Wire curiosity / orchestrator**

Preferred shape (keeps FieldState IO out of pure formatter):

1. In `curiosity_investigation` before `execute_unified_turn`, or inside orchestrator when `utterance_origin=="orion"`:
   - `notes = read_hop_notes(...)` → refusal progress
   - unused/recent PeerBrief for run → budget progress
   - latest `substrate_field_state` JSON → score/driver → queue progress
2. Pass `progress_lines` into splice.

Fail-open each source independently.

- [ ] **Step 4: Hub tests with fake FieldState payload — PASS**

- [ ] **Step 5: Re-run static gates** (must be green with Hub consumer present)

```bash
python3 scripts/check_metric_lineage.py --gate
python3 scripts/check_definition_drift.py --gate
python3 scripts/check_inner_state_registry.py
pytest tests/test_role_teach_disclosure.py orion/curiosity/tests/test_access_refusals.py services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py services/orion-field-digester/tests/test_queue_contention_digestion.py -q
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(hub): disclose official queue contention score into role teach"
```

- [ ] **Step 7: Grep acceptance — no Hub EWMA**

```bash
rg -n "orion:hire:queue_pressure|queue_pressure:ewma" services/orion-hub orion/hub || echo OK
```

Expected: no matches

---

### Task 9: Docs cross-link + PR report

**Files:**
- Modify: `orion/curiosity/README.md` (hire disclosure section)
- Modify: digester README short note for the new digestion step
- Create: `docs/superpowers/pr-reports/2026-09-20-hire-handoff-and-queue-pressure-pr.md` (use repo PR template sections)

- [ ] **Step 1: Update READMEs** pointing at spec + score field names

- [ ] **Step 2: PR report** includes gate evidence path, static gate commands/results, restart commands for digester + hub

- [ ] **Step 3: Commit + open/update impl PR**

```bash
git commit -m "docs(curiosity): hire handoff + queue contention impl notes"
```

---

## Spec coverage self-check

| Spec requirement | Task |
| --- | --- |
| Mind deep strong nudge | Task 1 |
| Teach rewrite / no expensive Cursor | Task 1 |
| ≥2 permission denied nudge | Task 2 |
| Budget spent → resume | Task 3 |
| Progress-only splice | Task 4 |
| §0A live gate before wire | Task 5 |
| Official digester EWMA score | Task 6 |
| Semantic layer + CI static gates | Task 7 (+8) |
| Hub read-only disclosure | Task 8 |
| No Hub Redis EWMA | Task 8 grep |
| No Python auto-MERGE | Global + all tasks |
| Supervisor second consumer | Deferred (registry note only in Task 7) |

## Placeholder scan

None intentional. Task 5 may mark a source `UNVERIFIED` only with Juniper go-ahead.

---
