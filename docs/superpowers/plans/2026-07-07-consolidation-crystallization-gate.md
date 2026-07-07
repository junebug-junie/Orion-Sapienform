# Consolidation Crystallization Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop automated consolidation from inserting graph drafts for every closed window; instead run a deterministic gate that **skips** low-signal windows or **proposes `MemoryCrystallizationV1`** for governor review.

**Architecture:** `consolidation_memory_gate()` reads existing turn signals (spark_meta appraisal, significance, shared `low_info_social`, read-only grammar event refs). `consolidate_window` branches skip vs propose. Graph draft path preserved behind `MEMORY_CONSOLIDATION_OUTPUT=graph_draft` legacy flag.

**Tech Stack:** Python 3.12, asyncpg, Redis bus, Pydantic v2, pytest, bash smoke.

**Spec:** `docs/superpowers/specs/2026-07-07-consolidation-crystallization-gate-design.md`

**Prerequisite:** `orion/memory/low_info_social.py` from PCR plan Task 1 (merge PCR branch first or cherry-pick that commit).

**Read-side partner:** `docs/superpowers/plans/2026-07-07-purpose-conditioned-recall.md` — PCR milestones A+B can ship before this plan.

**Branch:** `feat/consolidation-crystallization-gate`

**Worktree setup:**

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
git worktree add ../Orion-Sapienform-consolidation-gate -b feat/consolidation-crystallization-gate
cd ../Orion-Sapienform-consolidation-gate
# If low_info_social not on main yet:
# git cherry-pick <pcr-low-info-commit>
```

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/memory/low_info_social.py` | Use (PCR) | Shared courtesy detector for window turns |
| `orion/memory/consolidation_gate.py` | Create | Deterministic skip vs propose |
| `orion/memory/crystallization/intake_consolidation_window.py` | Create | Window → proposed crystallization |
| `orion/schemas/memory_consolidation.py` | Modify | Add `skipped` to `consolidation_status` literal |
| `services/orion-memory-consolidation/app/window_state.py` | Modify | `mark_consolidated_skipped`, `mark_crystallization_proposed` |
| `services/orion-memory-consolidation/app/worker.py` | Modify | Gate before suggest/draft insert |
| `services/orion-memory-consolidation/app/settings.py` | Modify | Gate env keys |
| `services/orion-memory-consolidation/.env_example` | Modify | Operator contract |
| `services/orion-memory-consolidation/tests/test_consolidation_gate.py` | Create | Gate unit tests |
| `services/orion-memory-consolidation/tests/test_intake_consolidation_window.py` | Create | Intake builder tests |
| `services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py` | Create | Worker branch tests |
| `scripts/smoke_memory_consolidation_gate.sh` | Create | Greeting skip + substantive propose smoke |

---

## Task 1: Schema — `consolidation_status=skipped`

**Files:**
- Modify: `orion/schemas/memory_consolidation.py`
- Create: `tests/test_memory_consolidation_schema_skipped.py`

- [ ] **Step 1: Write failing test**

```python
from orion.schemas.memory_consolidation import MemoryConsolidationWindowV1
from datetime import datetime, timezone


def test_consolidation_status_accepts_skipped():
    w = MemoryConsolidationWindowV1(
        memory_window_id="win-1",
        status="consolidated",
        consolidation_status="skipped",
        created_at=datetime.now(timezone.utc),
    )
    assert w.consolidation_status == "skipped"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_memory_consolidation_schema_skipped.py -v
```

Expected: validation error — `"skipped"` not in literal

- [ ] **Step 3: Extend literal**

In `orion/schemas/memory_consolidation.py`, change:

```python
consolidation_status: Optional[Literal["pending", "ok", "failed", "skipped"]] = None
```

Optionally add optional trace fields on window model if DB migration adds columns later (v1 can store skip reasons in spark_meta patch only).

- [ ] **Step 4: Run test + commit**

```bash
pytest tests/test_memory_consolidation_schema_skipped.py -v
git add orion/schemas/memory_consolidation.py tests/test_memory_consolidation_schema_skipped.py
git commit -m "feat(consolidation): add skipped consolidation_status"
```

---

## Task 2: `consolidation_memory_gate`

**Files:**
- Create: `orion/memory/consolidation_gate.py`
- Create: `services/orion-memory-consolidation/tests/test_consolidation_gate.py`

- [ ] **Step 1: Write failing gate tests**

```python
from orion.memory.consolidation_gate import consolidation_memory_gate


def _turn(prompt, response, novelty=0.1, shift="NONE", significance=0.2):
    return {
        "prompt": prompt,
        "response": response,
        "spark_meta": {
            "turn_change_appraisal": {
                "novelty_score": novelty,
                "shift_kind": shift,
                "turn_change_status": "ok",
            },
            "memory_significance_score": significance,
        },
    }


def test_gate_skips_greeting_window():
    turns = [
        _turn("hey", "hi there"),
        _turn("thanks", "anytime"),
    ]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "skip"
    assert "low_info_social" in result.reasons


def test_gate_proposes_topic_shift():
    turns = [_turn("move logistics alone", "that sounds heavy", novelty=0.72, shift="TOPIC", significance=0.55)]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "propose"
    assert result.dominant_shift == "TOPIC"


def test_gate_proposes_on_repair_signal():
    turns = [_turn("hey", "sorry about earlier")]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=True,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "propose"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest services/orion-memory-consolidation/tests/test_consolidation_gate.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement gate**

Create `orion/memory/consolidation_gate.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from orion.memory.low_info_social import is_low_info_social

_SHIFT_FORCE = frozenset({"TOPIC", "STANCE", "REPAIR"})


@dataclass
class ConsolidationGateResult:
    action: Literal["skip", "propose"]
    reasons: list[str] = field(default_factory=list)
    dominant_shift: str | None = None
    grammar_event_ids: list[str] = field(default_factory=list)
    window_novelty_max: float = 0.0
    window_significance_max: float = 0.0


def _appraisal(turn: dict[str, Any]) -> dict[str, Any]:
    spark = turn.get("spark_meta") if isinstance(turn.get("spark_meta"), dict) else {}
    appraisal = spark.get("turn_change_appraisal")
    return appraisal if isinstance(appraisal, dict) else {}


def _significance(turn: dict[str, Any]) -> float:
    spark = turn.get("spark_meta") if isinstance(turn.get("spark_meta"), dict) else {}
    val = spark.get("memory_significance_score", turn.get("memory_significance_score"))
    return float(val) if isinstance(val, (int, float)) else 0.0


def consolidation_memory_gate(
    *,
    turns: list[dict[str, Any]],
    grammar_repair_signal: bool,
    grammar_event_ids: list[str] | None = None,
    min_novelty: float = 0.35,
    min_significance: float = 0.40,
) -> ConsolidationGateResult:
    reasons: list[str] = []
    novelty_max = 0.0
    significance_max = 0.0
    dominant_shift: str | None = None
    best_novelty = -1.0

    for turn in turns:
        appraisal = _appraisal(turn)
        novelty = appraisal.get("novelty_score")
        n = float(novelty) if isinstance(novelty, (int, float)) else 0.0
        novelty_max = max(novelty_max, n)
        significance_max = max(significance_max, _significance(turn))
        shift = str(appraisal.get("shift_kind") or "NONE").upper()
        if n > best_novelty and shift in _SHIFT_FORCE:
            best_novelty = n
            dominant_shift = shift

    if grammar_repair_signal:
        return ConsolidationGateResult(
            action="propose",
            reasons=["repair_signal"],
            dominant_shift=dominant_shift or "REPAIR",
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    substantive_shift = any(
        str(_appraisal(t).get("shift_kind") or "NONE").upper() in _SHIFT_FORCE
        and float(_appraisal(t).get("novelty_score") or 0) >= min_novelty
        for t in turns
    )
    if substantive_shift:
        return ConsolidationGateResult(
            action="propose",
            reasons=["substantive_shift"],
            dominant_shift=dominant_shift,
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    if novelty_max >= min_novelty:
        return ConsolidationGateResult(
            action="propose",
            reasons=["novelty_above_floor"],
            dominant_shift=dominant_shift,
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    if significance_max >= min_significance:
        return ConsolidationGateResult(
            action="propose",
            reasons=["significance_above_floor"],
            dominant_shift=dominant_shift,
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    all_low_info = all(
        is_low_info_social(str(t.get("prompt") or ""))
        and is_low_info_social(str(t.get("response") or ""))
        for t in turns
    ) if turns else True
    if all_low_info:
        reasons.append("low_info_social")

    if novelty_max < min_novelty:
        reasons.append("novelty_below_floor")
    if significance_max < min_significance:
        reasons.append("significance_below_floor")

    return ConsolidationGateResult(
        action="skip",
        reasons=reasons or ["no_substantive_signal"],
        dominant_shift=None,
        grammar_event_ids=list(grammar_event_ids or []),
        window_novelty_max=novelty_max,
        window_significance_max=significance_max,
    )
```

- [ ] **Step 4: Run tests + commit**

```bash
pytest services/orion-memory-consolidation/tests/test_consolidation_gate.py -v
git add orion/memory/consolidation_gate.py services/orion-memory-consolidation/tests/test_consolidation_gate.py
git commit -m "feat(consolidation): deterministic consolidation_memory_gate"
```

---

## Task 3: `build_crystallization_from_window` intake

**Files:**
- Create: `orion/memory/crystallization/intake_consolidation_window.py`
- Create: `services/orion-memory-consolidation/tests/test_intake_consolidation_window.py`

- [ ] **Step 1: Write failing intake test**

```python
from orion.memory.consolidation_gate import ConsolidationGateResult
from orion.memory.crystallization.intake_consolidation_window import build_crystallization_from_window


def test_builds_proposed_semantic_crystallization():
    turns = [
        {
            "correlation_id": "corr-1",
            "prompt": "move logistics alone",
            "response": "that sounds heavy",
            "spark_meta": {},
        }
    ]
    gate = ConsolidationGateResult(action="propose", dominant_shift="TOPIC", grammar_event_ids=["evt-1"])
    crys = build_crystallization_from_window(
        memory_window_id="win-1",
        turns=turns,
        gate=gate,
    )
    assert crys.status == "proposed"
    assert crys.kind == "semantic"
    assert crys.governance.proposed_by == "memory_consolidation_intake"
    assert "corr-1" in [e.source_id for e in crys.evidence if e.source_kind == "chat_turn"]
    assert crys.source_grammar_event_ids == ["evt-1"]
```

- [ ] **Step 2: Implement intake (mirror autonomy episode pattern)**

Create `orion/memory/crystallization/intake_consolidation_window.py`:

```python
from __future__ import annotations

from typing import Any

from orion.memory.consolidation_gate import ConsolidationGateResult
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
    _utc_now,
    new_crystallization_id,
)

_PROPOSED_BY = "memory_consolidation_intake"
_POLICY = "consolidation_window_gate_v1"

_KIND_FOR_SHIFT = {
    "STANCE": "stance",
    "REPAIR": "open_loop",
    "TOPIC": "semantic",
}


def _window_summary(turns: list[dict[str, Any]]) -> str:
    for turn in reversed(turns):
        prompt = str(turn.get("prompt") or "").strip()
        if prompt:
            return prompt[:500]
    return "Consolidated chat window"


def _kind_for_gate(gate: ConsolidationGateResult, turns: list[dict[str, Any]]) -> str:
    if gate.dominant_shift:
        return _KIND_FOR_SHIFT.get(gate.dominant_shift, "episode")
    if len(turns) > 1:
        return "episode"
    return "episode"


def build_crystallization_from_window(
    *,
    memory_window_id: str,
    turns: list[dict[str, Any]],
    gate: ConsolidationGateResult,
    grammar_events: list[dict[str, Any]] | None = None,
) -> MemoryCrystallizationV1:
    now = _utc_now()
    summary = _window_summary(turns)
    evidence: list[CrystallizationEvidenceRefV1] = []
    for turn in turns:
        corr = str(turn.get("correlation_id") or "").strip()
        if not corr:
            continue
        excerpt = f"{turn.get('prompt', '')}\n{turn.get('response', '')}"[:2000] or None
        evidence.append(
            CrystallizationEvidenceRefV1(
                source_kind="chat_turn",
                source_id=corr,
                excerpt=excerpt,
                strength=0.75,
            )
        )
    grammar_ids = list(gate.grammar_event_ids)
    for event_id in grammar_ids:
        evidence.append(
            CrystallizationEvidenceRefV1(
                source_kind="grammar_event",
                source_id=event_id,
                strength=0.6,
            )
        )
    return MemoryCrystallizationV1(
        crystallization_id=new_crystallization_id(),
        kind=_kind_for_gate(gate, turns),
        subject=summary,
        summary=summary,
        status="proposed",
        scope=[f"memory_window:{memory_window_id}"],
        tags=["consolidation_window"],
        evidence=evidence,
        source_grammar_event_ids=grammar_ids,
        governance=CrystallizationGovernanceV1(
            proposed_by=_PROPOSED_BY,
            requires_manual_review=True,
            sensitivity="private",
            created_from_policy=_POLICY,
        ),
        provenance={
            "memory_window_id": memory_window_id,
            "dominant_shift": gate.dominant_shift,
            "window_novelty_max": gate.window_novelty_max,
            "window_significance_max": gate.window_significance_max,
            "gate_reasons": gate.reasons,
        },
        created_at=now,
        updated_at=now,
    )
```

- [ ] **Step 3: Run tests + commit**

```bash
pytest services/orion-memory-consolidation/tests/test_intake_consolidation_window.py -v
git add orion/memory/crystallization/intake_consolidation_window.py services/orion-memory-consolidation/tests/test_intake_consolidation_window.py
git commit -m "feat(consolidation): window intake builds proposed crystallization"
```

---

## Task 4: WindowStore skip + crystallization markers

**Files:**
- Modify: `services/orion-memory-consolidation/app/window_state.py`

- [ ] **Step 1: Write failing window_state test**

Add to `services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py` (create file):

```python
import pytest
from unittest.mock import AsyncMock

from app.window_state import WindowStore


@pytest.mark.asyncio
async def test_mark_consolidated_skipped():
    pool = AsyncMock()
    pool.execute = AsyncMock()
    store = WindowStore(pool)
    await store.mark_consolidated_skipped("win-1", reasons=["low_info_social"])
    sql = pool.execute.await_args.args[0]
    assert "skipped" in sql
```

- [ ] **Step 2: Add methods to WindowStore**

```python
    async def mark_consolidated_skipped(self, memory_window_id: str, *, reasons: list[str]) -> None:
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET status = 'consolidated', consolidation_status = 'skipped'
            WHERE memory_window_id = $1
            """,
            memory_window_id,
        )

    async def mark_crystallization_proposed(
        self,
        memory_window_id: str,
        *,
        crystallization_id: str,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE memory_consolidation_windows
            SET status = 'consolidated', consolidation_status = 'ok', draft_id = $2
            WHERE memory_window_id = $1
            """,
            memory_window_id,
            crystallization_id,
        )
```

Note: reuses `draft_id` column to store crystallization_id until a dedicated migration adds `crystallization_id` — document in README.

- [ ] **Step 3: Run test + commit**

```bash
pytest services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py::test_mark_consolidated_skipped -v
git add services/orion-memory-consolidation/app/window_state.py services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py
git commit -m "feat(consolidation): window store skip and crystallization markers"
```

---

## Task 5: Grammar evidence fetch helper

**Files:**
- Create: `orion/memory/consolidation_grammar.py`
- Modify: `services/orion-memory-consolidation/tests/test_consolidation_gate.py` (integration-style)

- [ ] **Step 1: Write failing grammar fetch test**

```python
import pytest
from unittest.mock import AsyncMock, patch

from orion.memory.consolidation_grammar import fetch_grammar_evidence_for_window


@pytest.mark.asyncio
async def test_fetch_collects_repair_signal_event_ids():
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=[{"event_id": "evt-1", "atoms": [{"semantic_role": "repair_signal"}]}])
    turns = [{"correlation_id": "corr-1"}]
    repair, event_ids = await fetch_grammar_evidence_for_window(
        pool,
        turns=turns,
        node_id="athena",
        enabled=True,
    )
    assert repair is True
    assert event_ids == ["evt-1"]
```

- [ ] **Step 2: Implement read-only grammar query**

Create `orion/memory/consolidation_grammar.py`:

```python
from __future__ import annotations

from typing import Any


async def fetch_grammar_evidence_for_window(
    pool,
    *,
    turns: list[dict[str, Any]],
    node_id: str,
    enabled: bool,
) -> tuple[bool, list[str]]:
    if not enabled or pool is None:
        return False, []
    event_ids: list[str] = []
    repair = False
    for turn in turns:
        corr = str(turn.get("correlation_id") or "").strip()
        if not corr:
            continue
        trace_id = f"hub.chat:{node_id}:{corr}"
        rows = await pool.fetch(
            """
            SELECT event_id, payload
            FROM grammar_events
            WHERE trace_id = $1
            ORDER BY created_at ASC
            """,
            trace_id,
        )
        for row in rows:
            eid = str(row["event_id"])
            event_ids.append(eid)
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            atoms = payload.get("atoms") or []
            if isinstance(atoms, list):
                for atom in atoms:
                    if isinstance(atom, dict) and atom.get("semantic_role") == "repair_signal":
                        repair = True
    return repair, event_ids
```

Adjust table/column names to match actual `grammar_events` schema in repo before implementing — inspect with:

```bash
rg "grammar_events" orion/ services/ --glob '*.sql' -l | head
```

- [ ] **Step 3: Run test + commit**

```bash
pytest services/orion-memory-consolidation/tests/test_consolidation_gate.py -v -k grammar
git add orion/memory/consolidation_grammar.py services/orion-memory-consolidation/tests/test_consolidation_gate.py
git commit -m "feat(consolidation): read-only grammar evidence for gate"
```

---

## Task 6: Settings + env

**Files:**
- Modify: `services/orion-memory-consolidation/app/settings.py`
- Modify: `services/orion-memory-consolidation/.env_example`

- [ ] **Step 1: Add settings fields**

```python
    memory_consolidation_output: str = Field("crystallization_propose", alias="MEMORY_CONSOLIDATION_OUTPUT")
    memory_consolidation_min_novelty: float = Field(0.35, alias="MEMORY_CONSOLIDATION_MIN_NOVELTY")
    memory_consolidation_min_significance: float = Field(0.40, alias="MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE")
    memory_consolidation_fetch_grammar_evidence: bool = Field(True, alias="MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE")
    memory_consolidation_grammar_dsn: str = Field("", alias="MEMORY_CONSOLIDATION_GRAMMAR_DSN")
```

- [ ] **Step 2: Update `.env_example` with comments**

Document `crystallization_propose` | `graph_draft` | `skip_only`.

- [ ] **Step 3: Sync local env**

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-memory-consolidation/app/settings.py services/orion-memory-consolidation/.env_example
git commit -m "feat(consolidation): gate env settings"
```

---

## Task 7: Wire `consolidate_window` — gate branch

**Files:**
- Modify: `services/orion-memory-consolidation/app/worker.py`
- Extend: `services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py`

- [ ] **Step 1: Write failing worker integration test**

```python
@pytest.mark.asyncio
async def test_consolidate_window_skips_greeting_without_draft(monkeypatch):
    # mock gate skip -> assert mark_consolidated_skipped called, insert_pending_draft NOT called
    ...


@pytest.mark.asyncio
async def test_consolidate_window_proposes_crystallization(monkeypatch):
    # mock gate propose -> insert_crystallization called, no suggest_with_escalation
    ...
```

- [ ] **Step 2: Refactor `consolidate_window`**

At start of `ConsolidationSuggestRunner.consolidate_window`:

```python
        output_mode = settings.memory_consolidation_output
        if output_mode == "crystallization_propose":
            from orion.memory.consolidation_gate import consolidation_memory_gate
            from orion.memory.consolidation_grammar import fetch_grammar_evidence_for_window
            from orion.memory.crystallization.intake_consolidation_window import build_crystallization_from_window
            from orion.memory.crystallization.repository import insert_crystallization

            repair, grammar_event_ids = await fetch_grammar_evidence_for_window(
                self._pool,
                turns=turns,
                node_id=settings.NODE_NAME,
                enabled=settings.memory_consolidation_fetch_grammar_evidence,
            )
            gate = consolidation_memory_gate(
                turns=turns,
                grammar_repair_signal=repair,
                grammar_event_ids=grammar_event_ids,
                min_novelty=settings.memory_consolidation_min_novelty,
                min_significance=settings.memory_consolidation_min_significance,
            )
            if gate.action == "skip":
                await self._window_store.mark_consolidated_skipped(window_id, reasons=gate.reasons)
                for corr in corr_ids:
                    await publish_spark_meta_patch(bus, corr, {"consolidation_gate": {"action": "skip", "reasons": gate.reasons}})
                return

            crystallization = build_crystallization_from_window(
                memory_window_id=window_id,
                turns=turns,
                gate=gate,
            )
            cid = await insert_crystallization(self._pool, crystallization)
            await self._window_store.mark_crystallization_proposed(window_id, crystallization_id=cid)
            # emit orion:memory:crystallization:proposed bus event — match Hub pattern
            for corr in corr_ids:
                await publish_spark_meta_patch(bus, corr, {"consolidation_gate": {"action": "propose", "crystallization_id": cid}})
            return

        # legacy graph_draft path unchanged below
```

- [ ] **Step 3: Run existing worker tests + new tests**

```bash
pytest services/orion-memory-consolidation/tests/test_worker_classify_patch.py -q
pytest services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py -q
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-memory-consolidation/app/worker.py services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py
git commit -m "feat(consolidation): gate wired into consolidate_window"
```

---

## Task 8: Legacy `graph_draft` escape hatch test

**Files:**
- Extend: `services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py`

- [ ] **Step 1: Test legacy mode**

```python
@pytest.mark.asyncio
async def test_graph_draft_legacy_mode_still_inserts_draft(monkeypatch):
    monkeypatch.setattr("app.settings.settings.memory_consolidation_output", "graph_draft")
    # assert suggest_with_escalation + insert_pending_draft still called
```

- [ ] **Step 2: Run test**

```bash
pytest services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py -v -k legacy
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test(consolidation): graph_draft legacy output mode"
```

---

## Task 9: README + deprecation note

**Files:**
- Modify: `services/orion-memory-consolidation/README.md`

- [ ] **Step 1: Document new default output**

Explain:
- Default `MEMORY_CONSOLIDATION_OUTPUT=crystallization_propose`
- Graph drafts are manual bridge path only
- `draft_id` column temporarily stores crystallization_id for proposed windows

- [ ] **Step 2: Commit**

```bash
git add services/orion-memory-consolidation/README.md
git commit -m "docs(consolidation): document crystallization gate default"
```

---

## Task 10: Smoke script + agent-check

**Files:**
- Create: `scripts/smoke_memory_consolidation_gate.sh`

- [ ] **Step 1: Create smoke**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== consolidation gate: greeting window skip =="
# Close a synthetic greeting window via test DB or bus fixture
# Assert consolidation_status=skipped

echo "== consolidation gate: topic window propose =="
# Assert memory_crystallizations row status=proposed

echo "consolidation gate smoke OK"
```

- [ ] **Step 2: Run agent-check**

```bash
make agent-check SERVICE=orion-memory-consolidation
python scripts/check_env_template_parity.py
```

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/smoke_memory_consolidation_gate.sh
git add scripts/smoke_memory_consolidation_gate.sh
git commit -m "chore(consolidation): add gate smoke script"
```

---

## Spec coverage self-check

| Spec requirement | Task |
|------------------|------|
| Deterministic consolidation gate | 2 |
| Skip greeting windows | 2, 7 |
| Propose crystallization on substantive signal | 3, 7 |
| Grammar read-only evidence refs | 5, 3 |
| No grammar schema changes | (no task — constraint) |
| `consolidation_status=skipped` | 1, 4 |
| Legacy `graph_draft` mode | 6, 8 |
| Env parity | 6 |
| Manual graph path unchanged | 7 (legacy branch preserved) |
| Traceability on spark_meta | 7 |
| README deprecation | 9 |

**Open spec questions (defaults chosen in plan):**
- Failed validation stays in inbox (no auto-quarantine task)
- `graph_draft` kept as env escape, default `crystallization_propose`
- Grammar DSN defaults to consolidation pool; override via `MEMORY_CONSOLIDATION_GRAMMAR_DSN`

---

**Plan complete.** Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks

**2. Inline Execution** — execute in this session via executing-plans with checkpoints

**Recommended program order:** PCR Milestones A→B first, then this plan Tasks 1–10 (shared `low_info_social` from PCR Task 1).

Which approach?
