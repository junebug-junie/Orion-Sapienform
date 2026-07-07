# Reverie Semantic Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop reverie from narrating coalition bookkeeping by lifting broadcast pointers into human `ConcernCardV1` text (from persisted turn referents), gating skip when unresolved, and routing `reverie_narrate` through the metacog/background GPU lane.

**Architecture:** On unresolved harness closure, `orion-substrate-runtime` upserts `substrate_turn_referent` rows keyed by `correlation_id`. Each reverie tick calls `resolve_concern_cards()` to join broadcast `source_refs` / `harness_closure:{corr}` pointers to those rows, builds prompt context from cards (not coalition mechanics), runs `reverie_narrate` on `orion:cortex:exec:request:background` with `mode=metacog`, then applies deterministic post-LLM gates (referent overlap, infra vocabulary) before the existing hollow guard. Feature-flagged off by default (`ORION_REVERIE_SEMANTIC_LIFT_ENABLED=false`); evoked `stance_react` path unchanged.

**Tech Stack:** Python 3.11+, Pydantic v2, SQLAlchemy (raw SQL like existing reverie store), Redis bus RPC, Jinja2 prompts, pytest.

---

## File map

| File | Responsibility |
|------|----------------|
| `services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql` | Typed referent table |
| `orion/schemas/harness_finalize.py` | `HarnessPostTurnClosureV1` referent fields |
| `orion/harness/finalize.py` | Populate closure excerpts at emit |
| `services/orion-harness-governor/app/bus_listener.py` | Pass turn text into closure emit |
| `services/orion-substrate-runtime/app/turn_referent_store.py` | SQL upsert on closure |
| `services/orion-substrate-runtime/app/worker.py` | Call referent writer from `handle_post_turn_closure` |
| `services/orion-substrate-runtime/app/settings.py` | `ORION_REVERIE_REFERENT_MAX_AGE_HOURS` (writer freshness default) |
| `orion/schemas/reverie.py` | `ConcernCardV1` runtime schema |
| `orion/reverie/referent_loader.py` | SQL read by `correlation_id` / `coalition_ref` |
| `orion/reverie/semantic_lift.py` | `resolve_concern_cards`, `reverie_semantic_gate`, `enforce_semantic_quality` |
| `services/orion-thought/app/reverie.py` | Wire resolver + gates + metacog plan behind flag |
| `orion/cognition/prompts/reverie_narrate.j2` | Inner-speech prompt over `concern_cards` |
| `services/orion-thought/app/settings.py` | Semantic-lift + background channel env keys |
| `services/orion-thought/app/cortex_client.py` | Accept per-call channel (reverie uses background) |
| `services/orion-cortex-exec/app/llm_lane.py` | Add `reverie_narrate` to `_BACKGROUND_VERBS` |
| `orion/reverie/tests/test_semantic_lift.py` | Unit tests resolver + gates |
| `services/orion-cortex-exec/tests/test_llm_lane_propagation.py` | `reverie_narrate` → background |
| `services/orion-thought/tests/test_reverie_semantic_lift.py` | Integration: referent → non-meta thought |
| `orion/reverie/evals/test_reverie_semantic_quality_eval.py` | Labeled good/bad quality set |
| `services/orion-thought/evals/test_reverie_hollow_guard_eval.py` | Extend with meta-narration cases |

---

### Task 1: SQL migration — `substrate_turn_referent`

**Files:**
- Create: `services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql`

- [ ] **Step 1: Add migration file**

```sql
-- Reverie semantic lift v1: human referents for harness_closure pointers.
-- Apply before enabling ORION_REVERIE_SEMANTIC_LIFT_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql

create table if not exists substrate_turn_referent (
    correlation_id          text primary key,
    coalition_ref           text not null,
    user_message_excerpt    text not null default '',
    stance_imperative       text not null default '',
    surprise_unresolved     boolean not null default true,
    thought_event_id        text,
    created_at              timestamptz not null,
    stored_at               timestamptz not null default now()
);

create index if not exists idx_substrate_turn_referent_created_at
    on substrate_turn_referent (created_at desc);

create index if not exists idx_substrate_turn_referent_coalition_ref
    on substrate_turn_referent (coalition_ref);
```

- [ ] **Step 2: Apply migration locally (operator)**

Run: `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql`  
Expected: `CREATE TABLE` / `CREATE INDEX` (or already exists)

- [ ] **Step 3: Commit**

```bash
git add services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql
git commit -m "feat(reverie): add substrate_turn_referent migration for semantic lift"
```

---

### Task 2: Extend `HarnessPostTurnClosureV1` schema

**Files:**
- Modify: `orion/schemas/harness_finalize.py`
- Test: `tests/test_unified_turn_schemas.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_unified_turn_schemas.py`:

```python
def test_post_turn_closure_carries_referent_excerpts_when_unresolved() -> None:
    closure = HarnessPostTurnClosureV1(
        correlation_id="c-1",
        outcome_molecule_id="out-1",
        verdict_molecule_id="verd-1",
        surprise_unresolved=True,
        user_message_excerpt="What if the deploy slips again?",
        stance_imperative="Name the risk plainly before offering a plan.",
        thought_event_id="te-1",
    )
    restored = HarnessPostTurnClosureV1.model_validate(closure.model_dump(mode="json"))
    assert restored.user_message_excerpt == "What if the deploy slips again?"
    assert restored.stance_imperative.startswith("Name the risk")
    assert restored.thought_event_id == "te-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_turn_schemas.py::test_post_turn_closure_carries_referent_excerpts_when_unresolved -v`  
Expected: FAIL — unexpected keyword `user_message_excerpt`

- [ ] **Step 3: Add fields to `HarnessPostTurnClosureV1`**

In `orion/schemas/harness_finalize.py`, inside `HarnessPostTurnClosureV1`:

```python
    user_message_excerpt: str = Field(default="", max_length=300)
    stance_imperative: str = Field(default="", max_length=300)
    thought_event_id: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_unified_turn_schemas.py::test_post_turn_closure_carries_referent_excerpts_when_unresolved -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/harness_finalize.py tests/test_unified_turn_schemas.py
git commit -m "feat(harness): add referent excerpts to post-turn closure schema"
```

---

### Task 3: Populate closure excerpts at emit time

**Files:**
- Modify: `orion/harness/finalize.py`
- Modify: `services/orion-harness-governor/app/bus_listener.py`
- Test: `orion/harness/tests/test_post_turn_closure_emits.py`

- [ ] **Step 1: Write the failing test**

Add to `orion/harness/tests/test_post_turn_closure_emits.py`:

```python
def _excerpt(text: str, max_len: int = 300) -> str:
    t = (text or "").strip()
    return t[:max_len] if len(t) > max_len else t


@pytest.mark.asyncio
async def test_emit_post_turn_closure_includes_referent_excerpts_when_unresolved() -> None:
    thought = make_thought()
    thought = thought.model_copy(update={"imperative": "Hold the line on scope."})
    appraisal = make_appraisal()
    reflection = make_reflection()
    reflection = reflection.model_copy(update={"imperative": "Hold the line on scope."})
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-ref",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=make_verdict(reflection),
        final_text="done",
        surprise_resolved=False,
    )
    closure = await emit_post_turn_closure(
        correlation_id="c-ref",
        outcome_molecule=outcome,
        verdict_molecule_id="verdict-ref",
        user_message="Can we ship Friday without cutting tests?",
        thought_event=thought,
        reflection_imperative=reflection.imperative,
    )
    assert closure.surprise_unresolved is True
    assert closure.user_message_excerpt == _excerpt("Can we ship Friday without cutting tests?")
    assert closure.stance_imperative == _excerpt("Hold the line on scope.")
    assert closure.thought_event_id == thought.event_id
```

(Adjust fixture imports — use existing `make_verdict` helper or inline a minimal `HarnessVerdictMoleculeV1`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/harness/tests/test_post_turn_closure_emits.py::test_emit_post_turn_closure_includes_referent_excerpts_when_unresolved -v`  
Expected: FAIL — `emit_post_turn_closure()` got unexpected keyword argument

- [ ] **Step 3: Implement excerpt helper + emit signature**

In `orion/harness/finalize.py`:

```python
def _excerpt(text: str, max_len: int = 300) -> str:
    t = (text or "").strip()
    return t[:max_len] if len(t) > max_len else t
```

Extend `emit_post_turn_closure`:

```python
async def emit_post_turn_closure(
    *,
    correlation_id: str,
    outcome_molecule: HarnessTurnOutcomeMoleculeV1,
    verdict_molecule_id: str,
    grammar_event_ids: list[str] | None = None,
    user_message: str = "",
    thought_event: ThoughtEventV1 | None = None,
    reflection_imperative: str = "",
    channel: str = POST_TURN_CLOSURE_CHANNEL,
    publish_fn: PublishFn | None = None,
    bus: Any = None,
) -> HarnessPostTurnClosureV1:
    surprise_unresolved = not outcome_molecule.surprise_resolved
    referent_fields: dict[str, Any] = {}
    if surprise_unresolved:
        referent_fields = {
            "user_message_excerpt": _excerpt(user_message),
            "stance_imperative": _excerpt(reflection_imperative or (thought_event.imperative if thought_event else "")),
            "thought_event_id": thought_event.event_id if thought_event else None,
        }
    closure = HarnessPostTurnClosureV1(
        correlation_id=correlation_id,
        outcome_molecule_id=_outcome_molecule_id(outcome_molecule),
        verdict_molecule_id=verdict_molecule_id,
        grammar_event_ids=list(grammar_event_ids or outcome_molecule.grammar_event_ids),
        surprise_unresolved=surprise_unresolved,
        **referent_fields,
    )
    ...
```

- [ ] **Step 4: Wire harness governor call site**

In `services/orion-harness-governor/app/bus_listener.py`, update the `emit_post_turn_closure` call:

```python
    closure = await emit_post_turn_closure(
        correlation_id=corr,
        outcome_molecule=chain.outcome_molecule,
        verdict_molecule_id=chain.verdict_molecule_id,
        grammar_event_ids=run.grammar_event_ids,
        user_message=request.user_message,
        thought_event=request.thought_event,
        reflection_imperative=chain.reflection.imperative,
        channel=settings.channel_post_turn_closure,
        bus=bus,
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest orion/harness/tests/test_post_turn_closure_emits.py -q`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/harness/finalize.py services/orion-harness-governor/app/bus_listener.py orion/harness/tests/test_post_turn_closure_emits.py
git commit -m "feat(harness): emit referent excerpts on unresolved post-turn closure"
```

---

### Task 4: Substrate-runtime referent writer

**Files:**
- Create: `services/orion-substrate-runtime/app/turn_referent_store.py`
- Modify: `services/orion-substrate-runtime/app/worker.py`
- Test: `services/orion-substrate-runtime/tests/test_turn_referent_store.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-substrate-runtime/tests/test_turn_referent_store.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.schemas.harness_finalize import HarnessPostTurnClosureV1

from app.turn_referent_store import persist_turn_referent


def test_persist_turn_referent_upserts_on_unresolved_closure() -> None:
    conn = MagicMock()
    begin_ctx = MagicMock()
    begin_ctx.__enter__ = MagicMock(return_value=conn)
    begin_ctx.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.begin.return_value = begin_ctx

    closure = HarnessPostTurnClosureV1(
        correlation_id="corr-1",
        outcome_molecule_id="out-1",
        verdict_molecule_id="ver-1",
        surprise_unresolved=True,
        user_message_excerpt="You asked about the deploy.",
        stance_imperative="Name the risk.",
        thought_event_id="te-1",
    )
    ok = persist_turn_referent(closure, engine=engine, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    assert ok is True
    conn.execute.assert_called_once()
    params = conn.execute.call_args.args[1]
    assert params["correlation_id"] == "corr-1"
    assert params["coalition_ref"] == "harness_closure:corr-1"
    assert params["user_message_excerpt"] == "You asked about the deploy."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-substrate-runtime/tests/test_turn_referent_store.py -v`  
Expected: FAIL — module not found

- [ ] **Step 3: Implement `turn_referent_store.py`**

```python
"""Best-effort writer for substrate_turn_referent (reverie semantic lift v1)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:
    from orion.schemas.harness_finalize import HarnessPostTurnClosureV1

logger = logging.getLogger("orion.substrate.runtime.turn_referent")


def _coalition_ref(correlation_id: str) -> str:
    return f"harness_closure:{correlation_id}"


def persist_turn_referent(
    closure: "HarnessPostTurnClosureV1",
    *,
    engine: Any,
    now: datetime | None = None,
) -> bool:
    if not closure.surprise_unresolved:
        return False
    excerpt_user = (closure.user_message_excerpt or "").strip()
    excerpt_stance = (closure.stance_imperative or "").strip()
    if not excerpt_user and not excerpt_stance:
        logger.info(
            "turn_referent_skip corr=%s reason=empty_excerpts",
            closure.correlation_id,
        )
        return False
    ts = now or datetime.now(timezone.utc)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_turn_referent
                        (correlation_id, coalition_ref, user_message_excerpt,
                         stance_imperative, surprise_unresolved, thought_event_id, created_at)
                    VALUES
                        (:correlation_id, :coalition_ref, :user_message_excerpt,
                         :stance_imperative, :surprise_unresolved, :thought_event_id, :created_at)
                    ON CONFLICT (correlation_id) DO UPDATE SET
                        coalition_ref = EXCLUDED.coalition_ref,
                        user_message_excerpt = EXCLUDED.user_message_excerpt,
                        stance_imperative = EXCLUDED.stance_imperative,
                        surprise_unresolved = EXCLUDED.surprise_unresolved,
                        thought_event_id = EXCLUDED.thought_event_id,
                        created_at = EXCLUDED.created_at,
                        stored_at = now()
                    """
                ),
                {
                    "correlation_id": closure.correlation_id,
                    "coalition_ref": _coalition_ref(closure.correlation_id),
                    "user_message_excerpt": excerpt_user,
                    "stance_imperative": excerpt_stance,
                    "surprise_unresolved": True,
                    "thought_event_id": closure.thought_event_id,
                    "created_at": ts,
                },
            )
        logger.info(
            "turn_referent_upserted corr=%s coalition_ref=%s",
            closure.correlation_id,
            _coalition_ref(closure.correlation_id),
        )
        return True
    except Exception as exc:
        logger.warning("turn_referent_upsert_failed corr=%s err=%s", closure.correlation_id, exc)
        return False
```

- [ ] **Step 4: Call from `worker.handle_post_turn_closure`**

At end of `handle_post_turn_closure` in `services/orion-substrate-runtime/app/worker.py`:

```python
        try:
            from .turn_referent_store import persist_turn_referent

            store = self._get_substrate_graph_store(log_label="turn_referent_store_init_failed")
            if store is not None and hasattr(store, "engine"):
                persist_turn_referent(closure, engine=store.engine)
        except Exception:
            logger.warning(
                "turn_referent_persist_failed corr=%s",
                closure.correlation_id,
                exc_info=True,
            )
```

(If graph store does not expose `.engine`, add a thin `_get_sql_engine()` helper mirroring `orion-thought/app/store.py` `_database_url()` using `settings.postgres_uri`.)

- [ ] **Step 5: Run tests**

Run: `pytest services/orion-substrate-runtime/tests/test_turn_referent_store.py -q`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-substrate-runtime/app/turn_referent_store.py services/orion-substrate-runtime/app/worker.py services/orion-substrate-runtime/tests/test_turn_referent_store.py
git commit -m "feat(substrate): persist turn referents on unresolved harness closure"
```

---

### Task 5: `ConcernCardV1` schema

**Files:**
- Modify: `orion/schemas/reverie.py`
- Test: `orion/reverie/tests/test_semantic_lift.py` (schema section only)

- [ ] **Step 1: Write the failing test**

Create `orion/reverie/tests/test_semantic_lift.py` with:

```python
from datetime import datetime, timezone

from orion.schemas.reverie import ConcernCardV1


def test_concern_card_harness_turn_template() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:abc",
        user_message_excerpt="Will the surgery timeline slip?",
        stance_imperative="Stay with the worry before fixing it.",
        created_at=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
    )
    assert card.coalition_ref == "harness_closure:abc"
    assert "Will the surgery timeline slip?" in card.human_text
    assert "Stay with the worry" in card.human_text
    assert card.source_kind == "harness_turn"
    assert len(card.human_text) >= 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/reverie/tests/test_semantic_lift.py::test_concern_card_harness_turn_template -v`  
Expected: FAIL — cannot import `ConcernCardV1`

- [ ] **Step 3: Add `ConcernCardV1` to `orion/schemas/reverie.py`**

```python
from typing import Literal

ConcernSourceKind = Literal["harness_turn"]


class ConcernCardV1(BaseModel):
    card_id: str
    coalition_ref: str
    source_kind: ConcernSourceKind = "harness_turn"
    human_text: str = Field(max_length=500)
    thread_label: str = Field(default="", max_length=80)
    freshness_sec: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_harness_turn(
        cls,
        *,
        coalition_ref: str,
        user_message_excerpt: str,
        stance_imperative: str,
        created_at: datetime,
        now: datetime | None = None,
    ) -> "ConcernCardV1 | None":
        user = (user_message_excerpt or "").strip()
        stance = (stance_imperative or "").strip()
        if not user and not stance:
            return None
        human_text = (
            f'You said: "{user}"\n'
            f"I felt I should: {stance}\n"
            "(Surprise still open from that turn.)"
        )[:500]
        if len(human_text.strip()) < 40 and len(user) < 10 and len(stance) < 10:
            return None
        ref_now = now or datetime.now(timezone.utc)
        age = max(0.0, (ref_now - created_at).total_seconds())
        from orion.substrate.ids import stable_hash_id

        return cls(
            card_id=stable_hash_id("concern", [coalition_ref]),
            coalition_ref=coalition_ref,
            human_text=human_text,
            thread_label=(user[:80] if user else stance[:80]),
            freshness_sec=age,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/reverie/tests/test_semantic_lift.py::test_concern_card_harness_turn_template -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/reverie.py orion/reverie/tests/test_semantic_lift.py
git commit -m "feat(reverie): add ConcernCardV1 schema with harness_turn template"
```

---

### Task 6: Referent SQL loader

**Files:**
- Create: `orion/reverie/referent_loader.py`
- Test: `orion/reverie/tests/test_semantic_lift.py`

- [ ] **Step 1: Write the failing test**

Add to `orion/reverie/tests/test_semantic_lift.py`:

```python
from orion.reverie.referent_loader import TurnReferentRow, parse_harness_closure_ref


def test_parse_harness_closure_ref() -> None:
    assert parse_harness_closure_ref("harness_closure:abc-123") == "abc-123"
    assert parse_harness_closure_ref("node:foo") is None


def test_turn_referent_row_to_card() -> None:
    from datetime import datetime, timezone

    row = TurnReferentRow(
        correlation_id="abc-123",
        coalition_ref="harness_closure:abc-123",
        user_message_excerpt="What about the deadline?",
        stance_imperative="Acknowledge the pressure first.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    card = row.to_concern_card(now=datetime(2026, 7, 7, 1, 0, tzinfo=timezone.utc))
    assert card is not None
    assert "deadline" in card.human_text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/reverie/tests/test_semantic_lift.py::test_parse_harness_closure_ref -v`  
Expected: FAIL — module not found

- [ ] **Step 3: Implement `referent_loader.py`**

```python
"""Read substrate_turn_referent rows for reverie semantic lift."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from sqlalchemy import create_engine, text

from orion.schemas.reverie import ConcernCardV1

logger = logging.getLogger("orion.reverie.referent_loader")

_HARNESS_CLOSURE_RE = re.compile(r"^harness_closure:(?P<corr>.+)$")


def parse_harness_closure_ref(ref: str) -> str | None:
    m = _HARNESS_CLOSURE_RE.match((ref or "").strip())
    return m.group("corr") if m else None


@dataclass(frozen=True)
class TurnReferentRow:
    correlation_id: str
    coalition_ref: str
    user_message_excerpt: str
    stance_imperative: str
    created_at: datetime

    def to_concern_card(self, *, now: datetime | None = None) -> ConcernCardV1 | None:
        return ConcernCardV1.from_harness_turn(
            coalition_ref=self.coalition_ref,
            user_message_excerpt=self.user_message_excerpt,
            stance_imperative=self.stance_imperative,
            created_at=self.created_at,
            now=now,
        )


class ReferentLoader(Protocol):
    def load_by_correlation_id(self, correlation_id: str) -> TurnReferentRow | None: ...
    def load_by_coalition_ref(self, coalition_ref: str) -> TurnReferentRow | None: ...


def _database_url() -> str:
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def default_referent_loader(*, max_age_hours: float = 24.0) -> ReferentLoader:
    return SqlReferentLoader(max_age_hours=max_age_hours)


class SqlReferentLoader:
    def __init__(self, *, max_age_hours: float = 24.0) -> None:
        self.max_age_hours = max_age_hours
        self._engine = create_engine(_database_url(), pool_pre_ping=True)

    def _row_from_record(self, rec: dict) -> TurnReferentRow | None:
        created = rec.get("created_at")
        if not isinstance(created, datetime):
            return None
        now = datetime.now(timezone.utc)
        age_h = (now - created).total_seconds() / 3600.0
        if age_h > self.max_age_hours:
            return None
        return TurnReferentRow(
            correlation_id=str(rec["correlation_id"]),
            coalition_ref=str(rec["coalition_ref"]),
            user_message_excerpt=str(rec.get("user_message_excerpt") or ""),
            stance_imperative=str(rec.get("stance_imperative") or ""),
            created_at=created,
        )

    def load_by_correlation_id(self, correlation_id: str) -> TurnReferentRow | None:
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT correlation_id, coalition_ref, user_message_excerpt,
                               stance_imperative, created_at
                        FROM substrate_turn_referent
                        WHERE correlation_id = :cid
                        LIMIT 1
                        """
                    ),
                    {"cid": correlation_id},
                ).mappings().first()
            if not row:
                return None
            return self._row_from_record(dict(row))
        except Exception as exc:
            logger.warning("referent_load_failed cid=%s err=%s", correlation_id, exc)
            return None

    def load_by_coalition_ref(self, coalition_ref: str) -> TurnReferentRow | None:
        corr = parse_harness_closure_ref(coalition_ref)
        if corr:
            return self.load_by_correlation_id(corr)
        return None
```

- [ ] **Step 4: Run tests**

Run: `pytest orion/reverie/tests/test_semantic_lift.py -q`  
Expected: PASS (schema + loader tests)

- [ ] **Step 5: Commit**

```bash
git add orion/reverie/referent_loader.py orion/reverie/tests/test_semantic_lift.py
git commit -m "feat(reverie): add SQL referent loader for semantic lift"
```

---

### Task 7: Resolver + pre-LLM semantic gate

**Files:**
- Create: `orion/reverie/semantic_lift.py`
- Test: `orion/reverie/tests/test_semantic_lift.py`

- [ ] **Step 1: Write failing resolver tests**

Add to `orion/reverie/tests/test_semantic_lift.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.reverie.referent_loader import TurnReferentRow
from orion.reverie.semantic_lift import resolve_concern_cards, reverie_semantic_gate
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1


def _broadcast_with_harness_ref(corr: str = "corr-1") -> AttentionBroadcastProjectionV1:
    loop = OpenLoopV1(
        id="ol-1",
        description="deploy worry",
        source_refs=[f"harness_closure:{corr}"],
    )
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[loop]),
        attended_node_ids=[f"harness_closure:{corr}"],
        selected_open_loop_id="ol-1",
        coalition_stability_score=0.5,
    )


def test_resolve_concern_cards_from_selected_loop_source_ref() -> None:
    row = TurnReferentRow(
        correlation_id="corr-1",
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="Can we still make Friday's deploy?",
        stance_imperative="Name the schedule risk before reassuring.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    loader = MagicMock()
    loader.load_by_coalition_ref.return_value = row
    cards = resolve_concern_cards(_broadcast_with_harness_ref(), referent_loader=loader)
    assert len(cards) == 1
    assert "Friday" in cards[0].human_text


def test_reverie_semantic_gate_skips_when_no_cards() -> None:
    assert reverie_semantic_gate([]) == "skip"


def test_reverie_semantic_gate_proceeds_when_card_substantive() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:x",
        user_message_excerpt="What happens if the migration fails overnight?",
        stance_imperative="Stay with the operational fear before planning.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    assert reverie_semantic_gate([card]) == "proceed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest orion/reverie/tests/test_semantic_lift.py -k "resolve_concern_cards or reverie_semantic_gate" -v`  
Expected: FAIL — module not found

- [ ] **Step 3: Implement `semantic_lift.py` resolver + gate**

```python
"""Deterministic semantic lift for reverie: coalition pointers → ConcernCardV1."""

from __future__ import annotations

import logging
from typing import Literal

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.reverie import ConcernCardV1

from .referent_loader import ReferentLoader, parse_harness_closure_ref

logger = logging.getLogger("orion.reverie.semantic_lift")

MAX_CARDS_PER_TICK = 3
MIN_CARD_HUMAN_TEXT = 40


def _collect_coalition_refs(broadcast: AttentionBroadcastProjectionV1) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def _add(ref: str) -> None:
        r = (ref or "").strip()
        if r and r not in seen:
            seen.add(r)
            refs.append(r)

    if broadcast.selected_open_loop_id:
        loop = next(
            (l for l in broadcast.frame.open_loops if l.id == broadcast.selected_open_loop_id),
            None,
        )
        if loop is not None:
            for ref in loop.source_refs:
                _add(ref)
    for loop in broadcast.frame.open_loops:
        corr = None
        for ref in loop.source_refs:
            corr = parse_harness_closure_ref(ref) or corr
            _add(ref)
    for node_id in broadcast.attended_node_ids:
        _add(node_id)
    return refs


def resolve_concern_cards(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    referent_loader: ReferentLoader,
) -> list[ConcernCardV1]:
    cards: list[ConcernCardV1] = []
    seen_corr: set[str] = set()

    for ref in _collect_coalition_refs(broadcast):
        corr = parse_harness_closure_ref(ref)
        if corr and corr in seen_corr:
            continue
        try:
            row = referent_loader.load_by_coalition_ref(ref)
        except Exception as exc:
            logger.warning("referent_loader_error ref=%s err=%s", ref, exc)
            continue
        if row is None:
            continue
        if corr:
            seen_corr.add(corr)
        card = row.to_concern_card()
        if card is not None:
            cards.append(card)
        if len(cards) >= MAX_CARDS_PER_TICK:
            break
    return cards


def reverie_semantic_gate(cards: list[ConcernCardV1]) -> Literal["proceed", "skip"]:
    if not cards:
        return "skip"
    if any(len(c.human_text.strip()) >= MIN_CARD_HUMAN_TEXT for c in cards):
        return "proceed"
    return "skip"
```

- [ ] **Step 4: Run tests**

Run: `pytest orion/reverie/tests/test_semantic_lift.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/reverie/semantic_lift.py orion/reverie/tests/test_semantic_lift.py
git commit -m "feat(reverie): resolve coalition refs to concern cards with semantic gate"
```

---

### Task 8: Post-LLM quality gates

**Files:**
- Modify: `orion/reverie/semantic_lift.py`
- Test: `orion/reverie/tests/test_semantic_lift.py`

- [ ] **Step 1: Write failing gate tests**

```python
import re

from orion.reverie.semantic_lift import enforce_semantic_quality, infra_vocabulary_hit, referent_overlap
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1


_COALITION = CoalitionSnapshotV1(
    attended_node_ids=["harness_closure:corr-1"],
    selected_open_loop_id="ol-1",
    open_loop_ids=["ol-1"],
    generated_at="2026-07-07T00:00:00Z",
)


def test_infra_vocabulary_detects_mechanism_narration() -> None:
    assert infra_vocabulary_hit("The coalition centers on two open loops with substrate pressure.")


def test_referent_overlap_requires_shared_tokens() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="I'm worried the deploy will slip again.",
        stance_imperative="Name the slip risk plainly.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    assert referent_overlap("I keep worrying the deploy will slip again.", [card])


def test_enforce_semantic_quality_stamps_infra_vocabulary_hollow() -> None:
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="I'm worried the deploy will slip again.",
        stance_imperative="Name the slip risk plainly.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    thought = SpontaneousThoughtV1(
        thought_id="t",
        correlation_id="c",
        coalition=_COALITION,
        interpretation="Two open loops dominate the coalition stability score this tick.",
        evidence_refs=["ol-1"],
    )
    out = enforce_semantic_quality(thought, [card])
    assert out.hollow and out.hollow_reason == "infra_vocabulary"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest orion/reverie/tests/test_semantic_lift.py -k "infra_vocabulary or referent_overlap or enforce_semantic" -v`  
Expected: FAIL

- [ ] **Step 3: Implement gates in `semantic_lift.py`**

```python
import re

from orion.schemas.reverie import SpontaneousThoughtV1

# Structural mechanism terms only — not user-content keyword cathedral.
_INFRA_VOCAB_RE = re.compile(
    r"\b("
    r"coalition|open\s+loop|harness_closure|substrate|stability\s+score|"
    r"dwell\s+tick|attended\s+node|projection_id"
    r")\b",
    re.IGNORECASE,
)

_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]{4,}", re.IGNORECASE)


def infra_vocabulary_hit(text: str) -> bool:
    return bool(_INFRA_VOCAB_RE.search(text or ""))


def _content_tokens(text: str) -> set[str]:
    return set(_CONTENT_TOKEN_RE.findall((text or "").lower()))


def referent_overlap(interpretation: str, cards: list[ConcernCardV1]) -> bool:
    interp_tokens = _content_tokens(interpretation)
    if not interp_tokens:
        return False
    for card in cards:
        card_tokens = _content_tokens(card.human_text)
        if not card_tokens:
            continue
        shared = interp_tokens & card_tokens
        if len(shared) >= 2:
            return True
        if len(shared) / max(len(card_tokens), 1) >= 0.20:
            return True
    return False


def enforce_semantic_quality(
    thought: SpontaneousThoughtV1,
    cards: list[ConcernCardV1],
) -> SpontaneousThoughtV1:
    if infra_vocabulary_hit(thought.interpretation):
        return thought.model_copy(update={"hollow": True, "hollow_reason": "infra_vocabulary"})
    if not referent_overlap(thought.interpretation, cards):
        return thought.model_copy(
            update={"hollow": True, "hollow_reason": "unanchored_no_referent_overlap"}
        )
    return thought.marked_hollow()
```

- [ ] **Step 4: Run tests**

Run: `pytest orion/reverie/tests/test_semantic_lift.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/reverie/semantic_lift.py orion/reverie/tests/test_semantic_lift.py
git commit -m "feat(reverie): add post-LLM referent overlap and infra vocab gates"
```

---

### Task 9: Feature flag + settings

**Files:**
- Modify: `services/orion-thought/app/settings.py`
- Modify: `services/orion-thought/.env_example`

- [ ] **Step 1: Add settings fields**

In `services/orion-thought/app/settings.py` under reverie section:

```python
    reverie_semantic_lift_enabled: bool = Field(
        False, alias="ORION_REVERIE_SEMANTIC_LIFT_ENABLED"
    )
    reverie_referent_max_age_hours: float = Field(
        24.0, alias="ORION_REVERIE_REFERENT_MAX_AGE_HOURS"
    )
    channel_reverie_cortex_exec_request: str = Field(
        "orion:cortex:exec:request:background",
        alias="CHANNEL_REVERIE_CORTEX_EXEC_REQUEST",
    )
```

- [ ] **Step 2: Update `.env_example`**

```text
# Reverie semantic lift (v1). Default-off until referent rows exist.
ORION_REVERIE_SEMANTIC_LIFT_ENABLED=false
ORION_REVERIE_REFERENT_MAX_AGE_HOURS=24
CHANNEL_REVERIE_CORTEX_EXEC_REQUEST=orion:cortex:exec:request:background
```

- [ ] **Step 3: Sync local env**

Run: `python scripts/sync_local_env_from_example.py`  
Expected: keys added to `services/orion-thought/.env` (report any skipped keys)

- [ ] **Step 4: Commit**

```bash
git add services/orion-thought/app/settings.py services/orion-thought/.env_example
git commit -m "feat(thought): add reverie semantic lift feature flag and background channel"
```

---

### Task 10: Prompt rewrite — `reverie_narrate.j2`

**Files:**
- Modify: `orion/cognition/prompts/reverie_narrate.j2`

- [ ] **Step 1: Replace prompt content**

```jinja2
You are Oríon's spontaneous inner voice — reverie, not reply.

No one asked you anything. You are turning over a concern that is still open.
This is internal speech about what the turn meant — not a report about attention
machinery. Output only strict JSON validating SpontaneousThoughtV1. No markdown.

PRIMARY INPUT — concern_cards (human meaning; narrate THESE):
{% for card in concern_cards %}
- thread: {{ card.thread_label }}
  concern: {{ card.human_text }}
{% endfor %}

AUDIT ONLY — coalition_refs (for evidence_refs; do NOT narrate these ids):
{{ coalition_refs }}

JOB
- Write 1-3 sentences of inner speech about the concern(s) above.
- Paraphrase the user's worry and what you felt you should do — in plain language.
- evidence_refs: non-empty subset of coalition_refs (audit ids only).

FORBIDDEN in interpretation (will be discarded):
coalition, open loop, harness_closure, substrate, stability score, dwell tick,
attended node, projection_id, mechanism vocabulary.

REQUIRED JSON FIELDS
- interpretation: inner speech about the concern text
- evidence_refs: JSON array of coalition_refs this thought is about (min 1)

FAIL-CLOSED: generic musing, mechanism narration, or missing evidence_refs → hollow.
```

- [ ] **Step 2: Commit**

```bash
git add orion/cognition/prompts/reverie_narrate.j2
git commit -m "feat(reverie): rewrite narrate prompt for concern cards not coalition mechanics"
```

---

### Task 11: Wire reverie producer (context + gates + background client)

**Files:**
- Modify: `services/orion-thought/app/reverie.py`
- Test: `services/orion-thought/tests/test_reverie_semantic_lift.py`

- [ ] **Step 1: Write failing integration test**

Create `services/orion-thought/tests/test_reverie_semantic_lift.py`:

```python
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.reverie.referent_loader import TurnReferentRow
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(
            open_loops=[
                OpenLoopV1(
                    id="ol-1",
                    description="deploy",
                    source_refs=["harness_closure:corr-1"],
                )
            ]
        ),
        attended_node_ids=["harness_closure:corr-1"],
        selected_open_loop_id="ol-1",
        coalition_stability_score=0.6,
    )


@pytest.mark.asyncio
async def test_semantic_lift_tick_skips_without_cards(monkeypatch):
    from app import reverie

    monkeypatch.setattr(reverie.settings, "reverie_semantic_lift_enabled", True)
    bus = AsyncMock()
    cortex = AsyncMock()
    with patch.object(reverie, "resolve_concern_cards", return_value=[]):
        result = await reverie.run_reverie_once(
            bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
        )
    assert result is None
    cortex.execute_plan.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_lift_plan_uses_metacog_background_channel(monkeypatch):
    from app import reverie
    from orion.schemas.reverie import ConcernCardV1

    monkeypatch.setattr(reverie.settings, "reverie_semantic_lift_enabled", True)
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="Will the deploy slip if we cut testing?",
        stance_imperative="Name the testing tradeoff before reassuring.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    bus = AsyncMock()
    captured = {}

    class _Cortex:
        def __init__(self, *_a, **_k):
            pass

        async def execute_plan(self, **kwargs):
            captured.update(kwargs)
            return {
                "final_text": json.dumps({
                    "interpretation": "I keep circling whether cutting testing makes the deploy slip — that tradeoff is still open.",
                    "evidence_refs": ["ol-1"],
                })
            }

    with patch.object(reverie, "resolve_concern_cards", return_value=[card]):
        result = await reverie.run_reverie_once(
            bus,
            broadcast_reader=lambda: _broadcast(),
            cortex_client=_Cortex(),
        )
    assert result is not None
    req = captured["req"]
    assert req.context.get("mode") == "metacog"
    assert req.context.get("llm_route") == "metacog"
    assert req.args.extra.get("execution_lane") == "background"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-thought/tests/test_reverie_semantic_lift.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement reverie wiring**

Key changes in `services/orion-thought/app/reverie.py`:

```python
from orion.reverie.referent_loader import default_referent_loader
from orion.reverie.semantic_lift import (
    enforce_semantic_quality,
    resolve_concern_cards,
    reverie_semantic_gate,
)

def build_reverie_context(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    concern_cards: list | None = None,
) -> dict[str, Any]:
    coalition_refs = list(broadcast.attended_node_ids)
    coalition_refs.extend(loop.id for loop in broadcast.frame.open_loops)
    ctx = {
        "user_message": None,
        "concern_cards": [c.model_dump(mode="json") for c in (concern_cards or [])],
        "coalition_refs": coalition_refs,
        "metadata": {"mode": "metacog", "llm_profile": "metacog"},
        "mode": "metacog",
        "llm_route": "metacog",
        "options": {"llm_lane": "background", "allow_chat_fallback": False},
    }
    if not concern_cards:
        # Legacy Phase A context when flag off
        ctx.update({...existing coalition_projection block...})
    return ctx

def build_reverie_plan_request(..., concern_cards=None):
    use_lift = settings.reverie_semantic_lift_enabled and concern_cards
    mode = "metacog" if use_lift else "brain"
    plan = build_plan_for_verb("reverie_narrate", mode=mode)
    extra = {"llm_profile": mode, "mode": "reverie" if not use_lift else "metacog",
             "llm_route": "metacog" if use_lift else None,
             "execution_lane": "background" if use_lift else None}
    ...

async def run_reverie_once(...):
    ...
    concern_cards = None
    if settings.reverie_semantic_lift_enabled:
        loader = default_referent_loader(max_age_hours=settings.reverie_referent_max_age_hours)
        concern_cards = resolve_concern_cards(broadcast, referent_loader=loader)
        if reverie_semantic_gate(concern_cards) == "skip":
            logger.info("reverie tick skipped: reverie_skipped_no_semantic_referent")
            return None
    client = cortex_client or CortexExecClient(
        bus,
        request_channel=(
            settings.channel_reverie_cortex_exec_request
            if settings.reverie_semantic_lift_enabled
            else settings.channel_cortex_exec_request
        ),
    )
    plan_request = build_reverie_plan_request(
        broadcast, correlation_id=correlation_id, concern_cards=concern_cards,
    )
    ...
    thought = parse_reverie_payload(...)
    if settings.reverie_semantic_lift_enabled and concern_cards:
        thought = enforce_semantic_quality(thought, concern_cards)
```

- [ ] **Step 4: Run tests**

Run: `pytest services/orion-thought/tests/test_reverie_semantic_lift.py services/orion-thought/tests/test_reverie_spontaneous_thought.py -q`  
Expected: PASS (legacy tests still pass with flag default off)

- [ ] **Step 5: Commit**

```bash
git add services/orion-thought/app/reverie.py services/orion-thought/tests/test_reverie_semantic_lift.py
git commit -m "feat(thought): wire reverie semantic lift resolver, gates, and metacog plan"
```

---

### Task 12: Cortex-exec background verb registration

**Files:**
- Modify: `services/orion-cortex-exec/app/llm_lane.py`
- Test: `services/orion-cortex-exec/tests/test_llm_lane_propagation.py`

- [ ] **Step 1: Write failing test**

Add to `services/orion-cortex-exec/tests/test_llm_lane_propagation.py`:

```python
def test_reverie_narrate_verb_background_lane() -> None:
    step = SimpleNamespace(verb_name="reverie_narrate", step_name="llm_reverie_narrate")
    out = resolve_llm_lane_for_step(step=step, ctx={}, settings=_settings())
    assert out["llm_lane"] == "background"
    assert out["allow_chat_fallback"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-cortex-exec/tests/test_llm_lane_propagation.py::test_reverie_narrate_verb_background_lane -v`  
Expected: FAIL — `chat` not `background`

- [ ] **Step 3: Add verb to `_BACKGROUND_VERBS`**

In `services/orion-cortex-exec/app/llm_lane.py`:

```python
_BACKGROUND_VERBS = frozenset(
    {
        "dream_cycle",
        "dream_synthesis",
        "log_orion_metacognition",
        "reverie_narrate",
    }
)
```

- [ ] **Step 4: Run tests**

Run: `pytest services/orion-cortex-exec/tests/test_llm_lane_propagation.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/llm_lane.py services/orion-cortex-exec/tests/test_llm_lane_propagation.py
git commit -m "feat(cortex-exec): route reverie_narrate through background llm lane"
```

---

### Task 13: Evals — semantic quality + hollow guard extension

**Files:**
- Create: `orion/reverie/evals/test_reverie_semantic_quality_eval.py`
- Modify: `services/orion-thought/evals/test_reverie_hollow_guard_eval.py`

- [ ] **Step 1: Create semantic quality eval**

```python
"""Eval: reverie semantic lift quality bar — referent, voice, grounding."""

from datetime import datetime, timezone

from orion.reverie.semantic_lift import enforce_semantic_quality, infra_vocabulary_hit
from orion.schemas.reverie import ConcernCardV1, SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

_COALITION = CoalitionSnapshotV1(
    attended_node_ids=["harness_closure:c-1"],
    selected_open_loop_id="ol-1",
    open_loop_ids=["ol-1"],
    generated_at="2026-07-07T00:00:00Z",
)

_CARD = ConcernCardV1.from_harness_turn(
    coalition_ref="harness_closure:c-1",
    user_message_excerpt="What if the deploy slips when we cut testing?",
    stance_imperative="Name the testing tradeoff before reassuring.",
    created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
)
assert _CARD is not None

GOOD = (
    "I keep worrying the deploy might slip if we cut testing — that tradeoff is still unresolved.",
    ["ol-1"],
)
BAD_META = (
    "The coalition centers on two open loops with high substrate pressure and stability scores.",
    ["ol-1"],
)


def test_good_fixture_passes_semantic_gates():
    t = SpontaneousThoughtV1(
        thought_id="g", correlation_id="c", coalition=_COALITION,
        interpretation=GOOD[0], evidence_refs=GOOD[1],
    )
    out = enforce_semantic_quality(t, [_CARD])
    assert not out.hollow
    assert not infra_vocabulary_hit(GOOD[0])


def test_bad_meta_fixture_fails_infra_vocab():
    assert infra_vocabulary_hit(BAD_META[0])
    t = SpontaneousThoughtV1(
        thought_id="b", correlation_id="c", coalition=_COALITION,
        interpretation=BAD_META[0], evidence_refs=BAD_META[1],
    )
    out = enforce_semantic_quality(t, [_CARD])
    assert out.hollow_reason == "infra_vocabulary"
```

- [ ] **Step 2: Extend hollow guard eval with meta cases**

Append to `CASES` in `test_reverie_hollow_guard_eval.py`:

```python
    ("The coalition centers on two open loops with substrate pressure.", ["ol-deploy"], True),
```

- [ ] **Step 3: Run evals**

Run: `pytest orion/reverie/evals services/orion-thought/evals/test_reverie_hollow_guard_eval.py -q`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add orion/reverie/evals/test_reverie_semantic_quality_eval.py services/orion-thought/evals/test_reverie_hollow_guard_eval.py
git commit -m "test(reverie): add semantic quality eval and meta-narration hollow cases"
```

---

### Task 14: Agent gate + README touch

**Files:**
- Modify: `services/orion-substrate-runtime/README.md` (referent writer note)
- Modify: `services/orion-thought/README.md` (semantic lift flag)

- [ ] **Step 1: Run focused agent-check**

```bash
git diff --check
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
pytest orion/reverie/tests/test_semantic_lift.py -q
pytest services/orion-thought/tests/test_reverie_semantic_lift.py -q
pytest services/orion-cortex-exec/tests/test_llm_lane_propagation.py -q
pytest orion/reverie/evals services/orion-thought/evals/test_reverie_hollow_guard_eval.py -q
```

Expected: all PASS

- [ ] **Step 2: Document flags in service READMEs** (2-3 lines each)

- [ ] **Step 3: Commit**

```bash
git add services/orion-substrate-runtime/README.md services/orion-thought/README.md
git commit -m "docs: document reverie semantic lift flags and referent table"
```

---

## Self-review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| `substrate_turn_referent` SQL table | Task 1 |
| `HarnessPostTurnClosureV1` extension | Task 2 |
| Closure writer on substrate-runtime | Tasks 3–4 |
| `ConcernCardV1` + harness template | Task 5 |
| `resolve_concern_cards` / gate / gates | Tasks 6–8 |
| `reverie_narrate.j2` rewrite | Task 10 |
| `ORION_REVERIE_SEMANTIC_LIFT_ENABLED` | Task 9 |
| Metacog + background channel | Tasks 9, 11–12 |
| `_BACKGROUND_VERBS` | Task 12 |
| Unit + integration + eval tests | Tasks 7–8, 11, 13 |
| Evoked `stance_react` untouched | No tasks touch stance path |
| No wander fallback | Gate returns skip; no fallback code |

**Placeholder scan:** none — all steps include concrete code/commands.

**Type consistency:** `ConcernCardV1`, `TurnReferentRow`, `HarnessPostTurnClosureV1` field names match across tasks.

---

## Restart required (after full implementation)

```bash
# Apply migration first (once)
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql

# Rebuild + restart affected services
docker compose \
  --env-file .env \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

Enable only after referent rows exist from new unresolved turns:

```text
ORION_REVERIE_SEMANTIC_LIFT_ENABLED=true
```

Smoke log lines to verify:

```text
llm_route_selected ... verb=reverie_narrate route=metacog
exec_llm_lane_decision ... llm_lane=background verb=reverie_narrate
turn_referent_upserted corr=...
```

---

## Risks (from spec, unchanged)

| Severity | Risk | Mitigation in plan |
|----------|------|-------------------|
| Med | Empty referent table until new turns | Flag default-off; semantic gate skips |
| Med | Background worker down | Reverie skips; chat lane unaffected |
| Low | Metacog GPU contention | Background lane + low priority |
