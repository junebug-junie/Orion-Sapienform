# Purpose-Conditioned Recall (PCR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blind pre-stance recall in `chat_general` with purpose-conditioned retrieval — Phase 0 skip, Phase 1 continuity, post-stance Phase 3 belief fetch — so crystallizations reach chat via `active_packet` and digests are split (`continuity_digest` / `belief_digest`).

**Architecture:** Cortex-exec orchestrates PCR in `pcr_chat_memory.py` (router + supervisor hooks). Recall worker branches on `recall_phase` + `retrieval_intent` with new profiles and collectors. Intent is derived deterministically from stance brief + appraisal (no LLM router). Legacy path preserved when `CHAT_PCR_ENABLED=false`.

**Tech Stack:** Python 3.12, Pydantic v2, asyncpg, Redis bus RPC, pytest, bash smoke, Jinja2 prompts.

**Spec:** `docs/superpowers/specs/2026-07-07-purpose-conditioned-recall-design.md`

**Program note:** Ship Milestones A→B **before** consolidation gate (write path). Shared `orion/memory/low_info_social.py` is owned here; consolidation plan depends on it.

**Branch:** `feat/pcr-chat-memory`

**Worktree setup:**

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
git worktree add ../Orion-Sapienform-pcr -b feat/pcr-chat-memory
cd ../Orion-Sapienform-pcr
```

---

## File Map

| File | Milestone | Action | Responsibility |
|------|-----------|--------|----------------|
| `orion/memory/low_info_social.py` | A | Create | Shared courtesy/greeting detector (extracted from recall fusion) |
| `orion/memory/recall_skip_gate.py` | A | Create | Phase 0 deterministic skip gate |
| `orion/schemas/recall_pcr.py` | A | Create | `RecallPhaseV1`, `RetrievalIntentV1`, `PcrChatMemoryV1` |
| `orion/core/contracts/recall.py` | A,B | Modify | Extend `RecallQueryV1` with PCR fields |
| `orion/schemas/registry.py` | A | Modify | Register PCR schema types if needed |
| `orion/recall/profiles/chat.continuity.v1.yaml` | A | Create | Phase 1 sql_chat-only profile |
| `services/orion-recall/app/fusion.py` | A,B | Modify | Use shared low_info; add `render_continuity_bundle`, `pcr_fuse_belief_candidates` |
| `services/orion-recall/app/worker.py` | A,B | Modify | Branch on `recall_phase` / `retrieval_intent` |
| `services/orion-cortex-exec/app/pcr_chat_memory.py` | A,B | Create | PCR orchestration helper |
| `services/orion-cortex-exec/app/router.py` | A,B | Modify | Replace single pre-recall with PCR hooks |
| `services/orion-cortex-exec/app/supervisor.py` | B | Modify | Supervisor parity via shared helper |
| `services/orion-cortex-exec/app/executor.py` | A,B | Modify | Pass PCR fields into `RecallQueryV1` |
| `services/orion-cortex-exec/app/chat_stance.py` | A | Modify | Hoist `continuity_digest` into stance inputs |
| `services/orion-cortex-exec/app/settings.py` | A,B | Modify | `CHAT_PCR_*` settings |
| `services/orion-cortex-exec/.env_example` | A,B | Modify | PCR env keys |
| `services/orion-recall/app/settings.py` | A,B | Modify | `RECALL_PCR_*` settings |
| `services/orion-recall/.env_example` | A,B | Modify | PCR env keys |
| `orion/cognition/prompts/chat_stance_brief.j2` | A | Modify | `continuity_digest` slot |
| `orion/cognition/prompts/chat_general.j2` | B | Modify | `continuity_digest` + `belief_digest` |
| `orion/memory/retrieval_intent.py` | B | Create | Post-stance intent derivation |
| `orion/recall/profiles/chat.belief.*.v1.yaml` | B | Create | Five belief profiles |
| `services/orion-recall/app/pcr_collectors.py` | B | Create | Backend plan per intent |
| `services/orion-recall/app/collectors/active_packet.py` | B | Create | In-process `retrieve_active_packet` collector |
| `tests/test_low_info_social.py` | A | Create | Shared detector tests |
| `tests/test_recall_skip_gate.py` | A | Create | Phase 0 tests |
| `tests/test_retrieval_intent.py` | B | Create | Intent rule table tests |
| `services/orion-recall/tests/test_pcr_continuity.py` | A | Create | Continuity fusion/worker |
| `services/orion-recall/tests/test_pcr_belief.py` | B | Create | Belief fusion + active_packet |
| `services/orion-cortex-exec/tests/test_pcr_chat_memory.py` | A,B | Create | Orchestration + greeting skip |
| `scripts/smoke_pcr_chat_memory_e2e.sh` | B | Create | Live stack smoke |

---

# Milestone A — Read relief (Phase 0 + Phase 1)

## Task 1: Shared `low_info_social` module

**Files:**
- Create: `orion/memory/low_info_social.py`
- Create: `tests/test_low_info_social.py`
- Modify: `services/orion-recall/app/fusion.py` (import shared helper)

- [ ] **Step 1: Write the failing test**

Create `tests/test_low_info_social.py`:

```python
from orion.memory.low_info_social import is_low_info_social


def test_greeting_is_low_info():
    assert is_low_info_social("hey Orion") is True


def test_substantive_is_not_low_info():
    assert is_low_info_social("still drowning in move logistics alone") is False


def test_thanks_is_low_info():
    assert is_low_info_social("thanks!") is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
pytest tests/test_low_info_social.py -v
```

Expected: `ModuleNotFoundError: orion.memory.low_info_social`

- [ ] **Step 3: Write minimal implementation**

Create `orion/memory/low_info_social.py`:

```python
from __future__ import annotations

import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_low_info_social(text: str) -> bool:
    """True for greetings, thanks, and other low-information social turns."""
    candidate = _normalize(text).lower()
    if not candidate:
        return True
    normalized = re.sub(r"[^a-z0-9' ]+", " ", candidate)
    tokens = re.findall(r"[a-z']{2,}", normalized)
    if len(tokens) > 18:
        return False
    courtesy_patterns = [
        r"^(hi|hey|hello|yo|sup|hiya|good (morning|afternoon|evening))( there| friend| orion| juniper)?[!. ]*$",
        r"^(thanks|thank you|awesome|cool|nice|sounds good|all good|doing good|doing well|glad to hear)[!. ]*$",
        r"^(how are you|how's it going|hope you're well|hope you are well)[?.! ]*$",
    ]
    if any(re.match(p, candidate, flags=re.I) for p in courtesy_patterns):
        return True
    if len(tokens) <= 6:
        low_info_terms = {
            "hi", "hey", "hello", "thanks", "thank", "good", "great", "cool", "nice",
            "fine", "well", "friend", "orion", "juniper", "all", "doing", "okay", "ok", "for", "now",
        }
        if all(t in low_info_terms for t in tokens):
            return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_low_info_social.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Wire fusion to shared helper**

In `services/orion-recall/app/fusion.py`, add at top:

```python
from orion.memory.low_info_social import is_low_info_social as _shared_is_low_info_social
```

Replace body of `_is_low_info_social_candidate` with:

```python
def _is_low_info_social_candidate(snippet: str) -> bool:
    text = _normalize_whitespace(snippet).lower()
    if not text:
        return True
    user, assistant = _extract_transcript_parts(text)
    candidate = _normalize_whitespace(f"{user} {assistant}" if (user or assistant) else text)
    return _shared_is_low_info_social(candidate)
```

- [ ] **Step 6: Run recall salience tests (regression)**

```bash
pytest services/orion-recall/tests/test_chat_general_recall_salience.py -q
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add orion/memory/low_info_social.py tests/test_low_info_social.py services/orion-recall/app/fusion.py
git commit -m "feat(pcr): extract shared low_info_social detector"
```

---

## Task 2: PCR schema types + RecallQueryV1 extensions

**Files:**
- Create: `orion/schemas/recall_pcr.py`
- Modify: `orion/core/contracts/recall.py`
- Modify: `orion/schemas/registry.py` (if registry lists contract literals)

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_recall_pcr_contract.py`:

```python
from orion.core.contracts.recall import RecallQueryV1
from orion.schemas.recall_pcr import RecallPhaseV1, RetrievalIntentV1


def test_recall_query_accepts_pcr_fields():
    q = RecallQueryV1(
        fragment="hello",
        profile="chat.continuity.v1",
        recall_phase="continuity",
        retrieval_intent="continuity",
        task_hints={"task_mode": "casual"},
    )
    assert q.recall_phase == "continuity"
    assert q.retrieval_intent == "continuity"
    assert RecallPhaseV1.__args__[0] == "skip"  # type: ignore[attr-defined]
    assert "semantic" in RetrievalIntentV1.__args__  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_recall_pcr_contract.py -v
```

Expected: `TypeError` or `ValidationError` — unknown fields on `RecallQueryV1`

- [ ] **Step 3: Create schema module**

Create `orion/schemas/recall_pcr.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RecallPhaseV1 = Literal["skip", "continuity", "purposeful"]

RetrievalIntentV1 = Literal[
    "none",
    "continuity",
    "relational",
    "semantic",
    "procedural",
    "open_loop",
    "contradiction",
]


@dataclass
class PcrChatMemoryV1:
    phase: RecallPhaseV1
    retrieval_intent: RetrievalIntentV1 | None
    continuity_digest: str = ""
    belief_digest: str = ""
    memory_digest: str = ""
    skip_reasons: list[str] = field(default_factory=list)
    recall_debug: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Extend RecallQueryV1**

In `orion/core/contracts/recall.py`, add imports and fields to `RecallQueryV1`:

```python
from typing import Any  # ensure present

# after reply_to field:
    recall_phase: Optional[str] = Field(None, description="PCR phase: skip|continuity|purposeful")
    retrieval_intent: Optional[str] = Field(None, description="PCR retrieval intent")
    task_hints: Optional[Dict[str, Any]] = Field(None, description="Stance/appraisal hints for PCR")
    seed_crystallization_id: Optional[str] = None
    continuity_digest_max_tokens: Optional[int] = None
    belief_digest_max_tokens: Optional[int] = None
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_recall_pcr_contract.py -v
```

Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/recall_pcr.py orion/core/contracts/recall.py tests/test_recall_pcr_contract.py
git commit -m "feat(pcr): add RecallQueryV1 PCR contract fields"
```

---

## Task 3: Phase 0 `recall_skip_gate`

**Files:**
- Create: `orion/memory/recall_skip_gate.py`
- Create: `tests/test_recall_skip_gate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recall_skip_gate.py`:

```python
from orion.memory.recall_skip_gate import recall_skip_gate


def test_skips_greeting_with_low_novelty():
    result = recall_skip_gate(
        user_message="hey Orion",
        appraisal={"novelty_score": 0.1, "shift_kind": "NONE"},
        has_repair_grammar_signal=False,
        max_novelty=0.25,
        shift_novelty_floor=0.35,
    )
    assert result.skip is True
    assert "low_info_social" in result.reasons


def test_no_skip_when_repair_signal():
    result = recall_skip_gate(
        user_message="hey",
        appraisal={"novelty_score": 0.1, "shift_kind": "NONE"},
        has_repair_grammar_signal=True,
        max_novelty=0.25,
        shift_novelty_floor=0.35,
    )
    assert result.skip is False


def test_no_skip_substantive_topic():
    result = recall_skip_gate(
        user_message="still drowning in move logistics alone",
        appraisal={"novelty_score": 0.72, "shift_kind": "TOPIC"},
        has_repair_grammar_signal=False,
        max_novelty=0.25,
        shift_novelty_floor=0.35,
    )
    assert result.skip is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_recall_skip_gate.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement gate**

Create `orion/memory/recall_skip_gate.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from orion.memory.low_info_social import is_low_info_social

_SHIFT_KINDS_FORCE_RECALL = frozenset({"TOPIC", "STANCE", "REPAIR"})


@dataclass
class RecallSkipGateResult:
    skip: bool
    reasons: list[str] = field(default_factory=list)


def recall_skip_gate(
    *,
    user_message: str,
    appraisal: dict | None,
    has_repair_grammar_signal: bool,
    max_novelty: float = 0.25,
    shift_novelty_floor: float = 0.35,
) -> RecallSkipGateResult:
    reasons: list[str] = []
    if has_repair_grammar_signal:
        return RecallSkipGateResult(skip=False, reasons=[])

    if is_low_info_social(user_message):
        reasons.append("low_info_social")
    else:
        return RecallSkipGateResult(skip=False, reasons=[])

    novelty = None
    shift_kind = "NONE"
    if isinstance(appraisal, dict):
        raw_novelty = appraisal.get("novelty_score")
        if isinstance(raw_novelty, (int, float)):
            novelty = float(raw_novelty)
        shift_kind = str(appraisal.get("shift_kind") or "NONE").upper()

    if novelty is None or novelty < max_novelty:
        reasons.append("novelty_below_floor")
    elif shift_kind in _SHIFT_KINDS_FORCE_RECALL and novelty >= shift_novelty_floor:
        return RecallSkipGateResult(skip=False, reasons=[])

    if shift_kind in _SHIFT_KINDS_FORCE_RECALL and novelty is not None and novelty < shift_novelty_floor:
        reasons.append("shift_below_novelty_floor")

    return RecallSkipGateResult(skip=True, reasons=reasons)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_recall_skip_gate.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add orion/memory/recall_skip_gate.py tests/test_recall_skip_gate.py
git commit -m "feat(pcr): add deterministic recall_skip_gate"
```

---

## Task 4: Continuity recall profile + fusion render lane

**Files:**
- Create: `orion/recall/profiles/chat.continuity.v1.yaml`
- Modify: `services/orion-recall/app/fusion.py`

- [ ] **Step 1: Write failing fusion test**

Create `services/orion-recall/tests/test_pcr_continuity.py`:

```python
from services.orion_recall.app.fusion import render_continuity_bundle  # adjust import path for test env


def test_render_continuity_bundle_user_prioritized():
    candidates = [
        {"id": "t1", "source": "sql_chat", "snippet": "User: hey\nAssistant: hi", "score": 0.5},
        {"id": "t2", "source": "sql_chat", "snippet": "User: move stress\nAssistant: I hear you", "score": 0.8},
    ]
    profile = {"render_budget_tokens": 96, "max_total_items": 6, "render_lane": "continuity"}
    bundle, _ = render_continuity_bundle(
        candidates=candidates,
        profile=profile,
        query_text="move stress",
        latency_ms=1,
    )
    assert "move stress" in bundle.rendered
    assert bundle.stats.profile == "continuity"
```

Use import that works in repo tests — typically:

```python
from app.fusion import render_continuity_bundle
```

when run from `services/orion-recall`.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/orion-recall && pytest tests/test_pcr_continuity.py::test_render_continuity_bundle_user_prioritized -v
```

Expected: `ImportError: cannot import name 'render_continuity_bundle'`

- [ ] **Step 3: Add profile YAML**

Create `orion/recall/profiles/chat.continuity.v1.yaml`:

```yaml
profile: chat.continuity.v1
enable_vector: false
vector_top_k: 0
rdf_top_k: 0
cards_top_k: 0
enable_sql_timeline: false
sql_chat_top_k: 6
sql_since_minutes: 120
max_per_source: 4
max_total_items: 6
render_budget_tokens: 96
render_lane: continuity
render_transcript_user_only: true
relevance:
  backend_weights:
    sql_chat: 1.0
  enable_recency: true
  recency_weight: 0.4
```

- [ ] **Step 4: Add `render_continuity_bundle` to fusion.py**

Add function after `fuse_candidates`:

```python
def render_continuity_bundle(
    *,
    candidates: Iterable[Dict[str, Any]],
    profile: Dict[str, Any],
    query_text: str | None = None,
    latency_ms: int = 0,
    session_id: str | None = None,
) -> Tuple[MemoryBundleV1, List[Dict[str, Any]]]:
    """PCR Phase 1: sql_chat-only transcript-shaped render."""
    narrowed = dict(profile)
    narrowed.setdefault("max_total_items", 6)
    narrowed.setdefault("render_budget_tokens", 96)
    narrowed["render_lane"] = "continuity"
    bundle, ranking = fuse_candidates(
        candidates=candidates,
        profile=narrowed,
        latency_ms=latency_ms,
        query_text=query_text or "",
        session_id=session_id,
        substantive_query=False,
        diagnostic=False,
    )
    header = "[Recent thread — continuity only]\n"
    rendered = header + (bundle.rendered or "")
    if bundle.stats:
        bundle.stats.profile = str(profile.get("profile") or "chat.continuity.v1")
    bundle.rendered = rendered.strip()
    return bundle, ranking
```

- [ ] **Step 5: Run test**

```bash
cd services/orion-recall && pytest tests/test_pcr_continuity.py -v
```

Expected: pass (adjust assertion if profile field differs)

- [ ] **Step 6: Commit**

```bash
git add orion/recall/profiles/chat.continuity.v1.yaml services/orion-recall/app/fusion.py services/orion-recall/tests/test_pcr_continuity.py
git commit -m "feat(pcr): add chat.continuity.v1 profile and render lane"
```

---

## Task 5: Recall worker Phase 1 branch

**Files:**
- Modify: `services/orion-recall/app/worker.py`
- Modify: `services/orion-recall/app/settings.py`
- Modify: `services/orion-recall/.env_example`

- [ ] **Step 1: Write failing worker test**

Add to `services/orion-recall/tests/test_pcr_continuity.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from app.worker import handle_recall_query


@pytest.mark.asyncio
async def test_worker_uses_continuity_render_when_phase_continuity():
    q = {
        "fragment": "hey",
        "profile": "chat.continuity.v1",
        "recall_phase": "continuity",
        "session_id": "sess-1",
    }
    with patch("app.worker.fetch_chat_history_pairs", new=AsyncMock(return_value=[])):
        with patch("app.worker.render_continuity_bundle") as mock_render:
            from orion.core.contracts.recall import MemoryBundleV1, MemoryBundleStatsV1
            mock_render.return_value = (MemoryBundleV1(rendered="ok", stats=MemoryBundleStatsV1()), [])
            # invoke internal handler used by worker — adapt to actual function name
            ...
```

Inspect `worker.py` for the main query handler (likely `handle_recall_query` or similar) and call it with a `RecallQueryV1`.

- [ ] **Step 2: Add settings**

In `services/orion-recall/app/settings.py`:

```python
    recall_pcr_enabled: bool = Field(True, alias="RECALL_PCR_ENABLED")
    recall_skip_max_novelty: float = Field(0.25, alias="RECALL_SKIP_MAX_NOVELTY")
    recall_continuity_sql_minutes: int = Field(120, alias="RECALL_CONTINUITY_SQL_MINUTES")
    recall_continuity_render_budget: int = Field(96, alias="RECALL_CONTINUITY_RENDER_BUDGET")
```

Mirror keys in `.env_example`.

- [ ] **Step 3: Branch in worker**

Near fusion call site in worker, before `fuse_candidates`:

```python
    recall_phase = getattr(q, "recall_phase", None) or (q.model_dump().get("recall_phase") if hasattr(q, "model_dump") else None)
    if settings.recall_pcr_enabled and recall_phase == "continuity":
        from .fusion import render_continuity_bundle
        profile["sql_since_minutes"] = settings.recall_continuity_sql_minutes
        profile["render_budget_tokens"] = settings.recall_continuity_render_budget
        bundle, ranking_debug = render_continuity_bundle(
            candidates=all_candidates,
            profile=profile,
            query_text=q.fragment,
            latency_ms=latency_ms,
            session_id=q.session_id,
        )
        # skip default fuse_candidates path
```

- [ ] **Step 4: Run recall tests**

```bash
pytest services/orion-recall/tests -q
```

- [ ] **Step 5: Sync env**

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-recall/app/worker.py services/orion-recall/app/settings.py services/orion-recall/.env_example services/orion-recall/tests/test_pcr_continuity.py
git commit -m "feat(pcr): recall worker continuity phase branch"
```

---

## Task 6: Cortex-exec settings + executor PCR query fields

**Files:**
- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/.env_example`
- Modify: `services/orion-cortex-exec/app/executor.py`

- [ ] **Step 1: Add cortex PCR settings**

```python
    chat_pcr_enabled: bool = Field(True, alias="CHAT_PCR_ENABLED")
    chat_pcr_post_stance_recall: bool = Field(True, alias="CHAT_PCR_POST_STANCE_RECALL")
    chat_pcr_skip_on_low_info: bool = Field(True, alias="CHAT_PCR_SKIP_ON_LOW_INFO")
    chat_pcr_quick_phase3: bool = Field(False, alias="CHAT_PCR_QUICK_PHASE3")
```

- [ ] **Step 2: Extend `run_recall_step` signature**

Add optional kwargs to `run_recall_step`:

```python
    recall_phase: str | None = None,
    retrieval_intent: str | None = None,
    task_hints: dict | None = None,
```

Pass into `RecallQueryV1(...)` construction:

```python
        recall_phase=recall_phase,
        retrieval_intent=retrieval_intent,
        task_hints=task_hints,
```

- [ ] **Step 3: Run executor recall tests**

```bash
pytest services/orion-cortex-exec/tests/test_recall_active_turn_exclusion_payload.py -q
```

- [ ] **Step 4: Sync env + commit**

```bash
python scripts/sync_local_env_from_example.py
git add services/orion-cortex-exec/app/settings.py services/orion-cortex-exec/.env_example services/orion-cortex-exec/app/executor.py
git commit -m "feat(pcr): cortex settings and recall step PCR fields"
```

---

## Task 7: `pcr_chat_memory.py` Phase 0+1 orchestration

**Files:**
- Create: `services/orion-cortex-exec/app/pcr_chat_memory.py`
- Create: `services/orion-cortex-exec/tests/test_pcr_chat_memory.py`

- [ ] **Step 1: Write failing orchestration test**

```python
import pytest
from unittest.mock import AsyncMock, patch

from app.pcr_chat_memory import run_pcr_phase0_and_1


@pytest.mark.asyncio
async def test_phase0_skips_recall_on_greeting():
    bus = AsyncMock()
    ctx = {"user_message": "hey", "spark_meta": {"turn_change_appraisal": {"novelty_score": 0.1, "shift_kind": "NONE"}}}
    with patch("app.settings.settings") as mock_settings:
        mock_settings.chat_pcr_enabled = True
        mock_settings.chat_pcr_skip_on_low_info = True
        result = await run_pcr_phase0_and_1(bus, ctx=ctx, correlation_id="c1", recall_cfg={}, settings=mock_settings)
    assert result.phase == "skip"
    assert result.continuity_digest == ""
    bus.publish.assert_not_called()
```

- [ ] **Step 2: Implement Phase 0+1 helper**

Create `services/orion-cortex-exec/app/pcr_chat_memory.py`:

```python
from __future__ import annotations

from typing import Any

from orion.memory.recall_skip_gate import recall_skip_gate
from orion.schemas.recall_pcr import PcrChatMemoryV1

from .executor import run_recall_step


def _appraisal_from_ctx(ctx: dict[str, Any]) -> dict | None:
    spark = ctx.get("spark_meta") if isinstance(ctx.get("spark_meta"), dict) else {}
    appraisal = spark.get("turn_change_appraisal")
    return appraisal if isinstance(appraisal, dict) else None


async def run_pcr_phase0_and_1(
    bus,
    *,
    ctx: dict[str, Any],
    correlation_id: str,
    recall_cfg: dict[str, Any],
    settings,
    source,
) -> PcrChatMemoryV1:
    if not settings.chat_pcr_enabled:
        return PcrChatMemoryV1(phase="continuity", retrieval_intent=None)

    user_message = str(ctx.get("user_message") or ctx.get("prompt") or "")
    appraisal = _appraisal_from_ctx(ctx)
    has_repair = bool(ctx.get("has_repair_grammar_signal"))

    if settings.chat_pcr_skip_on_low_info:
        gate = recall_skip_gate(
            user_message=user_message,
            appraisal=appraisal,
            has_repair_grammar_signal=has_repair,
            max_novelty=getattr(settings, "recall_skip_max_novelty", 0.25),
        )
        if gate.skip:
            return PcrChatMemoryV1(
                phase="skip",
                retrieval_intent="none",
                skip_reasons=gate.reasons,
            )

    recall_step, recall_debug, continuity_digest = await run_recall_step(
        bus,
        source=source,
        ctx=ctx,
        correlation_id=correlation_id,
        recall_cfg={**recall_cfg, "profile": "chat.continuity.v1"},
        recall_profile="chat.continuity.v1",
        recall_phase="continuity",
        step_name="pcr_continuity",
    )
    ctx["continuity_digest"] = continuity_digest
    return PcrChatMemoryV1(
        phase="continuity",
        retrieval_intent="continuity",
        continuity_digest=continuity_digest,
        memory_digest=continuity_digest,
        recall_debug=recall_debug if isinstance(recall_debug, dict) else {},
    )
```

- [ ] **Step 3: Run test + commit**

```bash
pytest services/orion-cortex-exec/tests/test_pcr_chat_memory.py -v
git add services/orion-cortex-exec/app/pcr_chat_memory.py services/orion-cortex-exec/tests/test_pcr_chat_memory.py
git commit -m "feat(pcr): add phase 0+1 orchestration helper"
```

---

## Task 8: Router hook + stance prompt (Milestone A complete)

**Files:**
- Modify: `services/orion-cortex-exec/app/router.py` (~1038–1055)
- Modify: `services/orion-cortex-exec/app/chat_stance.py`
- Modify: `orion/cognition/prompts/chat_stance_brief.j2`

- [ ] **Step 1: Replace blind pre-recall in router**

When `settings.chat_pcr_enabled` and verb is `chat_general` (or chat lane):

```python
from .pcr_chat_memory import run_pcr_phase0_and_1

pcr_early = await run_pcr_phase0_and_1(
    bus, ctx=ctx, correlation_id=correlation_id, recall_cfg=recall_cfg, settings=settings, source=source,
)
ctx["pcr_memory"] = pcr_early
if pcr_early.phase == "skip":
    should_recall = False  # skip legacy pre-recall
elif pcr_early.phase == "continuity":
    ctx["memory_digest"] = pcr_early.continuity_digest
    should_recall = False  # phase 1 already ran
```

Keep legacy `run_recall_step` block behind `if should_recall and not inline_recall and not settings.chat_pcr_enabled`.

- [ ] **Step 2: Hoist continuity_digest in chat_stance**

Ensure `build_chat_stance_inputs` includes `continuity_digest=ctx.get("continuity_digest", "")`.

- [ ] **Step 3: Update chat_stance_brief.j2**

```jinja2
- continuity_digest: {{ continuity_digest | default("") }}
```

Add instruction: continuity is thread orientation only, not durable belief.

- [ ] **Step 4: Run router tests**

```bash
pytest services/orion-cortex-exec/tests/test_router_chat_general_recall_profile_default.py -q
pytest services/orion-cortex-exec/tests/test_pcr_chat_memory.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/router.py services/orion-cortex-exec/app/chat_stance.py orion/cognition/prompts/chat_stance_brief.j2
git commit -m "feat(pcr): wire phase 0+1 in router and stance prompt"
```

---

# Milestone B — Purposeful recall (Phase 3)

## Task 9: `derive_retrieval_intent`

**Files:**
- Create: `orion/memory/retrieval_intent.py`
- Create: `tests/test_retrieval_intent.py`

- [ ] **Step 1: Write table-driven failing tests**

```python
import pytest
from orion.memory.recall_skip_gate import RecallSkipGateResult
from orion.memory.retrieval_intent import derive_retrieval_intent


@pytest.mark.parametrize(
    "stance,attention,appraisal,expected_intent,expected_rule",
    [
        ({"task_mode": "reflective_dialogue"}, {}, {"shift_kind": "NONE"}, "relational", "relational_mode"),
        ({"task_mode": "instrumental"}, {}, {"shift_kind": "TOPIC", "novelty_score": 0.7}, "semantic", "topic_shift"),
        ({}, {"open_loops": [{"id": "gpu"}]}, {}, "open_loop", "open_loops_present"),
    ],
)
def test_derive_retrieval_intent_rules(stance, attention, appraisal, expected_intent, expected_rule):
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief=stance,
        attention_frame=attention,
        appraisal=appraisal,
        hub_chat_lane=None,
        user_message="test",
        shift_novelty_floor=0.35,
    )
    assert intent == expected_intent
    assert rule_id == expected_rule
```

- [ ] **Step 2: Implement `derive_retrieval_intent`**

Create `orion/memory/retrieval_intent.py` per spec priority table (`phase0_skip`, `open_loops_present`, `repair_shift`, `relational_mode`, `stance_shift`, `procedural_mode`, `topic_shift`, `continuity_only`).

- [ ] **Step 3: Run tests + commit**

```bash
pytest tests/test_retrieval_intent.py -v
git add orion/memory/retrieval_intent.py tests/test_retrieval_intent.py
git commit -m "feat(pcr): deterministic retrieval intent derivation"
```

---

## Task 10: Belief recall profiles (5 YAML files)

**Files:**
- Create: `orion/recall/profiles/chat.belief.relational.v1.yaml`
- Create: `orion/recall/profiles/chat.belief.semantic.v1.yaml`
- Create: `orion/recall/profiles/chat.belief.procedural.v1.yaml`
- Create: `orion/recall/profiles/chat.belief.open_loop.v1.yaml`
- Create: `orion/recall/profiles/chat.belief.contradiction.v1.yaml`

- [ ] **Step 1: Create semantic profile (template for others)**

```yaml
profile: chat.belief.semantic.v1
enable_vector: false
vector_top_k: 0
rdf_top_k: 4
cards_top_k: 6
enable_sql_timeline: false
sql_chat_top_k: 0
max_per_source: 4
max_total_items: 8
render_budget_tokens: 128
render_lane: belief
relevance:
  backend_weights:
    active_packet: 1.0
    cards: 0.6
    rdf: 0.4
    rdf_chat: 0.45
  enable_recency: false
```

Relational: boost cards + rdf dispositions, `sql_chat_top_k: 0`. Procedural: `rdf_top_k: 0`, procedure tags. Open_loop: light rdf. Contradiction: enable graphiti when `RECALL_GRAPHITI_IN_CHAT=true`.

- [ ] **Step 2: Verify profiles load**

```bash
python -c "from services.orion_recall.app.profiles import get_profile; print(get_profile('chat.belief.semantic.v1')['profile'])"
```

Adjust import path to match repo layout (`app.profiles.get_profile` from `services/orion-recall`).

- [ ] **Step 3: Commit**

```bash
git add orion/recall/profiles/chat.belief.*.v1.yaml
git commit -m "feat(pcr): add belief recall profiles per intent"
```

---

## Task 11: `pcr_collectors` + `active_packet` collector

**Files:**
- Create: `services/orion-recall/app/pcr_collectors.py`
- Create: `services/orion-recall/app/collectors/active_packet.py`
- Create: `services/orion-recall/tests/test_pcr_belief.py`

- [ ] **Step 1: Write failing active_packet test**

```python
from unittest.mock import AsyncMock, patch
import pytest

from app.collectors.active_packet import fetch_active_packet_fragments


@pytest.mark.asyncio
async def test_active_packet_maps_crystallization_to_fragment():
    fake_packet = type("Pkt", (), {"items": [{"crystallization_id": "crys_abc", "summary": "Move solo", "salience": 0.9, "kind": "semantic", "bucket": "semantic"}]})()
    with patch("app.collectors.active_packet.retrieve_active_packet", new=AsyncMock(return_value=fake_packet)):
        frags = await fetch_active_packet_fragments(query=type("Q", (), {"session_id": "s1", "node_id": "n1", "retrieval_intent": "semantic"})(), pool=None, settings=type("S", (), {"recall_active_packet_enabled": True})())
    assert frags[0]["source"] == "active_packet"
    assert "Move solo" in frags[0]["snippet"]
```

- [ ] **Step 2: Implement collectors**

`pcr_collectors.py`:

```python
from orion.schemas.recall_pcr import RetrievalIntentV1

def collectors_for_intent(intent: RetrievalIntentV1) -> dict[str, bool]:
    plans = {
        "relational": {"cards": True, "rdf": True, "rdf_chat": True, "active_packet": True},
        "semantic": {"cards": True, "rdf": True, "rdf_chat": True, "active_packet": True},
        "procedural": {"cards": True, "active_packet": True},
        "open_loop": {"cards": True, "rdf_chat": True, "active_packet": True},
        "contradiction": {"cards": True, "rdf": True, "active_packet": True, "graphiti": True},
    }
    return plans.get(intent, {})
```

`active_packet.py` calls `orion.memory.crystallization.retriever.retrieve_active_packet` in-process.

- [ ] **Step 3: Run tests + commit**

```bash
pytest services/orion-recall/tests/test_pcr_belief.py -v
git add services/orion-recall/app/pcr_collectors.py services/orion-recall/app/collectors/active_packet.py services/orion-recall/tests/test_pcr_belief.py
git commit -m "feat(pcr): active_packet collector and intent backend plans"
```

---

## Task 12: Belief fusion `pcr_fuse_belief_candidates`

**Files:**
- Modify: `services/orion-recall/app/fusion.py`
- Modify: `services/orion-recall/app/worker.py`

- [ ] **Step 1: Write failing belief fusion test**

Assert `active_packet` ranks above cards; `low_info_social` always dropped; header contains `purposeful recall`.

- [ ] **Step 2: Implement `pcr_fuse_belief_candidates`**

```python
def pcr_fuse_belief_candidates(
    *,
    candidates: Iterable[Dict[str, Any]],
    profile: Dict[str, Any],
    retrieval_intent: str,
    query_text: str = "",
    latency_ms: int = 0,
) -> Tuple[MemoryBundleV1, List[Dict[str, Any]]]:
    filtered = [
        c for c in candidates
        if not _is_low_info_social_candidate(str(c.get("snippet") or c.get("text") or ""))
    ]
    # sort: active_packet salience desc, then cards, then rdf*
    ...
    header = f"[Durable beliefs — purposeful recall: {retrieval_intent}]\n"
```

Worker branch: `recall_phase == "purposeful"` → use collectors_for_intent + `pcr_fuse_belief_candidates`.

- [ ] **Step 3: Run recall tests + commit**

```bash
pytest services/orion-recall/tests/test_pcr_belief.py services/orion-recall/tests/test_chat_general_recall_salience.py -q
git commit -m "feat(pcr): belief fusion lane and purposeful worker branch"
```

---

## Task 13: Post-stance Phase 3 in cortex-exec

**Files:**
- Modify: `services/orion-cortex-exec/app/pcr_chat_memory.py`
- Modify: `services/orion-cortex-exec/app/router.py`
- Modify: `services/orion-cortex-exec/app/supervisor.py`

- [ ] **Step 1: Add `run_pcr_phase3`**

```python
PROFILE_FOR_INTENT = {
    "relational": "chat.belief.relational.v1",
    "semantic": "chat.belief.semantic.v1",
    "procedural": "chat.belief.procedural.v1",
    "open_loop": "chat.belief.open_loop.v1",
    "contradiction": "chat.belief.contradiction.v1",
}

async def run_pcr_phase3(bus, *, ctx, correlation_id, recall_cfg, settings, source, stance_brief, attention_frame):
    from orion.memory.retrieval_intent import derive_retrieval_intent
    skip_gate = ctx.get("pcr_skip_gate")  # or reconstruct from pcr_memory
    intent, rule_id = derive_retrieval_intent(...)
    if intent in ("none", "continuity"):
        return PcrChatMemoryV1(phase="continuity", retrieval_intent=intent, continuity_digest=ctx.get("continuity_digest", ""))
    profile = PROFILE_FOR_INTENT[intent]
    _, recall_debug, belief_digest = await run_recall_step(
        ...,
        recall_profile=profile,
        recall_phase="purposeful",
        retrieval_intent=intent,
        task_hints={"task_mode": stance_brief.get("task_mode"), "rule_id": rule_id},
    )
    continuity = ctx.get("continuity_digest", "")
    memory_digest = "\n\n".join(x for x in [continuity, belief_digest] if x)
    ctx["belief_digest"] = belief_digest
    ctx["memory_digest"] = memory_digest
    return PcrChatMemoryV1(...)
```

- [ ] **Step 2: Router hook after stance enforce**

After `enforce_chat_stance_quality` step completes (inspect step loop in `router.py`), call `run_pcr_phase3` when `settings.chat_pcr_post_stance_recall`.

- [ ] **Step 3: Supervisor parity**

In `supervisor.py` ~1644, replace duplicate `run_recall_step` with same PCR helpers.

- [ ] **Step 4: Integration test**

```python
async def test_pcr_semantic_runs_phase1_then_phase3(mock_bus):
    # assert two recall RPCs in order; second has retrieval_intent=semantic
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/pcr_chat_memory.py services/orion-cortex-exec/app/router.py services/orion-cortex-exec/app/supervisor.py services/orion-cortex-exec/tests/test_pcr_chat_memory.py
git commit -m "feat(pcr): post-stance phase 3 orchestration with supervisor parity"
```

---

## Task 14: Speech prompt contract + PCR telemetry

**Files:**
- Modify: `orion/cognition/prompts/chat_general.j2`
- Modify: `services/orion-recall/app/worker.py` (recall_debug.pcr block)

- [ ] **Step 1: Update chat_general.j2**

```jinja2
- continuity_digest: {{ continuity_digest | default("") }}
- belief_digest: {{ belief_digest | default("") }}
- memory_digest: {{ memory_digest | default("") }}
```

Replace stale-memory band-aid with belief/continuity instructions per spec.

- [ ] **Step 2: Attach `pcr` block to recall_debug**

When building `RecallDecisionV1`, set:

```python
decision.recall_debug["pcr"] = {
    "enabled": settings.recall_pcr_enabled,
    "phase": recall_phase,
    "retrieval_intent": retrieval_intent,
    "intent_rule_id": (q.task_hints or {}).get("rule_id"),
    "backend_plan": list(collectors_for_intent(retrieval_intent).keys()),
    ...
}
```

- [ ] **Step 3: Commit**

```bash
git add orion/cognition/prompts/chat_general.j2 services/orion-recall/app/worker.py
git commit -m "feat(pcr): speech prompt digests and recall telemetry block"
```

---

## Task 15: Legacy flag regression + agent-check

- [ ] **Step 1: Test legacy path**

```python
def test_chat_pcr_disabled_uses_legacy_pre_recall(monkeypatch):
    monkeypatch.setenv("CHAT_PCR_ENABLED", "false")
    # assert single run_recall_step before stance with chat.general.v1
```

- [ ] **Step 2: Run full gate suite**

```bash
make agent-check SERVICE=orion-cortex-exec
make agent-check SERVICE=orion-recall
pytest tests/test_recall_skip_gate.py tests/test_retrieval_intent.py tests/test_low_info_social.py -q
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test(pcr): legacy recall path regression"
```

---

## Task 16: Smoke script

**Files:**
- Create: `scripts/smoke_pcr_chat_memory_e2e.sh`

- [ ] **Step 1: Create smoke script**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== PCR smoke: greeting skip =="
# POST hub chat with "hey" -> grep trace for pcr.phase=skip

echo "== PCR smoke: semantic belief =="
# seed approved crystallization + topic message -> active_packet_refs non-empty

echo "PCR smoke OK"
```

Wire to existing hub chat curl patterns from `scripts/smoke_memory_crystallization_e2e.sh`.

- [ ] **Step 2: chmod + dry run**

```bash
chmod +x scripts/smoke_pcr_chat_memory_e2e.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_pcr_chat_memory_e2e.sh
git commit -m "chore(pcr): add e2e smoke script"
```

---

## Spec coverage self-check

| Spec requirement | Task |
|------------------|------|
| Phase 0 skip gate | 3, 7, 8 |
| Phase 1 continuity profile | 4, 5 |
| derive_retrieval_intent | 9 |
| Phase 3 purposeful recall | 10–13 |
| active_packet collector | 11 |
| Router + supervisor orchestration | 7, 8, 13 |
| Prompt contract | 8, 14 |
| RecallQueryV1 extensions | 2, 6 |
| PCR telemetry | 14 |
| Legacy CHAT_PCR_ENABLED=false | 15 |
| Smoke | 16 |
| Env parity | 5, 6, sync script |

**Open spec questions (operator defaults):** Quick lane phase 3 off (`CHAT_PCR_QUICK_PHASE3=false`). Contradiction v1 without Graphiti unless flag set. Verify spark_meta appraisal path in hub → exec ctx during Task 8 if phase 0 misses live turns.

---

**Plan complete.** Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks

**2. Inline Execution** — execute in this session via executing-plans with checkpoints

Which approach?
