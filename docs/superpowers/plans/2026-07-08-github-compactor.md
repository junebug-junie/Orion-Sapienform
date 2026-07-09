# GitHub Compactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Daily workflow `github_compactor_pass` fetches merged PR descriptions, compacts them into a bounded digest, supersedes one active `repo_dev_snapshot` memory card, and appends one journal entry.

**Architecture:** Extend the existing read-only GitHub PR skill to include truncated PR bodies; add pure digest/budget helpers in `orion/cognition/github_compactor/`; add a brain-lane LLM verb `github_compactor_digest_v1` (same pattern as `concept_induction_journal_synthesize`); implement `_execute_github_compactor_pass` in cortex-orch workflow runtime; register workflow + Hub catalogue. No new memory service, bus channel, or env keys.

**Tech Stack:** Python 3, Pydantic v2, asyncpg (memory cards via `RECALL_PG_DSN` on cortex-orch), pytest + pytest-asyncio, GitHub REST API (existing cortex-exec skill), Jinja2 prompt template.

**Design source:** `docs/superpowers/specs/2026-07-08-github-compactor-design.md`

---

## Context for the implementer (read before starting)

Work in a clean branch/worktree from repo root `/mnt/scripts/Orion-Sapienform`.

Verified facts:

- `skills.repo.github_recent_prs.v1` lives in `services/orion-cortex-exec/app/verb_adapters.py` — fetches merged PR metadata but **not** `body`.
- Workflows register in `orion/cognition/workflows/registry.py` and dispatch in `services/orion-cortex-orch/app/workflow_runtime.py`.
- Journal writes use `journal.entry.write.v1` via `_publish_journal_entry_write_or_fail`.
- Memory cards DAL: `orion/core/storage/memory_cards.py` (`insert_card`, `change_card_status`).
- Concept-induction journal synthesis pattern: orch sets verb `concept_induction_journal_synthesize` + metadata payload → cortex-exec brain lane + `.j2` prompt.
- **No** changes to `orion/bus/channels.yaml` or `orion/schemas/registry.py` in v1.
- **No** new env keys — reuse `GITHUB_TOKEN`, `ORION_ACTIONS_GITHUB_*`, `RECALL_PG_DSN`.

### Test commands

```bash
pytest orion/cognition/github_compactor/tests/test_digest.py -q
pytest tests/test_github_compactor_memory_cards.py -q
pytest services/orion-cortex-exec/tests/test_skill_verbs.py -k github -q
pytest services/orion-cortex-orch/tests/test_workflow_lane.py -k github_compactor -q
pytest services/orion-hub/tests/test_workflow_request_builder.py -k github -q
```

Full gate before PR:

```bash
pytest orion/cognition/github_compactor/tests -q
pytest tests/test_github_compactor_memory_cards.py -q
pytest services/orion-cortex-exec/tests/test_skill_verbs.py -q
pytest services/orion-cortex-orch/tests/test_workflow_lane.py -q
pytest services/orion-hub/tests/test_workflow_request_builder.py -q
```

---

## File structure

| File | Responsibility |
|------|----------------|
| `orion/schemas/actions/github_compactor.py` | `GithubCompactorDigestV1` output schema |
| `orion/schemas/actions/mesh_ops.py` | Add `body` to PR digest item |
| `orion/cognition/github_compactor/constants.py` | Slot key, char budgets |
| `orion/cognition/github_compactor/digest.py` | Budget validation, quiet-day stub, card/journal builders, stable entry id |
| `orion/cognition/github_compactor/tests/test_digest.py` | Pure unit tests (no PG, no LLM) |
| `orion/core/contracts/memory_cards.py` | Add `repo_compactor` provenance |
| `orion/core/storage/memory_cards.py` | Slot lookup + supersede-and-insert transaction |
| `tests/test_github_compactor_memory_cards.py` | DAL helper tests with mocked asyncpg pool |
| `services/orion-cortex-exec/app/verb_adapters.py` | PR `body` fetch + truncate |
| `orion/cognition/prompts/github_compactor_digest_v1.j2` | LLM compaction prompt |
| `orion/cognition/verbs/github_compactor_digest_v1.yaml` | Brain-lane verb registration |
| `services/orion-cortex-exec/app/router.py` | Allowlist verb name |
| `services/orion-cortex-orch/app/github_compactor_memory.py` | Thin orch wrapper: pool + supersede card |
| `services/orion-cortex-orch/app/workflow_runtime.py` | `_execute_github_compactor_pass` + dispatch branch |
| `orion/cognition/workflows/registry.py` | Register workflow |
| `services/orion-hub/templates/index.html` | Workflow dropdown entry |
| `services/orion-hub/static/js/app.js` | Workflow catalogue entry |
| `services/orion-actions/README.md` | Document workflow |

Build order: schemas/constants → PR body fetch → memory DAL → digest pure helpers → LLM verb → workflow executor → Hub/docs.

---

### Task 1: Schemas and constants

**Files:**
- Create: `orion/schemas/actions/github_compactor.py`
- Create: `orion/cognition/github_compactor/constants.py`
- Create: `orion/cognition/github_compactor/__init__.py`
- Modify: `orion/schemas/actions/mesh_ops.py`
- Modify: `orion/core/contracts/memory_cards.py`

- [ ] **Step 1: Write the failing digest schema test**

Create `orion/cognition/github_compactor/tests/test_digest.py` (first test only):

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.schemas.actions.github_compactor import GithubCompactorDigestV1


def test_github_compactor_digest_v1_rejects_empty_card_summary() -> None:
    with pytest.raises(ValidationError):
        GithubCompactorDigestV1(
            card_summary="",
            journal_title="Title",
            journal_body="Body",
            pr_refs=["#1"],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/cognition/github_compactor/tests/test_digest.py::test_github_compactor_digest_v1_rejects_empty_card_summary -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Add schema, constants, mesh_ops body field, provenance**

Create `orion/schemas/actions/github_compactor.py`:

```python
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GithubCompactorDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_summary: str = Field(..., min_length=1)
    journal_title: str = Field(..., min_length=1)
    journal_body: str = Field(..., min_length=1)
    pr_refs: List[str] = Field(default_factory=list)

    @field_validator("pr_refs")
    @classmethod
    def _normalize_pr_refs(cls, value: List[str]) -> List[str]:
        out = [str(item).strip() for item in value if str(item).strip()]
        return out
```

Create `orion/cognition/github_compactor/constants.py`:

```python
from __future__ import annotations

REPO_DEV_SNAPSHOT_SLOT = "repo_dev_snapshot"
REPO_DEV_SNAPSHOT_TAG = "repo_dev_snapshot"

PR_BODY_MAX_CHARS = 2000
CARD_SUMMARY_MAX_CHARS = 800
JOURNAL_TITLE_MAX_CHARS = 120
JOURNAL_BODY_MAX_CHARS = 4000

DEFAULT_LOOKBACK_DAYS = 1
```

Create `orion/cognition/github_compactor/__init__.py`:

```python
from orion.cognition.github_compactor.constants import REPO_DEV_SNAPSHOT_SLOT

__all__ = ["REPO_DEV_SNAPSHOT_SLOT"]
```

In `orion/schemas/actions/mesh_ops.py`, add to `RepoPullRequestDigestItemV1`:

```python
body: Optional[str] = None
```

In `orion/core/contracts/memory_cards.py`, extend `MemoryProvenance`:

```python
MemoryProvenance = Literal[
    "operator_highlight",
    "operator_distiller",
    "auto_extractor",
    "imported",
    "repo_compactor",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/cognition/github_compactor/tests/test_digest.py::test_github_compactor_digest_v1_rejects_empty_card_summary -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/actions/github_compactor.py orion/cognition/github_compactor/ orion/schemas/actions/mesh_ops.py orion/core/contracts/memory_cards.py
git commit -m "feat: add github compactor schema and constants"
```

---

### Task 2: Fetch PR body in GitHub skill

**Files:**
- Modify: `services/orion-cortex-exec/app/verb_adapters.py:1709-1747`
- Test: `services/orion-cortex-exec/tests/test_skill_verbs.py`

- [ ] **Step 1: Write the failing test**

Add to `services/orion-cortex-exec/tests/test_skill_verbs.py`:

```python
def test_github_recent_prs_includes_truncated_body(monkeypatch):
    sample_prs = [
        {
            "number": 42,
            "title": "Add compactor",
            "user": {"login": "juniper"},
            "state": "closed",
            "merged_at": "2026-07-08T12:00:00Z",
            "created_at": "2026-07-07T12:00:00Z",
            "updated_at": "2026-07-08T12:00:00Z",
            "labels": [],
            "base": {"ref": "main"},
            "head": {"ref": "feat/compactor"},
            "html_url": "https://github.com/acme/widgets/pull/42",
            "changed_files": 3,
            "body": "x" * 2500,
        }
    ]

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls = {"n": 0}

    def _urlopen(request, timeout=0):
        calls["n"] += 1
        url = request.full_url
        if url.endswith("/pulls?state=closed&sort=updated&direction=desc&per_page=20"):
            return _Resp(sample_prs)
        if url.endswith("/files?per_page=100"):
            return _Resp([{"filename": "services/orion-hub/app/main.py"}])
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(verb_adapters, "urlopen", _urlopen)
    monkeypatch.setattr(verb_adapters.settings, "github_owner", "acme")
    monkeypatch.setattr(verb_adapters.settings, "github_repo", "widgets")
    monkeypatch.setattr(verb_adapters.settings, "github_token", "test-token")
    monkeypatch.setattr(verb_adapters.settings, "mesh_default_lookback_days", 7)

    req = _plan_request("skills.repo.github_recent_prs.v1", skill_args={"lookback_days": 7})
    ctx = VerbContext(meta={"correlation_id": str(uuid4())})
    out, _ = asyncio.run(verb_adapters.GithubRecentPullRequestsVerb().execute(ctx, req))
    data = json.loads(out.final_text)
    assert data["available"] is True
    assert len(data["items"]) == 1
    assert data["items"][0]["body"] == ("x" * 2000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-cortex-exec/tests/test_skill_verbs.py::test_github_recent_prs_includes_truncated_body -v`

Expected: FAIL (`KeyError: 'body'` or `assert None == ...`).

- [ ] **Step 3: Implement body capture + truncation**

Near the top of `verb_adapters.py` (with other helpers), add:

```python
def _truncate_pr_body(body: object, *, max_chars: int = 2000) -> str | None:
    text = str(body or "").strip()
    if not text:
        return None
    return text[:max_chars]
```

In the PR item dict inside `GithubRecentPullRequestsVerb.execute`, add:

```python
"body": _truncate_pr_body(pr.get("body")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-cortex-exec/tests/test_skill_verbs.py::test_github_recent_prs_includes_truncated_body -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/verb_adapters.py services/orion-cortex-exec/tests/test_skill_verbs.py
git commit -m "feat: include truncated PR body in github recent prs skill"
```

---

### Task 3: Memory card compactor-slot helpers

**Files:**
- Modify: `orion/core/storage/memory_cards.py`
- Create: `tests/test_github_compactor_memory_cards.py`

- [ ] **Step 1: Write the failing DAL test**

Create `tests/test_github_compactor_memory_cards.py`:

```python
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.core.storage import memory_cards as mc_dal
from orion.cognition.github_compactor.constants import REPO_DEV_SNAPSHOT_SLOT


@pytest.mark.asyncio
async def test_supersede_and_insert_compactor_card_supersedes_existing() -> None:
    old_id = uuid4()
    new_id = uuid4()

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        side_effect=[
            {"card_id": old_id},  # FOR UPDATE lookup
            {"card_id": new_id, "slug": "recent-repo-development", "types": ["fact"], "anchor_class": "project",
             "status": "active", "confidence": "likely", "sensitivity": "private", "priority": "high_recall",
             "visibility_scope": ["chat"], "time_horizon": None, "provenance": "repo_compactor",
             "trust_source": None, "project": "acme/widgets", "title": "Recent repo development",
             "summary": "Built compactor.", "still_true": None, "anchors": None, "tags": ["repo_dev_snapshot"],
             "evidence": [], "subschema": {"compactor_slot": REPO_DEV_SNAPSHOT_SLOT},
             "created_at": "2026-07-08T12:00:00+00:00", "updated_at": "2026-07-08T12:00:00+00:00"},
        ]
    )
    conn.execute = AsyncMock()
    conn.transaction = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

    pool = AsyncMock()
    pool.acquire = MagicMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = acquire_cm

    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="project",
        status="active",
        priority="high_recall",
        provenance="repo_compactor",
        project="acme/widgets",
        title="Recent repo development",
        summary="Built compactor.",
        tags=[REPO_DEV_SNAPSHOT_SLOT],
        subschema={"compactor_slot": REPO_DEV_SNAPSHOT_SLOT},
    )

    # Patch _insert_card_on_conn to avoid full SQL in this unit test.
    async def _fake_insert(_conn, _card, *, actor, op="create"):
        return new_id

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mc_dal, "_insert_card_on_conn", _fake_insert)
    try:
        result_id = await mc_dal.supersede_and_insert_compactor_card(
            pool,
            slot=REPO_DEV_SNAPSHOT_SLOT,
            card=card,
            actor="github_compactor_pass",
        )
    finally:
        monkeypatch.undo()

    assert result_id == new_id
    assert conn.execute.await_count >= 1
    executed_sql = conn.execute.await_args_list[0].args[0]
    assert "status = 'superseded'" in executed_sql.lower() or "superseded" in executed_sql.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_github_compactor_memory_cards.py::test_supersede_and_insert_compactor_card_supersedes_existing -v`

Expected: FAIL with `AttributeError: module ... has no attribute 'supersede_and_insert_compactor_card'`.

- [ ] **Step 3: Implement helpers in memory_cards.py**

Add after `card_exists_by_fingerprint`:

```python
async def find_active_card_by_compactor_slot(pool: asyncpg.Pool, slot: str) -> Optional[MemoryCardV1]:
    key = (slot or "").strip()
    if not key:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM memory_cards
            WHERE status = 'active'
              AND subschema ->> 'compactor_slot' = $1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            key,
        )
        return _row_to_card(row) if row else None


async def supersede_and_insert_compactor_card(
    pool: asyncpg.Pool,
    *,
    slot: str,
    card: MemoryCardCreateV1,
    actor: str,
) -> UUID:
    key = (slot or "").strip()
    if not key:
        raise ValueError("compactor slot required")
    async with pool.acquire() as conn:
        async with conn.transaction():
            existing = await conn.fetchrow(
                """
                SELECT card_id FROM memory_cards
                WHERE status = 'active'
                  AND subschema ->> 'compactor_slot' = $1
                FOR UPDATE
                """,
                key,
            )
            if existing is not None:
                await conn.execute(
                    "UPDATE memory_cards SET status = 'superseded', updated_at = now() WHERE card_id = $1",
                    existing["card_id"],
                )
                before = await conn.fetchrow(
                    "SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1",
                    existing["card_id"],
                )
                after = await conn.fetchrow(
                    "SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1",
                    existing["card_id"],
                )
                await conn.execute(
                    """
                    INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                    VALUES ($1, NULL, 'status_change', $2, $3::jsonb, $4::jsonb)
                    """,
                    existing["card_id"],
                    actor,
                    json.dumps({"card": before["j"]}),
                    json.dumps({"card": after["j"], "reason": "compactor_supersede"}),
                )
            return await _insert_card_on_conn(conn, card, actor=actor, op="create")
```

Export from `orion/core/storage/__init__.py` if that module re-exports DAL helpers (match existing pattern for `insert_card`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_github_compactor_memory_cards.py::test_supersede_and_insert_compactor_card_supersedes_existing -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/core/storage/memory_cards.py orion/core/storage/__init__.py tests/test_github_compactor_memory_cards.py
git commit -m "feat: add compactor-slot memory card supersede helpers"
```

---

### Task 4: Pure digest helpers (budgets, quiet day, stable journal id)

**Files:**
- Create: `orion/cognition/github_compactor/digest.py`
- Test: `orion/cognition/github_compactor/tests/test_digest.py`

- [ ] **Step 1: Write failing tests**

Append to `orion/cognition/github_compactor/tests/test_digest.py`:

```python
from uuid import UUID

from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.github_compactor.digest import (
    assert_digest_within_budget,
    build_quiet_day_digest,
    parse_github_compactor_digest_json,
    stable_github_compactor_journal_entry_id,
)
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1


def test_assert_digest_within_budget_accepts_in_limit() -> None:
    digest = GithubCompactorDigestV1(
        card_summary="a" * CARD_SUMMARY_MAX_CHARS,
        journal_title="b" * JOURNAL_TITLE_MAX_CHARS,
        journal_body="c" * JOURNAL_BODY_MAX_CHARS,
        pr_refs=["#1"],
    )
    assert_digest_within_budget(digest)


def test_assert_digest_within_budget_rejects_over_limit() -> None:
    digest = GithubCompactorDigestV1(
        card_summary="a" * (CARD_SUMMARY_MAX_CHARS + 1),
        journal_title="title",
        journal_body="body",
        pr_refs=[],
    )
    with pytest.raises(ValueError, match="compactor_output_over_budget"):
        assert_digest_within_budget(digest)


def test_build_quiet_day_digest() -> None:
    digest = build_quiet_day_digest(repo="acme/widgets", window_label="2026-07-08")
    assert "No merges" in digest.journal_body
    assert digest.pr_refs == []


def test_stable_github_compactor_journal_entry_id_is_deterministic() -> None:
    a = stable_github_compactor_journal_entry_id(
        workflow_id="github_compactor_pass",
        calendar_date="2026-07-08",
        repo="acme/widgets",
    )
    b = stable_github_compactor_journal_entry_id(
        workflow_id="github_compactor_pass",
        calendar_date="2026-07-08",
        repo="acme/widgets",
    )
    assert a == b
    UUID(a)


def test_parse_github_compactor_digest_json() -> None:
    raw = '{"card_summary":"Card","journal_title":"Title","journal_body":"Body","pr_refs":["#9"]}'
    digest = parse_github_compactor_digest_json(raw)
    assert digest.card_summary == "Card"
    assert digest.pr_refs == ["#9"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest orion/cognition/github_compactor/tests/test_digest.py -v`

Expected: FAIL with `ModuleNotFoundError: orion.cognition.github_compactor.digest`.

- [ ] **Step 3: Implement digest.py**

Create `orion/cognition/github_compactor/digest.py`:

```python
from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5

from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1

_COMPACTOR_JOURNAL_ENTRY_NS = NAMESPACE_URL


def assert_digest_within_budget(digest: GithubCompactorDigestV1) -> None:
    if len(digest.card_summary) > CARD_SUMMARY_MAX_CHARS:
        raise ValueError("compactor_output_over_budget:card_summary")
    if len(digest.journal_title) > JOURNAL_TITLE_MAX_CHARS:
        raise ValueError("compactor_output_over_budget:journal_title")
    if len(digest.journal_body) > JOURNAL_BODY_MAX_CHARS:
        raise ValueError("compactor_output_over_budget:journal_body")


def build_quiet_day_digest(*, repo: str, window_label: str) -> GithubCompactorDigestV1:
    repo_label = (repo or "unknown repo").strip()
    return GithubCompactorDigestV1(
        card_summary=f"No merged PRs in {window_label} for {repo_label}.",
        journal_title=f"Repo development digest — {window_label}",
        journal_body=(
            f"No merged pull requests were found for {repo_label} during {window_label}. "
            "Previous repo development snapshot card was left unchanged."
        ),
        pr_refs=[],
    )


def parse_github_compactor_digest_json(raw: str) -> GithubCompactorDigestV1:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("compactor_digest_not_object")
    return GithubCompactorDigestV1.model_validate(payload)


def stable_github_compactor_journal_entry_id(
    *,
    workflow_id: str,
    calendar_date: str,
    repo: str,
) -> str:
    payload = "|".join([workflow_id.strip(), calendar_date.strip(), repo.strip()])
    return str(uuid5(_COMPACTOR_JOURNAL_ENTRY_NS, payload))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest orion/cognition/github_compactor/tests/test_digest.py -q`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/github_compactor/digest.py orion/cognition/github_compactor/tests/test_digest.py
git commit -m "feat: add github compactor digest budget helpers"
```

---

### Task 5: Brain-lane LLM verb `github_compactor_digest_v1`

**Files:**
- Create: `orion/cognition/prompts/github_compactor_digest_v1.j2`
- Create: `orion/cognition/verbs/github_compactor_digest_v1.yaml`
- Modify: `services/orion-cortex-exec/app/router.py`

- [ ] **Step 1: Write failing router allowlist test**

Add to `services/orion-cortex-exec/tests/test_router_final_text_assembly.py` (or create `services/orion-cortex-exec/tests/test_github_compactor_router.py`):

```python
def test_router_allows_github_compactor_digest_v1():
    from app.router import _brain_lane_verbs

    assert "github_compactor_digest_v1" in _brain_lane_verbs()
```

Adjust import/name to match actual router helper in the codebase (grep `concept_induction_journal_synthesize` in router tests and mirror exactly).

- [ ] **Step 2: Run test to verify it fails**

Run the new test with `-v`. Expected: FAIL — verb not allowlisted.

- [ ] **Step 3: Add prompt, verb yaml, router entry**

Create `orion/cognition/prompts/github_compactor_digest_v1.j2`:

```jinja2
You compact merged GitHub pull request activity into Orion's internal repo-development digest.

Input JSON is in metadata key `github_compactor_input`. It contains:
- repo: string
- lookback_days: integer
- items: list of merged PR objects (number, title, body, touched_paths, inferred_services, merged_at, url)

Return ONLY valid JSON with keys:
- card_summary (string, max {{ card_summary_max }} chars): compact gut-sense of what was built
- journal_title (string, max {{ journal_title_max }} chars)
- journal_body (string, max {{ journal_body_max }} chars): richer narrative with PR refs and service groupings
- pr_refs (array of strings like "#123")

Rules:
- Use PR descriptions when present; fall back to title + touched paths when body is empty.
- Do not invent PRs not present in input.
- Prefer themes ("hub UI polish", "substrate wiring") over file lists.
- No markdown fences.

Input:
{{ github_compactor_input | tojson }}
```

Create `orion/cognition/verbs/github_compactor_digest_v1.yaml`:

```yaml
name: github_compactor_digest_v1
label: GitHub Compactor Digest
description: Compact merged PR descriptions into bounded card + journal digest JSON.
category: Generative
priority: high
requires_gpu: true
requires_memory: false
timeout_ms: 90000
services:
  - LLMGatewayService
prompt_template: github_compactor_digest_v1.j2
steps:
  - name: github_compactor_digest_v1
    order: 0
    services: [LLMGatewayService]
    prompt_template: github_compactor_digest_v1.j2
```

Add `"github_compactor_digest_v1"` beside `"concept_induction_journal_synthesize"` in `services/orion-cortex-exec/app/router.py` brain-lane allowlist.

- [ ] **Step 4: Run router test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/prompts/github_compactor_digest_v1.j2 orion/cognition/verbs/github_compactor_digest_v1.yaml services/orion-cortex-exec/app/router.py services/orion-cortex-exec/tests/
git commit -m "feat: add github compactor digest brain-lane verb"
```

---

### Task 6: Workflow executor in cortex-orch

**Files:**
- Create: `services/orion-cortex-orch/app/github_compactor_memory.py`
- Modify: `services/orion-cortex-orch/app/workflow_runtime.py`
- Modify: `orion/cognition/workflows/registry.py`

- [ ] **Step 1: Write failing workflow lane test**

Add to `services/orion-cortex-orch/tests/test_workflow_lane.py`:

```python
def test_github_compactor_pass_writes_journal_and_supersedes_card(monkeypatch) -> None:
    bus = DummyBus()
    card_calls: dict = {}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.repo.github_recent_prs.v1":
            payload = {
                "available": True,
                "repo": "acme/widgets",
                "lookback_days": 1,
                "merged_pr_count": 1,
                "items": [
                    {
                        "number": 9,
                        "title": "Add compactor",
                        "body": "Compacts PR descriptions.",
                        "merged_at": "2026-07-08T10:00:00Z",
                        "touched_paths": ["services/orion-hub/app/main.py"],
                        "inferred_services": ["orion-hub"],
                        "url": "https://github.com/acme/widgets/pull/9",
                    }
                ],
            }
            return DummyVerbResult(payload={"result": {"status": "success", "final_text": json.dumps(payload)}})
        if req.verb == "github_compactor_digest_v1":
            digest = {
                "card_summary": "Added GitHub compactor workflow.",
                "journal_title": "Repo development — 2026-07-08",
                "journal_body": "Merged #9 added GitHub compactor.",
                "pr_refs": ["#9"],
            }
            ft = json.dumps({"mode": "manual", "title": digest["journal_title"], "body": digest["journal_body"]})
            return DummyVerbResult(payload={"result": {"status": "success", "final_text": ft, "metadata": {"github_compactor_digest": digest}}})
        raise AssertionError(f"unexpected verb {req.verb}")

    async def _fake_persist_card(**kwargs):
        card_calls.update(kwargs)
        return "00000000-0000-0000-0000-000000000099"

    monkeypatch.setattr(
        "app.workflow_runtime.persist_github_compactor_memory_card",
        _fake_persist_card,
    )

    result = asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name="cortex-orch"),
            req=_req("github_compactor_pass"),
            correlation_id="00000000-0000-0000-0000-000000000099",
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )

    assert result.ok is True
    assert result.metadata["workflow"]["workflow_id"] == "github_compactor_pass"
    assert result.metadata["workflow"]["merged_pr_count"] == 1
    assert card_calls.get("digest") is not None
    assert any(ch == "orion:journal:write" for ch, _ in bus.published)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-cortex-orch/tests/test_workflow_lane.py::test_github_compactor_pass_writes_journal_and_supersedes_card -v`

Expected: FAIL (`unknown_workflow` or missing executor).

- [ ] **Step 3: Register workflow**

Add to `orion/cognition/workflows/registry.py` `_WORKFLOWS` tuple:

```python
WorkflowDefinition(
    workflow_id="github_compactor_pass",
    display_name="GitHub Compactor",
    description="Daily compaction of merged PR descriptions into a repo development digest (memory card + journal).",
    aliases=[
        "run github compactor",
        "compact recent prs",
        "github compactor pass",
        "what have we been building",
    ],
    user_invocable=True,
    autonomous_invocable=True,
    execution_mode="sync",
    may_call_actions=False,
    persistence_policy="Supersede one active memory card (repo_dev_snapshot slot) and append one journal entry per run.",
    result_surface="Return PR count, card summary preview, and journal entry id.",
    steps=[
        WorkflowStepDefinition(step_id="fetch_prs", description="Fetch merged PRs with descriptions.", adapter="verb:skills.repo.github_recent_prs.v1"),
        WorkflowStepDefinition(step_id="compact", description="LLM compact digest JSON.", adapter="verb:github_compactor_digest_v1"),
        WorkflowStepDefinition(step_id="persist_card", description="Supersede repo_dev_snapshot card.", adapter="memory_cards:compactor_slot"),
        WorkflowStepDefinition(step_id="persist_journal", description="Append journal entry.", adapter="journaler:append_only_write"),
    ],
    planner_hints=["Use when the user asks to compact recent GitHub PR activity into repo development memory."],
),
```

- [ ] **Step 4: Add memory persist wrapper**

Create `services/orion-cortex-orch/app/github_compactor_memory.py`:

```python
from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from orion.cognition.github_compactor.constants import REPO_DEV_SNAPSHOT_SLOT, REPO_DEV_SNAPSHOT_TAG
from orion.core.contracts.memory_cards import MemoryCardCreateV1, EvidenceItemV1
from orion.core.storage import memory_cards as mc_dal
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1

from .memory_extractor import _get_memory_pool

logger = logging.getLogger("orion.cortex.github_compactor_memory")


async def persist_github_compactor_memory_card(
    *,
    digest: GithubCompactorDigestV1,
    repo: str,
    window_label: str,
    merged_pr_count: int,
    actor: str = "github_compactor_pass",
) -> Optional[UUID]:
    pool = await _get_memory_pool()
    if pool is None:
        raise RuntimeError("recall_pg_dsn_unavailable")

    evidence = [
        EvidenceItemV1(source=ref, ts=window_label)
        for ref in (digest.pr_refs or [])[:12]
    ]
    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="project",
        status="active",
        priority="high_recall",
        provenance="repo_compactor",
        project=repo,
        title=f"Recent repo development ({window_label})",
        summary=digest.card_summary,
        tags=[REPO_DEV_SNAPSHOT_TAG],
        evidence=evidence,
        subschema={
            "compactor_slot": REPO_DEV_SNAPSHOT_SLOT,
            "window_label": window_label,
            "merged_pr_count": merged_pr_count,
            "source_repo": repo,
        },
    )
    return await mc_dal.supersede_and_insert_compactor_card(
        pool,
        slot=REPO_DEV_SNAPSHOT_SLOT,
        card=card,
        actor=actor,
    )
```

- [ ] **Step 5: Implement `_execute_github_compactor_pass` and dispatch branch**

In `workflow_runtime.py`:

1. Import digest helpers + `persist_github_compactor_memory_card`.
2. Add helper `_run_github_compactor_digest(...)` mirroring `_run_concept_induction_grounded_journal_synthesis` but:
   - verb = `github_compactor_digest_v1`
   - metadata key `github_compactor_input` = fetch result dict
   - parse JSON from `metadata["github_compactor_digest"]` if present, else parse `final_text` via `parse_github_compactor_digest_json`
   - call `assert_digest_within_budget`
3. Add `_execute_github_compactor_pass`:
   - Resolve `lookback_days` from workflow policy metadata, default `DEFAULT_LOOKBACK_DAYS` (1)
   - Call fetch verb `skills.repo.github_recent_prs.v1`
   - If not `available`: raise `WorkflowExecutionError("github_repo_not_configured")`
   - If `merged_pr_count == 0`: use `build_quiet_day_digest`; **skip** `persist_github_compactor_memory_card`; still write journal
   - Else: run digest verb + persist card
   - Build journal draft from digest; use `stable_github_compactor_journal_entry_id(workflow_id, calendar_date, repo)` for entry id (override in `build_write_payload` path — set `write.entry_id` explicitly after build)
   - Publish journal via `_publish_journal_entry_write_or_fail`
   - Return workflow metadata with `merged_pr_count`, `card_id`, `journal_entry`, `card_summary_preview`
4. Add dispatch branch:

```python
elif workflow_id == "github_compactor_pass":
    result = await _execute_github_compactor_pass(...)
```

- [ ] **Step 6: Run workflow test**

Run: `pytest services/orion-cortex-orch/tests/test_workflow_lane.py::test_github_compactor_pass_writes_journal_and_supersedes_card -v`

Expected: PASS

- [ ] **Step 7: Add quiet-day + missing-config tests**

Add:

```python
def test_github_compactor_pass_quiet_day_skips_card_persist(monkeypatch) -> None:
    ...

def test_github_compactor_pass_missing_github_config_fails(monkeypatch) -> None:
    ...
```

Implement until green.

- [ ] **Step 8: Commit**

```bash
git add orion/cognition/workflows/registry.py services/orion-cortex-orch/app/workflow_runtime.py services/orion-cortex-orch/app/github_compactor_memory.py services/orion-cortex-orch/tests/test_workflow_lane.py
git commit -m "feat: add github_compactor_pass workflow executor"
```

---

### Task 7: Hub catalogue and actions docs

**Files:**
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-hub/static/js/app.js`
- Modify: `services/orion-actions/README.md`
- Modify: `services/orion-hub/tests/test_workflow_request_builder.py`

- [ ] **Step 1: Write failing workflow resolver test**

Add to `services/orion-hub/tests/test_workflow_request_builder.py`:

```python
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Run github compactor", "github_compactor_pass"),
        ("Compact recent prs", "github_compactor_pass"),
    ],
)
def test_resolve_github_compactor_pass(prompt, expected):
    from orion.cognition.workflows.registry import resolve_user_workflow_invocation

    match = resolve_user_workflow_invocation(prompt)
    assert match is not None
    assert match.workflow_id == expected
```

- [ ] **Step 2: Run test — expect PASS once registry landed** (FAIL only if Task 6 not done)

- [ ] **Step 3: Hub UI entries**

In `services/orion-hub/templates/index.html`, add workflow `<option>` near other workflows:

```html
<option value="Run github compactor." data-workflow-id="github_compactor_pass">Workflow · Run github compactor.</option>
```

In `services/orion-hub/static/js/app.js`, add catalogue item:

```javascript
{ workflowId: 'github_compactor_pass', prompt: 'Run github compactor.', label: 'Workflow · Run github compactor.' },
```

- [ ] **Step 4: Document in actions README**

Add section `### github_compactor_pass` mirroring `journal_pass` / `concept_induction_pass` format:

- Purpose: compact merged PR descriptions into repo development digest
- Schedulable: yes
- Notify: yes (via schedule policy)
- Persisted meaning: superseding `repo_dev_snapshot` memory card + append-only journal entry

- [ ] **Step 5: Run hub tests**

Run: `pytest services/orion-hub/tests/test_workflow_request_builder.py -k github -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/templates/index.html services/orion-hub/static/js/app.js services/orion-actions/README.md services/orion-hub/tests/test_workflow_request_builder.py
git commit -m "docs: register github compactor workflow in hub and actions"
```

---

### Task 8: Final verification and operator smoke

- [ ] **Step 1: Run full test gate**

```bash
pytest orion/cognition/github_compactor/tests -q
pytest tests/test_github_compactor_memory_cards.py -q
pytest services/orion-cortex-exec/tests/test_skill_verbs.py -k github -q
pytest services/orion-cortex-orch/tests/test_workflow_lane.py -k github_compactor -q
pytest services/orion-hub/tests/test_workflow_request_builder.py -k github -q
```

Expected: all PASS

- [ ] **Step 2: Manual smoke (operator)**

Prerequisites: `GITHUB_TOKEN`, `ORION_ACTIONS_GITHUB_OWNER`, `ORION_ACTIONS_GITHUB_REPO`, `RECALL_PG_DSN` on cortex-orch.

1. Hub → trigger workflow prompt: `Run github compactor.`
2. Verify response mentions PR count + journal entry id.
3. Hub Memory → filter tag `repo_dev_snapshot` → one active card.
4. Hub Journal → entry for today with PR refs.
5. Schedule daily cadence in Hub workflow schedule UI.

- [ ] **Step 3: Final commit if any fixups**

```bash
git status --short
git commit -m "test: github compactor verification fixups"  # only if needed
```

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| Fetch PR bodies (truncated) | Task 2 |
| `GithubCompactorDigestV1` + budgets | Tasks 1, 4 |
| `repo_dev_snapshot` card supersession | Tasks 3, 6 |
| Append-only journal + stable entry id | Tasks 4, 6 |
| `github_compactor_pass` workflow + schedule | Tasks 6, 7 |
| Quiet day behavior | Task 6 tests |
| GitHub unavailable fails loud | Task 6 tests |
| Hub catalogue | Task 7 |
| No bus/registry/env changes | respected |
| Concept induction passive (journal lineage) | no task (by design) |

## Placeholder scan

No TBD/TODO/Similar-to-task placeholders included.

## Type/name consistency

- Slot constant: `REPO_DEV_SNAPSHOT_SLOT` everywhere
- Verb names: `skills.repo.github_recent_prs.v1`, `github_compactor_digest_v1`
- Workflow id: `github_compactor_pass`
- Provenance: `repo_compactor`
- Error token: `compactor_output_over_budget`

---

## Restart required (after implementation)

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build orion-cortex-exec

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml \
  up -d --build orion-cortex-orch

docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

No orion-actions restart unless workflow registration caching is added later (not in v1).
