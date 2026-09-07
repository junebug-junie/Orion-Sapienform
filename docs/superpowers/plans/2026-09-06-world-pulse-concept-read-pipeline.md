# World-Pulse → Concept Atlas Read Pipeline (Stage 1 vertical slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a thin Hub sibling loop that dequeues world-pulse seeds (findings first), spends a **separate Wallet A** (~6/day), runs Stage 1 heavy read → typed handoff → Concept Atlas write with locked provenance, and journals a trace — without ever debiting Curiosity Atlas.

**Architecture:** Mirror `CuriosityInvestigation` lifecycle (tick / Redis daily cap + cooldown / `execute_unified_turn` with injectable Stage 1) but keep **new Redis key prefixes** and a **Postgres seed queue** on Hub's `memory_pg_pool`. Concept nodes come from a pure substrate adapter (`make_provenance`) then `SubstrateGraphMaterializer` on Hub's semantic store. Stage 2, Wallet B, hop loop-back, and operator backfill are **out of scope** (follow-on plans).

**Tech Stack:** Python 3.11, Pydantic v2, asyncpg, Redis (Hub bus), Falkor substrate via Hub, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-world-pulse-concept-read-pipeline-design.md`

## Global Constraints

- Curiosity Atlas budget is **never charged** — do not call `CuriosityInvestigation._record_investigation`, and never read/write `orion:curiosity:last_investigation_at`, `orion:curiosity:count:*`, or `orion:curiosity:last_run_id`.
- Wallet A default daily cap: **6**; Redis prefixes: `orion:wp_read:wallet_a:last_at`, `orion:wp_read:wallet_a:count:`.
- Seed order: **findings first**, then digest items.
- Provenance locked: `producer="world_pulse_read_pipeline"`; Stage 1 `source_kind="world_pulse.read"`; `evidence_refs` must include article URL + `run_id` + `trace_id`.
- Durable seed queue: **Postgres** table `world_pulse_read_seed` (Hub `RECALL_PG_DSN` / `memory_pg_pool`), not Redis.
- Stable `seed_id` for idempotent enqueue (exact formula in Task 2).
- No keyword-triggered conversational mode changes; no dumping raw scrape bodies into Falkor — only distilled handoff candidates.
- Sibling Hub loop file-bounded; do not merge into `curiosity_investigation.py`.
- Env parity: any new Hub key updates `.env_example`, `settings.py`, and local `.env` via `python scripts/sync_local_env_from_example.py`.
- Work only in a git worktree (not shared checkout); branch pattern `feat/world-pulse-concept-read-v1`.

### Deferred to follow-on plans (do not implement here)

- Stage 2 seeded curiosity pass + Wallet B (~6/day) + Stage 2→Stage 1 re-entry.
- Tool round-trip safety ceiling per `trace_id` (sketch for later: **5**).
- Operator backfill script for ~260 findings / ~636 items.
- Rich Hub status page (this plan ships a minimal JSON schedule endpoint only).

---

## File Structure

**New files:**
- `orion/schemas/world_pulse_read.py` — `WorldPulseReadSeedV1`, `WorldPulseReadHandoffV1`, concept/prior candidate models.
- `orion/world_pulse_read/__init__.py` — package marker.
- `orion/world_pulse_read/seeds.py` — pure `make_seed_id`, `seeds_from_digest_payload`.
- `orion/world_pulse_read/queue.py` — asyncpg enqueue/claim/complete helpers + DDL ensure.
- `orion/world_pulse_read/wallet_a.py` — Redis Wallet A gate + debit (isolated keys).
- `orion/substrate/adapters/world_pulse_read.py` — handoff → `SubstrateGraphRecordV1`.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py` — Hub loop (`WorldPulseReadPipeline`).
- `services/orion-hub/scripts/world_pulse_read_routes.py` — minimal `GET .../api/schedule`.
- `services/orion-sql-db/manual_migration_world_pulse_read_seed_queue_v1.sql`
- Tests: `tests/test_world_pulse_read_schemas.py`, `tests/test_world_pulse_read_seeds.py`, `tests/test_world_pulse_read_adapter.py`, `services/orion-hub/tests/test_world_pulse_read_wallet_a.py`, `services/orion-hub/tests/test_world_pulse_read_pipeline.py`

**Modified files:**
- `orion/schemas/registry.py` — register new models.
- `services/orion-hub/app/settings.py` + `.env_example` — Wallet A / enable / tick keys.
- `services/orion-hub/scripts/main.py` — construct + start loop; include router.
- Spec status line optional: mark design “Stage 1 plan ready” only if you touch it (not required).

---

### Task 0: Worktree / branch for implementation

**Files:** none (git only)

- [ ] **Step 1: Start from a clean worktree**

If implementing from the design worktree after this plan is approved, create a sibling feat worktree from updated `main` (or from this docs branch only if the design+plan are already merged):

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add ../Orion-Sapienform-world-pulse-concept-read-v1 -b feat/world-pulse-concept-read-v1 origin/main
cd ../Orion-Sapienform-world-pulse-concept-read-v1
# If design+plan landed on a docs PR first, cherry-pick or rebase that commit onto the feat branch before coding.
```

Expected: clean tree on `feat/world-pulse-concept-read-v1`. All later paths are relative to that worktree.

---

### Task 1: Handoff + seed schemas

**Files:**
- Create: `orion/schemas/world_pulse_read.py`
- Modify: `orion/schemas/registry.py` (import + `_MODELS` / registry dict entries matching existing world_pulse registrations)
- Test: `tests/test_world_pulse_read_schemas.py`

**Interfaces:**
- Produces: `WorldPulseReadSeedV1`, `WorldPulseReadConceptCandidateV1`, `WorldPulseReadPriorCandidateV1`, `WorldPulseReadHandoffV1`

- [ ] **Step 1: Write the failing test**

Create `tests/test_world_pulse_read_schemas.py`:

```python
from datetime import datetime, timezone

from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadPriorCandidateV1,
    WorldPulseReadSeedV1,
)


def test_seed_round_trip():
    seed = WorldPulseReadSeedV1(
        seed_id="finding:run-1:abcd",
        kind="finding",
        run_id="run-1",
        url="https://example.com/a",
        title="A",
        section="ai_technology",
    )
    assert WorldPulseReadSeedV1.model_validate(seed.model_dump()).seed_id == seed.seed_id


def test_handoff_requires_seed_ref_and_trace():
    handoff = WorldPulseReadHandoffV1(
        seed_ref=WorldPulseReadSeedV1(
            seed_id="finding:run-1:abcd",
            kind="finding",
            run_id="run-1",
            url="https://example.com/a",
            title="A",
            section="ai_technology",
        ),
        what_i_learned="Chip fab news.",
        candidate_priors=[
            WorldPulseReadPriorCandidateV1(claim="TSMC capacity is tight", confidence=0.6)
        ],
        concept_candidates=[
            WorldPulseReadConceptCandidateV1(label="advanced packaging", definition="chip packaging")
        ],
        open_threads=["Who supplies the lasers?"],
        trace_id="tr-1",
        created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
    )
    re = WorldPulseReadHandoffV1.model_validate(handoff.model_dump(mode="json"))
    assert re.producer_hint == "world_pulse_read_pipeline"
    assert re.concept_candidates[0].label == "advanced packaging"
    assert re.trace_id == "tr-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_world_pulse_read_schemas.py -v`  
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `orion.schemas.world_pulse_read`.

- [ ] **Step 3: Implement schemas + registry**

Create `orion/schemas/world_pulse_read.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorldPulseReadSeedV1(_Base):
    seed_id: str = Field(min_length=1)
    kind: Literal["finding", "digest_item"]
    run_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = ""
    section: str = ""
    item_id: str | None = None  # digest_item only


class WorldPulseReadConceptCandidateV1(_Base):
    label: str = Field(min_length=1)
    definition: str | None = None
    link_hints: list[str] = Field(default_factory=list)


class WorldPulseReadPriorCandidateV1(_Base):
    claim: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class WorldPulseReadHandoffV1(_Base):
    """Stage 1 → Stage 2 (and Concept Atlas) artifact."""

    seed_ref: WorldPulseReadSeedV1
    what_i_learned: str = Field(min_length=1)
    candidate_priors: list[WorldPulseReadPriorCandidateV1] = Field(default_factory=list)
    concept_candidates: list[WorldPulseReadConceptCandidateV1] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    trace_id: str = Field(min_length=1)
    created_at: datetime
    producer_hint: Literal["world_pulse_read_pipeline"] = "world_pulse_read_pipeline"
```

In `orion/schemas/registry.py`, add imports next to other world_pulse imports and register:

```python
from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadPriorCandidateV1,
    WorldPulseReadSeedV1,
)
# in the models dict (same pattern as DailyWorldPulseV1):
"WorldPulseReadSeedV1": WorldPulseReadSeedV1,
"WorldPulseReadHandoffV1": WorldPulseReadHandoffV1,
"WorldPulseReadConceptCandidateV1": WorldPulseReadConceptCandidateV1,
"WorldPulseReadPriorCandidateV1": WorldPulseReadPriorCandidateV1,
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_world_pulse_read_schemas.py -v`  
Expected: PASS. Also: `python scripts/check_schema_registry.py` → exit 0.

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/world_pulse_read.py orion/schemas/registry.py tests/test_world_pulse_read_schemas.py
git commit -m "$(cat <<'EOF'
feat(world-pulse-read): add Stage 1 seed and handoff schemas

EOF
)"
```

---

### Task 2: Seed extraction + stable `seed_id`

**Files:**
- Create: `orion/world_pulse_read/__init__.py`, `orion/world_pulse_read/seeds.py`
- Test: `tests/test_world_pulse_read_seeds.py`

**Interfaces:**
- Consumes: digest `payload_json` shaped like `DailyWorldPulseV1` (findings in `curiosity_followups[*].articles`, items in `items`)
- Produces: `make_seed_id(...) -> str`, `seeds_from_digest_payload(payload: dict, *, article_urls: dict[str, str] | None = None) -> list[WorldPulseReadSeedV1]` (findings first)

- [ ] **Step 1: Write the failing test**

```python
from orion.world_pulse_read.seeds import make_seed_id, seeds_from_digest_payload


def test_make_seed_id_is_stable_for_same_inputs():
    a = make_seed_id(kind="finding", run_id="r1", url="https://ex.com/a")
    b = make_seed_id(kind="finding", run_id="r1", url="https://ex.com/a")
    assert a == b
    assert a.startswith("finding:r1:")


def test_seeds_findings_before_digest_items():
    payload = {
        "run_id": "r1",
        "curiosity_followups": [
            {
                "section": "ai_technology",
                "articles": [
                    {"url": "https://ex.com/f1", "title": "F1", "description": "", "salience": 0.9}
                ],
            }
        ],
        "items": [
            {
                "item_id": "item-1",
                "run_id": "r1",
                "title": "Digest card",
                "category": "hardware_compute_gpu",
                "worth_reading": ["https://ex.com/d1"],
                "article_ids": [],
            }
        ],
    }
    seeds = seeds_from_digest_payload(payload)
    assert [s.kind for s in seeds] == ["finding", "digest_item"]
    assert seeds[0].url == "https://ex.com/f1"
    assert seeds[1].url == "https://ex.com/d1"
    assert seeds[1].item_id == "item-1"


def test_digest_item_resolves_url_via_article_map_when_worth_reading_empty():
    payload = {
        "run_id": "r1",
        "curiosity_followups": [],
        "items": [
            {
                "item_id": "item-2",
                "run_id": "r1",
                "title": "No worth_reading",
                "category": "ai_technology",
                "worth_reading": [],
                "article_ids": ["art-9"],
            }
        ],
    }
    seeds = seeds_from_digest_payload(
        payload, article_urls={"art-9": "https://ex.com/from-article"}
    )
    assert len(seeds) == 1
    assert seeds[0].url == "https://ex.com/from-article"


def test_digest_item_without_url_is_skipped():
    payload = {
        "run_id": "r1",
        "curiosity_followups": [],
        "items": [
            {
                "item_id": "item-3",
                "run_id": "r1",
                "title": "Orphan",
                "category": "ai_technology",
                "worth_reading": [],
                "article_ids": [],
            }
        ],
    }
    assert seeds_from_digest_payload(payload) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_world_pulse_read_seeds.py -v`  
Expected: FAIL import / missing functions.

- [ ] **Step 3: Implement**

`orion/world_pulse_read/__init__.py` — empty or docstring only.

`orion/world_pulse_read/seeds.py`:

```python
from __future__ import annotations

import hashlib
from typing import Any

from orion.schemas.world_pulse_read import WorldPulseReadSeedV1


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def make_seed_id(
    *,
    kind: str,
    run_id: str,
    url: str,
    item_id: str | None = None,
) -> str:
    h = _url_hash(url)
    if kind == "digest_item":
        if not item_id:
            raise ValueError("digest_item seed_id requires item_id")
        return f"digest_item:{run_id}:{item_id}:{h}"
    if kind == "finding":
        return f"finding:{run_id}:{h}"
    raise ValueError(f"unknown seed kind: {kind}")


def _http_url(value: str) -> str | None:
    v = (value or "").strip()
    if v.startswith("http://") or v.startswith("https://"):
        return v
    return None


def seeds_from_digest_payload(
    payload: dict[str, Any],
    *,
    article_urls: dict[str, str] | None = None,
) -> list[WorldPulseReadSeedV1]:
    """Findings first, then digest items. Skip rows with no resolvable URL."""
    article_urls = article_urls or {}
    run_id = str(payload.get("run_id") or "")
    out: list[WorldPulseReadSeedV1] = []

    for followup in payload.get("curiosity_followups") or []:
        section = str(followup.get("section") or "")
        for art in followup.get("articles") or []:
            url = _http_url(str(art.get("url") or ""))
            if not url or not run_id:
                continue
            out.append(
                WorldPulseReadSeedV1(
                    seed_id=make_seed_id(kind="finding", run_id=run_id, url=url),
                    kind="finding",
                    run_id=run_id,
                    url=url,
                    title=str(art.get("title") or ""),
                    section=section,
                )
            )

    for item in payload.get("items") or []:
        item_id = str(item.get("item_id") or "")
        if not item_id or not run_id:
            continue
        url = None
        for wr in item.get("worth_reading") or []:
            url = _http_url(str(wr))
            if url:
                break
        if url is None:
            for aid in item.get("article_ids") or []:
                mapped = article_urls.get(str(aid))
                if mapped:
                    url = _http_url(mapped)
                    if url:
                        break
        if not url:
            continue
        out.append(
            WorldPulseReadSeedV1(
                seed_id=make_seed_id(
                    kind="digest_item", run_id=run_id, url=url, item_id=item_id
                ),
                kind="digest_item",
                run_id=run_id,
                url=url,
                title=str(item.get("title") or ""),
                section=str(item.get("category") or ""),
                item_id=item_id,
            )
        )
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_world_pulse_read_seeds.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orion/world_pulse_read/ tests/test_world_pulse_read_seeds.py
git commit -m "$(cat <<'EOF'
feat(world-pulse-read): extract ordered seeds with stable seed_id

EOF
)"
```

---

### Task 3: Postgres seed queue

**Files:**
- Create: `services/orion-sql-db/manual_migration_world_pulse_read_seed_queue_v1.sql`
- Create: `orion/world_pulse_read/queue.py`
- Test: `services/orion-hub/tests/test_world_pulse_read_queue.py` (fake asyncpg conn)

**Interfaces:**
- Consumes: `WorldPulseReadSeedV1`, asyncpg connection
- Produces: `ensure_seed_queue_schema(conn)`, `enqueue_seeds(conn, seeds) -> int` (inserted count; ON CONFLICT DO NOTHING), `claim_next_seed(conn) -> WorldPulseReadSeedV1 | None`, `mark_seed_done(conn, seed_id, *, trace_id)`, `mark_seed_failed(conn, seed_id, *, error)`

- [ ] **Step 1: Write migration + failing queue tests**

Migration file:

```sql
-- World-pulse Concept Atlas read pipeline seed queue v1
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_world_pulse_read_seed_queue_v1.sql

create table if not exists world_pulse_read_seed (
    seed_id text primary key,
    kind text not null check (kind in ('finding', 'digest_item')),
    run_id text not null,
    url text not null,
    title text not null default '',
    section text not null default '',
    item_id text null,
    priority int not null default 100,
    status text not null default 'pending'
        check (status in ('pending', 'claimed', 'done', 'failed', 'skipped')),
    trace_id text null,
    last_error text null,
    created_at timestamptz not null default now(),
    claimed_at timestamptz null,
    completed_at timestamptz null
);

create index if not exists idx_world_pulse_read_seed_claim
    on world_pulse_read_seed (status, priority, created_at);

create index if not exists idx_world_pulse_read_seed_run
    on world_pulse_read_seed (run_id);
```

Test (in-memory fake recording SQL is enough — mirror Hub fake-conn style):

```python
import asyncio
from types import SimpleNamespace

from orion.schemas.world_pulse_read import WorldPulseReadSeedV1
from orion.world_pulse_read.queue import enqueue_seeds, claim_next_seed


class _FakeConn:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.executed = []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        if "INSERT INTO world_pulse_read_seed" in sql:
            seed_id = args[0]
            if seed_id in self.rows:
                return "INSERT 0 0"
            self.rows[seed_id] = {
                "seed_id": args[0],
                "kind": args[1],
                "run_id": args[2],
                "url": args[3],
                "title": args[4],
                "section": args[5],
                "item_id": args[6],
                "priority": args[7],
                "status": "pending",
            }
            return "INSERT 0 1"
        if "UPDATE world_pulse_read_seed" in sql and "claimed" in sql:
            pending = sorted(
                (r for r in self.rows.values() if r["status"] == "pending"),
                key=lambda r: (r["priority"], r["seed_id"]),
            )
            if not pending:
                return None
            row = pending[0]
            row["status"] = "claimed"
            return SimpleNamespace(**row)
        return "OK"

    async def fetchrow(self, sql, *args):
        return await self.execute(sql, *args)

    async def fetchval(self, sql, *args):
        return None


def test_enqueue_is_idempotent_on_seed_id():
    conn = _FakeConn()
    seed = WorldPulseReadSeedV1(
        seed_id="finding:r1:x",
        kind="finding",
        run_id="r1",
        url="https://ex.com/a",
        title="A",
        section="ai_technology",
    )

    async def _run():
        n1 = await enqueue_seeds(conn, [seed])
        n2 = await enqueue_seeds(conn, [seed])
        return n1, n2

    assert asyncio.run(_run()) == (1, 0)


def test_claim_prefers_lower_priority_number():
    """Findings priority=0, digest_item priority=10."""
    conn = _FakeConn()
    finding = WorldPulseReadSeedV1(
        seed_id="finding:r1:a",
        kind="finding",
        run_id="r1",
        url="https://ex.com/f",
        title="F",
        section="ai_technology",
    )
    item = WorldPulseReadSeedV1(
        seed_id="digest_item:r1:i1:b",
        kind="digest_item",
        run_id="r1",
        url="https://ex.com/d",
        title="D",
        section="hardware_compute_gpu",
        item_id="i1",
    )

    async def _run():
        await enqueue_seeds(conn, [item, finding])  # insert item first on purpose
        claimed = await claim_next_seed(conn)
        return claimed

    claimed = asyncio.run(_run())
    assert claimed is not None
    assert claimed.kind == "finding"
```

Implement `enqueue_seeds` / `claim_next_seed` so the fake above works — if the fake SQL matching is too brittle, prefer a small in-module queue store used only when `conn` is the test double; **preferred:** implement real SQL and adjust the fake to interpret the real SQL strings the module emits (copy patterns from `test_curiosity_investigation` fakes). Keep the public signatures exact.

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest services/orion-hub/tests/test_world_pulse_read_queue.py -v`  
Expected: FAIL missing module.

- [ ] **Step 3: Implement `orion/world_pulse_read/queue.py`**

```python
from __future__ import annotations

from typing import Any, Sequence

from orion.schemas.world_pulse_read import WorldPulseReadSeedV1

_PRIORITY = {"finding": 0, "digest_item": 10}

ENSURE_SQL = """
create table if not exists world_pulse_read_seed (
    seed_id text primary key,
    kind text not null check (kind in ('finding', 'digest_item')),
    run_id text not null,
    url text not null,
    title text not null default '',
    section text not null default '',
    item_id text null,
    priority int not null default 100,
    status text not null default 'pending'
        check (status in ('pending', 'claimed', 'done', 'failed', 'skipped')),
    trace_id text null,
    last_error text null,
    created_at timestamptz not null default now(),
    claimed_at timestamptz null,
    completed_at timestamptz null
);
create index if not exists idx_world_pulse_read_seed_claim
    on world_pulse_read_seed (status, priority, created_at);
"""

INSERT_SQL = """
INSERT INTO world_pulse_read_seed
    (seed_id, kind, run_id, url, title, section, item_id, priority, status)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'pending')
ON CONFLICT (seed_id) DO NOTHING
"""

CLAIM_SQL = """
UPDATE world_pulse_read_seed
SET status = 'claimed', claimed_at = now()
WHERE seed_id = (
    SELECT seed_id FROM world_pulse_read_seed
    WHERE status = 'pending'
    ORDER BY priority ASC, created_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING seed_id, kind, run_id, url, title, section, item_id
"""


async def ensure_seed_queue_schema(conn: Any) -> None:
    await conn.execute(ENSURE_SQL)


async def enqueue_seeds(conn: Any, seeds: Sequence[WorldPulseReadSeedV1]) -> int:
    inserted = 0
    for seed in seeds:
        priority = _PRIORITY[seed.kind]
        status = await conn.execute(
            INSERT_SQL,
            seed.seed_id,
            seed.kind,
            seed.run_id,
            seed.url,
            seed.title,
            seed.section,
            seed.item_id,
            priority,
        )
        # asyncpg returns "INSERT 0 1" / "INSERT 0 0"
        if isinstance(status, str) and status.endswith("1"):
            inserted += 1
    return inserted


async def claim_next_seed(conn: Any) -> WorldPulseReadSeedV1 | None:
    row = await conn.fetchrow(CLAIM_SQL)
    if not row:
        return None
    return WorldPulseReadSeedV1(
        seed_id=row["seed_id"],
        kind=row["kind"],
        run_id=row["run_id"],
        url=row["url"],
        title=row["title"] or "",
        section=row["section"] or "",
        item_id=row["item_id"],
    )


async def mark_seed_done(conn: Any, seed_id: str, *, trace_id: str) -> None:
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET status = 'done', trace_id = $2, completed_at = now(), last_error = null
        WHERE seed_id = $1
        """,
        seed_id,
        trace_id,
    )


async def mark_seed_failed(conn: Any, seed_id: str, *, error: str) -> None:
    await conn.execute(
        """
        UPDATE world_pulse_read_seed
        SET status = 'failed', last_error = $2, completed_at = now()
        WHERE seed_id = $1
        """,
        seed_id,
        error[:2000],
    )
```

Adjust the unit-test fake so `fetchrow(CLAIM_SQL)` returns the claimed row mapping; keep INSERT idempotency semantics.

Also add a helper used by the live loop (can live in same module):

```python
async def enqueue_from_recent_digests(conn: Any, *, limit_digests: int = 5) -> int:
    """Pull recent world_pulse_digest rows and enqueue seeds (findings first via priority)."""
    digests = await conn.fetch(
        """
        SELECT run_id, payload_json
        FROM world_pulse_digest
        ORDER BY created_at DESC NULLS LAST
        LIMIT $1
        """,
        limit_digests,
    )
    # load article urls for those runs
    run_ids = [d["run_id"] for d in digests]
    article_urls: dict[str, str] = {}
    if run_ids:
        rows = await conn.fetch(
            """
            SELECT article_id, url FROM world_pulse_article
            WHERE run_id = ANY($1::text[])
            """,
            run_ids,
        )
        article_urls = {r["article_id"]: r["url"] for r in rows}
    from orion.world_pulse_read.seeds import seeds_from_digest_payload

    total = 0
    # chronological enqueue: oldest of the limited set first
    for d in reversed(list(digests)):
        payload = d["payload_json"]
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        seeds = seeds_from_digest_payload(payload, article_urls=article_urls)
        total += await enqueue_seeds(conn, seeds)
    return total
```

Cover `enqueue_from_recent_digests` with a focused fake test that feeds one digest row + one article row.

- [ ] **Step 4: Run tests**

Run: `pytest services/orion-hub/tests/test_world_pulse_read_queue.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-db/manual_migration_world_pulse_read_seed_queue_v1.sql \
  orion/world_pulse_read/queue.py \
  services/orion-hub/tests/test_world_pulse_read_queue.py
git commit -m "$(cat <<'EOF'
feat(world-pulse-read): durable Postgres seed queue with idempotent enqueue

EOF
)"
```

---

### Task 4: Wallet A (isolated Redis budget)

**Files:**
- Create: `orion/world_pulse_read/wallet_a.py`
- Test: `services/orion-hub/tests/test_world_pulse_read_wallet_a.py`

**Interfaces:**
- Produces constants: `WALLET_A_COOLDOWN_KEY = "orion:wp_read:wallet_a:last_at"`, `WALLET_A_COUNT_KEY_PREFIX = "orion:wp_read:wallet_a:count:"`
- Produces: `wallet_a_block_reason(inp) -> str | None`, `WalletAInputs` dataclass, `async debit_wallet_a(redis, *, now, timezone_name, ttl_sec=172800)`, `async read_wallet_a_state(redis, *, now, timezone_name) -> tuple[float|None, int]`
- Must **not** import curiosity Redis key constants except in tests that assert curiosity keys are untouched after debit.

- [ ] **Step 1: Write failing tests**

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from orion.world_pulse_read import wallet_a as wa


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def incr(self, key):
        self.store[key] = str(int(self.store.get(key, "0")) + 1)
        return int(self.store[key])

    async def expire(self, key, ttl):
        return True


def test_block_reason_daily_cap():
    inp = wa.WalletAInputs(
        enabled=True,
        done_today=6,
        daily_cap=6,
        seconds_since_last=99999,
        min_cooldown_sec=60,
        now_hour=12,
        window_start_hour=8,
        window_end_hour=22,
    )
    assert wa.wallet_a_block_reason(inp) == "daily_cap"


def test_debit_uses_wallet_a_keys_only():
    r = _FakeRedis()
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)

    async def _run():
        await wa.debit_wallet_a(r, now=now, timezone_name="UTC")
        return sorted(r.store)

    keys = asyncio.run(_run())
    assert keys == sorted(
        [
            wa.WALLET_A_COOLDOWN_KEY,
            wa.WALLET_A_COUNT_KEY_PREFIX + "2026-09-06",
        ]
    )
    # Curiosity Atlas keys must not appear
    assert not any(k.startswith("orion:curiosity:") for k in keys)
```

Also add a regression that imports curiosity constants and asserts debit never writes them even if those keys pre-exist:

```python
def test_debit_does_not_touch_preexisting_curiosity_keys():
    from scripts.curiosity_investigation import _COOLDOWN_KEY, _DAILY_COUNT_KEY_PREFIX

    r = _FakeRedis()
    r.store[_COOLDOWN_KEY] = "already"
    r.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] = "3"
    now = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)
    asyncio.run(wa.debit_wallet_a(r, now=now, timezone_name="UTC"))
    assert r.store[_COOLDOWN_KEY] == "already"
    assert r.store[_DAILY_COUNT_KEY_PREFIX + "2026-09-06"] == "3"
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest services/orion-hub/tests/test_world_pulse_read_wallet_a.py -v`  
Expected: FAIL missing module.

- [ ] **Step 3: Implement `wallet_a.py`**

Mirror the gate order from `curiosity_investigation.scheduling_block_reason` (`disabled` → `daily_cap` → `outside_window` → `cooldown`), but with **local** key constants and `paced_cooldown_sec` either copied thinly or imported from `scripts.curiosity_investigation` **only if** that import stays test-safe; prefer duplicating the small pure `paced_cooldown_sec` / window helpers into `wallet_a.py` or a shared pure module **only if** already shared — YAGNI: copy the ~20-line pure helpers into `wallet_a.py` rather than coupling to the curiosity loop module.

```python
WALLET_A_COOLDOWN_KEY = "orion:wp_read:wallet_a:last_at"
WALLET_A_COUNT_KEY_PREFIX = "orion:wp_read:wallet_a:count:"
_STATE_TTL_SEC = 172800
```

`debit_wallet_a`: `setex` cooldown ISO timestamp; `incr` daily key; `expire` daily key TTL.

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/world_pulse_read/wallet_a.py services/orion-hub/tests/test_world_pulse_read_wallet_a.py
git commit -m "$(cat <<'EOF'
feat(world-pulse-read): Wallet A Redis budget isolated from Curiosity Atlas

EOF
)"
```

---

### Task 5: Substrate adapter (handoff → Concept Atlas record)

**Files:**
- Create: `orion/substrate/adapters/world_pulse_read.py`
- Test: `tests/test_world_pulse_read_adapter.py`

**Interfaces:**
- Consumes: `WorldPulseReadHandoffV1`
- Produces: `map_world_pulse_read_handoff_to_substrate(handoff: WorldPulseReadHandoffV1, *, observed_at: datetime | None = None) -> SubstrateGraphRecordV1`
- Each concept candidate → `ConceptNodeV1` with:
  - `provenance.producer == "world_pulse_read_pipeline"`
  - `provenance.source_kind == "world_pulse.read"`
  - `provenance.source_channel == "orion:world_pulse:read"`
  - `provenance.trace_id == handoff.trace_id`
  - `provenance.evidence_refs` containing the seed URL, `run_id`, and `trace_id` (as separate strings)
- Empty `concept_candidates` → empty nodes list (still valid record); callers decide whether to materialize.

- [ ] **Step 1: Write failing test**

```python
from datetime import datetime, timezone

from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
)
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate


def _handoff(**over):
    base = dict(
        seed_ref=WorldPulseReadSeedV1(
            seed_id="finding:r1:x",
            kind="finding",
            run_id="r1",
            url="https://ex.com/a",
            title="A",
            section="ai_technology",
        ),
        what_i_learned="Learned about packaging.",
        concept_candidates=[
            WorldPulseReadConceptCandidateV1(label="advanced packaging", definition="chip pkg")
        ],
        trace_id="tr-9",
        created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
    )
    base.update(over)
    return WorldPulseReadHandoffV1(**base)


def test_mapper_sets_locked_provenance():
    record = map_world_pulse_read_handoff_to_substrate(_handoff())
    concepts = [n for n in record.nodes if n.node_kind == "concept"]
    assert len(concepts) == 1
    p = concepts[0].provenance
    assert p.producer == "world_pulse_read_pipeline"
    assert p.source_kind == "world_pulse.read"
    assert p.trace_id == "tr-9"
    assert "https://ex.com/a" in p.evidence_refs
    assert "r1" in p.evidence_refs
    assert "tr-9" in p.evidence_refs
    assert concepts[0].label == "advanced packaging"


def test_mapper_skips_empty_labels_only_via_schema(min_length_enforced=True):
    record = map_world_pulse_read_handoff_to_substrate(
        _handoff(concept_candidates=[])
    )
    assert record.nodes == []
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement adapter**

Follow `concept_induction.py` / `topic_foundry.py` construction:

```python
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.adapters._common import make_provenance, make_temporal
import hashlib

def _concept_node_id(trace_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{trace_id}:{label}".encode()).hexdigest()[:16]
    return f"sub-concept-wp-read-{digest}"

def map_world_pulse_read_handoff_to_substrate(handoff, *, observed_at=None):
    observed = observed_at or handoff.created_at
    seed = handoff.seed_ref
    evidence = [seed.url, seed.run_id, handoff.trace_id]
    nodes = []
    for cand in handoff.concept_candidates:
        nodes.append(
            ConceptNodeV1(
                node_id=_concept_node_id(handoff.trace_id, cand.label),
                anchor_scope="orion",
                subject_ref="world_pulse",
                temporal=make_temporal(observed_at=observed),
                provenance=make_provenance(
                    source_kind="world_pulse.read",
                    source_channel="orion:world_pulse:read",
                    producer="world_pulse_read_pipeline",
                    correlation_id=seed.run_id,
                    trace_id=handoff.trace_id,
                    evidence_refs=evidence,
                ),
                label=cand.label,
                definition=cand.definition,
                signals=SubstrateSignalBundleV1(confidence=0.5, salience=0.5),
                metadata={
                    "seed_id": seed.seed_id,
                    "seed_kind": seed.kind,
                    "section": seed.section,
                },
            )
        )
    return SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="world_pulse",
        nodes=nodes,
        edges=[],
    )
```

`SubstrateGraphRecordV1` requires `anchor_scope` (see `cognitive_substrate.py`); match topic-foundry / concept-induction call sites if they set additional fields.

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/adapters/world_pulse_read.py tests/test_world_pulse_read_adapter.py
git commit -m "$(cat <<'EOF'
feat(world-pulse-read): map Stage 1 handoff to Concept Atlas nodes

EOF
)"
```

---

### Task 6: Stage 1 Hub loop (tick → read → materialize → journal)

**Files:**
- Create: `services/orion-hub/scripts/world_pulse_read_pipeline.py`
- Test: `services/orion-hub/tests/test_world_pulse_read_pipeline.py`

**Interfaces:**
- Class `WorldPulseReadPipeline` with `__init__(enabled, tick_interval_sec, min_cooldown_sec, daily_cap, window_start_hour, window_end_hour, timeout_sec, session_id, llm_route, timezone_name, pool_provider, source_ref, step_relay_provider, store_provider, ...)`
- Methods: `async start(bus, harness_rpc_bus=None)`, `async stop()`, `async tick(*, force: bool = False) -> str | None`
- Injectable: `async _stage1_read(self, seed) -> WorldPulseReadHandoffV1` (tests patch this; production calls `execute_unified_turn` and parses JSON handoff)
- On success: Wallet A debit **before** FCC (same debit-before-turn discipline as curiosity), claim seed, materialize via `SubstrateGraphMaterializer`, journal to `orion:journal:write` with `source_kind="world_pulse"`, `source_ref=f"world_pulse_read:{trace_id}"`, mark seed done
- On empty generation / parse failure: mark seed failed; Wallet A already debited
- **Never** call curiosity debit APIs

- [ ] **Step 1: Write failing pipeline tests**

Key cases (use `_FakeBus`, `_FakeRedis`, fake pool/conn from Task 3/4 patterns):

1. `tick` when Wallet A at cap → `"daily_cap"` and **no** seed claim.
2. Happy path with patched `_stage1_read` returning a handoff with one concept → materializer/store receives a node with `producer=world_pulse_read_pipeline`; journal publish happens; curiosity Redis keys unchanged.
3. `force=True` skips schedule gate (still debits Wallet A).

Sketch:

```python
def test_pipeline_happy_path_writes_concept_and_ignores_curiosity_wallet(monkeypatch):
    # build pipeline with fake redis/bus/pool/store
    # patch _stage1_read to return handoff with one concept candidate
    # asyncio.run(pipeline.tick(force=True))
    # assert store saw producer world_pulse_read_pipeline
    # assert journal envelope on fake bus
    # assert orion:curiosity:* keys untouched
    ...
```

Use `InMemorySubstrateGraphStore` + `SubstrateGraphMaterializer` for the store side when possible (see topic-foundry Hub tests).

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement loop**

Skeleton responsibilities inside `world_pulse_read_pipeline.py`:

1. `_run` loop sleeping `tick_interval_sec`.
2. `tick`: optional `enqueue_from_recent_digests` once per tick when pool available (cap `limit_digests=3` to stay thin).
3. Read Wallet A state from `bus._redis` (same access pattern as curiosity).
4. Gate via `wallet_a_block_reason`.
5. Claim seed; if none → return `"empty_queue"`.
6. `debit_wallet_a`.
7. `handoff = await self._stage1_read(seed)` (uuid `trace_id` if model omits — prefer model requires it; Stage 1 builder sets `trace_id=str(uuid4())` before prompt and forces it on parsed output).
8. `record = map_world_pulse_read_handoff_to_substrate(handoff)`; if nodes: materialize.
9. Publish journal entry summarizing `what_i_learned` + URL + `trace_id`.
10. `mark_seed_done`.

Production `_stage1_read`: build a prompt that instructs Orion to return a JSON object matching `WorldPulseReadHandoffV1` fields (seed_ref echoed, what_i_learned, candidates, open_threads). Parse fenced JSON; validate with Pydantic; on failure raise / return None handled by tick.

Keep file sibling-sized; do not paste the entire curiosity investigation module.

- [ ] **Step 4: Run tests — PASS**

Run: `pytest services/orion-hub/tests/test_world_pulse_read_pipeline.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/world_pulse_read_pipeline.py \
  services/orion-hub/tests/test_world_pulse_read_pipeline.py
git commit -m "$(cat <<'EOF'
feat(world-pulse-read): Stage 1 Hub loop with Concept Atlas + journal landing

EOF
)"
```

---

### Task 7: Env, Hub wiring, minimal schedule route

**Files:**
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-hub/scripts/main.py`
- Create: `services/orion-hub/scripts/world_pulse_read_routes.py`
- Test: `services/orion-hub/tests/test_world_pulse_read_routes.py` (schedule JSON shape + key import)
- Run: `python scripts/sync_local_env_from_example.py`

**Interfaces:**
- Settings (defaults):
  - `HUB_WORLD_PULSE_READ_ENABLED: bool = False`
  - `HUB_WORLD_PULSE_READ_TICK_SEC: float = 300`
  - `HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC: float = 1800`
  - `HUB_WORLD_PULSE_READ_DAILY_CAP: int = 6`  # Wallet A
  - `HUB_WORLD_PULSE_READ_WINDOW_START_HOUR: int = 8`
  - `HUB_WORLD_PULSE_READ_WINDOW_END_HOUR: int = 22`
  - `HUB_WORLD_PULSE_READ_TIMEOUT_SEC: float = 3500`
  - `HUB_WORLD_PULSE_READ_SESSION_ID: str = "orion_world_pulse_read"`
  - `HUB_WORLD_PULSE_READ_LLM_ROUTE: str = "agent"`
- `GET /world-pulse-read/api/schedule` returns `{enabled, done_today, daily_cap, cooldown_key, count_key_prefix, ...}` importing key constants from `wallet_a` (never retyped).

- [ ] **Step 1: Write route + settings tests first**

Assert settings defaults and that schedule payload includes `daily_cap == 6` and the Wallet A key strings.

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Wire settings, env example, main.py, router**

In `main.py`, mirror curiosity construction (same `pool_provider`, `timezone_name=HUB_ENDOGENOUS_OUTREACH_TZ`, `source_ref`, `step_relay_provider`, `harness_rpc_bus`). Default **enabled=False** so deploy is opt-in.

Document in `.env_example` comments: Wallet A is independent of `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`.

After `.env_example` change:

```bash
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
```

- [ ] **Step 4: Run focused suite**

```bash
pytest tests/test_world_pulse_read_schemas.py \
  tests/test_world_pulse_read_seeds.py \
  tests/test_world_pulse_read_adapter.py \
  services/orion-hub/tests/test_world_pulse_read_wallet_a.py \
  services/orion-hub/tests/test_world_pulse_read_queue.py \
  services/orion-hub/tests/test_world_pulse_read_pipeline.py \
  services/orion-hub/tests/test_world_pulse_read_routes.py -q
```

Expected: all PASS.

Operator note (print in PR, do not auto-apply):  

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_world_pulse_read_seed_queue_v1.sql
```

Hub restart after merge when enabling:

```bash
# Juniper runs — agents must not sudo
# rebuild/restart orion-hub via scripts/safe_docker_build.sh from a worktree
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/app/settings.py services/orion-hub/.env_example \
  services/orion-hub/scripts/main.py \
  services/orion-hub/scripts/world_pulse_read_routes.py \
  services/orion-hub/tests/test_world_pulse_read_routes.py
git commit -m "$(cat <<'EOF'
feat(world-pulse-read): wire Stage 1 loop, env, and schedule route

EOF
)"
```

---

## Acceptance mapping (spec → this plan)

| Spec check | Task |
|---|---|
| Concept Atlas nodes with `producer=world_pulse_read_pipeline` + URL in evidence | 5, 6 |
| Distinguishable from topic-foundry via producer | 5 |
| Wallet A increments; Curiosity Atlas counter does not | 4, 6 |
| Stage 2 re-enters Stage 1 / Wallet B | **Deferred** |
| Trace links Stage 1 (`trace_id` journal + provenance) | 6 |
| Backfill dedupe by `seed_id` | Queue idempotency in 3; full backfill script **Deferred** |
| Focused tests handoff + provenance | 1, 5, 6 |

Live end-to-end FCC read remains **UNVERIFIED** until enabled on Hub with migration applied — gate tests cover the deterministic path with injectable Stage 1.

---

## Self-review (plan author)

1. **Spec coverage:** Stage 1 vertical slice + wallet isolation + provenance + journal + durable queue covered. Stage 2/B/backfill explicitly deferred per design “Recommended next patch.”
2. **Placeholders:** None intentional; implementers must flesh fake-conn SQL matching in Task 3 if the sketched fake is too thin — keep public signatures stable.
3. **Type consistency:** `WorldPulseReadHandoffV1` / `WorldPulseReadSeedV1` names align across Tasks 1–6; Redis key strings align across Tasks 4 and 7; producer/source_kind strings match the design verbatim.
