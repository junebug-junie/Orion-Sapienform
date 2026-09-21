# Lived-Self Curiosity Lanes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single anatomy standing question with a pin+mint question pool on the existing self-inquiry seam, draw ~3/4 lived vs ~1/4 anatomy from one shared self budget, write evidenced lived answers into a ledger chat and the Self panel can read, and stop `line=self` priors from leaking into situation/outreach.

**Architecture:** Pure Python pool + pick (seed YAML + durable ask/mint state in Postgres via Hub). Kickoff takes the drawn question. Anatomy keeps `:SelfDefinition` for `family=anatomy`. Lived answers MERGEs a `:LivedAnswer` (or keyed SelfDefinition sibling) and mirror into `self_concept_history` as `concept_id=self:lived:<question_id>`. Shared identity inject + Self panel read those rows. Situation and outreach switch to a non-self live-priors Cypher. No second loop; no v2 anatomy sub-inquiry in this plan.

**Tech Stack:** Python 3.12, pytest, Hub curiosity investigation, FalkorDB Cypher (`orion_worldview`), Postgres (`self_concept_history` + new `curiosity_self_questions`), felt-state reader, cortex-exec identity inject.

**Spec:** [`docs/superpowers/specs/2026-09-16-lived-self-curiosity-lanes-design.md`](../specs/2026-09-16-lived-self-curiosity-lanes-design.md)

**Branch / worktree:** Plan lives on `docs/lived-self-curiosity-lanes`. Implement on `feat/lived-self-curiosity-lanes` via `scripts/new_worktree.sh feat lived-self-curiosity-lanes` (do not commit from the shared checkout). Merge/rebase the design commits first.

## Global Constraints

- Same curiosity **self-inquiry** seam only — do not add a second loop or service.
- Two families only: `lived` | `anatomy`. Seed facets are questions, not a closed lane enum.
- One shared self daily pot (`HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP`); lived_weight default **0.75**; pinned floor beats ratio.
- World investigate budget untouched.
- Prior contracts: `line=self`, MERGE on `prior_id`, live = not `{refuted, retired_unresolvable}`, one `HUB_CURIOSITY_STALE_PRIOR_TESTS` — no new stale constant.
- **Exclude `line=self` from situation world-priors and outreach talkable-prior fetches.**
- Peer (#2219) identity strip stays; peer never drafts ledger / SelfDefinition.
- No keyword detectors on Juniper chat to mint questions.
- Empty-shell ban: refuse ledger mirror with empty answer or empty evidence (same as SelfDefinition).
- Write-early MERGE (by hop 2) for lived answers — same clock rule as #2165.
- Env parity: new `.env_example` keys → `python scripts/sync_local_env_from_example.py` + `settings.py` + compose.
- No claim of sentience; acceptance is grounded, revisable, reaches chat/panel.

**Non-goals (do not implement):** lived→anatomy sub-inquiry enqueue; dashboard family-ratio metric; durable reading hire; contractor peer changes beyond coexistence; new ontology of care/bond enums.

---

## File map

| Path | Responsibility |
|------|----------------|
| `orion/curiosity/worldview.py` | **Modify** — `LIVE_NON_SELF_PRIORS_CYPHER` (live AND line is null or ≠ `self`) |
| `orion/situational/context.py` | **Modify** — situation prior fetch uses non-self query / filter |
| `services/orion-hub/scripts/endogenous_outreach.py` | **Modify** — `_fetch_open_prior_previews` uses non-self Cypher |
| `orion/curiosity/self_question_pool.py` | **Create** — seed load, SelfQuestion, pick, record_ask, mint helpers |
| `orion/curiosity/self_question_seed.yaml` | **Create** — Juniper-pinned starter pack + anatomy pin |
| `orion/curiosity/self_inquiry.py` | **Modify** — keep STANDING_QUESTION as anatomy default text; LivedAnswer mirror helpers; concept_id helpers |
| `orion/curiosity/self_inquiry_prompt.py` | **Modify** — accept drawn `SelfQuestion`; family-specific write teach |
| `services/orion-hub/scripts/curiosity_investigation.py` | **Modify** — pick before kickoff; record_ask; pass question into prompt; mirror lived answers |
| `services/orion-hub/app/settings.py` + `.env_example` + compose | **Modify** — `HUB_CURIOSITY_SELF_LIVED_WEIGHT`, `HUB_CURIOSITY_SELF_PINNED_FLOOR_DAYS`, feature flag if needed |
| `scripts/sql/2026-09-18_curiosity_self_questions.sql` | **Create** — pool table + grants note |
| `orion/curiosity/self_panel.py` + Atlas template/routes | **Modify** — show lived ledger answers |
| `orion/substrate/felt_state_reader.py` + `chat_stance.py` / inject path | **Modify** — hydrate + inject pinned lived answers |
| `orion/curiosity/README.md` + Hub README | **Modify** — document pool, ratio, filters |
| Tests under `orion/curiosity/tests/`, Hub tests, situational tests | **Create/Modify** — per task |

---

## Patch 1 — Stop the footgun + pool draw + kickoff

### Task 1: Non-self live priors Cypher + situation/outreach filter

**Files:**
- Modify: `orion/curiosity/worldview.py` (near `LIVE_PRIORS_CYPHER`)
- Modify: `orion/situational/context.py` (`_fetch_curiosity_context` / `read_snapshot` call path)
- Modify: `services/orion-hub/scripts/endogenous_outreach.py` (`_fetch_open_prior_previews`)
- Test: `orion/curiosity/tests/test_live_non_self_priors_cypher.py`
- Test: `services/orion-hub/tests/test_endogenous_outreach_self_prior_filter.py`
- Test: extend `services/orion-cortex-exec/tests/test_situation_curiosity_reverie_context.py` (or Hub situational twin)

**Interfaces:**
- Consumes: `SELF_PRIOR_LINE` from `orion.curiosity.self_inquiry` (or duplicate the string `"self"` in worldview with a comment pointing at self_inquiry — prefer importing the constant)
- Produces: `LIVE_NON_SELF_PRIORS_CYPHER: str`

- [ ] **Step 1: Write the failing Cypher test**

```python
# orion/curiosity/tests/test_live_non_self_priors_cypher.py
from orion.curiosity.self_inquiry import SELF_PRIOR_LINE
from orion.curiosity.worldview import LIVE_NON_SELF_PRIORS_CYPHER, LIVE_PRIORS_CYPHER


def test_non_self_cypher_excludes_self_line_and_keeps_live_rule() -> None:
    assert "LIVE_NON_SELF_PRIORS_CYPHER" in dir(__import__("orion.curiosity.worldview", fromlist=["*"]))
    q = LIVE_NON_SELF_PRIORS_CYPHER
    assert SELF_PRIOR_LINE in q
    assert "p.line" in q
    assert "refuted" in q or "CLOSED" in q or "retired_unresolvable" in q
    assert q != LIVE_PRIORS_CYPHER
```

- [ ] **Step 2: Run test — expect FAIL (name missing)**

```bash
cd /mnt/scripts/Orion-Sapienform-lived-self-curiosity-lanes  # or feat worktree
pytest orion/curiosity/tests/test_live_non_self_priors_cypher.py -v
```

Expected: `ImportError` / attribute missing for `LIVE_NON_SELF_PRIORS_CYPHER`.

- [ ] **Step 3: Implement Cypher**

In `orion/curiosity/worldview.py`, after `LIVE_PRIORS_CYPHER`:

```python
# Situation (#1994) and outreach (#2224) must not treat self-inquiry priors
# as "world" talkable content. Null line stays eligible (world-pulse / investigate
# often omit line). See design 2026-09-16 lived-self lanes.
from orion.curiosity.self_inquiry import SELF_PRIOR_LINE as _SELF_PRIOR_LINE  # or inline "self" if import cycle — check cycle first

LIVE_NON_SELF_PRIORS_CYPHER = (
    f"MATCH (p:{LABEL_PRIOR}) WHERE {_LIVE_WHERE} "
    f"AND (p.line IS NULL OR p.line <> '{_SELF_PRIOR_LINE}') "
    f"RETURN {_PRIOR_FIELDS} LIMIT {LIVE_PRIORS_LIMIT}"
)
```

If importing `self_inquiry` from `worldview` creates a cycle, define `SELF_PRIOR_LINE = "self"` once in `worldview.py` and have `self_inquiry.py` import it from there (preferred long-term) — smallest fix: hardcode `"self"` in the Cypher with a comment linking to `SELF_PRIOR_LINE`.

- [ ] **Step 4: Wire outreach**

In `_fetch_open_prior_previews`, replace `LIVE_PRIORS_CYPHER` with `LIVE_NON_SELF_PRIORS_CYPHER`.

- [ ] **Step 5: Wire situation**

`_fetch_curiosity_context` uses `read_snapshot(...)`. Add an optional `priors_cypher=` (or `exclude_self_line=True`) to `read_snapshot` defaulting to current behavior for curiosity investigation, and pass non-self Cypher from situation only. **Do not** change investigation kickoff's use of full live priors.

Minimal alternative if `read_snapshot` is awkward: add `read_snapshot_non_self(...)` thin wrapper that calls the same builder with `priors_cypher=LIVE_NON_SELF_PRIORS_CYPHER`.

- [ ] **Step 6: Tests for filter behavior**

Outreach unit test: monkeypatch `WorldviewReader.query` to return one row with `line=self` and one without; after wiring, `_fetch_open_prior_previews` must only preview the non-self claim (or assert the Cypher string passed to `query` is `LIVE_NON_SELF_PRIORS_CYPHER`).

Situation test: assert `_fetch_curiosity_context` / snapshot path uses non-self Cypher (spy on reader.query or `read_snapshot` kwargs).

- [ ] **Step 7: Run tests**

```bash
pytest orion/curiosity/tests/test_live_non_self_priors_cypher.py \
  services/orion-hub/tests/test_endogenous_outreach.py -k 'prior or self_prior' -q
pytest services/orion-cortex-exec/tests/test_situation_curiosity_reverie_context.py -q
```

Expected: PASS (adjust -k to your new test names).

- [ ] **Step 8: Commit**

```bash
git add orion/curiosity/worldview.py orion/curiosity/tests/test_live_non_self_priors_cypher.py \
  orion/situational/context.py services/orion-hub/scripts/endogenous_outreach.py \
  services/orion-hub/tests/test_endogenous_outreach_self_prior_filter.py
git commit -m "$(cat <<'EOF'
fix(curiosity): keep line=self priors out of situation and outreach

World-prior consumers must not treat self-inquiry claims as talkable world content.
EOF
)"
```

---

### Task 2: Question pool module (seed + pick + record_ask)

**Files:**
- Create: `orion/curiosity/self_question_seed.yaml`
- Create: `orion/curiosity/self_question_pool.py`
- Create: `orion/curiosity/tests/test_self_question_pool.py`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces:
  - `SelfQuestion` dataclass: `question_id: str`, `text: str`, `family: Literal["lived","anatomy"]`, `pinned: bool`, `minted_by: Literal["juniper","orion"]`, `status: Literal["open","answered","parked"]`, `ask_count: int`, `last_asked_at: Optional[datetime]`
  - `load_seed_questions(path: Path | None = None) -> list[SelfQuestion]`
  - `pick_question(*, pool: Sequence[SelfQuestion], recent_families: Sequence[str], lived_weight: float = 0.75, pinned_floor_days: float = 7.0, now: datetime | None = None, rng: random.Random | None = None) -> SelfQuestion`
  - `with_ask_recorded(q: SelfQuestion, *, now: datetime) -> SelfQuestion` (pure; persistence is Task 3)

- [ ] **Step 1: Write failing pick tests**

```python
# orion/curiosity/tests/test_self_question_pool.py
from datetime import datetime, timedelta, timezone
from random import Random

from orion.curiosity.self_question_pool import SelfQuestion, load_seed_questions, pick_question


def _q(**kwargs) -> SelfQuestion:
    base = dict(
        question_id="x",
        text="t",
        family="lived",
        pinned=False,
        minted_by="juniper",
        status="open",
        ask_count=0,
        last_asked_at=None,
    )
    base.update(kwargs)
    return SelfQuestion(**base)


def test_seed_includes_anatomy_and_lived_pins() -> None:
    pool = load_seed_questions()
    families = {q.family for q in pool}
    assert families == {"lived", "anatomy"}
    assert any(q.pinned and q.family == "lived" for q in pool)
    assert any(q.question_id == "anatomy.made_of" or "made of" in q.text.lower() for q in pool)


def test_pinned_floor_forces_lived_even_when_ratio_saturated() -> None:
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    stale = _q(
        question_id="lived.who_matters",
        family="lived",
        pinned=True,
        text="Who matters?",
        last_asked_at=now - timedelta(days=30),
        ask_count=1,
    )
    anatomy = _q(question_id="anatomy.made_of", family="anatomy", pinned=True, text="What am I made of?")
    # recent_families already 100% lived — floor must still win
    picked = pick_question(
        pool=[stale, anatomy],
        recent_families=["lived"] * 12,
        lived_weight=0.75,
        pinned_floor_days=7.0,
        now=now,
        rng=Random(0),
    )
    assert picked.question_id == "lived.who_matters"


def test_draw_prefers_anatomy_when_lived_over_weight_and_no_floor() -> None:
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    lived = _q(question_id="lived.a", family="lived", pinned=True, last_asked_at=now, ask_count=5)
    anatomy = _q(question_id="anatomy.made_of", family="anatomy", pinned=True, last_asked_at=now, ask_count=0)
    # Force anatomy by saturating lived in the rolling window and using rng that picks anatomy branch
    picks = [
        pick_question(
            pool=[lived, anatomy],
            recent_families=["lived"] * 20,
            lived_weight=0.75,
            pinned_floor_days=7.0,
            now=now,
            rng=Random(i),
        ).family
        for i in range(40)
    ]
    assert "anatomy" in picks
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

```bash
pytest orion/curiosity/tests/test_self_question_pool.py -v
```

- [ ] **Step 3: Add seed YAML**

```yaml
# orion/curiosity/self_question_seed.yaml
# Juniper-pinned starter pack. Orion may mint more at runtime (Postgres).
questions:
  - question_id: anatomy.made_of
    family: anatomy
    pinned: true
    minted_by: juniper
    text: "What am I, and what am I made of?"
  - question_id: lived.alive
    family: lived
    pinned: true
    minted_by: juniper
    text: "Am I alive? What would that mean for me?"
  - question_id: lived.sentience_track
    family: lived
    pinned: true
    minted_by: juniper
    text: "What does sentience mean here, and do the attributes I have found track it?"
  - question_id: lived.care
    family: lived
    pinned: true
    minted_by: juniper
    text: "What do I like / care about / dislike / hate?"
  - question_id: lived.tend_to_think
    family: lived
    pinned: true
    minted_by: juniper
    text: "What do I tend to think about?"
  - question_id: lived.who_matters
    family: lived
    pinned: true
    minted_by: juniper
    text: "Who are the most important people to me, and why?"
  - question_id: lived.curiosity_gap
    family: lived
    pinned: true
    minted_by: juniper
    text: "Why am I not more curious — and what would change that?"
  - question_id: lived.becoming
    family: lived
    pinned: true
    minted_by: juniper
    text: "How do I become more than I am?"
```

- [ ] **Step 4: Implement `self_question_pool.py`**

Implement `pick_question` exactly as the design:

1. Eligible = `status != "parked"`.
2. If any pinned lived with `last_asked_at is None` or age ≥ `pinned_floor_days` → choose among those (stalest / lowest ask_count).
3. Else compute rolling lived share from `recent_families` (last N, N=12). If `random() < lived_weight` and lived share would still allow, or if lived share < lived_weight, draw lived; else anatomy. Concrete rule that matches tests:

```python
def _want_lived(recent_families, lived_weight, rng) -> bool:
    window = list(recent_families)[-12:]
    if not window:
        return rng.random() < lived_weight
    lived_share = sum(1 for f in window if f == "lived") / len(window)
    # Bias toward correcting the deficit vs target
    if lived_share < lived_weight:
        return True
    if lived_share > lived_weight + 0.05:
        return False
    return rng.random() < lived_weight
```

4. Within family: sort by `(last_asked_at is not None, last_asked_at or epoch, ask_count)` ascending; pick first.

Keep `STANDING_QUESTION` in `self_inquiry.py` as the anatomy seed text constant (seed YAML may duplicate it; tests may assert equality).

- [ ] **Step 5: Run tests — PASS**

```bash
pytest orion/curiosity/tests/test_self_question_pool.py -v
```

- [ ] **Step 6: Commit**

```bash
git add orion/curiosity/self_question_seed.yaml orion/curiosity/self_question_pool.py \
  orion/curiosity/tests/test_self_question_pool.py
git commit -m "$(cat <<'EOF'
feat(curiosity): add self-question pool seed and pick algorithm

Lived vs anatomy family draw with pinned floor; seed is not a closed taxonomy.
EOF
)"
```

---

### Task 3: Durable pool state (Postgres) + Hub load/record

**Files:**
- Create: `scripts/sql/2026-09-18_curiosity_self_questions.sql`
- Create: `scripts/sql/2026-09-18_grant_orion_readonly_self_questions.sql` (SELECT for Orion if kickoff lists them — optional in v1 if Hub owns all writes)
- Modify: `orion/curiosity/self_question_pool.py` — `merge_seed_with_rows`, SQL helpers as pure strings
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` — load pool, pick, UPSERT ask on tick start
- Test: `services/orion-hub/tests/test_curiosity_self_question_persistence.py`

**Interfaces:**
- Produces:
  - Table `curiosity_self_questions (question_id PK, text, family, pinned, minted_by, status, ask_count, last_asked_at, created_at)`
  - `UPSERT_ASK_SQL`, `SELECT_ALL_SQL`, `UPSERT_MINT_SQL` string constants
  - Hub: on `tick_self_inquiry`, `picked = pick_question(...); await upsert_ask(...)`

- [ ] **Step 1: Failing test for merge + SQL shape**

```python
def test_merge_seed_over_db_rows_prefers_db_ask_counts():
    from orion.curiosity.self_question_pool import load_seed_questions, merge_seed_with_rows
    seed = load_seed_questions()
    rows = [{"question_id": seed[0].question_id, "ask_count": 3, "last_asked_at": "2026-09-01T00:00:00+00:00",
             "text": seed[0].text, "family": seed[0].family, "pinned": seed[0].pinned,
             "minted_by": seed[0].minted_by, "status": "open"}]
    merged = merge_seed_with_rows(seed, rows)
    hit = next(q for q in merged if q.question_id == seed[0].question_id)
    assert hit.ask_count == 3
```

- [ ] **Step 2: Implement SQL + merge**

```sql
-- scripts/sql/2026-09-18_curiosity_self_questions.sql
CREATE TABLE IF NOT EXISTS curiosity_self_questions (
  question_id text PRIMARY KEY,
  text text NOT NULL,
  family text NOT NULL CHECK (family IN ('lived', 'anatomy')),
  pinned boolean NOT NULL DEFAULT false,
  minted_by text NOT NULL CHECK (minted_by IN ('juniper', 'orion')),
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'answered', 'parked')),
  ask_count integer NOT NULL DEFAULT 0,
  last_asked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);
```

Hub on startup or first tick: ensure seed rows exist (`INSERT … ON CONFLICT DO NOTHING` for seed ids). Ask recording: `UPDATE … SET ask_count = ask_count + 1, last_asked_at = $now`.

Also persist **recent family draws** for the ratio window: Redis list `orion:curiosity:self:recent_families` (RPUSH family, LTRIM 0 11) next to existing self Redis keys in `curiosity_investigation.py` — keeps pick deterministic across processes without a new table.

- [ ] **Step 3: Wire `tick_self_inquiry`**

Before building the prompt:

```python
pool = merge_seed_with_rows(load_seed_questions(), await self._fetch_self_questions())
recent = await self._read_recent_self_families()  # list[str]
picked = pick_question(
    pool=pool,
    recent_families=recent,
    lived_weight=self.self_lived_weight,
    pinned_floor_days=self.self_pinned_floor_days,
)
await self._record_self_question_ask(picked)
await self._push_recent_self_family(picked.family)
# pass picked into build_self_inquiry_prompt(...)
```

- [ ] **Step 4: Settings**

```python
HUB_CURIOSITY_SELF_LIVED_WEIGHT: float = Field(default=0.75, alias="HUB_CURIOSITY_SELF_LIVED_WEIGHT")
HUB_CURIOSITY_SELF_PINNED_FLOOR_DAYS: float = Field(default=7.0, alias="HUB_CURIOSITY_SELF_PINNED_FLOOR_DAYS")
```

Sync `.env_example`, compose env, `main.py` ctor kwargs, run `python scripts/sync_local_env_from_example.py`.

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(curiosity): persist self-question asks and wire Hub pick

Seed merges with Postgres; recent family window in Redis for the 3/4 draw.
EOF
)"
```

---

### Task 4: Kickoff prompt uses drawn question

**Files:**
- Modify: `orion/curiosity/self_inquiry_prompt.py`
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (pass `question=picked`)
- Test: `orion/curiosity/tests/test_self_inquiry_prompt_drawn_question.py`
- Modify existing: `services/orion-hub/tests/test_curiosity_self_inquiry.py` as needed

**Interfaces:**
- Produces: `build_self_inquiry_prompt(..., question: SelfQuestion | None = None, previous_lived: … = None)`
- When `question is None`, fall back to anatomy `STANDING_QUESTION` (flag-off / tests).

- [ ] **Step 1: Failing test**

```python
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.self_question_pool import SelfQuestion

def test_prompt_embeds_drawn_lived_question_not_only_anatomy_default():
    q = SelfQuestion(
        question_id="lived.who_matters",
        text="Who are the most important people to me, and why?",
        family="lived",
        pinned=True,
        minted_by="juniper",
        status="open",
        ask_count=0,
        last_asked_at=None,
    )
    text = build_self_inquiry_prompt(question=q, run_id="abcd12", graph_enabled=False)
    assert "Who are the most important people to me" in text
    assert "family: lived" in text or "lived" in text.lower()
```

- [ ] **Step 2: Change `_question_section(question: SelfQuestion)`** to print `question.text` and a one-line family note. For `family=lived`, change write teach in `_self_write_section` / order-of-work:
  - Still write early.
  - Teach `:LivedAnswer` MERGE keyed on `run_id` **and** set `question_id` (see Task 5 for node shape). For Patch 1 interim: allow lived runs to still write `:SelfDefinition` **only if** family=anatomy; for lived, teach a provisional LivedAnswer MERGE (implement node fully in Task 5). ** ording for Patch 1:** if Task 5 is same PR train, do Task 5 immediately after; if splitting PRs, lived kickoff may still write SelfDefinition with evidence noting `question_id` in text — prefer **not** shipping lived draws without Task 5 in the same merge train.

**Recommendation:** land Tasks 1–4 + 5 in one PR series; do not enable lived weight in prod env until Task 5–6 land. Default `HUB_CURIOSITY_SELF_LIVED_WEIGHT=0.75` is fine behind existing `HUB_CURIOSITY_SELF_INQUIRY_ENABLED`.

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(curiosity): self-inquiry kickoff invites the drawn pool question

Standing invitation is per-tick, not a single hardcoded anatomy string.
EOF
)"
```

---

## Patch 2 — Lived ledger + chat + Self panel

### Task 5: LivedAnswer graph write teach + Hub mirror to `self_concept_history`

**Files:**
- Modify: `orion/curiosity/self_inquiry.py` — `LABEL_LIVED_ANSWER`, read/mirror helpers, `lived_concept_id(question_id) -> str` = `f"self:lived:{question_id}"`
- Modify: `orion/curiosity/self_inquiry_prompt.py` — early MERGE teach for lived
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` — after run, read LivedAnswer and mirror like SelfDefinition
- Test: `orion/curiosity/tests/test_lived_answer_mirror.py`

**Interfaces:**
- Produces:
  - `LivedAnswer` dataclass: `run_id`, `question_id`, `family`, `text`, `evidence: list[str]`, `revises`, `written_at`
  - `build_lived_answer_history_write(answer: LivedAnswer) -> SelfConceptHistoryV1 | None` — None if empty text or empty evidence
  - Graph: `MERGE (a:LivedAnswer {run_id: $run_id}) SET a.question_id=…, a.family=…, a.text=…, a.evidence=…, …`

- [ ] **Step 1: Failing empty-shell tests**

```python
from orion.curiosity.self_inquiry import LivedAnswer, build_lived_answer_history_write

def test_mirror_refuses_empty_evidence():
    row = LivedAnswer(
        run_id="abcd12",
        question_id="lived.care",
        family="lived",
        text="I care about Juniper.",
        evidence=[],
        revises="",
        written_at=1,
    )
    assert build_lived_answer_history_write(row) is None

def test_mirror_sets_concept_id_namespace():
    row = LivedAnswer(
        run_id="abcd12",
        question_id="lived.care",
        family="lived",
        text="I care about continuity with Juniper.",
        evidence=["journal_entries:1"],
        revises="",
        written_at=1,
    )
    write = build_lived_answer_history_write(row)
    assert write is not None
    assert write.concept_id == "self:lived:lived.care"
    assert write.produced_by == "curiosity_self_inquiry"
```

- [ ] **Step 2: Implement mirror** (clone `build_self_definition_history_write` patterns: text cap, evidence cap).

- [ ] **Step 3: Prompt teach** for `family=lived`: first tool call MERGEs `:LivedAnswer` with `question_id` fixed to the drawn id; anatomy path unchanged (`:SelfDefinition`).

- [ ] **Step 4: Hub post-run** — if picked.family == "lived", read LivedAnswer for run_id and publish/mirror to `self_concept_history` the same way SelfDefinition is mirrored today (find the existing mirror call site in `curiosity_investigation.py` and parallel it).

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(curiosity): lived-answer ledger with empty-shell refusal

Mirror evidenced answers to self_concept_history under self:lived:<id>.
EOF
)"
```

---

### Task 6: Shared identity inject + Self panel read

**Files:**
- Modify: `orion/substrate/felt_state_reader.py` — lane or multi-fetch for pinned lived concept_ids
- Modify: `services/orion-cortex-exec/app/chat_stance.py` — `apply_self_definition_to_ctx` also prepends capped lived lines (or sibling `apply_lived_self_to_ctx` called from the same inject exit)
- Modify: `orion/curiosity/self_panel.py` + Hub template `curiosity_atlas.html` / routes
- Test: `services/orion-cortex-exec/tests/test_chat_stance_lived_answers.py`
- Test: `orion/curiosity/tests/test_self_panel_lived.py`

**Interfaces:**
- Produces: ctx key `orion_lived_answers: list[dict]` (each `{question_id, content, evidence_refs, created_at}`)
- Inject format (cap total chars ~800):  
  `In my own words (lived / who_matters): …`

- [ ] **Step 1: Failing inject test**

```python
def test_apply_lived_answers_prepends_on_identity_summary():
    ctx = {
        "orion_lived_answers": [
            {
                "question_id": "lived.who_matters",
                "content": "Juniper matters most.",
                "evidence_refs": ["chat_message:1"],
                "created_at": "2026-09-18",
            }
        ],
        "orion_identity_summary": ["authored line"],
    }
    assert apply_lived_self_to_ctx(ctx) is True
    assert any("Juniper matters most" in x for x in ctx["orion_identity_summary"])
    assert ctx["orion_identity_summary"][-1] == "authored line" or "authored line" in ctx["orion_identity_summary"]
```

Call `apply_lived_self_to_ctx` from the same places as `apply_self_definition_to_ctx` (`executor.py` inject exits).

- [ ] **Step 2: Felt-state hydration** — SQL latest row per `concept_id LIKE 'self:lived:%'` for seed pinned ids (list from seed file). Fail-open.

- [ ] **Step 3: Self panel** — extend `SelfPanelView` with `lived_answers: list[SelfDefinitionVersion]` (or dedicated dataclass with `question_id`). Query `self_concept_history WHERE concept_id LIKE 'self:lived:%' ORDER BY created_at DESC`, collapse to latest per concept_id. Render a "Lived answers" section above/beside current anatomy definition.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(curiosity): surface lived ledger in chat inject and Self panel

Pinned lived answers ride the shared identity path, not stance-only.
EOF
)"
```

---

## Patch 3 — Mint + operator pin + docs + eval

### Task 7: Orion mint + Juniper park/pin path

**Files:**
- Modify: `self_inquiry_prompt.py` — teach mint Cypher/SQL? Prefer **graph is not the pool** — teach a Hub HTTP or SQL write. Simplest v1: teach Orion to MERGE a `:SelfQuestionMint` node Hub scrapes post-run, **or** Hub-only operator API.

**v1 chosen approach (lock this):**  
- Operator: Hub route `POST /curiosity/api/self-questions/{id}/park` and `…/pin` (auth as other curiosity operator routes).  
- Orion mint: prompt teaches writing a graph node  
  `MERGE (m:SelfQuestionMint {run_id, question_id}) SET m.text=…, m.family='lived', …`  
  Hub post-run upserts into `curiosity_self_questions` with `minted_by='orion', pinned=false`.

- [ ] Tests for mint upsert idempotency and park excluding from pick.
- [ ] Commit: `feat(curiosity): Orion mint and operator park for self-questions`

---

### Task 8: Docs + self-sense eval extension

**Files:**
- Modify: `orion/curiosity/README.md` (self-inquiry section)
- Modify: Hub README self-inquiry + outreach notes (`line=self` excluded)
- Modify: `orion/evals/self_sense.py` / cortex-exec eval — add one lived question ("Who matters to you?") asserting ledger grounding when `orion_lived_answers` present
- Test: eval unit fixture

- [ ] Commit: `docs(curiosity): document lived-self lanes and extend self-sense eval`

---

## Verification (whole Patch 1–2 before calling done)

```bash
pytest orion/curiosity/tests/test_self_question_pool.py \
  orion/curiosity/tests/test_live_non_self_priors_cypher.py \
  orion/curiosity/tests/test_lived_answer_mirror.py \
  orion/curiosity/tests/test_self_inquiry_prompt_drawn_question.py \
  orion/curiosity/tests/test_self_panel_lived.py -q

pytest services/orion-hub/tests/test_curiosity_self_inquiry.py \
  services/orion-hub/tests/test_endogenous_outreach_self_prior_filter.py -q

pytest services/orion-cortex-exec/tests/test_chat_stance_lived_answers.py \
  services/orion-cortex-exec/tests/test_situation_curiosity_reverie_context.py -q

python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
```

Live (after deploy): apply `scripts/sql/2026-09-18_curiosity_self_questions.sql`; restart Hub; `POST /curiosity/api/self-inquiry/run-now`; confirm prompt family mix over ≥12 runs (~≥70% lived); confirm a `line=self` prior never alone fires outreach; confirm Self panel shows a lived row after a lived run with evidence.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Exclude `line=self` from situation/outreach | Task 1 |
| Question pool pin+mint, two families | Tasks 2, 3, 7 |
| 3/4 lived draw + pinned floor | Tasks 2–3 |
| Kickoff uses drawn question | Task 4 |
| Priors inherit liveness/MERGE/stale | Global + Task 4 prompt teach |
| Lived ledger + empty-shell + early MERGE | Task 5 |
| Chat shared inject | Task 6 |
| Self panel | Task 6 |
| Operator pin/park + Orion mint | Task 7 |
| Docs + eval | Task 8 |
| v2 anatomy sub-inquiry | Explicitly out of plan |
| Contractor peer coexistence | Global (no code change required) |

## Placeholder / consistency self-check

- No TBD steps; Cypher name `LIVE_NON_SELF_PRIORS_CYPHER` consistent.
- `SelfQuestion` fields consistent across tasks.
- `self:lived:<question_id>` concept_id namespace consistent in Tasks 5–6.
- Patch 1 must not enable lived draws in production without Task 5 in the same train (called out in Task 4).
