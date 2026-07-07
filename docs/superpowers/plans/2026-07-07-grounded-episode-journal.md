# Grounded Autonomy Episode Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread real fetched-article evidence, the driving curiosity gap, and a deterministic salience score into the autonomy episode journal's narrative seed so the composed journal names *why* it fetched and *what* it read.

**Architecture:** Three additive, defaulted seams inside the `orion/autonomy/*` package: (1) stop dropping `title`/`description` in the Firecrawl backend and carry articles + query + salience into `ActionOutcomeRefV1`; (2) a new pure deterministic salience scorer (`orion/autonomy/salience.py`); (3) a structured narrative-seed builder in `policy_act.py` that stops discarding curiosity signals. No bus channels, no new events, no SQL projection change. The whole path stays gated by `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED`.

**Tech Stack:** Python 3, Pydantic v2 (`ConfigDict(extra="forbid")`), `pytest` + `pytest-asyncio`, stdlib `re`/`urllib`.

**Design source:** `docs/superpowers/specs/2026-07-07-grounded-episode-journal-design.md`

---

## Context for the implementer (read before starting)

You are working in `Orion-Sapienform`, a services-first "digital mind" repo. This plan touches only the shared `orion/autonomy/*` package (importable library code, not a `services/<name>/` service), so there is **no `.env_example`, Docker, or bus/schema-registry change** in scope. Verified facts:

- `ActionOutcomeRefV1` is **NOT** in `orion/schemas/registry.py` (grep returned no matches). Do **not** add a registry entry.
- `ActionOutcomeEmitV1` (the flat SQL projection in `orion/autonomy/models.py`) is **out of scope** — leave `from_outcome`/`to_outcome` unchanged. New fields ride only in the local JSON outcome store and the in-process `SubstrateActResultV1.fetch_outcome`.
- All autonomy models use `model_config = ConfigDict(extra="forbid", populate_by_name=True)`. New fields must be defaulted so existing persisted JSON still validates.
- Async tests use `@pytest.mark.asyncio`. Backend stubs use `unittest.mock.AsyncMock`.
- The outcome store path is overridable in tests via `monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", ...)`.

### Key existing signatures you will build on

- `orion/autonomy/fetch_backends.py::firecrawl_search_backend(query, *, max_articles, api_key) -> dict` — currently returns `{"success": bool, "urls": [str]}`.
- `orion/autonomy/episode_fetch.py::EpisodeFetchRequest` — frozen dataclass: `subject, goal_artifact_id, spawned_correlation_id, query, max_articles=2`.
- `orion/autonomy/episode_fetch.py::execute_readonly_fetch(req, *, fetch_backend) -> ActionOutcomeRefV1` — builds outcome, calls `append_action_outcome`.
- `orion/autonomy/policy_act.py::_GAP_SIGNAL = "world_coverage_gap"`, `curiosity_strength_from_signals(signals) -> float`, `build_readonly_fetch_query(signals) -> str`, `maybe_execute_readonly_fetch_after_goal(...)`, `maybe_compose_autonomy_episode_after_fetch(...)` (contains `del curiosity_signals` at the top and builds `narrative_seed = f"fetch outcome: {...}"`).
- `orion/core/schemas/frontier_curiosity.py::FrontierInvocationSignalV1` — has `signal_type`, `focal_node_refs: list[str]`, `signal_strength: float`.

### Test commands (run from repo root `/mnt/scripts/Orion-Sapienform`)

```bash
pytest orion/autonomy/tests/test_salience.py -q
pytest orion/autonomy/tests/test_fetch_backends.py -q
pytest orion/autonomy/tests/test_episode_fetch.py -q
pytest orion/autonomy/tests/test_policy_act.py -q
pytest orion/autonomy/tests -q   # full autonomy gate lane
```

---

## File structure (what each task creates/modifies)

- `orion/autonomy/salience.py` — **new**, pure deterministic scorer. One responsibility: gap-term extraction + article salience.
- `orion/autonomy/models.py` — add `FetchedArticleRefV1`; extend `ActionOutcomeRefV1` with `query`, `articles`, `salience`.
- `orion/autonomy/fetch_backends.py` — capture `title`/`description`, return `articles` alongside `urls`.
- `orion/autonomy/episode_fetch.py` — `EpisodeFetchRequest.gap_terms`; parse articles + score + carry `query`/aggregate salience into the outcome.
- `orion/autonomy/policy_act.py` — set `gap_terms` on the request; add `build_episode_narrative_seed`; stop discarding curiosity signals.
- `orion/autonomy/tests/test_salience.py` — **new** unit tests.
- `orion/autonomy/tests/test_fetch_backends.py` — add article-parsing test.
- `orion/autonomy/tests/test_episode_fetch.py` — assert new outcome fields.
- `orion/autonomy/tests/test_policy_act.py` — seed-builder tests + fix the now-stale `startswith("fetch outcome:")` assertion.

Build order matters: salience (no deps) → models → fetch_backends → episode_fetch (uses salience + models) → policy_act (uses salience + models + episode_fetch).

---

### Task 1: Deterministic salience scorer

**Files:**
- Create: `orion/autonomy/salience.py`
- Test: `orion/autonomy/tests/test_salience.py`

Design decision: the spec shows `gap_terms_from_signals(signals) -> set[str]` with a note "fall back to query tokens if no section refs." To keep it a pure function, this plan adds an optional `fallback_query: str = ""` parameter (the caller in `policy_act.py` already holds the query). This is the reasonable, testable choice.

- [ ] **Step 1: Write the failing tests**

Create `orion/autonomy/tests/test_salience.py`:

```python
from __future__ import annotations

from orion.autonomy.salience import gap_terms_from_signals, score_article_salience
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1


def _gap_signal(section: str = "section:hardware_compute_gpu") -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=[section],
        signal_strength=0.65,
        evidence_summary="gap",
        confidence=0.65,
    )


def test_gap_terms_from_section_refs() -> None:
    terms = gap_terms_from_signals([_gap_signal()])
    assert terms == {"hardware", "compute", "gpu"}


def test_gap_terms_ignores_non_gap_signals() -> None:
    sig = _gap_signal()
    other = sig.model_copy(update={"signal_type": "curiosity_candidate"})
    assert gap_terms_from_signals([other]) == set()


def test_gap_terms_falls_back_to_query_when_no_section() -> None:
    sig = _gap_signal(section="node:not_a_section")
    terms = gap_terms_from_signals([sig], fallback_query="GPU supply chain news")
    assert terms == {"gpu", "supply", "chain", "news"}


def test_score_article_salience_fraction() -> None:
    gap = {"hardware", "compute", "gpu"}
    # 2 of 3 gap terms present.
    score = score_article_salience("New GPU compute cluster launches", gap)
    assert abs(score - (2 / 3)) < 1e-9


def test_score_article_salience_empty_gap_is_zero() -> None:
    assert score_article_salience("anything at all", set()) == 0.0


def test_score_article_salience_empty_text_is_zero() -> None:
    assert score_article_salience("", {"gpu"}) == 0.0


def test_score_article_salience_clamped_to_one() -> None:
    gap = {"gpu"}
    assert score_article_salience("gpu gpu gpu", gap) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest orion/autonomy/tests/test_salience.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'orion.autonomy.salience'`.

- [ ] **Step 3: Write the implementation**

Create `orion/autonomy/salience.py`:

```python
"""Deterministic term-overlap salience scorer for readonly-fetch articles.

Narrow deterministic sensor (per repo no-regex-swamp note): a single scoring
function over gap terms vs article text. Not a cognition architecture.
"""

from __future__ import annotations

import re
from typing import Sequence

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1

_GAP_SIGNAL = "world_coverage_gap"
_SECTION_PREFIX = "section:"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def gap_terms_from_signals(
    signals: Sequence[FrontierInvocationSignalV1],
    *,
    fallback_query: str = "",
) -> set[str]:
    """Union of tokens from `section:` focal refs across world_coverage_gap signals.

    Falls back to query tokens only when no section-derived terms are found.
    """
    terms: set[str] = set()
    for sig in signals:
        if getattr(sig, "signal_type", None) != _GAP_SIGNAL:
            continue
        for ref in getattr(sig, "focal_node_refs", None) or []:
            section = str(ref or "").strip()
            if section.startswith(_SECTION_PREFIX):
                label = section[len(_SECTION_PREFIX):].replace("_", " ")
                terms |= _tokenize(label)
    if not terms and fallback_query:
        terms |= _tokenize(fallback_query)
    return terms


def score_article_salience(article_text: str, gap_terms: set[str]) -> float:
    """Fraction of gap terms present in the article text, clamped to [0, 1]."""
    if not gap_terms:
        return 0.0
    tokens = _tokenize(article_text)
    if not tokens:
        return 0.0
    overlap = gap_terms & tokens
    score = len(overlap) / len(gap_terms)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest orion/autonomy/tests/test_salience.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/salience.py orion/autonomy/tests/test_salience.py
git commit -m "feat(autonomy): deterministic gap-term article salience scorer"
```

---

### Task 2: Article + outcome model fields

**Files:**
- Modify: `orion/autonomy/models.py` (add `FetchedArticleRefV1` after imports/near `ActionOutcomeRefV1`; extend `ActionOutcomeRefV1` at lines 153-161)
- Test: `orion/autonomy/tests/test_substrate_act_models.py` (add cases)

Leave `ActionOutcomeEmitV1` (lines 164-201) untouched — SQL projection stays flat.

- [ ] **Step 1: Write the failing tests**

Add to `orion/autonomy/tests/test_substrate_act_models.py` (append at end of file; add imports at top if missing):

```python
def test_fetched_article_ref_defaults() -> None:
    from orion.autonomy.models import FetchedArticleRefV1

    art = FetchedArticleRefV1(url="https://example.com/a")
    assert art.title == ""
    assert art.description == ""
    assert art.salience == 0.0


def test_action_outcome_ref_carries_articles_and_query() -> None:
    from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1

    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        query="gpu recent news coverage",
        articles=[FetchedArticleRefV1(url="https://example.com/a", title="A", salience=0.5)],
        salience=0.5,
    )
    assert outcome.query == "gpu recent news coverage"
    assert outcome.articles[0].title == "A"
    assert outcome.salience == 0.5


def test_action_outcome_ref_backward_compatible_without_new_fields() -> None:
    from orion.autonomy.models import ActionOutcomeRefV1

    # Persisted JSON from before this patch (no query/articles/salience).
    outcome = ActionOutcomeRefV1.model_validate(
        {
            "action_id": "fetch-old",
            "kind": "web.fetch.readonly",
            "summary": "fetched 2 article(s)",
            "success": True,
            "surprise": 0.0,
        }
    )
    assert outcome.query is None
    assert outcome.articles == []
    assert outcome.salience == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest orion/autonomy/tests/test_substrate_act_models.py -q`
Expected: FAIL — `ImportError: cannot import name 'FetchedArticleRefV1'` (and `TypeError`/validation error for the new `ActionOutcomeRefV1` kwargs).

- [ ] **Step 3: Write the implementation**

In `orion/autonomy/models.py`, replace the existing `ActionOutcomeRefV1` class (lines 153-161):

```python
class ActionOutcomeRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action_id: str
    kind: str
    summary: str
    success: bool | None = None
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_at: datetime | None = None
```

with the new nested model **immediately before** it, plus the extended fields:

```python
class FetchedArticleRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    url: str
    title: str = ""
    description: str = ""
    salience: float = Field(default=0.0, ge=0.0, le=1.0)


class ActionOutcomeRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action_id: str
    kind: str
    summary: str
    success: bool | None = None
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_at: datetime | None = None
    query: str | None = None
    articles: list[FetchedArticleRefV1] = Field(default_factory=list)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
```

Do **not** modify `ActionOutcomeEmitV1.from_outcome` / `to_outcome` — they intentionally project only the flat SQL columns.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest orion/autonomy/tests/test_substrate_act_models.py -q`
Expected: PASS.

- [ ] **Step 5: Verify the SQL projection round-trip still passes (no regression)**

Run: `pytest orion/autonomy/tests/test_action_outcomes_sql.py services/orion-sql-writer/tests/test_action_outcome_sql_shape.py -q`
Expected: PASS (confirms `ActionOutcomeEmitV1` flat shape is untouched).

- [ ] **Step 6: Commit**

```bash
git add orion/autonomy/models.py orion/autonomy/tests/test_substrate_act_models.py
git commit -m "feat(autonomy): add FetchedArticleRefV1 and article/query/salience to ActionOutcomeRefV1"
```

---

### Task 3: Firecrawl backend keeps title/description

**Files:**
- Modify: `orion/autonomy/fetch_backends.py:38-47` (the `_call` result assembly)
- Test: `orion/autonomy/tests/test_fetch_backends.py`

- [ ] **Step 1: Write the failing test**

Add to `orion/autonomy/tests/test_fetch_backends.py` (after the existing `test_firecrawl_search_backend_parses_urls`):

```python
@pytest.mark.asyncio
async def test_firecrawl_search_backend_parses_articles() -> None:
    response_body = json.dumps(
        {
            "success": True,
            "data": [
                {"url": "https://example.com/a", "title": "A", "description": "desc a"},
                {"url": "https://example.com/b", "title": "B"},
            ],
        }
    )

    def _fake_urlopen(req, timeout=30.0):  # noqa: ARG001
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body.encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    with patch("orion.autonomy.fetch_backends.urllib.request.urlopen", side_effect=_fake_urlopen):
        result = await firecrawl_search_backend("gpu news", max_articles=2, api_key="test-key")

    assert result["success"] is True
    # urls preserved for back-compat
    assert result["urls"] == ["https://example.com/a", "https://example.com/b"]
    # articles carry title + description (missing description defaults to "")
    assert result["articles"] == [
        {"url": "https://example.com/a", "title": "A", "description": "desc a"},
        {"url": "https://example.com/b", "title": "B", "description": ""},
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest orion/autonomy/tests/test_fetch_backends.py::test_firecrawl_search_backend_parses_articles -q`
Expected: FAIL with `KeyError: 'articles'`.

- [ ] **Step 3: Write the implementation**

In `orion/autonomy/fetch_backends.py`, replace the article-assembly block (currently lines 38-47):

```python
        data = parsed.get("data") or []
        urls: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    url = item.get("url")
                    if url:
                        urls.append(str(url))
        success = bool(parsed.get("success")) and bool(urls)
        return {"success": success, "urls": urls}
```

with:

```python
        data = parsed.get("data") or []
        urls: list[str] = []
        articles: list[dict] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    url = item.get("url")
                    if url:
                        url_str = str(url)
                        urls.append(url_str)
                        articles.append(
                            {
                                "url": url_str,
                                "title": str(item.get("title") or ""),
                                "description": str(item.get("description") or ""),
                            }
                        )
        success = bool(parsed.get("success")) and bool(urls)
        return {"success": success, "urls": urls, "articles": articles}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest orion/autonomy/tests/test_fetch_backends.py -q`
Expected: PASS (existing URL/HTTP-error tests still pass; new article test passes).

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/fetch_backends.py orion/autonomy/tests/test_fetch_backends.py
git commit -m "feat(autonomy): firecrawl backend returns per-article title/description"
```

---

### Task 4: Fetch execution carries articles, query, and salience

**Files:**
- Modify: `orion/autonomy/episode_fetch.py` (imports; `EpisodeFetchRequest`; `execute_readonly_fetch`; add `_parse_articles` helper)
- Test: `orion/autonomy/tests/test_episode_fetch.py`

- [ ] **Step 1: Write the failing tests**

Add to `orion/autonomy/tests/test_episode_fetch.py` (append after existing tests; imports `AsyncMock`, `execute_readonly_fetch`, `EpisodeFetchRequest` are already at top):

```python
@pytest.mark.asyncio
async def test_execute_readonly_fetch_carries_articles_and_salience(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(
        return_value={
            "success": True,
            "urls": ["https://example.com/a", "https://example.com/b"],
            "articles": [
                {"url": "https://example.com/a", "title": "GPU compute news", "description": "hardware compute"},
                {"url": "https://example.com/b", "title": "Cooking", "description": "recipes"},
            ],
        }
    )
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="hardware compute gpu recent news coverage",
        gap_terms=("hardware", "compute", "gpu"),
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)

    assert outcome.query == "hardware compute gpu recent news coverage"
    assert len(outcome.articles) == 2
    assert outcome.articles[0].title == "GPU compute news"
    # article 0 covers all 3 gap terms -> salience 1.0; article 1 covers none -> 0.0
    assert outcome.articles[0].salience == 1.0
    assert outcome.articles[1].salience == 0.0
    # aggregate outcome salience is the max over articles
    assert outcome.salience == 1.0


@pytest.mark.asyncio
async def test_execute_readonly_fetch_backcompat_urls_only(tmp_path, monkeypatch) -> None:
    """Older backend returning only urls still yields article refs (salience 0)."""
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="gpu news",
        gap_terms=("gpu",),
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)
    assert [a.url for a in outcome.articles] == ["https://example.com/a"]
    assert outcome.articles[0].salience == 0.0
    assert outcome.salience == 0.0
    assert outcome.query == "gpu news"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest orion/autonomy/tests/test_episode_fetch.py -q`
Expected: FAIL — `EpisodeFetchRequest` has no `gap_terms` (`TypeError: __init__() got an unexpected keyword argument 'gap_terms'`).

- [ ] **Step 3: Write the implementation**

In `orion/autonomy/episode_fetch.py`, update the imports at the top:

```python
from orion.autonomy.action_outcomes import append_action_outcome
from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1
from orion.autonomy.salience import score_article_salience
```

Extend `EpisodeFetchRequest` (add the `gap_terms` field last so positional construction is unaffected):

```python
@dataclass(frozen=True)
class EpisodeFetchRequest:
    subject: str
    goal_artifact_id: str
    spawned_correlation_id: str
    query: str
    max_articles: int = 2
    gap_terms: tuple[str, ...] = ()
```

Add a parser helper (place it after `_build_summary`):

```python
def _parse_articles(result: dict, *, gap_terms: set[str]) -> list[FetchedArticleRefV1]:
    """Build scored article refs from a backend result.

    Prefers the rich `articles` list; falls back to bare `urls` for older backends.
    """
    parsed: list[FetchedArticleRefV1] = []
    raw = result.get("articles")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            title = str(item.get("title") or "")
            description = str(item.get("description") or "")
            salience = score_article_salience(f"{title} {description}", gap_terms)
            parsed.append(
                FetchedArticleRefV1(url=url, title=title, description=description, salience=salience)
            )
    if parsed:
        return parsed
    urls = result.get("urls")
    if isinstance(urls, list):
        for candidate in urls:
            url = str(candidate or "").strip()
            if url:
                parsed.append(FetchedArticleRefV1(url=url))
    return parsed
```

Update `execute_readonly_fetch` to score and carry the new fields. Replace the body from `success = False` through the `return outcome`:

```python
    action_id = f"fetch-{req.spawned_correlation_id}-{uuid.uuid4().hex[:8]}"
    observed_at = datetime.now(timezone.utc)
    success = False
    surprise = 1.0
    summary = "fetch failed"
    articles: list[FetchedArticleRefV1] = []
    aggregate_salience = 0.0

    try:
        result = await fetch_backend(req.query, max_articles=req.max_articles)
        if not isinstance(result, dict):
            result = {}
        success = bool(result.get("success"))
        summary = _build_summary(result)
        surprise = 0.0 if success else 1.0
        articles = _parse_articles(result, gap_terms=set(req.gap_terms))
        aggregate_salience = max((a.salience for a in articles), default=0.0)
    except Exception as exc:
        summary = f"fetch failed: {exc}"

    outcome = ActionOutcomeRefV1(
        action_id=action_id,
        kind=_FETCH_KIND,
        summary=summary,
        success=success,
        surprise=surprise,
        observed_at=observed_at,
        query=req.query,
        articles=articles,
        salience=aggregate_salience,
    )
    append_action_outcome(subject=req.subject, outcome=outcome)
    return outcome
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest orion/autonomy/tests/test_episode_fetch.py -q`
Expected: PASS — including the two pre-existing tests (their `{"urls": [...], "success": True}` backends now flow through `_parse_articles`' URL fallback with salience 0.0).

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/episode_fetch.py orion/autonomy/tests/test_episode_fetch.py
git commit -m "feat(autonomy): carry query, scored articles, and aggregate salience into fetch outcome"
```

---

### Task 5: Grounded narrative seed + stop discarding curiosity signals

**Files:**
- Modify: `orion/autonomy/policy_act.py` (imports; `maybe_execute_readonly_fetch_after_goal` request build; add `_gap_section_label` + `build_episode_narrative_seed`; rework `maybe_compose_autonomy_episode_after_fetch`)
- Test: `orion/autonomy/tests/test_policy_act.py` (add seed-builder tests; **fix the stale `startswith("fetch outcome:")` assertion**)

- [ ] **Step 1: Write the failing tests (new seed-builder coverage)**

Add to `orion/autonomy/tests/test_policy_act.py` (append at end; `ActionOutcomeRefV1` and `_gap_signal`/`_goal` helpers already exist in the file):

```python
def test_build_episode_narrative_seed_grounds_on_articles() -> None:
    from orion.autonomy.models import FetchedArticleRefV1
    from orion.autonomy.policy_act import build_episode_narrative_seed

    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 2 article(s)",
        success=True,
        query="hardware compute gpu recent news coverage",
        articles=[
            FetchedArticleRefV1(
                url="https://example.com/a",
                title="New GPU cluster",
                description="A big hardware compute launch.",
                salience=0.67,
            ),
            FetchedArticleRefV1(url="https://example.com/b", title="Side note", salience=0.0),
        ],
        salience=0.67,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)

    assert "hardware compute gpu" in seed          # the "why" (gap section)
    assert "New GPU cluster" in seed               # a real article title (the "what")
    assert "salience 0.67" in seed                 # salience marker
    assert "https://example.com/a" in seed         # real source url
    assert "Do not invent sources" in seed         # anti-confabulation ask
    assert "closes the gap" in seed                # satiation-assessment ask


def test_build_episode_narrative_seed_marks_unscored_when_no_gap_terms() -> None:
    from orion.autonomy.models import FetchedArticleRefV1
    from orion.autonomy.policy_act import build_episode_narrative_seed

    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        query="gpu news",
        articles=[FetchedArticleRefV1(url="https://example.com/a", title="A", salience=0.0)],
        salience=0.0,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)
    assert "unscored" in seed


def test_build_episode_narrative_seed_failure_branch_unchanged() -> None:
    from orion.autonomy.policy_act import build_episode_narrative_seed

    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetch failed: timeout",
        success=False,
        surprise=1.0,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)
    assert seed == "fetch failed: fetch failed: timeout"


def test_build_episode_narrative_seed_truncates_long_description() -> None:
    from orion.autonomy.models import FetchedArticleRefV1
    from orion.autonomy.policy_act import build_episode_narrative_seed

    long_desc = "x" * 500
    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        query="gpu news",
        articles=[FetchedArticleRefV1(url="https://example.com/a", title="A", description=long_desc, salience=0.5)],
        salience=0.5,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)
    assert "…" in seed
    assert "x" * 500 not in seed
```

- [ ] **Step 2: Fix the now-stale existing assertion**

The seed is no longer a `"fetch outcome:"` one-liner. In `orion/autonomy/tests/test_policy_act.py`, in `test_policy_act_dispatches_episode_journal_after_fetch`, give the fixture a real article and replace the final assertion.

Change the `fetch_outcome` construction in that test to include an article:

```python
    from orion.autonomy.models import FetchedArticleRefV1

    fetch_outcome = ActionOutcomeRefV1(
        action_id="fetch-test",
        kind="web.fetch.readonly",
        summary="fetched 2 article(s)",
        success=True,
        surprise=0.0,
        observed_at=datetime.now(timezone.utc),
        query="hardware compute gpu recent news coverage",
        articles=[
            FetchedArticleRefV1(url="https://example.com/a", title="GPU news", salience=0.67)
        ],
        salience=0.67,
    )
```

Replace the final assertion:

```python
    assert journal_dispatch.await_args.kwargs["narrative_seed"].startswith("fetch outcome:")
```

with:

```python
    seed = journal_dispatch.await_args.kwargs["narrative_seed"]
    assert "hardware compute gpu" in seed
    assert "GPU news" in seed
```

(Leave `test_policy_act_composes_episode_journal_on_fetch_failure` unchanged — it asserts `"fetch failed" in narrative_seed`, which the failure branch still satisfies.)

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest orion/autonomy/tests/test_policy_act.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_episode_narrative_seed'` for the new tests (the edited existing test will error on import too).

- [ ] **Step 4: Write the implementation**

In `orion/autonomy/policy_act.py`, add the salience import near the other `orion.autonomy` imports:

```python
from orion.autonomy.salience import gap_terms_from_signals
```

Set `gap_terms` when building the request inside `maybe_execute_readonly_fetch_after_goal`. Replace:

```python
    query = build_readonly_fetch_query(curiosity_signals)
    req = EpisodeFetchRequest(
        subject=goal.subject,
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        query=query,
    )
```

with:

```python
    query = build_readonly_fetch_query(curiosity_signals)
    gap_terms = gap_terms_from_signals(curiosity_signals, fallback_query=query)
    req = EpisodeFetchRequest(
        subject=goal.subject,
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        query=query,
        gap_terms=tuple(sorted(gap_terms)),
    )
```

Add the seed builder + section-label helper. Place them just above `maybe_compose_autonomy_episode_after_fetch` (constants `_GAP_SIGNAL`, `curiosity_strength_from_signals`, and type `Sequence`/`GoalProposalV1`/`FrontierInvocationSignalV1`/`ActionOutcomeRefV1` are already imported in this module):

```python
_MAX_SEED_DESC_CHARS = 300


def _gap_section_label(signals: Sequence[FrontierInvocationSignalV1]) -> str:
    for sig in signals:
        if sig.signal_type != _GAP_SIGNAL:
            continue
        for ref in sig.focal_node_refs:
            section = str(ref or "").strip()
            if section.startswith("section:"):
                return section.split(":", 1)[1].replace("_", " ")
    return ""


def build_episode_narrative_seed(
    goal: GoalProposalV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    fetch_outcome: ActionOutcomeRefV1,
) -> str:
    """Structured multi-line compose seed: why + what + salience + satiation ask.

    `goal` is part of the stable interface (spec contract + future goal_statement
    enrichment seam) even though the current body does not read it. If the repo's
    linter flags it as unused (e.g. ruff ARG001), suppress with a leading
    `del goal` line at the top of the body — do NOT invent usage and do NOT drop
    the parameter (tests and callers pass it positionally).
    """
    if not fetch_outcome.success:
        return f"fetch failed: {fetch_outcome.summary}"

    lines: list[str] = []
    strength = curiosity_strength_from_signals(curiosity_signals)
    section = _gap_section_label(curiosity_signals)
    if section:
        lines.append(f'Why: predictive coverage gap in "{section}" (strength {strength:.2f}).')
    else:
        lines.append(f"Why: predictive coverage gap (strength {strength:.2f}).")
    if fetch_outcome.query:
        lines.append(f'Query: "{fetch_outcome.query}"')

    articles = fetch_outcome.articles
    if articles:
        lines.append(f"Fetched {len(articles)} article(s):")
        any_scored = any(a.salience > 0.0 for a in articles)
        for idx, art in enumerate(articles, start=1):
            marker = f"salience {art.salience:.2f}" if any_scored else "unscored"
            title = art.title or "(untitled)"
            lines.append(f"  {idx}. [{marker}] {title} — {art.url}")
            desc = (art.description or "").strip()
            if desc:
                if len(desc) > _MAX_SEED_DESC_CHARS:
                    desc = desc[:_MAX_SEED_DESC_CHARS].rstrip() + "…"
                lines.append(f"     {desc}")
    else:
        lines.append(f"fetch outcome: {fetch_outcome.summary}")

    lines.append(
        "Reflect: summarize each article and assess whether it closes the gap that "
        "drove this fetch. Name what is still missing. Do not invent sources."
    )
    return "\n".join(lines)
```

In `maybe_compose_autonomy_episode_after_fetch`, remove the discard and use the builder. Delete this line near the top of the function:

```python
    del curiosity_signals
```

and replace the seed construction:

```python
    narrative_seed = (
        f"fetch outcome: {fetch_outcome.summary}"
        if fetch_outcome.success
        else f"fetch failed: {fetch_outcome.summary}"
    )
```

with:

```python
    narrative_seed = build_episode_narrative_seed(goal, curiosity_signals, fetch_outcome)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest orion/autonomy/tests/test_policy_act.py -q`
Expected: PASS (new seed tests + edited dispatch test + unchanged failure test).

- [ ] **Step 6: Run the full autonomy gate lane (no regressions)**

Run: `pytest orion/autonomy/tests -q`
Expected: PASS. If `orion/spark/concept_induction/tests/test_metabolism_hook.py` imports these symbols, also run:

```bash
pytest orion/spark/concept_induction/tests -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add orion/autonomy/policy_act.py orion/autonomy/tests/test_policy_act.py
git commit -m "feat(autonomy): grounded episode narrative seed (why + articles + salience)"
```

---

## Final verification & review (after all tasks)

- [ ] **Run the whole touched surface**

```bash
pytest orion/autonomy/tests orion/spark/concept_induction/tests -q
```

Expected: all PASS.

- [ ] **Lint check** on every changed file (fix anything you introduced): use ReadLints on
`orion/autonomy/salience.py`, `orion/autonomy/models.py`, `orion/autonomy/fetch_backends.py`,
`orion/autonomy/episode_fetch.py`, `orion/autonomy/policy_act.py`.

- [ ] **Code review gate:** run the code-review skill in a subagent against the branch diff; fix all material findings; re-run the autonomy gate lane.

- [ ] **Live smoke (runtime truth):** trigger a world-pulse run with the episode journal enabled
(`ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED=true`, `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED=true`,
a real `FIRECRAWL_API_KEY` / `ORION_EPISODE_FETCH_BACKEND=firecrawl`). Capture the
`substrate_episode_journal_dispatched` log line and the composed journal body. Confirm it names ≥1 real
article title/url, states the driving gap, and gives a satiation assessment. If the live path is not
exercised, mark this item `UNVERIFIED` in the PR report rather than claiming success.

## Acceptance checks (from the spec)

- [ ] Journal body names ≥1 fetched article title/url from the run.
- [ ] Journal body states the curiosity/gap that drove the fetch.
- [ ] Journal body assesses satiation (did it close the gap / what's missing).
- [ ] `ActionOutcomeRefV1` in the local outcome store carries `query`, `articles`, per-article `salience`, aggregate `salience`.
- [ ] All new unit tests pass; existing episode-journal/policy tests still pass.
- [ ] No new confabulated sources in a live sample.

## Notes on scope / non-goals (do not do these here)

- No embedding/LLM salience scorer (term overlap only).
- No Firecrawl `scrapeOptions`/markdown full-text scraping.
- No SQL `action_outcomes` JSONB column; `ActionOutcomeEmitV1` stays flat.
- No `goal_statement` enrichment; no capability-policy/threshold changes.
- No `orion/schemas/registry.py` change (`ActionOutcomeRefV1` is not registered — verified).
- No `.env_example`/Docker/bus-channel changes (pure `orion/autonomy/*` library work).

## Rollback / disable

- Whole path gated by `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED`; disabling reverts to no episode journal.
- Seams 1-2 are additive/defaulted. Reverting only Task 5 (seed builder) restores the old count-only
  seed without touching the schema.
