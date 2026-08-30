"""2026-08-30: curiosity world-priors and dream reveries in the situation
brief.

Two new, ON-by-default sections (Juniper's explicit call, unlike most other
sections in this file which default off/opt-in). Mirrors
`test_situation_affect_context.py`'s framing: the properties that matter
more than the happy path are (1) both are ON by default, (2) every failure
mode degrades to an honest "unavailable"/"do not infer" line rather than a
guess or an exception, and (3) the rendered prompt still respects the
production budget (raised 2026-08-30 from 1200 to 7200 chars, Juniper's
explicit request -- see `_DEFAULT_PROMPT_MAX_CHARS` in
`orion/situational/context.py`) alongside every other section.

Also covers a real bug caught in review the same day: `_fetch_curiosity_
context`'s confidence re-rank operated on a pool that `read_snapshot`
itself had already sliced down to its most-UNCERTAIN priors, so a live
pool larger than that slice could never surface a genuinely confident one
-- see `test_curiosity_confident_priors_survive_a_large_live_pool` below.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from orion.curiosity.worldview import (
    COUNTS_CYPHER,
    CONCEPT_COUNT_CYPHER,
    LIVE_PRIORS_CYPHER,
    RECENT_RUNS_CYPHER,
    RECENT_SETTLED_CYPHER,
    Prior,
    WorldviewSnapshot,
)
from orion.schemas.situation import (
    CuriosityPriorContextV1,
    CuriosityPriorSummaryV1,
    ReverieContextV1,
    ReverieSnippetV1,
    SituationBriefV1,
    SituationDiagnosticsV1,
)
from orion.situational import context as situation_mod
from orion.situational.context import (
    SituationSettings,
    _build_curiosity_context,
    _build_prompt_fragment,
    _build_reverie_context,
    settings_from_runtime,
)
from orion.situational.reverie_reader import ReverieRow


class _FakeGraphReader:
    """Stand-in for `WorldviewReader` that answers `read_snapshot`'s own
    fixed set of Cypher queries by exact string match, so the REAL
    `read_snapshot`/`select_priors` pipeline runs (unlike the other
    curiosity tests in this file, which monkeypatch `read_snapshot` itself
    and so cannot exercise the interaction between it and
    `_fetch_curiosity_context`'s re-rank)."""

    def __init__(self, rows_by_cypher: dict[str, list[dict]]) -> None:
        self._rows_by_cypher = rows_by_cypher

    def query(self, cypher: str) -> list[dict]:
        return self._rows_by_cypher.get(cypher, [])


@pytest.fixture(autouse=True)
def _clear_curiosity_and_reverie_caches():
    # _CURIOSITY_CACHE keys by "{host}:{port}:{name}" and _REVERIE_CACHE by a
    # constant key -- both would otherwise leak a mocked/cached result across
    # unrelated test cases, same leak risk test_situation_provider.py's own
    # _clear_runtime_cache/_clear_weather_cache fixtures guard against.
    situation_mod._CURIOSITY_CACHE.clear()
    situation_mod._REVERIE_CACHE.clear()
    yield
    situation_mod._CURIOSITY_CACHE.clear()
    situation_mod._REVERIE_CACHE.clear()


def _cfg(**overrides) -> SituationSettings:
    cfg = settings_from_runtime(SimpleNamespace())
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _diag() -> SituationDiagnosticsV1:
    return SituationDiagnosticsV1()


def _prior(**overrides) -> Prior:
    base = dict(
        prior_id="abc123",
        claim="Juniper prefers terse status updates over long explanations.",
        confidence=0.8,
        status="open",
        times_tested=2,
    )
    base.update(overrides)
    return Prior(**base)


# --- defaults --------------------------------------------------------------


def test_curiosity_enabled_by_default() -> None:
    assert settings_from_runtime(SimpleNamespace()).curiosity_enabled is True


def test_reverie_enabled_by_default() -> None:
    assert settings_from_runtime(SimpleNamespace()).reverie_enabled is True


# --- curiosity provider states ----------------------------------------------


@pytest.mark.asyncio
async def test_curiosity_disabled_yields_unavailable_not_an_error() -> None:
    diag = _diag()
    ctx = await _build_curiosity_context(_cfg(curiosity_enabled=False), diag)
    assert ctx.available is False
    assert ctx.source == "disabled"
    assert diag.provider_status["curiosity"] == "disabled"


@pytest.mark.asyncio
async def test_curiosity_no_graph_host_is_unconfigured_not_an_error() -> None:
    """cortex-exec has no established connection to orion_worldview today
    (only Hub does) -- this is a real, distinct, non-error state."""
    diag = _diag()
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host=""), diag
    )
    assert ctx.available is False
    assert ctx.source == "unconfigured"
    assert diag.provider_status["curiosity"] == "unconfigured"


@pytest.mark.asyncio
async def test_curiosity_live_priors_ranked_by_confidence(monkeypatch) -> None:
    """read_snapshot's own ordering is most-uncertain-first (picks what to
    TEST next); the prompt line wants the opposite -- highest confidence
    first, so it reads as "what Orion currently believes"."""
    snapshot = WorldviewSnapshot(
        live_priors=[
            _prior(prior_id="low", claim="low confidence claim", confidence=0.55),
            _prior(prior_id="high", claim="high confidence claim", confidence=0.95),
            _prior(prior_id="none", claim="unrated claim", confidence=None),
        ],
        live_total=3,
    )
    monkeypatch.setattr(situation_mod, "read_snapshot", lambda *a, **k: snapshot)
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host="127.0.0.1", curiosity_graph_port=6380),
        _diag(),
    )
    assert ctx.available is True
    assert ctx.source == "orion_worldview"
    assert ctx.live_total == 3
    # Capped to _CURIOSITY_MAX_PRIORS (2) -- the unrated claim sorts last and
    # does not survive the cap.
    claims = [s.claim for s in ctx.summaries]
    assert claims == ["high confidence claim", "low confidence claim"]


@pytest.mark.asyncio
async def test_curiosity_confidently_false_prior_ranks_with_confidently_true(monkeypatch) -> None:
    """Ranking must be by CERTAINTY (`Prior.uncertainty`, distance from
    0.5), not raw confidence descending. A prior Orion is 95% sure is
    FALSE (confidence=0.05) is just as much a current, decisive belief as
    one it is 95% sure is true -- raw-value sorting would bury it near the
    bottom, tied with unrated priors."""
    snapshot = WorldviewSnapshot(
        live_priors=[
            _prior(prior_id="toss-up", claim="near coin-flip claim", confidence=0.52),
            _prior(prior_id="sure-false", claim="confidently false claim", confidence=0.05),
            _prior(prior_id="sure-true", claim="confidently true claim", confidence=0.95),
        ],
        live_total=3,
    )
    monkeypatch.setattr(situation_mod, "read_snapshot", lambda *a, **k: snapshot)
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host="127.0.0.1"), _diag()
    )
    claims = {s.claim for s in ctx.summaries}
    assert claims == {"confidently false claim", "confidently true claim"}
    assert "near coin-flip claim" not in claims


@pytest.mark.asyncio
async def test_curiosity_confident_priors_survive_a_large_live_pool(monkeypatch) -> None:
    """Regression for a real bug caught in review (2026-08-30):
    `read_snapshot`'s Cypher pulls every live prior up to
    `LIVE_PRIORS_LIMIT`, but `select_priors` then SLICES that set down to
    `sample`, sorted MOST-UNCERTAIN-first (it exists to pick what Orion
    should test next). `_CURIOSITY_PRIOR_POOL_SAMPLE` used to be a small
    constant (20), which meant a live pool larger than that could fill the
    entire slice with near-coin-flip priors and permanently exclude every
    genuinely confident one -- they never survived `select_priors`' own
    cut to reach this module's confidence re-rank at all. Reproduced here
    with the REAL `read_snapshot`/`select_priors` pipeline (a fake
    `WorldviewReader.query()`, not a monkeypatched `read_snapshot`) so the
    interaction between the two is actually exercised, unlike the other
    tests in this file.
    """
    near_toss_up = [
        {
            "prior_id": f"toss-{i}",
            "claim": f"toss up claim {i}",
            "confidence": 0.50 + (i % 3) * 0.01,
            "status": "open",
            "times_tested": 0,
            "formed_from": "",
            "last_tested_at": "",
        }
        for i in range(25)
    ]
    confident = [
        {
            "prior_id": "sure-true",
            "claim": "Juniper prefers terse status updates.",
            "confidence": 0.97,
            "status": "open",
            "times_tested": 3,
            "formed_from": "",
            "last_tested_at": "",
        },
        {
            "prior_id": "sure-false",
            "claim": "Orion should announce every internal metric unprompted.",
            "confidence": 0.03,
            "status": "open",
            "times_tested": 5,
            "formed_from": "",
            "last_tested_at": "",
        },
    ]
    rows_by_cypher = {
        LIVE_PRIORS_CYPHER: near_toss_up + confident,
        COUNTS_CYPHER: [{"live_total": 27, "closed_total": 0}],
        CONCEPT_COUNT_CYPHER: [{"n": 0}],
        RECENT_SETTLED_CYPHER: [],
        RECENT_RUNS_CYPHER: [],
    }
    monkeypatch.setattr(
        situation_mod,
        "WorldviewReader",
        lambda **kwargs: _FakeGraphReader(rows_by_cypher),
    )
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host="127.0.0.1"), _diag()
    )
    assert ctx.available is True
    claims = [s.claim for s in ctx.summaries]
    assert "Juniper prefers terse status updates." in claims
    assert "Orion should announce every internal metric unprompted." in claims


@pytest.mark.asyncio
async def test_curiosity_empty_live_pool_is_unavailable(monkeypatch) -> None:
    snapshot = WorldviewSnapshot(live_priors=[], live_total=0)
    monkeypatch.setattr(situation_mod, "read_snapshot", lambda *a, **k: snapshot)
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host="127.0.0.1"), _diag()
    )
    assert ctx.available is False
    assert ctx.source == "unavailable"


@pytest.mark.asyncio
async def test_curiosity_graph_unreachable_fails_open(monkeypatch) -> None:
    snapshot = WorldviewSnapshot(unavailable_reason="ConnectionError: refused")
    monkeypatch.setattr(situation_mod, "read_snapshot", lambda *a, **k: snapshot)
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host="127.0.0.1"), _diag()
    )
    assert ctx.available is False
    assert ctx.source == "error"


@pytest.mark.asyncio
async def test_curiosity_reader_exception_fails_open(monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("redis gone")

    monkeypatch.setattr(situation_mod, "read_snapshot", _boom)
    diag = _diag()
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host="127.0.0.1"), diag
    )
    assert ctx.available is False
    assert ctx.source == "error"
    assert "redis gone" in diag.provider_errors["curiosity"]


@pytest.mark.asyncio
async def test_curiosity_claim_is_truncated_and_capped(monkeypatch) -> None:
    long_claim = "x" * 400
    priors = [
        _prior(prior_id=f"p{i}", claim=long_claim if i == 0 else f"claim {i}", confidence=0.9 - i * 0.1)
        for i in range(6)
    ]
    snapshot = WorldviewSnapshot(live_priors=priors, live_total=len(priors))
    monkeypatch.setattr(situation_mod, "read_snapshot", lambda *a, **k: snapshot)
    ctx = await _build_curiosity_context(
        _cfg(curiosity_enabled=True, curiosity_graph_host="127.0.0.1"), _diag()
    )
    # Capped small -- color, not a dump of the whole worldview.
    assert len(ctx.summaries) <= 2
    assert len(ctx.summaries[0].claim) < 120
    assert ctx.summaries[0].claim.endswith("…")


# --- reverie provider states -------------------------------------------------


@pytest.mark.asyncio
async def test_reverie_disabled_yields_unavailable_not_an_error() -> None:
    diag = _diag()
    ctx = await _build_reverie_context(_cfg(reverie_enabled=False), diag)
    assert ctx.available is False
    assert ctx.source == "disabled"
    assert diag.provider_status["reverie"] == "disabled"


@pytest.mark.asyncio
async def test_reverie_no_rows_is_unavailable_not_an_error(monkeypatch) -> None:
    monkeypatch.setattr(situation_mod, "fetch_recent_reverie_snippets", lambda limit: [])
    ctx = await _build_reverie_context(_cfg(reverie_enabled=True), _diag())
    assert ctx.available is False
    assert ctx.source == "unavailable"


@pytest.mark.asyncio
async def test_reverie_live_rows_are_available(monkeypatch) -> None:
    rows = [
        ReverieRow(text="a half-remembered thread about tide pools", observed_at=None, salience=0.7),
        ReverieRow(text="recurring motif: a lighthouse", observed_at=None, salience=0.4),
    ]
    monkeypatch.setattr(situation_mod, "fetch_recent_reverie_snippets", lambda limit: rows)
    ctx = await _build_reverie_context(_cfg(reverie_enabled=True), _diag())
    assert ctx.available is True
    assert ctx.source == "reverie_sql"
    assert len(ctx.snippets) == 2
    assert ctx.snippets[0].text == "a half-remembered thread about tide pools"


@pytest.mark.asyncio
async def test_reverie_snippet_text_is_truncated(monkeypatch) -> None:
    rows = [ReverieRow(text="y" * 400, observed_at=None, salience=None)]
    monkeypatch.setattr(situation_mod, "fetch_recent_reverie_snippets", lambda limit: rows)
    ctx = await _build_reverie_context(_cfg(reverie_enabled=True), _diag())
    assert ctx.available is True
    assert len(ctx.snippets[0].text) < 120
    assert ctx.snippets[0].text.endswith("…")


@pytest.mark.asyncio
async def test_reverie_reader_exception_fails_open(monkeypatch) -> None:
    def _boom(limit):
        raise RuntimeError("db gone")

    monkeypatch.setattr(situation_mod, "fetch_recent_reverie_snippets", _boom)
    diag = _diag()
    ctx = await _build_reverie_context(_cfg(reverie_enabled=True), diag)
    assert ctx.available is False
    assert ctx.source == "error"
    assert "db gone" in diag.provider_errors["reverie"]


# --- prompt rendering --------------------------------------------------------


def _brief(curiosity: CuriosityPriorContextV1, reverie: ReverieContextV1) -> SituationBriefV1:
    """Same technique test_situation_affect_context.py's own _brief() uses:
    build via the production helpers, swap in the sub-context under test."""
    import asyncio

    cfg = _cfg(curiosity_enabled=False, reverie_enabled=False)
    diag = _diag()
    time_ctx = situation_mod._build_time_context(cfg, diag)
    now = datetime.now(timezone.utc)
    return SituationBriefV1(
        generated_at=now,
        time=time_ctx,
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, now)),
        place=situation_mod._build_place_context(cfg),
        curiosity=curiosity,
        reverie=reverie,
    )


def test_available_priors_render_with_confidence_and_status() -> None:
    brief = _brief(
        CuriosityPriorContextV1(
            available=True,
            source="orion_worldview",
            summaries=[
                CuriosityPriorSummaryV1(
                    claim="Juniper prefers terse status updates.",
                    confidence=0.8,
                    status="open",
                    times_tested=2,
                )
            ],
        ),
        ReverieContextV1(),
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "Juniper prefers terse status updates." in text
    assert "confidence=0.80" in text
    assert "open" in text


def test_unavailable_priors_render_no_line_at_all() -> None:
    """Unlike weather/perception/affect above, an unavailable prior section
    is OMITTED entirely rather than rendered as a placeholder line -- see
    _build_prompt_fragment's own comment for why (budget overhead on the
    common no-graph-host path; a real regression on record from exactly this
    pattern with an always-on line)."""
    brief = _brief(CuriosityPriorContextV1(), ReverieContextV1())
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "world-priors" not in text.lower()


def test_available_reverie_renders_snippets() -> None:
    brief = _brief(
        CuriosityPriorContextV1(),
        ReverieContextV1(
            available=True,
            source="reverie_sql",
            snippets=[ReverieSnippetV1(text="a recurring motif: a lighthouse")],
        ),
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "a recurring motif: a lighthouse" in text


def test_unavailable_reverie_renders_no_line_at_all() -> None:
    """Same omission behavior as priors above -- see that test's docstring."""
    brief = _brief(CuriosityPriorContextV1(), ReverieContextV1())
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "reverie" not in text.lower()


def _everything_present_brief() -> SituationBriefV1:
    """Fixture sizes mirror what the real builder can actually produce
    (`_CURIOSITY_MAX_PRIORS`=2 summaries capped at
    `_CURIOSITY_CLAIM_MAX_CHARS`=110 chars, `_REVERIE_MAX_SNIPPETS`=2
    snippets capped at `_REVERIE_SNIPPET_MAX_CHARS`=110 chars) -- this
    renderer-level fixture constructs the schema directly, which has no cap
    of its own, so an unrealistically larger fixture here would not
    describe a reachable production state."""
    return _brief(
        CuriosityPriorContextV1(
            available=True,
            source="orion_worldview",
            summaries=[
                CuriosityPriorSummaryV1(
                    claim="x" * 110, confidence=0.8, status="open", times_tested=2
                ),
                CuriosityPriorSummaryV1(
                    claim="y" * 110, confidence=0.6, status="supported", times_tested=1
                ),
            ],
        ),
        ReverieContextV1(
            available=True,
            source="reverie_sql",
            snippets=[
                ReverieSnippetV1(text="a" * 110),
                ReverieSnippetV1(text="b" * 110),
            ],
        ),
    )


def test_fragment_stays_within_the_old_tight_1200_budget_with_everything_present() -> None:
    """All sections present at once, at the OLD 1200-char cap (pre-2026-08-30
    default) -- defense in depth: the new lines must not blow even the
    tighter historical budget every other section used to share, in case a
    future operator dials `ORION_SITUATION_PROMPT_MAX_CHARS` back down."""
    fragment = _build_prompt_fragment(_everything_present_brief(), 1200)
    assert len(fragment.compact_text) <= 1200
    # Cautions must survive truncation whole -- same non-negotiable this
    # file's affect/perception tests already assert.
    assert "not a requirement to mention" in fragment.compact_text


def test_fragment_stays_within_the_new_7200_production_budget_with_everything_present() -> None:
    """Same fixture, at the real 2026-08-30 production default
    (`_DEFAULT_PROMPT_MAX_CHARS` in orion/situational/context.py) -- with
    six times the room, both new sections plus every existing one and all
    cautions should fit without any truncation at all."""
    fragment = _build_prompt_fragment(_everything_present_brief(), situation_mod._DEFAULT_PROMPT_MAX_CHARS)
    assert len(fragment.compact_text) <= situation_mod._DEFAULT_PROMPT_MAX_CHARS
    assert "…" not in fragment.compact_text, "6x the old budget should not need any truncation here"
    for caution in fragment.caution_lines:
        assert caution in fragment.compact_text
