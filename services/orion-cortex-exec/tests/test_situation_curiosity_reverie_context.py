"""2026-08-30: curiosity world-priors and dream reveries in the situation
brief.

Two new, ON-by-default sections (Juniper's explicit call, unlike most other
sections in this file which default off/opt-in). Mirrors
`test_situation_affect_context.py`'s framing: the properties that matter
more than the happy path are (1) both are ON by default, (2) every failure
mode degrades to an honest "unavailable"/"do not infer" line rather than a
guess or an exception, and (3) the rendered prompt still respects the
existing 1200-char production budget alongside every other section.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from orion.curiosity.worldview import Prior, WorldviewSnapshot
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


def test_fragment_stays_within_production_budget_with_everything_present() -> None:
    """All sections present at once, at the real 1200-char production cap --
    the new lines must not blow the budget every other section already
    shares.

    Fixture sizes mirror what the real builder can actually produce
    (`_CURIOSITY_MAX_PRIORS`=2 summaries capped at
    `_CURIOSITY_CLAIM_MAX_CHARS`=110 chars, `_REVERIE_MAX_SNIPPETS`=2
    snippets capped at `_REVERIE_SNIPPET_MAX_CHARS`=110 chars) -- this
    renderer-level test constructs the schema directly, which has no cap of
    its own, so an unrealistically larger fixture here would not describe a
    reachable production state.
    """
    brief = _brief(
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
    fragment = _build_prompt_fragment(brief, 1200)
    assert len(fragment.compact_text) <= 1200
    # Cautions must survive truncation whole -- same non-negotiable this
    # file's affect/perception tests already assert.
    assert "not a requirement to mention" in fragment.compact_text
