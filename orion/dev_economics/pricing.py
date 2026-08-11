"""Small, explicitly versioned pricing table (model -> $/token), per
``docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md``'s
"Recommended next patch" follow-on: "a small, explicitly versioned pricing
table (model -> $/token), with the table's effective-date range recorded so
historical sessions are priced at the rate that was actually current, not
today's rate applied retroactively."

Real rates below are sourced from the ``claude-api`` skill's cached pricing
table (cached 2026-06-24, Anthropic first-party API rates) -- not guessed.
``claude-sonnet-5`` currently has two real, live rate windows: an
introductory rate through 2026-08-31, then the standard rate -- this is a
live, real example of the "effective-date range" requirement mattering
today, not a hypothetical.

Deliberately narrow: only models this repo's own real usage actually pays
for (see ``scripts/replay_dev_economics_ledger.py``'s real model-mix output)
have real, sourced rates. A model with no rate entry returns ``None`` from
``estimate_session_cost_usd`` -- never a fabricated $0.00 -- matching root
``CLAUDE.md``'s "no empty-shell cognition" rule and this module's own
metric-quality-gate discipline (a wrong number is worse than an honestly
absent one).

Who updates this when rates change: whoever notices a real rate change (a
new intro window ends, a model's price moves) and adds a new
``PricingRate`` window -- there is no automated rate-fetch here, matching
the design doc's own open question ("Where should a pricing table live, and
who updates it when rates change?"). This is a real, disclosed maintenance
cost, not a one-time fix -- same shape as the affective-state signal's
jargon-allowlist maintenance cost. An entry with ``effective_until=None`` is
the newest known rate for that model and is treated as still current for
any later date (an open-ended window, not a bug) -- what
``estimate_session_cost_usd`` actually refuses is a date with **no**
covering window at all (e.g. before the earliest known ``effective_from``,
or an unpriced model entirely), never a fabricated number in either case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone


@dataclass(frozen=True)
class PricingRate:
    """One priced window for one model. ``model_prefix`` is matched against
    the real ``model`` string transcripts carry (e.g.
    ``claude-haiku-4-5-20251001``), not an exact-equality match -- real
    transcripts observed in this repo's corpus carry a dated suffix on at
    least one model, and matching by prefix is robust to that without
    hand-listing every dated variant."""

    model_prefix: str
    input_per_mtok: float
    output_per_mtok: float
    effective_from: date
    effective_until: date | None  # None = still the current rate as of writing
    source: str

    # Cache multipliers are a fixed property of the API's cache mechanics,
    # not something that varies per pricing window (per the claude-api
    # skill's own cached "Prompt Caching" reference) -- not per-instance
    # fields, since every window for every model shares the same values.
    CACHE_WRITE_5M_MULTIPLIER = 1.25
    CACHE_WRITE_1H_MULTIPLIER = 2.0
    CACHE_READ_MULTIPLIER = 0.1

    def covers(self, observed_at: date) -> bool:
        if observed_at < self.effective_from:
            return False
        if self.effective_until is not None and observed_at > self.effective_until:
            return False
        return True


# Real Anthropic first-party API rates, cached 2026-06-24 (claude-api skill).
# Entries are written in whatever order is convenient (currently oldest-
# window-first per model) -- find_rate() below picks the covering entry
# with the latest `effective_from` itself, rather than relying on table
# order to encode "newest wins". Confirmed live 2026-08-11 (code review):
# an earlier draft's docstring claimed "ordered newest-window-first" while
# the literal tuple order was actually oldest-first -- harmless only
# because today's windows happen to be non-overlapping. Picking the latest
# effective_from explicitly makes "newest wins" a real guarantee instead of
# an unenforced convention that silently breaks if a future edit ever adds
# an overlapping window without also fixing the table's order.
PRICING_TABLE: tuple[PricingRate, ...] = (
    # claude-sonnet-5: real, live two-window case -- introductory rate
    # through 2026-08-31, standard rate after.
    PricingRate(
        model_prefix="claude-sonnet-5",
        input_per_mtok=2.00,
        output_per_mtok=10.00,
        effective_from=date(2026, 1, 1),  # launch; exact date not needed for this repo's real corpus
        effective_until=date(2026, 8, 31),
        source="claude-api skill pricing table (cached 2026-06-24): intro rate through 2026-08-31",
    ),
    PricingRate(
        model_prefix="claude-sonnet-5",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        effective_from=date(2026, 9, 1),
        effective_until=None,
        source="claude-api skill pricing table (cached 2026-06-24): standard rate after 2026-08-31",
    ),
    PricingRate(
        model_prefix="claude-opus-5",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        effective_from=date(2026, 1, 1),
        effective_until=None,
        source="claude-api skill pricing table (cached 2026-06-24)",
    ),
    PricingRate(
        model_prefix="claude-opus-4-8",
        input_per_mtok=5.00,
        output_per_mtok=25.00,
        effective_from=date(2026, 1, 1),
        effective_until=None,
        source="claude-api skill pricing table (cached 2026-06-24)",
    ),
    PricingRate(
        model_prefix="claude-fable-5",
        input_per_mtok=10.00,
        output_per_mtok=50.00,
        effective_from=date(2026, 1, 1),
        effective_until=None,
        source="claude-api skill pricing table (cached 2026-06-24)",
    ),
    PricingRate(
        model_prefix="claude-mythos-5",
        input_per_mtok=10.00,
        output_per_mtok=50.00,
        effective_from=date(2026, 1, 1),
        effective_until=None,
        source="claude-api skill pricing table (cached 2026-06-24)",
    ),
    PricingRate(
        model_prefix="claude-haiku-4-5",
        input_per_mtok=1.00,
        output_per_mtok=5.00,
        effective_from=date(2026, 1, 1),
        effective_until=None,
        source="claude-api skill pricing table (cached 2026-06-24)",
    ),
)


@dataclass(frozen=True)
class CostEstimate:
    input_cost_usd: float
    output_cost_usd: float
    cache_write_cost_usd: float
    cache_read_cost_usd: float
    rate: PricingRate

    @property
    def total_usd(self) -> float:
        return (
            self.input_cost_usd
            + self.output_cost_usd
            + self.cache_write_cost_usd
            + self.cache_read_cost_usd
        )


def find_rate(model: str, observed_at: date) -> PricingRate | None:
    """The ``PRICING_TABLE`` entry whose ``model_prefix`` matches ``model``,
    whose window covers ``observed_at``, and among any such matches, whose
    ``effective_from`` is latest -- "newest wins" is enforced here, not by
    relying on the table's own literal ordering (see ``PRICING_TABLE``'s
    own comment for why that distinction matters). Returns ``None`` for an
    unpriced model (e.g. this repo's real corpus also shows
    ``xai/grok-4.5``, ``openai/o3``, ``moonshotai/kimi-*`` -- non-Anthropic
    models this table deliberately does not price) or a date none of that
    model's known windows cover -- never guesses a nearby rate."""
    covering = [
        rate
        for rate in PRICING_TABLE
        if model.startswith(rate.model_prefix) and rate.covers(observed_at)
    ]
    if not covering:
        return None
    return max(covering, key=lambda rate: rate.effective_from)


def estimate_session_cost_usd(
    *,
    model: str,
    observed_at: datetime,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
) -> CostEstimate | None:
    """Real $ cost estimate for one session's real token counts, priced at
    the rate that was actually current on ``observed_at`` -- not today's
    rate applied retroactively (the design doc's own explicit requirement).
    Returns ``None``, never a fabricated number, when no covering rate
    exists: an unpriced model, or a date before the earliest known window
    for a model this table does price.

    Cache-creation tokens are priced at the 5-minute-TTL write multiplier
    (1.25x input rate) -- this repo's own transcripts don't currently
    distinguish 5m vs 1h cache writes at the ledger level (see
    ``SessionUsageRecord``), so 5m (the more common, cheaper default) is
    the honest assumption to state rather than silently picking whichever
    number is smaller. Cache-read tokens are priced at 0.1x input rate,
    per the claude-api skill's own cached economics reference."""
    rate = find_rate(model, observed_at.astimezone(timezone.utc).date())
    if rate is None:
        return None
    input_cost = input_tokens * rate.input_per_mtok / 1_000_000
    output_cost = output_tokens * rate.output_per_mtok / 1_000_000
    cache_write_cost = (
        cache_creation_input_tokens
        * rate.input_per_mtok
        * PricingRate.CACHE_WRITE_5M_MULTIPLIER
        / 1_000_000
    )
    cache_read_cost = (
        cache_read_input_tokens
        * rate.input_per_mtok
        * PricingRate.CACHE_READ_MULTIPLIER
        / 1_000_000
    )
    return CostEstimate(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        cache_write_cost_usd=cache_write_cost,
        cache_read_cost_usd=cache_read_cost,
        rate=rate,
    )
