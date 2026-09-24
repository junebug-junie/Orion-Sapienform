from __future__ import annotations

from app.lane_contention import (
    _never_a_swap_target,
    _parse_capacity_map,
    _parse_fallback_map,
    resolve_contention_fallback,
)
from app.upstream_admission import UpstreamAdmission

FALLBACK_MAP = _parse_fallback_map('{"metacog": ["quick", "agent"], "quick": ["metacog", "agent"]}')
CAPACITY_MAP = _parse_capacity_map('{"metacog": 1, "quick": 4, "agent": 1}')
ROUTE_URLS = {
    "metacog": "http://circe-worker-2:8080",
    "quick": "http://circe-worker-fast-1:8080",
    "agent": "http://circe-worker-agent-1:8080",
    "agent-burst": "http://circe-worker-agent-burst-1:8080",
    "chat": "http://circe-worker-1:8080",
}


def _gate() -> UpstreamAdmission:
    return UpstreamAdmission(max_inflight=8)


def _occupy(gate: UpstreamAdmission, url: str, n: int) -> None:
    lane = gate.lane(url)
    lane.inflight += n


def _resolve(route_key: str, gate: UpstreamAdmission, **overrides) -> tuple[str, bool]:
    kwargs = dict(
        enabled=True,
        fallback_map=FALLBACK_MAP,
        real_capacity_map=CAPACITY_MAP,
        default_capacity=8,
        gate=gate,
    )
    kwargs.update(overrides)
    return resolve_contention_fallback(route_key, ROUTE_URLS, **kwargs)


def test_no_contention_keeps_preferred_lane() -> None:
    gate = _gate()
    chosen, swapped = _resolve("metacog", gate)
    assert (chosen, swapped) == ("metacog", False)


def test_metacog_contended_swaps_to_quick() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["metacog"], 1)  # metacog's real capacity is 1
    chosen, swapped = _resolve("metacog", gate)
    assert (chosen, swapped) == ("quick", True)


def test_quick_contended_swaps_to_metacog() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["quick"], 4)  # quick's real capacity is 4
    chosen, swapped = _resolve("quick", gate)
    assert (chosen, swapped) == ("metacog", True)


def test_quick_at_three_of_four_is_not_contended() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["quick"], 3)
    chosen, swapped = _resolve("quick", gate)
    assert (chosen, swapped) == ("quick", False)


def test_sibling_and_own_lane_contended_falls_through_to_agent() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["metacog"], 1)
    _occupy(gate, ROUTE_URLS["quick"], 4)
    chosen, swapped = _resolve("metacog", gate)
    assert (chosen, swapped) == ("agent", True)


def test_all_three_contended_keeps_preferred() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["metacog"], 1)
    _occupy(gate, ROUTE_URLS["quick"], 4)
    _occupy(gate, ROUTE_URLS["agent"], 1)
    chosen, swapped = _resolve("metacog", gate)
    assert (chosen, swapped) == ("metacog", False)


def test_disabled_never_swaps_even_when_contended() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["metacog"], 1)
    chosen, swapped = _resolve("metacog", gate, enabled=False)
    assert (chosen, swapped) == ("metacog", False)


def test_chat_is_never_in_the_fallback_map() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["chat"], 8)
    chosen, swapped = _resolve("chat", gate)
    assert (chosen, swapped) == ("chat", False)


def test_missing_partner_url_is_skipped_tries_the_next_one() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["metacog"], 1)
    urls_without_quick = {k: v for k, v in ROUTE_URLS.items() if k != "quick"}
    chosen, swapped = resolve_contention_fallback(
        "metacog",
        urls_without_quick,
        enabled=True,
        fallback_map=FALLBACK_MAP,
        real_capacity_map=CAPACITY_MAP,
        default_capacity=8,
        gate=gate,
    )
    # quick's URL is unresolvable (skipped), agent's is fine and uncontended
    assert (chosen, swapped) == ("agent", True)


def test_all_partner_urls_missing_is_a_noop() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["metacog"], 1)
    urls_metacog_only = {"metacog": ROUTE_URLS["metacog"]}
    chosen, swapped = resolve_contention_fallback(
        "metacog",
        urls_metacog_only,
        enabled=True,
        fallback_map=FALLBACK_MAP,
        real_capacity_map=CAPACITY_MAP,
        default_capacity=8,
        gate=gate,
    )
    assert (chosen, swapped) == ("metacog", False)


def test_route_not_in_route_urls_is_a_noop() -> None:
    gate = _gate()
    chosen, swapped = resolve_contention_fallback(
        "unknown-route",
        ROUTE_URLS,
        enabled=True,
        fallback_map=FALLBACK_MAP,
        real_capacity_map=CAPACITY_MAP,
        default_capacity=8,
        gate=gate,
    )
    assert (chosen, swapped) == ("unknown-route", False)


def test_route_missing_from_fallback_map_is_a_noop() -> None:
    gate = _gate()
    _occupy(gate, ROUTE_URLS["agent"], 8)
    chosen, swapped = resolve_contention_fallback(
        "spark",
        ROUTE_URLS,
        enabled=True,
        fallback_map=FALLBACK_MAP,
        real_capacity_map=CAPACITY_MAP,
        default_capacity=8,
        gate=gate,
    )
    assert (chosen, swapped) == ("spark", False)


def test_default_capacity_used_when_route_not_in_real_capacity_map() -> None:
    gate = _gate()
    fallback_map = _parse_fallback_map('{"metacog": ["quick"]}')
    capacity_map: dict[str, int] = {}  # no real-capacity entries at all
    _occupy(gate, ROUTE_URLS["metacog"], 7)  # below default_capacity=8
    chosen, swapped = resolve_contention_fallback(
        "metacog",
        ROUTE_URLS,
        enabled=True,
        fallback_map=fallback_map,
        real_capacity_map=capacity_map,
        default_capacity=8,
        gate=gate,
    )
    assert (chosen, swapped) == ("metacog", False)
    _occupy(gate, ROUTE_URLS["metacog"], 1)  # now at 8/8
    chosen, swapped = resolve_contention_fallback(
        "metacog",
        ROUTE_URLS,
        enabled=True,
        fallback_map=fallback_map,
        real_capacity_map=capacity_map,
        default_capacity=8,
        gate=gate,
    )
    assert (chosen, swapped) == ("quick", True)


def test_json_parsing_helpers() -> None:
    assert _parse_fallback_map("") == {}
    assert _parse_fallback_map("not json") == {}
    assert _parse_fallback_map('["a", "b"]') == {}
    assert _parse_fallback_map('{"a": "b"}') == {"a": ("b",)}
    assert _parse_fallback_map('{"a": ["b", "c"]}') == {"a": ("b", "c")}
    assert _parse_capacity_map("") == {}
    assert _parse_capacity_map("not json") == {}
    assert _parse_capacity_map('{"a": "not-an-int"}') == {}
    assert _parse_capacity_map('{"a": 0, "b": -1, "c": 3}') == {"c": 3}


# --- BURST_LLM_ROUTES guard (review-caught 2026-09-24) ---------------------
# agent-burst/chat-burst require a durable capacity lease ordinary traffic
# never carries -- an unleased swap onto either always hard-fails at
# CapacityPermit.acquire() instead of queueing. These must never be chosen,
# both as parsed config and as a defense-in-depth check inside the resolver
# itself (in case a caller passes fallback_map directly, bypassing the parser).


def test_never_a_swap_target_covers_both_burst_routes() -> None:
    assert _never_a_swap_target("agent-burst") is True
    assert _never_a_swap_target("chat-burst") is True
    assert _never_a_swap_target("agent") is False
    assert _never_a_swap_target("quick") is False


def test_parse_fallback_map_drops_burst_route_partners() -> None:
    parsed = _parse_fallback_map('{"agent": ["agent-burst"], "chat": ["chat-burst"]}')
    assert parsed == {}


def test_parse_fallback_map_drops_only_the_burst_entry_keeps_the_rest() -> None:
    parsed = _parse_fallback_map('{"agent": ["agent-burst", "metacog"]}')
    assert parsed == {"agent": ("metacog",)}


def test_resolver_never_picks_agent_burst_even_if_fallback_map_lists_it() -> None:
    # Defense in depth: even if a caller bypasses _parse_fallback_map and hands
    # resolve_contention_fallback a raw dict containing the forbidden pairing
    # (e.g. a future refactor that skips the parser), it must still refuse.
    gate = _gate()
    _occupy(gate, ROUTE_URLS["agent"], 1)
    unsafe_fallback_map = {"agent": ("agent-burst",)}
    unsafe_capacity_map = {"agent": 1, "agent-burst": 1}
    chosen, swapped = resolve_contention_fallback(
        "agent",
        ROUTE_URLS,
        enabled=True,
        fallback_map=unsafe_fallback_map,
        real_capacity_map=unsafe_capacity_map,
        default_capacity=8,
        gate=gate,
    )
    assert (chosen, swapped) == ("agent", False)


def test_agent_burst_never_falls_back_to_agent() -> None:
    # unidirectional: agent-burst has no partner list of its own -- it's only
    # ever a would-be burst target, never a home lane that swaps away.
    gate = _gate()
    _occupy(gate, ROUTE_URLS["agent-burst"], 1)
    chosen, swapped = _resolve("agent-burst", gate)
    assert (chosen, swapped) == ("agent-burst", False)
