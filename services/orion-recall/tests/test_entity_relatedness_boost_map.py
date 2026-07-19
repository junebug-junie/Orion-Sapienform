"""_compute_entity_relatedness_boost_map (Phase 2 of entity-graph-reasoning,
docs/superpowers/specs/2026-07-19-recall-entity-graph-reasoning-arc.md):
best-effort (boost_map, injected_candidates) computation. Must degrade to
({}, []) on any failure -- never a hard dependency.

Returns a tuple, not just a dict -- live evidence (6 real queries, 3
profiles) showed the boost-only design never actually changed a ranking:
falkor_chat's own fetch is recency-windowed with no query filter, so an
entity from an older turn never entered the candidate pool for the boost to
act on. injected_candidates is what closes that gap.

A SECOND live regression was found after shipping the above: a mundane
message that simply addresses the assistant by name ("Orion, what do you
think...") extracted "orion" as a query entity and scored it a flat 1.0 --
since "orion"/"juniper" are the two most frequent nodes in the whole graph
(near-universal), this injected generic filler turns at full boost
strength purely because they mentioned Orion by name. Fixed with a
document-frequency discount (fetch_entity_degrees) applied to every target
entity, PLUS a minimum-score floor gating which entities are even allowed
to drive injection in the first place (discounting the score alone wasn't
enough -- an injected candidate still carries the same fixed base_score as
a genuinely recency-fetched one, so a heavily-discounted-but-nonzero boost
still let low-value injections compete)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch


def _run(coro):
    return asyncio.run(coro)


# Every non-early-return test needs fetch_entity_degrees mocked now (called
# unconditionally once target_scores is non-empty) -- default to returning
# {} (unknown degree -> the CONSERVATIVE fallback discount applies, not "no
# discount" -- see test_degree_lookup_gap_does_not_bypass_the_discount for
# why treating "unknown" as "full trust" was itself a live regression)
# unless a test cares about the discount specifically.
_NO_DEGREES = {}


def test_returns_empty_when_flag_disabled():
    with patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = False

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert boost_map == {}
        assert injected == []


def test_returns_empty_when_no_query_entities_extracted():
    with patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="what's the weather like today",  # no capitalized entities
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert boost_map == {}
        assert injected == []


def test_direct_query_entity_match_scores_full_and_related_scores_jaccard_when_rare():
    """A genuinely rare direct-matched entity (low degree, confirmed via
    fetch_entity_degrees) scores close to 1.0, and a Jaccard-related one
    keeps close to its own score -- both cleared by the frequency floor."""
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities",
        return_value=[{"name": "tesla", "shared_turns": 4, "jaccard": 0.5}],
    ) as mock_related, patch(
        "app.worker.fetch_entity_degrees", return_value={"nvidia": 5, "tesla": 5}
    ), patch(
        "app.worker.fetch_turns_mentioning_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_matches_for_turns",
        return_value={"turn-direct": ["nvidia"], "turn-related": ["tesla"]},
    ) as mock_matches:
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[
                    {"source": "falkor_chat", "uri": "turn-direct"},
                    {"source": "falkor_chat", "uri": "turn-related"},
                    {"source": "falkor_chat", "uri": "turn-nomatch"},
                ],
            )
        )

        # degree=5 < K=15 -> discount capped at 1.0, i.e. no discount for
        # either entity here.
        assert boost_map == {"turn-direct": 1.0, "turn-related": 0.5}
        assert injected == []
        mock_related.assert_called_once()
        assert mock_related.call_args.kwargs["name"] == "nvidia"
        mock_matches.assert_called_once()
        call_kwargs = mock_matches.call_args.kwargs
        assert set(call_kwargs["turn_ids"]) == {"turn-direct", "turn-related", "turn-nomatch"}
        assert "nvidia" in call_kwargs["target_names"]
        assert "tesla" in call_kwargs["target_names"]


def test_degree_lookup_gap_does_not_bypass_the_discount():
    """CRITICAL regression, found in code review: fetch_entity_degrees
    returning {} for an entity is indistinguishable, from the caller's
    side, between 'genuinely brand-new entity' and 'the Falkor call itself
    failed' (_safe_graph_query swallows exceptions into an empty result,
    never raises). A transient Falkor hiccup on this ONE call must not
    silently reopen the near-universal-entity bug this whole patch exists
    to fix -- 'orion' with NO degree data available must still be blocked
    from driving injection, not default back to a flat 1.0."""
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_degrees", return_value=_NO_DEGREES  # simulates the failure/unknown case
    ), patch(
        "app.worker.fetch_turns_mentioning_entities"
    ) as mock_mentioning, patch(
        "app.worker.fetch_entity_matches_for_turns", return_value={"turn-1": ["orion"]}
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="Orion, what do you think about the deploy?",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )

        # Must NOT be 1.0 -- the conservative fallback discount applies.
        # 0.15 matches app.worker._ENTITY_RELATEDNESS_MIN_INJECTION_SCORE.
        assert boost_map.get("turn-1", 0.0) < 0.15
        mock_mentioning.assert_not_called()
        assert injected == []


def test_near_universal_entity_gets_discounted_and_does_not_drive_injection():
    """The actual regression this patch fixes, live-confirmed: 'orion' at
    degree 282 must not score anywhere near 1.0, and must not be able to
    single-handedly pull generic filler turns into the pool."""
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_degrees", return_value={"orion": 282}
    ), patch(
        "app.worker.fetch_turns_mentioning_entities"
    ) as mock_mentioning, patch(
        "app.worker.fetch_entity_matches_for_turns", return_value={"turn-1": ["orion"]}
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="Orion, what do you think about the deploy?",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )

        # Discounted: 15/282 ~= 0.053, nowhere near the old flat 1.0.
        assert boost_map.get("turn-1", 0.0) < 0.1
        # Below the injection floor -- fetch_turns_mentioning_entities must
        # not even be called with "orion" as a target, since target_names
        # passed in would be empty after the floor filter.
        mock_mentioning.assert_not_called()
        assert injected == []


def test_specific_entity_clears_injection_floor_and_still_injects():
    """A genuinely rare/specific entity (low degree) must NOT be caught by
    the same floor that suppresses near-universal ones -- confirms the fix
    doesn't just kill injection outright."""
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_degrees", return_value={"nvidia": 23}
    ), patch(
        "app.worker.fetch_turns_mentioning_entities",
        return_value=[{"turn_id": "gpu-turn", "ts": "2026-05-01T00:00:00"}],
    ) as mock_mentioning, patch(
        "app.worker.fetch_chat_turns_by_id",
        return_value={"gpu-turn": ("show nvidia status", "gpu ok")},
    ), patch(
        "app.worker.fetch_entity_matches_for_turns", return_value={"gpu-turn": ["nvidia"]}
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(query_text="tell me about Nvidia", candidates=[])
        )

        mock_mentioning.assert_called_once()
        assert mock_mentioning.call_args.kwargs["target_names"] == ["nvidia"]
        assert len(injected) == 1
        assert injected[0]["id"] == "gpu-turn"
        # 15/23 ~= 0.65 -- discounted from 1.0 but well above the 0.15 floor.
        assert 0.5 < boost_map.get("gpu-turn", 0.0) < 1.0


def test_injects_entity_matched_turns_not_already_in_pool():
    """A turn that mentions the query's entity but was never fetched by the
    recency-windowed falkor_chat path gets fetched, hydrated, and returned
    as a new candidate."""
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_degrees", return_value={"nvidia": 5}  # rare -> clears the injection floor
    ), patch(
        "app.worker.fetch_turns_mentioning_entities",
        return_value=[{"turn_id": "old-turn-not-in-pool", "ts": "2025-10-21T00:00:00"}],
    ), patch(
        "app.worker.fetch_chat_turns_by_id",
        return_value={"old-turn-not-in-pool": ("what about nvidia?", "it's a GPU company")},
    ), patch(
        "app.worker.fetch_entity_matches_for_turns",
        return_value={"old-turn-not-in-pool": ["nvidia"]},
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "some-recent-unrelated-turn"}],
            )
        )

        assert len(injected) == 1
        frag = injected[0]
        assert frag["id"] == "old-turn-not-in-pool"
        assert frag["uri"] == "old-turn-not-in-pool"
        assert frag["source"] == "falkor_chat"
        assert "nvidia" in frag["text"].lower() or "GPU" in frag["text"]
        assert "entity_relatedness_injected" in frag["tags"]
        assert boost_map.get("old-turn-not-in-pool") == 1.0


def test_does_not_inject_turn_already_in_existing_pool():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_degrees", return_value={"nvidia": 5}  # rare -> clears the injection floor
    ), patch(
        "app.worker.fetch_turns_mentioning_entities",
        return_value=[{"turn_id": "already-here", "ts": "2025-10-21T00:00:00"}],
    ), patch("app.worker.fetch_chat_turns_by_id") as mock_hydrate, patch(
        "app.worker.fetch_entity_matches_for_turns", return_value={"already-here": ["nvidia"]}
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "already-here"}],
            )
        )

        assert injected == []
        mock_hydrate.assert_not_called()
        # Still boosted via the normal existing-candidate path.
        assert boost_map.get("already-here") == 1.0


def test_injected_turn_skipped_if_postgres_hydration_missing_it():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_degrees", return_value={"nvidia": 5}  # rare -> clears the injection floor
    ), patch(
        "app.worker.fetch_turns_mentioning_entities",
        return_value=[{"turn_id": "orphaned-turn", "ts": "2025-10-21T00:00:00"}],
    ), patch("app.worker.fetch_chat_turns_by_id", return_value={}), patch(
        "app.worker.fetch_entity_matches_for_turns", return_value={}
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(query_text="tell me about Nvidia", candidates=[])
        )
        assert injected == []


def test_failure_in_fetch_related_entities_degrades_not_raises():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", side_effect=RuntimeError("falkor down")
    ), patch("app.worker.fetch_entity_degrees", return_value=_NO_DEGREES), patch(
        "app.worker.fetch_turns_mentioning_entities", return_value=[]
    ), patch(
        "app.worker.fetch_entity_matches_for_turns", return_value={}
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert boost_map == {}
        assert injected == []


def test_failure_in_fetch_entity_degrees_degrades_not_raises():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch("app.worker.fetch_entity_degrees", side_effect=RuntimeError("falkor down")):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert boost_map == {}
        assert injected == []


def test_failure_in_fetch_turns_mentioning_entities_degrades_not_raises():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch("app.worker.fetch_entity_degrees", return_value=_NO_DEGREES), patch(
        "app.worker.fetch_turns_mentioning_entities", side_effect=RuntimeError("falkor down")
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        boost_map, injected = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert boost_map == {}
        assert injected == []
