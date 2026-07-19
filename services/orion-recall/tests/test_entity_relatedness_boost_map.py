"""_compute_entity_relatedness_boost_map (Phase 2 of entity-graph-reasoning,
docs/superpowers/specs/2026-07-19-recall-entity-graph-reasoning-arc.md):
best-effort (boost_map, injected_candidates) computation. Must degrade to
({}, []) on any failure -- never a hard dependency.

Returns a tuple now, not just a dict -- live evidence (6 real queries, 3
profiles) showed the boost-only design never actually changed a ranking:
falkor_chat's own fetch is recency-windowed with no query filter, so an
entity from an older turn never entered the candidate pool for the boost to
act on. injected_candidates is what closes that gap: real ChatTurn ids that
mention the query's own entities (or Jaccard-related ones), fetched and
hydrated independent of recency, added to the pool alongside whatever the
recency fetch already found."""

from __future__ import annotations

import asyncio
from unittest.mock import patch


def _run(coro):
    return asyncio.run(coro)


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


def test_direct_query_entity_match_scores_full_and_related_scores_jaccard():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities",
        return_value=[{"name": "tesla", "shared_turns": 4, "jaccard": 0.5}],
    ) as mock_related, patch(
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

        assert boost_map == {"turn-direct": 1.0, "turn-related": 0.5}
        assert injected == []
        mock_related.assert_called_once()
        assert mock_related.call_args.kwargs["name"] == "nvidia"
        mock_matches.assert_called_once()
        call_kwargs = mock_matches.call_args.kwargs
        assert set(call_kwargs["turn_ids"]) == {"turn-direct", "turn-related", "turn-nomatch"}
        assert "nvidia" in call_kwargs["target_names"]
        assert "tesla" in call_kwargs["target_names"]


def test_injects_entity_matched_turns_not_already_in_pool():
    """The actual fix for the live-confirmed gap: a turn that mentions the
    query's entity but was never fetched by the recency-windowed falkor_chat
    path gets fetched, hydrated, and returned as a new candidate."""
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
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
    ), patch("app.worker.fetch_turns_mentioning_entities", return_value=[]), patch(
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


def test_failure_in_fetch_turns_mentioning_entities_degrades_not_raises():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch("app.worker.fetch_turns_mentioning_entities", side_effect=RuntimeError("falkor down")):
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
