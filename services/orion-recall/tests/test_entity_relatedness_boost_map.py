"""_compute_entity_relatedness_boost_map (Phase 2 of entity-graph-reasoning,
docs/superpowers/specs/2026-07-19-recall-entity-graph-reasoning-arc.md):
best-effort boost-map computation feeding fuse_candidates' entity_boost_map
parameter. Must degrade to {} on any failure -- never a hard dependency."""

from __future__ import annotations

import asyncio
from unittest.mock import patch


def _run(coro):
    return asyncio.run(coro)


def test_returns_empty_when_flag_disabled():
    with patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = False

        from app.worker import _compute_entity_relatedness_boost_map

        out = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert out == {}


def test_returns_empty_when_no_query_entities_extracted():
    with patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        out = _run(
            _compute_entity_relatedness_boost_map(
                query_text="what's the weather like today",  # no capitalized entities
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert out == {}


def test_returns_empty_when_no_falkor_chat_candidates():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        out = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "sql_chat", "uri": "turn-1"}],  # wrong source
            )
        )
        assert out == {}


def test_direct_query_entity_match_scores_full_and_related_scores_jaccard():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities",
        return_value=[{"name": "tesla", "shared_turns": 4, "jaccard": 0.5}],
    ) as mock_related, patch(
        "app.worker.fetch_entity_matches_for_turns",
        return_value={"turn-direct": ["nvidia"], "turn-related": ["tesla"]},
    ) as mock_matches:
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        out = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[
                    {"source": "falkor_chat", "uri": "turn-direct"},
                    {"source": "falkor_chat", "uri": "turn-related"},
                    {"source": "falkor_chat", "uri": "turn-nomatch"},
                ],
            )
        )

        assert out == {"turn-direct": 1.0, "turn-related": 0.5}
        mock_related.assert_called_once()
        assert mock_related.call_args.kwargs["name"] == "nvidia"
        mock_matches.assert_called_once()
        call_kwargs = mock_matches.call_args.kwargs
        assert set(call_kwargs["turn_ids"]) == {"turn-direct", "turn-related", "turn-nomatch"}
        assert "nvidia" in call_kwargs["target_names"]
        assert "tesla" in call_kwargs["target_names"]


def test_failure_in_fetch_related_entities_degrades_not_raises():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", side_effect=RuntimeError("falkor down")
    ), patch("app.worker.fetch_entity_matches_for_turns", return_value={}):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        out = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert out == {}


def test_failure_in_fetch_entity_matches_degrades_not_raises():
    with patch("app.worker.settings") as mock_settings, patch(
        "app.worker.fetch_related_entities", return_value=[]
    ), patch("app.worker.fetch_entity_matches_for_turns", side_effect=RuntimeError("falkor down")):
        mock_settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED = True

        from app.worker import _compute_entity_relatedness_boost_map

        out = _run(
            _compute_entity_relatedness_boost_map(
                query_text="tell me about Nvidia",
                candidates=[{"source": "falkor_chat", "uri": "turn-1"}],
            )
        )
        assert out == {}
