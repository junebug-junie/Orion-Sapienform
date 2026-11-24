"""Unit tests for the field-native prediction-error instruments in
``orion/substrate/prediction_error.py`` -- 0-1 surprise scores diffing successive
reducer-projection snapshots (not ``SelfStateV1``, not ``tensions.py``'s bucket
vocabulary; see the Sentience Striving Program charter §9b item 3)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.chat_projection import ChatTurnStateV1
from orion.structural_mass.git_delta import GitChurnDelta
from orion.structural_mass.graph_delta import GraphStructuralDelta
from orion.structural_mass.pr_lifecycle import PrLifecycleDelta
from orion.substrate.prediction_error import (
    CodebaseMassBaseline,
    _DomainEwmaBaseline,
    codebase_prediction_error,
)

_NOW = datetime(2026, 7, 21, 0, 0, 0, tzinfo=timezone.utc)


# biometrics_prediction_error's tests went with it (retired 2026-11-18, #3575; successor
# orion/substrate/prediction_joint_surprise.py, tested in test_prediction_joint_surprise.py).


# execution_prediction_error's tests went with it (retired 2026-11-24, #3576; successors
# node:prediction.chat_turn's more-steps bet and node:prediction.body_joint, tested in
# test_prediction_turn_surprise.py / test_prediction_joint_surprise.py).


def test_execution_prediction_error_is_gone() -> None:
    import orion.substrate.prediction_error as pe

    assert not hasattr(pe, "execution_prediction_error")
    assert not hasattr(pe, "execution_raw_reading")
    assert not hasattr(pe, "_score_and_update_ewma_prediction_error")


# -- chat_prediction_error was deleted 2026-11-13 (PE suite rebuild step 6);
# its tests went with it. The chat reducer hint below is still live.


def _chat_turn(
    turn_id: str,
    *,
    word_count: int = 0,
    repair_pressure_level: float = 0.0,
    last_updated_at: datetime | None = None,
) -> ChatTurnStateV1:
    return ChatTurnStateV1(
        trace_id=f"hub.chat:athena:{turn_id}",
        turn_id=turn_id,
        session_id="orion_journal",
        node_id="athena",
        observed_at=_NOW,
        word_count=word_count,
        repair_pressure_level=repair_pressure_level,
        last_updated_at=last_updated_at or _NOW,
    )


def test_chat_pressure_hints_no_longer_carry_topic_coherence() -> None:
    from orion.substrate.chat_loop.grammar_extract import compute_chat_pressure_hints

    hints = compute_chat_pressure_hints(_chat_turn("t1", word_count=30, repair_pressure_level=0.4))
    assert set(hints) == {"conversation_load", "repair_pressure"}


# route_prediction_error and bus_synaptic_prediction_error were deleted 2026-11
# (PE suite rebuild step 5); their tests went with them.


class TestCodebaseMassBaselineSerialization:
    """Wire-format round-trip for the consumer patch's
    substrate_codebase_mass_baseline table -- CodebaseMassBaseline.to_json_dict()/
    from_json_dict()."""

    def test_round_trip_preserves_all_three_sub_baselines(self) -> None:
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=1.5, variance=2.5, n=3),
            pr=_DomainEwmaBaseline(ewma=4.5, variance=5.5, n=6),
            graph=_DomainEwmaBaseline(ewma=7.5, variance=8.5, n=9),
        )
        restored = CodebaseMassBaseline.from_json_dict(baseline.to_json_dict())
        assert restored == baseline

    def test_round_trip_default_baseline(self) -> None:
        baseline = CodebaseMassBaseline()
        restored = CodebaseMassBaseline.from_json_dict(baseline.to_json_dict())
        assert restored == baseline

    def test_from_json_dict_tolerates_missing_keys(self) -> None:
        """A row from before some future field addition, or a hand-edited
        JSON blob missing a sub-domain entirely, must not crash -- defaults
        to a fresh cold-start sub-baseline for whatever's missing."""
        restored = CodebaseMassBaseline.from_json_dict({"git": {"ewma": 1.0, "variance": 2.0, "n": 3}})
        assert restored.git == _DomainEwmaBaseline(ewma=1.0, variance=2.0, n=3)
        assert restored.pr == _DomainEwmaBaseline()
        assert restored.graph == _DomainEwmaBaseline()

    def test_from_json_dict_empty_dict_is_fresh_baseline(self) -> None:
        assert CodebaseMassBaseline.from_json_dict({}) == CodebaseMassBaseline()


class TestCodebasePredictionError:
    """Contract patch (docs/superpowers/specs/2026-07-30-codebase-mass-signal-
    design.md) -- composite scoring across all three structural_mass producer
    domains (git/pr/graph), each on its own independent EWMA baseline since each
    producer runs on its own interval and a given tick may populate only some of
    them. Not wired into any tick/bus channel yet -- that's a separate, later
    patch; this only proves the scoring function itself is correct."""

    def _git(
        self,
        *,
        lines_added: int = 0,
        lines_removed: int = 0,
        commit_count: int = 1,
        files_added: int = 0,
        files_deleted: int = 0,
        files_modified: int = 0,
    ) -> GitChurnDelta:
        return GitChurnDelta(
            prev_sha="a" * 40,
            head_sha="b" * 40,
            commit_count=commit_count,
            files_added=files_added,
            files_deleted=files_deleted,
            files_modified=files_modified,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

    def _pr(
        self,
        *,
        submitted_count: int = 0,
        merged_count: int = 0,
        closed_without_merge_count: int = 0,
    ) -> PrLifecycleDelta:
        _now = datetime(2026, 7, 30, tzinfo=timezone.utc)
        return PrLifecycleDelta(
            since=_now,
            until=_now,
            submitted_count=submitted_count,
            merged_count=merged_count,
            closed_without_merge_count=closed_without_merge_count,
        )

    def _graph(
        self,
        *,
        node_count_delta: int = 0,
        edge_count_delta: int = 0,
        community_count_delta: int = 0,
        god_node_jaccard_similarity: float | None = 1.0,
    ) -> GraphStructuralDelta:
        return GraphStructuralDelta(
            node_count_delta=node_count_delta,
            edge_count_delta=edge_count_delta,
            community_count_delta=community_count_delta,
            god_node_jaccard_similarity=god_node_jaccard_similarity,
        )

    def test_no_producers_fired_is_explicit_no_op_and_baseline_untouched(self) -> None:
        """All three domains None (no producer fired this tick) must read 0.0
        without mutating any sub-baseline -- a real 'nothing happened,' not a
        computed reading against absent data."""
        baseline = CodebaseMassBaseline()
        result = codebase_prediction_error(
            git_delta=None, pr_delta=None, graph_delta=None, baseline=baseline
        )
        assert result.score == 0.0
        assert result.baseline == baseline
        assert result.baseline.git.n == 0

    def test_first_tick_is_cold_start_but_seeds_baseline(self) -> None:
        """No established baseline yet for the one domain that fired -- must
        return 0.0 (no z-score to compute against), but must seed that domain's
        sub-baseline so the next tick has something real to compare against.
        Other two domains' sub-baselines stay untouched (n=0)."""
        baseline = CodebaseMassBaseline()
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=400, lines_removed=200),
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        assert result.score == 0.0
        assert result.baseline.git.ewma == pytest.approx(600.0)
        assert result.baseline.git.n == 1
        assert result.baseline.pr.n == 0
        assert result.baseline.graph.n == 0

    def test_single_domain_scores_against_its_own_established_baseline(self) -> None:
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=5))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=300, lines_removed=220),  # magnitude 520
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        # zscore = (520 - 500) / sqrt(10_000) = 0.2 -> mean of just this one zscore
        assert result.score == pytest.approx(0.2 / 3.0)
        assert result.baseline.git.n == 6

    def test_two_domains_average_their_independent_zscores(self) -> None:
        """git and pr both fire this tick; graph doesn't. Composite score is the
        mean of exactly the two available z-scores, not diluted by a phantom
        third 0.0 for the domain that didn't observe anything."""
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=5),
            pr=_DomainEwmaBaseline(ewma=2.0, variance=1.0, n=5),
        )
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=300, lines_removed=220),  # magnitude 520, z=0.2
            pr_delta=self._pr(submitted_count=3, merged_count=2),  # magnitude 5, z=(5-2)/1=3.0
            graph_delta=None,
            baseline=baseline,
        )
        expected_mean_z = (0.2 + 3.0) / 2
        assert result.score == pytest.approx(min(1.0, expected_mean_z / 3.0))
        assert result.baseline.git.n == 6
        assert result.baseline.pr.n == 6
        assert result.baseline.graph.n == 0

    def test_all_three_domains_average_together(self) -> None:
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=5),
            pr=_DomainEwmaBaseline(ewma=2.0, variance=1.0, n=5),
            graph=_DomainEwmaBaseline(ewma=100.0, variance=400.0, n=5),
        )
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=300, lines_removed=220),  # z=0.2
            pr_delta=self._pr(submitted_count=3, merged_count=2),  # z=3.0
            graph_delta=self._graph(node_count_delta=120, edge_count_delta=0),  # magnitude 120, z=(120-100)/20=1.0
            baseline=baseline,
        )
        expected_mean_z = (0.2 + 3.0 + 1.0) / 3
        assert result.score == pytest.approx(min(1.0, expected_mean_z / 3.0))

    def test_saturates_exactly_at_zscore_saturation_boundary(self) -> None:
        """No off-by-one at the saturation ceiling: a single domain's zscore of
        exactly 3.0 must score exactly 1.0."""
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=500.0, variance=100.0, n=10))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=530, lines_removed=0),  # magnitude 530, z=3.0 exactly
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        assert result.score == pytest.approx(1.0)

    def test_large_catch_up_diff_spanning_missed_ticks_scores_high(self) -> None:
        """A big diff accumulated across several missed ticks should read as a
        real spike relative to the domain's own established baseline."""
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=20))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=9000, lines_removed=6000, commit_count=14, files_modified=40),
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        # zscore = (15_000 - 500) / sqrt(10_000) = 145.0 -> far past saturation
        assert result.score == pytest.approx(1.0)

    def test_below_baseline_tick_clamps_to_zero_before_averaging(self) -> None:
        """A quieter-than-usual git tick (negative z-score) clamps to 0.0 before
        averaging with a genuinely surprising PR tick -- the PR spike must not
        be diluted toward 0 by an unclamped negative contribution from git."""
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=5000.0, variance=1_000_000.0, n=10),
            pr=_DomainEwmaBaseline(ewma=2.0, variance=1.0, n=10),
        )
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=10, lines_removed=5),  # magnitude 15, well below baseline
            pr_delta=self._pr(submitted_count=3, merged_count=2),  # magnitude 5, z=3.0
            graph_delta=None,
            baseline=baseline,
        )
        # If git's negative zscore weren't clamped, mean would pull below 3.0/3.0=1.0.
        # Clamped to 0.0: mean(0.0, 3.0) / 3.0 = 0.5.
        assert result.score == pytest.approx(0.5)

    def test_git_only_below_baseline_tick_is_exactly_zero(self) -> None:
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=5000.0, variance=1_000_000.0, n=10))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=10, lines_removed=5),
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        assert result.score == 0.0

    def test_baseline_state_flows_independently_of_module_globals(self) -> None:
        """Two independent baseline threads must not interfere with each other
        -- state is entirely caller-owned, no hidden module-level mutation."""
        baseline_a = CodebaseMassBaseline()
        baseline_b = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=9999.0, variance=1.0, n=50))
        result_a = codebase_prediction_error(
            git_delta=self._git(lines_added=100, lines_removed=0), pr_delta=None, graph_delta=None,
            baseline=baseline_a,
        )
        result_b = codebase_prediction_error(
            git_delta=self._git(lines_added=100, lines_removed=0), pr_delta=None, graph_delta=None,
            baseline=baseline_b,
        )
        assert result_a.baseline.git.n == 1
        assert result_b.baseline.git.n == 51
        assert result_a.baseline.git.ewma != result_b.baseline.git.ewma
