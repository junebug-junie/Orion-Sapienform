"""Per-stream scene label habituation (observed → believed tier)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BeliefObserveResult:
    believed_labels: frozenset[str]
    added: frozenset[str]
    removed: frozenset[str]
    observation_count: int


class SceneBeliefTracker:
    def __init__(self, *, vote_n: int = 3, enter_votes: int = 2, exit_votes: int = 1) -> None:
        self._vote_n = vote_n
        self._enter_votes = enter_votes
        self._exit_votes = exit_votes
        self._ring: deque[frozenset[str]] = deque(maxlen=vote_n)
        self._believed: frozenset[str] = frozenset()
        self._last_nonempty: frozenset[str] = frozenset()

    @property
    def believed_labels(self) -> frozenset[str]:
        return self._believed

    def _effective_labels(self, observed: frozenset[str]) -> frozenset[str]:
        return observed if observed else self._last_nonempty

    def _label_counts(self) -> dict[str, int]:
        """Enter votes: empty ring slots inherit last non-empty labels (flicker fix)."""
        counts: dict[str, int] = {}
        for entry in self._ring:
            for label in self._effective_labels(entry):
                counts[label] = counts.get(label, 0) + 1
        return counts

    def _raw_label_counts(self) -> dict[str, int]:
        """Exit votes: count only labels present in each raw observation."""
        counts: dict[str, int] = {}
        for entry in self._ring:
            for label in entry:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def observe(self, observed: frozenset[str]) -> BeliefObserveResult:
        if observed:
            self._last_nonempty = observed
        self._ring.append(observed)
        is_empty = not observed

        candidates = set(self._believed)
        for entry in self._ring:
            candidates.update(self._effective_labels(entry))

        enter_counts = self._label_counts()
        exit_counts = self._raw_label_counts()
        new_believed = set(self._believed)
        for label in candidates:
            enter_count = enter_counts.get(label, 0)
            exit_count = exit_counts.get(label, 0)
            can_enter = enter_count >= self._enter_votes and (
                len(self._ring) >= self._vote_n or (is_empty and enter_count >= self._enter_votes)
            )
            can_exit = len(self._ring) >= self._vote_n and exit_count <= self._exit_votes
            if label not in self._believed and can_enter:
                new_believed.add(label)
            elif label in self._believed and can_exit:
                new_believed.discard(label)

        prev = self._believed
        self._believed = frozenset(new_believed)
        added = self._believed - prev
        removed = prev - self._believed
        return BeliefObserveResult(
            believed_labels=self._believed,
            added=added,
            removed=removed,
            observation_count=len(self._ring),
        )

    def enrich_evidence(self, evidence: dict[str, Any]) -> dict[str, Any]:
        out = dict(evidence)
        out["believed_hard_labels"] = sorted(self._believed)
        out["belief"] = {
            "schema": "scene_belief.v1",
            "vote_n": self._vote_n,
            "enter_votes": self._enter_votes,
            "exit_votes": self._exit_votes,
            "observation_count": len(self._ring),
        }
        return out


class SceneBeliefRegistry:
    def __init__(self, *, vote_n: int = 3, enter_votes: int = 2, exit_votes: int = 1) -> None:
        self._vote_n = vote_n
        self._enter_votes = enter_votes
        self._exit_votes = exit_votes
        self._trackers: dict[str, SceneBeliefTracker] = {}

    def _tracker(self, stream_id: str) -> SceneBeliefTracker:
        if stream_id not in self._trackers:
            self._trackers[stream_id] = SceneBeliefTracker(
                vote_n=self._vote_n,
                enter_votes=self._enter_votes,
                exit_votes=self._exit_votes,
            )
        return self._trackers[stream_id]

    def observe(self, stream_id: str, observed: frozenset[str]) -> BeliefObserveResult:
        return self._tracker(stream_id).observe(observed)

    def enrich_evidence(self, stream_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return self._tracker(stream_id).enrich_evidence(evidence)
