from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class ClusterResult:
    clusters: List[List[str]]
    labels: Dict[int, str]


class ConceptClusterer:
    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def _cluster_with_embeddings(
        self, items: List[str], embeddings: Dict[str, List[float]]
    ) -> List[List[str]]:
        clusters: List[List[str]] = []
        for cand in items:
            vec = embeddings.get(cand)
            if not vec:
                continue
            placed = False
            for cluster in clusters:
                anchor_vec = embeddings.get(cluster[0])
                if not anchor_vec:
                    continue
                if _cosine(vec, anchor_vec) >= self.threshold:
                    cluster.append(cand)
                    placed = True
                    break
            if not placed:
                clusters.append([cand])
        # ensure deterministic ordering
        return [sorted(c) for c in clusters]

    def _cluster_with_strings(self, items: List[str]) -> List[List[str]]:
        clusters: List[List[str]] = []
        for cand in items:
            placed = False
            for cluster in clusters:
                anchor = cluster[0]
                if _jaccard(cand, anchor) >= 0.6:
                    cluster.append(cand)
                    placed = True
                    break
            if not placed:
                clusters.append([cand])
        return [sorted(c) for c in clusters]

    def cluster(
        self,
        items: List[str],
        embeddings: Dict[str, List[float]] | None = None,
    ) -> ClusterResult:
        if embeddings:
            clusters = self._cluster_with_embeddings(items, embeddings)
        else:
            clusters = self._cluster_with_strings(items)

        labels: Dict[int, str] = {}
        for idx, cluster in enumerate(clusters):
            labels[idx] = ", ".join(cluster[:3])
        return ClusterResult(clusters=clusters, labels=labels)
