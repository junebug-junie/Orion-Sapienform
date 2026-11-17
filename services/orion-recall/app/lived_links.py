"""Link-following over ``episode_memory_referent`` (issue #3534, "Recall rev 2" step 2).

A memory that fits the turn pulls in memories that share a referent with it (a person, a
project, a place). The style is personalized PageRank over the memory-referent graph
(HippoRAG 2, Gutierrez et al. 2025), truncated to two hops:

1. **Seeds.** The referents of the top few candidates (before rerank), plus referents whose
   name appears in the message.
2. **Spread.** Seed mass flows referent -> memory -> referent -> memory, with restart
   ``ALPHA`` back to the seeds. Each referent splits its mass over its memories, so a referent
   shared by many memories passes little to each one.
3. **Hubs.** A referent on more than ``HUB_MAX_DEGREE`` memories (the two main participants, on
   most of them) says nothing about this turn and is left out of the graph.

Neighbours are only *candidates*: the reranker scores every one, and the quality bar applies.
Nothing reaches the prompt from here without a rerank.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from orion.memory.lived_memory import KIND_EPISODE, doc_id

ALPHA = 0.5
HUB_MAX_DEGREE = 30
SEED_DOCS = 5
NAME_MIN_CHARS = 4
NAME_SEED_WEIGHT = 1.0

# Short function words that showed up as alias text ("that"); names must not be one of them.
_STOPWORDS = frozenset(
    """that this those these there here then them they with from what when where which about into
    over just only very really also some much more most other same such than your yours ours
    mine thing things stuff home""".split()
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_NON_ALNUM.sub(" ", str(text or "").lower()).split())


def usable_name(alias: str) -> bool:
    """A name alias worth matching: long enough, has a letter, not a function word."""
    a = normalize(alias)
    if len(a) < NAME_MIN_CHARS or not any(c.isalpha() for c in a):
        return False
    return a not in _STOPWORDS


class ReferentGraph:
    """Bipartite memory <-> referent graph over episode memories (doc ids ``episode_memory:<id>``)."""

    def __init__(self, edges: Iterable[Tuple[str, str]], names: Iterable[Tuple[str, str]] = (), *,
                 hub_max_degree: int = HUB_MAX_DEGREE) -> None:
        refs_of: Dict[str, Set[str]] = defaultdict(set)
        docs_of: Dict[str, Set[str]] = defaultdict(set)
        for memory_id, ref in edges:
            if not memory_id or not ref:
                continue
            d = doc_id(KIND_EPISODE, str(memory_id))
            refs_of[d].add(str(ref))
            docs_of[str(ref)].add(d)
        self.hubs = {r for r, ds in docs_of.items() if len(ds) > hub_max_degree}
        self.docs_of = {r: ds for r, ds in docs_of.items() if r not in self.hubs}
        self.refs_of = {d: {r for r in rs if r not in self.hubs} for d, rs in refs_of.items()}
        self.refs_of = {d: rs for d, rs in self.refs_of.items() if rs}
        self.names: Dict[str, Set[str]] = defaultdict(set)
        for alias, ref in names:
            ref = str(ref or "")
            if ref in self.docs_of and usable_name(alias):
                self.names[normalize(alias)].add(ref)

    def __len__(self) -> int:
        return len(self.refs_of)

    def mentions(self, text: str) -> Set[str]:
        """Referents whose name appears in ``text`` as whole words."""
        norm = f" {normalize(text)} "
        if not norm.strip():
            return set()
        out: Set[str] = set()
        for alias, refs in self.names.items():
            if f" {alias} " in norm:
                out |= refs
        return out

    def spread(self, seed_docs: Sequence[str], seed_refs: Iterable[str] = (), *,
               exclude: Set[str] = frozenset(), allowed: Optional[Callable[[str], bool]] = None,
               top_k: int = 6) -> List[Tuple[str, float]]:
        """Two-hop personalized PageRank from the seeds. Returns up to ``top_k`` memories not in
        ``exclude`` (and passing ``allowed``), best first, with their scores."""
        p: Dict[str, float] = defaultdict(float)
        for d in seed_docs:
            refs = self.refs_of.get(d) or set()
            for r in refs:
                p[r] += 1.0 / len(refs)
        for r in seed_refs:
            if r in self.docs_of:
                p[r] += NAME_SEED_WEIGHT
        total = sum(p.values())
        if total <= 0:
            return []
        p = {r: v / total for r, v in p.items()}

        def to_docs(ref_mass: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = defaultdict(float)
            for r, m in ref_mass.items():
                ds = self.docs_of.get(r) or ()
                for d in ds:
                    out[d] += m / len(ds)
            return out

        def to_refs(doc_mass: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = defaultdict(float)
            for d, m in doc_mass.items():
                rs = self.refs_of.get(d) or ()
                for r in rs:
                    out[r] += m / len(rs)
            return out

        hop1 = to_docs(p)
        back = to_refs(hop1)
        mixed: Dict[str, float] = defaultdict(float)
        for r, v in p.items():
            mixed[r] += ALPHA * v
        for r, v in back.items():
            mixed[r] += (1.0 - ALPHA) * v
        scores = to_docs(mixed)
        ranked = sorted(
            ((d, s) for d, s in scores.items()
             if s > 0 and d not in exclude and (allowed is None or allowed(d))),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[: max(0, int(top_k))]


async def load_graph(pool: Any) -> ReferentGraph:
    from orion.memory.lived_memory import REFERENT_EDGES_SQL, REFERENT_NAMES_SQL

    edges = [(str(r["memory_id"]), str(r["referent_key"])) for r in await pool.fetch(REFERENT_EDGES_SQL)]
    names = [(str(r["alias"]), str(r["referent_key"])) for r in await pool.fetch(REFERENT_NAMES_SQL)]
    return ReferentGraph(edges, names)


__all__ = ["ALPHA", "HUB_MAX_DEGREE", "ReferentGraph", "SEED_DOCS", "load_graph", "normalize", "usable_name"]
