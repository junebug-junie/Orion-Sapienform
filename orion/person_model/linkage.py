"""Deterministic two-threshold linkage reducer (spec section 5, rev 3). No LLM.

Pure: every input that is not a string comparison (extractor reading, embedding
similarity, shared evidence, co-occurring names, stated home) arrives through a
``LinkageContext`` of plain callables/sets, so tests drive it with fixtures and
the dry-run CLI drives it with read-only live data.

Outcomes, in evaluation order:
1. ``topic``  -> no_match (auto_reject): a Concept from a topic-cluster producer.
2. ``junk``   -> junk (auto_reject): the label is a leaked prompt field (a key the
   prompt templates render).
3. ``code_identifier`` -> no_match (auto_reject): a path, channel, file name, or
   snake_case identifier. A shape rule (characters), not a word list.
4. ``facet``  -> facet_of (auto_merge group, link only, never a merge): "<name>'s X",
   or "<name> X" where the extractor does not read the whole label as a name.
5. otherwise Fellegi-Sunter score: >= upper -> same_as auto_merge;
   <= lower -> no_match auto_reject; between -> same_as needs_your_call.
   Guard: names that occur side by side in one source are forced to review.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import yaml

from orion.person_model.normalize import as_key, fold, is_snake_key, possessive_head, tokens
from orion.schemas.person_model import LinkageFeatureV1, PersonLinkageDecisionV1

_NS = uuid.UUID("0c4b1f6e-2d7a-4b8e-9f3c-5a1e7d9b2c40")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("linkage_config.yaml")


@dataclass(frozen=True)
class Anchor:
    node_id: str
    label: str
    aliases: Tuple[str, ...] = ()
    producer: str = ""
    evidence_refs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    node_id: str
    label: str
    kind: str = "substrate_node"  # substrate_node | referent_alias | mention_cluster
    node_labels: Tuple[str, ...] = ()  # graph labels, e.g. ("SubstrateNode", "Entity")
    producer: str = ""
    evidence_refs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkageConfig:
    version: str
    upper: float
    lower: float
    weights: Dict[str, Dict[str, float]]
    embedding_bins: Dict[str, float]
    jw_bins: Dict[str, float]
    topic_producers: FrozenSet[str]

    def w(self, feature: str, level: str) -> float:
        return float(self.weights[feature][level])


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> LinkageConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = LinkageConfig(
        version=str(d["version"]),
        upper=float(d["upper_threshold"]),
        lower=float(d["lower_threshold"]),
        weights={k: {lk: float(lv) for lk, lv in v.items()} for k, v in d["weights"].items()},
        embedding_bins={k: float(v) for k, v in d["embedding_bins"].items()},
        jw_bins={k: float(v) for k, v in d["jaro_winkler_bins"].items()},
        topic_producers=frozenset(d["rules"]["topic_producers"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: LinkageConfig) -> None:
    """The guarantees spec rev 3 promises the interlocutor, checked on every load."""
    if not cfg.lower < cfg.upper:
        raise ValueError("lower_threshold must be below upper_threshold")
    max_single = max(max(levels.values()) for levels in cfg.weights.values())
    if max_single >= cfg.upper:
        raise ValueError(f"a single feature ({max_single}) could force an auto-merge (upper={cfg.upper})")
    emb_max = max(cfg.weights["embedding_centered"].values())
    if emb_max >= cfg.upper - cfg.lower:
        raise ValueError("embedding weight must be capped below the review-band width")


@dataclass
class LinkageContext:
    # Whole label read as one person name by the pinned extractor; None = unknown.
    is_person_name: Callable[[str], Optional[bool]] = lambda label: None
    # The folded label occurs as an extractor person-name mention in the
    # interlocutor's own messages; None = extractor not run.
    mentioned_as_person: Callable[[str], Optional[bool]] = lambda label: None
    # Centered cosine between two labels; None = vector host unavailable.
    embedding_sim: Callable[[str, str], Optional[float]] = lambda a, b: None
    # Prompt-context keys harvested from the templates (normalize.prompt_context_keys).
    prompt_keys: FrozenSet[str] = frozenset()
    # Pairs of folded names seen side by side in one source ("jay, jay").
    cooccurring: FrozenSet[Tuple[str, str]] = frozenset()


def decision_id(anchor_id: str, candidate_id: str, version: str) -> str:
    return str(uuid.uuid5(_NS, f"{anchor_id}|{candidate_id}|{version}"))


def anchor_tokens(anchor: Anchor) -> List[List[str]]:
    names = [anchor.label, *anchor.aliases]
    out, seen = [], set()
    for n in names:
        t = tokens(n)
        if t and tuple(t) not in seen:
            seen.add(tuple(t))
            out.append(t)
    return out


def blocks(anchor: Anchor, cand: Candidate) -> bool:
    """Pair only when the candidate shares a name token, or a 4-char first-token
    prefix, with the anchor. Keeps the pair count in the thousands at most."""
    ct = set(tokens(cand.label))
    if not ct:
        return False
    for at in anchor_tokens(anchor):
        if ct & set(at):
            return True
        first_c = tokens(cand.label)[0]
        if len(at[0]) >= 4 and len(first_c) >= 4 and at[0][:4] == first_c[:4]:
            return True
    return False


def candidate_pairs(anchors: Sequence[Anchor], cands: Sequence[Candidate]) -> List[Tuple[Anchor, Candidate]]:
    out = []
    for a in anchors:
        for c in cands:
            if c.node_id == a.node_id:
                continue
            if blocks(a, c):
                out.append((a, c))
    out.sort(key=lambda ac: (ac[0].node_id, ac[1].node_id))
    return out


def _jaro_winkler(a: str, b: str) -> float:
    from rapidfuzz.distance import JaroWinkler

    return float(JaroWinkler.similarity(a, b))


def _metaphone(word: str) -> str:
    import jellyfish

    return jellyfish.metaphone(word)


def _best_name(anchor: Anchor, cand: Candidate) -> str:
    """The anchor name (label or alias) closest to the candidate, by JW then text."""
    names = sorted({anchor.label, *anchor.aliases})
    cf = fold(cand.label)
    return max(names, key=lambda n: (_jaro_winkler(fold(n), cf), n))


_IDENT_CHARS = re.compile(r"[/:*\\]|\.[A-Za-z]{1,5}\b|\w_\w")


def is_code_identifier(label: str) -> bool:
    """Paths, bus channels, file names, snake_case identifiers ("x/y.py",
    "svc:vision:frames", "a_b"). Hyphens are left alone: people have hyphenated
    names, and "name-thing" labels go through the facet rule instead."""
    s = str(label).strip()
    return is_snake_key(s) or bool(_IDENT_CHARS.search(s))


def rule_outcome(anchor: Anchor, cand: Candidate, cfg: LinkageConfig, ctx: LinkageContext) -> Optional[Tuple[str, str]]:
    if "Concept" in cand.node_labels and cand.producer in cfg.topic_producers:
        return "no_match", "topic_cluster_concept"
    if as_key(cand.label) in ctx.prompt_keys:
        return "junk", "prompt_context_key"
    if is_code_identifier(cand.label):
        return "no_match", "code_identifier"
    ats = anchor_tokens(anchor)
    head = possessive_head(cand.label)
    if head and any(fold(head) == at[0] and len(at) == 1 for at in ats):
        return "facet_of", "possessive_facet"
    ct = tokens(cand.label)
    for at in ats:
        if len(ct) > len(at) and ct[: len(at)] == at:
            # "<name> <more words>" is a possible full name only if the
            # interlocutor ever wrote it as a person's name (a name counts only
            # if she said it). Without mention data, fall back to the
            # extractor's reading of the bare label.
            mentioned = ctx.mentioned_as_person(cand.label)
            if mentioned is False or (mentioned is None and ctx.is_person_name(cand.label) is False):
                return "facet_of", "name_plus_non_name_words"
    return None


def features(anchor: Anchor, cand: Candidate, cfg: LinkageConfig, ctx: LinkageContext) -> List[LinkageFeatureV1]:
    name = _best_name(anchor, cand)
    a, c = fold(name), fold(cand.label)
    at, ct = tokens(name), tokens(cand.label)
    exact = a == c
    feats: List[LinkageFeatureV1] = []

    def add(fname: str, level: str, detail: Optional[str] = None) -> None:
        feats.append(LinkageFeatureV1(name=fname, level=level, weight=cfg.w(fname, level), detail=detail))

    add("exact_label", "agree" if exact else "disagree", f"{a!r} vs {c!r}")
    if exact:
        add("token_containment", "exact")
        # JW and phonetic are transforms of the same comparison; when the labels
        # are identical they carry no extra evidence, so they add nothing.
        feats.append(LinkageFeatureV1(name="jaro_winkler", level="implied_by_exact", weight=0.0))
        feats.append(LinkageFeatureV1(name="phonetic_first", level="implied_by_exact", weight=0.0))
    else:
        contained = bool(at and ct) and (set(at) <= set(ct) or set(ct) <= set(at))
        add("token_containment", "agree" if contained else "disagree")
        jw = _jaro_winkler(a, c)
        level = "high" if jw >= cfg.jw_bins["high"] else "mid" if jw >= cfg.jw_bins["mid"] else "low"
        add("jaro_winkler", level, f"{jw:.3f}")
        same_ph = bool(at and ct) and _metaphone(at[0]) == _metaphone(ct[0])
        add("phonetic_first", "agree" if same_ph else "disagree", f"{_metaphone(at[0]) if at else ''}/{_metaphone(ct[0]) if ct else ''}")
    # One feature for the one model: the extractor read this label as a person,
    # in her messages or as a bare label. Two readings, same source, one weight.
    readings = [r for r in (ctx.mentioned_as_person(cand.label), ctx.is_person_name(cand.label)) if r is not None]
    add("extractor_person", "missing" if not readings else "agree" if any(readings) else "disagree")
    sim = ctx.embedding_sim(name, cand.label)
    if exact:
        # The embedder lowercases and strips accents too (bge tokenizer), so for
        # labels that fold to the same text it returns ~1.0: the same comparison
        # as exact_label, not independent evidence. It adds nothing here.
        feats.append(LinkageFeatureV1(name="embedding_centered", level="implied_by_exact", weight=0.0,
                                      detail=None if sim is None else f"{sim:.3f}"))
    elif sim is None:
        add("embedding_centered", "missing")
    else:
        lvl = "agree" if sim >= cfg.embedding_bins["agree"] else "weak" if sim >= cfg.embedding_bins["weak"] else "disagree"
        add("embedding_centered", lvl, f"{sim:.3f}")
    add("producer_pair", "cross" if anchor.producer != cand.producer else "same", f"{anchor.producer or '?'} / {cand.producer or '?'}")
    shared = sorted(set(cand.evidence_refs) & set(anchor.evidence_refs))
    add("shared_evidence", "agree" if shared else "missing", ",".join(shared[:3]) or None)
    return feats


def decide(anchor: Anchor, cand: Candidate, cfg: LinkageConfig, ctx: LinkageContext, evidence: Optional[dict] = None) -> PersonLinkageDecisionV1:
    base = dict(
        decision_id=decision_id(anchor.node_id, cand.node_id, cfg.version),
        linkage_version=cfg.version,
        anchor_id=anchor.node_id,
        anchor_label=anchor.label,
        candidate_id=cand.node_id,
        candidate_label=cand.label,
        candidate_kind=cand.kind,
        upper_threshold=cfg.upper,
        lower_threshold=cfg.lower,
        evidence=dict(evidence or {}),
    )
    ruled = rule_outcome(anchor, cand, cfg, ctx)
    if ruled:
        relation, rule = ruled
        group = "auto_merge" if relation == "facet_of" else "auto_reject"
        return PersonLinkageDecisionV1(relation=relation, group=group, rule=rule, **base)
    feats = features(anchor, cand, cfg, ctx)
    score = round(sum(f.weight for f in feats), 3)
    if score >= cfg.upper:
        relation, group, rule = "same_as", "auto_merge", None
    elif score <= cfg.lower:
        relation, group, rule = "no_match", "auto_reject", None
    else:
        relation, group, rule = "same_as", "needs_your_call", "possible_match_band"
    pair = tuple(sorted((fold(anchor.label), fold(cand.label))))
    if pair in ctx.cooccurring and group != "needs_your_call":
        group, relation, rule = "needs_your_call", "same_as", "cooccurring_names_guard"
    return PersonLinkageDecisionV1(relation=relation, group=group, score=score, rule=rule, features=feats, **base)


def run(anchors: Sequence[Anchor], cands: Sequence[Candidate], cfg: LinkageConfig, ctx: LinkageContext,
        evidence_for: Callable[[Anchor, Candidate], dict] = lambda a, c: {}) -> List[PersonLinkageDecisionV1]:
    """All decisions, in a stable order. A candidate that blocks against several
    anchors gets one decision per anchor; ``best_per_candidate`` collapses them."""
    return [decide(a, c, cfg, ctx, evidence_for(a, c)) for a, c in candidate_pairs(anchors, cands)]


_GROUP_RANK = {"auto_merge": 0, "needs_your_call": 1, "auto_reject": 2}


def best_per_candidate(decisions: Iterable[PersonLinkageDecisionV1]) -> List[PersonLinkageDecisionV1]:
    """One decision per candidate: the strongest claim wins (a merge or facet over
    a review over a reject; then higher score). Stable for identical input."""
    best: Dict[str, PersonLinkageDecisionV1] = {}
    for d in decisions:
        cur = best.get(d.candidate_id)
        key = (-_GROUP_RANK[d.group], d.score if d.score is not None else 0.0, d.anchor_id)
        if cur is None or key > (-_GROUP_RANK[cur.group], cur.score if cur.score is not None else 0.0, cur.anchor_id):
            best[d.candidate_id] = d
    return [best[k] for k in sorted(best)]


HOME_ALIAS = "home"  # the attribute category name a descriptor alias claims


def stated_home_conflicts(
    alias_rows: Iterable[dict],
    *,
    person: Anchor,
    stated_home: str,
    place_labels: Dict[str, str],
    cfg: LinkageConfig,
    stated_ref: str,
) -> List[PersonLinkageDecisionV1]:
    """A 'home' descriptor alias on a place that is not the person's stated home.

    The person's own answer decides it, so it is an auto outcome (auto_reject:
    the alias stops claiming home; the place itself is untouched)."""
    home_tokens = set(tokens(stated_home))
    out = []
    for row in alias_rows:
        if row.get("referent_kind") != "place" or fold(row.get("alias_norm", "")) != HOME_ALIAS:
            continue
        place = place_labels.get(row["node_id"], row["node_id"])
        if home_tokens & set(tokens(place)):
            continue
        cid = f"{row['node_id']}#alias:{row['alias_norm']}"
        out.append(
            PersonLinkageDecisionV1(
                decision_id=decision_id(person.node_id, cid, cfg.version),
                linkage_version=cfg.version,
                anchor_id=person.node_id,
                anchor_label=person.label,
                candidate_id=cid,
                candidate_label=f"home -> {place}",
                candidate_kind="referent_alias",
                relation="not_home",
                group="auto_reject",
                upper_threshold=cfg.upper,
                lower_threshold=cfg.lower,
                rule="stated_home_answer",
                evidence={"stated_home": stated_home, "stated_ref": stated_ref, "place_label": place,
                          "grounded_in": row.get("grounded_in")},
            )
        )
    return out


def repeated_name_splits(clusters: Dict[str, dict], cfg: LinkageConfig) -> List[PersonLinkageDecisionV1]:
    """A name listed twice side by side in one source ("Jay, Jay") is two people
    until the interlocutor says otherwise: one needs_your_call card per name.

    ``clusters``: folded name -> {"label", "node_id", "repeat_sources": [...]}."""
    out = []
    for key in sorted(clusters):
        c = clusters[key]
        if not c.get("repeat_sources"):
            continue
        out.append(
            PersonLinkageDecisionV1(
                decision_id=decision_id(c["node_id"], c["node_id"] + "#split", cfg.version),
                linkage_version=cfg.version,
                anchor_id=c["node_id"],
                anchor_label=c["label"],
                candidate_id=c["node_id"] + "#split",
                candidate_label=c["label"],
                candidate_kind="mention_cluster",
                relation="distinct",
                group="needs_your_call",
                upper_threshold=cfg.upper,
                lower_threshold=cfg.lower,
                rule="repeated_name_in_one_list",
                evidence={"repeat_sources": c["repeat_sources"]},
            )
        )
    return out
