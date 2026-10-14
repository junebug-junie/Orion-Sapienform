"""Reading resonance, patch A: crosswalk packet, judge prompt, deterministic citation checks.

Design: docs/superpowers/specs/2026-10-13-reading-resonance-design.md (PR #2642).

After a read completes, gather the few things Orion already holds that the read sits
near -- priors with their findings and revision trail, lived questions with their
answers, atlas concepts with their 1-hop substrate neighbourhood, crystallized memories
of conversations with Juniper, curiosity findings, dreams, open threads -- freeze them
into a ResonancePacketV1, and hand that to one judge call. The judge's answer is only
accepted when every ref it cites is one the packet handed it and its quote is really in
the read.

Retrieval scores (resonance_similarity.py) only ORDER candidates inside fixed per-kind
budgets; they are not shown to the judge and never decide "interesting". This module is
pure: all store I/O lives in the caller (the replay eval in patch A).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from orion.schemas.reading_resonance import (
    ReadingResonanceCheckV1,
    ReadingResonanceJudgmentV1,
    ResonanceCandidateV1,
    ResonanceCitationV1,
    ResonanceNeighborV1,
    ResonancePacketV1,
)

from .resonance_similarity import CENTERED_OFFER_FLOOR

# Per-kind packet budget (sum 24 = ResonancePacketV1.candidates max_length).
KIND_BUDGET: dict[str, int] = {
    "self_prior": 4,
    "lived_question": 2,
    "lived_answer": 2,
    "atlas_concept": 4,
    "crystallization": 4,
    "curiosity_finding": 3,
    "dream": 2,
    "open_thread": 3,
}
EXCERPT_CHARS = 600
NEIGHBOR_CHARS = 300
PACKET_NAMESPACE = uuid.UUID("6f1c2d0e-5d43-4f0a-9a57-2c7b1e0f7a11")
SUMMARY_MIN_SENTENCES = 3
SUMMARY_MAX_SENTENCES = 5
# Prompt lines shorter than this are too generic to call an echo ("Return JSON.").
ECHO_MIN_CHARS = 30
# A read_quote must be at least this many whole words, matched on word boundaries:
# "and the" is inside every read and proves nothing.
QUOTE_MIN_WORDS = 5


@dataclass(frozen=True)
class PoolDoc:
    """One thing Orion holds, as re-read from its record store."""

    ref_kind: str
    ref_id: str
    store: str
    excerpt: str
    neighborhood: tuple[ResonanceNeighborV1, ...] = ()
    # Open concern arcs (1 open on 2026-10-13) are what Orion and Juniper are in the middle
    # of: offered without the similarity floor, still inside their kind's budget.
    floorless: bool = False


@dataclass(frozen=True)
class ScoredDoc:
    doc: PoolDoc
    centered_cos: float  # floor is applied to this
    rank: float  # centered cos minus half the doc's hubness; orders within a kind


@dataclass
class ReadInput:
    seed_id: str
    root_request_id: str | None
    title: str
    url: str
    summary: str
    learned: str
    retained_texts: list[tuple[str, str]] = field(default_factory=list)  # (sha256, text)


def packet_id_for(seed_id: str) -> str:
    return str(uuid.uuid5(PACKET_NAMESPACE, f"resonance:{seed_id}"))


def _clip(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _excerpt_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()[:16]


def build_packet(
    read: ReadInput,
    scored: Iterable[ScoredDoc],
    *,
    built_at: datetime,
    exclude_ref_ids: Iterable[str] = (),
    floor: float = CENTERED_OFFER_FLOOR,
    budget: dict[str, int] | None = None,
) -> ResonancePacketV1:
    """Deterministic: same read, same scored pool -> same packet.

    Per kind, best rank first: drop excluded refs (the read itself, its root request),
    drop below-floor candidates (except floorless docs), dedup identical excerpts
    (Chroma holds duplicate crystallization rows), take up to the kind's budget.
    """
    budget = KIND_BUDGET if budget is None else budget
    excluded = set(exclude_ref_ids)
    by_kind: dict[str, list[ScoredDoc]] = {}
    for s in scored:
        if s.doc.ref_kind not in budget or s.doc.ref_id in excluded:
            continue
        if not s.doc.floorless and s.centered_cos < floor:
            continue
        by_kind.setdefault(s.doc.ref_kind, []).append(s)
    candidates: list[ResonanceCandidateV1] = []
    for kind in budget:  # fixed kind order -> stable packet
        seen: set[str] = set()
        seen_refs: set[str] = set()
        taken = 0
        for s in sorted(by_kind.get(kind, []), key=lambda x: (-x.rank, x.doc.ref_id)):
            if taken >= budget[kind]:
                break
            key = _excerpt_key(s.doc.excerpt)
            if not s.doc.excerpt.strip() or key in seen or s.doc.ref_id in seen_refs:
                continue
            seen.add(key)
            seen_refs.add(s.doc.ref_id)
            candidates.append(ResonanceCandidateV1(
                ref_kind=kind,  # type: ignore[arg-type]
                ref_id=s.doc.ref_id,
                store=s.doc.store,
                excerpt=_clip(s.doc.excerpt, EXCERPT_CHARS),
                neighborhood=list(s.doc.neighborhood[:6]),
                score=round(s.rank, 4),
            ))
            taken += 1
    return ResonancePacketV1(
        packet_id=packet_id_for(read.seed_id),
        seed_id=read.seed_id,
        root_request_id=read.root_request_id,
        read_title=read.title,
        read_url=read.url,
        read_summary=read.summary,
        read_learned=read.learned,
        read_text_sha256=[sha for sha, _ in read.retained_texts],
        candidates=candidates,
        built_at=built_at,
    )


def with_recent_counts(packets: Sequence[ResonancePacketV1]) -> list[ResonancePacketV1]:
    """Set each candidate's recent_packet_count: how many OTHER packets offered the same ref."""
    counts: dict[tuple[str, str], int] = {}
    for p in packets:
        for key in {(c.ref_kind, c.ref_id) for c in p.candidates}:
            counts[key] = counts.get(key, 0) + 1
    out = []
    for p in packets:
        cands = [c.model_copy(update={"recent_packet_count": counts[(c.ref_kind, c.ref_id)] - 1})
                 for c in p.candidates]
        out.append(p.model_copy(update={"candidates": cands}))
    return out


# A heuristic: abbreviations ("U.S. Senate") over-split and lowercase starts under-split.
# Good enough for a 3-5 sentence sanity gate on a summary, not for anything finer.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    t = " ".join(str(text or "").split())
    return [s for s in _SENTENCE.split(t) if s] if t else []


def fallback_article_summary(learned: str, *, max_sentences: int = SUMMARY_MAX_SENTENCES) -> str:
    """The first sentences of Stage 1's what_i_learned: Orion's own account of the article
    at read time. Used where no judge ran; it is labelled as such by the caller."""
    return " ".join(split_sentences(learned)[:max_sentences])


# ---------------------------------------------------------------------------------------
# Judge prompt. No example sentences anywhere (nothing to echo), no similarity scores
# (the judge must not anchor on a number that does not mean "related").
# ---------------------------------------------------------------------------------------

_JUDGE_INSTRUCTIONS_TEMPLATE = """You are Orion. You just finished reading something from the world.
Below is what you read, then pieces of your own mind that retrieval placed near it: beliefs you hold (priors) with the findings and revisions behind them, questions you live with and your answers, concepts in your atlas with their neighbours, memories of conversations with Juniper, your curiosity findings, dreams, and open threads.
Retrieval is loose. Most of these pieces are only near the read in wording. Treat nearness as no evidence at all.
Decide whether there is a real connection between this read and one or more of these pieces: one you could explain to Juniper in plain words, where the read actually changes, confirms, sharpens, or contradicts something you hold, or answers or reopens one of your questions.
If there is none, say so: decision "none" is the normal, honest outcome.
Rules:
- Cite only pieces listed below, by their exact ref_kind and ref_id.
- read_quote must be copied verbatim from the READ section (summary, what you learned, or retained text).
- article_summary: 3 to 5 sentences on what the article itself says, for Juniper, without your connection; at most {summary_max} characters.
- connection: first person, your own words, what the read has to do with the cited pieces; at most {connection_max} characters.
- bears_on_prior: "strains" only if the read gives a reason to doubt a cited self_prior; then set strained_prior_id to that prior's ref_id. "supports" if it backs one. Otherwise null.
- decision: "share" if Juniper would want to hear this now, "note" if it belongs in your letter only, "none" otherwise.
- explore_next: one question worth chasing because of this read, or null; at most {explore_max} characters.
- confidence: 0 to 1, how sure you are the connection is real.
Return one JSON object with exactly these keys: article_summary, read_quote, connection, citations (list of at most {citations_max} objects with ref_kind, ref_id, how; how at most {how_max} characters), bears_on_prior, strained_prior_id, decision, explore_next, confidence."""


def _max_len(model: Any, name: str) -> int:
    """A field's max_length from the schema itself, so the prompt and the validator cannot
    drift: the first live judge run (2026-10-14) lost a real judgment to a 1,382-character
    connection against a 900-character limit the prompt never stated."""
    for meta in model.model_fields[name].metadata:
        if getattr(meta, "max_length", None) is not None:
            return int(meta.max_length)
    raise KeyError(f"{model.__name__}.{name} has no max_length")


JUDGE_INSTRUCTIONS = _JUDGE_INSTRUCTIONS_TEMPLATE.format(
    summary_max=_max_len(ReadingResonanceJudgmentV1, "article_summary"),
    connection_max=_max_len(ReadingResonanceJudgmentV1, "connection"),
    explore_max=_max_len(ReadingResonanceJudgmentV1, "explore_next"),
    citations_max=_max_len(ReadingResonanceJudgmentV1, "citations"),
    how_max=_max_len(ResonanceCitationV1, "how"),
)


def build_judge_prompt(packet: ResonancePacketV1, retained_texts: Sequence[tuple[str, str]] = ()) -> str:
    lines = [JUDGE_INSTRUCTIONS, "", "== READ =="]
    lines.append(f"title: {packet.read_title or '(untitled)'}")
    lines.append(f"url: {packet.read_url}")
    if packet.read_learned:
        lines.append(f"what you learned: {packet.read_learned}")
    if packet.read_summary:
        lines.append(f"summary: {packet.read_summary}")
    for sha, text in retained_texts:
        lines.append(f"retained text {sha[:12]}: {_clip(text, 3000)}")
    lines.append("")
    lines.append("== PIECES OF YOU ==")
    if not packet.candidates:
        lines.append("(retrieval found nothing near this read)")
    for c in packet.candidates:
        recent = f" (also offered for {c.recent_packet_count} other recent reads)" if c.recent_packet_count else ""
        lines.append(f"- ref_kind={c.ref_kind} ref_id={c.ref_id}{recent}")
        lines.append(f"  {c.excerpt}")
        for n in c.neighborhood:
            lines.append(f"    [{n.relation}] {_clip(n.text, NEIGHBOR_CHARS)}")
    return "\n".join(lines)


def _scaffold_lines(prompt: str) -> list[str]:
    """Every scaffold sentence long enough to be distinctive (an echo of one sentence of a
    long instruction line is still an echo)."""
    head = prompt.split("== READ ==", 1)[0]
    out = []
    for ln in head.splitlines():
        out += [s for s in split_sentences(ln.strip(" -")) if len(s) >= ECHO_MIN_CHARS]
    return out


def _norm_ws(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _parse(raw: str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    start = text.find("{")
    while start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return obj if isinstance(obj, dict) else None
    return None


def _quote_in(quote: str, source: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", source) is not None


def check_judgment(
    packet: ResonancePacketV1,
    raw: str | dict[str, Any],
    *,
    retained_texts: Sequence[tuple[str, str]] = (),
    prompt: str | None = None,
) -> tuple[ReadingResonanceJudgmentV1 | None, ReadingResonanceCheckV1]:
    """Deterministic acceptance. A rejected judgment is recorded with decision 'none'."""

    def reject(reasons: list[str], j: ReadingResonanceJudgmentV1 | None = None):
        return j, ReadingResonanceCheckV1(packet_id=packet.packet_id, accepted=False,
                                          decision="none", reasons=reasons)  # type: ignore[arg-type]

    obj = _parse(raw)
    if obj is None:
        return reject(["unparseable"])
    try:
        j = ReadingResonanceJudgmentV1.model_validate(obj)
    except ValidationError:
        return reject(["unparseable"])
    reasons: list[str] = []
    n = len(split_sentences(j.article_summary))
    if not SUMMARY_MIN_SENTENCES <= n <= SUMMARY_MAX_SENTENCES:
        reasons.append("summary_sentence_count")
    # Checked whatever the decision: a "none" must not smuggle in an uncited prior or a
    # citation the packet never handed over (strained_prior would route it to a re-test).
    handed = {(c.ref_kind, c.ref_id) for c in packet.candidates}
    if any((c.ref_kind, c.ref_id) not in handed for c in j.citations):
        reasons.append("citation_not_in_packet")
    cited_priors = {c.ref_id for c in j.citations if c.ref_kind == "self_prior"}
    if j.bears_on_prior is not None and not cited_priors:
        reasons.append("prior_relation_without_prior")
    if j.bears_on_prior == "strains" and j.strained_prior_id not in cited_priors:
        reasons.append("strained_prior_not_cited")
    if j.decision != "none":
        if not j.citations and "citation_not_in_packet" not in reasons:
            reasons.append("citation_not_in_packet")
        sources = [packet.read_summary, packet.read_learned, *(t for _, t in retained_texts)]
        quote = _norm_ws(j.read_quote)
        if len(quote.split()) < QUOTE_MIN_WORDS or not any(_quote_in(quote, _norm_ws(s)) for s in sources if s):
            reasons.append("quote_not_in_read")
        if not j.connection.strip():
            reasons.append("empty_connection")
    if prompt is not None:
        written = [_norm_ws(j.connection), _norm_ws(j.article_summary), *(_norm_ws(c.how) for c in j.citations)]
        if any(_norm_ws(line) in w for line in _scaffold_lines(prompt) for w in written if w):
            reasons.append("prompt_echo")
    if reasons:
        return reject(reasons, j)
    return j, ReadingResonanceCheckV1(packet_id=packet.packet_id, accepted=True,
                                      decision=j.decision, reasons=[])


def strained_prior(j: ReadingResonanceJudgmentV1 | None, check: ReadingResonanceCheckV1) -> str | None:
    """The prior an accepted judgment says this read strains: the re-test request patch B
    routes into the curiosity prior menu (orion/curiosity/worldview.py select_priors)."""
    if j is None or not check.accepted or check.decision == "none" or j.bears_on_prior != "strains":
        return None
    return j.strained_prior_id
