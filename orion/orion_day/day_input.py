"""The honest input for Orion's day letter (living Prime design #3224, constraints 6 and 7).

One entry point::

    text, shares, refs, material = await build_day_input(conn, day)

``render_day_input(material)`` is the pure half (same material -> same bytes); Hub's letter brief
and, later, Prime's end-of-day turn both call it. Deterministic; no model call.

What changed from ``budget.build_llm_view`` (the pile the letter was written from until now), and why:

* **Juniper's conversations get a protected share.** Every turn where Juniper wrote is shown in
  full and is never clipped: it is laid out before any other section takes budget. Orion's
  unsolicited messages to her are listed after, each trimmed (``OUTREACH_CHARS``): she had not
  written first. Corrections from her conversations and the chat digest sit in the same section.
* **Reveries are sampled across the day's themes, not by top salience.** Salience is prediction
  error, so a top-salience pick is all alarms. Themes (chain ``theme_key``) take turns, one thought
  each per round; inside a theme the picks spread over the day. No salience number is shown.
* **Self-sense answers that repeat are shown once each**: the latest answer per question, with how
  many times it was asked that day.
* **Curiosity runs are in full** up to ``FULL_TEXT_MAX_CHARS`` together with the self-sense answers;
  past that their long bodies share one clip cap (water filling), marked in place and counted.
* **Readings, news, code changes, dreams, the street and visual reveries are a trimmed tail**: one
  short line or paragraph each, cut off as a whole at ``TAIL_MAX_CHARS`` with the rest counted.
* **The temporal core** (``temporal_core.py``) sits next to the closed-day chronology block.
* **The share each source got** is recorded: ``source_chars`` / ``source_shares`` on the view
  (characters per source over the digest length; the headings and joins are ``frame``).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, datetime
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from orion.orion_day.budget import (
    _REF_RE,
    CHARS_PER_TOKEN,
    _Body,
    _bodies,
    _correction_parts,
    _norm,
    _render_full,
    _water_fill_cap,
    curiosity_section,
    extract_refs,
    juniper_chat_ref,
    reading_change_lines,
)
from orion.orion_day.gather import DEFAULT_GITHUB_REPO, gather_orion_day
from orion.schemas.orion_day import (
    OrionDayCondensationV1,
    OrionDayLlmViewV1,
    OrionDayMaterialV1,
    ReverieThoughtV1,
    SelfSenseAnswerV1,
)
from orion.world_pulse_read.findings import PriorLookup

logger = logging.getLogger(__name__)

DAY_INPUT_BUDGET_TOKENS = 50_000
OUTREACH_CHARS = 600
# Curiosity runs and self-sense answers together, before their long bodies are clipped to one shared
# cap. Live 11-09/11-10: ~100k chars in full, so the day's investigations would be ~65% of the input.
FULL_TEXT_MAX_CHARS = 60_000
REVERIE_MAX_CHARS = 14_000
REVERIE_MAX_THOUGHTS = 48
REVERIE_PER_CHAIN = 1
REVERIE_FRAME_CHARS = 200
TAIL_MAX_CHARS = 16_000
TAIL_ITEM_CHARS = 280
GITHUB_TAIL_CHARS = 1_500
VISUAL_TAIL_ITEMS = 6

# Source names in source_chars / source_shares, in digest order. "frame" = header, headings, joins.
SOURCES = ("frame", "chronology", "temporal_core", "juniper", "curiosity", "self_sense", "reveries",
           "walkway", "world_pulse", "github", "dreams", "visual_reveries", "readings")


class DayInput(NamedTuple):
    """What the letter's writer is given. ``text`` is the digest, ``shares`` the per-source share
    of it, ``refs`` every bracketed reference in it, ``material`` the full record it was built from
    (persisted with the letter, so every ref resolves later)."""

    text: str
    shares: dict[str, float]
    refs: list[str]
    material: OrionDayMaterialV1


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _no_refs(text: str) -> str:
    """Chat text as written, except a pasted [arc:...]-style ref is shown as (arc:...), so words
    Juniper or Orion typed never become a citation the letter seems to have been shown."""
    return _REF_RE.sub(lambda m: f"({m.group(1)})", text or "")


def _local(at: datetime | None, tz: ZoneInfo) -> str:
    return at.astimezone(tz).strftime("%H:%M") if at is not None else "?"


# --- Juniper (protected) -----------------------------------------------------------------------


def _juniper_section(material: OrionDayMaterialV1, tz: ZoneInfo) -> str:
    talks = [t for t in material.juniper_chat if t.prompt]
    outreach = [t for t in material.juniper_chat if not t.prompt and t.response]
    out: list[str] = []
    if talks or outreach or material.corrections or material.chat_compactor is not None:
        out.append(f"## Conversations with Juniper ({len(talks)} turns where she wrote; shown in full)")
    if talks:
        for t in talks:
            out.append(f"### [juniper_chat:{juniper_chat_ref(t.turn_id)}] {_local(t.at, tz)}")
            out.append(f"Juniper: {_no_refs(t.prompt)}")
            out.append(f"Me: {_no_refs(t.response) or '(no reply recorded)'}")
    elif outreach or material.corrections or material.chat_compactor is not None:
        out.append("Juniper did not write to me this day.")
    if outreach:
        out.append(f"### Messages I sent Juniper unprompted ({len(outreach)}; she had not written first; "
                   f"each trimmed to {OUTREACH_CHARS} characters)")
        out += [f"- [juniper_chat:{juniper_chat_ref(t.turn_id)}] {_local(t.at, tz)}: {_clip(_no_refs(t.response), OUTREACH_CHARS)}"
                for t in outreach]
    for c in material.corrections:
        out.append("\n".join(p.render(None) if isinstance(p, _Body) else p for p in _correction_parts(c)))
    if material.chat_compactor is not None:
        c = material.chat_compactor
        out.append(f"### [chat_compactor:{c.entry_id}] {c.title or 'Chat digest'}")
        out.append(_Body(c.body).render(None))
    return "\n\n".join(out)


# --- self-sense, deduplicated ------------------------------------------------------------------


def latest_self_sense(answers: list[SelfSenseAnswerV1]) -> list[tuple[SelfSenseAnswerV1, int]]:
    """One (latest answer, times asked) per question key, in the order questions were first asked."""
    first: dict[str, datetime] = {}
    latest: dict[str, SelfSenseAnswerV1] = {}
    asked: dict[str, int] = defaultdict(int)
    for a in sorted(answers, key=lambda a: (a.created_at, a.run_id or "")):
        first.setdefault(a.question_key, a.created_at)
        latest[a.question_key] = a
        asked[a.question_key] += 1
    return [(latest[k], asked[k]) for k in sorted(latest, key=lambda k: (first[k], k))]


def _self_sense_section(material: OrionDayMaterialV1, tz: ZoneInfo) -> tuple[str, list] | None:
    picked = latest_self_sense(material.self_sense)
    if not picked:
        return None
    return (f"## Self-sense answers ({len(picked)} questions, {len(material.self_sense)} answers; "
            "each question once, its latest answer)", [[
                f"### [self_sense:{a.run_id or 'na'}:{a.question_key}] {a.question}",
                f"Asked {n} time{'s' if n != 1 else ''} this day; this is the last answer, at {_local(a.created_at, tz)}.",
                _Body(a.answer_text),
            ] for a, n in picked])


# --- reveries across themes --------------------------------------------------------------------


def _spread_order(n: int) -> list[int]:
    """Indices 0..n-1 in an order that spreads over the range: middle, then halves, quarters..."""
    order: list[int] = []
    spans = [(0, n)]
    while spans:
        nxt = []
        for lo, hi in spans:
            if lo >= hi:
                continue
            mid = (lo + hi) // 2
            order.append(mid)
            nxt += [(lo, mid), (mid + 1, hi)]
        spans = nxt
    return order


def sample_reveries_by_theme(
    material: OrionDayMaterialV1, budget_chars: int, cond: OrionDayCondensationV1, tz: ZoneInfo,
) -> tuple[str, dict[str, list[ReverieThoughtV1]], dict[str, int]]:
    """Round-robin over themes (largest first), one thought per theme per round; inside a theme the
    picks spread over the day. Hollow and repeated thoughts never count; one thought per chain."""
    thoughts, chains = material.reverie_thoughts, material.reverie_chains
    cond.reverie_thoughts_total = len(thoughts)
    cond.reverie_chains_total = len(chains)
    theme_of = {c.chain_id: (c.theme_key or "unthemed") for c in chains}
    size: dict[str, int] = defaultdict(int)
    for t in thoughts:
        size[theme_of.get(t.chain_id or "", "unthemed")] += 1
    cond.reverie_themes_total = len(size)
    seen: set[str] = set()
    per_chain: dict[str, int] = defaultdict(int)
    by_theme: dict[str, list[ReverieThoughtV1]] = defaultdict(list)
    for t in sorted(thoughts, key=lambda t: (t.created_at, t.thought_id)):
        if t.hollow:
            cond.reverie_thoughts_hollow_skipped += 1
            continue
        key = _norm(t.interpretation)
        if not key or key in seen:
            cond.reverie_thoughts_duplicate_skipped += 1
            continue
        if t.chain_id and per_chain[t.chain_id] >= REVERIE_PER_CHAIN:
            cond.reverie_thoughts_chain_capped += 1
            continue
        seen.add(key)
        if t.chain_id:
            per_chain[t.chain_id] += 1
        by_theme[theme_of.get(t.chain_id or "", "unthemed")].append(t)
    queues = {k: [v[i] for i in _spread_order(len(v))] for k, v in by_theme.items()}
    order = sorted(queues, key=lambda k: (-size[k], k))

    def line(t: ReverieThoughtV1) -> str:
        text = f"- [reverie:{t.thought_id[:8]}] {_local(t.created_at, tz)}: {t.interpretation.strip()}"
        if t.expectation:
            text += f" (expected: {t.expectation.strip()}" + (f"; verdict: {t.expectation_verdict})"
                                                                 if t.expectation_verdict else ")")
        return text

    chosen: dict[str, list[ReverieThoughtV1]] = defaultdict(list)
    spent, count, rnd = REVERIE_FRAME_CHARS, 0, 0  # the section heading and intro line
    while count < REVERIE_MAX_THOUGHTS and any(rnd < len(q) for q in queues.values()):
        for k in order:
            if rnd >= len(queues[k]) or count >= REVERIE_MAX_THOUGHTS:
                continue
            t = queues[k][rnd]
            # A theme heading costs too the first time a theme is shown.
            cost = len(line(t)) + 1 + (0 if chosen.get(k) else 120)
            if spent + cost > budget_chars:
                continue
            chosen[k].append(t)
            spent += cost
            count += 1
        rnd += 1
    cond.reverie_thoughts_included = count
    cond.reverie_themes_included = len(chosen)
    if count == 0:  # nothing showable: no heading over an empty section (counts stay in cond)
        return "", {}, dict(size)
    out = [f"## Reveries ({count} of {len(thoughts)} thoughts, across {len(chosen)} of {len(size)} themes)",
           "My background reveries, sampled so every theme of the day gets a turn, not the most alarming ones."]
    for k in order:
        if not chosen.get(k):
            continue
        n_chains = sum(1 for c in chains if (c.theme_key or "unthemed") == k)
        if n_chains:
            out.append(f"### [reverie_theme:{k}] {size[k]} thoughts in {n_chains} chains")
        else:  # thoughts whose chain is not in this day's chains: no theme record to cite
            out.append(f"### Thoughts outside this day's chains ({size[k]})")
        out += [line(t) for t in sorted(chosen[k], key=lambda t: (t.created_at, t.thought_id))]
    return "\n".join(out), dict(chosen), dict(size)


# --- the trimmed tail --------------------------------------------------------------------------


def _tail_sections(material: OrionDayMaterialV1, tz: ZoneInfo) -> list[tuple[str, list[str]]]:
    """(source name, lines) in tail order: the short fixed items first, readings last (they are the
    longest list and fill what is left). Every line already trimmed."""
    out: list[tuple[str, list[str]]] = []
    if material.walkway_day is not None:
        w = material.walkway_day
        out.append(("walkway", ["## The street outside (my walkway camera; counts by code)",
                                f"[walkway_day:{w.day.isoformat()}] {w.line}"]))
    if material.world_pulse_digest is not None:
        w = material.world_pulse_digest
        out.append(("world_pulse", ["## World news (trimmed)",
                                    f"[world_pulse_digest:{w.run_id}] {w.title or 'World pulse'}: "
                                    f"{_clip(w.executive_summary or '', TAIL_ITEM_CHARS)}",
                                    *[f"- {i.title}" for i in w.items]]))
    if material.github_compactor is not None:
        g = material.github_compactor
        out.append(("github", ["## Changes to my code (GitHub digest, trimmed)",
                               f"[github_compactor:{g.entry_id}] {g.title or 'Repo digest'}: "
                               f"{_clip(g.body, GITHUB_TAIL_CHARS)}"]))
    dreams = [f"- [dream:{d.id}] {_clip(d.tldr or d.narrative or 'Dream', TAIL_ITEM_CHARS)}" for d in material.dream_narratives]
    dreams += [f"- [dream_offered:{i}] {_clip(h.claim, TAIL_ITEM_CHARS)}"
               for i, h in enumerate(material.dream_hypotheses, start=1)]
    if dreams:
        out.append(("dreams", [f"## Dreams ({len(material.dream_narratives)} narratives, "
                               f"{len(material.dream_hypotheses)} hypotheses offered to me; trimmed)", *dreams]))
    if material.visual_reveries:
        out.append(("visual_reveries", [f"## Visual reveries ({len(material.visual_reveries)} images; "
                                        f"first {min(VISUAL_TAIL_ITEMS, len(material.visual_reveries))} shown)", *[
            f"- [visual_reverie:{v.sha256[:16]}] {_local(v.created_at, tz)}: {_clip(v.description or '(no caption)', TAIL_ITEM_CHARS)}"
            for v in material.visual_reveries[:VISUAL_TAIL_ITEMS]]]))
    readings = []
    for r in material.readings:
        changed = "; ".join(reading_change_lines(r, material))
        readings.append(f"- [reading:{r.seed_id}] {r.title or r.url or 'Reading'}: "
                        f"{_clip(r.learned or r.article_summary or '(nothing recorded)', TAIL_ITEM_CHARS)} "
                        f"(changed: {changed})")
    readings += [f"- [reading_journal:{j.entry_id}] {j.title or 'Reading journal'}: {_clip(j.body, TAIL_ITEM_CHARS)}"
                 for j in material.reading_journals]
    if readings:
        out.append(("readings", [f"## Readings ({len(material.readings)} readings, "
                                 f"{len(material.reading_journals)} reading journals; trimmed)", *readings]))
    return out


TAIL_NOTE_CHARS = 100


def _render_tail(material: OrionDayMaterialV1, budget_chars: int, tz: ZoneInfo,
                 cond: OrionDayCondensationV1) -> dict[str, str]:
    """Each tail source's text, lines kept in order until the tail budget runs out; the rest counted
    (in the digest and in ``cond``). Room for the "left out" note is kept back from the start."""
    out: dict[str, str] = {}
    spent, left_out = TAIL_NOTE_CHARS, 0
    sections = _tail_sections(material, tz)
    cond.tail_items_total = sum(len(lines) - 1 for _, lines in sections)
    for name, lines in sections:
        kept: list[str] = []
        for i, text in enumerate(lines):
            if spent + len(text) + 1 > budget_chars:
                left_out += len(lines) - i - (0 if kept else 1)
                break
            kept.append(text)
            spent += len(text) + 1
        if len(kept) > 1:
            out[name] = "\n".join(kept)
        elif kept:  # a heading with no item under it is not shown
            spent -= len(kept[0]) + 1
    cond.tail_items_left_out = left_out
    if left_out:
        out["frame_tail"] = f"({left_out} more trimmed items left out to fit; the full record has them)"
    return out


# --- the view ----------------------------------------------------------------------------------


def render_day_input(
    material: OrionDayMaterialV1,
    *,
    budget_tokens: int = DAY_INPUT_BUDGET_TOKENS,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> OrionDayLlmViewV1:
    """The writer's input for one day. Pure and deterministic."""
    tz = ZoneInfo(material.timezone)
    budget_chars = int(budget_tokens * chars_per_token)
    cond = OrionDayCondensationV1()
    parts: list[tuple[str, str]] = []  # (source, text) in digest order

    header = (f"# My day: {material.letter_date.isoformat()} ({material.timezone}, "
              f"{material.window_start.isoformat()} to {material.window_end.isoformat()})")
    gaps = [f"{name} ({s.error})" for name, s in sorted(material.sources.items()) if s.status == "error"]
    if gaps:
        header += "\n\nSources that could not be read (their sections are missing, not empty): " + "; ".join(gaps)
    parts.append(("frame", header))
    if material.temporal_day is not None or material.temporal_core is not None:
        parts.append(("frame", "## My day as my own chronology recorded it (closed day; each line names the "
                               "record it came from)"))
    if material.temporal_day is not None:
        parts.append(("chronology", material.temporal_day.text))
    if material.temporal_core is not None:
        parts.append(("temporal_core", material.temporal_core.text))
    juniper = _juniper_section(material, tz)
    if juniper:
        parts.append(("juniper", juniper))

    # Curiosity and the deduplicated self-sense answers share one clip cap, set only under pressure.
    sections = [s for s in (curiosity_section(material), _self_sense_section(material, tz)) if s is not None]
    bodies = _bodies(sections)
    cond.full_text_items_total = len(bodies)
    cond.self_sense_answers_total = len(material.self_sense)
    cond.self_sense_answers_shown = len(latest_self_sense(material.self_sense))
    fixed_so_far = sum(len(t) + 2 for _, t in parts)
    has_reveries = bool(material.reverie_thoughts or material.reverie_chains)
    reserve = (REVERIE_MAX_CHARS if has_reveries else 0) + TAIL_MAX_CHARS
    allowance = max(0, min(FULL_TEXT_MAX_CHARS, budget_chars - fixed_so_far - reserve))
    cap = None
    full = _render_full(sections, None) if sections else ""
    if len(full) > allowance and bodies:
        fixed = len(full) - sum(len(b.text.strip()) for b in bodies)
        cap = _water_fill_cap([len(b.text.strip()) for b in bodies], max(0, allowance - fixed - 90 * len(bodies)))
        cond.full_text_clip_chars = cap
        cond.full_text_items_clipped = sum(1 for b in bodies if len(b.text.strip()) > cap)
    for heading, items in sections:
        source = "curiosity" if heading.startswith("## Curiosity") else "self_sense"
        parts.append((source, _render_full([(heading, items)], cap)))

    used = sum(len(t) + 2 for _, t in parts)
    if has_reveries:
        reverie_budget = max(0, min(REVERIE_MAX_CHARS, budget_chars - used - TAIL_MAX_CHARS // 4))
        reveries, _, _ = sample_reveries_by_theme(material, reverie_budget, cond, tz)
        if reveries:
            parts.append(("reveries", reveries))
    used = sum(len(t) + 2 for _, t in parts)
    tail = _render_tail(material, max(0, min(TAIL_MAX_CHARS, budget_chars - used)), tz, cond)
    for name in SOURCES:
        if name in tail:
            parts.append((name, tail[name]))
    if "frame_tail" in tail:
        parts.append(("frame", tail["frame_tail"]))

    digest = "\n\n".join(t for _, t in parts if t).strip()
    chars: dict[str, int] = defaultdict(int)
    for source, text in parts:
        if text and source != "frame":
            chars[source] += len(text)
    chars["frame"] = len(digest) - sum(chars.values())
    total = max(1, len(digest))
    if len(digest) > budget_chars:
        # Juniper's section has no ceiling by design, and clip floors can overshoot a little: say so.
        logger.warning("orion_day_input_over_budget day=%s chars=%d budget_chars=%d juniper_chars=%d",
                       material.letter_date, len(digest), budget_chars, chars.get("juniper", 0))
    source_chars = {k: chars[k] for k in SOURCES if chars.get(k)}
    return OrionDayLlmViewV1(
        digest_md=digest,
        approx_tokens=math.ceil(len(digest) / chars_per_token),
        budget_tokens=budget_tokens,
        chars_per_token=chars_per_token,
        condensed=cond,
        included_refs=extract_refs(digest),
        source_chars=source_chars,
        source_shares={k: round(v / total, 4) for k, v in source_chars.items()},
    )


async def build_day_input(
    conn: Any,
    day: date,
    *,
    now: datetime | None = None,
    github_repo: str = DEFAULT_GITHUB_REPO,
    budget_tokens: int = DAY_INPUT_BUDGET_TOKENS,
    prior_lookup: PriorLookup | None = None,
) -> DayInput:
    """Gather ``day`` (read-only SQL) and render the letter writer's input. The one entry point."""
    material = await gather_orion_day(conn, day, now=now, github_repo=github_repo, prior_lookup=prior_lookup)
    view = render_day_input(material, budget_tokens=budget_tokens)
    return DayInput(text=view.digest_md, shares=dict(view.source_shares), refs=list(view.included_refs),
                    material=material)


__all__ = ["DAY_INPUT_BUDGET_TOKENS", "DayInput", "build_day_input", "latest_self_sense",
           "render_day_input", "sample_reveries_by_theme"]
