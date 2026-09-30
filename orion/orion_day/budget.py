"""Build the model's view of the day within a token budget. Deterministic.

Full text, in this order: curiosity runs (with their outcomes), self-sense
answers, readings and reading journals, dreams (narratives and offered
hypotheses), the chat and GitHub compactor digests, the world-pulse digest, and
visual reverie captions. Reveries (text thoughts, ~450k chars a day) are the one
section that is always condensed: chain themes plus the most salient distinct,
non-hollow thoughts that fit what is left.

If the full-text sections alone overrun their share, each long body is clipped to
one shared cap (the largest cap that fits -- "water filling"), and the clip is
marked in place and counted. The material itself is never touched: the email
shows everything.

Every item is rendered with a bracketed reference (``[curiosity:<run_id>]``,
``[reading:<seed_id>]`` ...). ``included_refs`` lists them, and the grounding eval
checks every one resolves to a material item.

Token counts are an estimate (``chars / CHARS_PER_TOKEN``), conservative for Qwen
tokenizers on English prose; the agent lane has a 131k-token context, and the
default 70k budget leaves room for the prompt frame, the note (carry-forward
input) and the completion.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from orion.schemas.orion_day import (
    OrionDayCondensationV1,
    OrionDayLlmViewV1,
    OrionDayMaterialV1,
    ReverieThoughtV1,
)

DEFAULT_BUDGET_TOKENS = 70_000
CHARS_PER_TOKEN = 3.5
# Reveries get at most this many tokens, and never more than what the full-text sections leave.
REVERIE_MAX_TOKENS = 12_000
# The full-text sections may use the budget minus this reserve before clipping starts,
# so a heavy day still shows some reverie texture.
REVERIE_RESERVE_TOKENS = 4_000
REVERIE_MAX_THOUGHTS = 80
REVERIE_THEME_SHARE = 0.25
# Reverie thoughts within one chain, and across chains on one theme, often open with the same
# sentence (live 2026-09-29: "The coalition is fixated on the unresolved prediction error from
# the intake pipeline, which persists despite full mapping." x3 at 06:45). One per chain, and a
# normalized-prefix dedupe across chains.
DEDUPE_PREFIX_CHARS = 100
REVERIE_PER_CHAIN = 1
REVERIE_FRAME_CHARS = 400
MIN_CLIP_CHARS = 400

_REF_RE = re.compile(r"\[((?:curiosity|curiosity_failed|self_sense|reading|reading_journal|dream|dream_offered|"
                     r"chat_compactor|github_compactor|world_pulse_digest|visual_reverie|reverie|reverie_theme)"
                     r":[^\]\s]+)\]")


def extract_refs(text: str) -> list[str]:
    """Every bracketed item reference in ``text``, in order, de-duplicated."""
    seen: dict[str, None] = {}
    for match in _REF_RE.finditer(text or ""):
        seen.setdefault(match.group(1), None)
    return list(seen)


_HEADING_RE = re.compile(r"^(#{1,6})(\s)", flags=re.MULTILINE)


@dataclass
class _Body:
    text: str

    def render(self, cap: int | None) -> str:
        body = (self.text or "").strip()
        if cap is not None and len(body) > cap:
            body = body[:cap].rstrip() + f"\n… [clipped here: {len(body) - cap} more characters in the full record]"
        # A body's own markdown headings sit below the digest's item headings (###).
        return _HEADING_RE.sub(lambda m: "####" + m.group(2), body)


def _fmt_time(value) -> str:
    return value.strftime("%H:%M UTC") if value is not None else "?"


def _yn(value) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _full_text_blocks(material: OrionDayMaterialV1) -> list[tuple[str, list]]:
    """(section heading, [str | _Body, ...]) in render order. _Body parts are clippable."""
    sections: list[tuple[str, list]] = []

    runs = []
    for run in material.curiosity_runs:
        head = f"### [curiosity:{run.run_id}] {run.journal_title or 'Curiosity run'} (line: {run.line or 'unknown'}"
        head += f", family: {run.self_question_family})" if run.self_question_family else ")"
        parts: list = [head, f"Finished {_fmt_time(run.completed_at)}. Line continues: {_yn(run.continue_line)}. "
                             f"Reached out: {_yn(run.reach_out)}" + (f" -- {run.reach_out_why}" if run.reach_out_why else "") + "."]
        if run.outcome:
            o = run.outcome
            priors = []
            for p in o.get("per_prior") or []:
                if isinstance(p, dict):
                    priors.append(f"{p.get('prior_id')} {p.get('kind')} {p.get('before')}->{p.get('after')}")
            parts.append(f"Outcome: turn ok {_yn(o.get('turn_ok'))}; tested {o.get('n_tested')}, moved "
                         f"{o.get('n_moved')}, formed {o.get('n_formed')}" + (f"; priors: {'; '.join(priors)}" if priors else "")
                         + (f"; unknown reason: {o.get('unknown_reason')}" if o.get("unknown_reason") else ""))
        if run.self_definition_text:
            parts += ["Self-definition after this run:", _Body(run.self_definition_text)]
        if run.lived_answer_text:
            parts += ["Lived answer after this run:", _Body(run.lived_answer_text)]
        body = run.journal_body or run.finding_text or ""
        if body:
            parts += ["Write-up:", _Body(body)]
        runs.append(parts)
    failed = [[f"- [curiosity_failed:{f.run_id}] {f.workflow} failed {_fmt_time(f.failed_at)}: {f.error or 'no error recorded'}"]
              for f in material.curiosity_failed]
    if runs or failed:
        sections.append((f"## Curiosity ({len(material.curiosity_runs)} runs finished, "
                         f"{len(material.curiosity_failed)} failed)", runs + failed))

    if material.self_sense:
        sections.append(("## Self-sense answers", [[
            f"### [self_sense:{a.run_id or 'na'}:{a.question_key}] {a.question}",
            f"Scores: self-label {a.self_label_score}, grounded-record {a.grounded_record_score}.",
            _Body(a.answer_text),
        ] for a in material.self_sense]))

    readings = [[
        f"### [reading:{r.seed_id}] {r.title or r.url or 'Reading'}",
        f"Source: {r.url or 'unknown'}. Status: {r.reading_status or 'unknown'}."
        + (f" Why I picked it: {r.why_now}" if r.why_now else ""),
        "What I learned:" if r.learned else "What I learned: (nothing recorded)",
        *([_Body(r.learned)] if r.learned else []),
    ] for r in material.readings]
    readings += [[f"### [reading_journal:{j.entry_id}] {j.title or 'Reading journal'}", _Body(j.body)]
                 for j in material.reading_journals]
    if readings:
        sections.append((f"## Readings ({len(material.readings)} readings, "
                         f"{len(material.reading_journals)} reading journals)", readings))

    dreams = [[f"### [dream:{d.id}] {d.tldr or 'Dream'}", *([_Body(d.narrative)] if d.narrative else [])]
              for d in material.dream_narratives]
    # No hypothesis id in the model's view: `dream_hypothesis:<id>` is the dream scorecard's
    # formed_from join key (orion/dream/hypotheses.py), and a prior formed from the letter must
    # never be credited to the blind offer. The material (email) keeps the ids.
    dreams += [[f"### [dream_offered:{i}] {h.claim}", *([_Body(h.why)] if h.why else [])]
               for i, h in enumerate(material.dream_hypotheses, start=1)]
    if dreams:
        sections.append((f"## Dreams ({len(material.dream_narratives)} narratives, "
                         f"{len(material.dream_hypotheses)} hypotheses offered to me)", dreams))

    if material.chat_compactor is not None:
        c = material.chat_compactor
        sections.append(("## Conversations with Juniper (chat digest)",
                         [[f"### [chat_compactor:{c.entry_id}] {c.title or 'Chat digest'}", _Body(c.body)]]))
    if material.github_compactor is not None:
        g = material.github_compactor
        sections.append(("## Changes to my code (GitHub digest)",
                         [[f"### [github_compactor:{g.entry_id}] {g.title or 'Repo digest'}", _Body(g.body)]]))
    if material.world_pulse_digest is not None:
        w = material.world_pulse_digest
        lines = [f"[world_pulse_digest:{w.run_id}] {w.title or 'World pulse'} -- {w.executive_summary or ''}".strip()]
        lines += [f"- ({i.category or 'general'}) {i.title}" + (f": {i.summary}" if i.summary and i.summary != i.title else "")
                  for i in w.items]
        sections.append(("## World pulse digest", [lines]))
    if material.visual_reveries:
        sections.append((f"## Visual reveries ({len(material.visual_reveries)} images)", [[
            f"- [visual_reverie:{v.sha256[:16]}] {_fmt_time(v.created_at)}"
            + (f" theme {v.theme_key}" if v.theme_key else "") + f": {v.description or '(no caption)'}"
        ] for v in material.visual_reveries]))
    return sections


def _render_full(sections: list[tuple[str, list]], cap: int | None) -> str:
    out: list[str] = []
    for heading, items in sections:
        out.append(heading)
        for parts in items:
            out.append("\n".join(p.render(cap) if isinstance(p, _Body) else p for p in parts))
        out.append("")
    return "\n\n".join(out).strip()


def _bodies(sections) -> list[_Body]:
    return [p for _, items in sections for parts in items for p in parts if isinstance(p, _Body)]


def _water_fill_cap(lengths: list[int], allowance: int) -> int:
    """Largest integer cap C with sum(min(len, C)) <= allowance (never below MIN_CLIP_CHARS)."""
    lo, hi = 0, max(lengths, default=0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= allowance:
            lo = mid
        else:
            hi = mid - 1
    return max(lo, MIN_CLIP_CHARS)


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())[:DEDUPE_PREFIX_CHARS]


def _render_reveries(material: OrionDayMaterialV1, budget_chars: int, cond: OrionDayCondensationV1) -> str:
    thoughts, chains = material.reverie_thoughts, material.reverie_chains
    cond.reverie_thoughts_total = len(thoughts)
    cond.reverie_chains_total = len(chains)
    if not thoughts and not chains:
        return ""
    themes: dict[str, dict] = defaultdict(lambda: {"chains": 0, "thoughts": 0, "peak": 0.0, "endings": defaultdict(int)})
    for c in chains:
        t = themes[c.theme_key or "unthemed"]
        t["chains"] += 1
        t["thoughts"] += c.thought_count
        t["peak"] = max(t["peak"], float(c.ema_salience or 0.0))
        t["endings"][c.terminal_reason or "unknown"] += 1
    cond.reverie_themes_total = len(themes)
    ordered = sorted(themes.items(), key=lambda kv: (-kv[1]["thoughts"], -kv[1]["chains"], kv[0]))
    theme_budget = int(budget_chars * REVERIE_THEME_SHARE)
    theme_lines: list[str] = []
    used = 0
    for key, t in ordered:
        endings = ", ".join(f"{k} x{v}" for k, v in sorted(t["endings"].items()))
        line = (f"- [reverie_theme:{key}] {t['chains']} chains, {t['thoughts']} thoughts, "
                f"peak salience {t['peak']:.2f}; endings: {endings}")
        if used + len(line) + 1 > theme_budget and theme_lines:
            break
        theme_lines.append(line)
        used += len(line) + 1
    cond.reverie_themes_included = len(theme_lines)

    seen: set[str] = set()
    per_chain: dict[str, int] = defaultdict(int)
    candidates: list[ReverieThoughtV1] = []
    for t in sorted(thoughts, key=lambda t: (-(t.salience or 0.0), t.created_at, t.thought_id)):
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
        candidates.append(t)
    thought_budget = budget_chars - used
    chosen: list[tuple[ReverieThoughtV1, str]] = []
    spent = 0
    for t in candidates:
        if len(chosen) >= REVERIE_MAX_THOUGHTS:
            break
        line = f"- [reverie:{t.thought_id[:8]}] {_fmt_time(t.created_at)}, salience {(t.salience or 0.0):.2f}: {t.interpretation.strip()}"
        if t.expectation:
            line += f" (expected: {t.expectation.strip()}" + (f"; verdict: {t.expectation_verdict})" if t.expectation_verdict else ")")
        if spent + len(line) + 1 > thought_budget:
            continue  # a shorter, less salient thought may still fit; order stays deterministic
        chosen.append((t, line))
        spent += len(line) + 1
    cond.reverie_thoughts_included = len(chosen)
    chosen.sort(key=lambda pair: (pair[0].created_at, pair[0].thought_id))
    out = [f"## Reveries (condensed: {len(chosen)} of {len(thoughts)} thoughts shown, "
           f"{len(theme_lines)} of {len(themes)} chain themes)",
           "These are my background reveries, condensed to the most salient distinct thoughts.",
           "### Chain themes", *(theme_lines or ["- (no chains)"]),
           "### Most salient thoughts, in time order", *([line for _, line in chosen] or ["- (none fit)"])]
    return "\n".join(out)


def build_llm_view(
    material: OrionDayMaterialV1,
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> OrionDayLlmViewV1:
    budget_chars = int(budget_tokens * chars_per_token)
    cond = OrionDayCondensationV1()
    header = (f"# My day: {material.letter_date.isoformat()} ({material.timezone}, "
              f"{material.window_start.isoformat()} to {material.window_end.isoformat()})")
    gaps = [f"{name} ({s.error})" for name, s in sorted(material.sources.items()) if s.status == "error"]
    if gaps:
        header += "\n\nSources that could not be read (their sections are missing, not empty): " + "; ".join(gaps)

    sections = _full_text_blocks(material)
    bodies = _bodies(sections)
    cond.full_text_items_total = len(bodies)
    full = _render_full(sections, None)
    reserve = min(int(REVERIE_RESERVE_TOKENS * chars_per_token), budget_chars // 4) \
        if (material.reverie_thoughts or material.reverie_chains) else 0
    allowance = budget_chars - len(header) - reserve
    if len(full) > allowance and bodies:
        fixed = len(full) - sum(len(b.text.strip()) for b in bodies)
        # Room for each clipped body's marker line.
        cap = _water_fill_cap([len(b.text.strip()) for b in bodies], max(0, allowance - fixed - 90 * len(bodies)))
        full = _render_full(sections, cap)
        cond.full_text_clip_chars = cap
        cond.full_text_items_clipped = sum(1 for b in bodies if len(b.text.strip()) > cap)

    # The reverie section's own headings and the joins between sections come out of the same budget.
    remaining = budget_chars - len(header) - len(full) - REVERIE_FRAME_CHARS
    reverie_chars = max(0, min(remaining, int(REVERIE_MAX_TOKENS * chars_per_token)))
    reveries = _render_reveries(material, reverie_chars, cond)
    digest = "\n\n".join(part for part in (header, full, reveries) if part).strip()
    return OrionDayLlmViewV1(
        digest_md=digest,
        approx_tokens=math.ceil(len(digest) / chars_per_token),
        budget_tokens=budget_tokens,
        chars_per_token=chars_per_token,
        condensed=cond,
        included_refs=extract_refs(digest),
    )


def material_refs(material: OrionDayMaterialV1) -> set[str]:
    """Every reference a digest may legitimately cite for this material (grounding checks)."""
    refs = {f"curiosity:{r.run_id}" for r in material.curiosity_runs}
    refs |= {f"curiosity_failed:{f.run_id}" for f in material.curiosity_failed}
    refs |= {f"self_sense:{a.run_id or 'na'}:{a.question_key}" for a in material.self_sense}
    refs |= {f"reading:{r.seed_id}" for r in material.readings}
    refs |= {f"reading_journal:{j.entry_id}" for j in material.reading_journals}
    refs |= {f"dream:{d.id}" for d in material.dream_narratives}
    refs |= {f"dream_offered:{i}" for i in range(1, len(material.dream_hypotheses) + 1)}
    if material.chat_compactor is not None:
        refs.add(f"chat_compactor:{material.chat_compactor.entry_id}")
    if material.github_compactor is not None:
        refs.add(f"github_compactor:{material.github_compactor.entry_id}")
    if material.world_pulse_digest is not None:
        refs.add(f"world_pulse_digest:{material.world_pulse_digest.run_id}")
    refs |= {f"visual_reverie:{v.sha256[:16]}" for v in material.visual_reveries}
    refs |= {f"reverie:{t.thought_id[:8]}" for t in material.reverie_thoughts}
    refs |= {f"reverie_theme:{c.theme_key or 'unthemed'}" for c in material.reverie_chains}
    return refs
