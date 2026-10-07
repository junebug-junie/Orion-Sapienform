#!/usr/bin/env python3
"""Orion's Day eval: on a fixture day (or a live day, read-only), check what the model is
given before any model is called.

Checks, each PASS/FAIL:

1. separation   -- the note prompt's instructions contain no carry-forward vocabulary and no
                   note placeholder; the carry-forward prompt contains the note and the digest.
2. grounding    -- every bracketed item reference in the digest resolves to a material item,
                   and every material item that must be shown in full (curiosity runs, self-sense
                   answers, readings, reading journals, dreams, compactors) is referenced.
3. full_text    -- unless the view reports clipping, each full-text body appears verbatim.
4. condensation -- reveries are condensed (shown <= total, hollow never shown, one per chain).
5. blind        -- no dream-hypothesis id (the scorecard's formed_from join key) in the view.
6. budget       -- approx_tokens <= budget_tokens; the note prompt fits the agent lane's 131k
                   context with the note's completion budget on top.
7. determinism  -- two builds of the same material are byte-identical.

Usage:
    python orion/orion_day/evals/run_orion_day_eval.py                 # fixture day
    python orion/orion_day/evals/run_orion_day_eval.py --live [--date YYYY-MM-DD] [--dsn ...]
Exit 0 when every check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from jinja2 import Environment  # noqa: E402

from orion.orion_day.budget import _Body, build_llm_view, extract_refs, material_refs  # noqa: E402
from orion.orion_day.gather import gather_orion_day  # noqa: E402
from orion.schemas.orion_day import OrionDayMaterialV1  # noqa: E402

PROMPTS = ROOT / "orion" / "cognition" / "prompts"
AGENT_CONTEXT_TOKENS = 131_072
NOTE_MAX_TOKENS = 12_000
CARRY_FORWARD_WORDS = ("carry", "forward", "future", "tomorrow", "next time", "follow up", "follow-up",
                       "thread", "to pursue", "open question", "revisit")


def _render(name: str, od: dict) -> str:
    return Environment(autoescape=False).from_string((PROMPTS / name).read_text()).render(
        metadata={"orion_day_input": od})


def required_refs(m: OrionDayMaterialV1) -> set[str]:
    refs = {f"curiosity:{r.run_id}" for r in m.curiosity_runs}
    refs |= {f"self_sense:{a.run_id or 'na'}:{a.question_key}" for a in m.self_sense}
    refs |= {f"reading:{r.seed_id}" for r in m.readings}
    refs |= {f"reading_journal:{j.entry_id}" for j in m.reading_journals}
    refs |= {f"dream:{d.id}" for d in m.dream_narratives}
    refs |= {f"dream_offered:{i}" for i in range(1, len(m.dream_hypotheses) + 1)}
    if m.chat_compactor:
        refs.add(f"chat_compactor:{m.chat_compactor.entry_id}")
    if m.github_compactor:
        refs.add(f"github_compactor:{m.github_compactor.entry_id}")
    return refs


def full_bodies(m: OrionDayMaterialV1) -> list[str]:
    bodies = [r.journal_body for r in m.curiosity_runs] + [a.answer_text for a in m.self_sense]
    bodies += [r.learned for r in m.readings] + [j.body for j in m.reading_journals]
    bodies += [d.narrative or "" for d in m.dream_narratives]
    bodies += [c.body for c in (m.chat_compactor, m.github_compactor) if c is not None]
    return [b.strip() for b in bodies if b and b.strip()]


def evaluate(material: OrionDayMaterialV1) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    view = build_llm_view(material)
    od = {"letter_date": material.letter_date.isoformat(), "timezone": material.timezone, "digest_md": "\x00DIGEST\x00"}
    note_instr = _render("orion_day_note_v1.j2", od).split("\x00DIGEST\x00")[0].lower()
    leaked = [w for w in CARRY_FORWARD_WORDS if w in note_instr]
    cf = _render("orion_day_carry_forward_v1.j2", {**od, "digest_md": view.digest_md, "note_md": "\x00NOTE\x00"})
    ok = not leaked and "note_md" not in note_instr and "\x00NOTE\x00" in cf and view.digest_md in cf
    results.append(("separation", ok, f"note-instruction leaks={leaked}"))

    refs = extract_refs(view.digest_md)
    unresolved = sorted(set(refs) - material_refs(material))
    missing = sorted(required_refs(material) - set(refs))
    results.append(("grounding", not unresolved and not missing,
                    f"refs={len(refs)} unresolved={unresolved[:5]} missing={missing[:5]}"))

    leaked_ids = [h.hypothesis_id for h in material.dream_hypotheses if h.hypothesis_id in view.digest_md]
    ok = not leaked_ids and "dream_hypothesis:" not in view.digest_md
    results.append(("blind", ok, f"hypotheses={len(material.dream_hypotheses)} ids_in_view={leaked_ids[:3]}"))

    if view.condensed.full_text_clip_chars is None:
        # Bodies appear verbatim apart from the documented heading demotion (their own markdown
        # headings are pushed below the digest's ### item headings).
        absent = [b[:60] for b in full_bodies(material) if _Body(b).render(None) not in view.digest_md]
        results.append(("full_text", not absent, f"bodies={len(full_bodies(material))} absent={absent[:3]}"))
    else:
        results.append(("full_text", True, f"clipped at {view.condensed.full_text_clip_chars} chars "
                                           f"({view.condensed.full_text_items_clipped} items) -- reported"))

    c = view.condensed
    shown = [r for r in refs if r.startswith("reverie:")]
    hollow_shown = [t.thought_id for t in material.reverie_thoughts
                    if t.hollow and t.interpretation.strip() and t.interpretation.strip() in view.digest_md]
    by_prefix = {}
    for t in material.reverie_thoughts:
        by_prefix.setdefault(t.thought_id[:8], t.chain_id)
    chains = [by_prefix.get(r.split(":", 1)[1]) for r in shown]
    ok = (c.reverie_thoughts_included <= c.reverie_thoughts_total and not hollow_shown
          and len([x for x in chains if x]) == len({x for x in chains if x}))
    results.append(("condensation", ok, f"shown {c.reverie_thoughts_included}/{c.reverie_thoughts_total}, "
                                        f"themes {c.reverie_themes_included}/{c.reverie_themes_total}, hollow_shown={len(hollow_shown)}"))

    note_prompt_tokens = int(len(_render("orion_day_note_v1.j2", {**od, "digest_md": view.digest_md})) / view.chars_per_token)
    ok = view.approx_tokens <= view.budget_tokens and note_prompt_tokens + NOTE_MAX_TOKENS < AGENT_CONTEXT_TOKENS
    results.append(("budget", ok, f"digest~{view.approx_tokens} tokens (budget {view.budget_tokens}); "
                                  f"note prompt~{note_prompt_tokens} + {NOTE_MAX_TOKENS} < {AGENT_CONTEXT_TOKENS}"))

    again = build_llm_view(OrionDayMaterialV1.model_validate_json(material.model_dump_json()))
    results.append(("determinism", again.digest_md == view.digest_md, "rebuilt from JSON"))
    return results


async def _fixture_material() -> OrionDayMaterialV1:
    from orion.orion_day.tests import fixtures as fx
    return await gather_orion_day(fx.FakeConn(), fx.LETTER_DATE, now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc))


async def _live_material(letter_date: date | None, dsn: str | None) -> OrionDayMaterialV1:
    import asyncpg

    sys.path.insert(0, str(ROOT / "scripts"))
    from smoke_orion_day_letter import _dsn  # noqa: E402
    from orion.orion_day.window import yesterday_letter_date

    conn = await asyncpg.connect(_dsn(dsn), server_settings={"default_transaction_read_only": "on"})
    try:
        return await gather_orion_day(conn, letter_date or yesterday_letter_date(datetime.now(timezone.utc)))
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--date")
    parser.add_argument("--dsn")
    args = parser.parse_args()
    if args.live:
        material = asyncio.run(_live_material(date.fromisoformat(args.date) if args.date else None, args.dsn))
    else:
        material = asyncio.run(_fixture_material())
    results = evaluate(material)
    print(f"orion_day eval -- {'live' if args.live else 'fixture'} day {material.letter_date}")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name:13s} {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
