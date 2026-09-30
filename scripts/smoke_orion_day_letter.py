#!/usr/bin/env python3
"""Live smoke for Orion's Day: gather a day from Postgres READ-ONLY, print what it holds.

Opens one asyncpg connection with ``default_transaction_read_only=on`` (the server refuses
any write), runs orion/orion_day/gather.py for the letter date, builds the budgeted model
view, renders both verb prompts, and prints section counts, source status, condensation
counts, prompt sizes and brief/request sizes. Writes nothing, calls no model.

Usage:
    python scripts/smoke_orion_day_letter.py                      # yesterday (America/Denver)
    python scripts/smoke_orion_day_letter.py --date 2026-09-29
    python scripts/smoke_orion_day_letter.py --dsn postgresql://...  # else ORION_DAY_SMOKE_DSN / DATABASE_URL
    python scripts/smoke_orion_day_letter.py --show-digest           # also print the digest text

Exit: 0 = gathered and budgeted; 1 = a source errored or the day is empty; 2 = could not connect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jinja2 import Environment  # noqa: E402

from orion.orion_day.brief import OrionDayEmptyError, brief_from_material, build_orion_day_request  # noqa: E402
from orion.orion_day.budget import material_refs  # noqa: E402
from orion.orion_day.gather import gather_orion_day  # noqa: E402
from orion.orion_day.window import yesterday_letter_date  # noqa: E402

PROMPTS = ROOT / "orion" / "cognition" / "prompts"


def _dsn(arg: str | None) -> str:
    if arg:
        return arg
    for key in ("ORION_DAY_SMOKE_DSN", "DATABASE_URL"):
        if os.environ.get(key):
            return os.environ[key]
    env = ROOT.parent / "Orion-Sapienform" / "services" / "orion-hub" / ".env"
    for candidate in (ROOT / "services" / "orion-hub" / ".env", env):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line.startswith("DATABASE_URL="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("no DSN: pass --dsn or set ORION_DAY_SMOKE_DSN / DATABASE_URL")


def render(template: str, metadata: dict) -> str:
    return Environment(autoescape=False).from_string((PROMPTS / template).read_text()).render(metadata=metadata)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="letter date YYYY-MM-DD (default: yesterday, America/Denver)")
    parser.add_argument("--dsn")
    parser.add_argument("--show-digest", action="store_true")
    args = parser.parse_args()
    letter_date = date.fromisoformat(args.date) if args.date else yesterday_letter_date(datetime.now(timezone.utc))

    import asyncpg

    try:
        conn = await asyncpg.connect(_dsn(args.dsn), server_settings={"default_transaction_read_only": "on"})
    except Exception as exc:  # noqa: BLE001
        print(f"connect failed: {type(exc).__name__}: {exc}")
        return 2
    try:
        readonly = await conn.fetchval("SHOW default_transaction_read_only")
        material = await gather_orion_day(conn, letter_date)
    finally:
        await conn.close()

    print(f"letter_date={letter_date} window=[{material.window_start.isoformat()}, {material.window_end.isoformat()}) "
          f"read_only_session={readonly}")
    print("\n== sections (material, full length)")
    sections = {
        "curiosity_runs": material.curiosity_runs, "curiosity_failed": material.curiosity_failed,
        "self_sense": material.self_sense, "readings": material.readings,
        "reading_journals": material.reading_journals, "dream_narratives": material.dream_narratives,
        "dream_hypotheses": material.dream_hypotheses, "reverie_thoughts": material.reverie_thoughts,
        "reverie_chains": material.reverie_chains, "visual_reveries": material.visual_reveries,
    }
    for name, items in sections.items():
        chars = sum(len(json.dumps(i.model_dump(mode="json"))) for i in items)
        print(f"  {name:18s} count={len(items):5d} json_chars={chars}")
    for name in ("chat_compactor", "github_compactor", "world_pulse_digest"):
        value = getattr(material, name)
        size = len(json.dumps(value.model_dump(mode="json"))) if value is not None else 0
        print(f"  {name:18s} present={value is not None} json_chars={size}")
    print("\n== sources")
    for name, status in sorted(material.sources.items()):
        print(f"  {name:18s} {status.status:5s} count={status.count}" + (f" error={status.error}" if status.error else ""))

    try:
        brief = brief_from_material(material)
    except OrionDayEmptyError as exc:
        print(f"\nEMPTY DAY: {exc}")
        return 1
    view = brief.llm_view
    print("\n== model view (budgeted)")
    print(f"  digest_chars={len(view.digest_md)} approx_tokens={view.approx_tokens} budget_tokens={view.budget_tokens}")
    print(f"  condensed={view.condensed.model_dump()}")
    unknown = [r for r in view.included_refs if r not in material_refs(material)]
    print(f"  included_refs={len(view.included_refs)} unresolved_refs={len(unknown)}")

    note_prompt = render("orion_day_note_v1.j2", {"orion_day_input": {
        "letter_date": str(letter_date), "timezone": brief.timezone, "digest_md": view.digest_md}})
    cf_prompt = render("orion_day_carry_forward_v1.j2", {"orion_day_input": {
        "letter_date": str(letter_date), "timezone": brief.timezone, "digest_md": view.digest_md,
        "note_md": "(note placeholder, ~12k chars in production)"}})
    print("\n== prompts")
    print(f"  note_prompt_chars={len(note_prompt)} (~{int(len(note_prompt) / view.chars_per_token)} tokens)")
    print(f"  carry_forward_prompt_chars={len(cf_prompt)} (+ the note)")

    request = build_orion_day_request(brief)
    print("\n== durable request")
    print(f"  run_id={request.run_id} workflow={request.workflow} resource={request.admission.resource} "
          f"priority={request.admission.priority} deadline_at={request.admission.deadline_at.isoformat()}")
    print(f"  minimum_context_tokens={request.admission.requirements.get('minimum_context_tokens')} "
          f"(the chat card serves 65,536; agent / agent-gpu2 serve 131,072 -- live /props 2026-09-30)")
    print(f"  request_json_bytes={len(request.model_dump_json())} material_json_bytes={len(material.model_dump_json())}")
    if args.show_digest:
        print("\n== digest\n" + view.digest_md)
    bad = [n for n, s in material.sources.items() if s.status == "error"] or unknown
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
