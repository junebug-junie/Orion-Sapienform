#!/usr/bin/env python3
"""Orion's Day email eval: does the rendered letter carry every material item's FULL text?

The letter promises "everything above is shown in full". This checks that promise against the
real renderer (scripts/orion_day_email.py), independently of it: the expected texts are pulled
straight from the material model, normalised the same way on both sides (markdown syntax,
HTML tags/entities and whitespace removed), and each must appear as a substring of both the
HTML part and the plain-text part. It also checks the note and carry-forward land in their own
sections, no dream hypothesis id leaks, and every cid: reference has an attachment.

    python services/orion-hub/evals/run_orion_day_email_eval.py                  # fixture day
    python services/orion-hub/evals/run_orion_day_email_eval.py --material m.json  # e.g. a live
        # day gathered read-only with orion.orion_day.gather (scripts/smoke_orion_day_letter.py)
    ... --write-preview out.html                                                  # also save HTML

Offline: no DB, notify or network. Exit 0 = every check passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import asyncio
import html as html_lib
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]
warnings.filterwarnings("ignore", message=r"Field \"model_", category=UserWarning)

PLACEHOLDER_NOTE = ("PLACEHOLDER NOTE -- not written by Orion. The real note is the durable run's "
                    "`orion_day_note_v1` output; this preview only shows where it goes and how it reads.")
PLACEHOLDER_CARRY = ("PLACEHOLDER CARRY-FORWARD -- not written by Orion.\n\n"
                     "1. PLACEHOLDER thread one.\n2. PLACEHOLDER thread two.")


def _paths() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_HUB_ROOT)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_HUB_ROOT))


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_MARKS = re.compile(r"[*_`#>|~\\]")
_LIST_MARK = re.compile(r"(?m)^\s*(?:[-+*]|\d+[.)])\s+")
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def normalise_source(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _LIST_MARK.sub(" ", text)
    text = _MD_MARKS.sub("", text)
    text = text.replace("-", "")  # table rules / hr / soft dashes all collapse the same way
    # Whitespace dropped entirely: markdown -> HTML moves spaces around inline tags
    # (`code`, **bold**), and what this eval checks is the words, in order.
    return _WS.sub("", text)


def normalise_doc(doc: str, *, is_html: bool) -> str:
    if is_html:
        doc = _TAGS.sub(" ", doc)
        doc = html_lib.unescape(doc)
        doc = _LIST_MARK.sub(" ", doc)
        doc = _MD_MARKS.sub("", doc).replace("-", "")
        return _WS.sub("", doc)
    return normalise_source(doc)


def material_texts(letter) -> list[tuple[str, str]]:
    """(label, full text) for every material item the letter promises to show in full."""
    m = letter.material
    # The note and carry-forward render part by part with a number between parts
    # (orion/orion_day/letter_parts.py), so each part must be present in full.
    from orion.orion_day.letter_parts import split_carry, split_note

    out: list[tuple[str, str]] = [(f"note:{i}", p.text) for i, p in enumerate(split_note(letter.note_md))]
    out += [(f"carry_forward:{i}", p.text) for i, p in enumerate(split_carry(letter.carry_forward_md))]
    for r in m.curiosity_runs:
        for name in ("journal_body", "self_definition_text", "lived_answer_text"):
            if getattr(r, name):
                out.append((f"curiosity:{r.run_id}:{name}", getattr(r, name)))
        if not r.journal_body and r.finding_text:
            out.append((f"curiosity:{r.run_id}:finding_text", r.finding_text))
    out += [(f"curiosity_failed:{f.run_id}", f.error) for f in m.curiosity_failed if f.error]
    for a in m.self_sense:
        out += [(f"self_sense:{a.question_key}:q", a.question), (f"self_sense:{a.question_key}:a", a.answer_text)]
    for r in m.readings:
        out += [(f"reading:{r.seed_id}:{k}", v) for k, v in
                (("title", r.title), ("learned", r.learned), ("why_now", r.why_now)) if v]
    out += [(f"reading_journal:{j.entry_id}", j.body) for j in m.reading_journals]
    for d in m.dream_narratives:
        out += [(f"dream:{d.id}:{k}", v) for k, v in (("tldr", d.tldr), ("narrative", d.narrative)) if v]
    for i, h in enumerate(m.dream_hypotheses):
        out += [(f"dream_hypothesis:{i}:{k}", v) for k, v in (("claim", h.claim), ("why", h.why)) if v]
    out += [(f"visual:{v.sha256[:12]}", v.description) for v in m.visual_reveries if v.description]
    if m.github_compactor:
        out.append(("github_compactor", m.github_compactor.body))
    if m.chat_compactor:
        out.append(("chat_compactor", m.chat_compactor.body))
    if m.world_pulse_digest:
        w = m.world_pulse_digest
        if w.executive_summary:
            out.append(("world:summary", w.executive_summary))
        for n, item in enumerate(w.items):
            out += [(f"world:{n}:{k}", v) for k, v in (("title", item.title), ("summary", item.summary)) if v]
    return [(label, text) for label, text in out if text and text.strip()]


def missing_texts(doc: str, texts: list[tuple[str, str]], *, is_html: bool) -> list[str]:
    hay = normalise_doc(doc, is_html=is_html)
    return [label for label, text in texts if normalise_source(text) not in hay]


def check(letter, images) -> list[str]:
    from scripts.orion_day_email import build_notification

    req = build_notification(letter, images)
    failures: list[str] = []
    texts = material_texts(letter)
    failures += [f"html missing {label}" for label in missing_texts(req.body_html, texts, is_html=True)]
    failures += [f"text missing {label}" for label in missing_texts(req.body_md, texts, is_html=False)]
    for h in letter.material.dream_hypotheses:
        if h.hypothesis_id in req.body_html or h.hypothesis_id in req.body_md:
            failures.append(f"hypothesis id leaked {h.hypothesis_id}")
    cids = set(re.findall(r'src="cid:([^"]+)"', req.body_html))
    attached = {a.content_id for a in (req.attachments or [])}
    if cids != attached:
        failures.append(f"cid mismatch refs={sorted(cids)} attachments={sorted(attached)}")
    note = re.search(r'data-section="note">(.*?)</td>', req.body_html, flags=re.S)
    carry = re.search(r'data-section="carry-forward">(.*?)</td>', req.body_html, flags=re.S)
    if not note or not carry:
        failures.append("note or carry-forward section missing")
    elif normalise_source(letter.carry_forward_md)[:60] in normalise_doc(note.group(1), is_html=True):
        failures.append("carry-forward text inside the note section")
    return failures


def _letter_from_material(material, *, note: str, carry: str):
    from orion.schemas.orion_day import OrionDayLetterSourcesV1, OrionDayLetterV1

    created = material.window_end + timedelta(hours=8, minutes=40)
    return OrionDayLetterV1(
        letter_date=material.letter_date, run_id=f"orion-day-{material.letter_date}-1",
        window_start=material.window_start, window_end=material.window_end,
        note_md=note, carry_forward_md=carry, material=material,
        sources=OrionDayLetterSourcesV1(by_source=material.sources),
        created_at=created, carry_forward_expires_at=created + timedelta(hours=36),
    )


def fixture_letter():
    from orion.orion_day.gather import gather_orion_day
    from orion.orion_day.tests import fixtures as fx

    github = {"entry_id": "gh01", "created_at": datetime(2026, 9, 30, 12, 2, tzinfo=timezone.utc),
              "title": "Repo digest", "body": "Merged the backend. " * 50,
              "source_ref": "github_compactor_pass:github_compactor:day:2026-09-29"}
    material = asyncio.run(gather_orion_day(fx.FakeConn(github=github), fx.LETTER_DATE,
                                            now=datetime(2026, 9, 30, 15, tzinfo=timezone.utc)))
    return _letter_from_material(material, note=PLACEHOLDER_NOTE, carry=PLACEHOLDER_CARRY)


def main(argv: list[str] | None = None) -> int:
    _paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", help="OrionDayMaterialV1 JSON (default: the fixture day)")
    ap.add_argument("--images-dir", default="", help="reverie image dir; empty = no images")
    ap.add_argument("--write-preview", default="", help="write the rendered HTML here")
    args = ap.parse_args(argv)

    from orion.schemas.orion_day import OrionDayMaterialV1
    from scripts.orion_day_email import build_notification, load_inline_images

    if args.material:
        material = OrionDayMaterialV1.model_validate_json(Path(args.material).read_text())
        letter = _letter_from_material(material, note=PLACEHOLDER_NOTE, carry=PLACEHOLDER_CARRY)
    else:
        letter = fixture_letter()
    images = load_inline_images(letter, storage_dir=args.images_dir) if args.images_dir else []
    failures = check(letter, images)
    texts = material_texts(letter)
    req = build_notification(letter, images)
    print(f"letter {letter.letter_date}: {len(texts)} full-text items, {len(images)} images, "
          f"html {len(req.body_html)} chars, text {len(req.body_md)} chars, "
          f"attachments {sum(len(a.content_base64) for a in (req.attachments or []))} b64 bytes")
    if args.write_preview:
        Path(args.write_preview).write_text(req.body_html)
        print(f"preview written: {args.write_preview}")
    for f in failures:
        print("FAIL", f)
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
