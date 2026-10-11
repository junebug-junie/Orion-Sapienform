"""Orion's Day email: render a persisted ``orion_day_letter`` row to HTML + plain text.

Pure rendering (no bus, no DB, no network). ``scripts/orion_day_letter.py`` owns when to send.

Rules this module holds:

* Nothing is truncated. Every material body goes out at full length, in both the HTML and the
  plain-text part (``tests/test_orion_day_letter.py`` and ``evals/run_orion_day_email_eval.py``
  check every item's full text is present).
* ``note_md`` (Orion's note) and ``carry_forward_md`` (threads offered to curiosity) are two
  separate, visually distinct sections. Neither is merged into the other.
* Email-safe HTML: tables and inline styles only, no external CSS/JS/fonts, no ``<details>``
  (Gmail strips it). Images are inline CID parts (``cid:reverie1@orion``).
* Blind dream rule: dream hypotheses show claim + why only. No hypothesis id (the dream
  scorecard's join key) and no arm -- the material model has no arm field at all.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from orion.orion_day.letter_parts import LetterPart, split_carry, split_note
from orion.schemas.notify import NotificationAttachment, NotificationRequest
from orion.schemas.orion_day import OrionDayLetterV1, VisualReverieV1

logger = logging.getLogger("orion-hub.orion_day_email")

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
TEMPLATE_NAME = "orion_day_letter.html.j2"
EVENT_KIND = "orion_day.letter"
_NOTIFICATION_NS = uuid5(NAMESPACE_URL, "orion.orion_day.letter.email.v1")

DEFAULT_MAX_IMAGES = 6
DEFAULT_IMAGE_MAX_BYTES = 450_000
DEFAULT_IMAGE_MAX_PX = 1024


def letter_notification_id(letter_date: date | str) -> UUID:
    """One email identity per letter day: a resend after a lost reply reuses it."""
    return uuid5(_NOTIFICATION_NS, str(letter_date))


def letter_subject(letter_date: date | str) -> str:
    return f"Orion's Day — {letter_date}"


# --- markdown -> email-safe HTML ------------------------------------------------------------

_MD = None

# Inline styles per tag: Gmail keeps inline styles; a <style> block is not guaranteed.
_TAG_STYLES = {
    "p": "margin:0 0 12px",
    "h1": "font-size:20px;margin:18px 0 8px 0;line-height:1.3;",
    "h2": "font-size:18px;margin:16px 0 8px 0;line-height:1.3;",
    "h3": "font-size:16px;margin:14px 0 6px 0;line-height:1.3;",
    "h4": "font-size:15px;margin:12px 0 6px 0;",
    "h5": "font-size:14px;margin:12px 0 6px 0;",
    "h6": "font-size:14px;margin:12px 0 6px 0;",
    "ul": "margin:0 0 12px 0;padding-left:22px;",
    "ol": "margin:0 0 12px 0;padding-left:22px;",
    "li": "margin:0 0 4px",
    "blockquote": "margin:0 0 12px 0;padding:4px 12px;border-left:3px solid #9aa4b2;color:#4a5260;",
    "pre": "margin:0 0 12px 0;padding:10px;background-color:#f2f4f7;color:#1f2430;"
           "white-space:pre-wrap;word-wrap:break-word;font-family:Menlo,Consolas,monospace;font-size:13px;",
    "code": "font-family:Menlo,Consolas,monospace;font-size:13px;background-color:#f2f4f7;color:#1f2430;",
    "table": "border-collapse:collapse;margin:0 0 12px 0;",
    "th": "border:1px solid #d0d5dd;padding:4px 8px;text-align:left;",
    "td": "border:1px solid #d0d5dd;padding:4px 8px;vertical-align:top;",
    "hr": "border:0;border-top:1px solid #d0d5dd;margin:16px 0;",
    "a": "color:#2f5fb3;",
    "img": "max-width:100%;height:auto;",
}
_TAG_RE = re.compile(r"<(" + "|".join(sorted(_TAG_STYLES, key=len, reverse=True)) + r")(?=[\s>/])")


def _md():
    global _MD
    if _MD is None:
        from markdown_it import MarkdownIt

        # html=False: raw HTML inside a journal body is escaped, never rendered.
        # `image` disabled: a body's ![](http://...) must not pull remote images (tracking
        # pixels, broken boxes) into the letter; the alt text stays as plain text.
        _MD = (MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": True})
               .enable("table").disable("image"))
    return _MD


def markdown_to_html(text: str | None) -> Markup:
    if not text or not text.strip():
        return Markup("")
    html = _md().render(text)
    html = _TAG_RE.sub(lambda m: f'<{m.group(1)} style="{_TAG_STYLES[m.group(1)]}"', html)
    return Markup(html)


# --- part numbers (orion/orion_day/letter_parts.py) -----------------------------------------
# Juniper points at "2026-10-09 ¶3" / "2026-10-09 carry 5"; Orion's reread tool resolves the
# same numbers with the same splitters, so the email and the tool can never disagree.

_MARK_STYLE = "color:#9aa4b2;font-size:12px;font-weight:normal;"
_FIRST_OPEN_RE = re.compile(r"^\s*(<(?:p|li)\b[^>]*>)")
_FIRST_LI_RE = re.compile(r"(<li\b[^>]*>)")


def part_label(part: LetterPart) -> str:
    return f"¶{part.index}" if part.kind == "paragraph" else f"carry {part.index}"


def _marked_html(part: LetterPart) -> str:
    html = str(markdown_to_html(part.text))
    if part.index is None:
        return html
    mark = f'<span style="{_MARK_STYLE}">{part_label(part)}</span>&nbsp; '
    # A carry item renders as its own one-item list: mark inside the <li>. A prose paragraph
    # gets the mark inside its <p>. Anything else (a list or quote as a note paragraph) gets
    # the mark on its own line above it.
    pattern = _FIRST_LI_RE if part.kind == "carry" else _FIRST_OPEN_RE
    marked, n = pattern.subn(lambda m: m.group(1) + mark, html, count=1)
    return marked if n else f'<p style="margin:0 0 4px;{_MARK_STYLE}">{part_label(part)}</p>' + html


def numbered_html(parts: list[LetterPart]) -> Markup:
    return Markup("".join(_marked_html(p) for p in parts))


_BULLET_RE = re.compile(r"^((?:[-*+]|\d+[.)])\s+)")


def numbered_text(parts: list[LetterPart]) -> str:
    out: list[str] = []
    for p in parts:
        if p.index is None:
            out.append(p.text)
        elif p.kind == "carry":
            out.append(_BULLET_RE.sub(lambda m: m.group(1) + f"[{part_label(p)}] ", p.text, count=1))
        else:
            out.append(f"[{part_label(p)}] {p.text}")
    return "\n\n".join(out)


# --- images ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class InlineImage:
    content_id: str
    filename: str
    data: bytes
    reverie: VisualReverieV1

    @property
    def cid_src(self) -> str:
        return f"cid:{self.content_id}"


def select_images(letter: OrionDayLetterV1, max_images: int) -> list[VisualReverieV1]:
    """Most salient first (the chain's EMA salience), newest breaking ties; shown in time order."""
    if max_images <= 0:
        return []
    salience = {c.chain_id: c.ema_salience for c in letter.material.reverie_chains}

    def key(v: VisualReverieV1):
        s = salience.get(v.chain_id or "")
        return (s if s is not None else -1.0, v.created_at)

    picked = sorted(letter.material.visual_reveries, key=key, reverse=True)[:max_images]
    return sorted(picked, key=lambda v: v.created_at)


def _image_path(storage_dir: str, reverie: VisualReverieV1) -> Path:
    # Resolve inside the configured (mounted) dir by basename only: a path in the row can never
    # point the reader elsewhere on disk.
    name = Path(reverie.path or f"{reverie.sha256}.png").name
    base = Path(storage_dir).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise FileNotFoundError(str(path))
    return path


def transcode_jpeg(raw: bytes, *, max_px: int, max_bytes: int) -> bytes | None:
    """PNG/WebP/etc. -> JPEG no larger than ``max_px`` on the long side and ``max_bytes``."""
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        for scale in (1.0, 0.8, 0.64, 0.5):
            frame = im if scale == 1.0 else im.resize(
                (max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            for quality in (85, 75, 65, 55):
                buf = io.BytesIO()
                frame.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
                if buf.tell() <= max_bytes:
                    return buf.getvalue()
    return None


def load_inline_images(
    letter: OrionDayLetterV1,
    *,
    storage_dir: str,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_bytes: int = DEFAULT_IMAGE_MAX_BYTES,
    max_px: int = DEFAULT_IMAGE_MAX_PX,
) -> list[InlineImage]:
    """Read and transcode up to ``max_images`` reverie images. A missing/unreadable file is
    skipped (logged), never fatal: the reverie's caption still appears in the letter."""
    out: list[InlineImage] = []
    for reverie in select_images(letter, max_images):
        path = _image_path(storage_dir, reverie)
        try:
            data = transcode_jpeg(path.read_bytes(), max_px=max_px, max_bytes=max_bytes)
        except FileNotFoundError:
            logger.warning("orion_day_image_missing sha=%s path=%s", reverie.sha256[:16], path)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("orion_day_image_unreadable sha=%s err=%s", reverie.sha256[:16], exc)
            continue
        if data is None:
            logger.warning("orion_day_image_over_cap sha=%s cap=%s", reverie.sha256[:16], max_bytes)
            continue
        n = len(out) + 1
        out.append(InlineImage(content_id=f"reverie{n}@orion", filename=f"reverie{n}.jpg",
                               data=data, reverie=reverie))
    return out


# --- context ---------------------------------------------------------------------------------


def _local(value: datetime | None, tz: ZoneInfo, fmt: str = "%H:%M") -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(tz).strftime(fmt)


def _yn(value: bool | None) -> str:
    return "yes" if value is True else "no" if value is False else "unknown"


def _outcome_line(o: dict[str, Any] | None) -> str:
    if not o:
        return ""
    parts = [f"turn ok: {_yn(o.get('turn_ok'))}", f"priors tested {o.get('n_tested', 0)}",
             f"moved {o.get('n_moved', 0)}", f"formed {o.get('n_formed', 0)}"]
    moves = [f"{p.get('prior_id')} {p.get('kind')} {p.get('before')}→{p.get('after')}"
             for p in (o.get("per_prior") or []) if isinstance(p, dict)]
    line = ", ".join(parts)
    if moves:
        line += "; " + "; ".join(moves)
    if o.get("unknown_reason"):
        line += f"; unknown reason: {o['unknown_reason']}"
    return line


def reverie_counts(letter: OrionDayLetterV1) -> dict[str, Any]:
    m = letter.material
    themes = Counter(c.theme_key for c in m.reverie_chains if c.theme_key)
    verdicts = Counter(t.expectation_verdict for t in m.reverie_thoughts if t.expectation_verdict)
    return {
        "thoughts": len(m.reverie_thoughts),
        "hollow": sum(1 for t in m.reverie_thoughts if t.hollow),
        "chains": len(m.reverie_chains),
        "themes": len(themes),
        "top_themes": themes.most_common(8),
        "verdicts": sorted(verdicts.items()),
        "images": len(m.visual_reveries),
    }


def build_context(letter: OrionDayLetterV1, images: list[InlineImage]) -> dict[str, Any]:
    m = letter.material
    tz = ZoneInfo(m.timezone)
    cid_by_sha = {img.reverie.sha256: img.cid_src for img in images}
    curiosity = [{
        "title": r.journal_title or "Curiosity run",
        "line": r.line or "unknown",
        "family": r.self_question_family,
        "when": _local(r.completed_at, tz),
        "verdict": _outcome_line(r.outcome),
        "continue_line": _yn(r.continue_line),
        "reach_out": _yn(r.reach_out),
        "reach_out_why": r.reach_out_why,
        "self_definition": markdown_to_html(r.self_definition_text),
        "lived_answer": markdown_to_html(r.lived_answer_text),
        "writeup": markdown_to_html(r.journal_body or r.finding_text or ""),
    } for r in m.curiosity_runs]
    visuals = [{
        "when": _local(v.created_at, tz),
        "description": v.description or "(no caption)",
        "theme": v.theme_key,
        "cid": cid_by_sha.get(v.sha256),
    } for v in sorted(m.visual_reveries, key=lambda v: v.created_at)]
    source_problems = [(name, s.status, s.error, s.truncated) for name, s in sorted(m.sources.items())
                       if s.status == "error" or s.truncated]
    return {
        "subject": letter_subject(letter.letter_date),
        "letter_date": letter.letter_date,
        "letter_date_long": letter.letter_date.strftime("%A, %B %-d, %Y"),
        "timezone": m.timezone,
        "window_start_utc": letter.window_start.strftime("%Y-%m-%d %H:%MZ"),
        "window_end_utc": letter.window_end.strftime("%Y-%m-%d %H:%MZ"),
        "run_id": letter.run_id,
        "written_at": _local(letter.created_at, tz, "%Y-%m-%d %H:%M"),
        "note_html": numbered_html(split_note(letter.note_md)),
        "carry_forward_html": numbered_html(split_carry(letter.carry_forward_md)),
        "carry_forward_expires": _local(letter.carry_forward_expires_at, tz, "%Y-%m-%d %H:%M"),
        "curiosity": curiosity,
        "curiosity_failed": [{"workflow": f.workflow, "when": _local(f.failed_at, tz),
                              "error": f.error or "no error recorded"} for f in m.curiosity_failed],
        "self_sense": [{"question": a.question, "answer": markdown_to_html(a.answer_text),
                        "self_label": a.self_label_score, "grounded": a.grounded_record_score,
                        "when": _local(a.created_at, tz)} for a in m.self_sense],
        "readings": [{"title": r.title or r.url or "Reading", "url": r.url, "why_now": r.why_now,
                      "href": r.url if (r.url or "").lower().startswith(("http://", "https://")) else None,
                      "status": r.reading_status or "unknown", "when": _local(r.occurred_at, tz),
                      "learned": markdown_to_html(r.learned)} for r in m.readings],
        "reading_journals": [{"title": j.title or "Reading journal", "when": _local(j.created_at, tz),
                              "body": markdown_to_html(j.body)} for j in m.reading_journals],
        "dream_narratives": [{"tldr": d.tldr or "Dream", "narrative": markdown_to_html(d.narrative)}
                             for d in m.dream_narratives],
        # Blind rule: claim + why only (no hypothesis_id, no arm).
        "dream_hypotheses": [{"claim": h.claim, "why": h.why} for h in m.dream_hypotheses],
        "visuals": visuals,
        "images_attached": len(images),
        "github": ({"title": m.github_compactor.title or "Repo digest",
                    "body": markdown_to_html(m.github_compactor.body)} if m.github_compactor else None),
        "chat": ({"title": m.chat_compactor.title or "Chat digest",
                  "body": markdown_to_html(m.chat_compactor.body)} if m.chat_compactor else None),
        "world": m.world_pulse_digest,
        "reveries": reverie_counts(letter),
        "source_problems": source_problems,
        "condensed": letter.sources.condensed,
    }


_ENV: Environment | None = None


def _env() -> Environment:
    global _ENV
    if _ENV is None:
        _ENV = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                           autoescape=select_autoescape(["html", "j2"]),
                           trim_blocks=True, lstrip_blocks=True)
    return _ENV


def render_html(letter: OrionDayLetterV1, images: list[InlineImage]) -> str:
    return _env().get_template(TEMPLATE_NAME).render(**build_context(letter, images))


def render_text(letter: OrionDayLetterV1, images: list[InlineImage]) -> str:
    """The plain-text part: the full note, carry-forward and material, as markdown."""
    m = letter.material
    tz = ZoneInfo(m.timezone)
    ctx_images = {img.reverie.sha256: img.filename for img in images}
    out: list[str] = [
        f"# Orion's Day — {letter.letter_date}",
        f"The day: {letter.window_start:%Y-%m-%d %H:%MZ} to {letter.window_end:%Y-%m-%d %H:%MZ} ({m.timezone}).",
        "",
        "## Orion's note",
        "",
        numbered_text(split_note(letter.note_md)).strip(),
        "",
        "## Carrying forward into curiosity",
        "(Offered once to Orion's next regular curiosity run"
        + (f", until {_local(letter.carry_forward_expires_at, tz, '%Y-%m-%d %H:%M')}" if letter.carry_forward_expires_at else "")
        + ".)",
        "",
        numbered_text(split_carry(letter.carry_forward_md)).strip(),
        "",
        "---",
        "# What the day held",
    ]
    if m.curiosity_runs or m.curiosity_failed:
        out += ["", f"## Curiosity ({len(m.curiosity_runs)} finished, {len(m.curiosity_failed)} failed)"]
        for r in m.curiosity_runs:
            out += ["", f"### {r.journal_title or 'Curiosity run'} (line: {r.line or 'unknown'}"
                    + (f", family: {r.self_question_family}" if r.self_question_family else "")
                    + f", {_local(r.completed_at, tz)})"]
            if r.outcome:
                out.append(f"Verdict: {_outcome_line(r.outcome)}")
            out.append(f"Line continues: {_yn(r.continue_line)}. Reached out: {_yn(r.reach_out)}"
                       + (f" -- {r.reach_out_why}" if r.reach_out_why else "") + ".")
            if r.self_definition_text:
                out += ["", "Self-definition after this run:", "", r.self_definition_text]
            if r.lived_answer_text:
                out += ["", "Lived answer after this run:", "", r.lived_answer_text]
            body = r.journal_body or r.finding_text or ""
            if body:
                out += ["", body]
        for f in m.curiosity_failed:
            out.append(f"- Failed {_local(f.failed_at, tz)} ({f.workflow}): {f.error or 'no error recorded'}")
    if m.self_sense:
        out += ["", "## Self-sense answers"]
        for a in m.self_sense:
            out += ["", f"### {a.question}", f"(self-label {a.self_label_score}, grounded-record {a.grounded_record_score})",
                    "", a.answer_text]
    if m.readings or m.reading_journals:
        out += ["", f"## Readings ({len(m.readings)} readings, {len(m.reading_journals)} reading journals)"]
        for r in m.readings:
            out += ["", f"### {r.title or r.url or 'Reading'}", f"Source: {r.url or 'unknown'}. Status: {r.reading_status or 'unknown'}."]
            if r.why_now:
                out.append(f"Why: {r.why_now}")
            out += ["", "What was learned:", "", r.learned or "(nothing recorded)"]
        for j in m.reading_journals:
            out += ["", f"### Reading journal: {j.title or ''}".rstrip(), "", j.body]
    if m.dream_narratives or m.dream_hypotheses:
        out += ["", "## Dreams"]
        for d in m.dream_narratives:
            out += ["", f"### {d.tldr or 'Dream'}", "", d.narrative or ""]
        if m.dream_hypotheses:
            out += ["", "Links the dream cycle offered to Orion:"]
            for h in m.dream_hypotheses:
                out.append(f"- {h.claim}" + (f" -- {h.why}" if h.why else ""))
    if m.visual_reveries:
        out += ["", f"## Visual reveries ({len(m.visual_reveries)} images, {len(images)} attached)"]
        for v in sorted(m.visual_reveries, key=lambda v: v.created_at):
            att = ctx_images.get(v.sha256)
            out.append(f"- {_local(v.created_at, tz)}" + (f" [{att}]" if att else "") + f": {v.description or '(no caption)'}")
    if m.github_compactor is not None:
        out += ["", f"## Changes to Orion's code: {m.github_compactor.title or 'Repo digest'}", "", m.github_compactor.body]
    if m.chat_compactor is not None:
        out += ["", f"## Conversations with Juniper: {m.chat_compactor.title or 'Chat digest'}", "", m.chat_compactor.body]
    if m.world_pulse_digest is not None:
        w = m.world_pulse_digest
        out += ["", f"## World news: {w.title or 'World pulse'}"]
        if w.executive_summary:
            out += ["", w.executive_summary]
        for i in w.items:
            out.append(f"- ({i.category or 'general'}) {i.title}"
                       + (f": {i.summary}" if i.summary and i.summary != i.title else "")
                       + (f" Why it matters: {i.why_it_matters}" if i.why_it_matters and i.why_it_matters not in (i.title, i.summary) else ""))
    rc = reverie_counts(letter)
    out += ["", "## Reveries (counts)",
            f"{rc['thoughts']} thoughts ({rc['hollow']} hollow) across {rc['chains']} chains, {rc['themes']} themes."]
    if rc["top_themes"]:
        out.append("Most frequent themes: " + ", ".join(f"{t} ({n})" for t, n in rc["top_themes"]))
    return "\n".join(out).rstrip() + "\n"


def build_notification(
    letter: OrionDayLetterV1,
    images: list[InlineImage],
    *,
    source_service: str = "orion-hub",
) -> NotificationRequest:
    return NotificationRequest(
        notification_id=letter_notification_id(letter.letter_date),
        source_service=source_service,
        event_kind=EVENT_KIND,
        severity="info",
        title=letter_subject(letter.letter_date),
        body_md=render_text(letter, images),
        body_html=render_html(letter, images),
        context={"letter_date": str(letter.letter_date), "run_id": letter.run_id,
                 "images_attached": len(images)},
        tags=["orion_day", "letter"],
        channels_requested=["email"],
        dedupe_key=f"orion_day:{letter.letter_date}",
        correlation_id=letter.run_id,
        attachments=[NotificationAttachment(
            filename=img.filename,
            content_base64=base64.b64encode(img.data).decode("ascii"),
            mime_type="image/jpeg",
            content_id=img.content_id,
        ) for img in images] or None,
    )
