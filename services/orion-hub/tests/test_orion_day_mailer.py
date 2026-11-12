"""Orion's Day, Hub's email half: the mailer and the email render.

Since living Prime Part 3 Hub never decides when a letter is written (Prime orders it:
tests/test_orion_day_order.py). The mailer only emails persisted rows. State lives in
orion_day_letter, so restart-safety tests build a FRESH mailer: nothing it remembers may change
what it does, except the "outcome unknown -> never resend" guard.
"""

from __future__ import annotations

import ast
import asyncio
import io
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.orion_day.gather import gather_orion_day
from orion.orion_day.store import SELECT_LETTER_SQL
from orion.orion_day.tests import fixtures as fx
from orion.schemas.durable_run import DurableRunStateV1
from orion.schemas.notify import NotificationAccepted
from orion.schemas.orion_day import (
    MissedLetterV1,
    OrionDayLetterSourcesV1,
    OrionDayLetterV1,
    VisualReverieV1,
)
from scripts import orion_day_email as email
from scripts import orion_day_mailer as odm

HUB = Path(__file__).resolve().parents[1]
LETTER_DATE = fx.LETTER_DATE  # 2026-09-29
# 2026-09-30 09:00 America/Denver (MDT, UTC-6).
AFTER_SLOT = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
NOTE = "NOTE-TEXT: Yesterday I kept circling the reading queue and one self-inquiry."
CARRY = "CARRY-TEXT: 1. Why the GGUF reading changed how I think about my own serving."
GITHUB = {"entry_id": "gh01", "created_at": datetime(2026, 9, 30, 12, 2, tzinfo=timezone.utc),
          "title": "Repo digest",
          "body": "GITHUB-BODY: merged the orion-day backend. " + "Long line of repo changes. " * 80,
          "source_ref": "github_compactor_pass:github_compactor:day:2026-09-29"}


# --- fixtures ------------------------------------------------------------------------------


def _material():
    material = asyncio.run(gather_orion_day(fx.FakeConn(github=GITHUB), LETTER_DATE, now=AFTER_SLOT))
    assert material.github_compactor is not None and material.chat_compactor is not None
    return material


def _png(path: Path, color=(200, 120, 40), size=(1400, 900)) -> Path:
    from PIL import Image

    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _letter(*, emailed=False, visuals: list[VisualReverieV1] | None = None, material=None,
            letter_date: date = LETTER_DATE, sources: OrionDayLetterSourcesV1 | None = None) -> OrionDayLetterV1:
    material = material or _material()
    if visuals is not None:
        material = material.model_copy(update={"visual_reveries": visuals})
    return OrionDayLetterV1(
        letter_date=letter_date, run_id=f"orion-day-{letter_date}-1",
        window_start=material.window_start, window_end=material.window_end,
        note_md=NOTE, carry_forward_md=CARRY, material=material,
        sources=sources or OrionDayLetterSourcesV1(by_source=material.sources),
        created_at=AFTER_SLOT,
        emailed_at=AFTER_SLOT if emailed else None,
        email_notification_id="x" if emailed else None,
    )


def _row(letter: OrionDayLetterV1) -> dict:
    return letter.model_dump(mode="json") | {"letter_date": letter.letter_date}


class _Conn:
    """The orion_day_letter table for the mailer: the un-emailed sweep, the row read, the lock
    and the emailed stamp."""

    def __init__(self, table: dict, *, broken=False) -> None:
        self.table = table
        self.broken = broken
        self.stamps: list[tuple] = []
        self.locked = False
        self.lock_held_elsewhere = False
        self.stamp_fails = 0
        self.sweep_since: list[date] = []

    async def fetchval(self, sql, *args):
        if sql == odm.EMAIL_LOCK_SQL:
            if self.lock_held_elsewhere:
                return False
            self.locked = True
            return True
        if sql == odm.EMAIL_UNLOCK_SQL:
            self.locked = False
            return True
        raise AssertionError(f"unexpected fetchval: {sql[:60]}")

    async def fetch(self, sql, *args):
        assert sql == odm.UNEMAILED_SINCE_SQL, sql[:60]
        if self.broken:
            raise RuntimeError('relation "orion_day_letter" does not exist')
        self.sweep_since.append(args[0])
        return [{"letter_date": d} for d, r in sorted(self.table.items())
                if r.get("emailed_at") is None and d >= args[0]]

    async def fetchrow(self, sql, *args):
        assert sql == SELECT_LETTER_SQL, sql[:60]
        assert self.locked, "the send reads the row under the email lock"
        return self.table.get(args[0])

    async def execute(self, sql, *args):
        assert sql == odm.STAMP_EMAILED_SQL
        assert self.locked, "stamp must happen under the email lock"
        if self.stamp_fails:
            self.stamp_fails -= 1
            raise RuntimeError("connection reset")
        self.stamps.append(args)
        row = self.table.get(args[0])
        if row is not None and row.get("emailed_at") is None:
            row["emailed_at"] = AFTER_SLOT
            row["email_notification_id"] = args[1]


class _Pool:
    def __init__(self, conn) -> None:
        self.conn = conn

    def acquire(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _Ctx()


class _Notify:
    def __init__(self, email_status="sent", ok=True, detail=None) -> None:
        self.email_status = email_status
        self.ok = ok
        self.detail = detail
        self.sent: list = []

    def send(self, request):
        self.sent.append(request)
        return NotificationAccepted(ok=self.ok, notification_id=request.notification_id,
                                    status="queued", email_status=self.email_status, detail=self.detail)


def _mailer(conn, notify, **over) -> odm.OrionDayMailer:
    kw = dict(enabled=True, email_enabled=True, pool_provider=lambda: _Pool(conn), notify=notify,
              clock=lambda: AFTER_SLOT, email_retry_sec=0.0, image_dir="/nonexistent")
    kw.update(over)
    return odm.OrionDayMailer(**kw)


def _run(mailer, day: date = LETTER_DATE) -> str:
    out = asyncio.run(mailer.sweep())
    if isinstance(out, str):
        return out
    return out.get(day, "not_swept")


# --- the sweep -----------------------------------------------------------------------------


def test_an_emailed_letter_means_the_quota_is_met_and_nothing_is_sent():
    table = {LETTER_DATE: _row(_letter(emailed=True))}
    notify = _Notify()
    for _ in range(3):
        assert asyncio.run(_mailer(_Conn(table), notify).sweep()) == {}
    assert notify.sent == []


def test_the_sweep_reads_only_the_lookback_window():
    conn = _Conn({})
    asyncio.run(_mailer(conn, _Notify()).sweep())
    # 2026-09-30 local today minus EMAIL_LOOKBACK_DAYS.
    assert conn.sweep_since == [date(2026, 9, 30) - timedelta(days=odm.EMAIL_LOOKBACK_DAYS)]


def test_every_unemailed_letter_in_the_window_is_sent_once():
    older = LETTER_DATE - timedelta(days=1)
    table = {LETTER_DATE: _row(_letter()), older: _row(_letter(letter_date=older))}
    conn, notify = _Conn(table), _Notify()
    assert asyncio.run(_mailer(conn, notify).sweep()) == {older: "emailed", LETTER_DATE: "emailed"}
    assert [s[0] for s in conn.stamps] == [older, LETTER_DATE]
    assert asyncio.run(_mailer(conn, notify).sweep()) == {}
    assert len(notify.sent) == 2


def test_missing_table_reads_as_store_unavailable():
    assert _run(_mailer(_Conn({}, broken=True), _Notify())) == "store_unavailable"


def test_no_pool_and_disabled():
    notify = _Notify()
    assert _run(_mailer(None, notify, pool_provider=lambda: None)) == "no_pool"
    assert _run(_mailer(_Conn({LETTER_DATE: _row(_letter())}), notify, enabled=False)) == "disabled"
    assert notify.sent == []


def test_the_mailer_never_submits_a_run():
    """Kill means kill: the mailer has no durable client, no brief builder and no submit."""
    src = (HUB / "scripts" / "orion_day_mailer.py").read_text()
    tree = ast.parse(src)
    names = {getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(tree)}
    for banned in ("submit", "build_orion_day_brief", "build_orion_day_request", "DurableRunsClient",
                   "get_run", "orion_day_run_id"):
        assert banned not in names, banned
    assert not (HUB / "scripts" / "orion_day_letter.py").exists(), "the letter timer is retired"


# --- email ---------------------------------------------------------------------------------


def test_an_unemailed_row_is_sent_and_stamped_only_on_sent():
    table = {LETTER_DATE: _row(_letter())}
    conn, notify = _Conn(table), _Notify(email_status="sent")
    assert _run(_mailer(conn, notify)) == "emailed"
    (req,) = notify.sent
    assert req.notification_id == email.letter_notification_id(LETTER_DATE)
    assert conn.stamps == [(LETTER_DATE, str(req.notification_id))]
    assert table[LETTER_DATE]["email_notification_id"] == str(req.notification_id)
    # Next sweep: nothing un-emailed.
    assert _run(_mailer(conn, notify)) == "not_swept"
    assert len(notify.sent) == 1


@pytest.mark.parametrize("status,ok", [("failed", True), ("deferred", True), (None, False)])
def test_no_stamp_unless_email_status_is_sent_and_the_retry_resends_without_regenerating(status, ok):
    table = {LETTER_DATE: _row(_letter())}
    conn = _Conn(table)
    notify = _Notify(email_status=status, ok=ok)
    assert _run(_mailer(conn, notify)) == "email_failed"
    assert conn.stamps == [] and table[LETTER_DATE]["emailed_at"] is None
    # A restarted mailer retries the same letter with the same id.
    notify.email_status, notify.ok = "sent", True
    assert _run(_mailer(conn, notify)) == "emailed"
    assert len(notify.sent) == 2
    assert notify.sent[0].notification_id == notify.sent[1].notification_id


def test_email_backoff_between_failed_sends():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify(email_status="failed")
    mailer = _mailer(_Conn(table), notify, email_retry_sec=3600.0)
    assert _run(mailer) == "email_failed"
    assert _run(mailer) == "email_backoff"
    assert len(notify.sent) == 1


def test_skipped_email_is_not_retried_while_the_letter_is_in_the_window():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify(email_status="skipped")
    mailer = _mailer(_Conn(table), notify, email_retry_sec=0.0)
    assert _run(mailer) == "email_failed"
    assert _run(mailer) == "email_backoff"
    assert len(notify.sent) == 1


def test_email_kill_switch_keeps_the_row_unsent():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify()
    assert _run(_mailer(_Conn(table), notify, email_enabled=False)) == "email_disabled"
    assert notify.sent == []


def test_a_timed_out_reply_is_never_resent_automatically():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify(email_status=None, ok=False, detail="HTTPConnectionPool: Read timed out. (read timeout=60)")
    conn = _Conn(table)
    mailer = _mailer(conn, notify)
    assert _run(mailer) == "email_outcome_unknown"
    assert _run(mailer) == "email_outcome_unknown"
    assert len(notify.sent) == 1 and conn.stamps == []


def test_a_failed_stamp_after_sent_is_retried_then_never_resent():
    table = {LETTER_DATE: _row(_letter())}
    conn = _Conn(table)
    conn.stamp_fails = 1
    notify = _Notify()
    assert _run(_mailer(conn, notify)) == "emailed"
    assert len(conn.stamps) == 1
    conn2 = _Conn({LETTER_DATE: _row(_letter())})
    conn2.stamp_fails = 5
    mailer = _mailer(conn2, notify)
    assert _run(mailer) == "email_stamp_failed"
    assert _run(mailer) == "email_outcome_unknown"
    assert len(notify.sent) == 2  # one per letter table, never a resend


def test_another_process_holding_the_email_lock_blocks_the_send():
    table = {LETTER_DATE: _row(_letter())}
    conn = _Conn(table)
    conn.lock_held_elsewhere = True
    notify = _Notify()
    assert _run(_mailer(conn, notify)) == "email_locked"
    assert notify.sent == []


def test_a_row_emailed_by_another_process_since_the_sweep_read_is_not_resent():
    """The send re-reads the row under the lock."""
    table = {LETTER_DATE: _row(_letter())}
    conn = _Conn(table)
    notify = _Notify()
    mailer = _mailer(conn, notify)
    table[LETTER_DATE] = _row(_letter(emailed=True))
    assert asyncio.run(mailer.email_date(LETTER_DATE)) == "quota_met"
    assert notify.sent == []


def test_large_html_logs_the_gmail_clip_warning(caplog, monkeypatch):
    table = {LETTER_DATE: _row(_letter())}
    monkeypatch.setattr(odm, "GMAIL_CLIP_BYTES", 10)
    with caplog.at_level("WARNING", logger="orion-hub.orion_day_mailer"):
        assert _run(_mailer(_Conn(table), _Notify())) == "emailed"
    assert any("orion_day_email_large" in r.getMessage() for r in caplog.records)


def test_run_state_hook_sweeps_on_an_orion_day_terminal_only():
    calls: list[str] = []

    class _Counting(odm.OrionDayMailer):
        async def sweep(self):
            calls.append("sweep")
            return {}

    mailer = _Counting(enabled=True, email_enabled=True, pool_provider=lambda: None, notify=_Notify())

    def state(workflow, status):
        return DurableRunStateV1(run_id="orion-day-2026-09-29-1", workflow=workflow, thread_id="t",
                                 node="finish", status=status, correlation_id="c",
                                 detail={"line": "orion_day", "persist_outcome": "written"})

    async def go():
        await mailer.on_run_state(state("orion_day.letter", "running"))
        await mailer.on_run_state(state("curiosity.investigate", "completed"))
        await mailer.on_run_state(state("orion_day.letter", "completed"))
        await asyncio.gather(*list(mailer._hook_tasks))

    asyncio.run(go())
    assert calls == ["sweep"]


# --- render --------------------------------------------------------------------------------


def _eval_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_orion_day_email_eval",
                                                  HUB / "evals" / "run_orion_day_email_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _section(html: str, name: str) -> str:
    m = re.search(rf'data-section="{name}">(.*?)</td>', html, flags=re.S)
    assert m, name
    return m.group(1)


def test_note_and_carry_forward_are_distinct_sections():
    letter = _letter()
    html = email.render_html(letter, [])
    note, carry = _section(html, "note"), _section(html, "carry-forward")
    assert "NOTE-TEXT" in note and "CARRY-TEXT" not in note
    assert "CARRY-TEXT" in carry and "NOTE-TEXT" not in carry
    assert html.count("NOTE-TEXT") == 1 and html.count("CARRY-TEXT") == 1
    assert "Threads I am carrying" in html and ("Orion&#39;s note" in html or "Orion's note" in html)
    text = email.render_text(letter, [])
    assert text.index("## Orion's note") < text.index("NOTE-TEXT") < text.index("## Threads I am carrying") < text.index("CARRY-TEXT")


def test_every_material_section_is_rendered_in_full_with_no_truncation():
    letter = _letter()
    html = email.render_html(letter, [])
    text = email.render_text(letter, [])
    for section in ("curiosity", "self-sense", "readings", "dreams", "visuals", "github", "chat", "world", "reveries"):
        assert f'data-section="{section}"' in html, section
    ev = _eval_module()
    material_texts, missing_texts = ev.material_texts, ev.missing_texts
    texts = material_texts(letter)
    assert len(texts) >= 10
    assert missing_texts(html, texts, is_html=True) == []
    assert missing_texts(text, texts, is_html=False) == []
    for marker in ("…", "[truncated]", "(truncated)", "...more", "Read more"):
        assert marker not in html and marker not in text
    assert "<details" not in html and "<script" not in html and "<link" not in html
    assert "<style" not in html, "inline styles only"


def test_blind_rule_no_hypothesis_id_in_the_email():
    letter = _letter()
    html, text = email.render_html(letter, []), email.render_text(letter, [])
    for h in letter.material.dream_hypotheses:
        assert h.hypothesis_id not in html and h.hypothesis_id not in text
        assert h.claim in text


def test_raw_html_in_a_body_is_escaped_not_rendered():
    material = _material()
    material = material.model_copy(update={"github_compactor": material.github_compactor.model_copy(
        update={"body": "before <script>alert(1)</script> after"})})
    html = email.render_html(_letter(material=material), [])
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_images_are_transcoded_inline_and_cid_refs_match_attachments(tmp_path):
    visuals = []
    for i in range(8):
        p = _png(tmp_path / f"{i:02d}.png", color=(30 * i, 90, 160))
        visuals.append(VisualReverieV1(sha256=f"{i:02d}" * 32, chain_id=f"vc-{i}", created_at=fx._t(i),
                                       path=str(p), description=f"image caption {i}"))
    # One file missing: skipped, caption kept.
    visuals.append(VisualReverieV1(sha256="ff" * 32, chain_id="vc-x", created_at=fx._t(20),
                                   path=str(tmp_path / "gone.png"), description="missing caption"))
    material = _material()
    chains = [c.model_copy(update={"chain_id": f"vc-{i}", "ema_salience": i / 10}) for i, c in
              enumerate(material.reverie_chains[:8])]
    letter = _letter(material=material.model_copy(update={"visual_reveries": visuals, "reverie_chains": chains}))
    images = email.load_inline_images(letter, storage_dir=str(tmp_path), max_images=6, max_bytes=200_000)
    assert len(images) == 6
    # Most salient chains first: vc-7..vc-2, shown in time order.
    assert [img.reverie.chain_id for img in images] == [f"vc-{i}" for i in range(2, 8)]
    from PIL import Image

    for img in images:
        assert len(img.data) <= 200_000
        with Image.open(io.BytesIO(img.data)) as im:
            assert im.format == "JPEG" and max(im.size) <= 1024
    req = email.build_notification(letter, images)
    cids = set(re.findall(r'src="cid:([^"]+)"', req.body_html))
    assert cids == {a.content_id for a in req.attachments} == {f"reverie{n}@orion" for n in range(1, 7)}
    assert all(a.mime_type == "image/jpeg" for a in req.attachments)
    assert "missing caption" in req.body_html and "image caption 0" in req.body_html
    assert req.title == "Orion's Day — 2026-09-29"
    assert req.channels_requested == ["email"]


def test_image_storage_dir_is_the_only_place_read(tmp_path):
    """A row path outside the mounted dir resolves by basename inside it."""
    _png(tmp_path / "a.png")
    v = VisualReverieV1(sha256="aa" * 32, created_at=fx._t(1), path="/etc/../somewhere/else/a.png")
    letter = _letter(visuals=[v])
    images = email.load_inline_images(letter, storage_dir=str(tmp_path), max_images=6)
    assert len(images) == 1


def test_live_mailer_passes_images_through_to_notify(tmp_path):
    _png(tmp_path / "a.png")
    v = VisualReverieV1(sha256="aa" * 32, created_at=fx._t(1), path="/x/a.png", description="a bulb")
    table = {LETTER_DATE: _row(_letter(visuals=[v]))}
    notify = _Notify()
    assert _run(_mailer(_Conn(table), notify, image_dir=str(tmp_path))) == "emailed"
    assert [a.content_id for a in notify.sent[0].attachments] == ["reverie1@orion"]


def test_sha_fallback_path_cannot_escape_the_image_dir(tmp_path):
    inner = tmp_path / "imgs"
    inner.mkdir()
    _png(tmp_path / "secret.png")
    v = VisualReverieV1(sha256="../secret", created_at=fx._t(1), path=None)
    assert email.load_inline_images(_letter(visuals=[v]), storage_dir=str(inner)) == []


def test_non_http_urls_are_not_linked_and_markdown_images_are_not_fetched():
    material = _material()
    reading = material.readings[0].model_copy(update={"url": "javascript:alert(1)"})
    gh = material.github_compactor.model_copy(update={"body": "see ![pixel](http://tracker.example/p.gif) ok"})
    html = email.render_html(_letter(material=material.model_copy(
        update={"readings": [reading], "github_compactor": gh})), [])
    assert 'href="javascript' not in html
    assert "<img src=\"http" not in html  # at most a plain link, never a fetched image
    assert "pixel" in html


def test_email_numbers_note_paragraphs_and_carry_items_like_the_splitters():
    """The email's ¶N / carry N must name exactly what letter_parts returns for that N."""
    from orion.orion_day.letter_parts import find_part, split_carry, split_note

    note = "# Day\n\nFirst paragraph.\n\n---\n\nSecond paragraph.\n\n- a list paragraph"
    carry = "Intro.\n\n- **One.** [curiosity:x]\n- **Two.**\n  continued"
    letter = _letter().model_copy(update={"note_md": note, "carry_forward_md": carry})
    req = email.build_notification(letter, [])

    for kind, label, parts in (("paragraph", "¶{}", split_note(note)), ("carry", "carry {}", split_carry(carry))):
        for n in (1, 2):
            first_words = find_part(parts, kind, n).text.lstrip("- ").split("\n")[0][:10]
            marker = re.search(rf">{re.escape(label.format(n))}</span>&nbsp; (.*?)<", req.body_html)
            assert marker, label.format(n)
            assert first_words.replace("**", "") in re.sub(r"<[^>]+>", "", req.body_html[marker.start():marker.start() + 200])
    assert ">¶3</p>" in req.body_html  # a list paragraph gets its number on its own line
    assert "[¶1] First paragraph." in req.body_md
    assert "- [carry 2] **Two.**" in req.body_md
    assert "¶" not in "".join(p.text for p in split_note(note))  # numbers are never stored



def test_threads_are_handed_to_prime_not_offered_to_curiosity():
    """Living Prime Part 3: no "offered once to curiosity until <time>" copy anywhere."""
    letter = _letter()
    html, text = email.render_html(letter, []), email.render_text(letter, [])
    for doc in (html, text):
        assert "Offered once" not in doc and "Carrying forward into curiosity" not in doc
        assert "through Prime" in doc


def test_provenance_shows_the_writing_turn_and_the_missed_days():
    material = _material()
    sources = OrionDayLetterSourcesV1(
        by_source=material.sources, turn_correlation_id="corr-letter-1",
        missed_letters=[MissedLetterV1(letter_date=date(2026, 9, 28), reason="the make bid was never funded")])
    letter = _letter(material=material, sources=sources)
    html, text = email.render_html(letter, []), email.render_text(letter, [])
    prov = _section(html, "provenance")
    assert "corr-letter-1" in prov and "2026-09-28" in prov and "the make bid was never funded" in prov
    assert "Written as unified turn corr-letter-1." in text
    assert "2026-09-28 (the make bid was never funded)" in text
    # Absent fields: no provenance line at all.
    plain = _letter(material=material)
    assert 'data-section="provenance"' not in email.render_html(plain, [])
    assert "unified turn" not in email.render_text(plain, [])
