"""Orion's Day, Hub half: scheduling, email render, carry-forward into curiosity.

State lives in orion_day_letter + the durable registry, so every loop test builds a FRESH
loop per tick where restart safety matters: nothing the loop remembers may change what it does.
"""

from __future__ import annotations

import ast
import asyncio
import io
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from orion.orion_day import carry_forward as cf
from orion.orion_day.brief import OrionDayEmptyError
from orion.orion_day.gather import gather_orion_day
from orion.orion_day.store import SELECT_LETTER_SQL
from orion.orion_day.tests import fixtures as fx
from orion.schemas.durable_run import DurableRunStateV1
from orion.schemas.notify import NotificationAccepted
from orion.schemas.orion_day import (
    OrionDayLetterSourcesV1,
    OrionDayLetterV1,
    VisualReverieV1,
)
from scripts import orion_day_email as email
from scripts import orion_day_letter as odl

HUB = Path(__file__).resolve().parents[1]
LETTER_DATE = fx.LETTER_DATE  # 2026-09-29
# 2026-09-30 09:00 America/Denver (MDT, UTC-6) -> letter 2026-09-29 is active.
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


def _letter(*, emailed=False, visuals: list[VisualReverieV1] | None = None, material=None) -> OrionDayLetterV1:
    material = material or _material()
    if visuals is not None:
        material = material.model_copy(update={"visual_reveries": visuals})
    return OrionDayLetterV1(
        letter_date=LETTER_DATE, run_id="orion-day-2026-09-29-1",
        window_start=material.window_start, window_end=material.window_end,
        note_md=NOTE, carry_forward_md=CARRY, material=material,
        sources=OrionDayLetterSourcesV1(by_source=material.sources),
        created_at=AFTER_SLOT, carry_forward_expires_at=AFTER_SLOT + timedelta(hours=36),
        emailed_at=AFTER_SLOT if emailed else None,
        email_notification_id="x" if emailed else None,
    )


def _row(letter: OrionDayLetterV1) -> dict:
    return letter.model_dump(mode="json") | {"letter_date": letter.letter_date}


class _Conn(fx.FakeConn):
    """Gather's fixture conn + the orion_day_letter row (read + emailed stamp)."""

    def __init__(self, table: dict, *, empty=False, broken=False) -> None:
        super().__init__(github=GITHUB)
        self.table = table
        self.empty = empty
        self.broken = broken
        self.stamps: list[tuple] = []
        self.locked = False
        self.lock_held_elsewhere = False
        self.stamp_fails = 0

    async def fetchval(self, sql, *args):
        if sql == odl.EMAIL_LOCK_SQL:
            if self.lock_held_elsewhere:
                return False
            self.locked = True
            return True
        if sql == odl.EMAIL_UNLOCK_SQL:
            self.locked = False
            return True
        return await super().fetchval(sql, *args)

    async def fetch(self, sql, *args):
        if self.empty:
            self.calls.append((sql, args))
            return []
        return await super().fetch(sql, *args)

    async def fetchrow(self, sql, *args):
        if sql == SELECT_LETTER_SQL:
            if self.broken:
                raise RuntimeError('relation "orion_day_letter" does not exist')
            return self.table.get(args[0])
        if self.empty:
            return None
        return await super().fetchrow(sql, *args)

    async def execute(self, sql, *args):
        assert sql == odl.STAMP_EMAILED_SQL
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


class _Durable:
    """The durable registry: {run_id: status dict}. submit() records and registers."""

    def __init__(self, runs: dict[str, dict] | None = None, *, down=False, submit_status=202) -> None:
        self.runs = dict(runs or {})
        self.down = down
        self.submit_status = submit_status
        self.submitted: list[dict] = []
        self.gets: list[str] = []

    async def get_run(self, run_id):
        if self.down:
            raise odl.DurableUnavailable("ConnectError")
        self.gets.append(run_id)
        return self.runs.get(run_id)

    async def submit(self, request_json):
        self.submitted.append(request_json)
        if self.submit_status < 400:
            self.runs[request_json["run_id"]] = {"run_id": request_json["run_id"], "status": "waiting_resource"}
        return self.submit_status, ""


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


def _loop(conn, durable, notify, **over) -> odl.OrionDayLetterLoop:
    kw = dict(enabled=True, email_enabled=True, pool_provider=lambda: _Pool(conn), durable=durable,
              notify=notify, clock=lambda: AFTER_SLOT, max_attempts=3, email_retry_sec=0.0,
              image_dir="/nonexistent")
    kw.update(over)
    return odl.OrionDayLetterLoop(**kw)


def _run(loop) -> str:
    return asyncio.run(loop.tick())


# --- the letter window ------------------------------------------------------------------------


def test_active_letter_is_yesterday_after_the_slot_and_the_day_before_until_then():
    before = datetime(2026, 9, 30, 14, 29, tzinfo=timezone.utc)  # 08:29 MDT
    after = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)   # 08:30 MDT
    assert odl.active_letter_date(before, hour=8, minute=30) == date(2026, 9, 28)
    assert odl.active_letter_date(after, hour=8, minute=30) == date(2026, 9, 29)
    # Hard stop: at the NEXT day's slot the 29th is no longer active.
    next_slot = datetime(2026, 10, 1, 14, 30, tzinfo=timezone.utc)
    assert odl.active_letter_date(next_slot, hour=8, minute=30) == date(2026, 9, 30)
    assert odl.letter_slot(LETTER_DATE, hour=8, minute=30) == datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)


def test_a_letter_not_done_by_the_next_slot_is_abandoned():
    """The 29th's letter exists unemailed, but it is now past 10-01 08:30: the active letter is
    the 30th, so the 29th is never emailed late."""
    table = {LETTER_DATE: _row(_letter())}
    notify, durable = _Notify(), _Durable()
    late = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)
    loop = _loop(_Conn(table), durable, notify, clock=lambda: late)
    assert _run(loop) == "submitted"
    assert notify.sent == []
    assert durable.submitted[0]["run_id"] == "orion-day-2026-09-30-1"


# --- scheduling ----------------------------------------------------------------------------


def test_an_emailed_letter_means_the_quota_is_met_and_nothing_else_happens():
    table = {LETTER_DATE: _row(_letter(emailed=True))}
    notify, durable = _Notify(), _Durable()
    for _ in range(3):
        assert _run(_loop(_Conn(table), durable, notify)) == "quota_met"
    assert notify.sent == [] and durable.submitted == [] and durable.gets == []


def test_no_row_and_no_run_submits_attempt_one_with_the_approved_ttl():
    durable = _Durable()
    assert _run(_loop(_Conn({}), durable, _Notify(), carry_forward_ttl_hours=36.0)) == "submitted"
    (req,) = durable.submitted
    assert req["run_id"] == "orion-day-2026-09-29-1"
    assert req["workflow"] == "orion_day.letter"
    assert req["brief"]["carry_forward_ttl_hours"] == 36.0
    assert req["admission"]["priority"] == "background"


def test_a_live_run_is_left_alone():
    durable = _Durable({"orion-day-2026-09-29-1": {"status": "waiting_resource"}})
    assert _run(_loop(_Conn({}), durable, _Notify())) == "run_live"
    assert durable.submitted == []


def test_a_failed_run_is_retried_as_the_next_attempt_derived_from_durable_state():
    """Fresh loop (a restart): the attempt number comes from the registry, not memory."""
    durable = _Durable({
        "orion-day-2026-09-29-1": {"status": "failed", "error": "HoldLost"},
        "orion-day-2026-09-29-2": {"status": "abandoned", "error": "deadline"},
    })
    assert _run(_loop(_Conn({}), durable, _Notify())) == "submitted"
    assert [r["run_id"] for r in durable.submitted] == ["orion-day-2026-09-29-3"]
    # A second restarted loop sees attempt 3 live and does nothing.
    assert _run(_loop(_Conn({}), durable, _Notify())) == "run_live"
    assert len(durable.submitted) == 1


def test_attempts_are_capped_with_one_notice():
    durable = _Durable({f"orion-day-2026-09-29-{n}": {"status": "failed", "error": f"e{n}"} for n in (1, 2, 3)})
    notify = _Notify()
    loop = _loop(_Conn({}), durable, notify, max_attempts=3)
    assert _run(loop) == "attempts_exhausted"
    assert _run(loop) == "attempts_exhausted"
    assert durable.submitted == []
    assert len(notify.sent) == 1
    assert notify.sent[0].event_kind == "orion_day.letter.exhausted"
    assert notify.sent[0].channels_requested == ["in_app"]


def test_an_operator_cancel_is_not_retried():
    durable = _Durable({"orion-day-2026-09-29-1": {"status": "cancelled"}})
    assert _run(_loop(_Conn({}), durable, _Notify())) == "cancelled"
    assert durable.submitted == []


def test_durable_down_submits_nothing():
    durable = _Durable(down=True)
    assert _run(_loop(_Conn({}), durable, _Notify())) == "durable_unavailable"
    assert durable.submitted == []


def test_a_refused_submit_backs_off_instead_of_regathering_every_tick():
    durable = _Durable(submit_status=409)
    conn = _Conn({})
    loop = _loop(conn, durable, _Notify())
    assert _run(loop) == "submit_refused"
    assert _run(loop) == "submit_backoff"
    assert len(durable.submitted) == 1


def test_missing_table_reads_as_store_unavailable_and_submits_nothing():
    durable = _Durable()
    assert _run(_loop(_Conn({}, broken=True), durable, _Notify())) == "store_unavailable"
    assert durable.submitted == [] and durable.gets == []


def test_an_empty_day_is_skipped_without_a_run_or_an_email():
    durable, notify = _Durable(), _Notify()
    loop = _loop(_Conn({}, empty=True), durable, notify)
    assert _run(loop) == "empty_day"
    assert _run(loop) == "empty_day"
    assert durable.submitted == [] and notify.sent == []


def test_brief_builder_really_raises_on_an_empty_day():
    """The skip above is only honest if the #2435 builder raises for this fixture."""
    from orion.orion_day.brief import build_orion_day_brief

    with pytest.raises(OrionDayEmptyError):
        asyncio.run(build_orion_day_brief(_Conn({}, empty=True), LETTER_DATE, now=AFTER_SLOT))


def test_disabled_loop_does_nothing():
    durable = _Durable()
    assert _run(_loop(_Conn({}), durable, _Notify(), enabled=False)) == "disabled"
    assert durable.submitted == []


# --- email ---------------------------------------------------------------------------------


def test_an_unemailed_row_is_sent_and_stamped_only_on_sent():
    table = {LETTER_DATE: _row(_letter())}
    conn, notify, durable = _Conn(table), _Notify(email_status="sent"), _Durable()
    assert _run(_loop(conn, durable, notify)) == "emailed"
    (req,) = notify.sent
    assert req.notification_id == email.letter_notification_id(LETTER_DATE)
    assert conn.stamps == [(LETTER_DATE, str(req.notification_id))]
    assert table[LETTER_DATE]["email_notification_id"] == str(req.notification_id)
    assert durable.submitted == [] and durable.gets == [], "no run for a persisted letter"
    # Next tick: quota met.
    assert _run(_loop(conn, durable, notify)) == "quota_met"
    assert len(notify.sent) == 1


@pytest.mark.parametrize("status,ok", [("failed", True), ("deferred", True), (None, False)])
def test_no_stamp_unless_email_status_is_sent_and_the_retry_resends_without_regenerating(status, ok):
    table = {LETTER_DATE: _row(_letter())}
    conn, durable = _Conn(table), _Durable()
    notify = _Notify(email_status=status, ok=ok)
    assert _run(_loop(conn, durable, notify)) == "email_failed"
    assert conn.stamps == [] and table[LETTER_DATE]["emailed_at"] is None
    # A restarted loop retries the same letter with the same id; still no durable run.
    notify.email_status, notify.ok = "sent", True
    assert _run(_loop(conn, durable, notify)) == "emailed"
    assert len(notify.sent) == 2
    assert notify.sent[0].notification_id == notify.sent[1].notification_id
    assert durable.submitted == [] and durable.gets == []


def test_email_backoff_between_failed_sends():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify(email_status="failed")
    loop = _loop(_Conn(table), _Durable(), notify, email_retry_sec=3600.0)
    assert _run(loop) == "email_failed"
    assert _run(loop) == "email_backoff"
    assert len(notify.sent) == 1


def test_email_kill_switch_keeps_the_row_unsent():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify()
    assert _run(_loop(_Conn(table), _Durable(), notify, email_enabled=False)) == "email_disabled"
    assert notify.sent == []


def test_a_completed_run_whose_row_landed_between_reads_is_emailed():
    class _LateConn(_Conn):
        def __init__(self, table):
            super().__init__(table)
            self.reads = 0

        async def fetchrow(self, sql, *args):
            if sql == SELECT_LETTER_SQL:
                self.reads += 1
                if self.reads == 1:
                    return None
            return await super().fetchrow(sql, *args)

    table = {LETTER_DATE: _row(_letter())}
    durable = _Durable({"orion-day-2026-09-29-1": {"status": "completed", "orion_day": {"persist_outcome": "written"}}})
    notify = _Notify()
    assert _run(_loop(_LateConn(table), durable, notify)) == "emailed"
    assert durable.submitted == []


def test_run_state_hook_ticks_on_an_orion_day_terminal_only():
    calls: list[str] = []

    class _Counting(odl.OrionDayLetterLoop):
        async def tick(self):
            calls.append("tick")
            return "x"

    loop = _Counting(enabled=True, email_enabled=True, pool_provider=lambda: None,
                     durable=_Durable(), notify=_Notify())

    def state(workflow, status):
        return DurableRunStateV1(run_id="orion-day-2026-09-29-1", workflow=workflow, thread_id="t",
                                 node="finish", status=status, correlation_id="c",
                                 detail={"line": "orion_day", "persist_outcome": "written"})

    async def go():
        await loop.on_run_state(state("orion_day.letter", "running"))
        await loop.on_run_state(state("curiosity.investigate", "completed"))
        await loop.on_run_state(state("orion_day.letter", "completed"))
        await asyncio.gather(*list(loop._hook_tasks))

    asyncio.run(go())
    assert calls == ["tick"]


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
    assert "Carrying forward into curiosity" in html and "Orion&#39;s note" in html or "Orion's note" in html
    text = email.render_text(letter, [])
    assert text.index("## Orion's note") < text.index("NOTE-TEXT") < text.index("## Carrying forward") < text.index("CARRY-TEXT")


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


def test_live_loop_passes_images_through_to_notify(tmp_path):
    _png(tmp_path / "a.png")
    v = VisualReverieV1(sha256="aa" * 32, created_at=fx._t(1), path="/x/a.png", description="a bulb")
    table = {LETTER_DATE: _row(_letter(visuals=[v]))}
    notify = _Notify()
    assert _run(_loop(_Conn(table), _Durable(), notify, image_dir=str(tmp_path))) == "emailed"
    assert [a.content_id for a in notify.sent[0].attachments] == ["reverie1@orion"]


# --- carry-forward into curiosity ----------------------------------------------------------


class _LetterTable:
    """orion_day_letter semantics for the TAKE / RELEASE statements."""

    def __init__(self, rows: list[dict], now: datetime) -> None:
        self.rows = rows
        self.now = now

    async def fetchrow(self, sql, run_id):
        assert sql == cf.TAKE_CARRY_FORWARD_SQL
        eligible = [r for r in self.rows if r["carry_forward_offered_at"] is None
                    and r["carry_forward_expires_at"] > self.now]
        if not eligible:
            return None
        row = max(eligible, key=lambda r: r["letter_date"])
        row["carry_forward_offered_at"], row["carry_forward_offered_run_id"] = self.now, run_id
        return {"letter_date": row["letter_date"], "carry_forward_md": row["carry_forward_md"]}

    async def execute(self, sql, run_id):
        assert sql == cf.RELEASE_CARRY_FORWARD_SQL
        for r in self.rows:
            if r["carry_forward_offered_run_id"] == run_id:
                r["carry_forward_offered_at"] = r["carry_forward_offered_run_id"] = None


def _cf_rows(now):
    return [
        {"letter_date": date(2026, 9, 28), "carry_forward_md": "OLD threads", "note_md": "old note",
         "carry_forward_expires_at": now + timedelta(hours=2), "carry_forward_offered_at": None,
         "carry_forward_offered_run_id": None},
        {"letter_date": date(2026, 9, 29), "carry_forward_md": CARRY, "note_md": NOTE,
         "carry_forward_expires_at": now + timedelta(hours=30), "carry_forward_offered_at": None,
         "carry_forward_offered_run_id": None},
    ]


def test_carry_forward_is_claimed_once_newest_first_and_released_by_run():
    table = _LetterTable(_cf_rows(AFTER_SLOT), AFTER_SLOT)
    pool = _Pool(table)
    first = asyncio.run(cf.take_carry_forward(pool, run_id="run-a"))
    assert first == cf.OfferedCarryForward(letter_date=date(2026, 9, 29), text=CARRY)
    second = asyncio.run(cf.take_carry_forward(pool, run_id="run-b"))
    assert second.letter_date == date(2026, 9, 28), "the 29th was already offered"
    assert asyncio.run(cf.take_carry_forward(pool, run_id="run-c")) is None
    asyncio.run(cf.release_carry_forward(pool, run_id="run-a"))
    again = asyncio.run(cf.take_carry_forward(pool, run_id="run-d"))
    assert again.letter_date == date(2026, 9, 29)


def test_expired_carry_forward_is_never_offered():
    rows = _cf_rows(AFTER_SLOT)
    for r in rows:
        r["carry_forward_expires_at"] = AFTER_SLOT - timedelta(minutes=1)
    assert asyncio.run(cf.take_carry_forward(_Pool(_LetterTable(rows, AFTER_SLOT)), run_id="r")) is None


def test_take_sql_never_selects_the_note():
    assert "note_md" not in cf.TAKE_CARRY_FORWARD_SQL
    assert "carry_forward_expires_at > now()" in cf.TAKE_CARRY_FORWARD_SQL
    assert "FOR UPDATE SKIP LOCKED" in cf.TAKE_CARRY_FORWARD_SQL


def test_take_failure_is_silent():
    class _Boom:
        async def fetchrow(self, *a):
            raise RuntimeError('relation "orion_day_letter" does not exist')

    assert asyncio.run(cf.take_carry_forward(_Pool(_Boom()), run_id="r")) is None
    assert asyncio.run(cf.take_carry_forward(None, run_id="r")) is None


def test_kickoff_prompt_has_its_own_carry_forward_section_apart_from_dream_and_material():
    from orion.curiosity.kickoff_prompt import build_kickoff_prompt
    from orion.curiosity.study_material import StudyMaterial
    from orion.curiosity.worldview import WorldviewSnapshot
    from orion.dream.hypotheses import OfferedHypothesis

    offered = cf.OfferedCarryForward(letter_date=LETTER_DATE, text=CARRY)
    prompt = build_kickoff_prompt(
        StudyMaterial(generated_at=AFTER_SLOT), view=WorldviewSnapshot(), run_id="abcd1234abcd",
        dream_hypotheses=(OfferedHypothesis("dh-1", "a dream claim long enough"),), carry_forward=offered,
    )
    header = "THREADS CARRIED FORWARD FROM YESTERDAY'S REFLECTION (Orion's Day, 2026-09-29)"
    assert prompt.count(header) == 1 and prompt.count("CARRY-TEXT") == 1
    assert prompt.index("WHILE YOU SLEPT") < prompt.index(header)
    assert "NOTE-TEXT" not in prompt
    without = build_kickoff_prompt(StudyMaterial(generated_at=AFTER_SLOT), view=WorldviewSnapshot(), run_id="abcd1234abcd")
    assert header not in without


def _curiosity_loop(monkeypatch, *, enabled=True, text="found it"):
    from test_curiosity_investigation import _FakeBus, _loop as curiosity_loop

    taken: list[str] = []
    released: list[str] = []

    async def fake_take(pool, *, run_id):
        taken.append(run_id)
        return cf.OfferedCarryForward(letter_date=LETTER_DATE, text=CARRY)

    async def fake_release(pool, *, run_id):
        released.append(run_id)

    loop = curiosity_loop(_FakeBus(), text=text, carry_forward_enabled=enabled)
    # Patch the globals the loop's class actually runs with: conftest re-imports `scripts.*`
    # per test, so `scripts.curiosity_investigation` in sys.modules may be a newer copy.
    module_globals = type(loop)._take_carry_forward.__globals__
    monkeypatch.setitem(module_globals, "take_carry_forward", fake_take)
    monkeypatch.setitem(module_globals, "release_carry_forward", fake_release)
    return loop, taken, released


def test_regular_investigation_offers_carry_forward_in_its_prompt(monkeypatch):
    loop, taken, released = _curiosity_loop(monkeypatch)
    asyncio.run(loop.tick())
    assert len(taken) == 1 and released == []
    assert "CARRY-TEXT" in loop.seen_prompt and "NOTE-TEXT" not in loop.seen_prompt
    assert "Orion's Day, 2026-09-29" in loop.seen_prompt


def test_carry_forward_kill_switch(monkeypatch):
    loop, taken, _ = _curiosity_loop(monkeypatch, enabled=False)
    asyncio.run(loop.tick())
    assert taken == [] and "CARRY-TEXT" not in loop.seen_prompt


def test_a_cancelled_turn_releases_the_carry_forward(monkeypatch):
    loop, taken, released = _curiosity_loop(monkeypatch)

    async def _cancelled(*a, **k):
        raise asyncio.CancelledError()

    loop._generate = _cancelled  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.tick())
    assert released == taken and len(taken) == 1


def test_an_empty_generation_keeps_the_claim(monkeypatch):
    """Orion saw the prompt; an empty answer is not a cancel."""
    loop, taken, released = _curiosity_loop(monkeypatch, text="")
    asyncio.run(loop.tick())
    assert len(taken) == 1 and released == []


def test_only_the_regular_investigate_line_claims_carry_forward():
    """Static: `_take_carry_forward` is called from `_investigate` only (not urgent, not
    self-inquiry, not self-sense-eval), and the module-level take is called from nowhere else."""
    tree = ast.parse((HUB / "scripts" / "curiosity_investigation.py").read_text())
    callers: dict[str, set[str]] = {"_take_carry_forward": set(), "take_carry_forward": set()}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in callers:
                    callers[name].add(fn.name)
    assert callers["_take_carry_forward"] == {"_investigate"}
    assert callers["take_carry_forward"] == {"_take_carry_forward"}


def test_frozen_prompt_is_resent_on_retry():
    """Retries resend the brief's prompt (plus a resume preamble), never a fresh kickoff, so the
    carry-forward shown on attempt 1 is the one every attempt sees and no second claim happens."""
    src = (HUB / "scripts" / "curiosity_investigation.py").read_text()
    body = src[src.index("async def _prompt_for_attempt"):]
    body = body[:body.index("\n    async def ", 10)]
    assert "return request.prompt" in body and "preamble + request.prompt" in body
    assert "build_kickoff_prompt" not in body and "take_carry_forward" not in body


# --- review fixes --------------------------------------------------------------------------


def test_the_durable_deadline_is_the_hub_hard_stop_not_midnight():
    durable = _Durable()
    _run(_loop(_Conn({}), durable, _Notify()))
    deadline = datetime.fromisoformat(durable.submitted[0]["admission"]["deadline_at"].replace("Z", "+00:00"))
    assert deadline == datetime(2026, 10, 1, 14, 30, tzinfo=timezone.utc)  # 10-01 08:30 MDT


def test_skipped_email_waits_until_the_hard_stop():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify(email_status="skipped")
    loop = _loop(_Conn(table), _Durable(), notify, email_retry_sec=0.0)
    assert _run(loop) == "email_failed"
    assert _run(loop) == "email_backoff"
    assert len(notify.sent) == 1


def test_a_timed_out_reply_is_never_resent_automatically():
    table = {LETTER_DATE: _row(_letter())}
    notify = _Notify(email_status=None, ok=False, detail="HTTPConnectionPool: Read timed out. (read timeout=60)")
    conn = _Conn(table)
    loop = _loop(conn, _Durable(), notify)
    assert _run(loop) == "email_outcome_unknown"
    assert _run(loop) == "email_outcome_unknown"
    assert len(notify.sent) == 1 and conn.stamps == []


def test_a_failed_stamp_after_sent_is_retried_then_never_resent():
    table = {LETTER_DATE: _row(_letter())}
    conn = _Conn(table)
    conn.stamp_fails = 1
    notify = _Notify()
    assert _run(_loop(conn, _Durable(), notify)) == "emailed"
    assert len(conn.stamps) == 1
    conn2 = _Conn({LETTER_DATE: _row(_letter())})
    conn2.stamp_fails = 5
    loop = _loop(conn2, _Durable(), notify)
    assert _run(loop) == "email_stamp_failed"
    assert _run(loop) == "email_outcome_unknown"
    assert len(notify.sent) == 2  # one per letter table, never a resend


def test_another_process_holding_the_email_lock_blocks_the_send():
    table = {LETTER_DATE: _row(_letter())}
    conn = _Conn(table)
    conn.lock_held_elsewhere = True
    notify = _Notify()
    assert _run(_loop(conn, _Durable(), notify)) == "email_locked"
    assert notify.sent == []


def test_repeated_submit_refusals_raise_one_notice():
    durable, notify = _Durable(submit_status=422), _Notify()
    conn = _Conn({})
    loop = _loop(conn, durable, notify, submit_refused_retry_sec=0.0)
    for _ in range(odl.MAX_SUBMIT_REFUSALS + 2):
        assert _run(loop) == "submit_refused"
    assert len(notify.sent) == 1 and "submit_refused_http_422" in notify.sent[0].body_text


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


def _release_loop(monkeypatch, history, status=200):
    loop, taken, released = _curiosity_loop(monkeypatch)
    loop.durable_runs_url = "http://durable"
    g = type(loop)._release_carry_forward_if_unseen.__globals__

    class _Resp:
        status_code = status

        def json(self):
            return {"history": history}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp()

    class _Httpx:
        AsyncClient = _Client

    monkeypatch.setitem(g, "httpx", _Httpx)
    return loop, released


def test_an_unstarted_durable_curiosity_run_gives_its_carry_forward_back(monkeypatch):
    loop, released = _release_loop(monkeypatch, [{"event": "run.accepted"}, {"event": "run.abandoned"}])
    asyncio.run(loop._release_carry_forward_if_unseen("run-x"))
    assert released == ["run-x"]


def test_a_started_or_unknown_run_keeps_the_claim(monkeypatch):
    loop, released = _release_loop(monkeypatch, [{"event": "run.started"}])
    asyncio.run(loop._release_carry_forward_if_unseen("run-x"))
    loop2, released2 = _release_loop(monkeypatch, [], status=404)
    asyncio.run(loop2._release_carry_forward_if_unseen("run-y"))
    assert released == [] and released2 == []
