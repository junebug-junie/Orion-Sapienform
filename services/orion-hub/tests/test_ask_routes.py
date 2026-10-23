"""Hub ask routes (walkway camera idea 3): list, answer, dismiss, publish.

A fake asyncpg pool stands in for Postgres; it enforces the same "only an open,
unexpired row moves" rule the real UPDATE does, so the 409 path is exercised.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.schemas.ask import OrionAskAnsweredV1  # noqa: E402
from scripts import ask_routes  # noqa: E402

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
MAIN_PY = (HUB_ROOT / "scripts" / "main.py").read_text(encoding="utf-8")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _FakeConn:
    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows
        self.sql: list[str] = []
        self.statements: dict[str, str] = {}
        self.outcomes: list[tuple] = []
        self.tx_depth = 0

    def _answerable(self, r: dict[str, Any]) -> bool:
        return r["status"] == "open" and (r["expires_at"] is None or r["expires_at"] > _now())

    async def fetch(self, sql: str, *args):
        self.sql.append(sql)
        if "jsonb_each_text" in sql:
            # CLOSE_NAMED_WHO_SQL: open "who is this?" cards for the named individuals.
            import json as _json

            names, pair_ask_id, kind = _json.loads(args[0]), args[1], args[2]
            out = []
            for r in self.rows.values():
                if r["source_kind"] == kind and r["source_ref"] in names and r["status"] == "open":
                    r.update(status="answered", answer=names[r["source_ref"]], answered_at=_now(),
                             answer_detail={"resolved_by": pair_ask_id})
                    out.append({k: r[k] for k in ("ask_id", "status", "answer", "answered_at", "source_kind", "source_ref")})
            return out
        if "FROM episode_memory" in sql:
            (ids,) = args
            return [{"memory_id": m, "statement": self.statements[m]} for m in ids if m in self.statements]
        status, limit = args
        out = [r for r in self.rows.values() if r["status"] == status and (status != "open" or self._answerable(r))]
        return sorted(out, key=lambda r: r["created_at"], reverse=True)[:limit]

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self_inner):
                conn.tx_depth = conn.tx_depth + 1
                return conn

            async def __aexit__(self_inner, *exc):
                conn.tx_depth -= 1
                return False

        return _Tx()

    async def fetchval(self, sql: str, *args):
        self.sql.append(sql)
        assert "FROM episode_memory" in sql
        return self.statements.get(args[0])

    async def execute(self, sql: str, *args):
        self.sql.append(sql)
        if "INSERT INTO attention_loop_outcome" in sql:
            assert self.tx_depth == 1, "outcome insert must share the card update's transaction"
            self.outcomes.append(args)
        return "INSERT 0 1"

    async def fetchrow(self, sql: str, *args):
        self.sql.append(sql)
        if sql.lstrip().startswith("UPDATE"):
            ask_id, new_status, answer, kinds, *same_rule = args
            detail = None
            if len(same_rule) == 3:
                *same_rule, detail = same_rule
            r = self.rows.get(ask_id)
            if r is None or not self._answerable(r):
                return None
            if same_rule:
                # The real SQL: "same person?" / "Is this X?" cards only take yes / no (or a dismissal).
                yes_no_kinds, yes_no = same_rule
                if r["source_kind"] in yes_no_kinds and answer is not None and answer not in yes_no:
                    return None
            # The real SQL: /answer|/dismiss exclude RESOLVABLE_KINDS, /resolve requires them.
            wants_resolvable = "= ANY($4" in sql
            if (r["source_kind"] in kinds) != wants_resolvable:
                return None
            if wants_resolvable:
                assert self.tx_depth == 1, "card update must run inside the transaction"
            r.update(status=new_status, answer=answer, answered_at=_now())
            if "answer_detail" in sql:
                import json as _json

                r["answer_detail"] = None if detail is None else _json.loads(detail)
            return {k: r[k] for k in ("ask_id", "status", "answer", "answered_at", "source_kind", "source_ref")}
        (ask_id,) = args
        r = self.rows.get(ask_id)
        if "FOR UPDATE" in sql:
            keys = ("source_kind", "source_ref", "image_ref", "image_refs", "status", "expires_at")
            return None if r is None else {k: r.get(k) for k in keys}
        return None if r is None else {"status": r["status"], "expires_at": r["expires_at"], "source_kind": r["source_kind"]}


class _FakePool:
    def __init__(self, rows):
        self.conn = _FakeConn(rows)

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self_inner):
                return pool.conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


def _row(ask_id: str, *, status: str = "open", expires_at=None, created_offset: int = 0) -> dict[str, Any]:
    return {
        "ask_id": ask_id,
        "asked_of": "juniper",
        "question": "I have seen this same person 12 times, usually around 08:10. Do you know who this is?",
        "evidence_refs": '["sighting-1", "sighting-2"]',
        "image_ref": "/mnt/telemetry/vision/crops/x.jpg",
        "status": status,
        "answer": None,
        "answered_at": None,
        "created_at": _now() - timedelta(minutes=created_offset),
        "expires_at": expires_at,
        "source_kind": "vision_individual",
        "source_ref": f"ind-{ask_id}",
    }


@pytest.fixture()
def env(monkeypatch):
    rows = {
        "a1": _row("a1", created_offset=5),
        "a2": _row("a2", created_offset=1),
        "old": _row("old", expires_at=_now() - timedelta(hours=1)),
        "done": _row("done", status="answered"),
    }
    published: list[OrionAskAnsweredV1] = []

    async def _publish(event):
        published.append(event)
        return True

    monkeypatch.setattr(ask_routes, "_publish_answered", _publish)
    app = FastAPI()
    app.include_router(ask_routes.router)
    app.state.memory_pg_pool = _FakePool(rows)
    return TestClient(app), rows, published


def test_list_open_hides_expired_and_closed(env):
    client, _rows, _pub = env
    r = client.get("/api/asks?status=open")
    assert r.status_code == 200
    ids = [a["ask_id"] for a in r.json()["asks"]]
    assert ids == ["a2", "a1"]
    first = r.json()["asks"][0]
    assert first["evidence_refs"] == ["sighting-1", "sighting-2"]
    assert first["created_at"].endswith("+00:00")


def test_answer_moves_open_row_and_publishes(env):
    client, rows, published = env
    r = client.post("/api/asks/a1/answer", json={"answer": "  the   mail carrier "})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["published"] is True
    assert body["ask"]["status"] == "answered"
    assert rows["a1"]["status"] == "answered"
    assert rows["a1"]["answer"] == "the mail carrier"
    assert rows["a1"]["answered_at"] is not None
    assert len(published) == 1
    ev = published[0]
    assert (ev.ask_id, ev.status, ev.answer, ev.source_kind, ev.source_ref) == (
        "a1", "answered", "the mail carrier", "vision_individual", "ind-a1"
    )


def test_second_answer_is_409_and_does_not_overwrite(env):
    client, rows, published = env
    assert client.post("/api/asks/a1/answer", json={"answer": "Bob"}).status_code == 200
    r = client.post("/api/asks/a1/answer", json={"answer": "Alice"})
    assert r.status_code == 409
    assert r.json()["detail"] == "ask_not_open:answered"
    assert rows["a1"]["answer"] == "Bob"
    assert len(published) == 1


def test_expired_open_row_is_409_expired(env):
    client, rows, published = env
    r = client.post("/api/asks/old/dismiss")
    assert r.status_code == 409
    assert r.json()["detail"] == "ask_not_open:expired"
    assert rows["old"]["status"] == "open"
    assert published == []


def test_unknown_ask_is_404(env):
    client, _rows, _pub = env
    assert client.post("/api/asks/nope/dismiss").status_code == 404


def test_dismiss_sets_status_and_publishes_without_answer(env):
    client, rows, published = env
    r = client.post("/api/asks/a2/dismiss")
    assert r.status_code == 200
    assert rows["a2"]["status"] == "dismissed"
    assert rows["a2"]["answer"] is None
    assert published[0].status == "dismissed" and published[0].answer is None


def test_blank_answer_rejected_without_touching_row(env):
    client, rows, published = env
    assert client.post("/api/asks/a1/answer", json={"answer": "   "}).status_code == 422
    assert client.post("/api/asks/a1/answer", json={}).status_code == 422
    assert rows["a1"]["status"] == "open"
    assert published == []


def test_publish_failure_still_saves_answer(monkeypatch, env):
    client, rows, _published = env

    async def _fail(_event):
        return False

    monkeypatch.setattr(ask_routes, "_publish_answered", _fail)
    r = client.post("/api/asks/a1/answer", json={"answer": "Bob"})
    assert r.status_code == 200
    assert r.json()["published"] is False
    assert rows["a1"]["status"] == "answered"


def test_no_pool_is_503():
    app = FastAPI()
    app.include_router(ask_routes.router)
    app.state.memory_pg_pool = None
    assert TestClient(app).get("/api/asks").status_code == 503


def test_answered_envelope_matches_catalog_kind():
    ev = OrionAskAnsweredV1(ask_id="a", status="answered", answer="x", source_kind="vision_individual", source_ref="i")
    env = ask_routes.build_answered_envelope(ev)
    assert env.kind == "orion.ask.answered.v1"
    assert OrionAskAnsweredV1.model_validate(env.payload).ask_id == "a"
    assert ask_routes.CHANNEL_ASK_ANSWERED == "orion:ask:answered"


def test_router_registered_in_main():
    assert "from scripts.ask_routes import router as ask_router" in MAIN_PY
    assert "app.include_router(ask_router)" in MAIN_PY


def test_template_declares_card_and_script_tag():
    assert 'id="visionAsksCard"' in INDEX_HTML
    assert 'id="visionAsksList"' in INDEX_HTML
    assert 'id="visionAsksStatus"' in INDEX_HTML
    assert '/static/js/vision-asks.js?v={{HUB_UI_ASSET_VERSION}}' in INDEX_HTML
    assert (HUB_ROOT / "static" / "js" / "vision-asks.js").is_file()
    # 2026-10-06: the card is the first thing on the Hub home, above the EKG (memory-episode spec
    # section 5), no longer inside the Vision panel.
    hub_panel = INDEX_HTML.index('id="hubTabPanel"')
    card = INDEX_HTML.index('id="visionAsksCard"')
    assert hub_panel < card < INDEX_HTML.index('id="ekgCardTitle"')
    assert card < INDEX_HTML.index('id="affectCaptureResult"')
    assert "Things Orion saw and wants your help naming" not in INDEX_HTML


def test_rendered_index_includes_card_and_script():
    from scripts.main import render_hub_index_html

    html = render_hub_index_html(memory_pool_ok=True)
    assert 'id="visionAsksCard"' in html
    assert "/static/js/vision-asks.js?v=" in html
    assert "{{HUB_UI_ASSET_VERSION}}" not in html


def test_update_sql_guards_open_and_unexpired(env):
    """The fake re-implements the WHERE in Python; pin the real SQL text too."""
    client, _rows, _pub = env
    client.post("/api/asks/a1/dismiss")
    pool = client.app.state.memory_pg_pool
    update = next(q for q in pool.conn.sql if q.lstrip().startswith("UPDATE"))
    norm = " ".join(update.split())
    assert "WHERE ask_id = $1 AND status = 'open' AND (expires_at IS NULL OR expires_at > now())" in norm
    assert "answered_at = now()" in norm
    assert "RETURNING ask_id, status, answer, answered_at, source_kind, source_ref" in norm


def test_connection_errors_map_to_503():
    from asyncpg.exceptions import ConnectionDoesNotExistError, InterfaceError

    class _BoomPool:
        def __init__(self, exc):
            self.exc = exc

        def acquire(self):
            exc = self.exc

            class _Ctx:
                async def __aenter__(self_inner):
                    raise exc

                async def __aexit__(self_inner, *a):
                    return False

            return _Ctx()

    for exc in (ConnectionDoesNotExistError("gone"), InterfaceError("closed")):
        app = FastAPI()
        app.include_router(ask_routes.router)
        app.state.memory_pg_pool = _BoomPool(exc)
        r = TestClient(app).get("/api/asks")
        assert r.status_code == 503 and r.json()["detail"] == "ask_store_unavailable"


# --- crop thumbnails (the ask card's picture) --------------------------------


def _thumb_client(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_routes, "_crop_thumb_dir", lambda: tmp_path)
    app = FastAPI()
    app.include_router(ask_routes.router)
    return TestClient(app)


def test_thumb_route_serves_a_stored_thumb_by_hash(tmp_path, monkeypatch) -> None:
    import hashlib

    data = b"\xff\xd8\xff\xe0fake-jpeg"
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / f"{digest}.jpg").write_bytes(data)
    r = _thumb_client(tmp_path, monkeypatch).get(f"/api/vision/crop-thumbs/{digest}")
    assert r.status_code == 200
    assert r.content == data
    assert r.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("bad", [
    "..%2F..%2Fetc%2Fpasswd",
    "a" * 63,
    "a" * 65,
    "A" * 64,
    "g" * 64,
    ("a" * 64) + ".jpg",
])
def test_thumb_route_rejects_bad_ids(tmp_path, monkeypatch, bad) -> None:
    # A file that WOULD be served if validation were skipped.
    (tmp_path / "secret.jpg").write_bytes(b"x")
    r = _thumb_client(tmp_path, monkeypatch).get(f"/api/vision/crop-thumbs/{bad}")
    assert r.status_code in (400, 404)
    assert r.content != b"x"


def test_thumb_route_404s_a_missing_or_tampered_thumb(tmp_path, monkeypatch) -> None:
    client = _thumb_client(tmp_path, monkeypatch)
    missing = "c" * 64
    assert client.get(f"/api/vision/crop-thumbs/{missing}").status_code == 404
    (tmp_path / f"{missing}.jpg").write_bytes(b"not the bytes that hash to the name")
    assert client.get(f"/api/vision/crop-thumbs/{missing}").status_code == 404


def test_hub_mounts_the_thumb_dir_read_only() -> None:
    compose = (HUB_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "crop_thumbs}:${HUB_VISION_CROP_THUMB_DIR:-/mnt/telemetry/orion-vision-host/crop_thumbs}:ro" in compose


def test_thumb_route_refuses_symlinks_fifos_dirs_and_oversize(tmp_path, monkeypatch) -> None:
    import hashlib
    import os

    client = _thumb_client(tmp_path, monkeypatch)
    secret = tmp_path.parent / "outside.jpg"
    secret.write_bytes(b"secret")
    link_id = hashlib.sha256(b"secret").hexdigest()
    (tmp_path / f"{link_id}.jpg").symlink_to(secret)
    assert client.get(f"/api/vision/crop-thumbs/{link_id}").status_code == 404

    fifo_id = "d" * 64
    os.mkfifo(tmp_path / f"{fifo_id}.jpg")
    assert client.get(f"/api/vision/crop-thumbs/{fifo_id}").status_code == 404  # and did not hang

    dir_id = "e" * 64
    (tmp_path / f"{dir_id}.jpg").mkdir()
    assert client.get(f"/api/vision/crop-thumbs/{dir_id}").status_code == 404

    big = b"x" * (ask_routes.MAX_THUMB_BYTES + 1)
    big_id = hashlib.sha256(big).hexdigest()
    (tmp_path / f"{big_id}.jpg").write_bytes(big)
    assert client.get(f"/api/vision/crop-thumbs/{big_id}").status_code == 404


def test_thumb_route_permission_error_is_404(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ask_routes.os, "open", lambda *a, **k: (_ for _ in ()).throw(PermissionError("no")))
    assert _thumb_client(tmp_path, monkeypatch).get(f"/api/vision/crop-thumbs/{'f' * 64}").status_code == 404


# --- memory confirmation cards: Confirm / Revise / Reject (2026-10-06) -------------------------

import json  # noqa: E402
import uuid  # noqa: E402

from orion.memory.episode.confirmation import ask_id_for, loop_id_for, outcome_id_for  # noqa: E402
from orion.schemas.attention_salience import AttentionLoopOutcomeV1  # noqa: E402

MEM_ID = str(uuid.UUID(int=7))


def _memory_row(memory_id: str = MEM_ID, **kw) -> dict[str, Any]:
    loop = loop_id_for(memory_id)
    row = _row(ask_id_for(loop), **kw)
    row.update(source_kind="memory_confirmation", source_ref=loop, image_ref=None, evidence_refs="[]",
               question="You told me something on Oct 3, and I wrote it down like this: “X.” Want me to remember that?")
    return row


@pytest.fixture()
def menv(monkeypatch, env):
    client, rows, published = env
    mem = _memory_row()
    rows[mem["ask_id"]] = mem
    pool = client.app.state.memory_pg_pool
    pool.conn.statements[MEM_ID] = "Juniper told me that X happened today."
    outcomes: list[AttentionLoopOutcomeV1] = []

    async def _pub_outcome(o):
        outcomes.append(o)
        return True

    monkeypatch.setattr(ask_routes, "_publish_loop_outcome", _pub_outcome)
    monkeypatch.setattr(ask_routes, "_confirmation_loop_enabled", lambda: True)
    return client, rows, published, outcomes, mem["ask_id"], pool.conn


@pytest.mark.parametrize("resolution,note,ask_status,verdict", [
    ("confirmed", "", "answered", "resolved"),
    ("revised", "  Juniper told me Y,  not X. ", "answered", "resolved"),
    ("rejected", "", "dismissed", "dismissed"),
])
def test_resolve_closes_card_writes_outcome_in_one_tx_and_publishes(menv, resolution, note, ask_status, verdict):
    client, rows, published, outcomes, ask_id, conn = menv
    r = client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": resolution, "note": note})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome_id"] == outcome_id_for(ask_id)
    assert rows[ask_id]["status"] == ask_status
    # The outcome row (asserted inside the fake to share the card update's transaction).
    assert len(conn.outcomes) == 1
    oid, loop_id, theme, v, actor, saved_note, _sal, _wv, feats, _at = conn.outcomes[0]
    feats = json.loads(feats)
    assert (oid, loop_id, theme, v, actor) == (outcome_id_for(ask_id), loop_id_for(MEM_ID), loop_id_for(MEM_ID), verdict, "juniper")
    assert feats["resolution"] == resolution and feats["ask_id"] == ask_id and feats["memory_id"] == MEM_ID
    assert feats["via"] == "orion_is_asking"
    assert saved_note == " ".join(note.split())
    # Published after commit: the single resolution record, plus the usual card event.
    assert [o.outcome_id for o in outcomes] == [oid]
    assert outcomes[0].verdict == verdict
    assert published[0].ask_id == ask_id and published[0].status == ask_status


@pytest.mark.parametrize("note,detail", [
    ("   ", "revised_needs_note"),
    ("no that's wrong", "revised_too_short"),          # a meta-note, not wording: Reject is the button
    (" juniper TOLD me that X  happened today. ", "revised_unchanged"),  # the prefilled text sent back as-is
])
def test_revise_refuses_a_note_that_is_not_new_wording_and_touches_nothing(menv, note, detail):
    client, rows, published, outcomes, ask_id, conn = menv
    conn.statements[MEM_ID] = "Juniper told me that X happened today."
    r = client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": "revised", "note": note})
    assert r.status_code == 422 and r.json()["detail"] == detail
    assert conn.outcomes == [] and outcomes == [] and published == []
    # The real route raises inside the transaction, so the card update rolls back. The fake has no
    # rollback; the Postgres test (test_ask_resolve_pg.py) pins that the card stays open.


def test_second_resolve_is_409_without_a_second_outcome(menv):
    client, rows, _pub, outcomes, ask_id, conn = menv
    assert client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": "confirmed"}).status_code == 200
    r = client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": "rejected"})
    assert r.status_code == 409 and r.json()["detail"] == "ask_not_open:answered"
    assert len(conn.outcomes) == 1 and len(outcomes) == 1


def test_memory_card_cannot_be_answered_or_dismissed_without_an_outcome(menv):
    client, rows, published, _o, ask_id, conn = menv
    for path, payload in (("answer", {"answer": "yes"}), ("dismiss", None)):
        r = client.post(f"/api/asks/{ask_id}/{path}", json=payload) if payload else client.post(f"/api/asks/{ask_id}/{path}")
        assert r.status_code == 409 and r.json()["detail"] == "ask_needs_resolution"
    assert rows[ask_id]["status"] == "open" and published == [] and conn.outcomes == []


def test_vision_card_cannot_be_resolved(menv):
    client, rows, _p, _o, _ask, conn = menv
    r = client.post("/api/asks/a1/resolve", json={"resolution": "confirmed"})
    assert r.status_code == 409 and r.json()["detail"] == "ask_not_resolvable"
    assert rows["a1"]["status"] == "open" and conn.outcomes == []


def test_resolve_disabled_is_404(menv, monkeypatch):
    client, rows, _p, _o, ask_id, _c = menv
    monkeypatch.setattr(ask_routes, "_confirmation_loop_enabled", lambda: False)
    assert client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": "confirmed"}).status_code == 404
    assert rows[ask_id]["status"] == "open"


def test_outcome_publish_failure_still_commits(menv, monkeypatch):
    client, rows, _p, _o, ask_id, conn = menv

    async def _fail(_o):
        return False

    monkeypatch.setattr(ask_routes, "_publish_loop_outcome", _fail)
    r = client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": "confirmed"})
    assert r.status_code == 200 and r.json()["published_outcome"] is False
    assert rows[ask_id]["status"] == "answered" and len(conn.outcomes) == 1


def test_list_adds_the_memory_statement_for_revise(menv):
    client, _rows, _p, _o, ask_id, _c = menv
    asks = {a["ask_id"]: a for a in client.get("/api/asks?status=open").json()["asks"]}
    assert asks[ask_id]["memory_statement"] == "Juniper told me that X happened today."
    assert "memory_statement" not in asks["a1"]


def test_outcome_envelope_is_the_catalog_kind():
    from scripts.bus_publish import build_loop_outcome_envelope

    o = AttentionLoopOutcomeV1(outcome_id="o", loop_id=loop_id_for(MEM_ID), theme_key="t", verdict="resolved")
    assert build_loop_outcome_envelope(o).kind == "attention.loop.outcome.v1"
    assert ask_routes.CHANNEL_LOOP_OUTCOME == "orion:attention:loop_outcome"


# --- visual confirmation asks (2026-10-15) ------------------------------------


def _pair_row(ask_id: str) -> dict[str, Any]:
    r = _row(ask_id)
    r.update(source_kind="vision_same_individual", source_ref="aaa|bbb", image_ref="thumb:" + "a" * 64,
             image_refs='["thumb:' + "a" * 64 + '", "thumb:' + "b" * 64 + '"]',
             question="Are these the same person?")
    return r


def test_list_returns_image_refs_and_backfills_old_rows(env):
    client, rows, _pub = env
    rows["p1"] = _pair_row("p1")
    asks = {a["ask_id"]: a for a in client.get("/api/asks?status=open").json()["asks"]}
    assert asks["p1"]["image_refs"] == ["thumb:" + "a" * 64, "thumb:" + "b" * 64]
    # A row from before image_refs: the single image_ref becomes the list.
    assert asks["a1"]["image_refs"] == ["/mnt/telemetry/vision/crops/x.jpg"]
    assert "image_refs" in ask_routes._SELECT_COLUMNS


def test_same_person_card_takes_yes_or_no_only(env):
    client, rows, published = env
    rows["p1"] = _pair_row("p1")
    r = client.post("/api/asks/p1/answer", json={"answer": "maybe Bob"})
    assert r.status_code == 422 and r.json()["detail"] == "answer_must_be_yes_or_no"
    assert rows["p1"]["status"] == "open" and published == []
    r = client.post("/api/asks/p1/answer", json={"answer": " Yes "})
    assert r.status_code == 200, r.text
    assert rows["p1"]["answer"] == "yes"
    assert (published[0].source_kind, published[0].source_ref, published[0].answer) == (
        "vision_same_individual", "aaa|bbb", "yes")


def test_same_person_card_can_be_dismissed_as_not_sure(env):
    client, rows, _pub = env
    rows["p2"] = _pair_row("p2")
    assert client.post("/api/asks/p2/dismiss").status_code == 200
    assert rows["p2"]["status"] == "dismissed"


def _confirm_row(ask_id: str) -> dict[str, Any]:
    r = _row(ask_id)
    r.update(source_kind="vision_person_confirm", source_ref="ind-9|Juniper", image_ref="thumb:" + "c" * 64,
             image_refs='["thumb:' + "c" * 64 + '"]', question="Is this Juniper? It is my best guess, but I am not sure.")
    return r


def test_is_this_x_card_takes_yes_or_no_only_and_publishes_the_ref(env):
    client, rows, published = env
    rows["c1"] = _confirm_row("c1")
    r = client.post("/api/asks/c1/answer", json={"answer": "it is Bob"})
    assert r.status_code == 422 and r.json()["detail"] == "answer_must_be_yes_or_no"
    assert rows["c1"]["status"] == "open" and published == []
    r = client.post("/api/asks/c1/answer", json={"answer": "No"})
    assert r.status_code == 200, r.text
    assert (published[0].source_kind, published[0].source_ref, published[0].answer) == (
        "vision_person_confirm", "ind-9|Juniper", "no")


def _confirm_row_3(ask_id: str) -> dict[str, Any]:
    r = _confirm_row(ask_id)
    refs = ["thumb:" + c * 64 for c in "def"]
    r.update(image_ref=refs[0], image_refs=json.dumps(refs))
    return r


def test_is_this_x_card_says_which_picture_is_the_person(env):
    """2026-10-23, Juniper: "one of the three is Emerald but I can't specify which"."""
    client, rows, published = env
    rows["c3"] = _confirm_row_3("c3")
    refs = ["thumb:" + c * 64 for c in "def"]
    r = client.post("/api/asks/c3/answer", json={"answer": "yes", "pictures": [
        {"image_ref": refs[2], "verdict": "unsure"},
        {"image_ref": refs[0], "verdict": "different"},
        {"image_ref": refs[1], "verdict": "same"},
    ]})
    assert r.status_code == 200, r.text
    assert rows["c3"]["answer"] == "yes"
    assert rows["c3"]["answer_detail"] == {"pictures": [
        {"image_ref": refs[0], "index": 0, "verdict": "different"},
        {"image_ref": refs[1], "index": 1, "verdict": "same"},
        {"image_ref": refs[2], "index": 2, "verdict": "unsure"},
    ]}
    assert published[0].source_kind == "vision_person_confirm"


@pytest.mark.parametrize("answer,pictures,detail", [
    ("no", [("d", "same"), ("e", "unsure"), ("f", "unsure")], "answer_does_not_match_pictures"),
    ("yes", [("d", "different"), ("e", "unsure"), ("f", "unsure")], "answer_does_not_match_pictures"),
    ("yes", [("d", "same"), ("d", "same"), ("f", "unsure")], "picture_listed_twice"),
    ("yes", [("d", "same"), ("e", "unsure"), ("z", "unsure")], "picture_not_on_card"),
])
def test_bad_picture_answers_on_is_this_x_are_refused(env, answer, pictures, detail):
    client, rows, published = env
    rows["c3"] = _confirm_row_3("c3")
    r = client.post("/api/asks/c3/answer", json={"answer": answer, "pictures": [
        {"image_ref": "thumb:" + c * 64, "verdict": v} for c, v in pictures]})
    assert r.status_code == 422 and r.json()["detail"] == detail, r.text
    assert rows["c3"]["status"] == "open" and published == []


def test_a_picture_the_card_could_not_show_counts_as_not_sure(env):
    client, rows, _pub = env
    rows["c3"] = _confirm_row_3("c3")
    r = client.post("/api/asks/c3/answer", json={"answer": "yes", "pictures": [
        {"image_ref": "thumb:" + "d" * 64, "verdict": "same"}, {"image_ref": "thumb:" + "e" * 64, "verdict": "different"}]})
    assert r.status_code == 200, r.text
    assert [p["verdict"] for p in rows["c3"]["answer_detail"]["pictures"]] == ["same", "different", "unsure"]


def test_a_name_on_an_is_this_x_picture_is_refused(env):
    client, rows, _pub = env
    rows["c3"] = _confirm_row_3("c3")
    r = client.post("/api/asks/c3/answer", json={"answer": "yes", "pictures": [
        {"image_ref": "thumb:" + c * 64, "verdict": "same", "name": "Bob" if c == "d" else ""} for c in "def"]})
    assert r.status_code == 422 and r.json()["detail"] == "no_names_on_is_this_cards"


def test_is_this_x_card_can_be_dismissed_as_cant_tell(env):
    client, rows, _pub = env
    rows["c2"] = _confirm_row("c2")
    assert client.post("/api/asks/c2/dismiss").status_code == 200
    assert rows["c2"]["status"] == "dismissed"


def test_answer_sql_enforces_yes_no_for_same_person_cards(env):
    client, rows, _pub = env
    client.post("/api/asks/a1/answer", json={"answer": "Bob"})
    pool = client.app.state.memory_pg_pool
    upd = next(q for q in pool.conn.sql if q.lstrip().startswith("UPDATE"))
    assert "source_kind <> ALL($5::text[]) OR $3::text IS NULL OR $3::text = ANY($6::text[])" in upd


@pytest.mark.skipif(not __import__("os").getenv("ORION_TEST_PG_URI"),
                    reason="ORION_TEST_PG_URI not set (throwaway Postgres with the walkway migrations only)")
def test_close_ask_sql_runs_on_real_postgres():
    """The fake pool never sends SQL to Postgres; this does. A statement Postgres
    cannot prepare broke every Answer/Dismiss in review."""
    import asyncio
    import os

    asyncpg = pytest.importorskip("asyncpg")
    dsn = os.environ["ORION_TEST_PG_URI"].replace("postgresql+psycopg2://", "postgresql://")

    import json

    async def go():
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute("DELETE FROM orion_ask WHERE ask_id LIKE 'sqltest-%'")
            for aid, kind in (("sqltest-p1", "vision_same_individual"), ("sqltest-p2", "vision_same_individual"),
                              ("sqltest-p3", "vision_same_individual"), ("sqltest-w1", "vision_individual"),
                              ("sqltest-c1", "vision_person_confirm"), ("sqltest-c2", "vision_person_confirm")):
                await conn.execute("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref) "
                                   "VALUES ($1, 'q', $2, $1)", aid, kind)
            args = (list(ask_routes.RESOLVABLE_KINDS), list(ask_routes.YES_NO_KINDS), list(ask_routes.YES_NO_ANSWERS))
            run = lambda aid, st, ans, d=None: conn.fetchrow(ask_routes.CLOSE_ASK_SQL, aid, st, ans, *args, d)  # noqa: E731
            assert (await run("sqltest-p1", "answered", "yes"))["answer"] == "yes"
            assert await run("sqltest-p2", "answered", "maybe") is None
            assert (await run("sqltest-p3", "dismissed", None))["status"] == "dismissed"
            assert (await run("sqltest-w1", "answered", "Bob"))["answer"] == "Bob"
            assert await run("sqltest-c1", "answered", "maybe Bob") is None
            assert (await run("sqltest-c2", "answered", "no"))["answer"] == "no"
            await conn.execute("UPDATE orion_ask SET status='open' WHERE ask_id = 'sqltest-p1'")
            assert (await run("sqltest-p1", "answered", "no", '{"names": {"x": "Theo"}}'))["answer"] == "no"
            stored = await conn.fetchval("SELECT answer_detail::text FROM orion_ask WHERE ask_id = 'sqltest-p1'")
            assert json.loads(stored) == {"names": {"x": "Theo"}}
            await conn.execute("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref) "
                               "VALUES ('sqltest-w2', 'q', 'vision_individual', 'sqltest-ind')")
            closed = await conn.fetch(ask_routes.CLOSE_NAMED_WHO_SQL, '{"sqltest-ind": "Theo"}', "sqltest-p1",
                                      ask_routes.WHO_IS_THIS_KIND)
            assert [(r["ask_id"], r["answer"]) for r in closed] == [("sqltest-w2", "Theo")]
            await conn.execute("DELETE FROM orion_ask WHERE ask_id LIKE 'sqltest-%'")
        finally:
            await conn.close()

    asyncio.run(go())


# --- names on visual cards (2026-10-15) ----------------------------------------


def _who_row(ask_id: str, individual_id: str, refs: list[str] | None = None) -> dict[str, Any]:
    import json as _json

    r = _row(ask_id)
    refs = refs or ["thumb:" + "a" * 64]
    r.update(source_ref=individual_id, image_ref=refs[0], image_refs=_json.dumps(refs))
    return r


def test_same_person_with_a_name_names_both_and_closes_their_who_is_this(env):
    client, rows, published = env
    rows["p1"] = _pair_row("p1")
    rows["w-a"] = _who_row("w-a", "aaa")
    rows["w-x"] = _who_row("w-x", "xxx")  # someone else: untouched
    r = client.post("/api/asks/p1/answer", json={"answer": "yes", "names": {"aaa": "  Kid  next door "}})
    assert r.status_code == 200, r.text
    assert rows["p1"]["answer"] == "yes"
    assert rows["p1"]["answer_detail"] == {"names": {"aaa": "Kid next door", "bbb": "Kid next door"}}
    assert r.json()["closed_asks"] == ["w-a"]
    assert (rows["w-a"]["status"], rows["w-a"]["answer"]) == ("answered", "Kid next door")
    assert rows["w-a"]["answer_detail"] == {"resolved_by": "p1"}
    assert rows["w-x"]["status"] == "open"
    # The pair event carries the names; the closed card publishes its own plain answer.
    assert [e.ask_id for e in published] == ["p1", "w-a"]
    assert published[0].names == {"aaa": "Kid next door", "bbb": "Kid next door"}
    assert (published[1].source_kind, published[1].answer, published[1].names) == ("vision_individual", "Kid next door", {})


def test_different_people_names_each_face(env):
    client, rows, published = env
    rows["p1"] = _pair_row("p1")
    r = client.post("/api/asks/p1/answer", json={"answer": "no", "names": {"aaa": "Theo", "bbb": "Maya"}})
    assert r.status_code == 200, r.text
    assert rows["p1"]["answer_detail"] == {"names": {"aaa": "Theo", "bbb": "Maya"}}
    assert published[0].names == {"aaa": "Theo", "bbb": "Maya"}


def test_empty_names_are_a_plain_verdict(env):
    client, rows, published = env
    rows["p1"] = _pair_row("p1")
    r = client.post("/api/asks/p1/answer", json={"answer": "no", "names": {"aaa": "  ", "bbb": ""}})
    assert r.status_code == 200, r.text
    assert rows["p1"]["answer_detail"] is None
    assert published[0].names == {}
    # Left out of the bus payload, so a consumer built before `names` still validates it.
    assert "names" not in ask_routes.build_answered_envelope(published[0]).payload


@pytest.mark.parametrize("body,detail", [
    ({"answer": "yes", "names": {"aaa": "Theo", "bbb": "Maya"}}, "same_person_takes_one_name"),
    ({"answer": "no", "names": {"zzz": "Theo"}}, "name_for_unknown_individual"),
    ({"answer": "no", "names": {"aaa": "x" * 121}}, "name_too_long"),
    ({"answer": "no", "names": {"aaa": "a", "bbb": "b", "ccc": "c"}}, "name_for_unknown_individual"),
])
def test_bad_names_are_refused_and_leave_the_card_open(env, body, detail):
    client, rows, published = env
    rows["p1"] = _pair_row("p1")
    r = client.post("/api/asks/p1/answer", json=body)
    assert r.status_code == 422 and r.json()["detail"] == detail, r.text
    assert rows["p1"]["status"] == "open" and published == []


def test_names_are_refused_on_a_who_is_this_card(env):
    client, rows, _pub = env
    r = client.post("/api/asks/a1/answer", json={"answer": "Bob", "names": {"ind-a1": "Bob"}})
    assert r.status_code == 422 and r.json()["detail"] == "names_only_on_same_person_cards"
    assert rows["a1"]["status"] == "open"


def test_who_is_this_with_three_pictures_records_each_picture(env):
    client, rows, published = env
    refs = ["thumb:" + c * 64 for c in "abc"]
    rows["w3"] = _who_row("w3", "ind-1", refs)
    r = client.post("/api/asks/w3/answer", json={"answer": "Theo", "pictures": [
        {"image_ref": refs[1], "verdict": "same"},
        {"image_ref": refs[2], "verdict": "different", "name": " Other  kid "},
    ]})
    assert r.status_code == 200, r.text
    assert rows["w3"]["answer"] == "Theo"  # the individual's name: the first (face) picture's
    assert rows["w3"]["answer_detail"] == {"pictures": [
        {"image_ref": refs[0], "index": 0, "verdict": "same", "name": "Theo"},
        {"image_ref": refs[1], "index": 1, "verdict": "same", "name": "Theo"},
        {"image_ref": refs[2], "index": 2, "verdict": "different", "name": "Other kid"},
    ]}
    assert published[0].answer == "Theo" and published[0].names == {}


@pytest.mark.parametrize("pictures,detail", [
    ([{"image_ref": "thumb:" + "z" * 64, "verdict": "same"}], "picture_not_on_card"),
    ([{"image_ref": "thumb:" + "b" * 64, "verdict": "same"}] * 2, "picture_listed_twice"),
    ([{"image_ref": "thumb:" + "a" * 64, "verdict": "different", "name": "x"}], "first_picture_is_the_answer"),
    ([{"image_ref": "thumb:" + "b" * 64, "verdict": "same", "name": "Not Theo"}], "same_picture_takes_the_answer"),
])
def test_bad_pictures_are_refused(env, pictures, detail):
    client, rows, _pub = env
    rows["w3"] = _who_row("w3", "ind-1", ["thumb:" + c * 64 for c in "abc"])
    r = client.post("/api/asks/w3/answer", json={"answer": "Theo", "pictures": pictures})
    assert r.status_code == 422 and r.json()["detail"] == detail, r.text
    assert rows["w3"]["status"] == "open"


def test_pictures_are_refused_on_a_pair_card(env):
    client, rows, _pub = env
    rows["p1"] = _pair_row("p1")
    r = client.post("/api/asks/p1/answer", json={"answer": "yes", "pictures": [
        {"image_ref": "thumb:" + "b" * 64, "verdict": "same"}]})
    assert r.status_code == 422 and r.json()["detail"] == "pictures_only_on_who_is_this_cards"


def test_unparseable_answer_with_names_is_still_yes_no_422(env):
    client, rows, _pub = env
    rows["p1"] = _pair_row("p1")
    r = client.post("/api/asks/p1/answer", json={"answer": "maybe", "names": {"aaa": "Theo"}})
    assert r.status_code == 422 and r.json()["detail"] == "answer_must_be_yes_or_no"


def test_missing_names_migration_is_503_not_500(env):
    asyncpg = pytest.importorskip("asyncpg")
    client, _rows, _pub = env
    pool = client.app.state.memory_pg_pool

    async def boom(*_a, **_k):
        raise asyncpg.exceptions.UndefinedColumnError('column "answer_detail" of relation "orion_ask" does not exist')

    pool.conn.fetchrow = boom
    r = client.post("/api/asks/a1/answer", json={"answer": "Bob"})
    assert r.status_code == 503 and r.json()["detail"] == "ask_schema_missing"


def test_names_on_a_closed_card_are_409_not_422(env):
    client, rows, _pub = env
    rows["p1"] = _pair_row("p1")
    rows["p1"]["status"] = "answered"
    r = client.post("/api/asks/p1/answer", json={"answer": "yes", "names": {"zzz": "Theo"}})
    assert r.status_code == 409 and r.json()["detail"] == "ask_not_open:answered"
    rows["done"]["status"] = "answered"
    r = client.post("/api/asks/done/answer", json={"answer": "Bob", "names": {"x": "Bob"}})
    assert r.status_code == 409


def test_picture_index_is_the_first_position_shown():
    refs = ["thumb:" + "a" * 64, "thumb:" + "b" * 64, "thumb:" + "a" * 64]
    out = ask_routes.picture_answer_detail("Theo", refs, [
        ask_routes.PictureAnswer(image_ref=refs[1], verdict="different", name="Maya")])
    assert [(p["index"], p["verdict"]) for p in out["pictures"]] == [(0, "same"), (1, "different")]
    with pytest.raises(ask_routes.HTTPException):  # the repeat of picture 1 is picture 1, the answer
        ask_routes.picture_answer_detail("Theo", refs, [ask_routes.PictureAnswer(image_ref=refs[2], verdict="unsure")])
