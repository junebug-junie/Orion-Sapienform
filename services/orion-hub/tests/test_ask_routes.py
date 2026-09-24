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

    def _answerable(self, r: dict[str, Any]) -> bool:
        return r["status"] == "open" and (r["expires_at"] is None or r["expires_at"] > _now())

    async def fetch(self, sql: str, status: str, limit: int):
        self.sql.append(sql)
        out = [r for r in self.rows.values() if r["status"] == status and (status != "open" or self._answerable(r))]
        return sorted(out, key=lambda r: r["created_at"], reverse=True)[:limit]

    async def fetchrow(self, sql: str, *args):
        self.sql.append(sql)
        if sql.lstrip().startswith("UPDATE"):
            ask_id, new_status, answer = args
            r = self.rows.get(ask_id)
            if r is None or not self._answerable(r):
                return None
            r.update(status=new_status, answer=answer, answered_at=_now())
            return {k: r[k] for k in ("ask_id", "status", "answer", "answered_at", "source_kind", "source_ref")}
        (ask_id,) = args
        r = self.rows.get(ask_id)
        return None if r is None else {"status": r["status"], "expires_at": r["expires_at"]}


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
    # Card lives inside the Vision panel, after its last control.
    assert INDEX_HTML.index('id="affectCaptureResult"') < INDEX_HTML.index('id="visionAsksCard"') < INDEX_HTML.index('id="visionFloatingContainer"')


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
