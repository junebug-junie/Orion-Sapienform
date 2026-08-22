"""Percepts and chat uploads must resolve from different stores.

This is the whole point of `AttachmentRefV1.kind`. A camera frame of a private
home and a file someone dragged into a chat box have different lifetimes and
different blast radii, and Hub -- which serves the chat store -- is also the
process holding the docker socket. If these two ever collapse onto one base,
the separation this service exists for is gone and nothing else would notice.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import vision  # noqa: E402
from app.vision import AttachmentFetchError, resolve_attachment_url  # noqa: E402

_SHA = "a" * 64


class _Ref:
    def __init__(self, kind: str, sha256: str = _SHA) -> None:
        self.kind = kind
        self.sha256 = sha256


@pytest.fixture()
def cfg(monkeypatch):
    monkeypatch.setattr(
        vision.settings, "llm_gateway_attachment_base_url",
        "http://hub-host:8080/api/chat/attachments", raising=False)
    monkeypatch.setattr(
        vision.settings, "llm_gateway_percept_base_url",
        "http://percept-host:8000/percepts", raising=False)
    monkeypatch.setattr(
        vision.settings, "llm_gateway_attachment_allowed_hosts",
        "hub-host,percept-host", raising=False)
    return vision.settings


def test_percept_resolves_from_the_percept_store(cfg) -> None:
    assert resolve_attachment_url(_Ref("percept")) == f"http://percept-host:8000/percepts/{_SHA}"


def test_image_still_resolves_from_the_chat_store(cfg) -> None:
    """Backward compatibility: every existing producer sends kind='image'."""
    assert resolve_attachment_url(_Ref("image")) == \
        f"http://hub-host:8080/api/chat/attachments/{_SHA}"


def test_missing_kind_defaults_to_the_chat_store(cfg) -> None:
    class _Legacy:
        sha256 = _SHA
    assert "chat/attachments" in resolve_attachment_url(_Legacy())


def test_unset_percept_base_refuses_and_does_not_fall_back(cfg, monkeypatch) -> None:
    """The dangerous failure is silent reuse of the chat base.

    An unconfigured percept store must refuse, not quietly start serving camera
    frames out of the chat upload namespace.
    """
    monkeypatch.setattr(vision.settings, "llm_gateway_percept_base_url", "", raising=False)
    with pytest.raises(AttachmentFetchError) as exc:
        resolve_attachment_url(_Ref("percept"))
    assert "LLM_GATEWAY_PERCEPT_BASE_URL" in str(exc.value)
    # and specifically not the other store
    assert "chat" not in str(exc.value).lower()


def test_percept_host_must_still_be_allowlisted(cfg, monkeypatch) -> None:
    """Defence in depth survives the new base -- a typo'd base still fails closed."""
    monkeypatch.setattr(
        vision.settings, "llm_gateway_attachment_allowed_hosts", "hub-host", raising=False)
    with pytest.raises(AttachmentFetchError, match="not allowlisted"):
        resolve_attachment_url(_Ref("percept"))


@pytest.mark.parametrize("bad", ["../../etc/passwd", "not-a-hash", "", "0" * 63, "0" * 65, "0" * 63 + "g"])
def test_percepts_get_the_same_sha_validation_as_chat_uploads(cfg, bad) -> None:
    """The only caller-supplied component stays a validated hex digest.

    There is no path or authority to inject into: the URL is rebuilt as
    `<trusted base>/<digest>` and the ref's own source_url is ignored.
    """
    with pytest.raises(AttachmentFetchError):
        resolve_attachment_url(_Ref("percept", sha256=bad))


def test_uppercase_digest_is_normalised_not_rejected(cfg) -> None:
    """Documents a real asymmetry rather than asserting a wrong expectation.

    `resolve_attachment_url` lowercases before validating, so an uppercase
    digest resolves to the canonical lowercase URL. The percept store itself is
    stricter and 400s on uppercase (verified live). That is harmless because
    the gateway is the only caller and it always emits the lowercase form -- but
    it is worth pinning, since a future direct caller would find the store
    stricter than the gateway that fronts it.
    """
    assert resolve_attachment_url(_Ref("percept", sha256="A" * 64)) == \
        f"http://percept-host:8000/percepts/{'a' * 64}"


def test_the_two_bases_are_not_the_same_url(cfg) -> None:
    """Guards the collapse case directly."""
    assert resolve_attachment_url(_Ref("percept")) != resolve_attachment_url(_Ref("image"))
