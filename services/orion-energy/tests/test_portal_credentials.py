from __future__ import annotations

import pytest

from portal.credentials import CredentialsFileTooOpen, PortalCredentials, load_credentials


def _write(path, text, mode=0o600):
    path.write_text(text)
    path.chmod(mode)
    return path


def test_missing_file_means_no_credentials(tmp_path) -> None:
    assert load_credentials(tmp_path / "credentials.env") is None


def test_owner_only_file_loads(tmp_path) -> None:
    path = _write(tmp_path / "c.env", "# rmp\nRMP_USERNAME=me@example.com\nRMP_PASSWORD=p=ss word\n")
    assert load_credentials(path) == PortalCredentials(username="me@example.com", password="p=ss word")


def test_quoted_values_are_unquoted(tmp_path) -> None:
    path = _write(tmp_path / "c.env", "RMP_USERNAME='me'\nRMP_PASSWORD=\"sek ret\"\n")
    assert load_credentials(path) == PortalCredentials(username="me", password="sek ret")


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644])
def test_group_or_world_readable_file_is_refused(tmp_path, mode) -> None:
    path = _write(tmp_path / "c.env", "RMP_USERNAME=me\nRMP_PASSWORD=x\n", mode=mode)
    with pytest.raises(CredentialsFileTooOpen):
        load_credentials(path)


@pytest.mark.parametrize("text", ["RMP_USERNAME=me\n", "RMP_PASSWORD=x\n", "RMP_USERNAME=\nRMP_PASSWORD=x\n", ""])
def test_incomplete_file_means_no_credentials(tmp_path, text) -> None:
    assert load_credentials(_write(tmp_path / "c.env", text)) is None


def test_repr_never_shows_the_password() -> None:
    creds = PortalCredentials(username="me@example.com", password="hunter2")
    assert "hunter2" not in repr(creds) and "me@example.com" not in repr(creds)
