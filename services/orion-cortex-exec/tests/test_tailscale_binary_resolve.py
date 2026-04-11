from __future__ import annotations

from app import verb_adapters


def test_resolve_tailscale_binary_finds_configured_absolute_path(tmp_path, monkeypatch):
    fake = tmp_path / "tailscale"
    fake.write_text("#!/bin/sh\necho\n")
    fake.chmod(0o755)
    monkeypatch.setattr(verb_adapters.settings, "tailscale_path", str(fake))
    resolved, tried = verb_adapters._resolve_tailscale_binary()
    assert resolved == str(fake.resolve())
    assert str(fake) in tried


def test_resolve_tailscale_binary_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(verb_adapters.settings, "tailscale_path", "/no/such/tailscale")
    monkeypatch.setattr(verb_adapters.shutil, "which", lambda _name: None)
    # Avoid picking up a real /usr/bin/tailscale on the runner host.
    monkeypatch.setattr(verb_adapters.Path, "is_file", lambda self: False)
    resolved, tried = verb_adapters._resolve_tailscale_binary()
    assert resolved is None
    assert "/usr/bin/tailscale" in tried
