from __future__ import annotations

from app.settings import Settings


def test_defaults_do_not_require_env():
    s = Settings(_env_file=None)
    assert s.SERVICE_NAME == "affectgpt-worker"
    assert s.NODE_NAME == "circe"


def test_face_or_frame_is_the_only_real_checkpoint_mode():
    # Confirmed live 2026-08-22: no frame-mode checkpoint is reachable.
    # This must never silently change without a corresponding checkpoint.
    s = Settings(_env_file=None)
    assert s.AFFECTGPT_FACE_OR_FRAME == "multiface_audio_face_text"


def test_ckpt_epoch_is_pinned_not_latest():
    s = Settings(_env_file=None)
    assert isinstance(s.AFFECTGPT_CKPT_EPOCH, int)
    assert s.AFFECTGPT_CKPT_EPOCH == 60
