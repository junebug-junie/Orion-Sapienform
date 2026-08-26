from __future__ import annotations

from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1] / "app"
_ENROLL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "enroll_identity_face.py"


def test_save_gallery_embedding_only_referenced_by_enrollment_script():
    """Structural pin for the design doc's own non-negotiable: 'One enrolled
    subject. Gallery does not grow.' The only code path allowed to call
    identity_gallery.save_gallery_embedding is the human-run enrollment CLI
    -- if any file under app/ (the request-handling path) ever references
    it, that path can grow the gallery at runtime, which is exactly what
    this test exists to catch before it ships."""
    offending_files = []
    for py_file in _APP_DIR.glob("*.py"):
        if py_file.name == "identity_gallery.py":
            continue  # the definition itself, not a caller
        text = py_file.read_text(encoding="utf-8")
        if "save_gallery_embedding" in text:
            offending_files.append(py_file.name)

    assert offending_files == [], (
        f"save_gallery_embedding referenced outside the enrollment script in: {offending_files} "
        "-- this would let the request-handling path grow the identity gallery at runtime, "
        "violating the design doc's 'gallery does not grow' non-negotiable."
    )


def test_enrollment_script_is_the_one_real_caller():
    assert _ENROLL_SCRIPT.is_file()
    assert "save_gallery_embedding" in _ENROLL_SCRIPT.read_text(encoding="utf-8")
