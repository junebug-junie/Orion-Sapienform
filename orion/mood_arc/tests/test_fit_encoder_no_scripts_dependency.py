"""Regression for a live outage (2026-07-21): orion/mood_arc/fit_encoder.py
had a module-level `from scripts.fit_phi_encoder import _pearson,
_percentile`. scripts/ is repo-root operator/CLI tooling, not shipped in
every service's Docker image -- orion-field-digester's container has no
scripts/ directory, so importing this module at all (which
app/anomaly_scorer.py does, for the inference-only load_artifacts/
score_windows functions) crash-looped the whole service.

_pearson/_percentile are only used inside training-only functions
(prune_correlated_fields, train_autoencoder) -- neither is reachable from
load_artifacts()/score_windows(), so the import was moved from module scope
into those two functions specifically. This test asserts the module-level
import stays gone by directly checking `scripts` is never touched by the
inference path, run in a subprocess with `scripts` genuinely unimportable
(not just "happens to not be imported yet" in this process, which could pass
even with a bug if pytest's own collection already imported `scripts` for an
unrelated reason).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_fit_encoder_importable_and_inference_usable_without_scripts_on_path() -> None:
    script = textwrap.dedent(
        """
        import sys
        sys.path[:] = [p for p in sys.path if p]  # drop the implicit '' cwd entry
        sys.path.insert(0, {repo_root!r})

        import builtins
        _real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "scripts" or name.startswith("scripts."):
                raise ImportError(f"simulated missing scripts/ package: {{name}}")
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = _blocking_import

        import orion.mood_arc.fit_encoder as fe

        assert callable(fe.load_artifacts)
        assert callable(fe.score_windows)
        assert "scripts" not in sys.modules
        print("OK")
        """
    ).format(repo_root=str(_REPO_ROOT))

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"importing fit_encoder without scripts/ available must succeed "
        f"(this is exactly what broke orion-field-digester's container).\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_scripts_import_still_works_lazily_when_available(monkeypatch) -> None:
    """The flip side: with scripts/ genuinely available (the normal dev/CLI
    environment), prune_correlated_fields()/train_autoencoder()'s lazy
    imports must still resolve correctly -- this isn't just a deletion, it's
    a relocation."""
    import numpy as np

    from orion.mood_arc.fit_encoder import prune_correlated_fields
    from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1
    from datetime import datetime, timezone

    rows = [
        FieldChannelCorpusRowV1(
            generated_at=datetime.now(timezone.utc),
            tick_id=f"t{i}",
            channels={"a": float(i), "b": float(i)},  # perfectly correlated
        )
        for i in range(5)
    ]
    kept = prune_correlated_fields(rows, fields=("a", "b"), corr_threshold=0.9)
    assert len(kept) == 1
