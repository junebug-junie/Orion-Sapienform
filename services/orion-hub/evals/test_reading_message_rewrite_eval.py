"""Offline eval of the readings rewrite checks (no DB, no model)."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.reading_finding import ReadingFindingBundleV1, ReadingFindingV1, ReadingOriginStepV1

T = datetime(2026, 11, 10, 3, 7, tzinfo=timezone.utc)


def _load():
    path = Path(__file__).with_name("run_reading_message_rewrite_eval.py")
    spec = importlib.util.spec_from_file_location("run_reading_message_rewrite_eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_checks_pass_on_the_new_prompt_and_keep_the_old_provenance():
    mod = _load()
    from orion.world_pulse_read.finding_prompt import build_reading_finding_prompt, origin_lines

    origin = [ReadingOriginStepV1(seed_id="d", kind="digest_item", url="u1", requested_by="world_pulse"),
              ReadingOriginStepV1(seed_id="r", kind="reading", url="u2", parent_request_id="req")]
    bundle = ReadingFindingBundleV1(
        seed_id="r", url="u2", learned="x", article_summary="y", origin=origin,
        findings=[ReadingFindingV1(finding_id="link:a", kind="link_made", seed_id="r", observed_at=T,
                                   headline="Link made: A is associated with B", evidence_text="A and B")])
    prompt = build_reading_finding_prompt(bundle)
    lines = origin_lines(origin)
    old_prompt = "\n".join(lines[:-1] + ["Your reading pipeline ran each of these reads on its own timer."])
    checks = mod.check_prompt(prompt, lines, old_prompt)
    assert checks == {"provenance_kept": True, "record_vocab_in_changes": [], "asks_why_juniper": True}
    assert mod.check_prompt(prompt, lines, "a different chain")["provenance_kept"] is False


def test_record_vocabulary_measure_on_real_old_messages():
    mod = _load()
    old = ("One edge landed in my concept graph tonight. It was accepted because the sentence was "
           "sitting word-for-word in the text I read. One link, one link's worth of change.")
    assert {"edge", "graph", "accepted", "word-for-word", "one link"} <= set(mod.record_vocab_hits(old))
    assert mod.record_vocab_hits("The piece argues strokes are rising in younger adults.") == []
